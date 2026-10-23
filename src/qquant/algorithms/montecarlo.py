"""Workload III — Monte Carlo estimation of bounded NVDA expectations.

Model (toy, documented, NOT a validated NVDA price model): geometric Brownian motion
    log S_T ~ Normal(log S0 + (mu - sigma^2/2) T, sigma^2 T)
calibrated on a trailing window of the snapshot. Payoffs are bounded in [0, 1]:
    digital      g(S) = 1[S > K]                         -> Pr(S_T > K)
    capped_call  g(S) = min((S - K)^+, cap) / cap         -> E[g(S_T)]

Every estimator targets the same quantity. Quantum amplitude estimation needs the
distribution discretized onto 2^n grid points, which changes the estimand slightly;
`discretized_exact` computes that discretized target classically, and both the
discretization bias and each method's error are reported separately.
"""
from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane.templates.subroutines.qmc import func_to_unitary, probs_to_unitary
from scipy.stats import norm, qmc

from ..config import ManifestError
from .base import AlgoOutput, Algorithm, Param, RunContext, register
from .qae import circuit_resources, run_mlae, schedule
from .tasks import MonteCarloTask


# ----------------------------------------------------------------------- ground truth
def payoff(task: MonteCarloTask, S: np.ndarray) -> np.ndarray:
    if task.payoff == "digital":
        return (S > task.strike).astype(float)
    if task.payoff == "capped_call":
        return np.minimum(np.maximum(S - task.strike, 0.0), task.cap) / task.cap
    raise ValueError(task.payoff)


def _call(task: MonteCarloTask, K: float) -> float:
    m, s = task.log_mean, task.log_sd
    d2 = (m - np.log(K)) / s
    return float(np.exp(m + s * s / 2) * norm.cdf(d2 + s) - K * norm.cdf(d2))


def analytic_value(task: MonteCarloTask) -> float:
    if task.payoff == "digital":
        return float(norm.cdf((task.log_mean - np.log(task.strike)) / task.log_sd))
    return (_call(task, task.strike) - _call(task, task.strike + task.cap)) / task.cap


def discretize(task: MonteCarloTask, n_qubits: int, n_sd: float) -> tuple[np.ndarray, np.ndarray]:
    """2^n equal-width bins in log-price over mean +- n_sd sd; midpoint values, renormalized."""
    m, s = task.log_mean, task.log_sd
    edges = np.linspace(m - n_sd * s, m + n_sd * s, 2**n_qubits + 1)
    probs = np.diff(norm.cdf(edges, loc=m, scale=s))
    probs /= probs.sum()
    mids = np.exp(0.5 * (edges[:-1] + edges[1:]))
    return probs, payoff(task, mids)


def _report(task: MonteCarloTask, est: float, half: float | None, extra: dict) -> dict:
    truth = analytic_value(task)
    out = {
        "estimate": float(est), "analytic_value": truth, "abs_error": float(abs(est - truth)),
        "ci_halfwidth": None if half is None else float(half),
        "ci_covers_truth": None if half is None else bool(abs(est - truth) <= half),
    }
    out.update(extra)
    return out


MC_COMMON = [
    Param("confidence", float, 0.95, min=0.5, max=0.9999),
]


# ----------------------------------------------------------------------- classical
@register
class PlainMonteCarlo(Algorithm):
    name = "mc_plain"
    workload = "montecarlo"
    kind = "classical"
    estimand = "mc_expectation"
    description = (
        "Vectorized pseudorandom Monte Carlo; optional antithetic variates and control variate "
        "(S_T, known mean). Fixed n_paths, or adaptive doubling until the CI half-width <= target_epsilon."
    )
    params = MC_COMMON + [
        Param("n_paths", int, 100_000, min=100, max=500_000_000),
        Param("target_epsilon", float, 0.0, "0 = fixed n_paths; >0 = adaptive", min=0.0),
        Param("antithetic", bool, False),
        Param("control_variate", bool, False),
        Param("max_paths", int, 50_000_000, min=1000),
    ]

    def _draw(self, task, n, rng, antithetic):
        if antithetic:
            z = rng.standard_normal((n + 1) // 2)
            z = np.concatenate([z, -z])[:n]
        else:
            z = rng.standard_normal(n)
        return np.exp(task.log_mean + task.log_sd * z)

    def _estimate(self, task, S, cv: bool, antithetic: bool):
        g = payoff(task, S)
        if cv:
            ES = np.exp(task.log_mean + task.log_sd**2 / 2)
            c = np.cov(g, S)
            beta = c[0, 1] / c[1, 1] if c[1, 1] > 0 else 0.0
            g = g - beta * (S - ES)
        if antithetic:  # pair-average for a valid variance estimate
            h = len(g) // 2
            g = 0.5 * (g[:h] + g[h:2 * h])
        return g.mean(), g.std(ddof=1) / np.sqrt(len(g))

    def run(self, task: MonteCarloTask, p, prof, ctx):
        rng = np.random.default_rng(ctx.seed)
        z = norm.ppf(0.5 + p["confidence"] / 2)
        n = p["n_paths"]
        total = 0
        with prof.stage("execute"):
            while True:
                S = self._draw(task, n, rng, p["antithetic"])
                est, se = self._estimate(task, S, p["control_variate"], p["antithetic"])
                total += n
                if p["target_epsilon"] <= 0 or z * se <= p["target_epsilon"] or total >= p["max_paths"]:
                    break
                n = min(n * 2, p["max_paths"] - total)
        if p["target_epsilon"] > 0 and z * se > p["target_epsilon"]:
            prof.warn("max_paths reached before target_epsilon")
        prof.set("samples", total)
        prof.set("final_batch_paths", n)
        return AlgoOutput(_report(task, est, z * se, {"paths": total, "std_error": float(se)}))


@register
class QuasiMonteCarlo(Algorithm):
    name = "mc_sobol"
    workload = "montecarlo"
    kind = "classical"
    estimand = "mc_expectation"
    description = "Randomized (scrambled) Sobol quasi-Monte Carlo; error from independent randomizations."
    params = MC_COMMON + [
        Param("log2_points", int, 14, "points per randomization = 2^m", min=4, max=26),
        Param("randomizations", int, 16, min=2, max=1000),
    ]

    def run(self, task: MonteCarloTask, p, prof, ctx):
        z = norm.ppf(0.5 + p["confidence"] / 2)
        ests = []
        with prof.stage("execute"):
            for r in range(p["randomizations"]):
                u = qmc.Sobol(d=1, scramble=True, seed=ctx.seed * 1000 + r).random_base2(p["log2_points"])[:, 0]
                S = np.exp(task.log_mean + task.log_sd * norm.ppf(np.clip(u, 1e-12, 1 - 1e-12)))
                ests.append(payoff(task, S).mean())
        ests = np.asarray(ests)
        se = ests.std(ddof=1) / np.sqrt(len(ests))
        total = len(ests) * 2 ** p["log2_points"]
        prof.set("samples", total)
        return AlgoOutput(_report(task, ests.mean(), z * se, {"paths": total, "std_error": float(se)}))


@register
class DiscretizedExact(Algorithm):
    name = "discretized_exact"
    workload = "montecarlo"
    kind = "classical"
    estimand = "mc_expectation"
    description = "Exact expectation over the same 2^n-point discretization QAE uses (O(2^n) work)."
    params = [
        Param("n_qubits", int, 6, "distribution register size", min=1, max=30),
        Param("n_sd", float, 4.0, "grid half-width in log-price standard deviations", min=1.0, max=10.0),
    ]

    def run(self, task: MonteCarloTask, p, prof, ctx):
        with prof.stage("preprocess"):
            probs, g = discretize(task, p["n_qubits"], p["n_sd"])
        with prof.stage("execute"):
            est = float(probs @ g)
        prof.set("grid_points", len(probs))
        truth = analytic_value(task)
        return AlgoOutput(_report(task, est, None, {"discretization_bias": est - truth}))


# ----------------------------------------------------------------------- quantum
QAE_COMMON = [
    Param("n_qubits", int, 5, "distribution register qubits (2^n grid points)", min=1, max=12),
    Param("n_sd", float, 4.0, "grid half-width in log-price standard deviations", min=1.0, max=10.0),
]


def _check_qubits(total: int, ctx: RunContext, name: str):
    if total > ctx.max_qubits:
        raise ManifestError(f"{name} needs {total} qubits > max_qubits={ctx.max_qubits}")


@register
class MaximumLikelihoodQAE(Algorithm):
    name = "qae_mlae"
    workload = "montecarlo"
    kind = "quantum"
    estimand = "mc_expectation"
    experimental = True
    description = (
        "Maximum-likelihood amplitude estimation (no phase register). A = P (distribution "
        "loading) followed by R (payoff rotation onto an ancilla), both as dense unitaries."
    )
    params = MC_COMMON + QAE_COMMON + [
        Param("shots", int, 100, "shots per Grover power", min=1, max=10_000_000),
        Param("n_powers", int, 6, min=1, max=14),
        Param("schedule", str, "exponential", choices=("exponential", "linear")),
    ]

    def check(self, p, ctx):
        _check_qubits(p["n_qubits"] + 1, ctx, self.name)

    def run(self, task: MonteCarloTask, p, prof, ctx):
        rng = np.random.default_rng(ctx.seed)
        n = p["n_qubits"]
        with prof.stage("preprocess"):
            probs, g = discretize(task, n, p["n_sd"])
            truth_discrete = float(probs @ g)
            P = probs_to_unitary(probs)
            R = func_to_unitary(lambda i: float(g[i]), 2**n)

        def prep(wires):
            qml.QubitUnitary(P, wires=wires[:n])
            qml.QubitUnitary(R, wires=wires)

        res = run_mlae(prep, n + 1, n, schedule(p["schedule"], p["n_powers"]), p["shots"],
                       1 - p["confidence"], rng, prof, ctx.backend)
        half = 0.5 * (res.ci_high - res.ci_low)
        extra = {
            "discretized_value": truth_discrete,
            "discretization_bias": truth_discrete - analytic_value(task),
            "abs_error_vs_discretized": abs(res.estimate - truth_discrete),
            "oracle_queries": res.oracle_queries, "total_shots": res.shots,
            "grid_points": 2**n,
        }
        prof.set("samples", res.shots)
        return AlgoOutput(_report(task, res.estimate, half, extra))


@register
class CanonicalQAE(Algorithm):
    name = "qae_canonical"
    workload = "montecarlo"
    kind = "quantum"
    estimand = "mc_expectation"
    experimental = True
    description = (
        "Canonical QPE-based amplitude estimation via qml.QuantumMonteCarlo; m estimation qubits "
        "give resolution ~pi/2^m; estimate from the most frequent phase outcome over `shots`. "
        "The payoff is halved to remove the template's a <-> 1-a phase ambiguity (costs 2x in error)."
    )
    params = MC_COMMON + QAE_COMMON + [
        Param("estimation_qubits", int, 6, min=1, max=12),
        Param("shots", int, 100, min=1, max=1_000_000),
    ]

    def check(self, p, ctx):
        _check_qubits(p["n_qubits"] + 1 + p["estimation_qubits"], ctx, self.name)

    def run(self, task: MonteCarloTask, p, prof, ctx):
        rng = np.random.default_rng(ctx.seed)
        n, m = p["n_qubits"], p["estimation_qubits"]
        with prof.stage("preprocess"):
            probs, g = discretize(task, n, p["n_sd"])
            truth_discrete = float(probs @ g)
        with prof.stage("compile"):
            target = list(range(n + 1))
            est_w = list(range(n + 1, n + 1 + m))
            dev = qml.device(ctx.backend, wires=n + 1 + m)

            @qml.qnode(dev)
            def circuit():
                # payoff halved: the template's phase readout is ambiguous between a and 1-a,
                # so we estimate a' = a/2 <= 1/2 (unambiguous) and report 2a'
                qml.QuantumMonteCarlo(probs, lambda i: 0.5 * float(g[i]), target_wires=target, estimation_wires=est_w)
                return qml.probs(wires=est_w)
        with prof.stage("instrumentation"):
            for k, v in circuit_resources(circuit).items():
                prof.maximum(f"max_{k}", v)
        prof.set("qubits", n + 1 + m)
        with prof.stage("execute"):
            dist = np.clip(np.asarray(circuit()), 0, None)
            counts = rng.multinomial(p["shots"], dist / dist.sum())
            prof.count("circuit_executions", 1)
            prof.count("shots", p["shots"])
            prof.count("oracle_queries", p["shots"] * (2 * (2**m - 1) + 1))  # 2^m-1 Q calls, 2 A each, + initial A
        with prof.stage("postprocess"):
            # outcomes j and 2^m - j encode the same amplitude; fold them, then take the mode
            N = 2**m
            folded = counts[: N // 2 + 1].astype(float)
            folded[1:N // 2] += counts[N - 1:N // 2:-1]
            phase = np.argmax(folded) / N
            a_half = float((1 - np.cos(np.pi * phase)) / 2)  # estimate of a/2 (PennyLane convention)
            est = 2 * a_half
            # canonical-AE error bound (Brassard et al.) on a/2, doubled for a
            half = 2 * (2 * np.pi * np.sqrt(a_half * (1 - a_half)) / N + np.pi**2 / N**2)
        prof.set("samples", p["shots"])
        extra = {
            "discretized_value": truth_discrete,
            "discretization_bias": truth_discrete - analytic_value(task),
            "abs_error_vs_discretized": abs(est - truth_discrete),
            "oracle_queries": int(p["shots"] * (2 * (2**m - 1) + 1)), "total_shots": p["shots"], "grid_points": 2**n,
        }
        return AlgoOutput(_report(task, est, half, extra))
