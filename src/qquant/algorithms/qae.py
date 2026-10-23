"""Amplitude estimation on PennyLane simulators.

Implements maximum-likelihood amplitude estimation (MLAE; Suzuki et al. 2020): run
Q^m A|0> for a schedule of Grover powers m_k, measure the objective qubit with N shots
each, and maximize the joint likelihood over theta, where a = sin^2(theta) is the
amplitude of the "good" state |1> on the objective qubit.

    Q = A S_0 A^dagger S_chi,   S_chi = Z on objective qubit,  S_0 = reflection about |0..0>

Query accounting: one shot of Q^m A uses (2m + 1) applications of A (the "oracle").
Classical Monte Carlo needs O(1/eps^2) samples; ideal AE needs O(1/eps) oracle calls.
That idealized scaling is what is being tested here — *on a classical simulator*, so
wall-clock time is NOT evidence of physical quantum speedup (proposal section 8.4).

Finite-shot measurement is emulated by binomial draws from the simulator's exact
probability of |1>; for a noiseless simulator this is distributionally identical to
shot-based sampling and much faster.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pennylane as qml
from scipy.optimize import minimize_scalar

from ..benchmark.profiler import Profiler

PrepFn = Callable[[Sequence[int]], None]  # applies A to the given wires


def circuit_resources(qnode, *args) -> dict:
    out = {}
    for level, prefix in (("top", "logical_"), ("device", "")):
        try:
            r = qml.specs(qnode, level=level)(*args).resources
            out[prefix + "depth"] = int(r.depth)
            out[prefix + "gates"] = int(r.num_gates)
        except Exception:  # decomposition may be unavailable for large QubitUnitary
            pass
    return out


def schedule(kind: str, n_powers: int) -> list[int]:
    if kind == "exponential":
        return [0] + [2**i for i in range(n_powers - 1)]
    if kind == "linear":
        return list(range(n_powers))
    raise ValueError(f"unknown schedule {kind}")


@dataclass
class AEResult:
    estimate: float
    ci_low: float
    ci_high: float
    std_error: float
    oracle_queries: int
    shots: int
    grover_powers: list[int]
    hits: list[int]


def mle_amplitude(powers: Sequence[int], hits: Sequence[int], shots: int, alpha: float) -> tuple[float, float, float, float]:
    powers = np.asarray(powers)
    hits = np.asarray(hits)
    mult = 2 * powers + 1
    eps = 1e-12

    def nll(theta):
        p = np.clip(np.sin(mult * theta) ** 2, eps, 1 - eps)
        return -np.sum(hits * np.log(p) + (shots - hits) * np.log(1 - p))

    grid = np.linspace(1e-6, np.pi / 2 - 1e-6, 20_000)
    # vectorized grid search, then bounded refinement around the best grid point
    P =np.clip(np.sin(np.outer(grid, mult)) ** 2, eps, 1 - eps)
    vals = -(np.log(P) @ hits + np.log(1 - P) @ (shots - hits))
    i = int(np.argmin(vals))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    theta = minimize_scalar(nll, bounds=(lo, hi), method="bounded").x
    fisher = np.sum(4 * shots * mult**2)  # Fisher information for theta
    se_theta = 1 / np.sqrt(fisher)
    from scipy.stats import norm

    z = norm.ppf(1 - alpha / 2)
    a = np.sin(theta) ** 2
    lo_t, hi_t = max(theta - z * se_theta, 0.0), min(theta + z * se_theta, np.pi / 2)
    se_a = abs(np.sin(2 * theta)) * se_theta
    return float(a), float(np.sin(lo_t) ** 2), float(np.sin(hi_t) ** 2), float(se_a)


def run_mlae(
    prep: PrepFn,
    n_wires: int,
    objective_wire: int,
    powers: Sequence[int],
    shots: int,
    alpha: float,
    rng: np.random.Generator,
    prof: Profiler,
    backend: str = "default.qubit",
) -> AEResult:
    wires = list(range(n_wires))
    with prof.stage("compile"):
        dev = qml.device(backend, wires=n_wires)

        def grover(m: int):
            prep(wires)
            for _ in range(m):
                qml.Z(objective_wire)  # S_chi: phase-flip the good state
                qml.adjoint(prep)(wires)
                qml.FlipSign([0] * n_wires, wires=wires)  # S_0 (up to global phase)
                prep(wires)
            return qml.probs(wires=objective_wire)

        qnode = qml.QNode(grover, dev)
    with prof.stage("instrumentation"):  # resource counting; excluded from end-to-end time
        res = circuit_resources(qnode, max(powers))
        for k, v in res.items():
            prof.maximum(f"max_{k}", v)
    prof.set("qubits", n_wires)
    prof.set("grover_power_max", max(powers))

    hits = []
    with prof.stage("execute"):
        for m in powers:
            p1 = float(qnode(m)[1])
            hits.append(int(rng.binomial(shots, min(max(p1, 0.0), 1.0))))
            prof.count("circuit_executions", 1)
            prof.count("shots", shots)
            prof.count("oracle_queries", shots * (2 * m + 1))

    with prof.stage("postprocess"):
        a, lo, hi, se = mle_amplitude(powers, hits, shots, alpha)
    return AEResult(
        estimate=a, ci_low=lo, ci_high=hi, std_error=se,
        oracle_queries=int(sum(shots * (2 * m + 1) for m in powers)),
        shots=int(shots * len(powers)), grover_powers=list(powers), hits=hits,
    )
