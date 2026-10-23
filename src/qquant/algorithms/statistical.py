"""Workload I — statistical hypothesis testing.

Estimands
  ols_slope              b_j in y_t = a_j + b_j x_t + e_t       (classical only; see note)
  directional_hit_rate   a_j = P[(x_t - median(x)) * y_t > 0]   (classical AND quantum)

Note on quantum: there is no claim that OLS or t-tests are accelerated by quantum
computing. The quantum candidate estimates a *bounded* statistic (a probability) by
amplitude estimation, with the state preparation and its classical data-loading cost
stated and timed explicitly (proposal 4.3).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pennylane as qml
import statsmodels.api as sm
from scipy import stats

from ..config import ManifestError
from .base import AlgoOutput, Algorithm, Param, RunContext, register
from .qae import run_mlae, schedule
from .tasks import SignalTask


def auto_hac_lags(T: int, h: int) -> int:
    """Newey-West rule of thumb, floored at h-1 for overlapping h-day labels."""
    return int(max(h - 1, math.floor(4 * (T / 100) ** (2 / 9))))


def auto_block_len(T: int, h: int) -> int:
    return int(max(h, round(T ** (1 / 3))))


def _slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Row-wise OLS slopes for stacked resamples x, y of shape (B, T)."""
    xm = x - x.mean(axis=1, keepdims=True)
    ym = y - y.mean(axis=1, keepdims=True)
    return (xm * ym).sum(axis=1) / (xm**2).sum(axis=1)


def _block_indices(T: int, L: int, B: int, method: str, rng: np.random.Generator) -> np.ndarray:
    n_blocks = int(np.ceil(T / L))
    if method == "stationary":
        idx = np.empty((B, T), dtype=np.int64)
        for b in range(B):
            out, pos = [], 0
            while pos < T:
                start = rng.integers(0, T)
                length = rng.geometric(1 / L)
                out.extend((start + np.arange(length)) % T)
                pos += length
            idx[b] = np.asarray(out[:T])
        return idx
    hi = T if method == "circular" else T - L + 1
    starts = rng.integers(0, hi, size=(B, n_blocks))
    idx = (starts[:, :, None] + np.arange(L)[None, None, :]).reshape(B, -1)[:, :T]
    return idx % T


def _controls(task: SignalTask, which: str) -> pd.DataFrame | None:
    if which == "none":
        return None
    bench = which.removeprefix("bench_momentum_")
    col = f"bench_{bench}"
    if col not in task.frame:
        raise ManifestError(f"control {which} needs benchmark {bench} in the snapshot")
    k = task.hypothesis.lookback
    b = task.frame[col]
    c = (b / b.shift(k) - 1.0).rename(f"{bench}_mom{k}")
    return c.reindex(task.data.index).to_frame()


@register
class OLSRobust(Algorithm):
    name = "ols_robust"
    workload = "statistical"
    kind = "classical"
    estimand = "ols_slope"
    description = "OLS slope with HAC (Newey-West) / HC standard errors; optional benchmark control."
    params = [
        Param("cov", str, "HAC", "covariance estimator", choices=("HAC", "HC1", "nonrobust")),
        Param("hac_lags", int, -1, "-1 = auto: max(h-1, 4(T/100)^(2/9))", min=-1),
        Param("controls", str, "none", "'none' or 'bench_momentum_<TICKER>'"),
        Param("alpha", float, 0.05, "significance level", min=1e-4, max=0.5),
    ]

    def run(self, task: SignalTask, p, prof, ctx):
        with prof.stage("preprocess"):
            df = task.data
            X = df[["x"]]
            ctrl = _controls(task, p["controls"])
            if ctrl is not None:
                X = X.join(ctrl).dropna()
            y = df["y"].loc[X.index]
            X = sm.add_constant(X)
            T = len(y)
            lags = p["hac_lags"] if p["hac_lags"] >= 0 else auto_hac_lags(T, task.horizon)
        with prof.stage("execute"):
            model = sm.OLS(y.values, X.values)
            if p["cov"] == "HAC":
                res = model.fit(cov_type="HAC", cov_kwds={"maxlags": lags})
            elif p["cov"] == "HC1":
                res = model.fit(cov_type="HC1")
            else:
                res = model.fit()
        with prof.stage("postprocess"):
            ci = res.conf_int(alpha=p["alpha"])
            out = {
                "estimate": float(res.params[1]), "std_error": float(res.bse[1]),
                "t_stat": float(res.tvalues[1]), "p_value": float(res.pvalues[1]),
                "ci_low": float(ci[1][0]), "ci_high": float(ci[1][1]),
                "intercept": float(res.params[0]), "r2": float(res.rsquared),
                "n_obs": T, "hac_lags": lags if p["cov"] == "HAC" else None,
            }
        prof.set("samples", T)
        return AlgoOutput(out)


@register
class BlockBootstrapOLS(Algorithm):
    name = "block_bootstrap_ols"
    workload = "statistical"
    kind = "classical"
    estimand = "ols_slope"
    description = "Vectorized block bootstrap of the OLS slope (moving / circular / stationary)."
    params = [
        Param("n_boot", int, 1000, "bootstrap repetitions", min=50, max=100_000),
        Param("block_len", int, -1, "-1 = auto max(h, T^(1/3))", min=-1),
        Param("method", str, "moving", choices=("moving", "circular", "stationary")),
        Param("alpha", float, 0.05, min=1e-4, max=0.5),
    ]

    def run(self, task: SignalTask, p, prof, ctx):
        rng = np.random.default_rng(ctx.seed)
        with prof.stage("preprocess"):
            x, y = task.data["x"].to_numpy(), task.data["y"].to_numpy()
            T = len(x)
            L = p["block_len"] if p["block_len"] > 0 else auto_block_len(T, task.horizon)
            b_hat = float(_slopes(x[None], y[None])[0])
        with prof.stage("execute"):
            idx = _block_indices(T, L, p["n_boot"], p["method"], rng)
            b_star = _slopes(x[idx], y[idx])
        with prof.stage("postprocess"):
            a = p["alpha"]
            lo, hi = np.quantile(b_star, [a / 2, 1 - a / 2])
            # p-value from the bootstrap distribution re-centred under H0 (b = 0)
            centred = b_star - b_star.mean()
            pval = (1 + np.sum(np.abs(centred) >= abs(b_hat))) / (1 + len(b_star))
            out = {
                "estimate": b_hat, "std_error": float(b_star.std(ddof=1)),
                "p_value": float(pval), "ci_low": float(lo), "ci_high": float(hi),
                "n_obs": T, "block_len": L,
            }
        prof.set("samples", T)
        prof.set("resamples", p["n_boot"])
        return AlgoOutput(out)


@register
class BlockPermutationTest(Algorithm):
    name = "block_permutation"
    workload = "statistical"
    kind = "classical"
    estimand = "ols_slope"
    description = "Permutation test of b=0 shuffling contiguous blocks of x against fixed y."
    params = [
        Param("n_perm", int, 1000, min=50, max=100_000),
        Param("block_len", int, -1, "-1 = auto max(h, T^(1/3))", min=-1),
    ]

    def run(self, task: SignalTask, p, prof, ctx):
        rng = np.random.default_rng(ctx.seed)
        prof.warn(
            "Block permutation assumes blocks of x are exchangeable under H0; "
            "serial dependence longer than block_len invalidates the test."
        )
        with prof.stage("preprocess"):
            x, y = task.data["x"].to_numpy(), task.data["y"].to_numpy()
            T = len(x)
            L = p["block_len"] if p["block_len"] > 0 else auto_block_len(T, task.horizon)
            n_blocks = T // L
            Tt = n_blocks * L  # drop the ragged tail so all blocks are equal length
            xb = x[:Tt].reshape(n_blocks, L)
            yt = y[:Tt]
            b_hat = float(_slopes(x[None, :Tt], yt[None])[0])
        with prof.stage("execute"):
            perms = np.argsort(rng.random((p["n_perm"], n_blocks)), axis=1)
            xp = xb[perms].reshape(p["n_perm"], Tt)
            b_null = _slopes(xp, np.broadcast_to(yt, xp.shape))
        with prof.stage("postprocess"):
            pval = (1 + np.sum(np.abs(b_null) >= abs(b_hat))) / (1 + len(b_null))
            out = {"estimate": b_hat, "p_value": float(pval), "n_obs": Tt, "block_len": L,
                   "null_sd": float(b_null.std(ddof=1))}
        prof.set("samples", Tt)
        prof.set("resamples", p["n_perm"])
        return AlgoOutput(out)


# ---------------------------------------------------------------- hit rate (matched pair)
def hit_indicator(df: pd.DataFrame) -> np.ndarray:
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    return ((x - np.median(x)) * y > 0).astype(float)


@register
class HitRateClassical(Algorithm):
    name = "hit_rate_classical"
    workload = "statistical"
    kind = "classical"
    estimand = "directional_hit_rate"
    description = "Exact directional hit rate with HAC standard error; z-test vs 0.5."
    params = [
        Param("alpha", float, 0.05, min=1e-4, max=0.5),
        Param("hac_lags", int, -1, "-1 = auto", min=-1),
    ]

    def run(self, task: SignalTask, p, prof, ctx):
        with prof.stage("preprocess"):
            s = hit_indicator(task.data)
            T = len(s)
            lags = p["hac_lags"] if p["hac_lags"] >= 0 else auto_hac_lags(T, task.horizon)
        with prof.stage("execute"):
            res = sm.OLS(s, np.ones(T)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
        with prof.stage("postprocess"):
            a, se = float(res.params[0]), float(res.bse[0])
            z = stats.norm.ppf(1 - p["alpha"] / 2)
            zstat = (a - 0.5) / se if se > 0 else 0.0
            out = {
                "estimate": a, "std_error": se, "ci_low": float(a - z * se), "ci_high": float(a + z * se),
                "z_stat": float(zstat), "p_value": float(2 * stats.norm.sf(abs(zstat))), "n_obs": T,
            }
        prof.set("samples", T)
        return AlgoOutput(out)


@register
class HitRateQAE(Algorithm):
    name = "hit_rate_qae"
    workload = "statistical"
    kind = "quantum"
    experimental = True
    estimand = "directional_hit_rate"
    description = (
        "Maximum-likelihood amplitude estimation of the directional hit rate. "
        "'index_register' loads all T indicator values into a log2(T)+1 qubit state "
        "(Mottonen); 'amplitude_rotation' precomputes the rate classically and only "
        "benchmarks AE mechanics."
    )
    params = [
        Param("state_prep", str, "amplitude_rotation", choices=("amplitude_rotation", "index_register")),
        Param("shots", int, 100, "shots per Grover power", min=1, max=1_000_000),
        Param("n_powers", int, 5, "number of Grover powers in the schedule", min=1, max=12),
        Param("schedule", str, "exponential", choices=("exponential", "linear")),
        Param("alpha", float, 0.05, min=1e-4, max=0.5),
    ]

    def check(self, p, ctx):
        if p["state_prep"] == "index_register" and ctx.max_qubits < 3:
            raise ManifestError("index_register needs max_qubits >= 3")

    def run(self, task: SignalTask, p, prof, ctx):
        rng = np.random.default_rng(ctx.seed)
        with prof.stage("load"):
            s = hit_indicator(task.data)  # classical O(T) pass: unavoidable data loading
            T = len(s)
        with prof.stage("preprocess"):
            if p["state_prep"] == "amplitude_rotation":
                prof.warn(
                    "state_prep=amplitude_rotation: the target amplitude is computed classically "
                    "before encoding, so the classical answer is already known. This measures "
                    "amplitude-estimation mechanics only, not an end-to-end quantum method."
                )
                a_true = float(s.mean())
                theta = 2 * np.arcsin(np.sqrt(a_true))
                n_wires, obj = 1, 0

                def prep(wires):
                    qml.RY(theta, wires=wires[0])
            else:
                n_idx = max(1, int(np.ceil(np.log2(T))))
                n_wires = n_idx + 1
                if n_wires > ctx.max_qubits:
                    raise ManifestError(
                        f"index_register needs {n_wires} qubits for T={T} > max_qubits={ctx.max_qubits}"
                    )
                prof.warn(
                    f"index_register: Mottonen state preparation over 2^{n_idx} amplitudes; its "
                    "gate count grows ~linearly in T and dominates any query savings."
                )
                amp = np.zeros(2**n_wires)
                # basis |i>|s_i>, objective qubit is the last wire (least significant)
                amp[2 * np.arange(T) + s.astype(int)] = 1 / np.sqrt(T)
                obj = n_wires - 1

                def prep(wires):
                    qml.MottonenStatePreparation(amp, wires=wires)
        powers = schedule(p["schedule"], p["n_powers"])
        res = run_mlae(prep, n_wires, obj, powers, p["shots"], p["alpha"], rng, prof, ctx.backend)
        with prof.stage("postprocess"):
            # Both encodings put probability (#hits)/T on the objective qubit (padded index
            # states carry zero amplitude), so the AE estimate targets the empirical hit rate.
            # The AE interval quantifies *estimation* error of that number, not sampling
            # uncertainty of the hit rate itself, so no hypothesis-test p-value is reported:
            # inference uses the classical HAC standard error (hit_rate_classical).
            out = {
                "estimate": res.estimate, "ae_std_error": res.std_error,
                "ae_ci_low": res.ci_low, "ae_ci_high": res.ci_high,
                "n_obs": T, "oracle_queries": res.oracle_queries, "total_shots": res.shots,
            }
        prof.set("samples", T)
        return AlgoOutput(out)
