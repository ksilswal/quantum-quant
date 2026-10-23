"""Benchmark aggregation: timing summaries, scaling, speedup, matched-quality frontiers.

S(N) = T_classical(N) / T_quantum(N) is a *measured* ratio for specific
implementations, precision, hardware and workload (proposal eq. 5). On a simulator it
says nothing about physical quantum advantage; the tables carry the backend label.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from ..benchmark.profiler import STAGES


def config_label(df: pd.DataFrame) -> pd.Series:
    """algorithm, or algorithm#<params-hash> when one algorithm ran with several configs."""
    h = df["params_json"].map(lambda s: hashlib.sha1(s.encode()).hexdigest()[:5])
    n_cfg = df.assign(_h=h).groupby("algorithm")["_h"].transform("nunique")
    return np.where(n_cfg > 1, df["algorithm"] + "#" + h, df["algorithm"])


def _boot_ci(x: np.ndarray, stat=np.median, B: int = 2000, alpha: float = 0.05, seed: int = 0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    bs = stat(x[rng.integers(0, len(x), (B, len(x)))], axis=1)
    return tuple(np.quantile(bs, [alpha / 2, 1 - alpha / 2]))


def run_summary(runs: pd.DataFrame) -> pd.DataFrame:
    """Per-run end-to-end time and stage breakdown, aggregated per algorithm config."""
    r = runs[runs["workload"] != "signal"].copy()
    if r.empty:
        return pd.DataFrame()
    r["config"] = config_label(r)
    rows = []
    for (w, cfg, kind, exp), g in r.groupby(["workload", "config", "kind", "experimental"]):
        ok = g[g["status"] == "ok"]
        t = ok["total_seconds"].to_numpy()
        lo, hi = _boot_ci(t)
        row = {
            "workload": w, "config": cfg, "kind": kind, "experimental": bool(exp),
            "runs": len(g), "failed": int((g["status"] != "ok").sum()),
            "median_s": np.median(t) if len(t) else np.nan,
            "median_ci_low": lo, "median_ci_high": hi,
            "iqr_s": np.subtract(*np.percentile(t, [75, 25])) if len(t) else np.nan,
        }
        for st in (*STAGES, "inference", "instrumentation", "untracked"):
            col = f"t_{st}"
            if col in ok:
                row[f"median_{st}_s"] = ok[col].median()
        for c in ("c_qubits", "c_max_depth", "c_max_gates", "c_shots", "c_oracle_queries",
                  "c_circuit_executions", "c_samples"):
            if c in ok and ok[c].notna().any():
                row[c[2:]] = ok[c].median()
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["workload", "kind", "median_s"])


def scaling_table(batches: pd.DataFrame) -> pd.DataFrame:
    """Batch wall time vs N with dispersion across repetitions; throughput = N / T(N)."""
    if batches.empty:
        return pd.DataFrame()
    rows = []
    for (w, a, n), g in batches.groupby(["workload", "algorithm", "n_tasks"]):
        t = g["wall_seconds"].to_numpy()
        lo, hi = _boot_ci(t)
        rows.append({
            "workload": w, "algorithm": a, "N": int(n), "reps": len(t),
            "median_wall_s": float(np.median(t)), "ci_low": lo, "ci_high": hi,
            "min_s": float(t.min()), "max_s": float(t.max()),
            "throughput_tasks_per_s": float(n / np.median(t)) if np.median(t) > 0 else np.nan,
        })
    return pd.DataFrame(rows).sort_values(["workload", "algorithm", "N"])


def speedup_table(batches: pd.DataFrame, classical: str, quantum: str, seed: int = 0) -> pd.DataFrame:
    """S(N) = median T_classical / median T_quantum with a bootstrap CI over repetitions."""
    rng = np.random.default_rng(seed)
    rows = []
    for n in sorted(set(batches["n_tasks"])):
        c = batches[(batches.algorithm == classical) & (batches.n_tasks == n)]["wall_seconds"].to_numpy()
        q = batches[(batches.algorithm == quantum) & (batches.n_tasks == n)]["wall_seconds"].to_numpy()
        if len(c) == 0 or len(q) == 0:
            continue
        s = np.median(c) / np.median(q)
        if len(c) > 1 and len(q) > 1:
            bc = np.median(c[rng.integers(0, len(c), (2000, len(c)))], axis=1)
            bq = np.median(q[rng.integers(0, len(q), (2000, len(q)))], axis=1)
            lo, hi = np.quantile(bc / bq, [0.025, 0.975])
        else:
            lo = hi = np.nan
        rows.append({"N": int(n), "classical": classical, "quantum": quantum, "S": s, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def matched_quality(runs: pd.DataFrame) -> pd.DataFrame:
    """Accuracy and cost of every run whose estimand has a known reference value.

    montecarlo            reference = analytic value; for QAE also the discretized value
    directional_hit_rate  reference = exact empirical hit rate (hit_rate_classical)
    """
    r = runs[(runs["status"] == "ok") & runs["workload"].isin(["montecarlo", "statistical"])].copy()
    if r.empty:
        return pd.DataFrame()
    r["config"] = config_label(r)
    out = []
    mc = r[r["workload"] == "montecarlo"]
    for _, row in mc.iterrows():
        out.append({
            "estimand": "mc_expectation", "task_id": row.task_id, "config": row.config, "kind": row.kind,
            "repetition": row.repetition, "abs_error": row.get("m_abs_error"),
            # QAE targets the discretized distribution; classical MC targets the continuous one
            "abs_error_vs_own_target": row.get("m_abs_error_vs_discretized")
            if pd.notna(row.get("m_abs_error_vs_discretized", np.nan)) else row.get("m_abs_error"),
            "discretization_bias": row.get("m_discretization_bias"),
            "ci_halfwidth": row.get("m_ci_halfwidth"), "seconds": row.total_seconds,
            "samples_or_queries": row.get("c_oracle_queries") if row.kind == "quantum" else row.get("c_samples"),
        })
    hr = r[r["algorithm"].isin(["hit_rate_classical", "hit_rate_qae"])]
    if not hr.empty:
        ref = hr[hr.algorithm == "hit_rate_classical"].groupby("task_id")["m_estimate"].first()
        for _, row in hr.iterrows():
            if row.task_id not in ref:
                continue
            err = abs(row["m_estimate"] - ref[row.task_id])
            out.append({
                "estimand": "directional_hit_rate", "task_id": row.task_id, "config": row.config,
                "kind": row.kind, "repetition": row.repetition, "abs_error": err,
                "abs_error_vs_own_target": err,
                "ci_halfwidth": (row.get("m_ae_ci_high", np.nan) - row.get("m_ae_ci_low", np.nan)) / 2
                if row.kind == "quantum" else 0.0,
                "seconds": row.total_seconds,
                "samples_or_queries": row.get("c_oracle_queries") if row.kind == "quantum" else row.get("c_samples"),
            })
    return pd.DataFrame(out)


def quality_frontier(mq: pd.DataFrame) -> pd.DataFrame:
    """Median error vs median cost per config: the accuracy-cost frontier (proposal 6.4)."""
    if mq.empty:
        return mq
    g = mq.groupby(["estimand", "config", "kind"])
    return g.agg(
        median_abs_error=("abs_error", "median"),
        median_abs_error_vs_own_target=("abs_error_vs_own_target", "median"),
        p90_abs_error=("abs_error", lambda x: np.nanquantile(x.astype(float), 0.9)),
        median_seconds=("seconds", "median"),
        median_samples_or_queries=("samples_or_queries", "median"),
        runs=("abs_error", "size"),
    ).reset_index()
