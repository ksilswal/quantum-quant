"""Secondary NVDA signal research report: hypothesis-level results, multiple-testing
control, and post-hoc cross-signal correlation matrices.

Three different correlation objects are produced and must not be conflated:
  (i)   standardized signal values           -> do signals carry similar information?
  (ii)  out-of-sample predictions            -> do models built on them agree?
  (iii) hypothetical strategy returns        -> would positions co-move?
Correlation is neither complementarity nor causation; see `stability` for drift.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from ..storage import RunStore
from .benchmarks import config_label


def _canonical_runs(runs: pd.DataFrame, workload: str) -> pd.DataFrame:
    """One run per (hypothesis, config): repetition 0 of the largest batch."""
    r = runs[(runs.workload == workload) & (runs.status == "ok")].copy()
    if r.empty:
        return r
    r["config"] = config_label(r)
    r = r[r.repetition == 0]
    return r.drop_duplicates(["config", "task_id"], keep="last")


def hypothesis_table(runs: pd.DataFrame, fdr_method: str = "fdr_bh", alpha: float = 0.05) -> pd.DataFrame:
    """Statistical-workload estimates with unadjusted and adjusted p-values per config."""
    r = _canonical_runs(runs, "statistical")
    if r.empty:
        return r
    keep = ["task_id", "config", "kind", "m_estimate", "m_std_error", "m_ci_low", "m_ci_high", "m_p_value", "m_n_obs"]
    t = r[[c for c in keep if c in r]].rename(columns=lambda c: c.removeprefix("m_"))
    t["family"] = t["task_id"].str.split("-").str[0]
    parts = []
    for cfg, g in t.groupby("config"):
        g = g.copy()
        if "p_value" in g and g["p_value"].notna().sum() > 1:
            mask = g["p_value"].notna()
            rej, padj, _, _ = multipletests(g.loc[mask, "p_value"], alpha=alpha, method=fdr_method)
            g.loc[mask, "p_adjusted"] = padj
            g.loc[mask, "reject_adjusted"] = rej
            g["n_tests"] = int(mask.sum())
            g["n_families"] = g["family"].nunique()
        parts.append(g)
    return pd.concat(parts).sort_values(["config", "task_id"])


def predictive_table(runs: pd.DataFrame) -> pd.DataFrame:
    r = _canonical_runs(runs, "predictive")
    if r.empty:
        return r
    cols = ["task_id", "config", "kind"] + [c for c in r.columns if c.startswith("m_") and r[c].notna().any()]
    return r[cols].rename(columns=lambda c: c.removeprefix("m_"))


def _signal_panel(store: RunStore, sig_ids: list[str]) -> pd.DataFrame:
    """Dates x signal definitions. Hypotheses that differ only in horizon share a signal,
    so columns are keyed by family-lookback (e.g. momentum-k21) and deduplicated."""
    s = store.series(sig_ids, "signal")
    s["signal"] = s["run_id"].str.split("::").str[-1].str.rsplit("-", n=2).str[0]
    s = s.drop_duplicates(["signal", "date"])
    return s.pivot_table(index="date", columns="signal", values="value")


def correlation_matrices(store: RunStore, experiment_id: str, runs: pd.DataFrame, predictive_config: str | None = None) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    sig_ids = runs.loc[runs.workload == "signal", "run_id"].tolist()
    if sig_ids:
        out["signals"] = _signal_panel(store, sig_ids).corr(min_periods=60)
    pr = _canonical_runs(runs, "predictive")
    if not pr.empty:
        non_baseline = [c for c in pr["config"].unique() if not c.startswith("baseline")]
        cfg = predictive_config or (non_baseline or list(pr["config"].unique()))[0]
        sel = pr[pr.config == cfg]
        idmap = dict(zip(sel.run_id, sel.task_id))
        for kind, key in (("prediction", "predictions"), ("strategy", "strategy_returns")):
            s = store.series(sel.run_id, kind)
            if s.empty:
                continue
            s["hyp"] = s["run_id"].map(idmap)
            piv = s.pivot_table(index="date", columns="hyp", values="value")
            piv = piv.loc[:, piv.std() > 0]  # constant forecasts have undefined correlation
            out[key] = piv.corr(min_periods=30)
    return out


def partial_correlation(corr: pd.DataFrame, ridge: float = 1e-6) -> pd.DataFrame:
    c = corr.dropna(how="all").dropna(axis=1, how="all").fillna(0.0)
    P = np.linalg.pinv(c.values + ridge * np.eye(len(c)))
    d = np.sqrt(np.diag(P))
    pc = -P / np.outer(d, d)
    np.fill_diagonal(pc, 1.0)
    return pd.DataFrame(pc, index=c.index, columns=c.columns)


def stability(store: RunStore, runs: pd.DataFrame) -> pd.DataFrame:
    """Signal correlations in the first vs second half of the sample."""
    sig_ids = runs.loc[runs.workload == "signal", "run_id"].tolist()
    if not sig_ids:
        return pd.DataFrame()
    piv = _signal_panel(store, sig_ids).sort_index()
    half = len(piv) // 2
    c1, c2 = piv.iloc[:half].corr(min_periods=60), piv.iloc[half:].corr(min_periods=60)
    iu = np.triu_indices(len(c1), 1)
    return pd.DataFrame({
        "pair": [f"{c1.index[i]} | {c1.columns[j]}" for i, j in zip(*iu)],
        "corr_first_half": c1.values[iu], "corr_second_half": c2.values[iu],
    }).assign(abs_change=lambda d: (d.corr_second_half - d.corr_first_half).abs()).sort_values("abs_change", ascending=False)
