"""Acceptance: 50 NVDA hypotheses end-to-end from a manifest, no code edits; all three
workloads run; exports contain timing breakdowns, manifests and correlation matrices."""
import json

import pandas as pd

from qquant.analysis.report import export
from qquant.config import ExperimentManifest
from qquant.scheduler import run_experiment
from qquant.storage import RunStore


def test_fifty_hypotheses_all_workloads(tmp_path, snapshot):
    m = ExperimentManifest.from_dict({
        "name": "e2e",
        "hypotheses": {},  # default grid = 50
        "statistical": {"enabled": True, "algorithms": ["ols_robust", "hit_rate_classical", {"name": "hit_rate_qae", "params": {"n_powers": 3}}]},
        "predictive": {"enabled": True, "algorithms": ["baseline_mean", "linear", {"name": "quantum_kernel", "params": {"max_train_samples": 60}}]},
        "montecarlo": {"enabled": True, "algorithms": ["mc_plain", {"name": "qae_mlae", "params": {"n_qubits": 3, "n_powers": 3}}]},
        "montecarlo_tasks": {"n_tasks": 4},
        "run": {"repetitions": 2, "warmups": 0, "scaling_N": [10, 50], "db_path": str(tmp_path / "r.sqlite")},
    })
    store = RunStore(m.run.db_path)
    exp = run_experiment(m, store, snapshot=snapshot)
    runs = store.runs(exp)
    real = runs[runs.workload != "signal"]
    assert (real.status == "ok").all(), real.loc[real.status != "ok", "error"].tolist()
    stat = real[(real.workload == "statistical") & (real.algorithm == "ols_robust")]
    assert stat.task_id.nunique() == 50
    assert set(real.workload) == {"statistical", "predictive", "montecarlo"}
    # identical seeds and inputs across batch sizes -> identical estimates
    q = real[real.algorithm == "hit_rate_qae"]
    per = q.groupby(["task_id", "repetition"])["m_estimate"].nunique()
    assert (per == 1).all()
    assert {"t_load", "t_preprocess", "t_compile", "t_execute", "t_postprocess"} <= set(runs.columns)

    out = export(store, exp, tmp_path / "exports")
    for f in ("manifest.yaml", "environment.json", "runs_raw.csv", "scaling.csv", "run_summary.csv",
              "quality_frontier.csv", "hypotheses_statistical.csv", "corr_signals.csv"):
        assert (out / f).exists(), f
    env = json.loads((out / "environment.json").read_text())
    assert env["snapshot"]["sha256"] == snapshot.hash
    hs = pd.read_csv(out / "hypotheses_statistical.csv")
    assert {"p_value", "p_adjusted"} <= set(hs.columns)
