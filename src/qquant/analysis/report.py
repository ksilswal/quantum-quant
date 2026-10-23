"""Exportable experiment reports: raw measurements plus every aggregate table."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from ..storage import RunStore
from . import benchmarks as bm
from . import signals as sg


def experiment_tables(store: RunStore, experiment_id: str) -> dict[str, pd.DataFrame]:
    runs = store.runs(experiment_id)
    batches = store.batches(experiment_id)
    tables = {
        "runs_raw": runs.drop(columns=["params", "warnings"], errors="ignore"),
        "batches_raw": batches,
        "run_summary": bm.run_summary(runs),
        "scaling": bm.scaling_table(batches),
    }
    mq = bm.matched_quality(runs)
    tables["matched_quality"] = mq
    tables["quality_frontier"] = bm.quality_frontier(mq)
    tables["hypotheses_statistical"] = sg.hypothesis_table(runs)
    tables["hypotheses_predictive"] = sg.predictive_table(runs)
    for name, c in sg.correlation_matrices(store, experiment_id, runs).items():
        tables[f"corr_{name}"] = c
    return tables


def export(store: RunStore, experiment_id: str, out_dir: str | Path, fmt: str = "csv") -> Path:
    out = Path(out_dir) / experiment_id
    out.mkdir(parents=True, exist_ok=True)
    exp = store.experiments().set_index("experiment_id").loc[experiment_id]
    (out / "manifest.yaml").write_text(yaml.safe_dump(json.loads(exp["manifest_json"]), sort_keys=False))
    (out / "environment.json").write_text(json.dumps(json.loads(exp["environment_json"]), indent=2))
    for name, df in experiment_tables(store, experiment_id).items():
        if df is None or df.empty:
            continue
        idx = name.startswith("corr_")
        if fmt == "json":
            df.to_json(out / f"{name}.json", orient="split" if idx else "records", indent=1)
        else:
            df.to_csv(out / f"{name}.csv", index=idx)
    return out
