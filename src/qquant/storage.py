"""SQLite run store. Raw per-run measurements are kept; aggregation happens at read time."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    name TEXT,
    manifest_hash TEXT,
    manifest_json TEXT,
    dataset_hash TEXT,
    environment_json TEXT,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    experiment_id TEXT,
    batch_id TEXT,
    task_id TEXT,
    hypothesis_id TEXT,
    workload TEXT,
    algorithm TEXT,
    algorithm_version TEXT,
    kind TEXT,
    experimental INTEGER,
    params_json TEXT,
    seed INTEGER,
    repetition INTEGER,
    backend TEXT,
    dataset_hash TEXT,
    started_at TEXT,
    finished_at TEXT,
    status TEXT,
    error TEXT,
    stages_json TEXT,
    counters_json TEXT,
    metrics_json TEXT,
    warnings_json TEXT,
    total_seconds REAL,
    peak_mem_mb REAL
);
CREATE TABLE IF NOT EXISTS batches (
    batch_id TEXT PRIMARY KEY,
    experiment_id TEXT,
    workload TEXT,
    algorithm TEXT,
    n_tasks INTEGER,
    repetition INTEGER,
    wall_seconds REAL,
    workers INTEGER
);
CREATE TABLE IF NOT EXISTS series (
    run_id TEXT,
    kind TEXT,
    date TEXT,
    value REAL
);
CREATE INDEX IF NOT EXISTS idx_runs_exp ON runs(experiment_id);
CREATE INDEX IF NOT EXISTS idx_series_run ON series(run_id);
"""


def _json(obj: Any) -> str:
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    return json.dumps(obj, default=default, sort_keys=True)


class RunStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------ writes
    def add_experiment(self, row: dict[str, Any]) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO experiments VALUES (?,?,?,?,?,?,?)",
                (
                    row["experiment_id"], row["name"], row["manifest_hash"], _json(row["manifest"]),
                    row["dataset_hash"], _json(row["environment"]), row["created_at"],
                ),
            )

    def add_run(self, rec: dict[str, Any]) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO runs VALUES (" + ",".join("?" * 25) + ")",
                (
                    rec["run_id"], rec["experiment_id"], rec.get("batch_id"), rec["task_id"],
                    rec.get("hypothesis_id"), rec["workload"], rec["algorithm"], rec["algorithm_version"],
                    rec["kind"], int(rec["experimental"]), _json(rec["params"]), rec["seed"],
                    rec["repetition"], rec["backend"], rec["dataset_hash"], rec["started_at"],
                    rec["finished_at"], rec["status"], rec.get("error"), _json(rec["stages"]),
                    _json(rec["counters"]), _json(rec["metrics"]), _json(rec["warnings"]),
                    rec["total_seconds"], rec.get("peak_mem_mb"),
                ),
            )
            series = rec.get("series") or {}
            rows = [
                (rec["run_id"], kind, str(idx), float(val))
                for kind, s in series.items()
                for idx, val in zip(s.index, s.values)
            ]
            if rows:
                c.executemany("INSERT INTO series VALUES (?,?,?,?)", rows)

    def add_batch(self, row: dict[str, Any]) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO batches VALUES (?,?,?,?,?,?,?,?)",
                (
                    row["batch_id"], row["experiment_id"], row["workload"], row["algorithm"],
                    row["n_tasks"], row["repetition"], row["wall_seconds"], row["workers"],
                ),
            )

    # ------------------------------------------------------------------ reads
    def experiments(self) -> pd.DataFrame:
        with self._conn() as c:
            return pd.read_sql("SELECT * FROM experiments ORDER BY created_at DESC", c)

    def runs(self, experiment_id: str | None = None, expand: bool = True) -> pd.DataFrame:
        q, args = "SELECT * FROM runs", ()
        if experiment_id:
            q, args = q + " WHERE experiment_id=?", (experiment_id,)
        with self._conn() as c:
            df = pd.read_sql(q, c, params=args)
        if expand and len(df):
            df = expand_json_columns(df)
        return df

    def batches(self, experiment_id: str | None = None) -> pd.DataFrame:
        q, args = "SELECT * FROM batches", ()
        if experiment_id:
            q, args = q + " WHERE experiment_id=?", (experiment_id,)
        with self._conn() as c:
            return pd.read_sql(q, c, params=args)

    def series(self, run_ids: Iterable[str], kind: str) -> pd.DataFrame:
        run_ids = list(run_ids)
        if not run_ids:
            return pd.DataFrame(columns=["run_id", "date", "value"])
        ph = ",".join("?" * len(run_ids))
        with self._conn() as c:
            return pd.read_sql(
                f"SELECT run_id, date, value FROM series WHERE kind=? AND run_id IN ({ph})",
                c, params=(kind, *run_ids),
            )


def expand_json_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten stages/metrics/counters JSON into prefixed columns for analysis."""
    out = df.copy()
    for col, prefix in (("stages_json", "t_"), ("metrics_json", "m_"), ("counters_json", "c_")):
        parsed = out[col].map(lambda s: json.loads(s) if s else {})
        flat = pd.DataFrame(list(parsed), index=out.index).add_prefix(prefix)
        out = pd.concat([out, flat], axis=1)
    out["params"] = out["params_json"].map(json.loads)
    out["warnings"] = out["warnings_json"].map(json.loads)
    return out
