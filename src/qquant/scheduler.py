"""Experiment scheduler: manifest -> validated task batches -> profiled runs -> run store.

Batching: for each enabled workload, algorithm and batch size N in `run.scaling_N`
(capped at the number of available tasks), the first N tasks are run `repetitions`
times after `warmups` unrecorded warm-up runs. Every task gets a seed derived from
(base seed, task id, repetition) only, so all algorithms and all batch sizes see common
random numbers and identical inputs.

Concurrency: `workers > 1` uses a process pool. `timeout_s` is enforced per task in
pool mode (the task is reported as `timeout`; the worker is not killed mid-call). In
sequential mode overruns are flagged after the fact.
"""
from __future__ import annotations

import hashlib
import platform
import time
import traceback
import uuid
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutTimeout
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from typing import Callable

import numpy as np
import pandas as pd

from . import algorithms as algos
from .algorithms.base import RunContext
from .algorithms.tasks import MonteCarloTask, SignalTask
from .benchmark.profiler import Profiler
from .config import ExperimentManifest, ManifestError
from .data import DatasetSnapshot, load_snapshot
from .hypotheses import Hypothesis, generate
from .storage import RunStore

Progress = Callable[[int, int, str], None]


def environment_info(backend: str) -> dict:
    pkgs = {}
    for name in ("numpy", "pandas", "scipy", "statsmodels", "scikit-learn", "pennylane", "qquant"):
        try:
            pkgs[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pkgs[name] = None
    try:
        import os

        cpus = os.cpu_count()
    except Exception:  # pragma: no cover
        cpus = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count": cpus,
        "quantum_backend": backend,
        "quantum_hardware": "classical simulator (no QPU)" if backend.startswith(("default.", "lightning.")) else backend,
        "packages": pkgs,
    }


def derive_seed(base: int, task_id: str, rep: int) -> int:
    h = hashlib.sha256(f"{base}|{task_id}|{rep}".encode()).hexdigest()
    return int(h[:8], 16)


# ----------------------------------------------------------------------------- tasks
def signal_tasks(hyps: list[Hypothesis], snap: DatasetSnapshot) -> list[SignalTask]:
    out = []
    for h in hyps:
        d = h.dataset(snap.frame)
        if len(d) < 100:
            raise ManifestError(f"{h.id}: only {len(d)} usable observations in the window")
        out.append(SignalTask(task_id=h.id, hypothesis=h, data=d, frame=snap.frame))
    return out


def montecarlo_tasks(m: ExperimentManifest, snap: DatasetSnapshot) -> list[MonteCarloTask]:
    cfg = m.montecarlo_tasks
    if cfg.model != "gbm":
        raise ManifestError("only model=gbm is implemented")
    close = snap.frame["close"].dropna()
    lr = np.log(close).diff().dropna().iloc[-cfg.calibration_window_days:]
    sigma = float(lr.std(ddof=1) * np.sqrt(252))
    mu = float(lr.mean() * 252 + 0.5 * sigma**2)
    S0 = float(close.iloc[-1])
    grid = [(h, k) for h in cfg.horizons_days for k in cfg.moneyness]
    if cfg.n_tasks > len(grid):
        raise ManifestError(
            f"montecarlo_tasks.n_tasks={cfg.n_tasks} exceeds horizons x moneyness = {len(grid)}"
        )
    notes = [
        f"Toy GBM calibrated on last {len(lr)} daily log returns ending {close.index[-1].date()}: "
        f"mu={mu:.4f}, sigma={sigma:.4f} (annualized). Not a validated NVDA model."
    ]
    tasks = []
    for h, k in grid[: cfg.n_tasks]:
        tid = f"mc-{cfg.payoff}-h{h}-m{k:g}"
        tasks.append(MonteCarloTask(
            task_id=tid, S0=S0, mu=mu, sigma=sigma, horizon_days=int(h), strike=k * S0,
            payoff=cfg.payoff, cap=cfg.cap_fraction * S0, notes=notes,
        ))
    return tasks


# ----------------------------------------------------------------------------- planning
@dataclass
class PlannedAlgo:
    workload: str
    name: str
    params: dict
    algo: algos.Algorithm


def plan(m: ExperimentManifest) -> list[PlannedAlgo]:
    """Validate every algorithm/parameter combination before anything runs."""
    if not m.enabled_workloads():
        raise ManifestError("No workload enabled")
    ctx = RunContext(seed=m.run.seed, backend=m.run.backend, max_qubits=m.run.max_qubits)
    out, errors = [], []
    for w in m.enabled_workloads():
        spec = m.workload(w)
        if not spec.algorithms:
            errors.append(f"{w}: enabled but no algorithms selected")
        for a in spec.algorithms:
            try:
                algo = algos.get(a.name)
                if algo.workload != w:
                    raise ManifestError(f"{a.name} is a {algo.workload} algorithm, listed under {w}")
                out.append(PlannedAlgo(w, a.name, algos.resolve_params(algo, a.params, ctx), algo))
            except ManifestError as e:
                errors.append(str(e))
    if m.run.workers < 1:
        errors.append("run.workers must be >= 1")
    if m.run.repetitions < 1:
        errors.append("run.repetitions must be >= 1")
    if errors:
        raise ManifestError("Invalid manifest:\n  - " + "\n  - ".join(errors))
    return out


# ----------------------------------------------------------------------------- execution
def execute_one(algo_name: str, params: dict, task, ctx: RunContext, track_memory: bool = False) -> dict:
    """Run one algorithm on one task with staged profiling. Never raises."""
    algo = algos.get(algo_name)
    prof = Profiler(track_memory=track_memory)
    started = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    status, err, out = "ok", None, None
    prof.start()
    try:
        out = algo.run(task, params, prof, ctx)
    except Exception as e:  # recorded, not swallowed silently
        status = "failed"
        err = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"
    prof.stop()
    return {
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "status": status, "error": err,
        "stages": prof.stages, "counters": dict(prof.counters), "warnings": prof.warnings,
        "metrics": out.metrics if out else {}, "series": out.series if out else {},
        "total_seconds": prof.end_to_end, "peak_mem_mb": prof.peak_mem_mb,
    }


def run_experiment(
    m: ExperimentManifest,
    store: RunStore | None = None,
    progress: Progress | None = None,
    snapshot: DatasetSnapshot | None = None,
) -> str:
    planned = plan(m)
    store = store or RunStore(m.run.db_path)
    progress = progress or (lambda i, n, msg: None)

    t_load = time.perf_counter()
    snap = snapshot or load_snapshot(m.data)
    t_load = time.perf_counter() - t_load
    t_build = time.perf_counter()
    hyps = generate(m.hypotheses, benchmark=(m.data.benchmarks or ["QQQ"])[0])
    tasks: dict[str, list] = {}
    if {"statistical", "predictive"} & set(m.enabled_workloads()):
        st = signal_tasks(hyps, snap)
        tasks["statistical"] = tasks["predictive"] = st
    if "montecarlo" in m.enabled_workloads():
        tasks["montecarlo"] = montecarlo_tasks(m, snap)
    t_build = time.perf_counter() - t_build

    exp_id = f"{m.name}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{m.hash()[:6]}"
    env = environment_info(m.run.backend)
    store.add_experiment({
        "experiment_id": exp_id, "name": m.name, "manifest_hash": m.hash(), "manifest": m.to_dict(),
        "dataset_hash": snap.hash,
        # shared costs, identical for every algorithm, so reported once rather than per run
        "environment": {**env, "snapshot": snap.meta.__dict__,
                        "shared_costs_seconds": {"snapshot_load": t_load, "task_build": t_build}},
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })

    # standardized signal values, stored once per hypothesis for the correlation report
    if "statistical" in tasks or "predictive" in tasks:
        for t in tasks.get("statistical", []):
            x = t.data["x"]
            store.add_run({
                "run_id": f"signal::{exp_id}::{t.task_id}", "experiment_id": exp_id, "batch_id": None,
                "task_id": t.task_id, "hypothesis_id": t.task_id, "workload": "signal", "algorithm": "signal",
                "algorithm_version": "-", "kind": "data", "experimental": False, "params": {}, "seed": 0,
                "repetition": 0, "backend": "-", "dataset_hash": snap.hash, "started_at": "", "finished_at": "",
                "status": "ok", "stages": {}, "counters": {}, "metrics": {}, "warnings": [],
                "total_seconds": 0.0, "series": {"signal": (x - x.mean()) / x.std()},
            })

    # schedule
    jobs = []
    for pa in planned:
        avail = tasks[pa.workload]
        sizes = sorted({min(n, len(avail)) for n in (m.run.scaling_N or [len(avail)])})
        for N in sizes:
            for rep in range(m.run.repetitions):
                jobs.append((pa, N, rep))
    total_runs = sum(N for _, N, _ in jobs)
    done = 0

    pool = ProcessPoolExecutor(max_workers=m.run.workers) if m.run.workers > 1 else None
    try:
        warmed: set[str] = set()
        for pa, N, rep in jobs:
            batch_tasks = tasks[pa.workload][:N]
            if pa.name not in warmed:
                for t in batch_tasks[: m.run.warmups]:
                    execute_one(pa.name, pa.params, t, RunContext(seed=0, backend=m.run.backend, max_qubits=m.run.max_qubits))
                warmed.add(pa.name)
            batch_id = uuid.uuid4().hex[:12]
            ctxs = [
                RunContext(seed=derive_seed(m.run.seed, t.task_id, rep), backend=m.run.backend,
                           max_qubits=m.run.max_qubits, repetition=rep)
                for t in batch_tasks
            ]
            t0 = time.perf_counter()
            if pool is None:
                results = [execute_one(pa.name, pa.params, t, c) for t, c in zip(batch_tasks, ctxs)]
            else:
                futs = [pool.submit(execute_one, pa.name, pa.params, t, c) for t, c in zip(batch_tasks, ctxs)]
                results = []
                for f in futs:
                    try:
                        results.append(f.result(timeout=m.run.timeout_s))
                    except FutTimeout:
                        results.append(_timeout_record(m.run.timeout_s))
            wall = time.perf_counter() - t0
            for t, c, r in zip(batch_tasks, ctxs, results):
                if r["status"] == "ok" and r["total_seconds"] > m.run.timeout_s:
                    r["warnings"].append(f"exceeded timeout_s={m.run.timeout_s} (sequential mode: not interrupted)")
                store.add_run({
                    "run_id": uuid.uuid4().hex, "experiment_id": exp_id, "batch_id": batch_id,
                    "task_id": t.task_id,
                    "hypothesis_id": t.task_id if isinstance(t, SignalTask) else None,
                    "workload": pa.workload, "algorithm": pa.name, "algorithm_version": pa.algo.version,
                    "kind": pa.algo.kind, "experimental": pa.algo.experimental, "params": pa.params,
                    "seed": c.seed, "repetition": rep, "backend": m.run.backend if pa.algo.kind == "quantum" else "cpu",
                    "dataset_hash": snap.hash, **r,
                })
                done += 1
            store.add_batch({
                "batch_id": batch_id, "experiment_id": exp_id, "workload": pa.workload, "algorithm": pa.name,
                "n_tasks": N, "repetition": rep, "wall_seconds": wall, "workers": m.run.workers,
            })
            progress(done, total_runs, f"{pa.workload}/{pa.name} N={N} rep={rep + 1}: {wall:.2f}s")
    finally:
        if pool is not None:
            pool.shutdown(cancel_futures=True)
    return exp_id


def _timeout_record(limit: float) -> dict:
    now = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    return {
        "started_at": now, "finished_at": now, "status": "timeout", "error": f"exceeded {limit}s",
        "stages": {}, "counters": {}, "warnings": [], "metrics": {}, "series": {},
        "total_seconds": float(limit), "peak_mem_mb": None,
    }
