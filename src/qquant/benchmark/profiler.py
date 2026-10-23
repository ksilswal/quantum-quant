"""Staged timing and resource accounting (proposal eq. 3).

    T_end-to-end = T_load + T_preprocess + T_compile + T_execute + T_postprocess

Algorithms wrap their phases in `with prof.stage("execute"):` and report resource
counters (`prof.count("circuit_executions", 1)`). Unknown stage names are allowed but
the five canonical stages are always present (zero if unused) so tables line up.

The extra `instrumentation` stage holds work done only to *report* resources (e.g.
decomposing a circuit to count gates). It is recorded but excluded from `end_to_end`.
"""
from __future__ import annotations

import time
import tracemalloc
from collections import defaultdict
from contextlib import contextmanager

STAGES = ("load", "preprocess", "compile", "execute", "postprocess")


class Profiler:
    def __init__(self, track_memory: bool = False):
        self.stages: dict[str, float] = {s: 0.0 for s in STAGES}
        self.counters: dict[str, float] = defaultdict(float)
        self.warnings: list[str] = []
        self.track_memory = track_memory
        self.peak_mem_mb: float | None = None
        self._t0: float | None = None
        self._total: float | None = None

    # --- whole-run bracket -------------------------------------------------
    def start(self) -> None:
        if self.track_memory:
            tracemalloc.start()
        self._t0 = time.perf_counter()

    def stop(self) -> None:
        self._total = time.perf_counter() - (self._t0 or time.perf_counter())
        if self.track_memory and tracemalloc.is_tracing():
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            self.peak_mem_mb = peak / 2**20
        untracked = self._total - sum(self.stages.values())
        self.stages["untracked"] = max(untracked, 0.0)

    @property
    def total(self) -> float:
        return self._total if self._total is not None else sum(self.stages.values())

    @property
    def end_to_end(self) -> float:
        """Total minus the `instrumentation` stage (resource counting done only for reporting)."""
        return self.total - self.stages.get("instrumentation", 0.0)

    # --- stages & counters ---------------------------------------------------
    @contextmanager
    def stage(self, name: str):
        t = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] = self.stages.get(name, 0.0) + time.perf_counter() - t

    def count(self, key: str, value: float = 1) -> None:
        self.counters[key] += value

    def set(self, key: str, value: float) -> None:
        self.counters[key] = value

    def maximum(self, key: str, value: float) -> None:
        self.counters[key] = max(self.counters.get(key, value), value)

    def warn(self, msg: str) -> None:
        if msg not in self.warnings:
            self.warnings.append(msg)
