"""Algorithm registry.

Each algorithm declares its workload, classical/quantum kind, tunable parameters (with
types, bounds and defaults — the UI renders widgets from these), an `estimand` tag, and
whether it is experimental. Two algorithms with the same `estimand` compute the same
target quantity on the same input, so their outputs can be compared at matched quality
(e.g. the classical and QAE estimates of the same Monte Carlo expectation). Algorithms
with different estimands are "different-model" comparisons and are labeled as such.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import pandas as pd

from ..benchmark.profiler import Profiler
from ..config import ManifestError


@dataclass
class Param:
    name: str
    type: type
    default: Any
    help: str = ""
    choices: tuple | None = None
    min: float | None = None
    max: float | None = None

    def coerce(self, value: Any, algo: str) -> Any:
        try:
            if self.type is bool and isinstance(value, str):
                v = value.lower() in ("1", "true", "yes")
            elif self.type in (list, tuple):
                v = list(value)
            else:
                v = self.type(value)
        except (TypeError, ValueError) as e:
            raise ManifestError(f"{algo}.{self.name}: cannot interpret {value!r} as {self.type.__name__}") from e
        if self.choices is not None and v not in self.choices:
            raise ManifestError(f"{algo}.{self.name}: {v!r} not in {self.choices}")
        if self.min is not None and v < self.min:
            raise ManifestError(f"{algo}.{self.name}: {v} < min {self.min}")
        if self.max is not None and v > self.max:
            raise ManifestError(f"{algo}.{self.name}: {v} > max {self.max}")
        return v


@dataclass
class RunContext:
    seed: int
    backend: str = "default.qubit"
    max_qubits: int = 14
    repetition: int = 0


@dataclass
class AlgoOutput:
    metrics: dict[str, Any] = field(default_factory=dict)
    series: dict[str, pd.Series] = field(default_factory=dict)


class Algorithm:
    name: ClassVar[str]
    workload: ClassVar[str]
    kind: ClassVar[str]  # "classical" | "quantum"
    version: ClassVar[str] = "0.1"
    estimand: ClassVar[str] = ""
    experimental: ClassVar[bool] = False
    description: ClassVar[str] = ""
    params: ClassVar[list[Param]] = []

    def check(self, p: dict[str, Any], ctx: RunContext) -> None:
        """Cross-parameter / resource validation. Raise ManifestError on incompatibility."""

    def run(self, task: Any, p: dict[str, Any], prof: Profiler, ctx: RunContext) -> AlgoOutput:
        raise NotImplementedError


REGISTRY: dict[str, type[Algorithm]] = {}


def register(cls: type[Algorithm]) -> type[Algorithm]:
    if cls.name in REGISTRY:
        raise RuntimeError(f"duplicate algorithm name {cls.name}")
    if cls.kind not in ("classical", "quantum"):
        raise RuntimeError(f"{cls.name}: kind must be classical|quantum")
    REGISTRY[cls.name] = cls
    return cls


def get(name: str) -> Algorithm:
    if name not in REGISTRY:
        raise ManifestError(f"Unknown algorithm {name!r}. Available: {sorted(REGISTRY)}")
    return REGISTRY[name]()


def resolve_params(algo: Algorithm, given: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
    known = {p.name: p for p in algo.params}
    unknown = set(given) - set(known)
    if unknown:
        raise ManifestError(f"{algo.name}: unknown parameter(s) {sorted(unknown)}; accepts {sorted(known)}")
    resolved = {n: p.coerce(given.get(n, p.default), algo.name) for n, p in known.items()}
    algo.check(resolved, ctx)
    return resolved


def catalog(workload: str | None = None) -> pd.DataFrame:
    rows = []
    for name, cls in sorted(REGISTRY.items()):
        if workload and cls.workload != workload:
            continue
        rows.append(
            {
                "algorithm": name, "workload": cls.workload, "kind": cls.kind,
                "estimand": cls.estimand, "experimental": cls.experimental,
                "version": cls.version, "description": cls.description,
                "params": ", ".join(f"{p.name}={p.default}" for p in cls.params),
            }
        )
    return pd.DataFrame(rows)
