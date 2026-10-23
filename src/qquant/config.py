"""Experiment manifests: the single reproducible description of a run.

A manifest fully determines an experiment (data snapshot request, hypothesis grid,
workloads, algorithms + parameters, seeds, resources). It is hashed and stored with
every run record so any result can be traced back to its exact configuration.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

WORKLOADS = ("statistical", "predictive", "montecarlo")


class ManifestError(ValueError):
    """Raised when a manifest is invalid or contains incompatible settings."""


@dataclass
class DataConfig:
    source: str = "synthetic"  # yfinance | csv | synthetic
    ticker: str = "NVDA"
    benchmarks: list[str] = field(default_factory=lambda: ["QQQ"])
    start: str = "2015-01-01"
    end: str = "2025-12-31"
    frequency: str = "1d"
    csv_path: str | None = None  # for source=csv: columns date,ticker,close,adj_close,volume
    cache_dir: str = "data_cache"
    synthetic_seed: int = 7


@dataclass
class HypothesisConfig:
    families: list[str] = field(
        default_factory=lambda: ["momentum", "mean_reversion", "volatility", "volume", "relative_strength"]
    )
    lookbacks: list[int] = field(default_factory=lambda: [5, 10, 21, 63, 126])
    horizons: list[int] = field(default_factory=lambda: [1, 5])
    n_hypotheses: int | None = None  # None = full grid; otherwise first N in registry order
    question: str = "association"


@dataclass
class AlgorithmSpec:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkloadSpec:
    enabled: bool = False
    algorithms: list[AlgorithmSpec] = field(default_factory=list)


@dataclass
class MonteCarloTaskConfig:
    n_tasks: int = 10
    moneyness: list[float] = field(default_factory=lambda: [0.9, 0.95, 1.0, 1.05, 1.1])
    horizons_days: list[int] = field(default_factory=lambda: [21, 63])
    payoff: str = "digital"  # digital = Pr(S_T > K); capped_call = E[min((S_T-K)+, cap)]/cap
    cap_fraction: float = 0.25  # cap = cap_fraction * S0 for capped_call
    model: str = "gbm"
    calibration_window_days: int = 252


@dataclass
class RunConfig:
    seed: int = 1234
    repetitions: int = 3
    warmups: int = 1
    timeout_s: float = 600.0
    workers: int = 1
    scaling_N: list[int] = field(default_factory=lambda: [10, 25, 50])
    backend: str = "default.qubit"
    db_path: str = "results/qquant.sqlite"
    export_format: str = "csv"
    max_qubits: int = 14  # hard resource cap for simulated circuits


@dataclass
class ExperimentManifest:
    name: str = "unnamed"
    description: str = ""
    data: DataConfig = field(default_factory=DataConfig)
    hypotheses: HypothesisConfig = field(default_factory=HypothesisConfig)
    statistical: WorkloadSpec = field(default_factory=WorkloadSpec)
    predictive: WorkloadSpec = field(default_factory=WorkloadSpec)
    montecarlo: WorkloadSpec = field(default_factory=WorkloadSpec)
    montecarlo_tasks: MonteCarloTaskConfig = field(default_factory=MonteCarloTaskConfig)
    run: RunConfig = field(default_factory=RunConfig)

    # ------------------------------------------------------------------ io
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExperimentManifest":
        return _build(cls, d, path="manifest")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentManifest":
        with open(path) as fh:
            return cls.from_dict(yaml.safe_load(fh) or {})

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_yaml(self, path: str | Path | None = None) -> str:
        text = yaml.safe_dump(self.to_dict(), sort_keys=False)
        if path is not None:
            Path(path).write_text(text)
        return text

    def hash(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def workload(self, name: str) -> WorkloadSpec:
        if name not in WORKLOADS:
            raise ManifestError(f"Unknown workload {name!r}")
        return getattr(self, name)

    def enabled_workloads(self) -> list[str]:
        return [w for w in WORKLOADS if self.workload(w).enabled]


def _build(cls, d: Any, path: str):
    """Strict recursive dataclass construction: unknown keys are errors, not silently dropped."""
    if not dataclasses.is_dataclass(cls):
        return d
    if d is None:
        return cls()
    if not isinstance(d, dict):
        raise ManifestError(f"{path}: expected a mapping, got {type(d).__name__}")
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(d) - set(fields)
    if unknown:
        raise ManifestError(f"{path}: unknown key(s) {sorted(unknown)}")
    kwargs = {}
    hints = {
        "data": DataConfig,
        "hypotheses": HypothesisConfig,
        "statistical": WorkloadSpec,
        "predictive": WorkloadSpec,
        "montecarlo": WorkloadSpec,
        "montecarlo_tasks": MonteCarloTaskConfig,
        "run": RunConfig,
    }
    for k, v in d.items():
        sub = f"{path}.{k}"
        if cls is ExperimentManifest and k in hints:
            kwargs[k] = _build(hints[k], v, sub)
        elif cls is WorkloadSpec and k == "algorithms":
            if not isinstance(v, list):
                raise ManifestError(f"{sub}: expected a list")
            specs = []
            for i, a in enumerate(v):
                if isinstance(a, str):
                    a = {"name": a}
                specs.append(_build(AlgorithmSpec, a, f"{sub}[{i}]"))
            kwargs[k] = specs
        else:
            kwargs[k] = v
    return cls(**kwargs)
