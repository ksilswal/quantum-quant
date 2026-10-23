"""Hypothesis registry: fully specified, uniquely identified experimental units.

A hypothesis = signal definition + parameters + target horizon + statistical question
+ data window + evaluation rule. Parameterizations of one family are strongly
correlated; `cluster` records this so that later inference does not treat 50 related
jobs as 50 independent pieces of evidence.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import asdict, dataclass, field

import pandas as pd

from ..config import HypothesisConfig
from ..data.returns import forward_return
from .signals import SIGNALS, compute_signal

QUESTIONS = {
    "association": "H0: b_j = 0 in y_t = a_j + b_j x_t + e_t (y = h-day forward NVDA return)",
}


@dataclass(frozen=True)
class Hypothesis:
    family: str
    lookback: int
    horizon: int
    question: str = "association"
    benchmark: str | None = None
    window_start: str | None = None
    window_end: str | None = None
    eval_rule: str = "two_sided_5pct"
    extra: tuple = field(default_factory=tuple)

    @property
    def id(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True, default=str).encode()
        return f"{self.family}-k{self.lookback}-h{self.horizon}-{hashlib.sha1(blob).hexdigest()[:6]}"

    @property
    def cluster(self) -> str:
        return self.family

    def describe(self) -> str:
        meta = SIGNALS[self.family]
        return f"{meta['formula']} (k={self.lookback}) -> r(t,t+{self.horizon}); {meta['hypothesis']}"

    def dataset(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Aligned (x_t, y_t) with NaNs dropped. y_t = r_{t,t+h}; x_t uses data <= t."""
        kw = {"benchmark": self.benchmark} if self.benchmark else {}
        x = compute_signal(frame, self.family, self.lookback, **kw)
        y = forward_return(frame["close"], self.horizon)
        df = pd.DataFrame({"x": x, "y": y}).loc[self.window_start:self.window_end]
        return df.dropna()


def generate(cfg: HypothesisConfig, benchmark: str = "QQQ", window: tuple[str | None, str | None] = (None, None)) -> list[Hypothesis]:
    """Deterministic grid. Order interleaves families so any prefix of length N is diverse."""
    unknown = set(cfg.families) - set(SIGNALS)
    if unknown:
        raise KeyError(f"Unknown signal families: {sorted(unknown)}")
    if cfg.question not in QUESTIONS:
        raise KeyError(f"Unknown question {cfg.question!r}")
    out: list[Hypothesis] = []
    for k, h, fam in itertools.product(cfg.lookbacks, cfg.horizons, cfg.families):
        if fam in ("mean_reversion", "volatility") and k < 2:
            continue
        out.append(
            Hypothesis(
                family=fam, lookback=int(k), horizon=int(h), question=cfg.question,
                benchmark=benchmark if SIGNALS[fam]["needs_benchmark"] else None,
                window_start=window[0], window_end=window[1],
            )
        )
    ids = [h.id for h in out]
    assert len(ids) == len(set(ids)), "hypothesis id collision"
    if cfg.n_hypotheses is not None:
        if cfg.n_hypotheses > len(out):
            raise ValueError(
                f"Requested {cfg.n_hypotheses} hypotheses but grid has only {len(out)}; "
                "add families/lookbacks/horizons."
            )
        out = out[: cfg.n_hypotheses]
    return out


def registry_table(hyps: list[Hypothesis]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"hypothesis_id": h.id, "cluster": h.cluster, **asdict(h), "description": h.describe()} for h in hyps]
    )
