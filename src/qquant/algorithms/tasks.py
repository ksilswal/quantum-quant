"""Task inputs handed to algorithms."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..hypotheses.registry import Hypothesis


@dataclass
class SignalTask:
    """Statistical-testing and predictive-training tasks: one hypothesis on NVDA."""

    task_id: str
    hypothesis: Hypothesis
    data: pd.DataFrame  # columns x, y aligned on decision date t
    frame: pd.DataFrame  # full snapshot frame (for controls); never used for labels

    @property
    def horizon(self) -> int:
        return self.hypothesis.horizon


@dataclass
class MonteCarloTask:
    """Bounded expectation E[g(S_T)] under a documented toy GBM for NVDA."""

    task_id: str
    S0: float
    mu: float  # annualized arithmetic drift
    sigma: float  # annualized volatility
    horizon_days: int
    strike: float
    payoff: str  # digital | capped_call
    cap: float
    model: str = "gbm"
    notes: list[str] = field(default_factory=list)

    @property
    def T(self) -> float:
        return self.horizon_days / 252.0

    @property
    def log_mean(self) -> float:
        import numpy as np

        return float(np.log(self.S0) + (self.mu - 0.5 * self.sigma**2) * self.T)

    @property
    def log_sd(self) -> float:
        import numpy as np

        return float(self.sigma * np.sqrt(self.T))
