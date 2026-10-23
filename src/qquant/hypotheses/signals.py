"""Signal families. Every signal is causal: x_t depends only on data at dates <= t.

Add a family by decorating a function `f(frame, lookback, **kw) -> pd.Series`.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

SignalFn = Callable[..., pd.Series]
SIGNALS: dict[str, dict] = {}


def signal(name: str, formula: str, hypothesis: str, needs_benchmark: bool = False):
    def deco(fn: SignalFn) -> SignalFn:
        SIGNALS[name] = {
            "fn": fn, "formula": formula, "hypothesis": hypothesis,
            "needs_benchmark": needs_benchmark,
        }
        return fn

    return deco


@signal("momentum", "M_t^(k) = P_t / P_{t-k} - 1", "Trend persistence over horizon h")
def momentum(frame: pd.DataFrame, k: int, **_) -> pd.Series:
    p = frame["close"]
    return p / p.shift(k) - 1.0


@signal("mean_reversion", "Z_t = (P_t - mean_{t,k}(P)) / sd_{t,k}(P)", "Large deviations reverse")
def mean_reversion(frame: pd.DataFrame, k: int, **_) -> pd.Series:
    if k < 2:
        raise ValueError("mean_reversion needs lookback >= 2")
    p = frame["close"]
    roll = p.rolling(k, min_periods=k)
    return (p - roll.mean()) / roll.std(ddof=1)


@signal("volatility", "sigma_hat_{t,k} = sd of daily returns over k days (annualized)",
        "Volatility conditions predictability")
def volatility(frame: pd.DataFrame, k: int, **_) -> pd.Series:
    if k < 2:
        raise ValueError("volatility needs lookback >= 2")
    r = frame["close"].pct_change()
    return r.rolling(k, min_periods=k).std(ddof=1) * np.sqrt(252)


@signal("vol_scaled_momentum", "M_t^(k) / (sigma_hat_{t,k} * sqrt(k/252))",
        "Risk-adjusted trend persists")
def vol_scaled_momentum(frame: pd.DataFrame, k: int, **_) -> pd.Series:
    sig = volatility(frame, max(k, 2)) * np.sqrt(k / 252)
    return momentum(frame, k) / sig


@signal("volume", "V_t / mean_{t,k}(V)", "Abnormal volume modifies returns")
def abnormal_volume(frame: pd.DataFrame, k: int, **_) -> pd.Series:
    v = frame["volume"]
    return v / v.rolling(k, min_periods=k).mean()


@signal("relative_strength", "M_{t,NVDA}^(k) - M_{t,BENCH}^(k)", "Relative performance persists",
        needs_benchmark=True)
def relative_strength(frame: pd.DataFrame, k: int, benchmark: str = "QQQ", **_) -> pd.Series:
    col = f"bench_{benchmark}"
    if col not in frame:
        raise KeyError(f"relative_strength needs benchmark column {col}")
    b = frame[col]
    return momentum(frame, k) - (b / b.shift(k) - 1.0)


def compute_signal(frame: pd.DataFrame, family: str, lookback: int, **kw) -> pd.Series:
    if family not in SIGNALS:
        raise KeyError(f"Unknown signal family {family!r}; have {sorted(SIGNALS)}")
    s = SIGNALS[family]["fn"](frame, lookback, **kw)
    s = s.replace([np.inf, -np.inf], np.nan)
    s.name = f"{family}_{lookback}"
    return s
