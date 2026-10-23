"""Return definitions with explicit timing conventions.

Convention (close-to-close, decision at close t):
    the signal x_t uses information available at the close of day t;
    the hypothetical position is entered at that close;
    the label is r_{t,t+h} = P_{t+h} / P_t - 1   (proposal eq. 1).

So x_t and r_{t,t+h} are aligned on the same index t, and the last h labels are NaN.
"""
from __future__ import annotations

import pandas as pd


def forward_return(close: pd.Series, h: int) -> pd.Series:
    if h < 1:
        raise ValueError("horizon h must be >= 1")
    out = close.shift(-h) / close - 1.0
    out.name = f"fwd_{h}d"
    return out


def trailing_return(close: pd.Series, k: int) -> pd.Series:
    return close / close.shift(k) - 1.0


def daily_returns(close: pd.Series) -> pd.Series:
    return close.pct_change()


def label_end_index(index: pd.DatetimeIndex, h: int) -> pd.Series:
    """Date on which each label r_{t,t+h} becomes known (used for purging/embargo)."""
    s = pd.Series(index, index=index)
    return s.shift(-h)
