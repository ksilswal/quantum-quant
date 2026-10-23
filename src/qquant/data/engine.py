"""Data engine: ingestion, adjustment metadata, immutable content-hashed snapshots.

A snapshot is a daily frame indexed by trading date with columns
    close        corporate-action-adjusted close of the target (NVDA)
    raw_close    unadjusted close (when the vendor supplies it)
    volume       share volume (as reported by the vendor; see adjustment notes)
    bench_<TKR>  adjusted closes of benchmark series (explanatory only; never traded)

Every snapshot carries provenance metadata and a SHA-256 hash of its canonical CSV
encoding. Experiments reference snapshots by hash, so a run can always be tied to the
exact bytes it saw.

Point-in-time caveat (documented, not solved): vendor "adjusted close" series are
back-adjusted *after the fact* (e.g. NVDA's 4:1 split in 2021 and 10:1 split in 2024).
Ratios such as returns and P_t/P_{t-k} are invariant to proportional back-adjustment,
so price-ratio signals are unaffected; level-based features (raw price thresholds,
dollar volume) are not, and should be built from raw_close with explicit split factors.
"""
from __future__ import annotations

import hashlib
import io
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import DataConfig


@dataclass
class SnapshotMeta:
    ticker: str
    benchmarks: list[str]
    source: str
    vendor: str
    retrieved_at_utc: str
    start: str
    end: str
    frequency: str
    calendar: str
    timezone: str
    adjustment: str
    n_rows: int
    missing_observations: dict[str, int]
    notes: list[str] = field(default_factory=list)
    sha256: str = ""
    version: str = "v1"


@dataclass
class DatasetSnapshot:
    frame: pd.DataFrame
    meta: SnapshotMeta

    @property
    def hash(self) -> str:
        return self.meta.sha256

    @property
    def short_hash(self) -> str:
        return self.meta.sha256[:12]

    def benchmark(self, ticker: str) -> pd.Series:
        col = f"bench_{ticker}"
        if col not in self.frame:
            raise KeyError(f"Benchmark {ticker} not in snapshot (have {self.meta.benchmarks})")
        return self.frame[col]

    def window(self, start: str | None = None, end: str | None = None) -> "DatasetSnapshot":
        """Date-restricted view. Hash is inherited; the window is recorded by the caller."""
        return DatasetSnapshot(self.frame.loc[start:end], self.meta)


def frame_hash(frame: pd.DataFrame) -> str:
    buf = io.StringIO()
    frame.to_csv(buf, float_format="%.10g", date_format="%Y-%m-%d")
    return hashlib.sha256(buf.getvalue().encode()).hexdigest()


# ---------------------------------------------------------------------- sources
def _load_synthetic(cfg: DataConfig) -> tuple[pd.DataFrame, dict]:
    """Regime-switching GBM with NVDA-like scale. For tests and offline development ONLY.

    The synthetic target has a small *planted* momentum effect (see `planted_beta`) so
    that pipelines can be validated against a known ground truth.
    """
    rng = np.random.default_rng(cfg.synthetic_seed)
    dates = pd.bdate_range(cfg.start, cfg.end, name="date")
    n = len(dates)
    # volatility regimes
    regime = np.zeros(n, dtype=int)
    for t in range(1, n):
        regime[t] = regime[t - 1] if rng.random() > 0.01 else 1 - regime[t - 1]
    vol = np.where(regime == 0, 0.022, 0.040)
    mkt = rng.normal(0.0004, 0.012, n)
    eps = rng.normal(0, 1, n) * vol
    planted_beta = 0.04  # next-day return loads on trailing 5d return
    r = np.zeros(n)
    for t in range(n):
        mom5 = r[max(0, t - 5):t].sum() if t > 0 else 0.0
        r[t] = 0.0008 + 1.4 * mkt[t] + planted_beta * mom5 + eps[t]
    close = 3.0 * np.exp(np.cumsum(r))
    qqq = 100.0 * np.exp(np.cumsum(mkt))
    volume = np.exp(rng.normal(17.5, 0.25, n) + 8 * np.abs(r))
    frame = pd.DataFrame(
        {"close": close, "raw_close": close, "volume": volume, "bench_QQQ": qqq}, index=dates
    )
    for b in cfg.benchmarks:
        if f"bench_{b}" not in frame:
            frame[f"bench_{b}"] = 100.0 * np.exp(np.cumsum(mkt + rng.normal(0, 0.004, n)))
    info = {
        "vendor": "synthetic (qquant regime-switching GBM)",
        "adjustment": "none required (synthetic)",
        "calendar": "pandas business days (no exchange holidays)",
        "timezone": "naive daily",
        "notes": [
            f"SYNTHETIC DATA, seed={cfg.synthetic_seed}. Not NVDA. Planted 5d-momentum beta={planted_beta}.",
        ],
    }
    return frame, info


def _load_yfinance(cfg: DataConfig) -> tuple[pd.DataFrame, dict]:
    try:
        import yfinance as yf
    except ImportError as e:  # pragma: no cover
        raise ImportError("pip install 'qquant[data]' to use the yfinance source") from e
    tickers = [cfg.ticker, *cfg.benchmarks]
    raw = yf.download(
        tickers, start=cfg.start, end=cfg.end, interval=cfg.frequency,
        auto_adjust=False, actions=False, progress=False, group_by="column",
    )
    if raw.empty:
        raise RuntimeError(f"yfinance returned no data for {tickers}")
    tgt_adj = raw["Adj Close"][cfg.ticker]
    frame = pd.DataFrame(
        {
            "close": tgt_adj,
            "raw_close": raw["Close"][cfg.ticker],
            "volume": raw["Volume"][cfg.ticker].astype(float),
        }
    )
    for b in cfg.benchmarks:
        frame[f"bench_{b}"] = raw["Adj Close"][b]
    frame = frame[frame["close"].notna()]  # target trading calendar defines the index
    frame.index = pd.DatetimeIndex(frame.index.date, name="date")
    info = {
        "vendor": f"Yahoo Finance via yfinance {getattr(yf, '__version__', '?')}",
        "adjustment": "Adj Close = split- and dividend-back-adjusted at retrieval time",
        "calendar": f"{cfg.ticker} trading days (exchange calendar as reported by vendor)",
        "timezone": "America/New_York session dates",
        "notes": [
            "Adjusted prices are revised retroactively; ratio-based signals are invariant to this.",
            "Volume is vendor-reported (split-adjusted by Yahoo); verify before level comparisons.",
            "Close-to-close convention: a signal computed from the close at t is assumed tradable "
            "no earlier than the close at t (MOC) — forward returns start at t's close.",
        ],
    }
    return frame, info


def _load_csv(cfg: DataConfig) -> tuple[pd.DataFrame, dict]:
    if not cfg.csv_path:
        raise ValueError("data.csv_path is required for source=csv")
    df = pd.read_csv(cfg.csv_path, parse_dates=["date"])
    tgt = df[df["ticker"] == cfg.ticker].set_index("date").sort_index()
    frame = pd.DataFrame(
        {
            "close": tgt.get("adj_close", tgt["close"]),
            "raw_close": tgt["close"],
            "volume": tgt["volume"].astype(float),
        }
    )
    for b in cfg.benchmarks:
        bt = df[df["ticker"] == b].set_index("date").sort_index()
        frame[f"bench_{b}"] = bt.get("adj_close", bt["close"]).reindex(frame.index)
    frame = frame.loc[cfg.start:cfg.end]
    frame.index.name = "date"
    info = {
        "vendor": f"local CSV: {cfg.csv_path}",
        "adjustment": "as supplied (adj_close if present)",
        "calendar": "as supplied",
        "timezone": "as supplied",
        "notes": [],
    }
    return frame, info


SOURCES = {"synthetic": _load_synthetic, "yfinance": _load_yfinance, "csv": _load_csv}


# ---------------------------------------------------------------------- public API
def load_snapshot(cfg: DataConfig, refresh: bool = False) -> DatasetSnapshot:
    """Return a cached immutable snapshot for `cfg`, creating it if needed.

    Cache key = request parameters. A cached snapshot is never mutated; `refresh=True`
    writes a *new* snapshot (new hash) alongside the old one.
    """
    if cfg.source not in SOURCES:
        raise ValueError(f"Unknown data source {cfg.source!r}; choose from {list(SOURCES)}")
    cache = Path(cfg.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    key_src = json.dumps(
        {k: v for k, v in asdict(cfg).items() if k != "cache_dir"}, sort_keys=True
    )
    key = hashlib.sha256(key_src.encode()).hexdigest()[:12]
    pointer = cache / f"{cfg.source}_{cfg.ticker}_{key}.latest"

    if pointer.exists() and not refresh:
        return read_snapshot(cache / pointer.read_text().strip())

    frame, info = SOURCES[cfg.source](cfg)
    frame = frame.sort_index()
    missing = {c: int(frame[c].isna().sum()) for c in frame.columns}
    meta = SnapshotMeta(
        ticker=cfg.ticker,
        benchmarks=list(cfg.benchmarks),
        source=cfg.source,
        vendor=info["vendor"],
        retrieved_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        start=str(frame.index.min().date()) if len(frame) else cfg.start,
        end=str(frame.index.max().date()) if len(frame) else cfg.end,
        frequency=cfg.frequency,
        calendar=info["calendar"],
        timezone=info["timezone"],
        adjustment=info["adjustment"],
        n_rows=len(frame),
        missing_observations=missing,
        notes=info["notes"],
        sha256=frame_hash(frame),
    )
    stem = f"{cfg.source}_{cfg.ticker}_{meta.sha256[:12]}"
    frame.to_csv(cache / f"{stem}.csv.gz", float_format="%.10g", date_format="%Y-%m-%d")
    (cache / f"{stem}.meta.json").write_text(json.dumps(asdict(meta), indent=2))
    pointer.write_text(stem)
    return DatasetSnapshot(frame, meta)


def read_snapshot(stem_path: str | Path) -> DatasetSnapshot:
    stem_path = Path(stem_path)
    frame = pd.read_csv(f"{stem_path}.csv.gz", index_col="date", parse_dates=["date"])
    meta = SnapshotMeta(**json.loads(Path(f"{stem_path}.meta.json").read_text()))
    actual = frame_hash(frame)
    if actual != meta.sha256:
        raise RuntimeError(
            f"Snapshot integrity check failed for {stem_path}: stored {meta.sha256[:12]}, "
            f"computed {actual[:12]}"
        )
    return DatasetSnapshot(frame, meta)
