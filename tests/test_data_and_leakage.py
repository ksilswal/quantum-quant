"""Snapshot integrity, return conventions, signal causality and chronological splits."""
import gzip

import numpy as np
import pandas as pd
import pytest

from qquant.algorithms.predictive import make_split
from qquant.algorithms.tasks import SignalTask
from qquant.config import DataConfig, HypothesisConfig
from qquant.data import forward_return, load_snapshot, read_snapshot
from qquant.hypotheses import SIGNALS, compute_signal, generate


def test_snapshot_cached_and_hash_stable(tmp_path):
    cfg = DataConfig(source="synthetic", cache_dir=str(tmp_path), start="2020-01-01", end="2021-12-31")
    a = load_snapshot(cfg)
    b = load_snapshot(cfg)
    assert a.hash == b.hash and len(a.hash) == 64
    pd.testing.assert_frame_equal(a.frame, b.frame, check_freq=False)


def test_snapshot_tamper_detected(tmp_path):
    cfg = DataConfig(source="synthetic", cache_dir=str(tmp_path), start="2020-01-01", end="2020-12-31")
    snap = load_snapshot(cfg)
    stem = tmp_path / f"synthetic_NVDA_{snap.hash[:12]}"
    df = pd.read_csv(f"{stem}.csv.gz", index_col="date")
    df.iloc[5, 0] *= 1.01
    with gzip.open(f"{stem}.csv.gz", "wt") as fh:
        df.to_csv(fh, float_format="%.10g")
    with pytest.raises(RuntimeError, match="integrity"):
        read_snapshot(stem)


def test_forward_return_definition():
    p = pd.Series([100.0, 110.0, 99.0, 120.0])
    r = forward_return(p, 2)
    assert r.iloc[0] == pytest.approx(99 / 100 - 1)
    assert r.iloc[1] == pytest.approx(120 / 110 - 1)
    assert r.iloc[2:].isna().all()


@pytest.mark.parametrize("family", sorted(SIGNALS))
def test_signals_are_causal(snapshot, family):
    """Changing prices/volumes after date t must not change any signal value at or before t."""
    frame = snapshot.frame.iloc[:600].copy()
    t = 400
    base = compute_signal(frame, family, 21, benchmark="QQQ")
    shocked = frame.copy()
    shocked.iloc[t + 1:] *= 1.7
    after = compute_signal(shocked, family, 21, benchmark="QQQ")
    pd.testing.assert_series_equal(base.iloc[: t + 1], after.iloc[: t + 1])


def test_dataset_alignment_has_no_future_features(snapshot):
    h = generate(HypothesisConfig(families=["momentum"], lookbacks=[5], horizons=[5]))[0]
    d = h.dataset(snapshot.frame)
    close = snapshot.frame["close"]
    t = d.index[100]
    i = close.index.get_loc(t)
    assert d.loc[t, "y"] == pytest.approx(close.iloc[i + 5] / close.iloc[i] - 1)
    assert d.loc[t, "x"] == pytest.approx(close.iloc[i] / close.iloc[i - 5] - 1)


@pytest.mark.parametrize("h", [1, 5, 21])
def test_chronological_split_purges_overlapping_labels(snapshot, h):
    hyp = generate(HypothesisConfig(families=["momentum"], lookbacks=[10], horizons=[h]))[0]
    task = SignalTask(hyp.id, hyp, hyp.dataset(snapshot.frame), snapshot.frame)
    p = {"n_lags": 2, "train_frac": 0.6, "val_frac": 0.2, "target": "return"}
    s = make_split(task, p)
    idx = task.data.index
    tr_last = pd.Timestamp(s.info["train_period"].split("..")[1])
    va_first = pd.Timestamp(s.info["val_period"].split("..")[0])
    va_last = pd.Timestamp(s.info["val_period"].split("..")[1])
    te_first = s.idx_te[0]
    # label of the last training row ends h trading rows later, strictly before validation starts
    assert idx[idx.get_loc(tr_last) + h] < va_first
    assert idx[idx.get_loc(va_last) + h] < te_first
    assert (s.X_tr.mean(0) == pytest.approx(0, abs=1e-8))  # scaler fit on train only
