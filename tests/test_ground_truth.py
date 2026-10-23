"""Numerical validation against known ground truth (acceptance criterion 1)."""
import numpy as np
import pandas as pd
import pennylane as qml
import pytest

from qquant import algorithms as algos
from qquant.algorithms.base import RunContext
from qquant.algorithms.montecarlo import analytic_value, discretize
from qquant.algorithms.qae import run_mlae
from qquant.algorithms.tasks import MonteCarloTask, SignalTask
from qquant.benchmark.profiler import Profiler
from qquant.config import HypothesisConfig
from qquant.hypotheses import generate


def run(name, task, seed=0, **params):
    a = algos.get(name)
    ctx = RunContext(seed=seed)
    p = algos.resolve_params(a, params, ctx)
    prof = Profiler()
    prof.start()
    out = a.run(task, p, prof, ctx)
    prof.stop()
    return out.metrics, prof


def synthetic_task(beta, T=3000, seed=0, h=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=T)
    x = rng.standard_normal(T)
    y = 0.001 + beta * x + 0.02 * rng.standard_normal(T)
    hyp = generate(HypothesisConfig(families=["momentum"], lookbacks=[5], horizons=[h]))[0]
    frame = pd.DataFrame({"close": np.ones(T), "volume": np.ones(T), "bench_QQQ": np.ones(T)}, index=idx)
    return SignalTask("synthetic", hyp, pd.DataFrame({"x": x, "y": y}, index=idx), frame)


@pytest.mark.parametrize("name", ["ols_robust", "block_bootstrap_ols"])
def test_slope_recovered(name):
    m, _ = run(name, synthetic_task(beta=0.004))
    assert m["ci_low"] < 0.004 < m["ci_high"]
    assert m["p_value"] < 0.01


@pytest.mark.parametrize("name", ["ols_robust", "block_permutation", "block_bootstrap_ols"])
def test_null_not_rejected_too_often(name):
    rejections = sum(run(name, synthetic_task(beta=0.0, T=800, seed=s), seed=s)[0]["p_value"] < 0.05 for s in range(20))
    assert rejections <= 4  # nominal 1/20; generous bound for 20 draws


def test_planted_effect_found_in_synthetic_snapshot(snapshot):
    hyp = generate(HypothesisConfig(families=["momentum"], lookbacks=[5], horizons=[1]))[0]
    task = SignalTask(hyp.id, hyp, hyp.dataset(snapshot.frame), snapshot.frame)
    m, _ = run("ols_robust", task)
    assert m["estimate"] > 0 and m["p_value"] < 0.01


@pytest.mark.parametrize("a", [0.05, 0.3, 0.5, 0.8, 0.97])
def test_mlae_recovers_known_amplitude(a):
    theta = 2 * np.arcsin(np.sqrt(a))
    prof = Profiler()
    res = run_mlae(lambda w: qml.RY(theta, wires=w[0]), 1, 0, [0, 1, 2, 4, 8, 16], shots=200,
                   alpha=0.01, rng=np.random.default_rng(1), prof=prof)
    assert abs(res.estimate - a) < 0.01
    assert res.oracle_queries == 200 * sum(2 * m + 1 for m in [0, 1, 2, 4, 8, 16])


def _mc_task(payoff, moneyness=1.0):
    return MonteCarloTask("t", S0=100.0, mu=0.1, sigma=0.4, horizon_days=63, strike=100 * moneyness,
                          payoff=payoff, cap=25.0)


@pytest.mark.parametrize("payoff", ["digital", "capped_call"])
def test_classical_mc_matches_analytic(payoff):
    task = _mc_task(payoff)
    m, _ = run("mc_plain", task, n_paths=2_000_000)
    assert abs(m["estimate"] - analytic_value(task)) < 4 * m["std_error"] + 1e-12
    m, _ = run("mc_sobol", task)
    assert m["abs_error"] < 1e-3


def test_discretization_converges():
    task = _mc_task("capped_call")
    errs = [abs(float(np.dot(*discretize(task, n, 5.0))) - analytic_value(task)) for n in (3, 6, 10)]
    assert errs[0] > errs[1] > errs[2] and errs[2] < 1e-3


@pytest.mark.parametrize("payoff,mny", [("digital", 0.8), ("digital", 1.2), ("capped_call", 1.0)])
def test_qae_matches_discretized_target(payoff, mny):
    task = _mc_task(payoff, mny)
    m, prof = run("qae_mlae", task, n_qubits=4, shots=200, n_powers=6)
    assert m["abs_error_vs_discretized"] < 0.02
    assert prof.counters["qubits"] == 5
    m, _ = run("qae_canonical", task, n_qubits=4, estimation_qubits=7, shots=200)
    assert m["abs_error_vs_discretized"] <= m["ci_halfwidth"]


def test_hit_rate_qae_matches_classical(snapshot):
    hyp = generate(HypothesisConfig(families=["momentum"], lookbacks=[5], horizons=[1]))[0]
    task = SignalTask(hyp.id, hyp, hyp.dataset(snapshot.frame).iloc[:500], snapshot.frame)
    ref, _ = run("hit_rate_classical", task)
    q, _ = run("hit_rate_qae", task, shots=300, n_powers=6)
    assert abs(q["estimate"] - ref["estimate"]) < 0.02
    qi, prof = run("hit_rate_qae", task, state_prep="index_register", shots=200, n_powers=3)
    assert abs(qi["estimate"] - ref["estimate"]) < 0.03
    assert prof.counters["qubits"] == 10  # ceil(log2 500) + 1
