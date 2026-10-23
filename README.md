# Quantum Quant

High-throughput quantum–classical benchmarking on reproducible NVDA quantitative-research
workloads.

Alpha-signal evaluation is the **workload**, not the objective. The question is whether,
when, and why quantum algorithms offer computational advantages over strong classical
methods, measured on elapsed time, throughput, scaling, precision and resource use.
Finding a profitable NVDA signal is explicitly not a goal, and no claim of quantum
advantage is drawn from running a PennyLane simulator on a classical computer.

## Quick start

```bash
pip install -e ".[all]"            # core + yfinance + streamlit + pytest
pytest                             # 43 tests: ground truth, leakage, e2e (≈30 s)

qquant algorithms                  # registered algorithms and their parameters
qquant validate configs/example_all_workloads.yaml
qquant run configs/smoke.yaml --export results/exports      # ~15 s
qquant ui                          # experiment builder + results dashboard
```

Every config defaults to `data.source: synthetic`, a regime-switching GBM with a
*planted* 5-day momentum effect, so the pipeline can be validated against a known
answer. For real data set `source: yfinance` (or `csv`).

## Layout

```
src/qquant/
  config.py              experiment manifests (strict YAML schema, hashed)
  data/engine.py         ingestion, provenance, content-hashed immutable snapshots
  data/returns.py        r_{t,t+h} = P_{t+h}/P_t - 1 and timing conventions
  hypotheses/            signal families + deterministic hypothesis registry
  algorithms/
    base.py              algorithm registry, typed parameter schemas, validation
    statistical.py       Workload I  — OLS/HAC, block bootstrap, block permutation, hit rate (+QAE)
    predictive.py        Workload II — baselines, linear, SVM, MLP, RBF kernel ridge, VQC, quantum kernel
    montecarlo.py        Workload III — MC, Sobol QMC, discretized exact, MLAE, canonical QAE
    qae.py               shared maximum-likelihood amplitude estimation + resource counting
  benchmark/profiler.py  staged timing (load/preprocess/compile/execute/postprocess) and counters
  scheduler.py           batches, scaling over N, repetitions, warm-ups, workers, seeds
  storage.py             SQLite run store (raw per-run records + series)
  analysis/              summaries, scaling, S(N), accuracy–cost frontiers, FDR, correlations
  ui/app.py              Streamlit builder and dashboard
configs/                 smoke, full 50-hypothesis example, Monte Carlo precision study
tests/                   ground truth, causality/leakage, manifest validation, end-to-end
docs/ROADMAP.md          phase-by-phase status against the proposal
```

## Workloads and algorithms

| Workload | Estimand | Classical | Quantum (experimental) |
|---|---|---|---|
| Statistical | `ols_slope` | `ols_robust`, `block_bootstrap_ols`, `block_permutation` | — (no claim that OLS is quantum-accelerated) |
| Statistical | `directional_hit_rate` | `hit_rate_classical` | `hit_rate_qae` (MLAE; `amplitude_rotation` or `index_register` state prep) |
| Predictive | `forecast` | `baseline_mean`, `baseline_zero`, `linear`, `svm_rbf`, `mlp`, `kernel_ridge_rbf` | `vqc`, `quantum_kernel` |
| Monte Carlo | `mc_expectation` | `mc_plain` (antithetic / control variate / adaptive ε), `mc_sobol`, `discretized_exact` | `qae_mlae`, `qae_canonical` |

Algorithms sharing an estimand compute the same quantity on the same input, so they are
compared at **matched quality** (accuracy–cost frontier). Predictive comparisons are
different-model comparisons and are labelled as such.

## Design decisions worth knowing

- **Timing convention.** A signal at the close of day *t* uses data ≤ *t*; the label is
  the forward return from that close. Causality is tested by shocking prices after *t*
  and asserting every signal ≤ *t* is unchanged.
- **Purged splits.** Chronological train/val/test; the last *h* rows of train and
  validation are dropped so no training label overlaps a later block. Scalers fit on train.
- **Common random numbers.** Each task's seed depends only on (base seed, task, repetition),
  so every algorithm and every batch size sees identical inputs.
- **Honest quantum accounting.** Data loading, compilation, execution and post-processing
  are timed separately; shots, oracle queries (2m+1 per shot of Q^m A), qubits, depth and
  gate counts are recorded. Gate counting itself is timed under `instrumentation` and
  excluded from end-to-end time.
- **QAE vs discretization.** QAE estimates the expectation over a 2^n-point grid, not the
  continuous model. Reports show error against the analytic value *and* against each
  method's own target, plus the discretization bias, so neither method gets credit or
  blame for the other's error.
- **`hit_rate_qae` with `amplitude_rotation`** computes the hit rate classically before
  encoding it. It measures amplitude-estimation mechanics only, and says so in its run
  warnings. `index_register` loads every observation into a log₂T+1-qubit state, and its
  gate count shows why data loading erases query savings.
- **Canonical QAE** halves the payoff to remove PennyLane's a ↔ 1−a phase ambiguity.
- **Multiple testing.** BH (default) within each algorithm config; the family count is
  reported because related parameterizations are not independent evidence.

## Adding an algorithm

```python
from qquant.algorithms.base import Algorithm, Param, AlgoOutput, register

@register
class MyEstimator(Algorithm):
    name = "my_estimator"
    workload = "montecarlo"          # statistical | predictive | montecarlo
    kind = "classical"               # or "quantum" (then set experimental = True)
    estimand = "mc_expectation"      # same estimand => matched-quality comparison
    params = [Param("n_paths", int, 10_000, min=100)]

    def run(self, task, p, prof, ctx):
        with prof.stage("execute"):
            ...
        prof.set("samples", p["n_paths"])
        return AlgoOutput({"estimate": ..., "ci_halfwidth": ...})
```

It appears in the CLI, the manifest validator and the UI automatically.

## Caveats

Simulated circuits run on CPUs; any S(N) > 1 is a statement about these implementations
on this machine, nothing more. Vendor-adjusted prices are back-adjusted (NVDA split 4:1 in
2021 and 10:1 in 2024); ratio-based signals are unaffected, level-based ones are not.
The GBM used for Monte Carlo is a toy model, calibrated on the trailing year of the snapshot.
