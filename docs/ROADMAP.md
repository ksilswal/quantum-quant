# Roadmap status

Status against the proposal's development phases (section 11) and acceptance criteria
(section 12). ✅ done · 🟡 partial / first version · ⬜ not started.

## Phase 1 — Foundations
- ✅ NVDA data pipeline: yfinance, CSV and synthetic sources; provenance metadata; SHA-256 snapshot hashing; integrity check on read
- ✅ Hypothesis registry: 6 signal families, deterministic interleaved grid, stable IDs, cluster labels
- ✅ Experiment schema: strict YAML manifests, hashing, round-trip
- ✅ Run record schema (section 10.3): dataset hash, hypothesis, workload, algorithm version, params, seed, backend, environment, timestamps, per-stage times, counters, metrics, warnings, failures
- ✅ Reproducibility tests
- 🟡 Point-in-time integrity: timing convention enforced and tested; vendor back-adjustment documented, not corrected
- ⬜ Exchange trading calendar for synthetic data (uses business days)

## Phase 2 — Classical baselines
- ✅ All three workloads with strong baselines (HAC OLS, block bootstrap/permutation; tuned ridge/logistic, SVM, MLP; MC with antithetic/control variates/adaptive ε, Sobol QMC)
- ✅ Numerical validation against synthetic ground truth and analytic values
- ✅ Staged profiling, scaling over N, repetitions, warm-ups
- 🟡 Optimized baselines: NumPy-vectorized; no Numba/JAX/GPU variants yet
- ⬜ Memory tracking enabled by default (supported via `Profiler(track_memory=True)`, not exposed in manifests)

## Phase 3 — Quantum prototypes
- ✅ Variational circuits (`vqc`), fidelity quantum kernels (`quantum_kernel`)
- ✅ Amplitude estimation: MLAE (`qae_mlae`, `hit_rate_qae`) and canonical QPE (`qae_canonical`)
- ✅ Documented statistical-estimation formulation (directional hit rate) with explicit state-prep cost
- ⬜ Iterative QAE (Grinko et al.) and low-depth variants
- ⬜ Shot-based training for VQC (currently analytic backprop; shots at inference only)
- ⬜ `lightning.qubit` / JAX-compiled pipelines benchmarked side by side

## Phase 4 — Controlled comparisons
- ✅ Matched-estimand pairing, accuracy–cost frontier, discretization bias reported separately
- ✅ S(N) with bootstrap CIs; resource accounting (qubits, depth, gates, shots, oracle queries)
- 🟡 Precision study config exists (`configs/mc_precision_study.yaml`); no automated ε-sweep yet
- ⬜ Scaling along T, d, ε, L, q, p axes (only N is automated; others via multiple configs)
- ⬜ Noise models

## Phase 5 — Research interface
- ✅ Streamlit builder with parameter widgets generated from algorithm schemas, validation, manifest download/upload
- ✅ Dashboard tabs: benchmarks, output quality, hypotheses, correlations, provenance; CSV downloads
- ✅ FDR-adjusted hypothesis tables; correlation matrices of signals / predictions / strategy returns; partial correlations; temporal stability
- ⬜ Saved configuration library in the UI; cancel a running experiment; full PDF report

## Phase 6 — Optional extension
- ⬜ Real QPU access (queue time, calibration, noise)
- ⬜ Web deployment
- ⬜ Additional algorithm families (e.g. QAOA portfolio formulation)

## Acceptance criteria (section 12)
| Criterion | Status |
|---|---|
| Classical baselines reproduce independent checks / synthetic ground truth | ✅ `tests/test_ground_truth.py` |
| ≥ 50 NVDA hypotheses end-to-end without code edits | ✅ `tests/test_end_to_end.py`, `configs/example_all_workloads.yaml` |
| Each workload configurable and runnable from the UI | ✅ |
| Export full timing breakdowns and manifests | ✅ `qquant run --export`, `qquant export` |
| Stable chronological splits, no forward leakage | ✅ `tests/test_data_and_leakage.py` |
| Correlation matrices only after independent evaluations | ✅ computed post hoc from stored series |
| ≥ 1 scaling experiment and 1 matched-quality comparison per feasible pairing | 🟡 tooling done; runs on real NVDA data pending |
| Unsupported comparisons labelled; simulator ≠ physical advantage | ✅ `experimental` flag, estimand labels, UI banner |
