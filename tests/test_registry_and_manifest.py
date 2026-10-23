import pytest

from qquant import algorithms as algos
from qquant.algorithms.base import RunContext
from qquant.config import ExperimentManifest, HypothesisConfig, ManifestError
from qquant.hypotheses import generate
from qquant.scheduler import plan


def test_default_grid_is_50_unique_hypotheses():
    hyps = generate(HypothesisConfig())
    assert len(hyps) == 50
    assert len({h.id for h in hyps}) == 50


def test_prefix_is_diverse_across_families():
    hyps = generate(HypothesisConfig(n_hypotheses=10))
    assert len({h.family for h in hyps}) == 5


def test_requesting_too_many_hypotheses_fails():
    with pytest.raises(ValueError):
        generate(HypothesisConfig(n_hypotheses=500))


def test_unknown_manifest_key_rejected():
    with pytest.raises(ManifestError, match="unknown key"):
        ExperimentManifest.from_dict({"run": {"sead": 1}})


def test_unknown_algorithm_param_rejected():
    m = ExperimentManifest.from_dict({"statistical": {"enabled": True, "algorithms": [{"name": "ols_robust", "params": {"lags": 3}}]}})
    with pytest.raises(ManifestError, match="unknown parameter"):
        plan(m)


def test_algorithm_under_wrong_workload_rejected():
    m = ExperimentManifest.from_dict({"montecarlo": {"enabled": True, "algorithms": ["ols_robust"]}})
    with pytest.raises(ManifestError, match="statistical algorithm"):
        plan(m)


def test_qubit_cap_enforced():
    m = ExperimentManifest.from_dict({
        "montecarlo": {"enabled": True, "algorithms": [{"name": "qae_canonical", "params": {"n_qubits": 8, "estimation_qubits": 8}}]},
        "run": {"max_qubits": 12},
    })
    with pytest.raises(ManifestError, match="qubits"):
        plan(m)


def test_incompatible_target_rejected():
    with pytest.raises(ManifestError, match="supports target"):
        algos.resolve_params(algos.get("baseline_zero"), {"target": "direction"}, RunContext(seed=0))


def test_every_algorithm_has_valid_defaults():
    for name in algos.REGISTRY:
        algos.resolve_params(algos.get(name), {}, RunContext(seed=0))


def test_quantum_algorithms_marked_experimental():
    for cls in algos.REGISTRY.values():
        if cls.kind == "quantum":
            assert cls.experimental, cls.name


def test_manifest_roundtrip(tmp_path):
    m = ExperimentManifest.from_yaml("configs/example_all_workloads.yaml")
    path = tmp_path / "m.yaml"
    m.to_yaml(path)
    assert ExperimentManifest.from_yaml(path).hash() == m.hash()
    assert len(plan(m)) == 15
