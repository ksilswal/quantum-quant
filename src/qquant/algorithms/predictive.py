"""Workload II — predictive model training on one hypothesis (signal) at a time.

Design: features are lags x_t, x_{t-1}, ..., x_{t-n_lags+1} of the hypothesis' signal,
fixed before testing. Target is the h-day forward NVDA return (regression) or its sign
(classification). Splits are chronological train / validation / test with the last h
rows of train and validation *purged*, so no training label overlaps a later period.
Scalers are fit on train only. The test block is touched once, after selection on
validation.

All predictive comparisons are *different-model* comparisons (different hypothesis
classes); report fit time and inference latency jointly with quality.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pennylane as qml
from pennylane import numpy as pnp
from scipy.stats import norm
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR

from ..config import ManifestError
from .base import AlgoOutput, Algorithm, Param, RunContext, register
from .qae import circuit_resources
from .tasks import SignalTask

COMMON = [
    Param("target", str, "return", "regress the return or classify its sign", choices=("return", "direction")),
    Param("n_lags", int, 2, "lags of the signal used as features", min=1, max=16),
    Param("train_frac", float, 0.6, min=0.2, max=0.9),
    Param("val_frac", float, 0.2, min=0.05, max=0.5),
    Param("cost_bps", float, 5.0, "hypothetical one-way cost for descriptive strategy metrics", min=0),
]


@dataclass
class Split:
    X_tr: np.ndarray
    y_tr: np.ndarray
    X_va: np.ndarray
    y_va: np.ndarray
    X_te: np.ndarray
    y_te: np.ndarray
    r_te: np.ndarray  # raw forward returns on the test block
    idx_te: pd.Index
    info: dict


def make_split(task: SignalTask, p: dict) -> Split:
    df = task.data
    feats = pd.concat({f"lag{i}": df["x"].shift(i) for i in range(p["n_lags"])}, axis=1)
    d = feats.join(df["y"]).dropna()
    n, h = len(d), task.horizon
    tr_end = int(n * p["train_frac"])
    va_end = tr_end + int(n * p["val_frac"])
    if va_end >= n - 10:
        raise ManifestError("train_frac + val_frac leaves < 10 test observations")
    tr = slice(0, tr_end - h)  # purge: last h training labels overlap validation
    va = slice(tr_end, va_end - h)
    te = slice(va_end, n)
    X = d.drop(columns="y").to_numpy()
    r = d["y"].to_numpy()
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
    Xs = (X - mu) / sd
    y = (r > 0).astype(int) if p["target"] == "direction" else r
    idx = d.index
    info = {
        "train_period": f"{idx[tr][0].date()}..{idx[tr][-1].date()}",
        "val_period": f"{idx[va][0].date()}..{idx[va][-1].date()}",
        "test_period": f"{idx[te][0].date()}..{idx[te][-1].date()}",
        "n_train": tr_end - h, "n_val": va_end - h - tr_end, "n_test": n - va_end, "purge": h,
    }
    return Split(Xs[tr], y[tr], Xs[va], y[va], Xs[te], y[te], r[te], idx[te], info)


def evaluate(target: str, s: Split, pred: np.ndarray, prob: np.ndarray | None, cost_bps: float) -> tuple[dict, dict]:
    m: dict = {}
    if target == "return":
        err = pred - s.y_te
        m["rmse"] = float(np.sqrt(np.mean(err**2)))
        m["mae"] = float(np.mean(np.abs(err)))
        bench = np.mean((s.y_tr.mean() - s.y_te) ** 2)
        m["r2_oos"] = float(1 - np.mean(err**2) / bench)
        pos = np.sign(pred)
    else:
        prob = np.clip(prob, 1e-6, 1 - 1e-6)
        m["accuracy"] = float(np.mean((prob > 0.5) == s.y_te))
        m["brier"] = float(np.mean((prob - s.y_te) ** 2))
        m["log_loss"] = float(log_loss(s.y_te, prob, labels=[0, 1]))
        m["auc"] = float(roc_auc_score(s.y_te, prob)) if len(np.unique(s.y_te)) == 2 else None
        pos = np.where(prob > 0.5, 1.0, -1.0)
    # undefined when the model never takes a side (e.g. the zero forecast)
    m["directional_accuracy"] = float(np.mean(np.sign(pos) == np.sign(s.r_te))) if np.any(pos != 0) else None
    turnover = np.abs(np.diff(np.concatenate([[0.0], pos])))
    strat = pos * s.r_te - turnover * cost_bps / 1e4
    m["strategy_mean_return"] = float(strat.mean())
    m["strategy_hit_rate"] = float(np.mean(strat > 0))
    series = {
        "prediction": pd.Series(pred if target == "return" else prob, index=s.idx_te),
        "strategy": pd.Series(strat, index=s.idx_te),
    }
    return m, series


class Predictive(Algorithm):
    workload = "predictive"
    extra_params: list[Param] = []
    supports = ("return", "direction")

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        cls.params = COMMON + cls.extra_params

    def check(self, p, ctx):
        if p["target"] not in self.supports:
            raise ManifestError(f"{self.name} supports target in {self.supports}, got {p['target']!r}")

    # subclasses return (predictions on test, probabilities on test or None)
    def fit_predict(self, s: Split, p: dict, prof, ctx: RunContext):
        raise NotImplementedError

    def run(self, task: SignalTask, p, prof, ctx):
        with prof.stage("preprocess"):
            s = make_split(task, p)
        pred, prob = self.fit_predict(s, p, prof, ctx)
        with prof.stage("postprocess"):
            m, series = evaluate(p["target"], s, pred, prob, p["cost_bps"])
            m.update(s.info)
        prof.set("samples", s.info["n_train"])
        prof.set("features", s.X_tr.shape[1])
        return AlgoOutput(m, series)


def _timed_fit(prof, fit, predict):
    with prof.stage("execute"):
        fit()
    with prof.stage("inference"):
        return predict()


# ------------------------------------------------------------------------- classical
@register
class BaselineMean(Predictive):
    name = "baseline_mean"
    kind = "classical"
    estimand = "forecast"
    description = "Historical mean return (regression) or training up-rate (classification)."

    def fit_predict(self, s, p, prof, ctx):
        with prof.stage("execute"):
            c = float(s.y_tr.mean())
        with prof.stage("inference"):
            out = np.full(len(s.y_te), c)
        return (out, None) if p["target"] == "return" else (out, out)


@register
class BaselineZero(Predictive):
    name = "baseline_zero"
    kind = "classical"
    estimand = "forecast"
    supports = ("return",)
    description = "Zero-return forecast."

    def fit_predict(self, s, p, prof, ctx):
        with prof.stage("inference"):
            return np.zeros(len(s.y_te)), None


def _select(models: list, s: Split, target: str, prof):
    """Fit candidates on train, select on validation; returns best fitted model and its index."""
    best, best_score, best_i = None, np.inf, 0
    for i, mdl in enumerate(models):
        mdl.fit(s.X_tr, s.y_tr)
        if target == "return":
            score = np.mean((mdl.predict(s.X_va) - s.y_va) ** 2)
        else:
            score = log_loss(s.y_va, np.clip(_proba(mdl, s.X_va), 1e-6, 1 - 1e-6), labels=[0, 1])
        if score < best_score:
            best, best_score, best_i = mdl, score, i
    prof.set("tuning_candidates", len(models))
    return best, best_i


def _proba(mdl, X):
    if hasattr(mdl, "predict_proba"):
        return mdl.predict_proba(X)[:, 1]
    return np.clip((mdl.predict(X) + 1) / 2, 0, 1)  # +-1 regression -> probability


@register
class LinearModel(Predictive):
    name = "linear"
    kind = "classical"
    estimand = "forecast"
    description = "Ridge regression (return) or L2 logistic regression (direction); penalty tuned on validation."
    extra_params = [Param("alphas", list, [0.01, 0.1, 1.0, 10.0, 100.0], "penalty grid searched on validation")]

    def fit_predict(self, s, p, prof, ctx):
        if p["target"] == "return":
            cands = [Ridge(alpha=a) for a in p["alphas"]]
        else:
            cands = [LogisticRegression(C=1 / a, max_iter=1000) for a in p["alphas"]]
        with prof.stage("execute"):
            mdl, i = _select(cands, s, p["target"], prof)
        prof.set("selected_alpha", p["alphas"][i])
        with prof.stage("inference"):
            if p["target"] == "return":
                return mdl.predict(s.X_te), None
            pr = mdl.predict_proba(s.X_te)[:, 1]
            return pr, pr


@register
class SVMModel(Predictive):
    name = "svm_rbf"
    kind = "classical"
    estimand = "forecast"
    description = "RBF support vector regression / classification."
    extra_params = [
        Param("C", float, 1.0, min=1e-4),
        Param("epsilon", float, 0.001, "SVR epsilon (in return units)", min=0),
        Param("max_train_samples", int, 2000, "most-recent training rows kept", min=50),
    ]

    def fit_predict(self, s, p, prof, ctx):
        n = p["max_train_samples"]
        Xtr, ytr = s.X_tr[-n:], s.y_tr[-n:]
        if p["target"] == "return":
            mdl = SVR(C=p["C"], epsilon=p["epsilon"], gamma="scale")
            pr = _timed_fit(prof, lambda: mdl.fit(Xtr, ytr), lambda: mdl.predict(s.X_te))
            return pr, None
        mdl = SVC(C=p["C"], gamma="scale", probability=True, random_state=ctx.seed)
        pr = _timed_fit(prof, lambda: mdl.fit(Xtr, ytr), lambda: mdl.predict_proba(s.X_te)[:, 1])
        return pr, pr


@register
class MLPModel(Predictive):
    name = "mlp"
    kind = "classical"
    estimand = "forecast"
    description = "Compact feed-forward network with early stopping on an internal validation slice."
    extra_params = [
        Param("hidden", int, 16, min=1, max=1024),
        Param("layers", int, 1, min=1, max=4),
        Param("learning_rate", float, 1e-3, min=1e-6, max=1.0),
        Param("l2", float, 1e-3, min=0),
        Param("epochs", int, 200, min=1, max=10_000),
    ]

    def fit_predict(self, s, p, prof, ctx):
        kw = dict(
            hidden_layer_sizes=(p["hidden"],) * p["layers"], learning_rate_init=p["learning_rate"],
            alpha=p["l2"], max_iter=p["epochs"], early_stopping=True, random_state=ctx.seed,
        )
        if p["target"] == "return":
            sd = s.y_tr.std() + 1e-12
            mdl = MLPRegressor(**kw)
            pr = _timed_fit(prof, lambda: mdl.fit(s.X_tr, s.y_tr / sd), lambda: mdl.predict(s.X_te) * sd)
            return pr, None
        mdl = MLPClassifier(**kw)
        pr = _timed_fit(prof, lambda: mdl.fit(s.X_tr, s.y_tr), lambda: mdl.predict_proba(s.X_te)[:, 1])
        return pr, pr


@register
class KernelRidgeRBF(Predictive):
    name = "kernel_ridge_rbf"
    kind = "classical"
    estimand = "forecast"
    description = "Classical RBF kernel ridge; the like-for-like baseline for quantum_kernel (same sample cap)."
    extra_params = [
        Param("alpha", float, 1.0, min=1e-8),
        Param("gamma", float, -1.0, "-1 = 1/n_features", min=-1),
        Param("max_train_samples", int, 400, "most-recent training rows kept", min=20, max=20_000),
    ]

    def fit_predict(self, s, p, prof, ctx):
        n = p["max_train_samples"]
        Xtr, ytr = s.X_tr[-n:], s.y_tr[-n:]
        g = p["gamma"] if p["gamma"] > 0 else 1.0 / Xtr.shape[1]
        mdl = KernelRidge(alpha=p["alpha"], kernel="rbf", gamma=g)
        if p["target"] == "return":
            sd = ytr.std() + 1e-12
            pr = _timed_fit(prof, lambda: mdl.fit(Xtr, ytr / sd), lambda: mdl.predict(s.X_te) * sd)
            return pr, None
        pr = _timed_fit(prof, lambda: mdl.fit(Xtr, 2 * ytr - 1.0), lambda: np.clip((mdl.predict(s.X_te) + 1) / 2, 0, 1))
        return pr, pr


# ------------------------------------------------------------------------- quantum
def _angles(X: np.ndarray, bandwidth: float) -> np.ndarray:
    """Standardized features -> rotation angles in (0, pi) via the normal CDF, then scaled."""
    return bandwidth * np.pi * norm.cdf(X)


@register
class VariationalCircuit(Predictive):
    name = "vqc"
    kind = "quantum"
    estimand = "forecast"
    experimental = True
    description = (
        "Variational circuit: angle embedding (1 qubit per feature) + StronglyEntanglingLayers, "
        "readout <Z0> with trainable affine head. Trained analytically with backprop on the "
        "simulator; `shots` applies to inference only."
    )
    extra_params = [
        Param("layers", int, 2, min=1, max=20),
        Param("learning_rate", float, 0.05, min=1e-5, max=1.0),
        Param("epochs", int, 30, min=1, max=2000),
        Param("batch_size", int, 64, min=1),
        Param("max_train_samples", int, 1000, "most-recent training rows kept", min=20),
        Param("bandwidth", float, 1.0, "angle scale", min=0.01, max=4.0),
        Param("shots", int, 0, "0 = analytic expectation at inference", min=0),
    ]

    def check(self, p, ctx):
        super().check(p, ctx)
        if p["n_lags"] > ctx.max_qubits:
            raise ManifestError(f"vqc needs n_lags={p['n_lags']} qubits > max_qubits={ctx.max_qubits}")

    def fit_predict(self, s, p, prof, ctx):
        rng = np.random.default_rng(ctx.seed)
        n_q = s.X_tr.shape[1]
        n = p["max_train_samples"]
        Xtr, ytr = _angles(s.X_tr[-n:], p["bandwidth"]), s.y_tr[-n:]
        Xva, Xte = _angles(s.X_va, p["bandwidth"]), _angles(s.X_te, p["bandwidth"])
        reg = p["target"] == "return"
        y_mu, y_sd = (ytr.mean(), ytr.std() + 1e-12) if reg else (0.0, 1.0)
        ytr_t, yva_t = (ytr - y_mu) / y_sd, (s.y_va - y_mu) / y_sd

        with prof.stage("compile"):
            wires = list(range(n_q))
            dev = qml.device(ctx.backend, wires=n_q)

            @qml.qnode(dev, diff_method="backprop")
            def circuit(x, w):
                qml.AngleEmbedding(x, wires=wires, rotation="Y")
                qml.StronglyEntanglingLayers(w, wires=wires)
                return qml.expval(qml.Z(0))

            shape = qml.StronglyEntanglingLayers.shape(n_layers=p["layers"], n_wires=n_q)
            w = pnp.array(rng.uniform(0, 2 * np.pi, shape), requires_grad=True)
            head = pnp.array([1.0, 0.0], requires_grad=True)
            n_params = int(np.prod(shape)) + 2
        with prof.stage("instrumentation"):
            for k, v in circuit_resources(circuit, Xtr[:1], w).items():
                prof.maximum(f"max_{k}", v)
        prof.set("qubits", n_q)
        prof.set("trainable_params", n_params)

        def model(x, w, head):
            return head[0] * circuit(x, w) + head[1]

        def loss(w, head, x, y):
            out = model(x, w, head)
            if reg:
                return pnp.mean((out - y) ** 2)
            pr = 1 / (1 + pnp.exp(-out))
            pr = pnp.clip(pr, 1e-6, 1 - 1e-6)
            return -pnp.mean(y * pnp.log(pr) + (1 - y) * pnp.log(1 - pr))

        opt = qml.AdamOptimizer(stepsize=p["learning_rate"])
        best = (np.inf, w, head)
        with prof.stage("execute"):
            for epoch in range(p["epochs"]):
                order = rng.permutation(len(Xtr))
                for b0 in range(0, len(Xtr), p["batch_size"]):
                    bi = order[b0:b0 + p["batch_size"]]
                    w, head = opt.step(lambda w_, h_: loss(w_, h_, Xtr[bi], ytr_t[bi]), w, head)
                    prof.count("circuit_executions", len(bi))  # forward evals (backprop)
                    prof.count("param_shift_equivalent_evals", len(bi) * (2 * (n_params - 2) + 1))
                vl = float(loss(w, head, Xva, yva_t))
                prof.count("circuit_executions", len(Xva))
                if vl < best[0]:
                    best = (vl, w.copy(), head.copy())
                    prof.set("best_epoch", epoch + 1)
        _, w, head = best

        with prof.stage("inference"):
            if p["shots"] > 0:
                sdev = qml.device(ctx.backend, wires=n_q, shots=p["shots"])
                sc = qml.QNode(circuit.func, sdev)
                raw = np.asarray(sc(Xte, w))
                prof.count("shots", p["shots"] * len(Xte))
            else:
                raw = np.asarray(circuit(Xte, w))
            prof.count("circuit_executions", len(Xte))
            out = float(head[0]) * raw + float(head[1])
        if reg:
            return out * y_sd + y_mu, None
        pr = 1 / (1 + np.exp(-out))
        return pr, pr


@register
class QuantumKernel(Predictive):
    name = "quantum_kernel"
    kind = "quantum"
    estimand = "forecast"
    experimental = True
    description = (
        "Fidelity quantum kernel k(x,x') = |<phi(x')|phi(x)>|^2 with an IQP-style feature map, "
        "used in kernel ridge. Simulated via statevectors (n_train + n_test circuit evaluations); "
        "hardware would need one overlap circuit per kernel entry (counted separately)."
    )
    extra_params = [
        Param("alpha", float, 1.0, min=1e-8),
        Param("reps", int, 2, "feature-map repetitions", min=1, max=10),
        Param("bandwidth", float, 0.5, "angle scale (kernel bandwidth)", min=0.01, max=4.0),
        Param("max_train_samples", int, 400, "most-recent training rows kept", min=20, max=5000),
    ]

    def check(self, p, ctx):
        super().check(p, ctx)
        if p["n_lags"] > ctx.max_qubits:
            raise ManifestError(f"quantum_kernel needs {p['n_lags']} qubits > max_qubits={ctx.max_qubits}")

    def fit_predict(self, s, p, prof, ctx):
        n = p["max_train_samples"]
        Xtr, ytr = _angles(s.X_tr[-n:], p["bandwidth"]), s.y_tr[-n:]
        Xte = _angles(s.X_te, p["bandwidth"])
        n_q = Xtr.shape[1]
        with prof.stage("compile"):
            wires = list(range(n_q))
            dev = qml.device(ctx.backend, wires=n_q)

            @qml.qnode(dev)
            def feature_state(x):
                for _ in range(p["reps"]):
                    for i in wires:
                        qml.Hadamard(i)
                        qml.RZ(x[..., i], wires=i)
                    for i in range(n_q - 1):
                        qml.IsingZZ(x[..., i] * x[..., i + 1], wires=[i, i + 1])
                return qml.state()
        with prof.stage("instrumentation"):
            for k, v in circuit_resources(feature_state, Xtr[0]).items():
                prof.maximum(f"max_{k}", v)
        prof.set("qubits", n_q)

        reg = p["target"] == "return"
        y_mu, y_sd = (ytr.mean(), ytr.std() + 1e-12) if reg else (0.0, 1.0)
        target = (ytr - y_mu) / y_sd if reg else 2 * ytr - 1.0
        mdl = KernelRidge(alpha=p["alpha"], kernel="precomputed")
        with prof.stage("execute"):
            S_tr = np.asarray(feature_state(Xtr))
            K_tr = np.abs(S_tr.conj() @ S_tr.T) ** 2
            mdl.fit(K_tr, target)
        with prof.stage("inference"):
            S_te = np.asarray(feature_state(Xte))
            K_te = np.abs(S_te.conj() @ S_tr.T) ** 2
            out = mdl.predict(K_te)
        m, t = len(Xtr), len(Xte)
        prof.count("circuit_executions", m + t)
        prof.count("hardware_overlap_circuits", m * (m - 1) // 2 + m * t)
        if reg:
            return out * y_sd + y_mu, None
        pr = np.clip((out + 1) / 2, 0, 1)
        return pr, pr
