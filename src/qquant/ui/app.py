"""Streamlit interface: experiment builder + results dashboard.

Run with `qquant ui` or `streamlit run src/qquant/ui/app.py`.
"""
from __future__ import annotations

import json
from datetime import date

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import yaml

from qquant import algorithms as algos
from qquant.algorithms.base import REGISTRY
from qquant.analysis import benchmarks as bm
from qquant.analysis import signals as sg
from qquant.analysis.report import experiment_tables
from qquant.config import WORKLOADS, ExperimentManifest, ManifestError
from qquant.hypotheses import SIGNALS, generate
from qquant.scheduler import plan, run_experiment
from qquant.storage import RunStore

st.set_page_config(page_title="Quantum Quant", layout="wide")

SIM_BANNER = (
    "Quantum timings below come from **classical simulation** of circuits "
    "(PennyLane `{backend}`). They benchmark software pipelines and resource counts; "
    "they are not evidence of physical quantum speedup."
)
WORKLOAD_LABEL = {"statistical": "Statistical testing", "predictive": "Predictive training", "montecarlo": "Monte Carlo"}


# --------------------------------------------------------------------------- helpers
def param_widget(algo_name: str, idx: int, p) -> object:
    key = f"{algo_name}-{idx}-{p.name}"
    label = f"{p.name}" + (f" — {p.help}" if p.help else "")
    if p.choices:
        return st.selectbox(label, p.choices, index=p.choices.index(p.default), key=key)
    if p.type is bool:
        return st.checkbox(label, value=p.default, key=key)
    if p.type is int:
        return int(st.number_input(label, value=int(p.default), step=1, key=key,
                                   min_value=None if p.min is None else int(p.min),
                                   max_value=None if p.max is None else int(p.max)))
    if p.type is float:
        return float(st.number_input(label, value=float(p.default), key=key, format="%g",
                                     min_value=None if p.min is None else float(p.min),
                                     max_value=None if p.max is None else float(p.max)))
    if p.type in (list, tuple):
        txt = st.text_input(label, value=", ".join(str(v) for v in p.default), key=key)
        return [float(v) for v in txt.split(",") if v.strip()]
    return st.text_input(label, value=str(p.default), key=key)


def algo_label(name: str) -> str:
    cls = REGISTRY[name]
    tag = " · quantum" if cls.kind == "quantum" else ""
    return name + tag + (" · experimental" if cls.experimental else "")


def download(df: pd.DataFrame, name: str, index: bool = False):
    if df is not None and not df.empty:
        st.download_button(f"Download {name}.csv", df.to_csv(index=index).encode(), f"{name}.csv", "text/csv",
                           key=f"dl-{name}")


# --------------------------------------------------------------------------- builder
def builder(db_path: str):
    st.header("Experiment builder")
    up = st.file_uploader("Load a manifest (YAML) — or build one below", type=["yaml", "yml"])
    base = ExperimentManifest.from_dict(yaml.safe_load(up.read())) if up else ExperimentManifest()

    c1, c2 = st.columns([2, 3])
    name = c1.text_input("Experiment name", base.name if up else "nvda-benchmark")
    desc = c2.text_input("Description", base.description)

    with st.expander("Data — NVDA snapshot", expanded=True):
        d = base.data
        c = st.columns(4)
        source = c[0].selectbox("Source", ["synthetic", "yfinance", "csv"], index=["synthetic", "yfinance", "csv"].index(d.source))
        c[1].text_input("Target asset", "NVDA", disabled=True, help="All forecasts, positions and P&L concern NVDA only.")
        start = c[2].date_input("Start", date.fromisoformat(d.start))
        end = c[3].date_input("End", date.fromisoformat(d.end))
        benches = st.multiselect("Benchmark / explanatory series (never traded)", ["QQQ", "SPY", "SOXX", "SMH"], default=d.benchmarks)
        csv_path = st.text_input("CSV path (source=csv)", d.csv_path or "") if source == "csv" else None
        if source == "synthetic":
            st.caption("Synthetic regime-switching GBM with a planted 5-day momentum effect — for validation, not research conclusions.")

    with st.expander("Hypotheses", expanded=True):
        h = base.hypotheses
        fams = st.multiselect("Signal families", sorted(SIGNALS), default=h.families)
        c = st.columns(3)
        lbs = c[0].multiselect("Lookbacks k (days)", [2, 5, 10, 21, 42, 63, 126, 252], default=h.lookbacks)
        hzs = c[1].multiselect("Forward-return horizons h (days)", [1, 2, 5, 10, 21], default=h.horizons)
        grid_n = len(fams) * len(lbs) * len(hzs)
        nh = c[2].number_input(f"Use first N hypotheses (grid = {grid_n}; 0 = all)", 0, max(grid_n, 1), h.n_hypotheses or 0)
        st.caption("Related parameterizations are correlated: N jobs are not N independent pieces of evidence.")

    workload_specs = {}
    st.subheader("Workloads and algorithms")
    tabs = st.tabs([WORKLOAD_LABEL[w] for w in WORKLOADS])
    for w, tab in zip(WORKLOADS, tabs):
        with tab:
            spec = base.workload(w)
            enabled = st.checkbox(f"Enable {WORKLOAD_LABEL[w].lower()}", value=spec.enabled or not up, key=f"en-{w}")
            names = sorted(n for n, c in REGISTRY.items() if c.workload == w)
            chosen = st.multiselect("Algorithms", names, default=[a.name for a in spec.algorithms if a.name in names] or names[:2],
                                    format_func=algo_label, key=f"alg-{w}", disabled=not enabled)
            given = {a.name: a.params for a in spec.algorithms}
            entries = []
            for i, n in enumerate(chosen):
                cls = REGISTRY[n]
                with st.expander(f"{n} parameters", expanded=False):
                    st.caption(cls.description)
                    if cls.experimental:
                        st.warning("Experimental quantum method — results are labelled as such in every report.")
                    params = {}
                    for p in cls.params:
                        if n in given and p.name in given[n]:
                            p = type(p)(**{**p.__dict__, "default": given[n][p.name]})
                        params[p.name] = param_widget(n, i, p)
                entries.append({"name": n, "params": params})
            workload_specs[w] = {"enabled": enabled, "algorithms": entries}
            if w == "montecarlo" and enabled:
                mc = base.montecarlo_tasks
                c = st.columns(4)
                mc_n = c[0].number_input("Tasks", 1, 100, mc.n_tasks)
                payoff = c[1].selectbox("Payoff", ["digital", "capped_call"], index=["digital", "capped_call"].index(mc.payoff))
                mny = c[2].text_input("Moneyness K/S0", ", ".join(map(str, mc.moneyness)))
                hz = c[3].text_input("Horizons (days)", ", ".join(map(str, mc.horizons_days)))
                st.caption("Toy GBM calibrated on the snapshot's trailing year; assumptions are recorded with every run.")
                workload_specs["_mc"] = {
                    "n_tasks": int(mc_n), "payoff": payoff, "cap_fraction": mc.cap_fraction,
                    "moneyness": [float(x) for x in mny.split(",") if x.strip()],
                    "horizons_days": [int(x) for x in hz.split(",") if x.strip()],
                }

    with st.expander("Run, repetition and resource settings", expanded=False):
        r = base.run
        c = st.columns(4)
        seed = c[0].number_input("Seed", value=r.seed, step=1)
        reps = c[1].number_input("Repetitions", 1, 100, r.repetitions)
        warm = c[2].number_input("Warm-up runs", 0, 10, r.warmups)
        workers = c[3].number_input("Parallel workers", 1, 64, r.workers)
        c = st.columns(4)
        scaling = c[0].multiselect("Scaling N", [5, 10, 25, 50, 100, 250, 500], default=r.scaling_N)
        backend = c[1].selectbox("Quantum backend", ["default.qubit", "lightning.qubit"], index=0)
        maxq = c[2].number_input("Max qubits (resource cap)", 1, 30, r.max_qubits)
        timeout = c[3].number_input("Timeout per task (s)", 1.0, 86400.0, float(r.timeout_s))

    manifest_dict = {
        "name": name, "description": desc,
        "data": {"source": source, "ticker": "NVDA", "benchmarks": benches, "start": str(start), "end": str(end),
                 "csv_path": csv_path or None, "cache_dir": base.data.cache_dir},
        "hypotheses": {"families": fams, "lookbacks": lbs, "horizons": hzs, "n_hypotheses": int(nh) or None},
        **{w: workload_specs[w] for w in WORKLOADS},
        "run": {"seed": int(seed), "repetitions": int(reps), "warmups": int(warm), "workers": int(workers),
                "scaling_N": scaling, "backend": backend, "max_qubits": int(maxq), "timeout_s": float(timeout),
                "db_path": db_path},
    }
    if "_mc" in workload_specs:
        manifest_dict["montecarlo_tasks"] = workload_specs["_mc"]

    st.subheader("Validate and run")
    try:
        m = ExperimentManifest.from_dict(manifest_dict)
        planned = plan(m)
        n_h = len(generate(m.hypotheses)) if {"statistical", "predictive"} & set(m.enabled_workloads()) else 0
        st.success(f"Valid · {len(planned)} algorithm configs · {n_h} hypotheses · manifest {m.hash()}")
    except (ManifestError, KeyError, ValueError) as e:
        st.error(str(e))
        return
    c1, c2 = st.columns([1, 4])
    c1.download_button("Download manifest", m.to_yaml().encode(), f"{m.name}.yaml", "text/yaml")
    with c2.expander("Manifest YAML"):
        st.code(m.to_yaml(), language="yaml")
    if st.button("Run experiment", type="primary"):
        bar = st.progress(0.0, text="Loading data…")
        log = st.empty()

        def prog(i, n, msg):
            bar.progress(min(i / max(n, 1), 1.0), text=f"{i}/{n} runs")
            log.caption(msg)

        try:
            exp_id = run_experiment(m, RunStore(db_path), progress=prog)
        except Exception as e:  # surface, don't hide
            st.exception(e)
            return
        st.session_state["experiment"] = exp_id
        st.success(f"Finished: {exp_id}. Open **Results** in the sidebar.")


# --------------------------------------------------------------------------- results
def heatmap(c: pd.DataFrame, title: str):
    d = c.reset_index(names="a").melt(id_vars="a", var_name="b", value_name="r")
    ch = alt.Chart(d).mark_rect().encode(
        x=alt.X("b:N", title=None, sort=None), y=alt.Y("a:N", title=None, sort=None),
        color=alt.Color("r:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1], reverse=True)),
        tooltip=["a", "b", alt.Tooltip("r:Q", format=".2f")],
    ).properties(title=title, height=max(300, 14 * len(c)))
    st.altair_chart(ch, width="stretch")


def results(db_path: str):
    store = RunStore(db_path)
    exps = store.experiments()
    if exps.empty:
        st.info("No experiments in this database yet.")
        return
    default = st.session_state.get("experiment")
    ids = exps["experiment_id"].tolist()
    exp_id = st.selectbox("Experiment", ids, index=ids.index(default) if default in ids else 0)
    exp = exps.set_index("experiment_id").loc[exp_id]
    manifest = json.loads(exp["manifest_json"])
    env = json.loads(exp["environment_json"])
    st.info(SIM_BANNER.format(backend=manifest["run"]["backend"]))
    with st.spinner("Aggregating…"):
        T = experiment_tables(store, exp_id)
    runs = store.runs(exp_id)

    tabs = st.tabs(["Computational benchmarks", "Output quality", "Hypotheses", "Signal correlations", "Provenance"])
    with tabs[0]:
        rs = T["run_summary"]
        st.subheader("Per-run cost by algorithm")
        st.dataframe(rs, width="stretch", hide_index=True)
        download(rs, "run_summary")
        stage_cols = [c for c in rs.columns if c.startswith("median_") and c.endswith("_s") and c not in ("median_s",)]
        if not rs.empty and stage_cols:
            d = rs.melt(id_vars=["workload", "config"], value_vars=stage_cols, var_name="stage", value_name="seconds")
            d["stage"] = d["stage"].str.removeprefix("median_").str.removesuffix("_s")
            st.altair_chart(alt.Chart(d.dropna()).mark_bar().encode(
                y=alt.Y("config:N", title=None), x=alt.X("seconds:Q", title="median seconds per task"),
                color="stage:N", row=alt.Row("workload:N", title=None), tooltip=["config", "stage", "seconds"],
            ).resolve_scale(y="independent"), width="stretch")
        sc = T["scaling"]
        if not sc.empty:
            st.subheader("Scaling with number of tasks N")
            wl = st.selectbox("Workload", sorted(sc.workload.unique()), key="sc-w")
            d = sc[sc.workload == wl]
            base = alt.Chart(d).encode(x=alt.X("N:Q", scale=alt.Scale(type="log")), color="algorithm:N")
            st.altair_chart(
                base.mark_line(point=True).encode(y=alt.Y("median_wall_s:Q", scale=alt.Scale(type="log"), title="batch wall time (s)"),
                                                  tooltip=list(d.columns))
                + base.mark_errorbar().encode(y="ci_low:Q", y2="ci_high:Q"),
                width="stretch",
            )
            st.dataframe(d, width="stretch", hide_index=True)
            download(sc, "scaling")
            cl = sorted(runs.loc[(runs.workload == wl) & (runs.kind == "classical"), "algorithm"].unique())
            qu = sorted(runs.loc[(runs.workload == wl) & (runs.kind == "quantum"), "algorithm"].unique())
            if cl and qu:
                st.subheader("Measured speedup S(N) = T_classical / T_quantum")
                c1, c2 = st.columns(2)
                a = c1.selectbox("Classical", cl, key="sp-c")
                b = c2.selectbox("Quantum", qu, key="sp-q")
                sp = bm.speedup_table(store.batches(exp_id), a, b)
                st.dataframe(sp, width="stretch", hide_index=True)
                st.caption("S > 1 means the quantum *simulation* was faster for these implementations, settings and hardware only.")

    with tabs[1]:
        fr, mq = T["quality_frontier"], T["matched_quality"]
        if fr.empty:
            st.info("No matched-estimand runs (needs Monte Carlo or hit-rate classical + quantum pairs).")
        else:
            st.subheader("Accuracy–cost frontier")
            est = st.selectbox("Estimand", sorted(fr.estimand.unique()))
            err_col = st.radio("Error measured against", ["abs_error", "abs_error_vs_own_target"], horizontal=True,
                               format_func=lambda s: "analytic / exact reference" if s == "abs_error" else "each method's own target (QAE: discretized)")
            d = mq[mq.estimand == est].copy()
            d = d[d[err_col] > 0]
            st.altair_chart(alt.Chart(d).mark_circle(opacity=0.6).encode(
                x=alt.X("seconds:Q", scale=alt.Scale(type="log"), title="seconds per task"),
                y=alt.Y(f"{err_col}:Q", scale=alt.Scale(type="log"), title="absolute error"),
                color="config:N", shape="kind:N", tooltip=list(d.columns),
            ), width="stretch")
            st.dataframe(fr[fr.estimand == est], width="stretch", hide_index=True)
            download(fr, "quality_frontier")
            download(mq, "matched_quality")
            if "discretization_bias" in mq:
                b = mq.loc[mq.estimand == "mc_expectation", "discretization_bias"].dropna()
                if len(b):
                    st.caption(f"Median |discretization bias| of the QAE grid: {b.abs().median():.4g} — compare with sampling error before crediting either method.")

    with tabs[2]:
        ht = T["hypotheses_statistical"]
        if not ht.empty:
            st.subheader("Statistical tests (FDR-adjusted within each config)")
            cfg = st.selectbox("Config", sorted(ht.config.unique()), key="ht-c")
            st.dataframe(ht[ht.config == cfg], width="stretch", hide_index=True)
            download(ht, "hypotheses_statistical")
        pt = T["hypotheses_predictive"]
        if not pt.empty:
            st.subheader("Out-of-sample predictive metrics (chronological test block)")
            cfg = st.selectbox("Config", sorted(pt.config.unique()), key="pt-c")
            st.dataframe(pt[pt.config == cfg], width="stretch", hide_index=True)
            download(pt, "hypotheses_predictive")
            st.caption("Strategy metrics are descriptive, net of the assumed cost, and use overlapping labels when h > 1.")

    with tabs[3]:
        mats = {k.removeprefix("corr_"): v for k, v in T.items() if k.startswith("corr_")}
        if not mats:
            st.info("No signal/prediction series stored.")
        else:
            which = st.selectbox("Matrix", list(mats), format_func=lambda s: {
                "signals": "(i) standardized signal values", "predictions": "(ii) out-of-sample predictions",
                "strategy_returns": "(iii) hypothetical strategy returns"}.get(s, s))
            partial = st.checkbox("Partial correlations")
            c = mats[which]
            heatmap(sg.partial_correlation(c) if partial else c, which)
            download(c, f"corr_{which}", index=True)
            if which == "signals":
                stab = sg.stability(store, runs)
                st.subheader("Temporal stability (first vs second half)")
                st.dataframe(stab.head(25), width="stretch", hide_index=True)
            st.caption("Computed only after every signal was evaluated independently. Correlation ≠ complementarity or causation.")

    with tabs[4]:
        snap = env.get("snapshot", {})
        c1, c2 = st.columns(2)
        c1.subheader("Dataset snapshot")
        c1.json({k: snap.get(k) for k in ("source", "vendor", "sha256", "start", "end", "n_rows", "adjustment",
                                         "calendar", "timezone", "missing_observations", "notes", "retrieved_at_utc")})
        c2.subheader("Environment")
        c2.json({k: v for k, v in env.items() if k != "snapshot"})
        st.subheader("Manifest")
        st.code(yaml.safe_dump(manifest, sort_keys=False), language="yaml")
        bad = runs[runs.status != "ok"]
        if len(bad):
            st.subheader(f"Failures ({len(bad)})")
            st.dataframe(bad[["algorithm", "task_id", "status", "error"]], width="stretch", hide_index=True)
        w = runs.explode("warnings").dropna(subset=["warnings"])
        if len(w):
            st.subheader("Warnings")
            st.dataframe(w.groupby(["algorithm", "warnings"]).size().rename("runs").reset_index(), width="stretch", hide_index=True)
        download(T["runs_raw"], "runs_raw")


# --------------------------------------------------------------------------- main
st.sidebar.title("Quantum Quant")
st.sidebar.caption("Quantum–classical benchmarking on NVDA research workloads")
page = st.sidebar.radio("Page", ["Build & run", "Results"], index=1 if "experiment" in st.session_state else 0)
db = st.sidebar.text_input("Results database", "results/qquant.sqlite")
with st.sidebar.expander("Algorithm catalog"):
    st.dataframe(algos.catalog()[["algorithm", "workload", "kind", "experimental"]], hide_index=True)
(builder if page == "Build & run" else results)(db)
