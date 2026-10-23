"""Command-line entry point.

    qquant algorithms [--workload W]       list registered algorithms and parameters
    qquant validate CONFIG                 check a manifest without running it
    qquant hypotheses CONFIG               print the hypothesis registry for a manifest
    qquant run CONFIG [--export DIR]       run an experiment and optionally export tables
    qquant export EXPERIMENT_ID [--db DB]  export tables for a stored experiment
    qquant ui                              launch the Streamlit interface
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from .config import ExperimentManifest, ManifestError


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="qquant")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("algorithms")
    a.add_argument("--workload")
    v = sub.add_parser("validate")
    v.add_argument("config")
    h = sub.add_parser("hypotheses")
    h.add_argument("config")
    r = sub.add_parser("run")
    r.add_argument("config")
    r.add_argument("--export", default=None, help="directory for exported tables")
    e = sub.add_parser("export")
    e.add_argument("experiment_id")
    e.add_argument("--db", default="results/qquant.sqlite")
    e.add_argument("--out", default="results/exports")
    e.add_argument("--format", default="csv", choices=("csv", "json"))
    sub.add_parser("ui")
    args = ap.parse_args(argv)

    from . import algorithms as algos

    pd.set_option("display.width", 200, "display.max_colwidth", 80)
    try:
        if args.cmd == "algorithms":
            print(algos.catalog(args.workload).to_string(index=False))
        elif args.cmd == "validate":
            from .scheduler import plan

            m = ExperimentManifest.from_yaml(args.config)
            planned = plan(m)
            print(f"OK: {m.name} ({m.hash()}): " + ", ".join(f"{p.workload}/{p.name}" for p in planned))
        elif args.cmd == "hypotheses":
            from .hypotheses import generate, registry_table

            m = ExperimentManifest.from_yaml(args.config)
            print(registry_table(generate(m.hypotheses))[["hypothesis_id", "cluster", "description"]].to_string(index=False))
        elif args.cmd == "run":
            from .analysis.report import export
            from .scheduler import run_experiment
            from .storage import RunStore

            m = ExperimentManifest.from_yaml(args.config)
            store = RunStore(m.run.db_path)
            exp_id = run_experiment(m, store, progress=lambda i, n, msg: print(f"[{i}/{n}] {msg}", flush=True))
            print(f"experiment_id = {exp_id}")
            if args.export:
                print(f"exported -> {export(store, exp_id, args.export, m.run.export_format)}")
        elif args.cmd == "export":
            from .analysis.report import export
            from .storage import RunStore

            print(export(RunStore(args.db), args.experiment_id, args.out, args.format))
        elif args.cmd == "ui":
            app = Path(__file__).parent / "ui" / "app.py"
            return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app)])
    except ManifestError as e:
        print(f"Manifest error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
