"""Training versus test errors (internal-role diagnostic) at a chosen period.

    python -m repro.own_period.train_vs_test run --periods damit --out-dir D [--folds 0]
    python -m repro.own_period.train_vs_test run --periods own-periods-170.json --out-dir O
    python -m repro.own_period.train_vs_test report --cells O --output O-report.json [--compare-published]

``run`` follows ``lc_pipeline.k3.internal_validation.run`` cell by cell (the
locked rows, bundles, lightcurve reader, ``diagnostic_predict`` and error
function of the published diagnostic), except that each lightcurve is folded
with the period of ``--periods``.  With ``damit`` the locked period of the
published study (the first DAMIT period) is used.  Training, validation and
test roles are all evaluated with the same period source.

``report`` pairs the mean training-role error of each asteroid with its
test-role error of the same run and applies the published fold-wise paired
bootstrap (``internal_validation_reporting.fold_bootstrap``, seed 20260916).
The test-role errors come from this run, so both sides use the same period.

Caveat: the networks were trained with the DAMIT period, so at the own period
the training asteroids are no longer the inputs that were fitted.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from lc_pipeline.k3.internal_validation import _error, diagnostic_predict
from lc_pipeline.k3.internal_validation_reporting import fold_bootstrap
from lc_pipeline.k3.reliability_study import unseal, verify
from repro import network_root as network

from .nettests_common import FOLLOWUP, max_abs, period_override, write_json_once

PUBLISHED = FOLLOWUP / "data/internal-validation-20260916/run-1"


def run(args) -> int:
    import torch

    from lc_pipeline.k3.bundle import K3EnsemblePredictor
    from lc_pipeline.k3.reliability_sampling import read_lc, to_epochs
    from lc_pipeline.v2.preprocessing import KnownPeriod

    periods, provenance, label = period_override(args.periods)
    torch.set_num_threads(6)
    lock = unseal(PUBLISHED / "study.json")
    bundles = [network.bundle_record(f, lock['bundles']) for f in range(5)]
    by_fold = {
        json.loads((Path(b["directory"]) / "bundle.json").read_text())["fold"]: b["directory"]
        for b in bundles
    }
    network.guard_output(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for fold in args.folds:
        destination = args.out_dir / f"cells-fold-{fold}.jsonl"
        if destination.exists():
            network.check_jsonl(destination, identity_fields=('object_id', 'fold', 'role'))
            continue
        rows = [r for r in lock["rows"] if r["fold"] == fold]
        rows = [r for r in rows if r["role"] == "test"] + [r for r in rows if r["role"] != "test"]
        if args.limit:
            rows = rows[: args.limit]
        cells = []
        if True:
            predictor = K3EnsemblePredictor(by_fold[fold], device=args.device or lock["device"],
                                            allow_legacy=not network.switched())
            for row in rows:
                period = row["period_hours"] if periods is None else periods[row["object_id"]]
                prov = "locked-internal-diagnostic" if periods is None else provenance
                started = time.monotonic()
                curve = read_lc(verify(row["lightcurve"]), object_id=row["object_id"], period_hours=period)
                result = diagnostic_predict(
                    predictor,
                    to_epochs(curve),
                    known_period=KnownPeriod(hours=period, provenance=prov),
                    object_id=row["object_id"],
                    expected_role=row["role"] + "_ids",
                )
                axes = [x["axis_xyz"] for x in result["axes"]]
                value = {
                    "object_id": row["object_id"],
                    "fold": fold,
                    "role": row["role"],
                    "period_source": label,
                    "period_hours": float(period),
                    "axes": axes,
                    "error_deg": _error(axes, row["references"]),
                    "runtime_s": time.monotonic() - started,
                    **network.json_metadata(),
                }
                cells.append(value)
            del predictor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            with destination.open("x") as stream:
                stream.writelines(json.dumps(v) + "\n" for v in cells)
            print(json.dumps({"fold": fold, "cells": len(cells),
                              "mean_error_by_role": {r: float(np.mean([v["error_deg"] for v in cells if v["role"] == r]))
                                                     for r in ("train", "validation", "test")}}), flush=True)
    return 0


def report(args) -> int:
    network.guard_output(args.output)
    if network.switched() and args.compare_published:
        raise ValueError('historical numerical parity is checked separately with the old networks')
    for path in args.cells.glob('cells-fold-*.jsonl'):
        network.check_jsonl(path, identity_fields=('object_id', 'fold', 'role'))
    cells = [
        json.loads(line)
        for path in sorted(args.cells.glob("cells-fold-*.jsonl"))
        for line in path.read_text().splitlines()
    ]
    keys = [(c["object_id"], c["fold"], c["role"]) for c in cells]
    if len(keys) != len(set(keys)):
        raise SystemExit("duplicate cells")
    lock = unseal(PUBLISHED / "study.json")
    expected = {(r["object_id"], r["fold"], r["role"]) for r in lock["rows"]}
    complete = set(keys) == expected
    test = {c["object_id"]: c for c in cells if c["role"] == "test"}
    train = defaultdict(list)
    for c in cells:
        if c["role"] == "train":
            train[c["object_id"]].append(c["error_deg"])
    pairs = [
        {
            "object_id": oid,
            "mean_train_error_deg": float(np.mean(v)),
            "oof_error_deg": test[oid]["error_deg"],
            "difference_deg": float(np.mean(v)) - test[oid]["error_deg"],
            "train_repetitions": len(v),
        }
        for oid, v in sorted(train.items())
        if oid in test
    ]
    folds = {oid: c["fold"] for oid, c in test.items()}
    if network.switched() and not complete:
        raise ValueError('incomplete internal-validation role cells')
    value = {
        **network.json_metadata(),
        "cells": str(args.cells),
        "period_source": sorted({c["period_source"] for c in cells}),
        "complete": complete,
        "n_cells": len(cells),
        "by_role": {
            role: {
                "n": sum(c["role"] == role for c in cells),
                "mean_error_deg": float(np.mean([c["error_deg"] for c in cells if c["role"] == role])),
            }
            for role in ("train", "validation", "test")
            if any(c["role"] == role for c in cells)
        },
        "train_oof_pairs": {"n": len(pairs), "bootstrap": fold_bootstrap(pairs, folds) if pairs else None},
        "pairs": pairs,
    }
    if args.compare_published:
        old = {}
        for path in (PUBLISHED / "cells").glob("*/*/*/result.json"):
            cell = unseal(path)
            old[(cell["object_id"], cell["fold"], cell["role"])] = cell
        diffs = [max_abs(c["axes"], old[(c["object_id"], c["fold"], c["role"])]["axes"]) for c in cells]
        errs = [abs(c["error_deg"] - old[(c["object_id"], c["fold"], c["role"])]["error_deg"]) for c in cells]
        value["comparison_with_published"] = {
            "cells_compared": len(cells),
            "max_abs_axis_diff": float(max(diffs)),
            "max_abs_error_diff_deg": float(max(errs)),
            "passes_1e-6": float(max(diffs)) <= 1e-6,
        }
        published = json.loads((PUBLISHED / "report-export-publication/summary.json").read_text())
        value["published_train_oof_pairs"] = published["train_oof_pairs"]
    write_json_once(args.output, value)
    print(json.dumps({k: v for k, v in value.items() if k != "pairs"}, indent=1))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    a = commands.add_parser("run")
    a.add_argument("--periods", required=True)
    a.add_argument("--out-dir", type=Path, required=True)
    a.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    a.add_argument("--device", default=None, help="default: the locked device (cuda)")
    a.add_argument("--limit", type=int, default=0, help="smoke tests only: first N cells per fold")
    b = commands.add_parser("report")
    b.add_argument("--cells", type=Path, required=True, help="directory with cells-fold-*.jsonl")
    b.add_argument("--output", type=Path, required=True)
    b.add_argument("--compare-published", action="store_true")
    args = parser.parse_args()
    return run(args) if args.command == "run" else report(args)


if __name__ == "__main__":
    raise SystemExit(main())
