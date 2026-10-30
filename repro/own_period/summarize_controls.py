"""Validate and summarize the single-network runs of ``network_controls``.

    python -m repro.own_period.summarize_controls --run-dir D --compare-published --output V.json
    python -m repro.own_period.summarize_controls --run-dir O --output S.json [--folds ...]

``--compare-published`` compares every file of the run directory with the
published file of the same name (object order, donors, oracle errors and score
maps).  The summary gives, per seed and pooled over the selected folds, the
mean oracle error of every control and its change against the ``raw`` run of
the same directory: the lightcurve-exchange gap (mean of swap minus raw over
the asteroids, as ``aggregate_real_oof_input_swap_artifacts``) and the removal
gaps (mean of the removal minus mean of raw, as the input-removal figure),
with the mean, minimum and maximum over the seeds (the published ranges of
the ``\\Removal*`` values are the minimum and maximum over the seeds).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from repro import network_root as network

from .network_controls import CONTROLS, DIAGNOSTIC, output_name
from .nettests_common import EVALUATIONS, FOLLOWUP, SEEDS, max_abs, write_json_once

FINER_PUBLISHED = FOLLOWUP / "data/interpretation-20260923/ablations"


def published_path(fold, seed, control) -> Path:
    name = output_name(fold, seed, control)
    if control in ("raw", "input-swap") or control in DIAGNOSTIC:
        return EVALUATIONS / name
    return FINER_PUBLISHED / name


def compare(run_dir: Path) -> dict:
    if network.switched():
        raise ValueError("published comparison is only for the original networks; use Check 3")
    rows, worst_error, worst_grid = [], 0.0, 0.0
    for path in sorted(run_dir.glob("real-fold-*.npz")):
        stem = path.stem.split("-")
        fold, seed = int(stem[2]), int(stem[4])
        control = "-".join(stem[5:]) or "raw"
        reference = published_path(fold, seed, control)
        with np.load(path) as new, np.load(reference) as old:
            same_ids = bool(np.array_equal(new["object_ids"], old["object_ids"]))
            same_donors = (
                bool(np.array_equal(new["input_donor_ids"], old["input_donor_ids"]))
                if "input_donor_ids" in old
                else None
            )
            error_diff = max_abs(new["oracle_errors_deg"], old["oracle_errors_deg"])
            grid_diff = max_abs(new["score_grids"], old["score_grids"])
        worst_error, worst_grid = max(worst_error, error_diff), max(worst_grid, grid_diff)
        rows.append(
            {
                "file": path.name,
                "published": str(reference.relative_to(FOLLOWUP)),
                "same_object_order": same_ids,
                "same_donors": same_donors,
                "max_abs_oracle_error_diff_deg": error_diff,
                "max_abs_score_grid_diff": grid_diff,
            }
        )
    return {
        "files_compared": len(rows),
        "all_object_orders_equal": all(r["same_object_order"] for r in rows),
        "all_donors_equal": all(r["same_donors"] in (None, True) for r in rows),
        "max_abs_oracle_error_diff_deg": worst_error,
        "max_abs_score_grid_diff": worst_grid,
        "passes_1e-6_deg": worst_error <= 1e-6,
        "files": rows,
    }


def summarize(run_dir: Path, folds) -> dict:
    def load(seed, control):
        values, ids = [], []
        for fold in folds:
            path = run_dir / output_name(fold, seed, control)
            if not path.exists():
                return None, None
            network.check_npz(path)
            with np.load(path) as data:
                values.append(np.asarray(data["oracle_errors_deg"], dtype=np.float64))
                ids.extend(data["object_ids"].astype(str).tolist())
        return np.concatenate(values), ids

    result = {"folds": list(folds), "controls": {}}
    raw = {seed: load(seed, "raw") for seed in SEEDS}
    result["raw_mean_by_seed"] = {
        str(s): float(v[0].mean()) for s, v in raw.items() if v[0] is not None
    }
    for control in CONTROLS:
        if control == "raw":
            continue
        by_seed, means = {}, {}
        for seed in SEEDS:
            values, ids = load(seed, control)
            base, base_ids = raw[seed]
            if values is None or base is None:
                continue
            if ids != base_ids:
                raise ValueError(f"object order differs for {control} seed {seed}")
            by_seed[str(seed)] = float(np.mean(values - base))
            means[str(seed)] = float(values.mean())
        if not by_seed:
            continue
        gaps = np.asarray(list(by_seed.values()))
        result["controls"][control] = {
            "n_objects": len(base_ids),
            "n_seeds": len(by_seed),
            "mean_error_by_seed_deg": means,
            "gap_by_seed_deg": by_seed,
            "gap_mean_deg": float(gaps.mean()),
            "gap_min_deg": float(gaps.min()),
            "gap_max_deg": float(gaps.max()),
            "gap_positive_for_every_seed": bool(np.all(gaps > 0)),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--compare-published", action="store_true")
    args = parser.parse_args()
    network.guard_output(args.output)
    network.check_json(args.run_dir / "period-source.json")
    value = {
        **network.json_metadata(),
        "run_dir": str(args.run_dir),
        "period_source": json.loads((args.run_dir / "period-source.json").read_text()),
        "summary": summarize(args.run_dir, args.folds),
    }
    if args.compare_published:
        value["comparison_with_published"] = compare(args.run_dir)
    write_json_once(args.output, value)
    brief = {
        k: round(v["gap_mean_deg"], 3) for k, v in value["summary"]["controls"].items()
    }
    print(json.dumps({"output": str(args.output), "gaps": brief}, indent=1))
    if args.compare_published:
        c = value["comparison_with_published"]
        print(json.dumps({k: v for k, v in c.items() if k != "files"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
