"""Score-map analyses of Paper I rerun on a chosen out-of-fold run (CPU only).

    python -m repro.own_period.maps.analyses --run RUN_DIR --periods damit|FILE --out OUT_DIR

RUN_DIR/evaluations holds files in the layout of the published
``real-oof-ensemble.npz`` and ``real-fold-<f>-seed-<s>.npz`` (written by
``predict_oof_full.py``, or the published directory itself for a check of
this script).  Every analysis is the published script, either imported with
its input and output paths redirected (mirror symmetry, candidate count) or
copied with only the paths changed (amplitude-aspect comparison and paired
comparison, primary intervals, reference strata and Spearman matrix of the
reliability supplement, reference reassignment of the internal validation).
Seeds, draws and bins are the published ones.  Outputs are written once.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPRO = HERE.parents[1]
sys.path.insert(0, str(REPRO.parent))
sys.path.insert(0, str(REPRO))
import amplitude_aspect_comparison as aa  # noqa: E402
import candidate_count as cc  # noqa: E402
import compute_reference_baselines as base  # noqa: E402
import mirror_symmetry as ms  # noqa: E402

from lc_pipeline.k3.evaluation import modes_from_score_grid  # noqa: E402
from lc_pipeline.k3.grid import axial_healpix_grid  # noqa: E402
from lc_pipeline.k3.internal_validation_reporting import reassignment_null  # noqa: E402
from lc_pipeline.k3.reliability_supplement import _unseal  # noqa: E402

from repro.own_period import DataPath

FOLLOWUP = DataPath("../..")
RELIABILITY = FOLLOWUP / "data/reliability-20260915/run-1"
OWN = DataPath()
SEEDS = (17, 42, 137, 777, 2027)


def write_once(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=1) + "\n")


def run_mirror_symmetry(ensemble: Path, out: Path) -> dict:
    target = out / "mirror-symmetry.json"
    if target.exists():
        raise FileExistsError(target)
    ms.ENSEMBLE, ms.OUTPUT = ensemble, target
    with contextlib.redirect_stdout(io.StringIO()):
        ms.main()
    return json.loads(target.read_text())


def run_candidate_count(ensemble: Path, out: Path) -> dict:
    target = out / "candidate-count.json"
    if target.exists():
        raise FileExistsError(target)
    cc.ENSEMBLE, cc.OUTPUT = ensemble, target
    with contextlib.redirect_stdout(io.StringIO()):
        cc.main()
    return json.loads(target.read_text())


def run_amplitude_aspect(ensemble_path: Path, out: Path) -> tuple[list, dict]:
    """``amplitude_aspect_comparison.main`` with the output paths changed and the
    printed null kept in a JSON file."""
    ensemble = np.load(ensemble_path)
    grid = axial_healpix_grid(32)
    poles = base.reference_poles()
    standard = base.standard_axes()
    rows, maps, kept = [], [], []
    for index, name in enumerate(ensemble["object_ids"]):
        name = str(name)
        amplitudes, directions = aa.lightcurve_amplitudes(name)
        if len(amplitudes) < 3:
            continue
        physical = aa.amplitude_aspect_scores(amplitudes, directions, grid.vectors)
        maps.append(physical.astype(np.float32))
        kept.append(index)
        delphi = ensemble["score_grids"][index].astype(float)
        modes = modes_from_score_grid(physical.astype(np.float32))
        axes = np.array([mode.axis_xyz for mode in modes])
        rows.append(
            {
                "object_id": name,
                "lightcurves_used": int(len(amplitudes)),
                "map_correlation": float(np.corrcoef(physical, delphi)[0, 1]),
                "amplitude_aspect_oracle_deg": base.axial_error_deg(axes, poles[name]),
                "delphi_oracle_deg": float(ensemble["oracle_errors_deg"][index]),
                "delphi_grid_oracle_deg": float(ensemble["grid_oracle_errors_deg"][index]),
                "standard_oracle_deg": base.axial_error_deg(standard, poles[name]),
            }
        )
    write_once(out / "amplitude-aspect.json", rows)
    maps = np.array(maps)
    delphi_maps = ensemble["score_grids"][kept].astype(float)
    with (out / "amplitude-aspect.npz").open("xb") as handle:
        np.savez_compressed(
            handle,
            object_ids=np.array([rows[i]["object_id"] for i in range(len(rows))]),
            amplitude_aspect_maps=maps,
        )
    generator = np.random.default_rng(20260923)
    null = []
    for _ in range(20):
        order = generator.permutation(len(kept))
        while np.any(order == np.arange(len(kept))):
            order = generator.permutation(len(kept))
        null.append(
            np.median([np.corrcoef(maps[order[i]], delphi_maps[i])[0, 1] for i in range(len(kept))])
        )
    table = {key: np.array([row[key] for row in rows]) for key in rows[0] if key != "object_id"}
    summary = {
        "n": len(rows),
        "cross_asteroid_null_median_correlation": {
            "mean": float(np.mean(null)), "min": float(min(null)), "max": float(max(null))},
        "map_correlation_median": float(np.median(table["map_correlation"])),
        "map_correlation_iqr": [float(v) for v in np.percentile(table["map_correlation"], (25, 75))],
    }
    for key in ("amplitude_aspect_oracle_deg", "delphi_grid_oracle_deg", "delphi_oracle_deg",
                "standard_oracle_deg"):
        summary[key] = {"mean": float(table[key].mean()), "median": float(np.median(table[key])),
                        "within20": float(np.mean(table[key] < 20))}
    # Agreement groups of Section 6.2.2 and Figure 9d (same bin edges).
    groups = []
    edges = (-1.0, 0.5, 0.7, 0.8, 1.0)
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (table["map_correlation"] >= low) & (table["map_correlation"] < high)
        groups.append({"low": low, "high": high, "n": int(mask.sum()),
                       "delphi_mean_deg": float(table["delphi_oracle_deg"][mask].mean()),
                       "amplitude_aspect_mean_deg": float(table["amplitude_aspect_oracle_deg"][mask].mean())})
    summary["agreement_groups"] = groups
    write_once(out / "amplitude-aspect-summary.json", summary)
    return rows, summary


def run_amplitude_aspect_paired(rows: list, out: Path) -> dict:
    """``amplitude_aspect_paired.main`` with the folder changed."""
    delphi = np.array([row["delphi_oracle_deg"] for row in rows])
    amplitude = np.array([row["amplitude_aspect_oracle_deg"] for row in rows])
    standard = np.array([row["standard_oracle_deg"] for row in rows])
    draws = np.random.default_rng(20260930).integers(0, len(rows), size=(10000, len(rows)))

    def paired(difference):
        boot = difference[draws].mean(axis=1)
        return [float(difference.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]

    result = {
        "n_objects": len(rows),
        "mean_error_deg": {"delphi": float(delphi.mean()), "amplitude_aspect": float(amplitude.mean()),
                           "standard": float(standard.mean())},
        "within20": {"delphi": float(np.mean(delphi < 20)), "amplitude_aspect": float(np.mean(amplitude < 20)),
                     "standard": float(np.mean(standard < 20))},
        "amplitude_aspect_minus_delphi_deg": paired(amplitude - delphi),
        "standard_minus_amplitude_aspect_deg": paired(standard - amplitude),
        "delphi_closer_count": int(np.sum(delphi < amplitude)),
    }
    write_once(out / "amplitude-aspect-paired.json", result)
    return result


def run_primary_intervals(errors: np.ndarray, out: Path) -> dict:
    """Bootstrap part of ``primary_intervals.main`` (seed 20260930); the random-axes
    curve is reference-only and kept as published."""
    draws = np.random.default_rng(20260930).integers(0, len(errors), (10000, len(errors)))
    within = errors <= 20
    within30 = errors <= 30
    summary = dict(
        n=int(len(errors)),
        mean_deg=float(errors.mean()),
        mean_ci95_deg=[float(v) for v in np.percentile(errors[draws].mean(1), (2.5, 97.5))],
        within20=float(within.mean()),
        within20_ci95=[float(v) for v in np.percentile(within[draws].mean(1), (2.5, 97.5))],
        within30=float(within30.mean()),
        median_deg=float(np.median(errors)),
    )
    write_once(out / "primary-intervals.json", summary)
    return summary


def avgrank(x):
    order = np.argsort(x, kind="stable")
    result = np.empty(len(x))
    pos = 0
    while pos < len(x):
        end = pos + 1
        while end < len(x) and x[order[end]] == x[order[pos]]:
            end += 1
        result[order[pos:end]] = (pos + end - 1) / 2 + 1
        pos = end
    return result


def run_reliability_strata(error_of: dict, out: Path, own_periods: dict | None) -> dict:
    """Reference strata and Spearman matrix of ``reliability_supplement.export``,
    with the oracle error of each asteroid taken from this run.  The DAMIT period
    stays the "period_hours" property; the own period is an extra column."""
    study = _unseal(RELIABILITY / "study.json")
    scored = _unseal(RELIABILITY / "scored-rows.json")
    objects = study["objects"]
    full = [r for r in scored if r["condition"] == "full" and r["status"] == "ok"]
    by_id = {r["object_id"]: r for r in full}
    association = []
    for field, label in [("reference_primary_abs_latitude_deg", "primary_reference_abs_latitude_deg"),
                         ("reference_solution_count", "reference_solution_count")]:
        vals = np.array([float(by_id[o["object_id"]][field]) for o in objects])
        err = np.array([float(error_of[o["object_id"]]) for o in objects])
        rho = float(np.corrcoef(avgrank(vals), avgrank(err))[0, 1])
        bins = [(1, 1), (2, 2), (3, 3)] if field == "reference_solution_count" else [(0, 30), (30, 60), (60, 90)]
        for i, (low, high) in enumerate(bins):
            if field == "reference_solution_count":
                mask = (vals >= low) & (vals <= high)
            else:
                mask = (vals >= low) & ((vals < high) if i < len(bins) - 1 else (vals <= high))
            association.append({"property": label, "spearman_rho": rho, "stratum": f"{low}-{high}",
                                "n": int(mask.sum()),
                                "mean_error_deg": float(err[mask].mean()) if mask.any() else None})
    fields = {
        "oracle_error_deg": np.array([error_of[o["object_id"]] for o in objects]),
        "log_observations": np.log(np.array([o["native_observations"] for o in objects])),
        "native_sessions": np.array([o["native_sessions"] for o in objects]),
        "merged_blocks": np.array([o["merged_blocks"] for o in objects]),
        "baseline_years": np.array([o["time_span_days"] / 365.25 for o in objects]),
        "period_hours": np.array([o["period_hours"] for o in objects]),
        "primary_abs_latitude_deg": np.array(
            [by_id[o["object_id"]]["reference_primary_abs_latitude_deg"] for o in objects]),
        "reference_count": np.array([by_id[o["object_id"]]["reference_solution_count"] for o in objects]),
    }
    if own_periods is not None:
        fields["own_period_hours"] = np.array([own_periods[o["object_id"]] for o in objects])
    names = list(fields)
    corr = {a: {b: float(np.corrcoef(avgrank(fields[a]), avgrank(fields[b]))[0, 1]) for b in names}
            for a in names}
    result = {"reference_strata": association, "spearman_matrix": corr,
              "note": "period_hours is the DAMIT period (asteroid property); own_period_hours is extra"}
    write_once(out / "reliability-strata-spearman.json", result)
    return result


def run_reassignment(ensemble, out: Path) -> dict:
    """``internal_validation_reporting.reassignment_null`` (seed 20260916, 10,000
    draws) on the published row order and strata, with this run's axes and errors."""
    from lc_pipeline.v2.data import load_catalog

    scored = _unseal(RELIABILITY / "scored-rows.json")
    oof = [x for x in scored
           if x.get("condition") == "full" and x.get("repeat") == 0 and x.get("status") == "ok"]
    catalog = load_catalog(base.CATALOG)  # the catalogue of the published study (same SHA-256)
    refs = {x["object_id"]: [list(v) for v in catalog[x["object_id"]].solution_vectors] for x in oof}
    index = {str(v): i for i, v in enumerate(ensemble["object_ids"])}
    rows = []
    for x in oof:
        i = index[x["object_id"]]
        rows.append({"object_id": x["object_id"], "fold": x["fold"],
                     "axes": ensemble["refined_axes"][i].tolist(),
                     "error_deg": float(ensemble["oracle_errors_deg"][i])})
    result = reassignment_null(rows, refs)
    write_once(out / "reference-reassignment.json", result)
    return {k: v for k, v in result.items() if k != "null_draws_mean_error_deg"}


def phase_bin_occupancy(periods: dict, provenance: str) -> dict:
    from lc_pipeline.k3.damit import load_damit_object
    from lc_pipeline.k3.tokenizer import tokenize_epochs
    from lc_pipeline.v2.preprocessing import KnownPeriod

    dump = FOLLOWUP.parent / "damit-20250610T000301Z"
    filled, points = [], []
    for oid in sorted(base.reference_poles()):
        source = load_damit_object(dump, oid)
        tokenized = tokenize_epochs(source.epochs, known_period=KnownPeriod(float(periods[oid]), provenance))
        filled.extend(np.asarray(tokenized.phase_mask).sum(axis=1).tolist())
    filled = np.asarray(filled)
    return {"lightcurves": int(filled.size), "median_filled_bins": float(np.median(filled)),
            "quartiles_filled_bins": [float(v) for v in np.percentile(filled, (25, 75))]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="directory holding evaluations/")
    parser.add_argument("--periods", required=True, help="damit or a periods JSON (for M5 and the own-period row)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--skip", nargs="*", default=[])
    args = parser.parse_args()
    evaluations = args.run / "evaluations" if (args.run / "evaluations").is_dir() else args.run
    ensemble_path = evaluations / "real-oof-ensemble.npz"
    args.out.mkdir(parents=True, exist_ok=True)
    ensemble = np.load(ensemble_path)
    errors = np.asarray(ensemble["oracle_errors_deg"], dtype=float)
    grid_errors = np.asarray(ensemble["grid_oracle_errors_deg"], dtype=float)
    names = [str(v) for v in ensemble["object_ids"]]
    error_of = dict(zip(names, errors.tolist()))

    if args.periods == "damit":
        catalog = {r["object_id"]: r for r in map(json.loads, base.CATALOG.read_text().splitlines())}
        periods = {oid: float(catalog[oid]["solutions"][0]["period_hours"]) for oid in names}
        provenance, own_periods = "frozen-DAMIT-primary-solution-fixed-period", None
    else:
        value = json.loads(Path(args.periods).read_text())
        periods, provenance = value["periods_hours"], value["provenance"]
        own_periods = periods

    summary: dict = {"run": str(ensemble_path), "periods": args.periods}
    folds = ensemble["folds"]
    summary["overall"] = {
        "n": len(errors), "mean_deg": float(errors.mean()), "median_deg": float(np.median(errors)),
        "within20": float(np.mean(errors <= 20)), "within30": float(np.mean(errors <= 30)),
        "min_deg": float(errors.min()), "max_deg": float(errors.max()),
        "sd_ddof1_deg": float(errors.std(ddof=1)), "sd_ddof0_deg": float(errors.std()),
        "grid_mean_deg": float(grid_errors.mean()), "adjusted_mean_deg": float(errors.mean()),
    }
    summary["folds"] = [
        {"fold": f, "n": int((folds == f).sum()), "mean_deg": float(errors[folds == f].mean()),
         "median_deg": float(np.median(errors[folds == f]))} for f in range(5)]
    seeds = {}
    for s in SEEDS:
        files = [evaluations / f"real-fold-{f}-seed-{s}.npz" for f in range(5)]
        if all(p.exists() for p in files):
            seeds[str(s)] = float(np.concatenate([np.load(p)["oracle_errors_deg"] for p in files]).mean())
    summary["per_seed_mean_deg"] = seeds
    if seeds:
        summary["ensemble_better_than_every_seed"] = bool(errors.mean() < min(seeds.values()))
    seconds = np.asarray(ensemble["inference_wall_seconds"], dtype=float)
    summary["network_seconds"] = {"median": float(np.median(seconds)), "mean": float(np.mean(seconds))}
    names_index = {n: i for i, n in enumerate(names)}
    summary["phaethon_2514_oracle_deg"] = float(errors[names_index["asteroid_2514"]])

    if "phase-bins" not in args.skip:
        summary["phase_bin_occupancy"] = phase_bin_occupancy(periods, provenance)
    if "intervals" not in args.skip:
        summary["primary_intervals"] = run_primary_intervals(errors, args.out)
    if "mirror" not in args.skip:
        mirror = run_mirror_symmetry(ensemble_path, args.out)["summary"]
        summary["mirror_symmetry"] = {k: mirror[k] for k in (
            "mirror_correlation_median", "symmetry_by_observer_latitude", "mirror_pairs_among_candidates",
            "reference_preferred_over_mirror", "main_sample_direction", "coordinate_split")}
    if "count" not in args.skip:
        count = run_candidate_count(ensemble_path, args.out)
        summary["candidate_count"] = count
        areas = [row["search_area_mean_radius"] for row in count["k"]]
        summary["candidate_count_smallest_area_k"] = int(count["k"][int(np.argmin(areas))]["k"])
    if "amplitude" not in args.skip:
        rows, amp = run_amplitude_aspect(ensemble_path, args.out)
        summary["amplitude_aspect"] = amp
        summary["amplitude_aspect_paired"] = run_amplitude_aspect_paired(rows, args.out)
    if "strata" not in args.skip:
        strata = run_reliability_strata(error_of, args.out, own_periods)
        summary["reference_strata"] = strata["reference_strata"]
        summary["spearman_with_oracle_error"] = strata["spearman_matrix"]["oracle_error_deg"]
    if "reassignment" not in args.skip:
        summary["reference_reassignment"] = run_reassignment(ensemble, args.out)
    if own_periods is not None and "ztf" not in args.skip:
        ztf = json.loads((OWN / "own-period-scores.json").read_text())["ztf"]
        at90, excl = ztf["own_period_failed_at_90"], ztf["own_period_failed_excluded"]
        summary["ztf_all_intended"] = {
            "source": "own-period-scores.json ztf; nine published no-data failures at 90 deg",
            "both_failure_kinds_at_90_deg": (at90["n"] * at90["mean_deg"] + 9 * 90.0) / (at90["n"] + 9),
            "n_with_both_at_90": at90["n"] + 9,
            "period_failures_excluded_deg": (excl["n"] * excl["mean_deg"] + 9 * 90.0) / (excl["n"] + 9),
            "n_with_period_failures_excluded": excl["n"] + 9,
        }
    write_once(args.out / "summary.json", summary)
    print(json.dumps(summary, indent=1)[:6000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
