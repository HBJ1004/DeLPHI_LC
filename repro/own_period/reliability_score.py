"""Score the own-period reliability runs (reference poles and periods are read here only).

    python -m repro.own_period.reliability_score validate-scoring
    python -m repro.own_period.reliability_score score

``validate-scoring`` applies this scoring code to the published sealed rows and
fits of ``data/reliability-20260915/run-1`` and checks that it reproduces the
published sampling table, data-degradation table, sampling baselines and the
withheld table.  ``score`` then scores the own-period predictions in the same
way, with failures at 90 deg (as published) and, separately, excluded.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from repro.own_period import reliability as R

FIG8_BLOCKS = (1, 3, 5, 10)
CAPS = ("10", "20", "50", "100", "200", "500", "None")
DAMIT_CATALOG = R.FOLLOWUP / "inputs/training-run/repro/data/damit-20250610T000301Z/catalog.jsonl"


def catalog_references(lock):
    from lc_pipeline.v2.data import load_catalog
    path = Path(lock["catalog"]["path"])
    if R.sha256_file(path) != lock["catalog"]["sha256"]:
        raise ValueError("catalog changed")
    return load_catalog(path)


def damit_periods() -> dict[str, float]:
    out = {}
    for line in DAMIT_CATALOG.read_text().splitlines():
        row = json.loads(line)
        if row.get("solutions"):
            out[row["object_id"]] = float(row["solutions"][0]["period_hours"])
    return out


def period_class(own, reference):
    if own is None or not np.isfinite(own):
        return "failed"
    for name, factor in (("P", 1.0), ("P/2", 0.5), ("2P", 2.0)):
        if abs(own / reference - factor) <= 0.01 * factor:
            return name
    return "other"


# ----------------------------------------------------------------------------- rows

def build_rows(records: dict, lock, schedule, catalog):
    """Published-format scored rows (``reliability_study.score``) from prediction records."""
    from lc_pipeline.k3.reliability_study import oracle_error
    grid = set(lock["grid_ids"])
    conditions = R.sampling_conditions(lock)
    rows = []
    for entry in lock["objects"]:
        oid = entry["object_id"]
        refs = catalog[oid].solution_vectors
        for config in conditions:
            for repeat in R.repeats_of(config, schedule):
                if config["family"] == "two_d" and oid not in grid:
                    rows.append({"object_id": oid, "fold": entry["fold"], "condition": config["condition"],
                                 "repeat": repeat, "seed": R.SEEDS[repeat], "status": "ineligible",
                                 "axes": None, "error_deg": None})
                    continue
                rec = records[(oid, config["condition"], repeat)]
                value = {"object_id": oid, "fold": entry["fold"], "condition": config["condition"],
                         "repeat": repeat, "seed": R.SEEDS[repeat], "status": rec["status"],
                         "axes": rec["axes"] if rec["status"] == "ok" else None,
                         "period_hours": rec.get("period_hours"),
                         "n_points": rec.get("metadata", {}).get("n_points")}
                value["error_deg"] = oracle_error(rec["axes"], refs) if rec["status"] == "ok" else 90.0
                rows.append(value)
        base = next(r for r in rows if r["object_id"] == oid and r["condition"] == "full")
        if oid in grid:
            rows.append({**base, "condition": "grid-full"})
        else:
            rows.append({"object_id": oid, "fold": entry["fold"], "condition": "grid-full", "repeat": 0,
                         "seed": R.SEEDS[0], "status": "ineligible", "axes": None, "error_deg": None})
    return rows


def summarize(rows, lock, schedule):
    from lc_pipeline.k3.reliability_scoring import render_reliability_csv, score_reliability
    from lc_pipeline.k3.reliability_study import baselines_by_condition
    conditions = R.sampling_conditions(lock)
    repeats = {c["condition"]: list(R.repeats_of(c, schedule)) for c in conditions}
    repeats["grid-full"] = [0]
    clean = [{k: v for k, v in r.items() if k not in ("period_hours", "n_points")} for r in rows]
    summary = score_reliability(clean, expected_objects={r["object_id"]: r["fold"] for r in lock["objects"]},
                                expected_conditions=list(repeats), expected_repeats_by_condition=repeats,
                                baseline_by_condition=baselines_by_condition(conditions))
    return summary, render_reliability_csv(summary)


def grid_table(rows, *, exclude_failed=False):
    """``reliability_publication.recompute_two_d`` without the archived-value check."""
    from lc_pipeline.k3.reliability_scoring import stratified_asteroid_bootstrap
    usable = [r for r in rows if r["status"] in ("ok", "failed") and (r["status"] == "ok" or not exclude_failed)]
    grouped = defaultdict(lambda: defaultdict(list))
    for r in usable:
        if r["condition"].startswith("two_d-"):
            grouped[r["condition"]][r["object_id"]].append(r)
    baseline = {r["object_id"]: r["error_deg"] for r in usable if r["condition"] == "grid-full"}
    all_rows = [r for r in rows if r["condition"].startswith("two_d-") and r["status"] != "ineligible"]
    out = []
    for condition, objects in sorted(grouped.items()):
        objects = dict(sorted(objects.items()))
        means = {oid: float(np.mean([v["error_deg"] for v in vals])) for oid, vals in objects.items()}
        folds = {oid: int(vals[0]["fold"]) for oid, vals in objects.items()}
        counts = [int(v["n_points"]) for vals in objects.values() for v in vals if v.get("n_points") is not None]
        boot = stratified_asteroid_bootstrap([{"object_id": o, "fold": folds[o], "value": m} for o, m in means.items()],
                                             value_key="value", resamples=10_000, seed=20260915)
        block, cap = condition.replace("two_d-block_count-", "").replace("-per_block_cap-", " ").split()
        result = {"condition": condition, "block_count": "all" if block == "None" else int(block),
                  "observation_cap_per_block": "all" if cap == "None" else int(cap),
                  "n_objects": len(means), "mean_error_deg": float(np.mean(list(means.values()))),
                  "ci95_low_deg": boot["bootstrap_lower_95"], "ci95_high_deg": boot["bootstrap_upper_95"],
                  "median_error_deg": float(np.median(list(means.values()))),
                  "retained_observations_min": min(counts) if counts else None,
                  "retained_observations_median": float(np.median(counts)) if counts else None,
                  "retained_observations_max": max(counts) if counts else None,
                  "failed_predictions": sum(r["status"] == "failed" for r in all_rows if r["condition"] == condition),
                  "scheduled_predictions": sum(1 for r in all_rows if r["condition"] == condition)}
        matched = [o for o in means if o in baseline]
        diffs = [{"object_id": o, "fold": folds[o], "value": means[o] - baseline[o]} for o in matched]
        if diffs:
            paired = stratified_asteroid_bootstrap(diffs, value_key="value", resamples=10_000, seed=20260915)
            result.update(n_matched=len(diffs), paired_change_deg=float(np.mean([d["value"] for d in diffs])),
                          paired_ci95_low_deg=paired["bootstrap_lower_95"],
                          paired_ci95_high_deg=paired["bootstrap_upper_95"])
        out.append(result)
    return out


def sampling_baselines(rows, *, exclude_failed=False):
    """``repro/sampling_baselines.py`` on scored rows (same seeds and definitions)."""
    import repro.compute_reference_baselines as base
    rows = [r for r in rows if r.get("status") != "ineligible" and (r["status"] == "ok" or not exclude_failed)]
    out = {"groups": {}}
    for blocks in FIG8_BLOCKS:
        cond = f"two_d-block_count-{blocks}-per_block_cap-None"
        chosen = [r for r in rows if r["condition"] == cond]
        per_repeat = [float(np.mean([r["error_deg"] < 20 for r in chosen if r["repeat"] == k])) for k in (0, 1, 2)]
        out["groups"][blocks] = {"within20_by_repeat": per_repeat, "within20_mean": float(np.mean(per_repeat)),
                                 "mean_error_deg": float(np.mean([r["error_deg"] for r in chosen])),
                                 "n_cells": len(chosen)}
    ids = sorted({r["object_id"] for r in rows if r["condition"] == "grid-full"})
    poles = base.reference_poles()
    standard = base.standard_axes()
    std = {i: base.axial_error_deg(standard, poles[i]) for i in ids}
    paired = {}
    for blocks in FIG8_BLOCKS:
        cond = f"two_d-block_count-{blocks}-per_block_cap-None"
        have = [i for i in ids if any(r["condition"] == cond and r["object_id"] == i for r in rows)]
        delphi = np.array([np.mean([r["error_deg"] for r in rows if r["condition"] == cond and r["object_id"] == i])
                           for i in have])
        diff = np.array([std[i] for i in have]) - delphi
        draws = np.random.default_rng(20260930).integers(0, len(have), size=(10000, len(have)))
        boot = diff[draws].mean(axis=1)
        paired[blocks] = [float(diff.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)),
                          len(have)]
    out["standard_minus_delphi_deg"] = paired
    per_object = np.load(R.FOLLOWUP / "source-op-reliability/repro/baselines_out/reference-baselines-per-object.npz")
    standard_all = dict(zip([str(v) for v in per_object["object_ids"]], per_object["standard_errors_deg"]))
    caps = {}
    for cap in CAPS:
        cond = f"observation_cap-cap-{cap}"
        errors = {}
        for r in rows:
            # The published script keeps only status "ok"; failures are therefore
            # added at 90 deg explicitly in the 90-deg version.
            if r["condition"] == cond and (r.get("status") == "ok" or not exclude_failed):
                errors.setdefault(r["object_id"], []).append(r["error_deg"])
        diff = np.array([standard_all[i] - np.mean(v) for i, v in sorted(errors.items())])
        draws = np.random.default_rng(20260930).integers(0, len(diff), size=(10000, len(diff)))
        boot = diff[draws].mean(axis=1)
        caps[cap] = [int(len(diff)), float(diff.mean()), float(np.percentile(boot, 2.5)),
                     float(np.percentile(boot, 97.5))]
    out["total_cap_standard_minus_delphi_deg"] = caps
    out.update(n_objects=len(ids), standard_mean_error_deg=float(np.mean([std[i] for i in ids])),
               standard_within20=float(np.mean([std[i] < 20 for i in ids])))
    return out


def excluded_summary(rows, lock, schedule):
    """Every condition with failed cells removed (asteroids with no success dropped)."""
    from lc_pipeline.k3.reliability_scoring import stratified_asteroid_bootstrap
    out = {}
    names = [c["condition"] for c in R.sampling_conditions(lock)] + ["grid-full"]
    for condition in names:
        cells = [r for r in rows if r["condition"] == condition and r["status"] != "ineligible"]
        per = defaultdict(list)
        for r in cells:
            if r["status"] == "ok":
                per[r["object_id"]].append(r)
        means = {o: float(np.mean([v["error_deg"] for v in vals])) for o, vals in per.items()}
        within = {o: float(np.mean([v["error_deg"] <= 20 for v in vals])) for o, vals in per.items()}
        value = {"n_eligible_objects": len({r["object_id"] for r in cells}), "n_objects_with_success": len(means),
                 "n_cells": len(cells), "n_failed_cells": sum(r["status"] == "failed" for r in cells)}
        if means:
            boot = stratified_asteroid_bootstrap(
                [{"object_id": o, "fold": per[o][0]["fold"], "v": m} for o, m in means.items()],
                value_key="v", resamples=10_000, seed=20260915)
            value.update(mean_error_deg=float(np.mean(list(means.values()))),
                         median_error_deg=float(np.median(list(means.values()))),
                         within20_fraction=float(np.mean(list(within.values()))),
                         ci95_low_deg=boot["bootstrap_lower_95"], ci95_high_deg=boot["bootstrap_upper_95"])
        out[condition] = value
    return out


def table_value(csv_text, condition, key):
    for row in csv.DictReader(io.StringIO(csv_text)):
        if row["condition"] == condition:
            return float(row[key]) if row[key] not in ("", None) else None
    return None


# ----------------------------------------------------------------------------- withheld

def score_withheld(root: Path, ids, lock, catalog, arms=("classical20", "k3")):
    """Withheld normalized RMS as in ``reliability_study.score`` for the given fits root."""
    from lc_pipeline.k3.reliability_forward import ForwardLightcurve, ForwardObservation, holdout_shape_rmse, load_solution
    from lc_pipeline.k3.reliability_sampling import read_lc, select_holdout
    from lc_pipeline.k3.reliability_scoring import stratified_asteroid_bootstrap
    from lc_pipeline.k3.reliability_study import validate_start
    by_id = {r["object_id"]: r for r in lock["objects"]}
    rows = []
    for oid in ids:
        entry = by_id[oid]
        curve = read_lc(Path(entry["lightcurve"]["path"]), object_id=oid, period_hours=None)
        withheld = select_holdout(curve).withheld
        curves = tuple(ForwardLightcurve(tuple(ForwardObservation(o.time_jd, o.flux, o.sun, o.observer)
                                               for o in s.observations), relative=s.calibrated == 0)
                       for s in withheld.sessions)
        for sampling in R.WITHHELD_SAMPLINGS:
            for arm in arms:
                path = root / "fits" / oid / sampling / arm / "result.json"
                row = {"object_id": oid, "fold": entry["fold"], "sampling": sampling, "arm": arm,
                       "withheld_normalized_curve_rms": None, "status": "missing"}
                if path.exists():
                    fitted = json.loads(path.read_text())["payload"]
                    row["status"] = fitted["status"]
                    row["period_hours"] = None
                    if fitted["selected_fit"] is not None:
                        selected = path.parent / f"start-{fitted['selected_fit']['start_index']:03d}"
                        validate_start(selected)
                        solution = load_solution(selected / "solution-parameters.txt",
                                                 selected / "solution-areas.txt")
                        row["period_hours"] = solution.period_hours if hasattr(solution, "period_hours") else None
                        try:
                            row["withheld_normalized_curve_rms"] = holdout_shape_rmse(solution, curves)
                        except ValueError as exc:
                            row.update(status="failed", failure_code=f"forward_scoring: {exc}")
                        row["training_rms"] = fitted["selected_fit"]["final_relative_rms"]
                        lon, lat = np.radians([solution.longitude_deg, solution.latitude_deg])
                        d = np.asarray([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
                        ref = np.asarray(catalog[oid].solution_vectors, float)
                        ref /= np.linalg.norm(ref, axis=1, keepdims=True)
                        row["catalog_axis_disagreement_deg"] = float(np.degrees(np.arccos(np.clip(
                            np.abs(ref @ d).max(), 0, 1))))
                rows.append(row)
    summary = {}
    for sampling in R.WITHHELD_SAMPLINGS:
        base = {r["object_id"]: r for r in rows if r["sampling"] == sampling and r["arm"] == "classical20"
                and r["withheld_normalized_curve_rms"] is not None}
        for arm in arms:
            valid = [r for r in rows if r["sampling"] == sampling and r["arm"] == arm
                     and r["withheld_normalized_curve_rms"] is not None]
            value = {"n_scheduled": len(ids), "n_valid": len(valid)}
            if valid:
                value["withheld_rms"] = stratified_asteroid_bootstrap(valid, value_key="withheld_normalized_curve_rms")
            if arm != "classical20":
                matched = [{"object_id": r["object_id"], "fold": r["fold"],
                            "d": r["withheld_normalized_curve_rms"] - base[r["object_id"]]["withheld_normalized_curve_rms"]}
                           for r in valid if r["object_id"] in base]
                value["n_completed_pairs"] = len(matched)
                if matched:
                    value["candidate_minus_classical_rms"] = stratified_asteroid_bootstrap(matched, value_key="d")
                    # The same pairs, for a like-for-like classical mean.
                    pair_ids = {m["object_id"] for m in matched}
                    value["classical_rms_on_pairs"] = float(np.mean(
                        [base[o]["withheld_normalized_curve_rms"] for o in pair_ids]))
                    value["guided_rms_on_pairs"] = float(np.mean(
                        [r["withheld_normalized_curve_rms"] for r in valid if r["object_id"] in pair_ids]))
            summary[f"{sampling}-{arm}"] = value
    return rows, summary


# ----------------------------------------------------------------------------- commands

def published_records():
    rows = json.loads((R.RUN / "scored-rows.json").read_text())["payload"]
    return {(r["object_id"], r["condition"], r["repeat"]): {**r, "metadata": r.get("metadata", {})}
            for r in rows if r.get("status") != "ineligible" and r["condition"] != "grid-full"}


def command_validate_scoring(args) -> int:
    lock, schedule = R.published_lock(), R.published_schedule()
    catalog = catalog_references(lock)
    rows = build_rows(published_records(), lock, schedule, catalog)
    summary, table = summarize(rows, lock, schedule)
    published = {r["condition"]: r for r in csv.DictReader((R.RUN / "sampling-table.csv").open())}
    mine = {r["condition"]: r for r in csv.DictReader(io.StringIO(table))}
    table_mismatch = [c for c in mine if any(mine[c][k] != published[c][k] for k in mine[c])]
    grid = {r["condition"]: r for r in grid_table(rows)}
    pub_grid = {r["condition"]: r for r in csv.DictReader(
        (R.FOLLOWUP / "data/reliability-publication-20260916-final/k3_data_degradation.csv").open())}
    grid_worst = max(abs(grid[c][k] - float(pub_grid[c][k])) for c in pub_grid
                     for k in ("mean_error_deg", "ci95_low_deg", "ci95_high_deg", "paired_change_deg",
                               "paired_ci95_low_deg", "paired_ci95_high_deg"))
    baselines = sampling_baselines(rows)
    pub_base = json.loads((R.FOLLOWUP / "data/interpretation-20260930/sampling-baselines.json").read_text())
    base_diff = max(
        [abs(baselines["groups"][b]["within20_mean"] - pub_base["groups"][str(b)]["within20_mean"]) for b in FIG8_BLOCKS] +
        [abs(baselines["standard_minus_delphi_deg"][b][i] - pub_base["standard_minus_delphi_deg"][str(b)][i])
         for b in FIG8_BLOCKS for i in range(3)] +
        [abs(baselines["total_cap_standard_minus_delphi_deg"][c][i] - pub_base["total_cap_standard_minus_delphi_deg"][c][i])
         for c in CAPS for i in range(4)])
    _, withheld = score_withheld(R.RUN, schedule["holdout_ids"], lock, catalog)
    report = json.loads((R.RUN / "report.json").read_text())["payload"]["withheld"]
    withheld_diff = max(
        abs(withheld[k]["withheld_rms"][s] - report[k]["withheld_rms"][s])
        for k in ("full-classical20", "full-k3", "thin10-classical20", "thin10-k3")
        for s in ("estimate", "bootstrap_lower_95", "bootstrap_upper_95"))
    withheld_diff = max(withheld_diff, *(
        abs(withheld[k]["candidate_minus_classical_rms"][s] - report[k]["candidate_minus_classical_rms"][s])
        for k in ("full-k3", "thin10-k3") for s in ("estimate", "bootstrap_lower_95", "bootstrap_upper_95")))
    result = {"sampling_table_rows_compared": len(mine), "sampling_table_mismatched_conditions": table_mismatch,
              "data_degradation_max_abs_difference": grid_worst,
              "sampling_baselines_max_abs_difference": base_diff,
              "withheld_table_max_abs_difference": withheld_diff,
              "published_withheld_recomputed": {k: v.get("withheld_rms", {}).get("estimate") for k, v in withheld.items()}}
    with (R.OUT / "validation-scoring.json").open("x") as stream:
        stream.write(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))
    return 0


def _period_stats(lock, schedule):
    periods = R.read_jsonl(R.OUT / "search" / "periods-main.jsonl")
    reference = damit_periods()
    by_condition = defaultdict(list)
    for r in periods:
        by_condition[r["condition"]].append(r)
    full_own = json.loads(R.OWN_FULL.read_text())["periods_hours"]
    out = {}
    for condition, recs in sorted(by_condition.items()):
        classes = defaultdict(int)
        regimes = defaultdict(int)
        statuses = defaultdict(int)
        for r in recs:
            p = full_own[r["object_id"]] if condition == "full" else r["period_hours"]
            classes[period_class(p, reference[r["object_id"]])] += 1
            regimes[str(r.get("input_regime"))] += 1
            statuses[str(r["status"])] += 1
        searched = [r["seconds"] for r in recs if r.get("reused_from") is None]
        out[condition] = {"n_cells": len(recs), "n_with_period": sum(r["period_hours"] is not None for r in recs),
                          "success_rate": float(np.mean([r["period_hours"] is not None for r in recs])),
                          "classes": dict(classes), "input_regimes": dict(regimes), "statuses": dict(statuses),
                          "n_searched": len(searched), "search_seconds_mean": float(np.mean(searched)) if searched else 0.0,
                          "search_seconds_max": float(np.max(searched)) if searched else 0.0}
    # The full-condition searches check that this run reproduces own-periods-170.json.
    full = {r["object_id"]: r["period_hours"] for r in by_condition["full"]}
    diffs = [abs(full[o] - full_own[o]) / full_own[o] for o in full if full[o] is not None and full_own[o] is not None]
    out["_full_reproduces_own_periods_170"] = {
        "n": len(full), "n_identical": sum(full[o] == full_own[o] for o in full),
        "max_relative_difference": float(max(diffs)) if diffs else None,
        "n_missing": sum(full[o] is None for o in full)}
    return out


def command_score(args) -> int:
    lock, schedule = R.published_lock(), R.published_schedule()
    catalog = catalog_references(lock)
    records = {(r["object_id"], r["condition"], r["repeat"]): r
               for r in R.read_jsonl(R.OUT / "predictions" / "own.jsonl")}
    rows = build_rows(records, lock, schedule, catalog)
    summary, table = summarize(rows, lock, schedule)
    out = R.OUT / "score"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "sampling-table-own.csv").open("x") as s:
        s.write(table)
    grid90 = grid_table(rows)
    grid_ex = grid_table(rows, exclude_failed=True)
    for name, cells in (("k3_data_degradation-own-failed90.csv", grid90),
                        ("k3_data_degradation-own-failed-excluded.csv", grid_ex)):
        with (out / name).open("x", newline="") as s:
            fields = list(dict.fromkeys(k for c in cells for k in c))
            writer = csv.DictWriter(s, fieldnames=fields)
            writer.writeheader()
            writer.writerows(cells)
    base90 = sampling_baselines(rows)
    base_ex = sampling_baselines(rows, exclude_failed=True)
    excluded = excluded_summary(rows, lock, schedule)
    with (out / "sampling-table-own-failed-excluded.csv").open("x", newline="") as s:
        writer = csv.writer(s)
        writer.writerow(["condition", "n_eligible_objects", "n_objects_with_success", "n_cells", "n_failed_cells",
                         "mean_error_deg", "mean_error_bootstrap_lower_95", "mean_error_bootstrap_upper_95",
                         "median_object_repeat_mean_error_deg", "mean_repeat_within_20_fraction"])
        for condition, v in excluded.items():
            writer.writerow([condition, v["n_eligible_objects"], v["n_objects_with_success"], v["n_cells"],
                             v["n_failed_cells"], v.get("mean_error_deg"), v.get("ci95_low_deg"),
                             v.get("ci95_high_deg"), v.get("median_error_deg"), v.get("within20_fraction")])
    with (out / "scored-rows-own.json").open("x") as s:
        json.dump(rows, s)
    withheld_rows, withheld = ({}, {})
    if (R.withheld_root("own") / "fits.complete").exists():
        withheld_rows, withheld = score_withheld(R.withheld_root("own"), schedule["holdout_ids"], lock, catalog)
    result = {"failed_at_90": {"sampling_summary": summary["models"]["k3"],
                               "matched": summary["matched_comparisons"]["k3"],
                               "data_degradation": grid90, "sampling_baselines": base90},
              "failed_excluded": {"conditions": excluded, "data_degradation": grid_ex,
                                  "sampling_baselines": base_ex},
              "period_search": _period_stats(lock, schedule),
              "withheld": withheld, "withheld_rows": withheld_rows}
    with (out / "reliability-own-period.json").open("x") as s:
        json.dump(result, s, indent=1)
    print("written", out)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate-scoring")
    sub.add_parser("score")
    args = parser.parse_args()
    return {"validate-scoring": command_validate_scoring, "score": command_score}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
