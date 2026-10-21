"""Scores of the lower-quality census, recomputed with the published code.

    python -m repro.own_period.lowq_score --predictions Y.jsonl --periods OP/lowq/own-periods-lowq.json --output OP/lowq/scores-own
    python -m repro.own_period.lowq_score --predictions published --output OP/lowq/scores-published-check

Every number the manuscript prints from the census is recomputed by calling the
published functions on the new object-level errors:

* ``run_k3_lowq_census.analyze`` (primary, quality strata, train-only atlas);
* ``analyze_k3_lowq_lineage.analyze`` (shared-source grouping, Gaia DR3, ASAS-SN);
* the lower-quality cohort blocks of ``compute_reference_baselines.main``
  (six standard poles, standard minus DeLPHI with the same seeds);
* the ``lower_quality_direction`` block of ``mirror_symmetry.main`` (latitude
  bins, Spearman, longitude Kruskal-Wallis test).

Two variants are written: asteroids without a period excluded (the census
``analyze`` treats them as rejected inputs) and at 90 deg (same set as the
published 9,783).  With ``--predictions published`` the published prediction
file is scored, which must reproduce the published outputs exactly.

Reference periods (the DAMIT model-table period of the census index) are read
only here, after the predictions are written, for the alias classes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import kruskal, spearmanr

from repro import analyze_k3_lowq_lineage as lineage
from repro import compute_reference_baselines as base
from repro import mirror_symmetry as ms
from repro import run_k3_lowq_census as census

from .lowq_search import CENSUS, FOLLOWUP, INDEX, SNAPSHOT, sha256
from .score import FAILED_DEG, SEED, paired, period_class, summary

REFERENCE = CENSUS / "reference-cohort.json"
ATLAS = CENSUS / "fold0-train-only-atlas.json"
PUBLISHED_PREDICTIONS = CENSUS / "predictions-fold0-five-seed"
PUBLISHED_ANALYSIS = FOLLOWUP / "data/publication-revision-20260917-r2/lowq-analysis.json"
SPLITS = (
    FOLLOWUP / "inputs/training-run/repro/data/damit-20250610T000301Z/publication-splits-v2.3.json"
)


def write_once(path: Path, value: object) -> None:
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n")


def prediction_directory(predictions: Path, destination: Path) -> Path:
    """Wrap a JSONL file in the census prediction-directory layout."""
    destination.mkdir(parents=True)
    target = destination / "predictions.jsonl"
    target.write_bytes(predictions.read_bytes())
    manifest = {
        "schema": census.PREDICTION_SCHEMA,
        "predictions_path": target.name,
        "predictions_sha256": sha256(target),
        "source_predictions": str(predictions),
    }
    write_once(destination / "manifest.json", manifest)
    return destination


def failed_at_90(excluded: dict, predictions: Path) -> dict:
    """The census analysis with failures (no period) added at 90 deg, in census order."""
    reference = json.loads(REFERENCE.read_text())
    reference_by_id = {row["object_id"]: row for row in reference["objects"]}
    atlas_axes = np.asarray(json.loads(ATLAS.read_text())["axes"], dtype=np.float64)
    from lc_pipeline.physics.axial import axial_angular_error_deg

    analyzed = {row["object_id"]: row for row in excluded["objects"]}
    rows = []
    with predictions.open() as handle:
        for line in handle:
            item = json.loads(line)
            object_id = item["object_id"]
            if object_id in analyzed:
                rows.append({**analyzed[object_id], "own_period_failed": False})
            elif item["status"] == "no_period":
                refs = np.asarray(
                    [r["axis"] for r in reference_by_id[object_id]["references"]], dtype=np.float64
                )
                atlas_error = float(
                    np.min(axial_angular_error_deg(atlas_axes[:, None, :], refs[None, :, :]))
                )
                rows.append({
                    "object_id": object_id,
                    "quality_flag": int(reference_by_id[object_id]["quality_flag"]),
                    "oracle_at_3_error_deg": FAILED_DEG,
                    "atlas_oracle_at_3_error_deg": atlas_error,
                    "wall_seconds": 0.0,
                    "epoch_count": 0,
                    "observation_count": 0,
                    "own_period_failed": True,
                })
    errors = np.asarray([r["oracle_at_3_error_deg"] for r in rows])
    atlas_errors = np.asarray([r["atlas_oracle_at_3_error_deg"] for r in rows])
    improvement = atlas_errors - errors
    low, high = census._bootstrap(improvement)
    strata = {}
    for quality in (1, 2):
        chosen = np.asarray([r["oracle_at_3_error_deg"] for r in rows if r["quality_flag"] == quality])
        if chosen.size:
            strata[str(quality)] = census._summary(chosen)
    result = {key: value for key, value in excluded.items() if key not in ("objects",)}
    result.update({
        "variant": "own_period_failed_at_90",
        "analyzed_denominator": len(rows),
        "own_period_failed_count": int(sum(r["own_period_failed"] for r in rows)),
        "primary": census._summary(errors),
        "quality_strata": strata,
        "train_only_atlas": census._summary(atlas_errors),
        "paired_atlas_minus_k3": {
            "mean_improvement_deg": float(np.mean(improvement)),
            "bootstrap_ci95_deg": [low, high],
        },
        "objects": rows,
    })
    return result


def baselines_block(analysis_rows: list[dict], lineage_csv: Path | None) -> dict:
    """The lower-quality cohort blocks of compute_reference_baselines.main (no random axes)."""
    table = base.damit_table_poles()
    poles = base.reference_poles()
    axes = base.standard_axes()
    out = {}
    names = [row["object_id"] for row in analysis_rows]
    delphi = np.array([float(row["oracle_at_3_error_deg"]) for row in analysis_rows])
    references, kept = [], []
    for name, value in zip(names, delphi, strict=True):
        found = base.cohort_reference(name, table, poles)
        if found is None:
            continue
        references.append(found)
        kept.append(value)
    kept = np.array(kept)
    standard = np.array([base.axial_error_deg(axes, item) for item in references])
    out["lowq"] = {
        "object_count": len(references),
        "unmatched_count": len(names) - len(references),
        "delphi": base.summarize(kept),
        "standard": base.summarize(standard),
        "standard_minus_delphi": base.paired_difference(standard, kept, base.BOOTSTRAP_SEED + 2),
    }
    if lineage_csv is None:
        return out
    strata: dict[str, list[tuple[str, float]]] = {"gaia": [], "asas": [], "shared": [], "unshared": []}
    with lineage_csv.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            keys = row.get("source_keys", "").lower()
            value = (row["object_id"], float(row["oracle_at_3_error_deg"]))
            if base.GAIA_SOURCE_KEY in keys:
                strata["gaia"].append(value)
            if base.ASAS_SOURCE_KEY in keys:
                strata["asas"].append(value)
            if row.get("overlapping_training_source_keys", "").strip():
                strata["shared"].append(value)
            else:
                strata["unshared"].append(value)
    for label, rows in strata.items():
        refs, measured = [], []
        for name, value in rows:
            found = base.cohort_reference(name, table, poles)
            if found is None:
                continue
            refs.append(found)
            measured.append(value)
        measured = np.array(measured)
        standard = np.array([base.axial_error_deg(axes, item) for item in refs])
        out[f"lowq_{label}"] = {
            "object_count": len(refs),
            "unmatched_count": len(rows) - len(refs),
            "delphi": base.summarize(measured),
            "standard": base.summarize(standard),
            "standard_minus_delphi": base.paired_difference(
                standard, measured, base.BOOTSTRAP_SEED + 6
            ),
        }
    return out


def direction_block(analysis_rows: list[dict]) -> dict:
    """The lower_quality_direction block of mirror_symmetry.main."""
    from collections import defaultdict

    solutions = defaultdict(list)
    with (ms.DUMP / "tables/asteroid_models.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                solutions[int(row["asteroid_id"])].append((float(row["lambda"]), float(row["beta"])))
            except ValueError:
                continue
    standard = base.standard_axes()
    low_rows = []
    for entry in analysis_rows:
        listed = solutions[int(entry["object_id"].split(":")[1])]
        references = np.array([base.unit_from_degrees(lam, b) for lam, b in listed])
        low_rows.append((
            listed[0][0] % 360.0,
            abs(listed[0][1]),
            float(entry["oracle_at_3_error_deg"]),
            base.axial_error_deg(standard, references),
        ))
    low = np.array(low_rows)
    return {
        "n": int(len(low)),
        "spearman_abs_latitude": float(spearmanr(low[:, 1], low[:, 2])[0]),
        "spearman_abs_latitude_p": float(spearmanr(low[:, 1], low[:, 2])[1]),
        "latitude_bins": ms.binned(low[:, 1], low[:, 2], [0, 15, 30, 45, 60, 75, 90]),
        "standard_latitude_bins": ms.binned(low[:, 1], low[:, 3], [0, 15, 30, 45, 60, 75, 90]),
        "longitude_kruskal_p": float(
            kruskal(
                *[low[(low[:, 0] >= lo) & (low[:, 0] < lo + 60), 2] for lo in range(0, 360, 60)]
            ).pvalue
        ),
        "longitude_bins": ms.binned(low[:, 0], low[:, 2], list(range(0, 361, 60))),
        # Per-asteroid series for redrawing Figure 11 (absolute latitude of the
        # first listed model, DeLPHI error, six-standard-pole error).
        "series": {
            "object_id": [row["object_id"] for row in analysis_rows],
            "abs_latitude_deg": low[:, 1].tolist(),
            "delphi_error_deg": low[:, 2].tolist(),
            "standard_error_deg": low[:, 3].tolist(),
        },
    }


def variant_outputs(analysis: dict, name: str, output: Path) -> dict:
    analysis_path = output / f"census-analysis-{name}.json"
    if not analysis_path.exists():
        write_once(analysis_path, analysis)
    lineage_dir = output / f"lineage-{name}"
    try:
        lineage.analyze(SNAPSHOT, SPLITS, analysis_path, lineage_dir)
    except ValueError as error:  # a small subset may lack one lineage stratum
        lin, lineage_csv = {"error": str(error)}, None
    else:
        lin, lineage_csv = json.loads((lineage_dir / "analysis.json").read_text()), lineage_dir / "objects.csv"
    baselines = baselines_block(analysis["objects"], lineage_csv)
    direction = direction_block(analysis["objects"])
    write_once(output / f"baselines-{name}.json", baselines)
    write_once(output / f"direction-{name}.json", direction)
    return {
        "census_primary": analysis["primary"],
        "census_train_only_atlas": analysis["train_only_atlas"],
        "census_paired_atlas_minus_k3": analysis["paired_atlas_minus_k3"],
        "lineage_error": lin.get("error"),
        "lineage_counts": lin.get("lineage_counts"),
        "lineage_disjoint": lin.get("structured_reference_disjoint"),
        "lineage_overlap": lin.get("structured_reference_overlap"),
        "lineage_source_groups": lin.get("structured_reference_disjoint_source_groups"),
        "baselines": baselines,
        "direction": {k: v for k, v in direction.items() if k != "series"},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="own JSONL or 'published'")
    parser.add_argument("--periods", type=Path, default=None, help="own-period file (alias classes)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True)
    if args.predictions == "published":
        directory = PUBLISHED_PREDICTIONS
        jsonl = directory / "predictions.jsonl"
    else:
        jsonl = Path(args.predictions)
        directory = prediction_directory(jsonl, args.output / "prediction-directory")
    excluded_path = args.output / "census-analysis-excluded.json"
    census.analyze(REFERENCE, directory, ATLAS, SPLITS, excluded_path, fold=0)
    excluded = json.loads(excluded_path.read_text())
    report: dict = {
        "schema": "delphi.own-period-lowq-scores.v1",
        "predictions": str(jsonl),
        "predictions_sha256": sha256(jsonl),
        "variants": {"failed_excluded": variant_outputs(excluded, "excluded", args.output)},
    }
    statuses: dict[str, int] = {}
    with jsonl.open() as handle:
        for line in handle:
            item = json.loads(line)
            statuses[item["status"]] = statuses.get(item["status"], 0) + 1
    report["prediction_status_counts"] = statuses
    if statuses.get("no_period"):
        at90 = failed_at_90(excluded, jsonl)
        report["variants"]["failed_at_90"] = variant_outputs(at90, "at90", args.output)
    else:
        at90 = excluded

    # DAMIT-period (published) DeLPHI against this run, paired over asteroids.
    published = {
        row["object_id"]: float(row["oracle_at_3_error_deg"])
        for row in json.loads(PUBLISHED_ANALYSIS.read_text())["objects"]
    }
    own = {row["object_id"]: row for row in at90["objects"]}
    ids = [i for i in published if i in own]
    pub = np.array([published[i] for i in ids])
    own_at90 = np.array([own[i]["oracle_at_3_error_deg"] for i in ids])
    ok = np.array([not own[i].get("own_period_failed", False) for i in ids])
    report["published_minus_own"] = {
        "failed_at_90": paired(pub, own_at90, SEED),
        "failed_excluded": paired(pub[ok], own_at90[ok], SEED + 1),
        "published_on_own_successes": summary(pub[ok]),
        "max_abs_difference_deg": float(np.max(np.abs(pub - own_at90))),
    }

    if args.periods is not None:
        # Reference periods are read only from here on.
        periods = json.loads(args.periods.read_text())
        damit = {r["object_id"]: float(r["known_period_hours"]) for r in json.loads(INDEX.read_text())["objects"]}
        classes = {i: period_class(periods["periods_hours"].get(i), damit[i]) for i in ids}
        counts = {}
        for c in ("P", "P/2", "2P", "other", "failed"):
            mask = np.array([classes[i] == c for i in ids])
            counts[c] = {"n": int(mask.sum())}
            if mask.any():
                counts[c]["own_mean_deg"] = float(own_at90[mask].mean())
                counts[c]["own_median_deg"] = float(np.median(own_at90[mask]))
                counts[c]["published_mean_deg"] = float(pub[mask].mean())
                counts[c]["published_median_deg"] = float(np.median(pub[mask]))
        regimes: dict[str, dict[str, int]] = {}
        seconds = []
        for i in ids:
            detail = periods["details"][i]
            key = f"{detail['status']}|{detail.get('input_regime')}"
            regimes.setdefault(key, {"n": 0})["n"] += 1
            seconds.append(float(detail["seconds"]))
        seconds = np.array(seconds)
        report["period_classes"] = counts
        report["search_status_regime"] = regimes
        report["search_seconds"] = {
            "sum": float(seconds.sum()),
            "mean": float(seconds.mean()),
            "median": float(np.median(seconds)),
            "max": float(seconds.max()),
            "n_over_180": int((seconds > 180).sum()),
        }
        with (args.output / "objects.csv").open("x", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["object_id", "own_period_hours", "damit_period_hours", "period_class",
                             "own_error_deg_or_90", "own_failed", "published_error_deg"])
            for i, value, failed, p in zip(ids, own_at90, ~ok, pub, strict=True):
                writer.writerow([i, periods["periods_hours"].get(i), damit[i], classes[i],
                                 f"{value:.10g}", int(failed), f"{p:.10g}"])
    write_once(args.output / "summary.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "variants"}, indent=1)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
