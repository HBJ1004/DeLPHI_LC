#!/usr/bin/env python3
"""Build the hash-bound public export for the 2026-09-17 analyses."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

SCHEMA = "delphi.k3-publication-revision-export.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _write(path: Path, value: object) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _require_schema(value: dict[str, object], expected: str, label: str) -> None:
    if value.get("schema") != expected:
        raise ValueError(f"{label} has the wrong schema")


def _timing_descriptive(csv_path: Path) -> dict[str, object]:
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 5100:
        raise ValueError("timing table does not contain 5100 cases")

    def selected(arm: str, mode: str) -> list[dict[str, str]]:
        return [row for row in rows if row["arm"] == arm and row["timing_mode"] == mode]

    guided = selected("guided20", "cold")
    classical_cold = selected("classical20", "cold")
    classical_warm = selected("classical20", "warm")
    starts = [int(row["starts_requested"]) for row in guided]
    inference = [float(row["inference_and_optional_load_seconds"]) for row in guided]
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in guided + classical_cold:
        grouped.setdefault((row["object_id"], row["arm"]), []).append(row)
    representative = {}
    repeat_stable = True
    for key, values in grouped.items():
        axes = [
            tuple(float(row[f"selected_axis_{axis}"]) for axis in "xyz") for row in values
        ]
        repeat_stable = repeat_stable and all(axis == axes[0] for axis in axes[1:])
        representative[key] = axes[0]
    disagreements = []
    for object_id in {row["object_id"] for row in guided}:
        left = representative[(object_id, "guided20")]
        right = representative[(object_id, "classical20")]
        dot = sum(a * b for a, b in zip(left, right, strict=True))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        disagreements.append(math.degrees(math.acos(min(1.0, abs(dot) / left_norm / right_norm))))
    return {
        "guided20_cold_mean_starts": sum(starts) / len(starts),
        "guided20_cold_min_starts": min(starts),
        "guided20_cold_max_starts": max(starts),
        "guided20_cold_mean_inference_and_optional_load_seconds": (
            sum(inference) / len(inference)
        ),
        "classical20_cold_mean_seconds": sum(
            float(row["wall_seconds"]) for row in classical_cold
        )
        / len(classical_cold),
        "classical20_warm_mean_seconds": sum(
            float(row["wall_seconds"]) for row in classical_warm
        )
        / len(classical_warm),
        "selected_poles_identical_across_three_repeats": repeat_stable,
        "objects_with_guided_classical_selected_axis_disagreement_over_20_deg": sum(
            value > 20.0 for value in disagreements
        ),
    }


def export(
    *,
    lowq_analysis: Path,
    lowq_lineage: Path,
    lowq_lineage_objects: Path,
    ensemble_ablation: Path,
    error_sensitivity: Path,
    ztf_predictions: Path,
    alcdef_predictions: Path,
    ztf_prepared_manifest: Path,
    alcdef_prepared_manifest: Path,
    timing_directory: Path,
    sampling_caps: Path,
    broad_grid_figure: Path,
    reliability_study: Path,
    random_atlas_rows: Path,
    reliability_sampling_table: Path,
    reliability_numbers: Path,
    output: Path,
) -> dict[str, object]:
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    lowq = _read(lowq_analysis)
    lineage = _read(lowq_lineage)
    ablation = _read(ensemble_ablation)
    sensitivity = _read(error_sensitivity)
    ztf_prepared = _read(ztf_prepared_manifest)
    alcdef_prepared = _read(alcdef_prepared_manifest)
    timing = _read(timing_directory / "manifest.json")
    reliability = _read(reliability_study)
    atlas_rows = _read(random_atlas_rows)
    reliability_publication = _read(reliability_numbers)
    _require_schema(lowq, "delphi.k3-lowq-damit-census-analysis.v1", "low-Q analysis")
    _require_schema(lineage, "delphi.k3-lowq-damit-lineage-analysis.v1", "lineage analysis")
    _require_schema(
        ablation, "delphi.k3-lowq-ensemble-ablation-analysis.v1", "ensemble ablation"
    )
    _require_schema(
        sensitivity, "delphi.k3-error-channel-sensitivity.v1", "error sensitivity"
    )
    _require_schema(
        timing, "delphi.k3-pole-grid-timing-public-export.v1", "timing export"
    )
    if lowq.get("analyzed_denominator") != 9783 or lineage.get("lineage_counts", {}).get(
        "source_disjoint"
    ) != 7420:
        raise ValueError("low-Q evidence has an unexpected denominator")

    study_payload = reliability.get("payload")
    atlas_payload = atlas_rows.get("payload")
    if not isinstance(study_payload, dict) or not isinstance(atlas_payload, list):
        raise ValueError("reliability study or atlas rows have an invalid seal")
    grid_ids = study_payload.get("grid_ids")
    if not isinstance(grid_ids, list) or len(grid_ids) != 80:
        raise ValueError("reliability grid cohort has an unexpected denominator")
    atlas_by_id = {str(row["object_id"]): float(row["atlas_error_deg"]) for row in atlas_payload}
    grid_atlas_errors = [atlas_by_id[str(object_id)] for object_id in grid_ids]
    with reliability_sampling_table.open(encoding="utf-8", newline="") as handle:
        sampling_rows = list(csv.DictReader(handle))
    capped_means = {
        int(str(row["condition"]).rsplit("-", 1)[-1]): float(row["mean_error_deg"])
        for row in sampling_rows
        if row.get("model_id") == "k3"
        and str(row.get("condition", "")).startswith("observation_cap-cap-")
        and str(row["condition"]).rsplit("-", 1)[-1] not in {"None", "all"}
    }
    one_block = next(
        row
        for row in reliability_publication["two_d_cells"]
        if row["block_count"] == 1 and row["observation_cap_per_block"] == "all"
    )

    sources = {
        "lowq-analysis.json": lowq_analysis,
        "lowq-lineage-analysis.json": lowq_lineage,
        "lowq-lineage-objects.csv": lowq_lineage_objects,
        "ensemble-ablation-analysis.json": ensemble_ablation,
        "error-channel-sensitivity.json": error_sensitivity,
        "error-channel-ztf-predictions.json": ztf_predictions,
        "error-channel-alcdef-predictions.json": alcdef_predictions,
        "error-channel-ztf-prepared-manifest.json": ztf_prepared_manifest,
        "error-channel-alcdef-prepared-manifest.json": alcdef_prepared_manifest,
        "timing-cases.csv": timing_directory / "timing-cases.csv",
        "timing-source-manifest.json": timing_directory / "manifest.json",
        "k3_sampling-caps.pdf": sampling_caps,
        "broad-grid-timing-rms.pdf": broad_grid_figure,
    }
    for name, path in sources.items():
        if not path.is_file():
            raise ValueError(f"missing revision evidence: {name}")

    output.mkdir(parents=True)
    for name, source in sources.items():
        shutil.copyfile(source, output / name)

    summary = {
        "schema": "delphi.k3-publication-revision-summary.v1",
        "scope": (
            "post hoc frozen-model analyses; no model selection, retraining, or new primary endpoint"
        ),
        "low_quality_damit": {
            "selected_denominator": lowq["selected_denominator"],
            "analyzed_denominator": lowq["analyzed_denominator"],
            "input_rejected_count": lowq["input_rejected_count"],
            "k3": lowq["primary"],
            "train_only_atlas": lowq["train_only_atlas"],
            "paired_atlas_minus_k3": lowq["paired_atlas_minus_k3"],
            "timing": lowq["timing"],
            "reference_quality_warning": lowq["reference_quality_warning"],
            "source_lineage": {
                "definition": lineage["lineage_definition"],
                "counts": lineage["lineage_counts"],
                "training_identity_count": lineage["training_identity_count"],
                "training_source_count": lineage["training_source_count"],
                "source_disjoint": lineage["source_disjoint"],
                "source_overlap": lineage["source_overlap"],
            },
            "ensemble_ablation": {
                "scope": ablation["study_scope"],
                "selection_warning": ablation["selection_warning"],
                "object_count": ablation["object_count"],
                "family_summary": ablation["family_summary"],
            },
        },
        "sparse_input_atlas_reference": {
            "grid_cohort_n": len(grid_ids),
            "grid_cohort_atlas_mean_deg": sum(grid_atlas_errors) / len(grid_atlas_errors),
            "one_block_k3_mean_deg": one_block["mean_error_deg"],
            "all_170_atlas_mean_deg": sum(atlas_by_id.values()) / len(atlas_by_id),
            "total_observation_cap_k3_means_deg": {
                str(cap): capped_means[cap] for cap in sorted(capped_means) if cap <= 200
            },
            "source_hashes": {
                "reliability_study": _sha256(reliability_study),
                "random_atlas_rows": _sha256(random_atlas_rows),
                "sampling_table": _sha256(reliability_sampling_table),
                "reliability_numbers": _sha256(reliability_numbers),
            },
        },
        "error_channel_sensitivity": {
            "intervention": sensitivity["intervention"],
            "ztf": sensitivity["ztf"],
            "alcdef_gaia": {
                key: value
                for key, value in sensitivity["alcdef_gaia"].items()
                if key not in {"objects", "without_error_channel_analysis"}
            },
            "alcdef_gaia_without_error_channel_analysis": {
                key: value
                for key, value in sensitivity["alcdef_gaia"][
                    "without_error_channel_analysis"
                ].items()
                if key != "objects"
            },
            "ztf_ready_objects_with_nonnull_errors": sum(
                int(row.get("removed_nonnull_measured_error_count", 0) > 0)
                for row in ztf_prepared["objects"]
            ),
            "alcdef_objects_with_nonnull_errors": sum(
                int(row.get("removed_nonnull_measured_error_count", 0) > 0)
                for row in alcdef_prepared["objects"]
            ),
        },
        "broad_grid_timing": {
            "host": timing["host"],
            "primary_result": timing["primary_result"],
            "thresholds": timing["thresholds"],
            "descriptive": _timing_descriptive(timing_directory / "timing-cases.csv"),
            "case_count": timing["case_count"],
            "interval_scope": "object resampling on the recorded host",
        },
    }
    _write(output / "revision-summary.json", summary)
    files = {
        path.name: _sha256(path)
        for path in sorted(output.iterdir())
        if path.is_file()
    }
    manifest = {
        "schema": SCHEMA,
        "date": "2026-09-17",
        "files": files,
        "source_sha256": {name: _sha256(path) for name, path in sources.items()},
    }
    _write(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "lowq-analysis",
        "lowq-lineage",
        "lowq-lineage-objects",
        "ensemble-ablation",
        "error-sensitivity",
        "ztf-predictions",
        "alcdef-predictions",
        "ztf-prepared-manifest",
        "alcdef-prepared-manifest",
        "timing-directory",
        "sampling-caps",
        "broad-grid-figure",
        "reliability-study",
        "random-atlas-rows",
        "reliability-sampling-table",
        "reliability-numbers",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest = export(
            lowq_analysis=args.lowq_analysis,
            lowq_lineage=args.lowq_lineage,
            lowq_lineage_objects=args.lowq_lineage_objects,
            ensemble_ablation=args.ensemble_ablation,
            error_sensitivity=args.error_sensitivity,
            ztf_predictions=args.ztf_predictions,
            alcdef_predictions=args.alcdef_predictions,
            ztf_prepared_manifest=args.ztf_prepared_manifest,
            alcdef_prepared_manifest=args.alcdef_prepared_manifest,
            timing_directory=args.timing_directory,
            sampling_caps=args.sampling_caps,
            broad_grid_figure=args.broad_grid_figure,
            reliability_study=args.reliability_study,
            random_atlas_rows=args.random_atlas_rows,
            reliability_sampling_table=args.reliability_sampling_table,
            reliability_numbers=args.reliability_numbers,
            output=args.output,
        )
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({"file_count": len(manifest["files"]), "passed": True}))


if __name__ == "__main__":
    main()
