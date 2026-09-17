#!/usr/bin/env python3
"""Stream the complete usable low-Q DAMIT census through frozen K3 models."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from lc_pipeline.k3.bundle import _model_inputs
from lc_pipeline.k3.evaluation import ensemble_score_grids, modes_from_score_grid
from lc_pipeline.k3.inference import refine_ensemble_axes, score_axial_grid
from lc_pipeline.k3.ztf_prediction import _load_models
from lc_pipeline.physics.axial import axial_angular_error_deg
from lc_pipeline.v2.catalog import ecliptic_vector
from lc_pipeline.v2.data import canonical_json, parse_damit_lightcurve
from lc_pipeline.v2.preprocessing import KnownPeriod

INDEX_SCHEMA = "delphi.k3-lowq-damit-census-index.v1"
REFERENCE_SCHEMA = "delphi.k3-lowq-damit-census-reference.v1"
PREDICTION_SCHEMA = "delphi.k3-lowq-damit-census-predictions.v1"
ANALYSIS_SCHEMA = "delphi.k3-lowq-damit-census-analysis.v1"
BOOTSTRAP_SEED = 20260917
BOOTSTRAP_RESAMPLES = 10_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def build_index(snapshot: Path, output: Path) -> dict[str, object]:
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    model_table = snapshot / "tables" / "asteroid_models.csv"
    rows = _read_csv(model_table)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["asteroid_id"]].append(row)
    inputs: list[dict[str, object]] = []
    references: list[dict[str, object]] = []
    exclusions: list[dict[str, object]] = []
    for asteroid_id in sorted(grouped, key=int):
        models = grouped[asteroid_id]
        qualified: list[tuple[dict[str, str], int]] = []
        for row in models:
            try:
                quality = int(float(row["quality_flag"]))
            except (KeyError, ValueError):
                continue
            qualified.append((row, quality))
        if not qualified:
            continue
        qualities = [quality for _row, quality in qualified]
        if any(quality >= 3 for quality in qualities):
            continue
        highest = max(qualities)
        if highest not in (1, 2):
            continue
        selected = sorted(
            [
                row for row, quality in qualified if quality == highest
            ],
            key=lambda row: int(row["id"]),
        )
        object_id = f"damit:{asteroid_id}"
        try:
            period = float(selected[0]["period"])
        except (IndexError, KeyError, ValueError):
            exclusions.append({"object_id": object_id, "reason": "invalid_period"})
            continue
        if not math.isfinite(period) or period <= 0:
            exclusions.append({"object_id": object_id, "reason": "invalid_period"})
            continue
        relative = Path("files") / f"asteroid_{asteroid_id}" / "lc.txt"
        if not (snapshot / relative).is_file():
            exclusions.append({"object_id": object_id, "reason": "missing_lightcurve"})
            continue
        reference_axes = []
        try:
            for model in selected:
                reference_axes.append(
                    {
                        "model_id": int(model["id"]),
                        "axis": list(
                            ecliptic_vector(float(model["lambda"]), float(model["beta"]))
                        ),
                    }
                )
        except (KeyError, ValueError):
            exclusions.append({"object_id": object_id, "reason": "invalid_reference_axis"})
            continue
        inputs.append(
            {
                "object_id": object_id,
                "damit_asteroid_id": int(asteroid_id),
                "quality_flag": highest,
                "known_period_hours": period,
                "period_model_id": int(selected[0]["id"]),
                "period_provenance": (
                    "DAMIT low-quality model-table period; reference-axis columns excluded"
                ),
                "lightcurve_path": relative.as_posix(),
            }
        )
        references.append(
            {
                "object_id": object_id,
                "quality_flag": highest,
                "references": reference_axes,
            }
        )
    output.mkdir(parents=True)
    index = {
        "schema": INDEX_SCHEMA,
        "study_scope": "complete_low_quality_DAMIT_same_source_census",
        "quality_definition": (
            "highest model-table quality is 1 or 2 and no model with quality >=3 exists"
        ),
        "selection_uses_reference_axes": False,
        "model_table_sha256": _sha256(model_table),
        "object_count": len(inputs),
        "quality_counts": {
            str(quality): sum(row["quality_flag"] == quality for row in inputs)
            for quality in (1, 2)
        },
        "exclusions": exclusions,
        "objects": inputs,
    }
    _write_json(output / "input-index.json", index)
    reference = {
        "schema": REFERENCE_SCHEMA,
        "input_index_sha256": _sha256(output / "input-index.json"),
        "reference_quality_warning": (
            "Q1/Q2 DAMIT axes are lower-confidence model solutions, not physical ground truth"
        ),
        "object_count": len(references),
        "objects": references,
    }
    _write_json(output / "reference-cohort.json", reference)
    return index


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def predict(
    snapshot: Path,
    index_path: Path,
    model_directory: Path,
    model_manifest: Path,
    output: Path,
    *,
    fold: int,
    device: str,
    resume: bool,
) -> dict[str, object]:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("schema") != INDEX_SCHEMA:
        raise ValueError("input index schema mismatch")
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    models, names, bindings = _load_models(model_directory, model_manifest, (fold,), str(target))
    if len(models) != 5:
        raise ValueError("census policy requires five seeds from one fixed fold")
    output.mkdir(parents=True, exist_ok=resume)
    partial = output / "predictions.jsonl.partial"
    completed: set[str] = set()
    if resume and partial.is_file():
        with partial.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    completed.add(str(json.loads(line)["object_id"]))
    elif partial.exists():
        raise ValueError("partial predictions exist; use --resume")
    mode = "a" if resume else "x"
    started = time.perf_counter()
    new_count = 0
    with partial.open(mode, encoding="utf-8", newline="\n", buffering=1) as handle:
        for position, row in enumerate(index["objects"], start=1):
            object_id = str(row["object_id"])
            if object_id in completed:
                continue
            source = (snapshot / str(row["lightcurve_path"])).resolve()
            expected_root = snapshot.resolve()
            try:
                source.relative_to(expected_root)
            except ValueError as exc:
                raise ValueError(f"lightcurve path escapes snapshot: {object_id}") from exc
            before = time.perf_counter()
            try:
                epochs = parse_damit_lightcurve(source)
                known_period = KnownPeriod(
                    float(row["known_period_hours"]), str(row["period_provenance"])
                )
                inputs = _model_inputs(epochs, known_period, target)
                _sync(target)
                with torch.inference_mode():
                    grids = np.stack(
                        [
                            score_axial_grid(model, inputs, chunk_size=1024)
                            for model in models
                        ]
                    )
                    mean = ensemble_score_grids(grids[:, None, :])[0]
                    modes = modes_from_score_grid(mean)
                with torch.enable_grad():
                    axes, scores = refine_ensemble_axes(
                        models,
                        inputs,
                        np.asarray([mode.axis_xyz for mode in modes]),
                    )
                _sync(target)
                result = {
                    "object_id": object_id,
                    "status": "ok",
                    "quality_flag": int(row["quality_flag"]),
                    "lightcurve_sha256": _sha256(source),
                    "epoch_count": len(epochs),
                    "observation_count": sum(len(epoch.observations) for epoch in epochs),
                    "wall_seconds": time.perf_counter() - before,
                    "axes": [
                        {
                            "axis_xyz": axis.tolist(),
                            "score": float(score),
                            "grid_index": int(mode_row.grid_index),
                        }
                        for axis, score, mode_row in zip(
                            axes, scores, modes, strict=True
                        )
                    ],
                }
            except (OSError, TypeError, ValueError, RuntimeError) as exc:
                result = {
                    "object_id": object_id,
                    "status": "input_rejected",
                    "quality_flag": int(row["quality_flag"]),
                    "wall_seconds": time.perf_counter() - before,
                    "reason": f"{type(exc).__name__}:{exc}",
                }
            handle.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
            new_count += 1
            if position % 25 == 0 or position == len(index["objects"]):
                elapsed = time.perf_counter() - started
                print(
                    f"processed {position}/{len(index['objects'])}; new={new_count}; "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )
    final = output / "predictions.jsonl"
    if final.exists():
        raise ValueError("final predictions already exist")
    os.replace(partial, final)
    counts = Counter()
    total_wall = 0.0
    with final.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            counts[str(row["status"])] += 1
            total_wall += float(row["wall_seconds"])
    manifest = {
        "schema": PREDICTION_SCHEMA,
        "study_scope": "complete_low_quality_DAMIT_same_source_census",
        "inference_policy": "fixed_fold_five_seed_ensemble",
        "fold": fold,
        "model_count": len(models),
        "checkpoint_names": names,
        "checkpoint_bindings": bindings,
        "device": str(target),
        "input_index_sha256": _sha256(index_path),
        "model_manifest_sha256": _sha256(model_manifest),
        "prediction_rows": sum(counts.values()),
        "status_counts": dict(sorted(counts.items())),
        "sum_object_wall_seconds": total_wall,
        "predictions_path": final.name,
        "predictions_sha256": _sha256(final),
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def _bootstrap(values: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    chunk = 128
    for start in range(0, BOOTSTRAP_RESAMPLES, chunk):
        stop = min(BOOTSTRAP_RESAMPLES, start + chunk)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        samples[start:stop] = np.mean(values[indices], axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _summary(errors: np.ndarray) -> dict[str, object]:
    low, high = _bootstrap(errors)
    return {
        "n_objects": len(errors),
        "mean_oracle_at_3_error_deg": float(np.mean(errors)),
        "mean_bootstrap_ci95_deg": [low, high],
        "median_oracle_at_3_error_deg": float(np.median(errors)),
        "within_20_deg": float(np.mean(errors <= 20.0)),
        "within_30_deg": float(np.mean(errors <= 30.0)),
    }


def analyze(
    reference_path: Path,
    prediction_directory: Path,
    atlas_path: Path,
    splits_path: Path,
    output: Path,
    *,
    fold: int,
) -> dict[str, object]:
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    manifest_path = prediction_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    predictions_path = prediction_directory / str(manifest["predictions_path"])
    if _sha256(predictions_path) != manifest["predictions_sha256"]:
        raise ValueError("prediction JSONL hash mismatch")
    atlas = json.loads(atlas_path.read_text(encoding="utf-8"))
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    fold_rows = [
        row
        for row in splits.get("folds", [])
        if isinstance(row, dict) and row.get("fold") == fold
    ]
    if len(fold_rows) != 1 or not isinstance(fold_rows[0].get("train_ids"), list):
        raise ValueError(f"split document has no unique train role for fold {fold}")
    expected_ids = sorted(str(value) for value in fold_rows[0]["train_ids"])
    expected_ids_sha256 = hashlib.sha256(
        canonical_json(expected_ids).encode("utf-8")
    ).hexdigest()
    if atlas.get("schema") != "delphi.axis-atlas.v1":
        raise ValueError("atlas schema mismatch")
    if atlas.get("object_count") != len(expected_ids):
        raise ValueError("atlas was not fitted to the selected fold training role")
    if atlas.get("train_object_ids_sha256") != expected_ids_sha256:
        raise ValueError("atlas training identity hash does not match the selected fold")
    atlas_axes = np.asarray(atlas["axes"], dtype=np.float64)
    if atlas_axes.shape != (3, 3) or not np.all(np.isfinite(atlas_axes)):
        raise ValueError("atlas must contain three finite axes")
    reference_by_id = {row["object_id"]: row for row in reference["objects"]}
    result_rows: list[dict[str, object]] = []
    rejection_reasons: Counter[str] = Counter()
    with predictions_path.open(encoding="utf-8") as handle:
        for line in handle:
            prediction = json.loads(line)
            if prediction["status"] != "ok":
                rejection_reasons[str(prediction["reason"])] += 1
                continue
            object_id = str(prediction["object_id"])
            refs = np.asarray(
                [row["axis"] for row in reference_by_id[object_id]["references"]],
                dtype=np.float64,
            )
            axes = np.asarray(
                [row["axis_xyz"] for row in prediction["axes"]], dtype=np.float64
            )
            error = float(
                np.min(axial_angular_error_deg(axes[:, None, :], refs[None, :, :]))
            )
            atlas_error = float(
                np.min(
                    axial_angular_error_deg(
                        atlas_axes[:, None, :], refs[None, :, :]
                    )
                )
            )
            result_rows.append(
                {
                    "object_id": object_id,
                    "quality_flag": int(reference_by_id[object_id]["quality_flag"]),
                    "oracle_at_3_error_deg": error,
                    "atlas_oracle_at_3_error_deg": atlas_error,
                    "wall_seconds": float(prediction["wall_seconds"]),
                    "epoch_count": int(prediction["epoch_count"]),
                    "observation_count": int(prediction["observation_count"]),
                }
            )
    errors = np.asarray(
        [row["oracle_at_3_error_deg"] for row in result_rows], dtype=np.float64
    )
    atlas_errors = np.asarray(
        [row["atlas_oracle_at_3_error_deg"] for row in result_rows], dtype=np.float64
    )
    improvement = atlas_errors - errors
    paired_low, paired_high = _bootstrap(improvement)
    strata: dict[str, object] = {}
    for quality in (1, 2):
        selected = np.asarray(
            [
                row["oracle_at_3_error_deg"]
                for row in result_rows
                if row["quality_flag"] == quality
            ],
            dtype=np.float64,
        )
        if selected.size:
            strata[str(quality)] = _summary(selected)
    wall = np.asarray([row["wall_seconds"] for row in result_rows], dtype=np.float64)
    result = {
        "schema": ANALYSIS_SCHEMA,
        "study_scope": "complete_low_quality_DAMIT_same_source_census",
        "reference_quality_warning": reference["reference_quality_warning"],
        "prediction_manifest_sha256": _sha256(manifest_path),
        "reference_sha256": _sha256(reference_path),
        "atlas_provenance": {
            "fold": fold,
            "atlas_file_sha256": _sha256(atlas_path),
            "atlas_sha256": atlas["atlas_sha256"],
            "axes": atlas["axes"],
            "object_count": atlas["object_count"],
            "train_object_ids_sha256": atlas["train_object_ids_sha256"],
            "split_sha256": _sha256(splits_path),
        },
        "selected_denominator": reference["object_count"],
        "analyzed_denominator": len(result_rows),
        "input_rejected_count": sum(rejection_reasons.values()),
        "input_rejection_reasons": dict(rejection_reasons),
        "primary": _summary(errors),
        "quality_strata": strata,
        "train_only_atlas": _summary(atlas_errors),
        "paired_atlas_minus_k3": {
            "mean_improvement_deg": float(np.mean(improvement)),
            "bootstrap_ci95_deg": [paired_low, paired_high],
        },
        "timing": {
            "scope": "raw_file_parse_through_refined_axes_with_models_already_loaded",
            "mean_seconds_per_analyzed_object": float(np.mean(wall)),
            "median_seconds_per_analyzed_object": float(np.median(wall)),
            "sum_seconds": float(np.sum(wall)),
        },
        "objects": result_rows,
    }
    _write_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    index = commands.add_parser("index")
    index.add_argument("--snapshot", type=Path, required=True)
    index.add_argument("--output", type=Path, required=True)
    prediction = commands.add_parser("predict")
    prediction.add_argument("--snapshot", type=Path, required=True)
    prediction.add_argument("--index", type=Path, required=True)
    prediction.add_argument("--model-directory", type=Path, required=True)
    prediction.add_argument("--model-manifest", type=Path, required=True)
    prediction.add_argument("--output", type=Path, required=True)
    prediction.add_argument("--fold", type=int, default=0, choices=range(5))
    prediction.add_argument("--device", default="cuda")
    prediction.add_argument("--resume", action="store_true")
    analysis = commands.add_parser("analyze")
    analysis.add_argument("--reference", type=Path, required=True)
    analysis.add_argument("--predictions", type=Path, required=True)
    analysis.add_argument("--atlas", type=Path, required=True)
    analysis.add_argument("--splits", type=Path, required=True)
    analysis.add_argument("--fold", type=int, default=0, choices=range(5))
    analysis.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "index":
            result = build_index(args.snapshot, args.output)
        elif args.command == "predict":
            result = predict(
                args.snapshot,
                args.index,
                args.model_directory,
                args.model_manifest,
                args.output,
                fold=args.fold,
                device=args.device,
                resume=args.resume,
            )
        else:
            result = analyze(
                args.reference,
                args.predictions,
                args.atlas,
                args.splits,
                args.output,
                fold=args.fold,
            )
    except ValueError as exc:
        parser.error(str(exc))
    if args.command == "index":
        summary = {
            "object_count": result["object_count"],
            "quality_counts": result["quality_counts"],
            "exclusion_count": len(result["exclusions"]),
        }
    elif args.command == "predict":
        summary = {
            "prediction_rows": result["prediction_rows"],
            "status_counts": result["status_counts"],
        }
    else:
        summary = {
            "selected_denominator": result["selected_denominator"],
            "analyzed_denominator": result["analyzed_denominator"],
            "input_rejected_count": result["input_rejected_count"],
        }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
