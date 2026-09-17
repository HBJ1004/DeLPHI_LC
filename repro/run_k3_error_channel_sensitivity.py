#!/usr/bin/env python3
"""Re-run frozen ZTF and ALCDEF predictions without reported-error channels.

This post hoc sensitivity keeps every photon, epoch, geometry vector, period,
checkpoint, and inference setting fixed.  It only removes the optional
``measured_error`` field from prepared observations.  Prediction remains
separate from reference scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from lc_pipeline.k3.bundle import _model_inputs
from lc_pipeline.k3.evaluation import ensemble_score_grids, modes_from_score_grid
from lc_pipeline.k3.generalization_validation import (
    _reference_axes,
    _score_prediction,
    analyze_generalization,
)
from lc_pipeline.k3.inference import refine_ensemble_axes, score_axial_grid
from lc_pipeline.k3.ztf_prediction import _load_models, _load_prepared
from lc_pipeline.physics.axial import axial_angular_error_deg

SCHEMA = "delphi.k3-error-channel-sensitivity.v1"
BOOTSTRAP_SEED = 20260917
BOOTSTRAP_RESAMPLES = 10_000


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
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError(f"refusing to overwrite {path}")
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


def remove_measured_errors(
    value: Mapping[str, object],
) -> tuple[dict[str, object], int, int, int]:
    """Return a JSON-safe copy with only measured-error fields removed."""
    result = json.loads(json.dumps(value))
    epochs = result.get("epochs")
    if not isinstance(epochs, list) or not epochs:
        raise ValueError("prepared object has no epochs")
    observations = 0
    removed_fields = 0
    removed_values = 0
    for epoch in epochs:
        if not isinstance(epoch, dict) or not isinstance(epoch.get("observations"), list):
            raise ValueError("prepared epoch is invalid")
        for observation in epoch["observations"]:
            if not isinstance(observation, dict):
                raise ValueError("prepared observation is invalid")
            observations += 1
            if "measured_error" in observation:
                if observation["measured_error"] is not None:
                    removed_values += 1
                observation.pop("measured_error")
                removed_fields += 1
    if observations == 0:
        raise ValueError("prepared object has no observations")
    result["error_channel_sensitivity"] = {
        "role": "post_hoc_input_sensitivity",
        "transform": "remove_optional_measured_error_values_and_availability_flags",
        "observation_count": observations,
        "removed_field_count": removed_fields,
        "removed_nonnull_value_count": removed_values,
    }
    return result, observations, removed_fields, removed_values


def prepare_directory(source: Path, destination: Path, *, cohort: str) -> dict[str, object]:
    manifest_path = source / "manifest.json"
    manifest = _read(manifest_path)
    rows = manifest.get("objects")
    if not isinstance(rows, list) or not rows:
        raise ValueError("prepared manifest has no objects")
    if destination.exists():
        raise ValueError(f"output already exists: {destination}")
    destination.mkdir(parents=True)
    output_rows = []
    total_observations = 0
    total_removed_fields = 0
    total_removed_values = 0
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("object_id"), str):
            raise ValueError("prepared manifest row is invalid")
        if row.get("status", "ready") != "ready":
            output_rows.append({**row, "path": None, "sha256": None})
            continue
        relative = row.get("path")
        if not isinstance(relative, str):
            raise ValueError("ready prepared row has no path")
        input_path = source / relative
        if _sha256(input_path) != row.get("sha256"):
            raise ValueError(f"prepared-object hash mismatch: {row['object_id']}")
        value, observations, removed_fields, removed_values = remove_measured_errors(
            _read(input_path)
        )
        output_path = destination / "objects" / Path(relative).name
        _write(output_path, value)
        output_rows.append(
            {
                **row,
                "path": output_path.relative_to(destination).as_posix(),
                "sha256": _sha256(output_path),
                "source_prepared_sha256": _sha256(input_path),
                "observation_count": observations,
                "removed_measured_error_field_count": removed_fields,
                "removed_nonnull_measured_error_count": removed_values,
            }
        )
        total_observations += observations
        total_removed_fields += removed_fields
        total_removed_values += removed_values
    output = {
        "schema": "delphi.k3-error-channel-sensitivity-prepared-directory.v1",
        "cohort": cohort,
        "source_manifest_sha256": _sha256(manifest_path),
        "object_count": len(rows),
        "ready_object_count": sum(row.get("status", "ready") == "ready" for row in rows),
        "observation_count": total_observations,
        "removed_measured_error_field_count": total_removed_fields,
        "removed_nonnull_measured_error_count": total_removed_values,
        "objects": output_rows,
    }
    _write(destination / "manifest.json", output)
    return output


def _predict_one(
    prepared_path: Path,
    models: Sequence[torch.nn.Module],
    device: torch.device,
    *,
    chunk_size: int,
) -> tuple[str, list[list[float]], float]:
    object_id, period, epochs = _load_prepared(prepared_path)
    inputs = _model_inputs(epochs, period, device)
    start = time.monotonic()
    with torch.inference_mode():
        grids = np.stack(
            [score_axial_grid(model, inputs, chunk_size=chunk_size) for model in models]
        )
        mean = ensemble_score_grids(grids[:, None, :])[0]
        modes = modes_from_score_grid(mean)
    with torch.enable_grad():
        axes, _ = refine_ensemble_axes(
            models,
            inputs,
            np.asarray([mode.axis_xyz for mode in modes]),
        )
    return object_id, axes.tolist(), time.monotonic() - start


def predict_ztf(
    prepared: Path,
    model_directory: Path,
    model_manifest: Path,
    destination: Path,
    *,
    device: str,
    chunk_size: int,
) -> dict[str, object]:
    manifest = _read(prepared / "manifest.json")
    rows = manifest["objects"]
    assert isinstance(rows, list)
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if isinstance(row, dict) and row.get("status", "ready") == "ready":
            fold = row.get("held_out_fold")
            if isinstance(fold, bool) or not isinstance(fold, int) or fold not in range(5):
                raise ValueError("ZTF sensitivity row lacks a held-out fold")
            grouped[fold].append(row)
    target = torch.device(device)
    outputs = []
    start = time.monotonic()
    for fold in sorted(grouped):
        models, names, bindings = _load_models(
            model_directory, model_manifest, (fold,), str(target)
        )
        for row in sorted(grouped[fold], key=lambda value: str(value["object_id"])):
            path = prepared / str(row["path"])
            object_id, axes, elapsed = _predict_one(
                path, models, target, chunk_size=chunk_size
            )
            if object_id != row["object_id"]:
                raise ValueError("ZTF prepared identity mismatch")
            outputs.append(
                {
                    "object_id": object_id,
                    "status": "ok",
                    "held_out_fold": fold,
                    "model_count": len(models),
                    "model_bindings": bindings,
                    "checkpoint_names": names,
                    "prepared_sha256": _sha256(path),
                    "axes": axes,
                    "elapsed_seconds": elapsed,
                }
            )
        del models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    result = {
        "schema": "delphi.k3-error-channel-sensitivity-predictions.v1",
        "cohort": "mapped_ztf_same_identity",
        "model_policy": "five_seed_held_out_fold_ensemble",
        "prepared_manifest_sha256": _sha256(prepared / "manifest.json"),
        "model_manifest_sha256": _sha256(model_manifest),
        "device": str(target),
        "chunk_size": chunk_size,
        "elapsed_seconds": time.monotonic() - start,
        "objects": sorted(outputs, key=lambda row: str(row["object_id"])),
    }
    _write(destination, result)
    return result


def predict_alcdef(
    prepared: Path,
    model_directory: Path,
    model_manifest: Path,
    destination: Path,
    *,
    device: str,
    chunk_size: int,
) -> dict[str, object]:
    manifest = _read(prepared / "manifest.json")
    rows = manifest["objects"]
    assert isinstance(rows, list)
    target = torch.device(device)
    models, names, bindings = _load_models(
        model_directory, model_manifest, tuple(range(5)), str(target)
    )
    outputs = []
    start = time.monotonic()
    for row in sorted(rows, key=lambda value: str(value["object_id"])):
        if not isinstance(row, dict) or row.get("status", "ready") != "ready":
            continue
        path = prepared / str(row["path"])
        object_id, axes, elapsed = _predict_one(path, models, target, chunk_size=chunk_size)
        if object_id != row["object_id"]:
            raise ValueError("ALCDEF prepared identity mismatch")
        outputs.append(
            {
                "object_id": object_id,
                "status": "ok",
                "model_count": len(models),
                "model_bindings": bindings,
                "checkpoint_names": names,
                "prepared_sha256": _sha256(path),
                "axes": axes,
                "elapsed_seconds": elapsed,
            }
        )
    result = {
        "schema": "delphi.k3-error-channel-sensitivity-predictions.v1",
        "cohort": "alcdef_gaia_new_identity",
        "model_policy": "all_25_frozen_models",
        "prepared_manifest_sha256": _sha256(prepared / "manifest.json"),
        "model_manifest_sha256": _sha256(model_manifest),
        "device": str(target),
        "chunk_size": chunk_size,
        "elapsed_seconds": time.monotonic() - start,
        "objects": sorted(outputs, key=lambda row: str(row["object_id"])),
    }
    _write(destination, result)
    return result


def _paired_interval(values: np.ndarray) -> dict[str, object]:
    if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("paired differences are invalid")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = np.empty(BOOTSTRAP_RESAMPLES, dtype=float)
    for index in range(BOOTSTRAP_RESAMPLES):
        selected = rng.integers(0, len(values), size=len(values))
        samples[index] = float(np.mean(values[selected]))
    return {
        "mean_difference_deg": float(np.mean(values)),
        "ci95_low_deg": float(np.quantile(samples, 0.025)),
        "ci95_high_deg": float(np.quantile(samples, 0.975)),
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED,
    }


def _descriptive(values: np.ndarray) -> dict[str, object]:
    if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("descriptive values are invalid")
    return {
        "n_objects": len(values),
        "mean_error_deg": float(np.mean(values)),
        "median_error_deg": float(np.median(values)),
        "within_20_fraction": float(np.mean(values <= 20.0)),
        "within_30_fraction": float(np.mean(values <= 30.0)),
    }


def analyze(
    *,
    ztf_predictions: Path,
    ztf_original_score: Path,
    reference_catalog: Path,
    ztf_atlas_directory: Path,
    alcdef_predictions: Path,
    alcdef_original_analysis: Path,
    alcdef_cohort: Path,
    alcdef_atlas: Path,
    output: Path,
) -> dict[str, object]:
    ztf = _read(ztf_predictions)
    original_ztf = _read(ztf_original_score)
    catalog = {}
    for line in reference_catalog.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        catalog[row["object_id"]] = row
    original_rows = {
        row["object_id"]: row
        for row in original_ztf["variants"]["original"]["objects"]
        if row.get("input_available")
    }
    atlas_by_fold = {
        fold: np.asarray(
            _read(ztf_atlas_directory / f"fold-{fold}.json")["axes"], dtype=float
        )
        for fold in range(5)
    }
    ztf_rows = []
    for prediction in ztf["objects"]:
        object_id = prediction["object_id"]
        references = _reference_axes(catalog[object_id])
        error, failure = _score_prediction(prediction, references, role="candidate")
        if failure is not None:
            raise ValueError(f"ZTF sensitivity prediction failed: {object_id}: {failure}")
        atlas_error = float(
            np.min(
                axial_angular_error_deg(
                    atlas_by_fold[int(original_rows[object_id]["held_out_fold"])][
                        :, None, :
                    ],
                    references[None, :, :],
                )
            )
        )
        ztf_rows.append(
            {
                "object_id": object_id,
                "original_error_deg": original_rows[object_id]["oracle_at_3_error_deg"],
                "train_only_atlas_error_deg": atlas_error,
                "without_error_channel_error_deg": error,
            }
        )
    ztf_original = np.asarray([row["original_error_deg"] for row in ztf_rows], dtype=float)
    ztf_without = np.asarray(
        [row["without_error_channel_error_deg"] for row in ztf_rows], dtype=float
    )
    ztf_atlas = np.asarray(
        [row["train_only_atlas_error_deg"] for row in ztf_rows], dtype=float
    )

    alcdef = _read(alcdef_predictions)
    alcdef_analysis = analyze_generalization(
        cohort=_read(alcdef_cohort),
        candidate_predictions={"objects": alcdef["objects"]},
        atlas_predictions=_read(alcdef_atlas),
    )
    original_alcdef = _read(alcdef_original_analysis)
    original_by_id = {
        row["object_id"]: row["oracle_at_3_error_deg"]
        for row in original_alcdef["objects"]
    }
    no_error_by_id = {
        row["object_id"]: row["oracle_at_3_error_deg"]
        for row in alcdef_analysis["objects"]
    }
    alcdef_ids = sorted(original_by_id)
    alcdef_original = np.asarray([original_by_id[key] for key in alcdef_ids], dtype=float)
    alcdef_without = np.asarray([no_error_by_id[key] for key in alcdef_ids], dtype=float)
    original_alcdef_objects = {
        row["object_id"]: row for row in original_alcdef["objects"]
    }
    alcdef_atlas_errors = np.asarray(
        [
            original_alcdef_objects[key]["atlas_oracle_at_3_error_deg"]
            for key in alcdef_ids
        ],
        dtype=float,
    )
    result = {
        "schema": SCHEMA,
        "role": "post_hoc_input_sensitivity",
        "intervention": (
            "remove reported measurement-error values and their availability flag; "
            "keep photometry, epochs, geometry, periods, models, and inference fixed"
        ),
        "ztf": {
            "n_objects": len(ztf_rows),
            "original_mean_deg": float(np.mean(ztf_original)),
            "train_only_atlas_mean_deg": float(np.mean(ztf_atlas)),
            "original_descriptive": _descriptive(ztf_original),
            "train_only_atlas_descriptive": _descriptive(ztf_atlas),
            "without_error_channel_descriptive": _descriptive(ztf_without),
            "original_minus_train_only_atlas": _paired_interval(
                ztf_original - ztf_atlas
            ),
            "without_error_channel_mean_deg": float(np.mean(ztf_without)),
            "without_minus_original": _paired_interval(ztf_without - ztf_original),
            "objects": ztf_rows,
        },
        "alcdef_gaia": {
            "n_objects": len(alcdef_ids),
            "original_mean_deg": float(np.mean(alcdef_original)),
            "without_error_channel_mean_deg": float(np.mean(alcdef_without)),
            "original_descriptive": _descriptive(alcdef_original),
            "train_only_atlas_descriptive": _descriptive(alcdef_atlas_errors),
            "without_error_channel_descriptive": _descriptive(alcdef_without),
            "original_minus_train_only_atlas": {
                "mean_difference_deg": -float(
                    original_alcdef["paired_train_only_atlas"]["mean_improvement_deg"]
                ),
                "ci95_low_deg": -float(
                    original_alcdef["paired_train_only_atlas"]["ci95_high_deg"]
                ),
                "ci95_high_deg": -float(
                    original_alcdef["paired_train_only_atlas"]["ci95_low_deg"]
                ),
                "resamples": original_alcdef["paired_train_only_atlas"]["resamples"],
                "seed": original_alcdef["paired_train_only_atlas"]["seed"],
            },
            "without_minus_original": _paired_interval(alcdef_without - alcdef_original),
            "without_error_channel_analysis": alcdef_analysis,
        },
        "source_hashes": {
            "ztf_predictions": _sha256(ztf_predictions),
            "ztf_original_score": _sha256(ztf_original_score),
            "reference_catalog": _sha256(reference_catalog),
            "ztf_atlases": {
                str(fold): _sha256(ztf_atlas_directory / f"fold-{fold}.json")
                for fold in range(5)
            },
            "alcdef_predictions": _sha256(alcdef_predictions),
            "alcdef_original_analysis": _sha256(alcdef_original_analysis),
            "alcdef_cohort": _sha256(alcdef_cohort),
            "alcdef_atlas": _sha256(alcdef_atlas),
        },
    }
    _write(output, result)
    return result


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.output.exists():
        raise ValueError(f"output already exists: {args.output}")
    args.output.mkdir(parents=True)
    ztf_prepared = args.output / "prepared-ztf-no-error"
    alcdef_prepared = args.output / "prepared-alcdef-no-error"
    prepare_directory(args.ztf_prepared, ztf_prepared, cohort="mapped_ztf_same_identity")
    prepare_directory(args.alcdef_prepared, alcdef_prepared, cohort="alcdef_gaia_new_identity")
    ztf_predictions = args.output / "ztf-predictions.json"
    alcdef_predictions = args.output / "alcdef-predictions.json"
    predict_ztf(
        ztf_prepared,
        args.model_directory,
        args.model_manifest,
        ztf_predictions,
        device=args.device,
        chunk_size=args.chunk_size,
    )
    predict_alcdef(
        alcdef_prepared,
        args.model_directory,
        args.model_manifest,
        alcdef_predictions,
        device=args.device,
        chunk_size=args.chunk_size,
    )
    return analyze(
        ztf_predictions=ztf_predictions,
        ztf_original_score=args.ztf_original_score,
        reference_catalog=args.reference_catalog,
        ztf_atlas_directory=args.ztf_atlas_directory,
        alcdef_predictions=alcdef_predictions,
        alcdef_original_analysis=args.alcdef_original_analysis,
        alcdef_cohort=args.alcdef_cohort,
        alcdef_atlas=args.alcdef_atlas,
        output=args.output / "analysis.json",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ztf-prepared", type=Path, required=True)
    parser.add_argument("--ztf-original-score", type=Path, required=True)
    parser.add_argument("--reference-catalog", type=Path, required=True)
    parser.add_argument("--ztf-atlas-directory", type=Path, required=True)
    parser.add_argument("--alcdef-prepared", type=Path, required=True)
    parser.add_argument("--alcdef-original-analysis", type=Path, required=True)
    parser.add_argument("--alcdef-cohort", type=Path, required=True)
    parser.add_argument("--alcdef-atlas", type=Path, required=True)
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk-size", type=int, default=1024)
    args = parser.parse_args()
    try:
        result = run(args)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "ztf_without_error_channel_mean_deg": result["ztf"][
                    "without_error_channel_mean_deg"
                ],
                "alcdef_without_error_channel_mean_deg": result["alcdef_gaia"][
                    "without_error_channel_mean_deg"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
