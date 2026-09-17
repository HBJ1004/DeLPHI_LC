#!/usr/bin/env python3
"""Measure frozen K3 ensemble size against low-Q DAMIT agreement and cost."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from lc_pipeline.k3.bundle import _model_inputs
from lc_pipeline.k3.evaluation import modes_from_score_grid
from lc_pipeline.k3.inference import refine_axes, refine_ensemble_axes, score_axial_grid
from lc_pipeline.k3.ztf_prediction import _load_models, _load_prepared
from lc_pipeline.physics.axial import axial_angular_error_deg

PREDICTION_SCHEMA = "delphi.k3-lowq-ensemble-ablation-predictions.v1"
ANALYSIS_SCHEMA = "delphi.k3-lowq-ensemble-ablation-analysis.v1"
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


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _policies(bindings: list[dict[str, object]]) -> list[dict[str, object]]:
    by_fold: dict[int, list[int]] = defaultdict(list)
    by_seed: dict[int, list[int]] = defaultdict(list)
    policies: list[dict[str, object]] = []
    for index, binding in enumerate(bindings):
        fold = int(binding["fold"])
        seed = int(binding["seed"])
        by_fold[fold].append(index)
        by_seed[seed].append(index)
        policies.append(
            {
                "policy_id": f"single-fold-{fold}-seed-{seed}",
                "family": "single_checkpoint",
                "indices": [index],
            }
        )
    for fold in sorted(by_fold):
        policies.append(
            {
                "policy_id": f"fold-{fold}-five-seed",
                "family": "five_seeds_one_fold",
                "indices": by_fold[fold],
            }
        )
    for seed in sorted(by_seed):
        policies.append(
            {
                "policy_id": f"seed-{seed}-five-fold",
                "family": "one_seed_each_fold",
                "indices": by_seed[seed],
            }
        )
    policies.append(
        {
            "policy_id": "all-25",
            "family": "all_25",
            "indices": list(range(len(bindings))),
        }
    )
    return policies


def predict(
    prepared: Path,
    model_directory: Path,
    model_manifest: Path,
    output: Path,
    *,
    device: str,
) -> dict[str, object]:
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    prepared_manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
    rows = prepared_manifest.get("objects")
    if not isinstance(rows, list) or not rows:
        raise ValueError("prepared manifest has no objects")
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    models, names, bindings = _load_models(
        model_directory, model_manifest, tuple(range(5)), str(target)
    )
    policies = _policies(bindings)
    output.mkdir(parents=True)
    started = time.perf_counter()
    result_rows: list[dict[str, object]] = []
    for object_index, row in enumerate(
        sorted(rows, key=lambda value: str(value["object_id"])), start=1
    ):
        source = prepared / str(row["path"])
        if _sha256(source) != row.get("sha256"):
            raise ValueError(f"prepared object hash mismatch: {row.get('object_id')}")
        object_id, period, epochs = _load_prepared(source)
        _sync(target)
        before_inputs = time.perf_counter()
        inputs = _model_inputs(epochs, period, target)
        _sync(target)
        input_seconds = time.perf_counter() - before_inputs
        score_maps: list[np.ndarray] = []
        grid_seconds: list[float] = []
        with torch.inference_mode():
            for model in models:
                _sync(target)
                before_grid = time.perf_counter()
                score_maps.append(score_axial_grid(model, inputs, chunk_size=1024))
                _sync(target)
                grid_seconds.append(time.perf_counter() - before_grid)
        grids = np.stack(score_maps)
        policy_rows: list[dict[str, object]] = []
        for policy in policies:
            indices = [int(value) for value in policy["indices"]]
            selected_models = [models[index] for index in indices]
            _sync(target)
            before_postprocess = time.perf_counter()
            mean = np.mean(grids[indices], axis=0)
            modes = modes_from_score_grid(mean)
            with torch.enable_grad():
                starts = np.asarray([mode.axis_xyz for mode in modes])
                if len(selected_models) == 1:
                    axes, _scores = refine_axes(selected_models[0], inputs, starts)
                else:
                    axes, _scores = refine_ensemble_axes(
                        selected_models,
                        inputs,
                        starts,
                    )
            _sync(target)
            postprocess_seconds = time.perf_counter() - before_postprocess
            grid_total = float(sum(grid_seconds[index] for index in indices))
            policy_rows.append(
                {
                    "policy_id": policy["policy_id"],
                    "family": policy["family"],
                    "model_count": len(indices),
                    "model_indices": indices,
                    "axes": axes.tolist(),
                    "input_seconds": input_seconds,
                    "grid_seconds": grid_total,
                    "mode_and_refinement_seconds": postprocess_seconds,
                    "estimated_warm_inference_seconds": (
                        input_seconds + grid_total + postprocess_seconds
                    ),
                }
            )
        object_payload = {
            "schema": "delphi.k3-lowq-ensemble-ablation-object.v1",
            "object_id": object_id,
            "prepared_sha256": _sha256(source),
            "policies": policy_rows,
        }
        destination = output / "objects" / f"{object_id.replace(':', '_')}.json"
        _write_json(destination, object_payload)
        result_rows.append(
            {
                "object_id": object_id,
                "path": destination.relative_to(output).as_posix(),
                "sha256": _sha256(destination),
            }
        )
        print(f"completed {object_index}/{len(rows)}: {object_id}", flush=True)
    manifest = {
        "schema": PREDICTION_SCHEMA,
        "study_scope": "post_hoc_cost_accuracy_sensitivity_not_model_selection",
        "prepared_manifest_sha256": _sha256(prepared / "manifest.json"),
        "model_manifest_sha256": _sha256(model_manifest),
        "device": str(target),
        "timing_scope": (
            "warm_loaded_models; per-object tokenization plus selected model grid scoring "
            "plus mode extraction and refinement; checkpoint loading excluded"
        ),
        "policy_count": len(policies),
        "policies": [
            {
                "policy_id": policy["policy_id"],
                "family": policy["family"],
                "model_count": len(policy["indices"]),
                "model_indices": policy["indices"],
            }
            for policy in policies
        ],
        "checkpoint_names": names,
        "checkpoint_bindings": bindings,
        "object_count": len(result_rows),
        "objects": result_rows,
        "total_combined_run_wall_seconds": time.perf_counter() - started,
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def _bootstrap_interval(values: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for start in range(0, BOOTSTRAP_RESAMPLES, 512):
        stop = min(BOOTSTRAP_RESAMPLES, start + 512)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        samples[start:stop] = np.mean(values[indices], axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def analyze(reference: Path, predictions: Path, baseline_analysis: Path, output: Path) -> dict[str, object]:
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    references = json.loads(reference.read_text(encoding="utf-8"))
    manifest = json.loads((predictions / "manifest.json").read_text(encoding="utf-8"))
    baseline = json.loads(baseline_analysis.read_text(encoding="utf-8"))
    if manifest.get("schema") != PREDICTION_SCHEMA:
        raise ValueError("prediction manifest schema mismatch")
    reference_by_id = {row["object_id"]: row for row in references["objects"]}
    atlas_by_id = {
        row["object_id"]: float(row["atlas_oracle_at_3_error_deg"])
        for row in baseline["objects"]
    }
    by_policy: dict[str, list[dict[str, float]]] = defaultdict(list)
    family_by_policy = {
        row["policy_id"]: row["family"] for row in manifest["policies"]
    }
    count_by_policy = {
        row["policy_id"]: int(row["model_count"]) for row in manifest["policies"]
    }
    for row in manifest["objects"]:
        payload_path = predictions / str(row["path"])
        if _sha256(payload_path) != row["sha256"]:
            raise ValueError(f"prediction object hash mismatch: {row['object_id']}")
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        object_id = payload["object_id"]
        target = np.asarray(
            [item["axis"] for item in reference_by_id[object_id]["references"]],
            dtype=np.float64,
        )
        for policy in payload["policies"]:
            axes = np.asarray(policy["axes"], dtype=np.float64)
            error = float(
                np.min(axial_angular_error_deg(axes[:, None, :], target[None, :, :]))
            )
            by_policy[policy["policy_id"]].append(
                {
                    "error_deg": error,
                    "atlas_error_deg": atlas_by_id[object_id],
                    "seconds": float(policy["estimated_warm_inference_seconds"]),
                }
            )
    all25 = np.asarray(
        [row["error_deg"] for row in by_policy["all-25"]], dtype=np.float64
    )
    policies: list[dict[str, object]] = []
    for policy_id in [row["policy_id"] for row in manifest["policies"]]:
        rows = by_policy[policy_id]
        errors = np.asarray([row["error_deg"] for row in rows], dtype=np.float64)
        atlas_errors = np.asarray(
            [row["atlas_error_deg"] for row in rows], dtype=np.float64
        )
        seconds = np.asarray([row["seconds"] for row in rows], dtype=np.float64)
        atlas_difference = atlas_errors - errors
        relative_to_all25 = errors - all25
        atlas_low, atlas_high = _bootstrap_interval(atlas_difference)
        all25_low, all25_high = _bootstrap_interval(relative_to_all25)
        policies.append(
            {
                "policy_id": policy_id,
                "family": family_by_policy[policy_id],
                "model_count": count_by_policy[policy_id],
                "n_objects": len(errors),
                "mean_oracle_at_3_error_deg": float(np.mean(errors)),
                "median_oracle_at_3_error_deg": float(np.median(errors)),
                "within_20_deg": float(np.mean(errors <= 20.0)),
                "within_30_deg": float(np.mean(errors <= 30.0)),
                "mean_warm_inference_seconds": float(np.mean(seconds)),
                "median_warm_inference_seconds": float(np.median(seconds)),
                "paired_atlas_minus_policy_mean_deg": float(np.mean(atlas_difference)),
                "paired_atlas_minus_policy_ci95_deg": [atlas_low, atlas_high],
                "paired_policy_minus_all25_mean_deg": float(np.mean(relative_to_all25)),
                "paired_policy_minus_all25_ci95_deg": [all25_low, all25_high],
            }
        )
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in policies:
        grouped[str(row["family"])].append(row)
    family_summary: list[dict[str, object]] = []
    all25_time = float(
        next(row for row in policies if row["policy_id"] == "all-25")[
            "mean_warm_inference_seconds"
        ]
    )
    for family in (
        "single_checkpoint",
        "five_seeds_one_fold",
        "one_seed_each_fold",
        "all_25",
    ):
        rows = grouped[family]
        errors = np.asarray(
            [float(row["mean_oracle_at_3_error_deg"]) for row in rows], dtype=np.float64
        )
        times = np.asarray(
            [float(row["mean_warm_inference_seconds"]) for row in rows], dtype=np.float64
        )
        family_summary.append(
            {
                "family": family,
                "policy_replicates": len(rows),
                "model_count": int(rows[0]["model_count"]),
                "mean_of_policy_mean_errors_deg": float(np.mean(errors)),
                "minimum_policy_mean_error_deg": float(np.min(errors)),
                "maximum_policy_mean_error_deg": float(np.max(errors)),
                "mean_warm_inference_seconds": float(np.mean(times)),
                "cost_fraction_of_all25": float(np.mean(times) / all25_time),
            }
        )
    result = {
        "schema": ANALYSIS_SCHEMA,
        "study_scope": "post_hoc_cost_accuracy_sensitivity_not_model_selection",
        "selection_warning": (
            "No best fold, seed, or checkpoint may be presented as prespecified; "
            "family averages and ranges are the intended comparison."
        ),
        "prediction_manifest_sha256": _sha256(predictions / "manifest.json"),
        "reference_sha256": _sha256(reference),
        "object_count": len(manifest["objects"]),
        "policy_count": len(policies),
        "family_summary": family_summary,
        "policies": policies,
    }
    _write_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prediction = commands.add_parser("predict")
    prediction.add_argument("--prepared", type=Path, required=True)
    prediction.add_argument("--model-directory", type=Path, required=True)
    prediction.add_argument("--model-manifest", type=Path, required=True)
    prediction.add_argument("--output", type=Path, required=True)
    prediction.add_argument("--device", default="cuda")
    analysis = commands.add_parser("analyze")
    analysis.add_argument("--reference", type=Path, required=True)
    analysis.add_argument("--predictions", type=Path, required=True)
    analysis.add_argument("--baseline-analysis", type=Path, required=True)
    analysis.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "predict":
            result = predict(
                args.prepared,
                args.model_directory,
                args.model_manifest,
                args.output,
                device=args.device,
            )
        else:
            result = analyze(
                args.reference,
                args.predictions,
                args.baseline_analysis,
                args.output,
            )
    except ValueError as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "object_count": result["object_count"],
                "policy_count": result["policy_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
