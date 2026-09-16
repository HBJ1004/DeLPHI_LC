"""Prepare and predict the already-exposed ZTF cohort; no new holdout is opened."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from lc_pipeline.k3.bundle import _model_inputs
from lc_pipeline.k3.evaluation import ensemble_score_grids, modes_from_score_grid
from lc_pipeline.k3.generalization_diagnostics import (
    VARIANTS,
    prepare_diagnostics,
    sha256,
    write_new_json,
)
from lc_pipeline.k3.inference import refine_ensemble_axes, score_axial_grid
from lc_pipeline.k3.survey_identity import validate_survey_binding
from lc_pipeline.k3.ztf_prediction import _fold_policy, _load_models, _load_prepared


def validate_prediction_manifest(manifest_path: Path, split_path: Path) -> dict:
    """Reject truncated cohorts and crosslinked photon identities before model loading."""
    manifest = json.loads(manifest_path.read_text())
    split_document = json.loads(split_path.read_text())
    folds = {}
    for fold in split_document["folds"]:
        for object_id in fold["test_ids"]:
            if object_id in folds:
                raise ValueError("object appears in multiple held-out test folds")
            folds[object_id] = fold["fold"]
    if (
        manifest.get("schema") != "delphi.k3-generalization-diagnostics.v1"
        or manifest.get("role") != "post_hoc_development_only"
        or manifest.get("object_count") != len(folds)
        or set(manifest.get("variants", [])) != set(VARIANTS)
        or len(manifest.get("variants", [])) != len(VARIANTS)
    ):
        raise ValueError("diagnostic manifest does not declare the complete development factorial")
    keys = [(row["object_id"], row["variant"]) for row in manifest["records"]]
    expected = {(object_id, variant) for object_id in folds for variant in VARIANTS}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("diagnostic manifest must retain every object in every variant exactly once")
    for row in manifest["records"]:
        if row["status"] == "input_unavailable":
            if row["path"] is not None or row["sha256"] is not None:
                raise ValueError("unavailable input cannot carry a prepared path")
            continue
        if row["status"] != "ready":
            raise ValueError("unknown diagnostic input status")
        path = manifest_path.parent / row["path"]
        if not path.resolve().is_relative_to(manifest_path.parent.resolve()):
            raise ValueError("prepared path escapes diagnostic manifest directory")
        if sha256(path) != row["sha256"]:
            raise ValueError("prepared hash mismatch")
        value = json.loads(path.read_text())
        validate_survey_binding(value)
        if (
            value["object_id"] != row["object_id"]
            or value["identity_binding"]["held_out_fold"] != folds[row["object_id"]]
            or value["identity_binding"]["identity_map_sha256"] != manifest["identity_map_sha256"]
            or value["generalization_transform"]["variant"] != row["variant"]
            or value["generalization_transform"]["study_spec_sha256"] != manifest["study_spec_sha256"]
        ):
            raise ValueError("prepared identity, held-out fold, or variant binding mismatch")
    return manifest


def predict(args: argparse.Namespace) -> dict:
    root = args.manifest.parent
    manifest = validate_prediction_manifest(args.manifest, args.splits)
    if args.variant is not None and (len(args.variant) != len(set(args.variant)) or set(args.variant) - set(VARIANTS)):
        raise ValueError("selected variants must be unique names from the factorial design")
    if args.output_directory.exists():
        raise ValueError("prediction output directory must be new")
    args.output_directory.mkdir(parents=True)
    records = manifest["records"]
    selected = [row for row in records if args.variant is None or row["variant"] in args.variant]
    if not selected:
        raise ValueError("no variants selected")
    fold_rows = {}
    for record in selected:
        folds = _fold_policy(record["object_id"], args.splits, "existing_identity")
        fold_rows.setdefault(folds, []).append(record)
    outputs = []
    start = time.monotonic()
    for folds, rows in sorted(fold_rows.items()):
        models, names, bindings = _load_models(args.model_directory, args.model_manifest, folds, args.device)
        for index, row in enumerate(rows):
            path = root / row["path"] if row["path"] is not None else None
            if path is not None and sha256(path) != row["sha256"]:
                raise ValueError("prepared hash mismatch")
            payload = {"schema": "delphi.k3-generalization-development-prediction.v1",
                       "object_id": row["object_id"], "variant": row["variant"], "folds": list(folds),
                       "candidate_semantics": "unordered_three_axis_set", "model_count": len(models),
                       "model_manifest_sha256": sha256(args.model_manifest), "prepared_sha256": row["sha256"],
                       "model_bindings": bindings, "checkpoint_names": names}
            object_start = time.monotonic()
            try:
                if row.get("status") == "input_unavailable" or path is None:
                    raise ValueError(f'input unavailable: {row.get("reason", "missing_prepared_input")}')
                loaded_id, period, epochs = _load_prepared(path)
                if loaded_id != row["object_id"]:
                    raise ValueError("prepared object identity changed after preflight")
                inputs = _model_inputs(epochs, period, torch.device(args.device))
                grids = np.stack([score_axial_grid(model, inputs, chunk_size=args.chunk_size) for model in models])
                mean = ensemble_score_grids(grids[:, None, :])[0]
                modes = modes_from_score_grid(mean)
                axes, _ = refine_ensemble_axes(models, inputs, np.asarray([mode.axis_xyz for mode in modes]))
                payload.update(status="ok", axes=axes.tolist())
            except (ValueError, RuntimeError) as exc:
                payload.update(status="failed", axes=[], reason=f"{type(exc).__name__}: {exc}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            payload["elapsed_seconds"] = time.monotonic() - object_start
            output = args.output_directory / row["variant"] / (row["object_id"] + ".json")
            write_new_json(output, payload)
            outputs.append({"object_id": row["object_id"], "variant": row["variant"],
                            "path": output.relative_to(args.output_directory).as_posix(), "sha256": sha256(output),
                            "status": payload["status"]})
            if index % 10 == 0:
                print(json.dumps({"fold": folds[0], "completed": len(outputs), "total": len(selected),
                                  "object_id": row["object_id"], "variant": row["variant"], "status": payload["status"]}), flush=True)
        del models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    receipt = {"schema": "delphi.k3-generalization-development-prediction-receipt.v1",
               "role": "post_hoc_development_only", "diagnostic_manifest_sha256": sha256(args.manifest),
               "model_manifest_sha256": sha256(args.model_manifest), "splits_sha256": sha256(args.splits),
               "runner_sha256": sha256(Path(__file__)), "device": args.device, "chunk_size": args.chunk_size,
               "elapsed_seconds": time.monotonic() - start, "predictions": outputs,
               "references_opened_by_prediction": False}
    write_new_json(args.output_directory / "prediction-receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--manifest", type=Path, required=True)
    prepare.add_argument("--spec", type=Path, required=True)
    prepare.add_argument("--output-directory", type=Path, required=True)
    run = commands.add_parser("predict")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--splits", type=Path, required=True)
    run.add_argument("--model-directory", type=Path, required=True)
    run.add_argument("--model-manifest", type=Path, required=True)
    run.add_argument("--output-directory", type=Path, required=True)
    run.add_argument("--device", default="cuda")
    run.add_argument("--chunk-size", type=int, default=256)
    run.add_argument("--variant", action="append")
    args = parser.parse_args()
    try:
        result = prepare_diagnostics(args.manifest, args.spec, args.output_directory) if args.command == "prepare" else predict(args)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({key: value for key, value in result.items() if key not in ("records", "predictions")}), flush=True)


if __name__ == "__main__":
    main()
