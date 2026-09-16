#!/usr/bin/env python3
"""Run sealed, label-free frozen-ensemble predictions for ALCDEF inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from lc_pipeline.k3.bundle import _model_inputs
from lc_pipeline.k3.evaluation import ensemble_score_grids, modes_from_score_grid
from lc_pipeline.k3.inference import refine_ensemble_axes, score_axial_grid
from lc_pipeline.k3.ztf_prediction import _load_models, _load_prepared


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_new(path: Path, payload: object) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def predict(prepared_directory: Path, model_directory: Path, model_manifest: Path, output: Path, *, device: str) -> dict[str, object]:
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    try:
        prepared_manifest = json.loads((prepared_directory / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read prepared directory manifest: {exc}") from exc
    if prepared_manifest.get("schema") != "delphi.k3-alcdef-prepared-directory.v1" or not isinstance(prepared_manifest.get("objects"), list):
        raise ValueError("prepared directory schema mismatch")
    rows = prepared_manifest["objects"]
    if not rows or len({row.get("object_id") for row in rows if isinstance(row, dict)}) != len(rows):
        raise ValueError("prepared directory identities invalid")
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    models, names, bindings = _load_models(model_directory, model_manifest, tuple(range(5)), str(target))
    if len(models) != 25:
        raise ValueError("requires the complete 25-model ensemble")
    output.mkdir(parents=True)
    result_rows = []
    for row in sorted(rows, key=lambda value: str(value["object_id"])):
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise ValueError("prepared manifest row is invalid")
        source = prepared_directory / str(row["path"])
        if _sha256(source) != row.get("sha256"):
            raise ValueError(f"prepared-object hash mismatch: {row.get('object_id')}")
        object_id, period, epochs = _load_prepared(source)
        inputs = _model_inputs(epochs, period, target)
        with torch.inference_mode():
            grids = np.stack([score_axial_grid(model, inputs, chunk_size=1024) for model in models])
            mean = ensemble_score_grids(grids[:, None, :])[0]
            modes = modes_from_score_grid(mean)
        # Projected refinement intentionally differentiates scores with
        # respect to the three selected candidate axes.
        with torch.enable_grad():
            axes, scores = refine_ensemble_axes(models, inputs, np.asarray([mode.axis_xyz for mode in modes]))
        payload = {"schema": "delphi.k3-alcdef-transfer-prediction.v1", "object_id": object_id, "policy": "all_25_frozen_models_for_new_identity", "model_count": len(models), "checkpoint_names": names, "checkpoint_bindings": bindings, "model_manifest_sha256": _sha256(model_manifest), "prepared_sha256": _sha256(source), "period_provenance": period.provenance, "axes": [{"axis_xyz": axis.tolist(), "score": float(score), "grid_index": int(mode.grid_index)} for axis, score, mode in zip(axes, scores, modes, strict=True)]}
        destination = output / "objects" / f"{object_id.replace(':', '_')}.json"
        destination.parent.mkdir(exist_ok=True)
        _write_new(destination, payload)
        result_rows.append({"object_id": object_id, "path": destination.relative_to(output).as_posix(), "sha256": _sha256(destination)})
    manifest = {"schema": "delphi.k3-alcdef-transfer-prediction-directory.v1", "prepared_manifest_sha256": _sha256(prepared_directory / "manifest.json"), "model_manifest_sha256": _sha256(model_manifest), "device": str(target), "object_count": len(result_rows), "objects": result_rows}
    _write_new(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-directory", type=Path, required=True)
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    try:
        result = predict(args.prepared_directory, args.model_directory, args.model_manifest, args.output, device=args.device)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps({"object_count": result["object_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
