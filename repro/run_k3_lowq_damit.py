#!/usr/bin/env python3
"""Run a label-separated frozen-K3 check on low-quality DAMIT records.

This is deliberately an exploratory, same-source evaluation.  It does not
retrain K3, and it does not establish cross-survey performance.  Selection and
input preparation use only DAMIT identity, quality flag, period, and raw
lightcurve bytes.  Reference axes are written separately and opened only by
the scoring command.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from lc_pipeline.k3.bundle import _model_inputs
from lc_pipeline.k3.evaluation import ensemble_score_grids, modes_from_score_grid
from lc_pipeline.k3.generalization_validation import analyze_generalization
from lc_pipeline.k3.inference import refine_ensemble_axes, score_axial_grid
from lc_pipeline.k3.tokenizer import tokenize_epochs
from lc_pipeline.k3.ztf_prediction import _load_models, _load_prepared
from lc_pipeline.v2.catalog import ecliptic_vector
from lc_pipeline.v2.data import parse_damit_lightcurve
from lc_pipeline.v2.preprocessing import KnownPeriod

INPUT_SCHEMA = "delphi.k3-lowq-damit-input-index.v1"
PREPARED_SCHEMA = "delphi.k3-lowq-damit-prepared-directory.v1"
REFERENCE_SCHEMA = "delphi.k3-lowq-damit-reference-cohort.v1"
PREDICTION_SCHEMA = "delphi.k3-lowq-damit-prediction-directory.v1"
SELECTION_SALT = "delphi-k3-lowq-damit-20260917-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _quality(value: str) -> int:
    number = float(value)
    if not math.isfinite(number) or number not in (1.0, 2.0):
        raise ValueError(f"invalid low-quality flag: {value!r}")
    return int(number)


def _rank(identity: str) -> str:
    return hashlib.sha256(f"{SELECTION_SALT}:{identity}".encode()).hexdigest()


def prepare(snapshot: Path, output: Path, *, q1_sample_size: int) -> dict[str, object]:
    """Select and prepare a label-blind low-Q cohort, with references separate."""
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    if q1_sample_size <= 0:
        raise ValueError("q1 sample size must be positive")
    asteroids = _read_csv(snapshot / "tables" / "asteroids.csv")
    models = _read_csv(snapshot / "tables" / "asteroid_models.csv")
    asteroid_by_id = {row["id"]: row for row in asteroids}
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in models:
        grouped[row["asteroid_id"]].append(row)

    candidates: dict[int, list[dict[str, object]]] = {1: [], 2: []}
    for asteroid_id, rows in grouped.items():
        # Any quality-3+ record makes this identity part of the high-quality
        # DAMIT population, so it is never treated as a new low-Q object.
        try:
            qualities = [int(float(row["quality_flag"])) for row in rows]
        except (KeyError, ValueError):
            continue
        if any(q >= 3 for q in qualities):
            continue
        highest = max(qualities)
        if highest not in (1, 2):
            continue
        selected_models = sorted(
            [row for row, q in zip(rows, qualities, strict=True) if q == highest],
            key=lambda row: int(row["id"]),
        )
        if not selected_models:
            continue
        try:
            period = float(selected_models[0]["period"])
        except (KeyError, ValueError):
            continue
        if not math.isfinite(period) or period <= 0:
            continue
        lightcurve = snapshot / "files" / f"asteroid_{asteroid_id}" / "lc.txt"
        if not lightcurve.is_file():
            continue
        identity = f"damit:{asteroid_id}"
        candidates[highest].append(
            {
                "object_id": identity,
                "damit_asteroid_id": int(asteroid_id),
                "damit_number": asteroid_by_id.get(asteroid_id, {}).get("number"),
                "quality_flag": highest,
                "known_period_hours": period,
                "period_model_id": int(selected_models[0]["id"]),
                "lightcurve_path": lightcurve,
                "models": selected_models,
                "selection_rank": _rank(identity),
            }
        )
    candidates[1].sort(key=lambda item: str(item["selection_rank"]))
    candidates[2].sort(key=lambda item: int(item["damit_asteroid_id"]))
    def valid_model_input(item: dict[str, object]) -> bool:
        """Apply the frozen tokenizer's label-free input contract before selection."""
        try:
            epochs = parse_damit_lightcurve(Path(item["lightcurve_path"]))
            tokenize_epochs(
                epochs,
                known_period=KnownPeriod(
                    float(item["known_period_hours"]), "DAMIT low-quality model-table period"
                ),
            )
        except (TypeError, ValueError):
            return False
        return True

    q2_selected = [item for item in candidates[2] if valid_model_input(item)]
    q1_selected: list[dict[str, object]] = []
    for item in candidates[1]:
        if valid_model_input(item):
            q1_selected.append(item)
        if len(q1_selected) == q1_sample_size:
            break
    selected = q2_selected + q1_selected
    if len(q1_selected) < q1_sample_size or not selected:
        raise ValueError("insufficient usable low-quality DAMIT objects")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        input_rows: list[dict[str, object]] = []
        reference_rows: list[dict[str, object]] = []
        prepared_rows: list[dict[str, object]] = []
        for item in sorted(selected, key=lambda value: str(value["object_id"])):
            object_id = str(item["object_id"])
            lightcurve = Path(item["lightcurve_path"])
            try:
                epochs = parse_damit_lightcurve(lightcurve)
            except ValueError as exc:
                raise ValueError(f"cannot prepare {object_id}: {exc}") from exc
            lightcurve_hash = _sha256(lightcurve)
            input_row = {
                "object_id": object_id,
                "damit_asteroid_id": item["damit_asteroid_id"],
                "damit_number": item["damit_number"],
                "quality_flag": item["quality_flag"],
                "known_period_hours": item["known_period_hours"],
                "period_model_id": item["period_model_id"],
                "period_provenance": "DAMIT low-quality model-table period; axis columns excluded from input preparation",
                "lightcurve_path": lightcurve.relative_to(snapshot).as_posix(),
                "lightcurve_sha256": lightcurve_hash,
                "selection_rank": item["selection_rank"],
            }
            input_rows.append(input_row)
            payload = {
                "schema": "delphi.k3-external-prepared.v1",
                "object_id": object_id,
                "external_role": "exploratory_same_source_low_quality_damit",
                "known_period_hours": item["known_period_hours"],
                "period_provenance": input_row["period_provenance"],
                "source": {
                    "repository": "DAMIT",
                    "quality_flag": item["quality_flag"],
                    "lightcurve_sha256": lightcurve_hash,
                },
                "epochs": [
                    {
                        "epoch_id": epoch.epoch_id,
                        "observations": [
                            {
                                "time_jd": obs.time_jd,
                                "relative_brightness": obs.relative_brightness,
                                "sun_asteroid_ecliptic_j2000_au": list(obs.sun_asteroid_ecliptic_j2000_au),
                                "observer_asteroid_ecliptic_j2000_au": list(obs.observer_asteroid_ecliptic_j2000_au),
                                "measured_error": obs.measured_error,
                            }
                            for obs in epoch.observations
                        ],
                    }
                    for epoch in epochs
                ],
            }
            serial = json.dumps(payload, sort_keys=True).lower()
            if any(token in serial for token in ('"lambda"', '"beta"', '"axis"', '"spin"', '"solution"')):
                raise ValueError(f"reference labels leaked into prepared object {object_id}")
            prepared_path = staging / "prepared" / "objects" / f"{object_id.replace(':', '_')}.json"
            _write_json(prepared_path, payload)
            prepared_rows.append({
                "object_id": object_id,
                "path": prepared_path.relative_to(staging / "prepared").as_posix(),
                "sha256": _sha256(prepared_path),
                "epoch_count": len(epochs),
                "observation_count": sum(len(epoch.observations) for epoch in epochs),
            })
            refs = []
            for model in item["models"]:
                try:
                    axis = ecliptic_vector(float(model["lambda"]), float(model["beta"]))
                except (KeyError, ValueError) as exc:
                    raise ValueError(f"invalid reference axis for {object_id}") from exc
                refs.append({"model_id": int(model["id"]), "axis": list(axis)})
            reference_rows.append({
                "object_id": object_id,
                "eligible": True,
                "source_id": "DAMIT_same_source_low_quality",
                "quality_flag": item["quality_flag"],
                "references": refs,
            })
        input_index = {
            "schema": INPUT_SCHEMA,
            "study_scope": "exploratory_same_source_low_quality_DAMIT_not_external_validation",
            "selection_uses_reference_axes": False,
            "q1_selection": {"method": "sha256_sort", "salt": SELECTION_SALT, "sample_size": q1_sample_size},
            "objects": input_rows,
        }
        _write_json(staging / "input-index.json", input_index)
        prepared_manifest = {
            "schema": PREPARED_SCHEMA,
            "input_index_sha256": _sha256(staging / "input-index.json"),
            "reference_directory_access_required": False,
            "object_count": len(prepared_rows),
            "objects": prepared_rows,
        }
        _write_json(staging / "prepared" / "manifest.json", prepared_manifest)
        references = {
            "schema": REFERENCE_SCHEMA,
            "input_index_sha256": _sha256(staging / "input-index.json"),
            "selection_uses_reference_axes": False,
            "objects": reference_rows,
        }
        _write_json(staging / "reference-cohort.json", references)
        os.replace(staging, output)
        return {"object_count": len(selected), "q1": len(q1_selected), "q2": len(q2_selected)}
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def predict(prepared: Path, model_directory: Path, model_manifest: Path, output: Path, *, device: str) -> dict[str, object]:
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != PREPARED_SCHEMA or not isinstance(manifest.get("objects"), list):
        raise ValueError("prepared manifest schema mismatch")
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    models, names, bindings = _load_models(model_directory, model_manifest, tuple(range(5)), str(target))
    output.mkdir(parents=True)
    rows = []
    for index, row in enumerate(sorted(manifest["objects"], key=lambda value: str(value["object_id"])), start=1):
        source = prepared / str(row["path"])
        if _sha256(source) != row.get("sha256"):
            raise ValueError(f"prepared object hash mismatch for {row.get('object_id')}")
        object_id, period, epochs = _load_prepared(source)
        inputs = _model_inputs(epochs, period, target)
        with torch.inference_mode():
            grids = np.stack([score_axial_grid(model, inputs, chunk_size=1024) for model in models])
            mean = ensemble_score_grids(grids[:, None, :])[0]
            modes = modes_from_score_grid(mean)
        with torch.enable_grad():
            axes, scores = refine_ensemble_axes(models, inputs, np.asarray([mode.axis_xyz for mode in modes]))
        item = {
            "schema": "delphi.k3-lowq-damit-prediction.v1",
            "object_id": object_id,
            "status": "ok",
            "policy": "all_25_frozen_models_no_training",
            "model_count": len(models),
            "checkpoint_names": names,
            "checkpoint_bindings": bindings,
            "model_manifest_sha256": _sha256(model_manifest),
            "prepared_sha256": _sha256(source),
            "period_provenance": period.provenance,
            "axes": [{"axis_xyz": axis.tolist(), "score": float(score), "grid_index": int(mode.grid_index)} for axis, score, mode in zip(axes, scores, modes, strict=True)],
        }
        destination = output / "objects" / f"{object_id.replace(':', '_')}.json"
        _write_json(destination, item)
        rows.append({"object_id": object_id, "path": destination.relative_to(output).as_posix(), "sha256": _sha256(destination)})
        print(f"completed {index}/{len(manifest['objects'])}: {object_id}", flush=True)
    result = {"schema": PREDICTION_SCHEMA, "prepared_manifest_sha256": _sha256(prepared / "manifest.json"), "model_manifest_sha256": _sha256(model_manifest), "device": str(target), "object_count": len(rows), "objects": rows}
    _write_json(output / "manifest.json", result)
    return result


def score(reference: Path, predictions: Path, atlas: Path, output: Path) -> dict[str, object]:
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    refs = json.loads(reference.read_text(encoding="utf-8"))
    pred_manifest = json.loads((predictions / "manifest.json").read_text(encoding="utf-8"))
    if pred_manifest.get("schema") != PREDICTION_SCHEMA:
        raise ValueError("prediction manifest schema mismatch")
    prediction_rows = []
    for row in pred_manifest["objects"]:
        item = json.loads((predictions / row["path"]).read_text(encoding="utf-8"))
        prediction_rows.append({"object_id": item["object_id"], "status": item["status"], "axes": [axis["axis_xyz"] for axis in item["axes"]]})
    atlas_payload = json.loads(atlas.read_text(encoding="utf-8"))
    result = analyze_generalization(cohort=refs, candidate_predictions={"objects": prediction_rows}, atlas_predictions=atlas_payload)
    result["study_scope"] = "exploratory_same_source_low_quality_DAMIT_not_external_validation"
    result["prediction_manifest_sha256"] = _sha256(predictions / "manifest.json")
    result["reference_cohort_sha256"] = _sha256(reference)
    _write_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--snapshot", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--q1-sample-size", type=int, default=143)
    pred = commands.add_parser("predict")
    pred.add_argument("--prepared", type=Path, required=True)
    pred.add_argument("--model-directory", type=Path, required=True)
    pred.add_argument("--model-manifest", type=Path, required=True)
    pred.add_argument("--output", type=Path, required=True)
    pred.add_argument("--device", default="cuda")
    scoring = commands.add_parser("score")
    scoring.add_argument("--reference", type=Path, required=True)
    scoring.add_argument("--predictions", type=Path, required=True)
    scoring.add_argument("--atlas", type=Path, required=True)
    scoring.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            value = prepare(args.snapshot, args.output, q1_sample_size=args.q1_sample_size)
        elif args.command == "predict":
            value = predict(args.prepared, args.model_directory, args.model_manifest, args.output, device=args.device)
        else:
            value = score(args.reference, args.predictions, args.atlas, args.output)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
