"""Label-blind external K3 prediction and separately invoked reference scoring."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal, Mapping, Sequence

import numpy as np
import torch

from ..physics.axial import axial_angular_error_deg
from ..v2.preprocessing import KnownPeriod, Observation, ObservationEpoch
from .bundle import _model_inputs
from .config import K3ScoreModelConfig
from .evaluation import ensemble_score_grids, modes_from_score_grid
from .inference import refine_ensemble_axes, score_axial_grid
from .model import CandidateConditionedScorer
from .publication_run import K3_OOF_SEEDS


class ZTFPredictionError(ValueError):
    """Raised when external prediction policy would leak labels or folds."""


def _load_prepared(path: str | Path) -> tuple[str, KnownPeriod, tuple[ObservationEpoch, ...]]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if value.get("schema") not in {
            "delphi.k3-ztf-prepared.v1",
            "delphi.k3-external-prepared.v1",
        }:
            raise ZTFPredictionError("prepared object schema is invalid")
        period = KnownPeriod(float(value["known_period_hours"]), str(value["period_provenance"]))
        epochs = tuple(ObservationEpoch(row["epoch_id"], tuple(Observation(**item) for item in row["observations"])) for row in value["epochs"])
        object_id = value["object_id"]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ZTFPredictionError(f"cannot load frozen prepared object: {exc}") from exc
    if not isinstance(object_id, str) or not object_id.strip():
        raise ZTFPredictionError("prepared object requires an object ID")
    return object_id, period, epochs


def _fold_policy(object_id: str, split_path: str | Path, policy: Literal["existing_identity", "strict_temporal_new"]) -> tuple[int, ...]:
    try:
        folds = sorted(json.loads(Path(split_path).read_text())["folds"], key=lambda row: int(row["fold"]))
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ZTFPredictionError(f"cannot load frozen splits: {exc}") from exc
    if [int(row["fold"]) for row in folds] != list(range(5)):
        raise ZTFPredictionError("external prediction requires frozen folds zero through four")
    test_folds = [int(row["fold"]) for row in folds if object_id in row.get("test_ids", [])]
    any_ids = {identifier for row in folds for role in ("train_ids", "validation_ids", "calibration_ids", "test_ids") for identifier in row.get(role, [])}
    if policy == "existing_identity":
        if len(test_folds) != 1:
            raise ZTFPredictionError("existing_identity requires membership in exactly one held-out test fold")
        return (test_folds[0],)
    if policy == "strict_temporal_new":
        if object_id in any_ids:
            raise ZTFPredictionError("strict_temporal_new rejects any frozen-split identity overlap")
        return tuple(range(5))
    raise ZTFPredictionError("prediction policy must be existing_identity or strict_temporal_new")


def _load_models(model_directory: str | Path, folds: Sequence[int], device: str) -> tuple[list[CandidateConditionedScorer], list[str]]:
    models: list[CandidateConditionedScorer] = []
    names: list[str] = []
    config: K3ScoreModelConfig | None = None
    root = Path(model_directory)
    for fold in folds:
        for seed in K3_OOF_SEEDS:
            path = root / f"real-fold-{fold}-seed-{seed}.pt"
            try:
                checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                if checkpoint.get("stage") != "real-oof" or int(checkpoint.get("config", {}).get("seed", -1)) != seed:
                    raise ZTFPredictionError(f"checkpoint policy mismatch: {path.name}")
                current = K3ScoreModelConfig(**checkpoint["model_config"])
                if config is not None and current != config:
                    raise ZTFPredictionError("external ensemble checkpoints have incompatible configurations")
                config = current
                model = CandidateConditionedScorer(current)
                model.load_state_dict(checkpoint["model_state_dict"])
            except ZTFPredictionError:
                raise
            except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
                raise ZTFPredictionError(f"cannot load checkpoint {path.name}: {exc}") from exc
            models.append(model.to(device).eval())
            names.append(path.name)
    return models, names


def predict_prepared_ztf_object(prepared_path: str | Path, model_directory: str | Path, split_path: str | Path, output_path: str | Path, *, policy: Literal["existing_identity", "strict_temporal_new"], device: str = "cpu") -> dict[str, object]:
    """Predict without loading any pole/reference label; output must be new."""
    destination = Path(output_path)
    if destination.exists():
        raise ZTFPredictionError("refusing to overwrite external prediction")
    object_id, period, epochs = _load_prepared(prepared_path)
    folds = _fold_policy(object_id, split_path, policy)
    models, names = _load_models(model_directory, folds, device)
    expected_count = 5 if policy == "existing_identity" else 25
    if len(models) != expected_count:
        raise ZTFPredictionError("model count does not match external prediction policy")
    inputs = _model_inputs(epochs, period, torch.device(device))
    grids = np.stack([score_axial_grid(model, inputs, chunk_size=1024) for model in models])
    mean = ensemble_score_grids(grids[:, None, :])[0]
    modes = modes_from_score_grid(mean)
    axes, scores = refine_ensemble_axes(models, inputs, np.asarray([mode.axis_xyz for mode in modes]))
    payload = {"schema": "delphi.k3-ztf-external-prediction.v1", "object_id": object_id, "policy": policy, "folds": list(folds), "model_count": len(models), "checkpoint_names": names, "prepared_sha256": hashlib.sha256(Path(prepared_path).read_bytes()).hexdigest(), "period_provenance": period.provenance, "axes": [{"axis_xyz": axis.tolist(), "score": float(score), "grid_index": int(mode.grid_index)} for axis, score, mode in zip(axes, scores, modes, strict=True)]}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def score_external_prediction(prediction_path: str | Path, reference_path: str | Path, output_path: str | Path) -> dict[str, object]:
    """Separate scoring command: reference axes are unavailable to prediction."""
    destination = Path(output_path)
    if destination.exists():
        raise ZTFPredictionError("refusing to overwrite external score")
    prediction, references = json.loads(Path(prediction_path).read_text()), json.loads(Path(reference_path).read_text())
    object_id = prediction.get("object_id")
    if prediction.get("schema") != "delphi.k3-ztf-external-prediction.v1" or not isinstance(references, Mapping) or object_id not in references:
        raise ZTFPredictionError("prediction/reference identities do not align")
    targets = np.asarray(references[object_id], dtype=float)
    axes = np.asarray([row["axis_xyz"] for row in prediction["axes"]], dtype=float)
    if targets.ndim != 2 or targets.shape[1] != 3 or axes.shape != (3, 3) or not np.all(np.isfinite(targets)):
        raise ZTFPredictionError("reference axes are invalid")
    error = float(np.min(axial_angular_error_deg(axes[:, None, :], targets[None, :, :])))
    payload = {"schema": "delphi.k3-ztf-external-score.v1", "prediction_sha256": hashlib.sha256(Path(prediction_path).read_bytes()).hexdigest(), "object_id": object_id, "oracle_at_3_error_deg": error}
    destination.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return payload
