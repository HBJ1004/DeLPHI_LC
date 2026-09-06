"""Portable ensemble metadata, tensor integrity, and single-object inference."""

from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pytest
from safetensors.torch import save_file

from lc_pipeline.k3 import bundle
from lc_pipeline.k3.cli import _smoke_geometry
from lc_pipeline.k3.config import K3ScoreModelConfig
from lc_pipeline.k3.model import CandidateConditionedScorer
from lc_pipeline.v2.preprocessing import KnownPeriod
from repro.train_k3_custom import export_inference_bundle


@pytest.fixture
def ensemble(tmp_path):
    splits = bundle.repository_root() / "repro/data/damit-20250610T000301Z/publication-splits-v2.3.json"
    roles = json.loads(splits.read_text())["folds"][0]
    config = K3ScoreModelConfig()
    state = CandidateConditionedScorer(config).state_dict()
    members = []
    for seed in bundle.SEEDS:
        path = tmp_path / f"seed-{seed}.safetensors"
        save_file(state, str(path))
        members.append({"seed": seed, "file": path.name, "sha256": bundle.sha256(path)})
    metadata = {"schema": "delphi.k3-ensemble-bundle.v1", "bundle_id": "test",
                "protocol_sha256": bundle.K3_PROTOCOL_SHA256,
                "tokenizer_sha256": bundle.K3_TOKENIZER_SCHEMA_SHA256,
                "splits_sha256": bundle.sha256(splits), "fold": 0,
                "object_roles": {key: value for key, value in roles.items() if key.endswith("_ids")},
                "members": members, "model_config": asdict(config),
                "calibration": {"cone90_deg": 40., "cone95_deg": 90.}}
    (tmp_path / "bundle.json").write_text(json.dumps(metadata))
    return tmp_path, metadata


def test_single_object_score_map_and_unseen_prediction(ensemble, monkeypatch):
    root, metadata = ensemble
    predictor = bundle.K3EnsemblePredictor(root)
    monkeypatch.setattr(bundle, "score_axial_grid", lambda *args, **kwargs: np.linspace(0, 1, 6144))
    monkeypatch.setattr(bundle, "refine_ensemble_axes", lambda models, inputs, axes: (axes, np.ones(3)))
    result = predictor.predict(_smoke_geometry(), known_period=KnownPeriod(6., "test"),
                               object_id=metadata["object_roles"]["test_ids"][0])
    assert len(result["axes"]) == 3
    assert result["risk_deg"] is None
    assert result["period_provenance"] == "test"
    assert np.allclose(np.linalg.norm([row["axis_xyz"] for row in result["axes"]], axis=1), 1)


def test_known_training_object_is_rejected(ensemble):
    root, metadata = ensemble
    predictor = bundle.K3EnsemblePredictor(root)
    with pytest.raises(ValueError, match="held-out"):
        predictor.predict(_smoke_geometry(), known_period=KnownPeriod(6., "test"),
                          object_id=metadata["object_roles"]["train_ids"][0])


@pytest.fixture
def custom_bundle(tmp_path):
    config = K3ScoreModelConfig()
    model = CandidateConditionedScorer(config)
    checkpoint = tmp_path / "custom-model.pt"
    torch_data = {
        "schema": "delphi.k3-checkpoint.v2",
        "stage": "custom",
        "model_config": config.as_mapping(),
        "model_state_dict": model.state_dict(),
    }
    import torch

    torch.save(torch_data, checkpoint)
    root = tmp_path / "inference-bundle"
    export_inference_bundle(
        checkpoint,
        root,
        bundle_id="custom-test",
        training_report_sha256="0" * 64,
    )
    return root


def test_custom_bundle_predicts_without_calibration(custom_bundle, monkeypatch):
    predictor = bundle.load_predictor(custom_bundle)
    monkeypatch.setattr(bundle, "score_axial_grid", lambda *args, **kwargs: np.linspace(0, 1, 6144))
    monkeypatch.setattr(bundle, "refine_axes", lambda model, inputs, axes: (axes, np.ones(3)))
    result = predictor.predict(
        _smoke_geometry(), known_period=KnownPeriod(6.0, "test"), object_id="new-object"
    )
    assert result["schema"] == "delphi.k3-custom-prediction.v1"
    assert result["calibration"] is None
    assert result["risk_deg"] is None
    assert result["candidate_semantics"] == "unordered axial set"
    assert "do not use" in result["score_semantics"]


def test_tampered_custom_bundle_is_rejected(custom_bundle):
    data = json.loads((custom_bundle / "bundle.json").read_text())
    data["member"]["sha256"] = "0" * 64
    (custom_bundle / "bundle.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="checksum"):
        bundle.load_predictor(custom_bundle)


@pytest.mark.parametrize("mutation", ["seed", "protocol", "split", "roles", "checksum", "path"])
def test_tampered_bundle_is_rejected(ensemble, mutation):
    root, data = ensemble
    if mutation == "seed":
        data["members"][0]["seed"] = 42
    elif mutation == "protocol":
        data["protocol_sha256"] = "0" * 64
    elif mutation == "split":
        data["splits_sha256"] = "0" * 64
    elif mutation == "roles":
        data["object_roles"]["test_ids"][0] = "asteroid_unknown"
    elif mutation == "checksum":
        data["members"][0]["sha256"] = "0" * 64
    else:
        data["members"][0]["file"] = "../outside.safetensors"
    (root / "bundle.json").write_text(json.dumps(data))
    with pytest.raises(ValueError):
        bundle.K3EnsemblePredictor(root)
