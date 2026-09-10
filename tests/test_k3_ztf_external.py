"""Offline ZTF conversion and leakage-safe external prediction contracts."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from lc_pipeline.k3.ztf_external import ZTFExternalError, fink_rows_to_epoch, prepared_ztf_object
from lc_pipeline.k3.ztf_prediction import (
    ZTFPredictionError,
    _fold_policy,
    predict_prepared_ztf_object,
)


def _cache(rows):
    metadata = {"target": "test", "center": "500@399", "frame": "ecliptic J2000"}

    def digest(value):
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    return {"source": "JPL Horizons", "query_metadata": metadata, "rows": rows,
            "query_metadata_sha256": digest(metadata), "rows_sha256": digest(rows)}


def _rows():
    return [{"jd": 2450000.0, "fid": 2, "magpsf": 18.0, "sigmapsf": .1, "Dhelio": 1., "Dobs": 1., "phase": 90.},
            {"jd": 2450000.1, "fid": 1, "magpsf": 18.0, "sigmapsf": .1, "Dhelio": 1., "Dobs": 1., "phase": 90.},
            {"jd": 2450000.2, "fid": 2, "magpsf": 19.0, "sigmapsf": .3, "Dhelio": 1., "Dobs": 1., "phase": 90.}]


def test_fink_conversion_is_rband_only_positive_and_hash_bound():
    cache = _cache([{"jd": 2450000.0, "asteroid_to_sun_ecliptic_j2000_au": [1, 0, 0], "asteroid_to_earth_ecliptic_j2000_au": [0, 1, 0]}])
    epoch = fink_rows_to_epoch("ztf-1", _rows(), cache)
    assert epoch.epoch_id == "ztf-r-ztf-1" and len(epoch.observations) == 1
    assert epoch.observations[0].relative_brightness == 1.0
    assert epoch.observations[0].measured_error > 0


@pytest.mark.parametrize("vector,error", [
    ([0, -1, 0], "RA/DEC"), ([2, 0, 0], "norm"),
])
def test_fink_conversion_rejects_wrong_cached_vector_sign_or_norm(vector, error):
    rows = [{"jd": 2450000.0, "asteroid_to_sun_ecliptic_j2000_au": [1, 0, 0], "asteroid_to_earth_ecliptic_j2000_au": vector}]
    source_rows = _rows()
    if error == "RA/DEC":
        source_rows[0].update({"ra": 270., "dec": 0.})
    with pytest.raises(ZTFExternalError, match=error):
        fink_rows_to_epoch("ztf-1", source_rows, _cache(rows))


def test_fink_conversion_rejects_bad_cache_hash_and_missing_vector():
    cache = _cache([])
    cache["rows"] = [{"jd": 2450000.0, "asteroid_to_sun_ecliptic_j2000_au": [1, 0, 0], "asteroid_to_earth_ecliptic_j2000_au": [0, 1, 0]}]
    with pytest.raises(ZTFExternalError, match="hash mismatch"):
        fink_rows_to_epoch("ztf-1", _rows(), cache)


def _splits(tmp_path):
    path = tmp_path / "splits.json"
    path.write_text(json.dumps({"folds": [{"fold": fold, "train_ids": [f"train-{fold}"], "validation_ids": [], "calibration_ids": [], "test_ids": [f"known-{fold}"]} for fold in range(5)]}))
    return path


def test_external_fold_policy_isolated_and_new_ids_use_all_folds(tmp_path):
    split = _splits(tmp_path)
    assert _fold_policy("known-3", split, "existing_identity") == (3,)
    assert _fold_policy("new", split, "post_cutoff_damit_record") == (0, 1, 2, 3, 4)
    with pytest.raises(ZTFPredictionError, match="overlap"):
        _fold_policy("known-3", split, "post_cutoff_damit_record")
    with pytest.raises(ZTFPredictionError, match="held-out"):
        _fold_policy("train-3", split, "existing_identity")


def test_prediction_is_label_blind_and_uses_policy_model_count(tmp_path, monkeypatch):
    cache = _cache([{"jd": 2450000.0, "asteroid_to_sun_ecliptic_j2000_au": [1, 0, 0], "asteroid_to_earth_ecliptic_j2000_au": [0, 1, 0]}])
    prepared = tmp_path / "prepared.json"
    prepared.write_text(
        json.dumps(
            prepared_ztf_object(
                "known-2",
                _rows(),
                cache,
                known_period_hours=7.0,
                period_provenance="external",
            )
        )
    )
    monkeypatch.setattr(
        "lc_pipeline.k3.ztf_prediction._load_models",
        lambda root, manifest, folds, device: (
            [object()] * (5 * len(folds)),
            [f"m{fold}" for fold in folds],
            [{"fold": fold} for fold in folds for _ in range(5)],
        ),
    )
    monkeypatch.setattr("lc_pipeline.k3.ztf_prediction._model_inputs", lambda *args: {})
    monkeypatch.setattr("lc_pipeline.k3.ztf_prediction.score_axial_grid", lambda *args, **kwargs: np.arange(6144, dtype=float))
    monkeypatch.setattr("lc_pipeline.k3.ztf_prediction.refine_ensemble_axes", lambda models, inputs, starts: (starts, np.ones(3)))
    output = tmp_path / "prediction.json"
    model_manifest = tmp_path / "model-manifest.json"
    model_manifest.write_text("{}")
    result = predict_prepared_ztf_object(
        prepared,
        tmp_path / "models",
        _splits(tmp_path),
        output,
        model_manifest_path=model_manifest,
        policy="existing_identity",
    )
    assert result["model_count"] == 5 and "reference" not in output.read_text().lower()
