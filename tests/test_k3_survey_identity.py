import json

import pytest

from lc_pipeline.k3.survey_identity import build_identity_map, validate_survey_binding
from lc_pipeline.k3.ztf_prediction import _load_prepared


def test_internal_database_id_is_not_mpc_number(tmp_path):
    table = tmp_path / "asteroids.csv"
    table.write_text('\ufeff"id","number","name","designation"\n"101","2","Pallas",""\n"999","101","Helena",""\n')
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"folds": [{"fold": 2, "test_ids": ["asteroid_101"]}]}))
    mapping = build_identity_map(table, splits)
    assert mapping["objects"][0]["mpc_number"] == 2
    assert mapping["objects"][0]["damit_id"] == 101


def test_binding_rejects_wrong_resolved_number():
    value = {"object_id": "asteroid_101", "held_out_fold": 2, "identity_binding": {
        "schema": "delphi.k3-survey-identity-binding.v1", "object_id": "asteroid_101",
        "damit_id": 101, "mpc_number": 2, "resolved_mpc_number": 101, "held_out_fold": 2,
        "identity_map_sha256": "a" * 64, "identity_table_sha256": "b" * 64,
    }}
    with pytest.raises(ValueError, match="differs from mapped"):
        validate_survey_binding(value)
    value["identity_binding"]["resolved_mpc_number"] = 2
    validate_survey_binding(value)
    del value["held_out_fold"]
    del value["identity_binding"]["held_out_fold"]
    with pytest.raises(ValueError, match="explicit valid held-out fold"):
        validate_survey_binding(value)


def test_legacy_ztf_input_rejected_before_model_loading(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"schema": "delphi.k3-ztf-prepared.v1", "object_id": "asteroid_101"}))
    with pytest.raises(ValueError, match="identity binding"):
        _load_prepared(path)
