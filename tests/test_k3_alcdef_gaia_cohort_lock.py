import json

from repro.lock_k3_alcdef_gaia_cohort import lock


def test_cohort_lock_uses_only_model_training_unexposed_identities(tmp_path):
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps({"schema": "delphi.k3-alcdef-gaia-eligibility.v1", "candidates": [
        {"object_id": "mpc:1", "alcdef": {"valid_point_count": 200}},
        {"object_id": "mpc:2", "alcdef": {"valid_point_count": 300}},
    ]}))
    exposure = tmp_path / "exposure.json"
    exposure.write_text(json.dumps({"schema": "delphi.k3-model-specific-exposure-audit.v1", "objects": [
        {"object_id": "mpc:1", "status": "model_training_exposed"},
        {"object_id": "mpc:2", "status": "model_training_unexposed"},
    ]}))
    result = lock(inventory, exposure, 1)
    assert result["objects"][0]["object_id"] == "mpc:2"
    assert result["scope"]["project_wide_history_unexposed"] is False
