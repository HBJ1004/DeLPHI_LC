import json

from repro.derive_k3_model_specific_exposure import derive


def test_model_specific_audit_separates_current_training_from_history(tmp_path):
    inventory = tmp_path / "candidates.json"
    inventory.write_text(json.dumps({"schema": "delphi.k3-alcdef-gaia-eligibility.v1", "candidates": [
        {"object_id": "mpc:1", "exposure": {"status": "exposed", "reasons": ["present_in:original_170_publication_splits"]}},
        {"object_id": "mpc:2", "exposure": {"status": "unknown", "reasons": ["unresolved_exposure_history_prevents_clearance"]}},
        {"object_id": "mpc:3", "exposure": {"status": "exposed", "reasons": ["present_in:historical_phase30_metadata"]}},
    ]}))
    result = derive(inventory)
    assert result["counts"] == {"model_training_exposed": 1, "model_training_unexposed": 2}
    assert result["objects"][2]["status"] == "model_training_unexposed"
    assert "prospective external validation" in result["forbidden_descriptions"]
