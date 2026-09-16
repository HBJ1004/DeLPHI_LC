import json

import pytest

from lc_pipeline.k3.generalization_diagnostic_scoring import FROZEN_SEEDS, score_diagnostics
from lc_pipeline.k3.generalization_diagnostics import VARIANTS, sha256, write_new_json


def _replace(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(tmp_path):
    splits = tmp_path / "splits.json"
    write_new_json(splits, {"folds": [{"fold": 0, "test_ids": ["asteroid_101", "asteroid_102"]}]})
    mapping = tmp_path / "identity-map.json"
    write_new_json(mapping, {
        "schema": "delphi.k3-survey-identity-map.v1", "splits_sha256": sha256(splits),
        "objects": [{"object_id": f"asteroid_{value}", "damit_id": value,
                     "mpc_number": value - 99, "physical_identity": f"mpc:{value - 99}",
                     "held_out_fold": 0} for value in (101, 102)],
    })
    models = tmp_path / "models.json"
    bindings = [{"filename": f"real-fold-0-seed-{seed}.pt", "fold": 0, "seed": seed,
                 "sha256": "a" * 64} for seed in FROZEN_SEEDS]
    write_new_json(models, {"models": bindings})
    diagnostic = tmp_path / "diagnostic" / "manifest.json"
    receipt = tmp_path / "predictions" / "receipt.json"
    records, predictions = [], []
    for object_id in ("asteroid_101", "asteroid_102"):
        for variant in VARIANTS:
            ready = object_id == "asteroid_101"
            prepared = diagnostic.parent / variant / f"{object_id}.json"
            if ready:
                write_new_json(prepared, {"fixture": True})
            record = {"object_id": object_id, "variant": variant,
                      "status": "ready" if ready else "input_unavailable",
                      "path": str(prepared.relative_to(diagnostic.parent)) if ready else None,
                      "sha256": sha256(prepared) if ready else None}
            records.append(record)
            prediction = receipt.parent / variant / f"{object_id}.json"
            axes = [[0, 1, 0]] * 3 if variant == "distance" else [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]
            write_new_json(prediction, {
                "schema": "delphi.k3-generalization-development-prediction.v1",
                "object_id": object_id, "variant": variant, "folds": [0], "model_count": 5,
                "model_bindings": bindings, "model_manifest_sha256": sha256(models),
                "prepared_sha256": record["sha256"], "candidate_semantics": "unordered_three_axis_set",
                "status": "ok" if ready else "failed", "axes": axes if ready else [],
                "reason": None if ready else "no_photometry",
            })
            predictions.append({"object_id": object_id, "variant": variant,
                                "path": str(prediction.relative_to(receipt.parent)),
                                "sha256": sha256(prediction), "status": "ok" if ready else "failed"})
    write_new_json(diagnostic, {
        "schema": "delphi.k3-generalization-diagnostics.v1", "role": "post_hoc_development_only",
        "identity_map_sha256": sha256(mapping), "object_count": 2,
        "variants": list(VARIANTS), "records": records,
    })
    write_new_json(receipt, {
        "schema": "delphi.k3-generalization-development-prediction-receipt.v1", "role": "post_hoc_development_only",
        "references_opened_by_prediction": False, "diagnostic_manifest_sha256": sha256(diagnostic),
        "splits_sha256": sha256(splits), "model_manifest_sha256": sha256(models), "predictions": predictions,
    })
    catalog = tmp_path / "catalog.jsonl"
    catalog.write_text("\n".join(json.dumps({"object_id": f"asteroid_{value}", "eligible": True,
                                         "solutions": [{"vector": [1, 0, 0]}]}) for value in (101, 102)))
    return {"diagnostic_manifest": diagnostic, "prediction_receipt": receipt, "identity_map": mapping,
            "splits": splits, "model_manifest": models, "reference_catalog": catalog,
            "output": tmp_path / "score.json", "expected_object_count": 2,
            "expected_reference_sha256": sha256(catalog)}


def test_full_factorial_preserves_failures_and_reports_paired_changes(tmp_path):
    result = score_diagnostics(**_fixture(tmp_path))
    assert result["new_identity_external_validation"] is False
    assert result["n_intended_objects"] == 2 and result["n_available_inputs"] == 1
    original = result["variants"]["original"]
    assert original["prediction_failure_count"] == 1
    assert original["all_intended_objects"]["mean_oracle_at_3_error_deg"] == pytest.approx(45)
    assert original["fixed_available_input_cohort"]["mean_oracle_at_3_error_deg"] == pytest.approx(0)
    assert result["variants"]["distance"]["paired_change_vs_original"]["mean_improvement_deg"] == pytest.approx(-45)
    assert result["factorial_average_effects"]["distance"]["mean_improvement_deg"] == pytest.approx(-11.25)


def test_truncated_prediction_receipt_rejected_before_reference_access(tmp_path):
    args = _fixture(tmp_path)
    path = args["prediction_receipt"]
    value = json.loads(path.read_text())
    value["predictions"].pop()
    _replace(path, value)
    args["reference_catalog"] = tmp_path / "must-not-open.jsonl"
    with pytest.raises(ValueError, match="every object in all eight"):
        score_diagnostics(**args)


@pytest.mark.parametrize("field,value", [("folds", [1]), ("object_id", "asteroid_102"), ("prepared_sha256", "b" * 64)])
def test_bound_but_wrong_prediction_identity_or_model_is_rejected(tmp_path, field, value):
    args = _fixture(tmp_path)
    receipt = json.loads(args["prediction_receipt"].read_text())
    row = receipt["predictions"][0]
    path = args["prediction_receipt"].parent / row["path"]
    prediction = json.loads(path.read_text())
    prediction[field] = value
    _replace(path, prediction)
    row["sha256"] = sha256(path)
    _replace(args["prediction_receipt"], receipt)
    with pytest.raises(ValueError, match="prediction identity"):
        score_diagnostics(**args)


def test_prediction_tamper_and_reference_hash_change_rejected(tmp_path):
    args = _fixture(tmp_path)
    receipt = json.loads(args["prediction_receipt"].read_text())
    path = args["prediction_receipt"].parent / receipt["predictions"][0]["path"]
    original = path.read_text()
    path.write_text(original + "\n")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        score_diagnostics(**args)
    path.write_text(original)
    args["expected_reference_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="frozen original catalog hash"):
        score_diagnostics(**args)


def test_report_no_overwrite(tmp_path):
    args = _fixture(tmp_path)
    score_diagnostics(**args)
    before = sha256(args["output"])
    with pytest.raises(ValueError, match="overwrite"):
        score_diagnostics(**args)
    assert sha256(args["output"]) == before


def test_unknown_input_status_is_not_an_unavailable_input(tmp_path):
    args = _fixture(tmp_path)
    manifest = json.loads(args["diagnostic_manifest"].read_text())
    for row in manifest["records"]:
        if row["status"] == "input_unavailable":
            row["status"] = "garbage"
    _replace(args["diagnostic_manifest"], manifest)
    receipt = json.loads(args["prediction_receipt"].read_text())
    receipt["diagnostic_manifest_sha256"] = sha256(args["diagnostic_manifest"])
    _replace(args["prediction_receipt"], receipt)
    with pytest.raises(ValueError, match="unknown diagnostic input status"):
        score_diagnostics(**args)
