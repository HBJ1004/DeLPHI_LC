import json

import pytest

from lc_pipeline.k3.generalization_diagnostics import VARIANTS, sha256, write_new_json
from repro.run_k3_generalization_diagnostics import validate_prediction_manifest


def _fixture(tmp_path):
    splits = tmp_path / "splits.json"
    write_new_json(splits, {"folds": [{"fold": 0, "test_ids": ["asteroid_101", "asteroid_102"]}]})
    path = tmp_path / "manifest.json"
    records = []
    for object_id in ("asteroid_101", "asteroid_102"):
        for variant in VARIANTS:
            if object_id == "asteroid_102":
                records.append({"object_id": object_id, "variant": variant, "status": "input_unavailable",
                                "path": None, "sha256": None})
                continue
            target = tmp_path / variant / f"{object_id}.json"
            write_new_json(target, {
                "object_id": object_id, "held_out_fold": 0,
                "identity_binding": {"schema": "delphi.k3-survey-identity-binding.v1", "object_id": object_id,
                                     "damit_id": 101, "mpc_number": 2, "resolved_mpc_number": 2,
                                     "held_out_fold": 0, "identity_map_sha256": "a" * 64, "identity_table_sha256": "c" * 64},
                "generalization_transform": {"variant": variant, "study_spec_sha256": "b" * 64},
            })
            records.append({"object_id": object_id, "variant": variant, "status": "ready",
                            "path": str(target.relative_to(tmp_path)), "sha256": sha256(target)})
    write_new_json(path, {"schema": "delphi.k3-generalization-diagnostics.v1", "role": "post_hoc_development_only",
                          "object_count": 2, "variants": list(VARIANTS), "records": records,
                          "identity_map_sha256": "a" * 64, "study_spec_sha256": "b" * 64})
    return path, splits


def test_full_missing_denominator_accepted(tmp_path):
    manifest, splits = _fixture(tmp_path)
    assert len(validate_prediction_manifest(manifest, splits)["records"]) == 16


def test_truncated_and_duplicate_denominators_rejected(tmp_path):
    manifest, splits = _fixture(tmp_path)
    value = json.loads(manifest.read_text())
    value["records"].pop()
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="every object"):
        validate_prediction_manifest(manifest, splits)
    value["records"].append(value["records"][0])
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="every object"):
        validate_prediction_manifest(manifest, splits)


def test_crosslinked_object_even_with_matching_file_hash_is_rejected(tmp_path):
    manifest, splits = _fixture(tmp_path)
    value = json.loads(manifest.read_text())
    row = value["records"][0]
    target = tmp_path / row["path"]
    prepared = json.loads(target.read_text())
    prepared["object_id"] = "asteroid_102"
    prepared["identity_binding"].update(object_id="asteroid_102", damit_id=102, mpc_number=3, resolved_mpc_number=3)
    target.write_text(json.dumps(prepared))
    row["sha256"] = sha256(target)
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="prepared identity"):
        validate_prediction_manifest(manifest, splits)
