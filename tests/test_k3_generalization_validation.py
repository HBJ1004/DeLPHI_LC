from __future__ import annotations

import json
from pathlib import Path

import pytest

from lc_pipeline.k3.generalization_validation import (
    FAILURE_ERROR_DEG,
    GeneralizationValidationError,
    analyze_generalization,
    audit_identity_exposure,
    classify_reference_photometry_overlap,
    create_prediction_receipt,
    create_protocol_lock,
    sha256_file,
    sha256_path,
    validate_external_exposure_clearance,
    validate_prediction_receipt,
    validate_protocol_lock,
)
from repro.analyze_k3_generalization import _candidate_ids


def _axis_rows(*ids: str) -> dict[str, object]:
    return {
        "objects": [
            {
                "object_id": object_id,
                "status": "ok",
                "axes": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            }
            for object_id in ids
        ]
    }


def _cohort(*ids: str) -> dict[str, object]:
    return {
        "objects": [
            {
                "object_id": object_id,
                "eligible": True,
                "references": [[1, 0, 0]],
            }
            for object_id in ids
        ]
    }


def _atlas(*ids: str) -> dict[str, object]:
    return {"train_only": True, **_axis_rows(*ids)}


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def test_exposure_audit_finds_split_and_all_donor_roles() -> None:
    sources = {
        "original_split": {"folds": [{"train_ids": ["asteroid_1"]}]},
        "shape_donors": [{"shape_donor_id": "old-name"}],
        "geometry_donors": {"geometry_donor_ids": ["asteroid_3"]},
        "noise_donors": {"objects": [{"noise_donor_id": "asteroid_4"}]},
        "training_history": {"history_object_ids": ["asteroid_5"]},
    }
    aliases = {
        "official_table_sha256": "f" * 64,
        "aliases": [
            {"alias": "asteroid_1", "canonical_id": "mpc:1"},
            {"alias": "old-name", "canonical_id": "mpc:2"},
            {"alias": "asteroid_3", "canonical_id": "mpc:3"},
            {"alias": "asteroid_4", "canonical_id": "mpc:4"},
            {"alias": "asteroid_5", "canonical_id": "mpc:5"},
            {"alias": "unresolved-name", "status": "unknown"},
        ]
    }
    result = audit_identity_exposure(
        ["mpc:6", "mpc:2", "unresolved-name", "mpc:1"],
        exposure_sources=sources,
        aliases=aliases,
    )
    statuses = {row["object_id"]: row["status"] for row in result["objects"]}
    assert statuses == {
        "mpc:1": "exposed",
        "mpc:2": "exposed",
        "mpc:6": "unexposed",
        "unresolved-name": "unknown",
    }
    assert result["all_candidates_unexposed"] is False


def test_exposure_unknown_source_alias_remains_visible_and_duplicate_rejected() -> None:
    result = audit_identity_exposure(
        ["mpc:999"],
        exposure_sources={"history": {"history_object_id": "mystery"}},
        aliases={"official_table_sha256": "f" * 64, "aliases": {"mystery": None}},
    )
    assert result["unresolved_source_aliases"] == [
        {"object_id": "mystery", "roles": ["history"]}
    ]
    with pytest.raises(GeneralizationValidationError, match="duplicates"):
        audit_identity_exposure(["mpc:9", "mpc:9"], exposure_sources={"split": {}})


def test_empty_exposure_sources_never_certify_unexposed() -> None:
    result = audit_identity_exposure(["mpc:999"], exposure_sources={})
    assert result["objects"][0]["status"] == "unknown"
    assert result["unknown_history_roles"] == ["__no_exposure_sources_declared__"]


def test_incomplete_history_prevents_unexposed_certification() -> None:
    result = audit_identity_exposure(
        ["mpc:999"],
        exposure_sources={
            "known_split": {"object_ids": ["mpc:1"]},
            "historical_projects": {"status": "unknown", "registry_complete": False},
        },
    )
    assert result["all_candidates_unexposed"] is False
    assert result["objects"][0]["status"] == "unknown"
    assert result["unknown_history_roles"] == ["historical_projects"]

    empty = audit_identity_exposure(["mpc:999"], exposure_sources={})
    assert empty["objects"][0]["status"] == "unknown"
    assert empty["unknown_history_roles"] == ["__no_exposure_sources_declared__"]


def test_identity_namespaces_prevent_damit_mpc_numeric_collision() -> None:
    aliases = {
        "official_table_sha256": "e" * 64,
        "aliases": {
            "asteroid_101": "damit:101",
            "damit:101": "mpc:2",
        },
    }
    result = audit_identity_exposure(
        ["mpc:2", "mpc:101"],
        exposure_sources={"original_split": {"test_ids": ["asteroid_101"]}},
        aliases=aliases,
    )
    statuses = {row["object_id"]: row["status"] for row in result["objects"]}
    assert statuses == {"mpc:101": "unexposed", "mpc:2": "exposed"}

    with_actual_ztf_target = audit_identity_exposure(
        ["mpc:2", "mpc:101"],
        exposure_sources={
            "original_split": {"test_ids": ["asteroid_101"]},
            "legacy_ztf": {"observed_mpc_number": 101},
        },
        aliases=aliases,
    )
    assert all(
        row["status"] == "exposed" for row in with_actual_ztf_target["objects"]
    )


def test_external_clearance_requires_exact_conclusive_audit() -> None:
    cohort = _cohort("mpc:10", "provisional:2026-AB")
    audit = audit_identity_exposure(
        ["provisional:2026-AB", "mpc:10"],
        exposure_sources={"history": {"object_ids": ["mpc:20"]}},
    )
    validate_external_exposure_clearance(cohort, audit)
    audit["objects"][0]["status"] = "unknown"
    with pytest.raises(GeneralizationValidationError, match="not conclusively"):
        validate_external_exposure_clearance(cohort, audit)


def test_official_survey_identity_map_is_accepted_as_alias_crosswalk() -> None:
    aliases = {
        "schema": "delphi.k3-survey-identity-map.v1",
        "identity_table_sha256": "d" * 64,
        "objects": [{
            "object_id": "asteroid_101",
            "damit_identity": "damit:101",
            "physical_identity": "mpc:2",
        }],
    }
    result = audit_identity_exposure(
        ["mpc:2", "mpc:101"],
        exposure_sources={"original_split": {"test_ids": ["asteroid_101"]}},
        aliases=aliases,
    )
    statuses = {row["object_id"]: row["status"] for row in result["objects"]}
    assert statuses == {"mpc:101": "unexposed", "mpc:2": "exposed"}


def test_official_identity_map_without_schema_is_accepted() -> None:
    aliases = {
        "identity_table_sha256": "c" * 64,
        "objects": [{
            "object_id": "asteroid_101",
            "damit_id": 101,
            "mpc_number": 2,
        }],
    }
    result = audit_identity_exposure(
        ["mpc:2", "mpc:101"],
        exposure_sources={"original_split": {"test_ids": ["asteroid_101"]}},
        aliases=aliases,
    )
    statuses = {row["object_id"]: row["status"] for row in result["objects"]}
    assert statuses == {"mpc:101": "unexposed", "mpc:2": "exposed"}


def test_numeric_only_official_identity_map_and_shape_model_suffix() -> None:
    aliases = {
        "identity_table_sha256": "c" * 64,
        "objects": [{"object_id": "asteroid_101", "damit_id": 101, "mpc_number": 2}],
    }
    result = audit_identity_exposure(
        ["mpc:2", "mpc:101"],
        exposure_sources={
            "published_shard": {"shape_donor_ids": ["asteroid_101/model_102"]}
        },
        aliases=aliases,
    )
    statuses = {row["object_id"]: row["status"] for row in result["objects"]}
    assert statuses == {"mpc:101": "unexposed", "mpc:2": "exposed"}


def test_source_census_candidate_identity_extraction() -> None:
    document = {
        "reference_availability": {
            "poles_loaded": False,
            "records": [
                {"candidate_identity": "mpc:2"},
                {"candidate_identity": "mpc:101"},
            ],
        }
    }
    assert _candidate_ids(document) == ["mpc:2", "mpc:101"]


def test_reference_overlap_classification_is_conservative() -> None:
    digest_a = "a" * 64
    digest_b = "b" * 64
    overlap = classify_reference_photometry_overlap(
        {
            "object_id": "mpc:1",
            "prediction_photometry_sha256": digest_a,
            "reference_photometry_hashes": [digest_a],
            "independent_support": ["occultation"],
        }
    )
    independent = classify_reference_photometry_overlap(
        {
            "object_id": "mpc:2",
            "prediction_photometry_sha256": digest_a,
            "reference_photometry_sha256": digest_b,
            "overlap_declared": False,
            "independent_support": ["radar"],
        }
    )
    unknown = classify_reference_photometry_overlap(
        {"object_id": "mpc:3", "independent_support": ["claimed"]}
    )
    assert overlap["classification"] == "catalog_agreement"
    assert independent["classification"] == "independent_supported"
    assert unknown["classification"] == "unknown"

    identifier_overlap = classify_reference_photometry_overlap(
        {
            "object_id": "mpc:4",
            "prediction_photometry_ids": ["lc:one", "lc:two"],
            "reference_photometry_ids": ["lc:two", "lc:three"],
            "independent_support": ["occultation"],
        }
    )
    assert identifier_overlap["classification"] == "catalog_agreement"
    assert identifier_overlap["overlap_evidence_identifiers"] == ["lc:two"]

    unequal_files_only = classify_reference_photometry_overlap(
        {
            "object_id": "mpc:5",
            "prediction_photometry_sha256": digest_a,
            "reference_photometry_sha256": digest_b,
            "independent_support": ["radar"],
        }
    )
    assert unequal_files_only["classification"] == "unknown"


def test_protocol_lock_hashes_files_and_directories_and_detects_change(tmp_path: Path) -> None:
    spec = tmp_path / "spec.yaml"
    code = tmp_path / "code.py"
    model = tmp_path / "model.bin"
    data = tmp_path / "data.json"
    refs = tmp_path / "refs"
    refs.mkdir()
    for path, text in ((spec, "spec"), (code, "code"), (model, "model"), (data, "data")):
        path.write_text(text, encoding="utf-8")
    (refs / "one.txt").write_text("reference", encoding="utf-8")
    lock = create_protocol_lock(
        spec_path=spec,
        code_paths=[code],
        model_paths=[model],
        data_paths=[data],
        reference_paths=[refs],
    )
    validate_protocol_lock(lock)
    tree_hash, kind, size = sha256_path(refs)
    assert kind == "directory" and size == len("reference")
    assert lock["resources"]["reference"][0]["sha256"] == tree_hash
    code.write_text("changed", encoding="utf-8")
    with pytest.raises(GeneralizationValidationError, match="changed"):
        validate_protocol_lock(lock)


def test_receipt_binds_lock_and_prediction_bytes(tmp_path: Path) -> None:
    paths = {name: tmp_path / name for name in ("spec", "code", "model", "data", "reference")}
    for name, path in paths.items():
        path.write_text(name, encoding="utf-8")
    lock = create_protocol_lock(
        spec_path=paths["spec"], code_paths=[paths["code"]], model_paths=[paths["model"]],
        data_paths=[paths["data"]], reference_paths=[paths["reference"]],
    )
    lock_path = tmp_path / "lock.json"
    _write(lock_path, lock)
    prediction = tmp_path / "prediction.json"
    _write(prediction, _axis_rows("mpc:9"))
    receipt = create_prediction_receipt(
        lock_path=lock_path, prediction_paths={"candidate": prediction}
    )
    validate_prediction_receipt(
        receipt, lock_path=lock_path, expected_predictions={"candidate": prediction}
    )
    prediction.write_text("{}\n", encoding="utf-8")
    with pytest.raises(GeneralizationValidationError, match="changed"):
        validate_prediction_receipt(
            receipt, lock_path=lock_path, expected_predictions={"candidate": prediction}
        )


def test_analysis_is_order_invariant_and_uses_all_references() -> None:
    cohort = {
        "objects": [
            {"object_id": "mpc:2", "eligible": True, "references": [[0, 0, 1]]},
            {
                "object_id": "mpc:1",
                "eligible": True,
                "references": [[0, 1, 0], [-1, 0, 0]],
                "source_id": "survey-a",
            },
        ]
    }
    candidate = _axis_rows("mpc:1", "mpc:2")
    atlas = _atlas("mpc:1", "mpc:2")
    first = analyze_generalization(
        cohort=cohort,
        candidate_predictions=candidate,
        atlas_predictions=atlas,
        uniform_resamples=25,
    )
    second = analyze_generalization(
        cohort={"objects": list(reversed(cohort["objects"]))},
        candidate_predictions={"objects": list(reversed(candidate["objects"]))},
        atlas_predictions={"train_only": True, "objects": list(reversed(atlas["objects"]))},
        uniform_resamples=25,
    )
    assert first == second
    assert first["primary"]["value"] == pytest.approx(0.0)
    assert first["primary"]["identity_bootstrap_ci95"]["resamples"] == 10_000
    assert "mean_error_mc_ci95_low_deg" not in first["uniform_random_three"]
    assert first["uniform_random_three"]["randomization_interval95_low_deg"] >= 0.0
    assert first["reference_count_strata"].keys() == {"1", "2"}
    assert first["source_strata"].keys() == {"survey-a", "unknown"}
    assert [row["object_id"] for row in first["objects"]] == ["mpc:1", "mpc:2"]


def test_fixed_train_only_atlas_is_supported_but_cannot_include_test_identity() -> None:
    fixed = {
        "train_only": True,
        "axes": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "source_identity_ids": ["mpc:99"],
    }
    result = analyze_generalization(
        cohort=_cohort("mpc:1"),
        candidate_predictions=_axis_rows("mpc:1"),
        atlas_predictions=fixed,
        uniform_resamples=10,
    )
    assert result["paired_train_only_atlas"]["atlas_kind"].startswith("one_fixed")
    fixed["source_identity_ids"] = ["mpc:1"]
    with pytest.raises(GeneralizationValidationError, match="evaluation identities"):
        analyze_generalization(
            cohort=_cohort("mpc:1"),
            candidate_predictions=_axis_rows("mpc:1"),
            atlas_predictions=fixed,
            uniform_resamples=10,
        )


def test_missing_and_invalid_candidate_predictions_score_as_failures() -> None:
    candidate = {
        "objects": [
            {"object_id": "mpc:1", "status": "ok", "axes": [[0, 0, 0]] * 3},
        ]
    }
    result = analyze_generalization(
        cohort=_cohort("mpc:1", "mpc:2"),
        candidate_predictions=candidate,
        atlas_predictions=_atlas("mpc:1", "mpc:2"),
        uniform_resamples=10,
    )
    assert result["eligible_denominator"] == 2
    assert result["failures"]["n_objects"] == 2
    assert result["primary"]["value"] == FAILURE_ERROR_DEG
    assert all(row["oracle_at_3_error_deg"] == FAILURE_ERROR_DEG for row in result["objects"])


def test_invalid_reference_is_not_converted_to_prediction_failure() -> None:
    cohort = _cohort("mpc:1")
    cohort["objects"][0]["references"] = [[0, 0, 0]]
    with pytest.raises(GeneralizationValidationError, match="invalid axis"):
        analyze_generalization(
            cohort=cohort,
            candidate_predictions=_axis_rows("mpc:1"),
            atlas_predictions=_atlas("mpc:1"),
            uniform_resamples=10,
        )


def test_duplicate_prediction_and_missing_or_invalid_atlas_are_rejected() -> None:
    duplicate = _axis_rows("mpc:1", "mpc:1")
    with pytest.raises(GeneralizationValidationError, match="duplicate"):
        analyze_generalization(
            cohort=_cohort("mpc:1"),
            candidate_predictions=duplicate,
            atlas_predictions=_atlas("mpc:1"),
            uniform_resamples=10,
        )
    with pytest.raises(GeneralizationValidationError, match="missing eligible"):
        analyze_generalization(
            cohort=_cohort("mpc:1", "mpc:2"),
            candidate_predictions=_axis_rows("mpc:1", "mpc:2"),
            atlas_predictions={"train_only": True, "objects": [{
                "object_id": "mpc:1", "axes": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            }]},
            uniform_resamples=10,
        )
    with pytest.raises(GeneralizationValidationError, match="atlas baseline is invalid"):
        analyze_generalization(
            cohort=_cohort("mpc:1"),
            candidate_predictions=_axis_rows("mpc:1"),
            atlas_predictions={"train_only": True, "objects": [{
                "object_id": "mpc:1", "status": "failed", "failure_reason": "not generated"
            }]},
            uniform_resamples=10,
        )


def test_file_hash_is_stable(tmp_path: Path) -> None:
    path = tmp_path / "value"
    path.write_bytes(b"value")
    assert sha256_file(path) == "cd42404d52ad55ccfa9aca4adc828aa5800ad9d385a0671fbcbf724118320619"
