from __future__ import annotations

import json
from pathlib import Path

import pytest

from lc_pipeline.k3.generalization_validation import GeneralizationValidationError
from repro.audit_k3_original_synthetic_overlap import build_overlap_report, write_report


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _fixtures(tmp_path: Path, *, donor: str = "asteroid_2") -> dict[str, Path]:
    paths = {
        "splits": tmp_path / "splits.json",
        "aliases": tmp_path / "aliases.json",
        "metadata": tmp_path / "metadata",
    }
    _write(
        paths["splits"],
        {
            "folds": [{
                "fold": 0,
                "train_ids": ["asteroid_101"],
                "validation_ids": ["asteroid_101"],
                "calibration_ids": [],
                "test_ids": ["asteroid_103"],
            }]
        },
    )
    _write(
        paths["aliases"],
        {
            "identity_table_sha256": "a" * 64,
            "objects": [
                {"object_id": "asteroid_101", "damit_id": 101, "mpc_number": 2},
                {"object_id": "asteroid_103", "damit_id": 103, "mpc_number": 5},
                {"object_id": "asteroid_2", "damit_id": 2, "mpc_number": 101},
            ],
        },
    )
    _write(
        paths["metadata"] / "train" / "train-0000.npz.manifest.json",
        {
            "schema": "delphi.k3-synthetic-shard.v1",
            "split": "train",
            "shape_donor_ids": [f"{donor}/model_9"],
            "geometry_donor_ids": [donor],
        },
    )
    return paths


def test_original_donor_audit_is_namespace_safe_and_bounded(tmp_path: Path) -> None:
    paths = _fixtures(tmp_path)
    report = build_overlap_report(
        original_splits=paths["splits"],
        identity_aliases=paths["aliases"],
        published_metadata_root=paths["metadata"],
        expected_manifest_count=1,
        expected_original_count=2,
        expected_donor_counts={"train": 1, "validation": 0, "test": 0},
    )
    assert report["checks"] == {
        "all_expected_donor_counts_match": True,
        "all_role_overlaps_zero": True,
        "all_shape_geometry_parent_sets_equal": True,
    }
    assert report["synthetic_splits"]["train"]["roles"]["shape"][
        "overlap_count_physical"
    ] == 0
    assert report["namespace_resolution"]["numeric_suffix_collapse_across_namespaces"] is False
    assert report["retraining_verified"] is False
    assert report["npz_contents_verified"] is False

    output = tmp_path / "report.json"
    write_report(output, report)
    with pytest.raises(GeneralizationValidationError, match="refusing to overwrite"):
        write_report(output, report)


def test_original_donor_audit_detects_physical_overlap_and_bad_suffix(tmp_path: Path) -> None:
    paths = _fixtures(tmp_path, donor="asteroid_101")
    report = build_overlap_report(
        original_splits=paths["splits"],
        identity_aliases=paths["aliases"],
        published_metadata_root=paths["metadata"],
        expected_manifest_count=1,
        expected_original_count=2,
        expected_donor_counts={"train": 1, "validation": 0, "test": 0},
    )
    assert report["checks"]["all_role_overlaps_zero"] is False
    assert report["all_overlap_physical_identities"] == ["mpc:2"]

    manifest = paths["metadata"] / "train" / "train-0000.npz.manifest.json"
    document = json.loads(manifest.read_text())
    document["shape_donor_ids"] = ["asteroid_101/not_a_model"]
    _write(manifest, document)
    with pytest.raises(GeneralizationValidationError, match="shape donor"):
        build_overlap_report(
            original_splits=paths["splits"],
            identity_aliases=paths["aliases"],
            published_metadata_root=paths["metadata"],
            expected_manifest_count=1,
            expected_original_count=2,
            expected_donor_counts={"train": 1, "validation": 0, "test": 0},
        )
