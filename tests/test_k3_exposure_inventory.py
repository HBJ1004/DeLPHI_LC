from __future__ import annotations

import json
from pathlib import Path

import pytest

from lc_pipeline.k3.generalization_validation import GeneralizationValidationError
from repro.build_k3_exposure_inventory import (
    _write_outputs,
    build_exposure_inventory,
)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _fixtures(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "census": tmp_path / "census.json",
        "aliases": tmp_path / "aliases.json",
        "splits": tmp_path / "splits.json",
        "published": tmp_path / "published",
        "historical": tmp_path / "historical",
        "eef": tmp_path / "eef" / "synthetic" / "validation",
        "legacy": tmp_path / "legacy.json",
        "fink": tmp_path / "fink",
        "fetch": tmp_path / "fetch_ztf.py",
    }
    _write(
        paths["census"],
        {
            "crossmatch_count": 3,
            "reference_availability": {
                "poles_loaded": False,
                "solutions_loaded": False,
                "records": [
                    {"candidate_identity": "mpc:2"},
                    {"candidate_identity": "mpc:101"},
                    {"candidate_identity": "mpc:500"},
                ],
            },
        },
    )
    _write(
        paths["aliases"],
        {
            "identity_table_sha256": "a" * 64,
            "objects": [
                {"object_id": "asteroid_101", "damit_id": 101, "mpc_number": 2},
            ],
        },
    )
    _write(
        paths["splits"],
        {"folds": [{"fold": 0, "test_ids": ["asteroid_101"], "train_ids": []}]},
    )
    manifest = {
        "schema": "delphi.k3-synthetic-shard.v1",
        "shape_donor_ids": ["asteroid_101/model_102"],
        "geometry_donor_ids": ["asteroid_101"],
    }
    for root in (paths["published"], paths["historical"]):
        _write(root / "train" / "train-0000.npz.manifest.json", manifest)
    paths["eef"].mkdir(parents=True)
    _write(paths["legacy"], {"objects": [{"observed_mpc_number": 101}]})
    _write(paths["fink"] / "2.json", {})
    _write(paths["fink"] / "777.json", {})
    paths["fetch"].write_text(
        '"""IAU asteroid number."""\npayload = {"n_or_d": str(asteroid_number)}\n',
        encoding="utf-8",
    )
    return paths


def test_inventory_is_namespace_safe_conservative_and_no_overwrite(tmp_path: Path) -> None:
    paths = _fixtures(tmp_path)
    census, prior, receipt = build_exposure_inventory(
        census_crossmatch=paths["census"],
        identity_aliases=paths["aliases"],
        original_splits=paths["splits"],
        published_metadata_root=paths["published"],
        historical_k3_root=paths["historical"],
        eef6d4ec_root=paths["eef"].parents[1],
        legacy_ztf_audit=paths["legacy"],
        historical_fink_raw_dir=paths["fink"],
        historical_fink_fetch_script=paths["fetch"],
        expected_manifest_count=1,
        expected_census_count=3,
        expected_fink_file_count=2,
    )
    statuses = {row["object_id"]: row["status"] for row in census["objects"]}
    assert statuses == {"mpc:101": "exposed", "mpc:2": "exposed", "mpc:500": "unknown"}
    assert census["all_candidates_unexposed"] is False
    assert receipt["inputs"]["published_and_historical_manifests_byte_identical"] is True
    assert receipt["inputs"]["historical_k3_eef6d4ec"]["coverage"] == "unknown"
    assert receipt["coverage_boundary"]["exhaustive_project_history_certified"] is False
    assert all(row["status"] == "exposed" for row in prior["objects"])
    assert prior["known_exposed_by_source_nonexclusive"] == {
        "prior_post_cutoff_evaluated_cases": 3
    }
    assert all(
        row["enumerated_source_overlap"]["original_170_member"] is False
        for row in prior["objects"]
    )

    output = tmp_path / "outputs"
    _write_outputs(output, census, prior, receipt)
    assert (output / "k3-census-exposure-audit.json").is_file()
    inventory = json.loads((output / "k3-exposure-source-inventory.json").read_text())
    assert inventory["derived_outputs"]["census_audit"]["sha256"]
    with pytest.raises(GeneralizationValidationError, match="refusing to overwrite"):
        _write_outputs(output, census, prior, receipt)


def test_inventory_rejects_unproven_fink_namespace(tmp_path: Path) -> None:
    paths = _fixtures(tmp_path)
    paths["fetch"].write_text("payload = {}\n", encoding="utf-8")
    with pytest.raises(GeneralizationValidationError, match="does not establish"):
        build_exposure_inventory(
            census_crossmatch=paths["census"],
            identity_aliases=paths["aliases"],
            original_splits=paths["splits"],
            published_metadata_root=paths["published"],
            historical_k3_root=paths["historical"],
            eef6d4ec_root=paths["eef"].parents[1],
            legacy_ztf_audit=paths["legacy"],
            historical_fink_raw_dir=paths["fink"],
            historical_fink_fetch_script=paths["fetch"],
            expected_manifest_count=1,
            expected_census_count=3,
            expected_fink_file_count=2,
        )
