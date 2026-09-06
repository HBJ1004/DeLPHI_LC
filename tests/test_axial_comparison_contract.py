from __future__ import annotations

from pathlib import Path

import pytest

from lc_pipeline.v2.comparison_contract import (
    AXIAL_COMPARISON_SPEC_SHA256,
    ComparisonContractError,
    load_axial_comparison_spec,
    sha256_bytes,
)


def test_axial_comparison_policy_and_splits_are_hash_bound():
    document = load_axial_comparison_spec()
    assert document["status"] == "locked_before_new_comparison_runs"
    assert document["decision_values"] == [
        "V2_DIRECT",
        "V2_DOWNSTREAM_OVERRIDE",
        "V1_RETAINED",
    ]


def test_declared_spec_digest_matches_exact_repository_bytes():
    root = Path(__file__).resolve().parents[1]
    assert sha256_bytes((root / "repro/axial_comparison_spec.yaml").read_bytes()) == (
        AXIAL_COMPARISON_SPEC_SHA256
    )


def test_modified_policy_is_rejected_before_parsing(tmp_path):
    source = Path(__file__).resolve().parents[1] / "repro/axial_comparison_spec.yaml"
    modified = tmp_path / "modified.json"
    modified.write_bytes(source.read_bytes() + b"\n")
    with pytest.raises(ComparisonContractError, match="hash mismatch"):
        load_axial_comparison_spec(modified, verify_splits=False)
