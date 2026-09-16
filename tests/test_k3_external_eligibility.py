from __future__ import annotations

import json

import pytest

from lc_pipeline.k3.external_eligibility import (
    ExternalEligibilityError,
    build_eligibility_inventory,
)


def _row(identity: str = "mpc:2", **updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "physical_identity": identity,
        "source_database_id": "alcdef:2",
        "alias_receipt": "a" * 64,
        "input_source": "dense_alcdef",
        "raw_source_version": "alcdef-pds-v1",
        "input_receipt_hash": "b" * 64,
        "valid_points": 200,
        "native_sessions": 6,
        "documented_apparitions": 2,
        "apparition_evidence": {"type": "observatory_log", "years": [2020, 2022]},
        "observer": "facility-a",
        "time_scale": "UTC",
        "photometric_convention": "relative_magnitude",
        "filter": "R",
        "period_source": "SBDB",
        "period_precision": "0.001 h",
        "period_quality": "documented",
        "reference_source": "Gaia-DR3",
        "reference_quality_tier": "independent_supported",
        "reference_input_overlap": False,
        "direct_training_exposure": "clear",
        "synthetic_donor_exposure": "clear",
        "selection_exposure": "clear",
        "history_unknown": False,
    }
    row.update(updates)
    return row


def test_metadata_only_selection_is_deterministic_and_sorted() -> None:
    first = build_eligibility_inventory([_row("mpc:7"), _row("mpc:2")])
    second = build_eligibility_inventory([_row("mpc:2"), _row("mpc:7")])
    assert first == second
    assert [row["physical_identity"] for row in first["rows"]] == ["mpc:2", "mpc:7"]
    assert first["counts"] == {"confirmatory_candidate": 2}
    assert first["policy"]["reference_pole_values_used"] is False


def test_unknown_history_is_not_cleared_or_excluded_as_confirmatory() -> None:
    result = build_eligibility_inventory([_row(history_unknown=True)])
    assert result["rows"][0]["eligible_role"] == "unknown"
    assert "history_unknown" in result["rows"][0]["exclusion_reasons"]


def test_metadata_thresholds_and_source_are_explicit() -> None:
    result = build_eligibility_inventory([
        _row("mpc:2", valid_points=149),
        _row("mpc:3", input_source="atlas_sscat", documented_apparitions=1),
        _row("mpc:4", input_source="tess_tssys_dr1", reference_quality_tier="catalog_agreement"),
    ])
    by_id = {row["physical_identity"]: row for row in result["rows"]}
    assert by_id["mpc:2"]["eligible_role"] == "excluded"
    assert by_id["mpc:3"]["eligible_role"] == "excluded"
    assert by_id["mpc:4"]["eligible_role"] == "development_candidate"


def test_reference_pole_or_score_cannot_influence_eligibility() -> None:
    clean = build_eligibility_inventory([_row()])
    assert clean["rows"][0]["eligible_role"] == "confirmatory_candidate"
    with pytest.raises(ExternalEligibilityError, match="outcome fields"):
        build_eligibility_inventory([_row(reference_axis=[1, 0, 0])])
    with pytest.raises(ExternalEligibilityError, match="outcome fields"):
        build_eligibility_inventory([_row(oracle_error=0.1)])


def test_duplicate_identity_source_and_bad_identity_fail_closed() -> None:
    with pytest.raises(ExternalEligibilityError, match="duplicate"):
        build_eligibility_inventory([_row(), _row()])
    with pytest.raises(ExternalEligibilityError, match="canonical namespace"):
        build_eligibility_inventory([_row("asteroid_101")])


def test_alias_and_input_receipts_are_hash_bound() -> None:
    with pytest.raises(ExternalEligibilityError, match="alias_receipt"):
        build_eligibility_inventory([_row(alias_receipt="not-a-hash")])
    with pytest.raises(ExternalEligibilityError, match="input_receipt_hash"):
        build_eligibility_inventory([_row(input_receipt_hash="not-a-hash")])


def test_cli_writes_metadata_inventory(tmp_path, capsys) -> None:
    source = tmp_path / "rows.json"
    output = tmp_path / "inventory.json"
    source.write_text(json.dumps([_row()]), encoding="utf-8")
    from repro.build_k3_external_eligibility import main

    assert main(["--input", str(source), "--output", str(output)]) == 0
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["schema"] == "delphi.k3-external-eligibility.v1"
    assert document["purpose"].startswith("metadata-only")
    assert capsys.readouterr().out == ""
