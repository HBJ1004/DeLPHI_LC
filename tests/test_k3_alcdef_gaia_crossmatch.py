import json

from repro.crossmatch_k3_alcdef_gaia import crossmatch


def test_crossmatch_retains_unknown_history_and_model_derived_reference(tmp_path):
    census = tmp_path / "census.json"
    census.write_text(json.dumps({"schema": "delphi.k3-external-source-census.v1", "objects": [{"object_number": 5, "primary_screen_approximation": True, "valid_point_count": 200, "session_count": 5, "distinct_session_years_approximation": 2, "filters": ["R"]}]}))
    table = tmp_path / "table3.dat"
    table.write_text(f"{5:6d} {'Astraea':<16} {122:3d} {37:3d} {312:3d} {40:3d} {16.8008:11.6f} {36:2d} C\n")
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"schema": "delphi.k3-identity-exposure-audit.v1", "objects": [{"object_id": "mpc:5", "status": "unknown", "reasons": ["history_not_auditable"]}]}))
    result = crossmatch(census, table, audit)
    assert result["counts"] == {"gaia_reference_rows": 1, "joined_candidates": 1, "exposure_status": {"unknown": 1}}
    assert result["candidates"][0]["gaia_reference"]["reference_type"] == "Gaia_DR3_model_derived_spin_solution"
    assert "full historical training/data exposure" in result["eligibility"]["not_yet_checked"]


def test_crossmatch_marks_audited_historical_candidate_as_exposed(tmp_path):
    census = tmp_path / "census.json"
    census.write_text(json.dumps({"schema": "delphi.k3-external-source-census.v1", "objects": [{"object_number": 5, "primary_screen_approximation": True, "valid_point_count": 200, "session_count": 5, "distinct_session_years_approximation": 2, "filters": ["R"]}]}))
    table = tmp_path / "table3.dat"
    table.write_text(f"{5:6d} {'Astraea':<16} {122:3d} {37:3d} {312:3d} {40:3d} {16.8008:11.6f} {36:2d} C\n")
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"schema": "delphi.k3-identity-exposure-audit.v1", "objects": []}))
    historical = tmp_path / "historical.json"
    historical.write_text(json.dumps({"schema": "delphi.k3-historical-metadata-exposure-audit.v1", "phase30": {"candidate_inventory_overlap": ["mpc:5"]}}))
    result = crossmatch(census, table, audit, historical)
    assert result["candidates"][0]["exposure"]["status"] == "exposed"
    assert "present_in:historical_phase30_metadata" in result["candidates"][0]["exposure"]["reasons"]
