import json

from lc_pipeline.k3.external_sources import (
    census_external_source,
    parse_alcdef_archive,
    parse_gaia_dr3_spins,
    parse_tssys_lightcurve,
    parse_tssys_release,
)


def test_gaia_dr3_spin_parser_reads_cds_fixed_width_table(tmp_path):
    table = tmp_path / "table3.dat"
    table.write_text(
        f"{5:6d} {'Astraea':<16} {122:3d} {37:3d} {312:3d} {40:3d} {16.8008:11.6f} {36:2d} C\n"
        f"{26:6d} {'Proserpina':<16} {83:3d} {-46:3d} {'':3} {'':3} {13.1092:11.6f} {58:2d} E\n"
    )
    rows = parse_gaia_dr3_spins(table)
    assert len(rows) == 2
    assert rows[0].object_number == 5
    assert rows[0].lambda2_deg == 312.0
    assert rows[1].lambda2_deg is None
    assert rows[1].period_hours == 13.1092


def _alcdef_fixture(root):
    root.mkdir()
    (root / "alcdef_metadata-1-99.csv").write_text(
        """# Column definitions,,,,,,,,,,,,
ID,Submitter,ObjNumber,ObjName,Desig,Facility,SessionDateTime,Filter,MagBand,LTCApplied,LTCType,LTCDays,ReducedMags,UnityCor,DiffMags,DiffZeroMag,MagStd,CICorr
integer,string,integer,string,string,string,datetime,string,string,Boolean,string,real,string,string,Boolean,magnitude,string,Boolean
-9,-,-9,-,-,-,1899-12-30T00:00:00Z,V,V,N,NONE,-99.9,NONE,-99.9,N,-99.9,NONE,N
\"1\",\"A. Observer\",\"3\",\"Juno\",\"-\",\"TESS\",\"2020-01-01T01:00:00Z\",\"R\",\"R\",\"N\",\"NONE\",\"-99.9\",\"NONE\",\"-99.9\",\"N\",\"-99.9\",\"INTSD\",\"N\"
\"2\",\"A. Observer\",\"3\",\"Juno\",\"-\",\"TESS\",\"2021-01-01T01:00:00Z\",\"R\",\"R\",\"N\",\"NONE\",\"-99.9\",\"NONE\",\"-99.9\",\"N\",\"-99.9\",\"INTSD\",\"N\"
""",
        encoding="utf-8",
    )
    (root / "alcdef_lcdata-1-99.csv").write_text(
        """# Column definitions,,
ObjectNumber,ObjectName,MDID,JD,Mag,MagErr
integer,string,integer,real,real,real
-9,-,-9,-99.9,-99.9,-99.9
\"000003\",\"Juno\",\"1\",\"2458849.5\",\"10.0\",\"0.1\"
\"000003\",\"Juno\",\"1\",\"2458849.5\",\"10.0\",\"0.1\"
\"000003\",\"Juno\",\"2\",\"2459215.5\",\"10.1\",\"-\"
""",
        encoding="utf-8",
    )
    return root


def test_alcdef_pds_parser_skips_schema_rows_preserves_corrections_and_duplicates(tmp_path):
    metadata, points, issues = parse_alcdef_archive(_alcdef_fixture(tmp_path / "alcdef"))
    assert not issues
    assert len(metadata) == 2
    assert metadata[0].candidate_identity == "mpc:3"
    assert metadata[0].correction_metadata["LTCApplied"] == "N"
    assert len(points) == 3
    assert points[1].valid is False
    assert "duplicate_observation_id" in points[1].exclusion_reasons
    assert points[2].magnitude_error is None


def test_alcdef_metadata_reports_truncated_row_and_keeps_session_mean(tmp_path):
    path = tmp_path / "alcdef_metadata-1-99.csv"
    path.write_text(
        """ID,Submitter,ObjNumber,ObjName,Desig,Facility,SessionDateTime,Filter,MagBand,LTCApplied,LTCType,LTCDays,ReducedMags,UnityCor,DiffMags,DiffZeroMag,MagStd,CICorr,Comments
integer,string,integer,string,string,string,datetime,string,string,Boolean,string,real,string,string,Boolean,magnitude,string,Boolean,string
\"7\",\"Observer\",\"3\",\"Juno\",\"-\",\"TESS\",\"2020-01-01T01:00:00Z\",\"R\",\"R\",\"N\",\"NONE\",\"-99.9\",\"NONE\",\"-99.9\",\"N\",\"-99.9\",\"INTSD\",\"N\",\"SessionDateTime is the source block mean\"
\"8\",\"truncated
""",
        encoding="utf-8",
    )
    from lc_pipeline.k3.external_sources import parse_alcdef_metadata

    rows, issues = parse_alcdef_metadata(path)
    assert len(rows) == 1
    assert rows[0].session_datetime == "2020-01-01T01:00:00Z"
    assert any(issue.startswith("malformed_row:") for issue in issues)


def test_tssys_release_and_lightcurve_are_metadata_only(tmp_path):
    release = tmp_path / "release.merge"
    release.write_text("  3      3.32896   P2      7.20946 0.1083  597  5 1 1  6.810  6.938  5.33\n")
    metadata, issues = parse_tssys_release(release)
    assert not issues
    assert metadata[0].object_number == 3
    assert metadata[0].period_hours == 7.20946
    lightcurve = tmp_path / "3.lc"
    lightcurve.write_text("3 2450000.0 1 2 10.0 0.1 0 0 0 4 2 1 3\n")
    points, issues = parse_tssys_lightcurve(lightcurve)
    assert not issues and points[0].valid
    assert points[0].magnitude_error == 0.1


def test_census_is_hash_bound_and_never_claims_independence(tmp_path):
    root = _alcdef_fixture(tmp_path / "alcdef")
    census = census_external_source(root)
    assert census["identity"]["namespace"] == "mpc"
    assert census["identity"]["independent_label"] is False
    assert census["observation_counts"]["duplicate_observation_ids"] == 1
    assert census["objects"][0]["documented_apparitions"] == "not_computed"
    assert census["input_files"] and len(census["input_files"][0]["sha256"]) == 64


def test_census_cli_help_and_json(tmp_path, capsys):
    from repro.census_k3_external_sources import main

    root = _alcdef_fixture(tmp_path / "alcdef")
    assert main([str(root)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["schema"] == "delphi.k3-external-source-census.v1"


def test_metadata_crossmatch_does_not_load_poles(tmp_path):
    census = tmp_path / "census.json"
    census.write_text(json.dumps({"schema": "delphi.k3-external-source-census.v1", "objects": [{"candidate_identity": "mpc:3", "object_number": 3, "session_count": 5, "valid_point_count": 150, "distinct_session_years_approximation": 2}]}))
    mapping = tmp_path / "asteroids.csv"
    mapping.write_text('"id","number","name","designation"\n102,3,"Juno",\n')
    from lc_pipeline.k3.external_sources import crossmatch_damit_metadata

    result = crossmatch_damit_metadata(census, mapping)
    assert result["crossmatch_count"] == 1
    assert result["reference_availability"]["damit_present_only"] is True
    assert result["reference_availability"]["poles_loaded"] is False
