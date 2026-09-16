import json
import zipfile

from repro.extract_k3_locked_alcdef import extract


def test_locked_extract_keeps_raw_rows_and_session_corrections(tmp_path):
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"schema": "delphi.k3-alcdef-gaia-cohort-lock.v1", "objects": [{"object_id": "mpc:5"}]}))
    source = tmp_path / "source.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("data/0/alcdef_metadata-0-99.csv", "ID,ObjNumber,SessionDateTime,LTCApplied\n1,5,2020-01-01T00:00:00Z,N\n")
        archive.writestr("data/0/alcdef_lcdata-0-99.csv", "ObjectNumber,MDID,JD,Mag,MagErr\n5,1,2450000.0,10.0,0.1\n")
    result = extract(lock, source)
    assert result["objects"][0]["raw_point_count"] == 1
    assert result["objects"][0]["sessions"][0]["LTCApplied"] == "N"
