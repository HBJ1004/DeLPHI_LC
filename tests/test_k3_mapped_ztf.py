"""Mapped DAMIT-internal to MPC-numbered ZTF preparation contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import lc_pipeline.k3.mapped_ztf as mapped_ztf
from lc_pipeline.k3.mapped_ztf import (
    MappedZTFError,
    fetch_mapped_fink,
    ingest_mapped_fink,
    prepare_mapped_ztf,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity_map(path: Path) -> Path:
    objects = [
        {"object_id": "asteroid_101", "damit_id": 101, "mpc_number": 2, "name": "Pallas", "held_out_fold": 2}
    ]
    objects.extend(
        {"object_id": f"asteroid_{1000 + index}", "damit_id": 1000 + index, "mpc_number": 10_000 + index, "name": f"X{index}", "held_out_fold": index % 5}
        for index in range(169)
    )
    path.write_text(json.dumps({"schema": "delphi.k3-survey-identity-map.v1", "identity_table_sha256": "a" * 64, "splits_sha256": "b" * 64, "objects": objects}), encoding="utf-8")
    return path


def _periods(path: Path, identity_path: Path) -> Path:
    identities = json.loads(identity_path.read_text())["objects"]
    path.write_text(json.dumps({"schema": "delphi.k3-mapped-ztf-period-manifest.v1", "identity_map_sha256": _sha(identity_path), "objects": [{"object_id": row["object_id"], "mpc_number": row["mpc_number"], "held_out_fold": row["held_out_fold"], "known_period_hours": 8.0, "period_provenance": {"externally_supplied": True}} for row in identities]}), encoding="utf-8")
    return path


def _raw_row() -> dict[str, object]:
    return {"i:ssnamenr": 2, "i:fid": 2, "i:jd": 2450000.25, "i:magpsf": 18.0, "i:sigmapsf": 0.1, "Dhelio": 1.0, "Dobs": 1.0, "Phase": 90.0}


def test_mapped_ingest_uses_mpc_filename_and_alias_but_internal_prepared_key(tmp_path: Path):
    identity, periods = _identity_map(tmp_path / "identity.json"), _periods(tmp_path / "periods.json", tmp_path / "identity.json")
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "2.json").write_text(json.dumps([_raw_row()]), encoding="utf-8")

    manifest = ingest_mapped_fink(identity_map_path=identity, period_manifest_path=periods, raw_directory=raw, output_directory=tmp_path / "ingest")

    row = next(item for item in manifest["objects"] if item["object_id"] == "asteroid_101")
    assert row["status"] == "ready" and row["source_filename"] == "2.json"
    normalized = json.loads((tmp_path / "ingest" / "objects" / "asteroid_101.json").read_text())
    assert normalized["object_id"] == "asteroid_101"
    assert normalized["acquisition_alias"] == "asteroid_2"
    assert normalized["identity_binding"]["mpc_number"] == 2
    assert normalized["rows"][0]["fink_ssnamenr"] == "2"
    assert not (tmp_path / "ingest" / "objects" / "asteroid_2.json").exists()


def test_mapped_fetch_resume_verifies_declared_response_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    identity = _identity_map(tmp_path / "identity.json")
    row = _raw_row()
    row.update({"sso_name": "Pallas", "i:ra": 10.0, "i:dec": 20.0})
    body = json.dumps([row]).encode()

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return body

    monkeypatch.setattr(mapped_ztf, "urlopen", lambda *_args, **_kwargs: Response())
    kwargs = {"identity_map_path": identity, "output_directory": tmp_path / "fetch", "object_ids": ["asteroid_101"]}
    fetch_mapped_fink(**kwargs)
    (tmp_path / "fetch" / "responses" / "2.json").write_bytes(b"tampered")
    with pytest.raises(MappedZTFError, match="response body hash mismatch"):
        fetch_mapped_fink(**kwargs)


def test_mapped_ingest_canonicalizes_only_identity_map_provisional_aliases(tmp_path: Path):
    identity = _identity_map(tmp_path / "identity.json")
    document = json.loads(identity.read_text())
    document["objects"][0].update(
        {"object_id": "asteroid_1726", "damit_id": 1726, "mpc_number": 44612, "name": "", "designation": "1999 RP27"}
    )
    identity.write_text(json.dumps(document), encoding="utf-8")
    periods = _periods(tmp_path / "periods.json", identity)
    raw = tmp_path / "raw"
    raw.mkdir()
    source = _raw_row()
    source.update({"i:ssnamenr": "1999RP27", "sso_name": "1999 RP27"})
    (raw / "44612.json").write_text(json.dumps([source]), encoding="utf-8")

    manifest = ingest_mapped_fink(identity_map_path=identity, period_manifest_path=periods, raw_directory=raw, output_directory=tmp_path / "ingest")

    row = next(item for item in manifest["objects"] if item["object_id"] == "asteroid_1726")
    normalized = json.loads((tmp_path / "ingest" / "objects" / "asteroid_1726.json").read_text())
    assert row["status"] == "ready"
    assert normalized["identity_alias_normalizations"] == [{
        "source_row_index": 0, "identity_map_designation": "1999 RP27", "canonical_comparison_form": "1999RP27",
        "source_i:ssnamenr": "1999RP27", "source_sso_name": "1999 RP27", "normalizer_sso_name": "1999RP27",
    }]
    source.update({"i:ssnamenr": "2000AA", "sso_name": "2000 AA"})
    bad_raw = tmp_path / "bad-raw"
    bad_raw.mkdir()
    (bad_raw / "44612.json").write_text(json.dumps([source]), encoding="utf-8")
    rejected = ingest_mapped_fink(identity_map_path=identity, period_manifest_path=periods, raw_directory=bad_raw, output_directory=tmp_path / "bad-ingest")
    assert next(item for item in rejected["objects"] if item["object_id"] == "asteroid_1726")["status"] == "invalid_local_raw"


def test_prepare_rejects_legacy_unbound_normalized_input(tmp_path: Path):
    identity = _identity_map(tmp_path / "identity.json")
    periods = _periods(tmp_path / "periods.json", identity)
    normalized_root = tmp_path / "ingest"
    object_path = normalized_root / "objects" / "asteroid_101.json"
    object_path.parent.mkdir(parents=True)
    object_path.write_text(json.dumps({"schema": "delphi.k3-fink-normalized.v1", "object_id": "asteroid_101", "rows": [_raw_row()]}), encoding="utf-8")
    all_ids = [row["object_id"] for row in json.loads(identity.read_text())["objects"]]
    ingest = {
        "schema": "delphi.k3-mapped-fink-ingest-manifest.v1",
        "identity_map_sha256": _sha(identity),
        "objects": [
            {"object_id": "asteroid_101", "status": "ready", "normalized_path": "objects/asteroid_101.json", "normalized_sha256": _sha(object_path)},
            *({"object_id": object_id, "status": "missing_local_raw_may_be_fetchable"} for object_id in all_ids if object_id != "asteroid_101"),
        ],
    }
    (normalized_root / "manifest.json").write_text(json.dumps(ingest), encoding="utf-8")
    horizons = tmp_path / "horizons"
    horizons.mkdir()
    (horizons / "manifest.json").write_text(json.dumps({"schema": "delphi.k3-mapped-horizons-manifest.v1", "identity_map_sha256": _sha(identity), "objects": [{"object_id": "asteroid_101", "status": "fetched", "cache_path": "asteroid_101/cache.json", "cache_sha256": "x"}]}), encoding="utf-8")

    with pytest.raises(MappedZTFError, match="legacy unbound"):
        prepare_mapped_ztf(ingest_manifest_path=normalized_root / "manifest.json", horizons_manifest_path=horizons / "manifest.json", period_manifest_path=periods, identity_map_path=identity, output_directory=tmp_path / "prepared")


def test_prepare_retains_one_point_source_as_explicit_insufficient_observations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    identity = _identity_map(tmp_path / "identity.json")
    periods = _periods(tmp_path / "periods.json", identity)
    all_ids = [row["object_id"] for row in json.loads(identity.read_text())["objects"]]
    ingest_root, horizons_root = tmp_path / "ingest", tmp_path / "horizons"
    object_path, cache_path = ingest_root / "objects" / "asteroid_101.json", horizons_root / "asteroid_101" / "cache.json"
    object_path.parent.mkdir(parents=True)
    cache_path.parent.mkdir(parents=True)
    object_path.write_text("{}", encoding="utf-8")
    cache_path.write_text("{}", encoding="utf-8")
    (ingest_root / "manifest.json").write_text(json.dumps({
        "schema": "delphi.k3-mapped-fink-ingest-manifest.v1", "identity_map_sha256": _sha(identity),
        "ready_object_count": 1, "missing_local_raw_count": 169,
        "objects": [
            {"object_id": "asteroid_101", "status": "ready", "normalized_path": "objects/asteroid_101.json", "normalized_sha256": _sha(object_path)},
            *({"object_id": object_id, "status": "missing_local_raw_may_be_fetchable"} for object_id in all_ids if object_id != "asteroid_101"),
        ],
    }), encoding="utf-8")
    (horizons_root / "manifest.json").write_text(json.dumps({
        "schema": "delphi.k3-mapped-horizons-manifest.v1", "identity_map_sha256": _sha(identity),
        "objects": [{"object_id": "asteroid_101", "status": "fetched", "cache_path": "asteroid_101/cache.json", "cache_sha256": _sha(cache_path)}],
    }), encoding="utf-8")
    monkeypatch.setattr(mapped_ztf, "_mapped_normalized", lambda *_: {"rows": []})
    monkeypatch.setattr(mapped_ztf, "_validate_cache", lambda *_args, **_kwargs: {"query_metadata": {}})
    monkeypatch.setattr(
        mapped_ztf,
        "fink_rows_to_epoch",
        lambda *_: type("Epoch", (), {"observations": (object(),)})(),
    )

    manifest = prepare_mapped_ztf(
        ingest_manifest_path=ingest_root / "manifest.json", horizons_manifest_path=horizons_root / "manifest.json",
        period_manifest_path=periods, identity_map_path=identity, output_directory=tmp_path / "prepared",
    )

    row = next(item for item in manifest["objects"] if item["object_id"] == "asteroid_101")
    assert row == {"object_id": "asteroid_101", "status": "insufficient_observations", "retained_observation_count": 1}
    assert manifest["insufficient_observations_count"] == 1
    assert len(manifest["objects"]) == 170
