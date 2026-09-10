"""Focused contracts for label-blind Fink ingestion and Horizons caching."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest
import yaml

from lc_pipeline.k3.ztf_external import (
    HorizonsCache,
    ZTFExternalError,
    _parse_horizons_vectors,
    fetch_horizons_cache,
    fink_rows_to_epoch,
    ingest_fink_directory,
    normalize_fink_rows,
    plan_horizons_directory,
)


def _splits(path: Path) -> Path:
    folds = []
    for fold in range(5):
        first = fold * 34 + 1
        folds.append(
            {
                "fold": fold,
                "test_ids": [f"asteroid_{number}" for number in range(first, first + 34)],
            }
        )
    path.write_text(json.dumps({"folds": folds}), encoding="utf-8")
    return path


def _spec(path: Path) -> Path:
    value = {
        "ztf_cross_survey": {
            "photometry": {
                "filter": "ztf_r",
                "maximum_sigma_magnitude": 0.2,
                "epoch_policy": "one_source_epoch_per_ztf_object",
            }
        }
    }
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    return path


def _raw_row(**changes: object) -> dict[str, object]:
    row: dict[str, object] = {
        "i:ssnamenr": 1,
        "i:fid": 2,
        "i:jd": 2450000.25,
        "i:magpsf": 18.0,
        "i:sigmapsf": 0.1,
        "Dhelio": 1.0,
        "Dobs": 1.0,
        "Phase": 90.0,
        "RA": 270.0,
        "DEC": 0.0,
        "Date": 2450000.2501,
        "sso_name": "Test",
        "pole_label_that_must_not_escape": [1, 0, 0],
    }
    row.update(changes)
    return row


def test_ingest_fink_directory_normalizes_filters_and_never_copies_labels(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    source_rows = [
        _raw_row(),
        _raw_row(**{"i:jd": 2450001.0, "i:fid": 1}),
        _raw_row(**{"i:jd": 2450002.0, "i:sigmapsf": 0.3}),
    ]
    (raw / "1.json").write_text(json.dumps(source_rows), encoding="utf-8")
    output = tmp_path / "normalized"

    manifest = ingest_fink_directory(
        raw_directory=raw,
        splits_path=_splits(tmp_path / "splits.json"),
        study_spec_path=_spec(tmp_path / "spec.yaml"),
        output_directory=output,
    )

    assert manifest["ready_object_count"] == 1
    assert manifest["missing_object_count"] == 169
    assert manifest["retained_observation_count"] == 1
    prepared = json.loads((output / "objects" / "asteroid_1.json").read_text())
    assert prepared["rows"][0]["jd"] == 2450000.25
    assert prepared["rows"][0]["phase"] == 90.0
    assert prepared["rejected_row_counts"] == {
        "invalid_photometry": 0,
        "named_designation_rows": 0,
        "sigma_rejected": 1,
        "wrong_filter": 1,
    }
    assert "pole_label" not in json.dumps(prepared)


def test_fink_named_designation_is_retained_but_numeric_identity_remains_strict():
    numeric = _raw_row(**{"i:ssnamenr": "1", "sso_name": "Ceres"})
    named = _raw_row(
        **{
            "i:ssnamenr": "Ceres",
            "sso_name": "Ceres",
            "i:jd": 2450001.25,
        }
    )

    rows, counts = normalize_fink_rows("asteroid_1", [numeric, named])

    assert [row["fink_ssnamenr"] for row in rows] == ["1", "Ceres"]
    assert counts["named_designation_rows"] == 1
    with pytest.raises(ZTFExternalError, match="identity mismatch"):
        normalize_fink_rows("asteroid_1", [_raw_row(**{"i:ssnamenr": "2"})])
    with pytest.raises(ZTFExternalError, match="named designation mismatch"):
        normalize_fink_rows(
            "asteroid_1", [_raw_row(**{"i:ssnamenr": "Ceres", "sso_name": "Pallas"})]
        )


def _fake_horizons(url: str, _timeout: float) -> tuple[bytes, str, str, str]:
    query = parse_qs(urlparse(url).query)
    center = query["CENTER"][0].strip("'")
    raw_times = query["TLIST"][0].split()
    jds = [float(value.strip("'")) for value in raw_times]
    vector = (-1.0, 0.0, 0.0) if center == "500@10" else (0.0, -1.0, 0.0)
    rows = "\n".join(
        f"{jd:.12f}, A.D. fake, {vector[0]}, {vector[1]}, {vector[2]}," for jd in jds
    )
    payload = {
        "signature": {"source": "NASA/JPL Horizons API", "version": "test"},
        "result": f"header\n$$SOE\n{rows}\n$$EOE\nfooter",
    }
    return json.dumps(payload).encode(), url, "Thu, 10 Sep 2026 00:00:00 GMT", "application/json"


def test_horizons_parser_accepts_observed_fractional_second_jd_rounding():
    requested = 2460195.0118981
    returned = 2460195.011898099
    payload = json.dumps(
        {
            "signature": {"source": "NASA/JPL Horizons API", "version": "test"},
            "result": f"$$SOE\n{returned:.15f}, A.D. fake, 1, 2, 3,\n$$EOE",
        }
    ).encode()

    assert _parse_horizons_vectors(payload, [requested]) == ((1.0, 2.0, 3.0),)

    shifted = json.dumps(
        {
            "signature": {"source": "NASA/JPL Horizons API", "version": "test"},
            "result": f"$$SOE\n{requested + 3e-9:.15f}, A.D. fake, 1, 2, 3,\n$$EOE",
        }
    ).encode()
    with pytest.raises(ZTFExternalError, match="epoch mismatch"):
        _parse_horizons_vectors(shifted, [requested])


def test_horizons_fetch_retains_raw_bytes_hashes_and_builds_usable_cache(tmp_path: Path):
    normalized = tmp_path / "asteroid_1.json"
    normalized.write_text(
        json.dumps(
            {
                "schema": "delphi.k3-fink-normalized.v1",
                "object_id": "asteroid_1",
                "rows": [
                    {
                        "jd": 2450000.25,
                        "fid": 2,
                        "magpsf": 18.0,
                        "sigmapsf": 0.1,
                        "Dhelio": 1.0,
                        "Dobs": 1.0,
                        "phase": 90.0,
                        "ra": 270.0,
                        "dec": -23.4392911,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "horizons"

    cache = fetch_horizons_cache(
        normalized_object_path=normalized,
        output_directory=output,
        request_interval_seconds=0,
        fetcher=_fake_horizons,
    )

    parsed = HorizonsCache.from_mapping(cache)
    assert len(parsed.query_metadata["raw_responses"]) == 2
    assert all((output / row["path"]).is_file() for row in parsed.query_metadata["raw_responses"])
    assert parsed.rows[0]["asteroid_to_sun_ecliptic_j2000_au"] == [1.0, -0.0, -0.0]
    epoch = fink_rows_to_epoch(
        "asteroid_1", json.loads(normalized.read_text())["rows"], cache
    )
    assert len(epoch.observations) == 1
    assert epoch.observations[0].relative_brightness == 1.0


def test_horizons_dry_run_reports_pending_then_validated_resume_cache(tmp_path: Path):
    normalized_root = tmp_path / "normalized"
    object_path = normalized_root / "objects" / "asteroid_1.json"
    object_path.parent.mkdir(parents=True)
    object_path.write_text(
        json.dumps(
            {
                "schema": "delphi.k3-fink-normalized.v1",
                "object_id": "asteroid_1",
                "rows": [{"jd": 2450000.25}],
            }
        ),
        encoding="utf-8",
    )
    object_hash = hashlib.sha256(object_path.read_bytes()).hexdigest()
    manifest_path = normalized_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "delphi.k3-fink-ingest-manifest.v1",
                "objects": [
                    {
                        "object_id": "asteroid_1",
                        "status": "ready",
                        "normalized_path": "objects/asteroid_1.json",
                        "normalized_sha256": object_hash,
                        "retained_row_count": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    cache_root = tmp_path / "horizons"

    pending = plan_horizons_directory(
        normalized_manifest_path=manifest_path,
        output_directory=cache_root,
        batch_size=20,
    )

    assert pending["pending_object_count"] == 1
    assert pending["cached_object_count"] == 0
    assert pending["estimated_request_count"] == 2
    assert not cache_root.exists()

    fetch_horizons_cache(
        normalized_object_path=object_path,
        output_directory=cache_root / "asteroid_1",
        request_interval_seconds=0,
        fetcher=_fake_horizons,
    )
    resumed = plan_horizons_directory(
        normalized_manifest_path=manifest_path,
        output_directory=cache_root,
        batch_size=20,
    )
    assert resumed["pending_object_count"] == 0
    assert resumed["cached_object_count"] == 1
    assert resumed["estimated_request_count"] == 0
