"""Contracts for strict-temporal DAMIT acquisition and label-blind preparation."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import shutil
from pathlib import Path

import pytest
import yaml

from lc_pipeline.k3.temporal_external import (
    DAMIT_ASTEROID_TABLE_URL,
    DAMIT_MODEL_TABLE_URL,
    TemporalDAMITError,
    fetch_temporal_damit_snapshot,
    prepare_temporal_damit_inputs,
)
from lc_pipeline.k3.ztf_prediction import _load_prepared
from repro.prepare_k3_followup_inputs import _parser


def _csv_bytes(fieldnames: list[str], rows: list[dict[str, object]]) -> bytes:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode()


def _spec(path: Path) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "temporal_external_case_series": {
                    "cutoff_utc": "2025-06-10T00:03:01Z",
                    "identities": ["asteroid_49", "asteroid_279", "asteroid_366"],
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def _official_fixture() -> dict[str, bytes]:
    asteroids = [
        {
            "id": internal,
            "number": number,
            "name": name,
            "designation": "",
            "comment": "",
            "created": created,
            "modified": created,
        }
        for internal, number, name, created in (
            (10867, 49, "Pales", "2025-07-11 10:05:39"),
            (10868, 279, "Thule", "2025-07-17 09:10:10"),
            (10869, 366, "Vincentina", "2025-07-18 11:11:17"),
        )
    ]
    models = [
        {
            "id": model_id,
            "asteroid_id": internal,
            "lambda": longitude,
            "beta": latitude,
            "period": period,
            "quality_flag": 3,
            "created": created,
            "modified": created,
        }
        for model_id, internal, longitude, latitude, period, created in (
            (16309, 10867, 95, -89, 20.70802, "2025-07-11 10:33:04"),
            (16313, 10868, 58, 2, 23.8964, "2025-07-17 09:10:10"),
            (16314, 10868, 236, 3, 23.89632, "2025-07-17 09:10:13"),
            (16317, 10869, 51, 19, 17.3484, "2025-07-18 11:18:29"),
            (16318, 10869, 234, 0, 17.34844, "2025-07-18 11:18:30"),
        )
    ]
    lightcurve = (
        "1\n"
        "2 0\n"
        "2450000.0 1.0 1.0 0.0 0.0 0.0 1.0 0.0\n"
        "2450000.1 1.1 1.0 0.0 0.0 0.0 1.0 0.0\n"
    ).encode()
    result = {
        DAMIT_ASTEROID_TABLE_URL: _csv_bytes(list(asteroids[0]), asteroids),
        DAMIT_MODEL_TABLE_URL: _csv_bytes(list(models[0]), models),
    }
    for internal in (10867, 10868, 10869):
        result[
            "https://damit.cuni.cz/projects/damit/light_curves/"
            f"exportAllForAsteroid/{internal}/plaintext"
        ] = lightcurve
    for model in models:
        result[
            "https://damit.cuni.cz/projects/damit/generated_files/open/"
            f"AsteroidModel/{model['id']}/spin.txt"
        ] = f"{model['lambda']} {model['beta']} {model['period']}\n".encode()
    return result


def test_temporal_snapshot_retains_sources_and_prepares_without_references(tmp_path: Path):
    responses = _official_fixture()

    def fetcher(url: str, _timeout: float) -> tuple[bytes, str, str, str]:
        return responses[url], url, "Thu, 10 Sep 2026 00:00:00 GMT", "text/plain"

    snapshot = tmp_path / "snapshot"
    receipt = fetch_temporal_damit_snapshot(
        study_spec_path=_spec(tmp_path / "spec.yaml"),
        output_directory=snapshot,
        request_interval_seconds=0,
        fetcher=fetcher,
    )

    assert len(receipt["fetches"]) == 10
    for record in receipt["fetches"]:
        raw = snapshot / record["path"]
        assert raw.is_file()
        assert hashlib.sha256(raw.read_bytes()).hexdigest() == record["sha256"]
        assert record["http_date"] == "Thu, 10 Sep 2026 00:00:00 GMT"
    safe_text = (snapshot / "input-index.json").read_text().lower()
    assert all(token not in safe_text for token in ('"lambda_deg"', '"beta_deg"', '"axis_'))
    references = json.loads(
        (snapshot / "reference" / "reference-manifest.json").read_text()
    )
    assert sum(len(row["models"]) for row in references["objects"]) == 5
    assert references["objects"][0]["models"][0]["lambda_deg"] == 95

    # Prediction preparation must remain operable after both reference-bearing
    # locations are made unavailable.
    shutil.rmtree(snapshot / "reference")
    shutil.rmtree(snapshot / "raw" / "reference")
    prepared_root = tmp_path / "prepared"
    manifest = prepare_temporal_damit_inputs(
        snapshot_directory=snapshot, output_directory=prepared_root
    )

    assert manifest["object_count"] == 3
    assert manifest["reference_directory_access_required"] is False
    prepared_text = (prepared_root / "objects" / "asteroid_49.json").read_text().lower()
    assert '"lambda_deg"' not in prepared_text
    assert '"beta_deg"' not in prepared_text
    object_id, period, epochs = _load_prepared(
        prepared_root / "objects" / "asteroid_49.json"
    )
    assert object_id == "asteroid_49"
    assert period.hours == 20.70802
    assert sum(len(epoch.observations) for epoch in epochs) == 2


def test_temporal_fetch_fails_closed_when_an_object_is_not_after_cutoff(tmp_path: Path):
    responses = _official_fixture()
    rows = list(csv.DictReader(io.StringIO(responses[DAMIT_ASTEROID_TABLE_URL].decode())))
    rows[0]["created"] = "2025-06-10 00:03:01"
    responses[DAMIT_ASTEROID_TABLE_URL] = _csv_bytes(list(rows[0]), rows)

    def fetcher(url: str, _timeout: float) -> tuple[bytes, str, str, str]:
        return responses[url], url, "Thu, 10 Sep 2026 00:00:00 GMT", "text/plain"

    output = tmp_path / "snapshot"
    with pytest.raises(TemporalDAMITError, match="not strictly temporal"):
        fetch_temporal_damit_snapshot(
            study_spec_path=_spec(tmp_path / "spec.yaml"),
            output_directory=output,
            request_interval_seconds=0,
            fetcher=fetcher,
        )
    assert not output.exists()


def test_followup_cli_freezes_safe_horizons_batch_and_has_no_scoring_command():
    parser = _parser()
    args = parser.parse_args(
        [
            "plan-horizons",
            "--normalized-manifest",
            "normalized/manifest.json",
            "--output-directory",
            "horizons",
        ]
    )
    assert args.batch_size == 20
    with pytest.raises(SystemExit):
        parser.parse_args(["score-temporal"])
