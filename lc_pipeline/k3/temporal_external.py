"""Fetch and prepare the frozen strict-temporal DAMIT case series.

Network acquisition and prediction preparation are separate operations.  The
fetch snapshot retains official table, lightcurve, and spin bytes.  Preparation
is allowed to open only ``input-index.json`` and the label-free lightcurves; pole
labels remain under ``reference/`` and are not needed to construct model input.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import yaml

from ..v2.catalog import audit_lightcurve_file, ecliptic_vector, parse_spin
from ..v2.data import parse_damit_lightcurve
from .ztf_external import HttpFetcher, ZTFExternalError, _default_http_fetcher

DAMIT_BASE_URL = "https://damit.cuni.cz/projects/damit"
DAMIT_ASTEROID_TABLE_URL = f"{DAMIT_BASE_URL}/exports/table/asteroids"
DAMIT_MODEL_TABLE_URL = f"{DAMIT_BASE_URL}/exports/table/asteroid_models"
TEMPORAL_INPUT_INDEX_SCHEMA = "delphi.k3-temporal-damit-input-index.v1"
TEMPORAL_REFERENCE_SCHEMA = "delphi.k3-temporal-damit-reference.v1"
TEMPORAL_FETCH_RECEIPT_SCHEMA = "delphi.k3-temporal-damit-fetch-receipt.v1"
EXTERNAL_PREPARED_SCHEMA = "delphi.k3-external-prepared.v1"
TEMPORAL_PREPARED_MANIFEST_SCHEMA = "delphi.k3-temporal-prepared-manifest.v1"


class TemporalDAMITError(ValueError):
    """Raised when the strict-temporal source or separation contract fails."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_csv(payload: bytes, name: str) -> list[dict[str, str]]:
    try:
        text = payload.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text))
        rows = [dict(row) for row in reader]
    except (UnicodeError, csv.Error) as exc:
        raise TemporalDAMITError(f"cannot parse official DAMIT {name} CSV: {exc}") from exc
    if not reader.fieldnames or not rows:
        raise TemporalDAMITError(f"official DAMIT {name} CSV is empty")
    return rows


def _strict_identities(spec_path: Path) -> tuple[int, ...]:
    try:
        spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
        section = spec["temporal_external_case_series"]
        cutoff = str(section["cutoff_utc"])
        identities = tuple(
            int(str(value).removeprefix("asteroid_")) for value in section["identities"]
        )
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise TemporalDAMITError(f"cannot read strict-temporal study policy: {exc}") from exc
    if identities != (49, 279, 366):
        raise TemporalDAMITError("strict-temporal identities must remain 49, 279, and 366")
    try:
        datetime.fromisoformat(cutoff.replace("Z", "+00:00"))
    except ValueError as exc:
        raise TemporalDAMITError("strict-temporal cutoff is not ISO-8601") from exc
    return identities


def _cutoff(spec_path: Path) -> datetime:
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    return datetime.fromisoformat(
        str(spec["temporal_external_case_series"]["cutoff_utc"]).replace("Z", "+00:00")
    )


def _source_datetime(value: str, field: str) -> datetime:
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise TemporalDAMITError(f"invalid DAMIT {field} timestamp: {value!r}") from exc
    return parsed


def _fetch(
    *,
    url: str,
    destination: Path,
    role: str,
    timeout_seconds: float,
    fetcher: HttpFetcher,
) -> tuple[bytes, dict[str, object]]:
    try:
        body, final_url, http_date, content_type = fetcher(url, timeout_seconds)
    except (OSError, ZTFExternalError) as exc:
        raise TemporalDAMITError(f"cannot fetch {role}: {exc}") from exc
    if not body:
        raise TemporalDAMITError(f"official DAMIT returned an empty {role}")
    _write_bytes(destination, body)
    return body, {
        "role": role,
        "requested_url": url,
        "final_url": final_url,
        "http_date": http_date,
        "content_type": content_type,
        "path": destination.as_posix(),
        "bytes": len(body),
        "sha256": _sha256_bytes(body),
    }


def _relative_record(record: Mapping[str, object], root: Path) -> dict[str, object]:
    value = dict(record)
    value["path"] = Path(str(record["path"])).relative_to(root).as_posix()
    return value


def fetch_temporal_damit_snapshot(
    *,
    study_spec_path: str | Path,
    output_directory: str | Path,
    timeout_seconds: float = 120.0,
    request_interval_seconds: float = 0.25,
    fetcher: HttpFetcher | None = None,
) -> dict[str, object]:
    """Freeze official DAMIT bytes and separated input/reference manifests."""
    spec_path = Path(study_spec_path)
    destination = Path(output_directory)
    if destination.exists():
        raise TemporalDAMITError("temporal DAMIT snapshot directory must not already exist")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise TemporalDAMITError("timeout_seconds must be finite and positive")
    if not math.isfinite(request_interval_seconds) or request_interval_seconds < 0:
        raise TemporalDAMITError("request_interval_seconds must be finite and nonnegative")
    identities = _strict_identities(spec_path)
    cutoff = _cutoff(spec_path)
    active_fetcher = fetcher or _default_http_fetcher
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    fetch_records: list[dict[str, object]] = []
    try:
        asteroid_body, record = _fetch(
            url=DAMIT_ASTEROID_TABLE_URL,
            destination=staging / "raw" / "input" / "asteroids.csv",
            role="DAMIT asteroid table",
            timeout_seconds=timeout_seconds,
            fetcher=active_fetcher,
        )
        fetch_records.append(_relative_record(record, staging))
        if request_interval_seconds:
            time.sleep(request_interval_seconds)
        model_body, record = _fetch(
            url=DAMIT_MODEL_TABLE_URL,
            destination=staging / "raw" / "reference" / "asteroid_models.csv",
            role="DAMIT model table (reference-bearing)",
            timeout_seconds=timeout_seconds,
            fetcher=active_fetcher,
        )
        fetch_records.append(_relative_record(record, staging))
        asteroid_rows = _read_csv(asteroid_body, "asteroid")
        model_rows = _read_csv(model_body, "model")
        asteroids: dict[int, dict[str, str]] = {}
        for number in identities:
            matches = [row for row in asteroid_rows if row.get("number") == str(number)]
            if len(matches) != 1:
                raise TemporalDAMITError(
                    f"official DAMIT table has {len(matches)} rows for asteroid {number}"
                )
            created = _source_datetime(matches[0]["created"], f"asteroid {number} creation")
            if created <= cutoff:
                raise TemporalDAMITError(
                    f"asteroid_{number} is not strictly temporal: {created.isoformat()}"
                )
            asteroids[number] = matches[0]

        input_objects: list[dict[str, object]] = []
        reference_objects: list[dict[str, object]] = []
        for number in identities:
            asteroid = asteroids[number]
            internal_id = int(asteroid["id"])
            candidates = [row for row in model_rows if row.get("asteroid_id") == str(internal_id)]
            if not candidates:
                raise TemporalDAMITError(f"official DAMIT has no models for asteroid_{number}")
            qualifying: list[dict[str, str]] = []
            for model in candidates:
                try:
                    quality = float(model["quality_flag"])
                except (KeyError, ValueError):
                    continue
                if math.isfinite(quality) and quality >= 3:
                    created = _source_datetime(
                        model["created"], f"asteroid {number} model creation"
                    )
                    if created <= cutoff:
                        raise TemporalDAMITError(
                            f"asteroid_{number} model {model['id']} predates the cutoff"
                        )
                    qualifying.append(model)
            qualifying.sort(key=lambda row: int(row["id"]))
            if not qualifying:
                raise TemporalDAMITError(
                    f"official DAMIT has no quality-3 temporal model for asteroid_{number}"
                )
            period_model = qualifying[0]
            period = float(period_model["period"])
            if not math.isfinite(period) or period <= 0:
                raise TemporalDAMITError(f"invalid period for asteroid_{number}")

            if request_interval_seconds:
                time.sleep(request_interval_seconds)
            lightcurve_url = (
                f"{DAMIT_BASE_URL}/light_curves/exportAllForAsteroid/"
                f"{internal_id}/plaintext"
            )
            lightcurve_path = staging / "raw" / "input" / f"asteroid_{number}" / "lc.txt"
            _lightcurve_body, lc_record = _fetch(
                url=lightcurve_url,
                destination=lightcurve_path,
                role=f"asteroid_{number} lightcurves",
                timeout_seconds=timeout_seconds,
                fetcher=active_fetcher,
            )
            fetch_records.append(_relative_record(lc_record, staging))
            try:
                lightcurve_audit = audit_lightcurve_file(lightcurve_path)
            except ValueError as exc:
                raise TemporalDAMITError(
                    f"invalid official lightcurve for asteroid_{number}: {exc}"
                ) from exc

            reference_models: list[dict[str, object]] = []
            for model in qualifying:
                if request_interval_seconds:
                    time.sleep(request_interval_seconds)
                model_id = int(model["id"])
                spin_url = (
                    f"{DAMIT_BASE_URL}/generated_files/open/AsteroidModel/"
                    f"{model_id}/spin.txt"
                )
                spin_path = (
                    staging
                    / "raw"
                    / "reference"
                    / f"asteroid_{number}"
                    / f"model_{model_id}"
                    / "spin.txt"
                )
                _spin_body, spin_record = _fetch(
                    url=spin_url,
                    destination=spin_path,
                    role=f"asteroid_{number} model_{model_id} spin label",
                    timeout_seconds=timeout_seconds,
                    fetcher=active_fetcher,
                )
                fetch_records.append(_relative_record(spin_record, staging))
                try:
                    spin = parse_spin(spin_path)
                    table_values = (
                        float(model["lambda"]),
                        float(model["beta"]),
                        float(model["period"]),
                    )
                except (ValueError, KeyError) as exc:
                    raise TemporalDAMITError(
                        f"invalid reference model {model_id} for asteroid_{number}: {exc}"
                    ) from exc
                spin_values = (spin.lambda_deg, spin.beta_deg, spin.period_hours)
                if any(
                    abs(left - right) > 1e-9
                    for left, right in zip(spin_values, table_values, strict=True)
                ):
                    raise TemporalDAMITError(
                        f"spin/table mismatch for asteroid_{number} model_{model_id}"
                    )
                reference_models.append(
                    {
                        "model_id": model_id,
                        "quality_flag": float(model["quality_flag"]),
                        "lambda_deg": spin.lambda_deg % 360.0,
                        "beta_deg": spin.beta_deg,
                        "period_hours": spin.period_hours,
                        "axis_ecliptic_j2000": list(
                            ecliptic_vector(spin.lambda_deg, spin.beta_deg)
                        ),
                        "created": model["created"],
                        "modified": model["modified"],
                        "spin_path": spin_path.relative_to(staging).as_posix(),
                        "spin_sha256": _sha256_file(spin_path),
                    }
                )

            input_objects.append(
                {
                    "object_id": f"asteroid_{number}",
                    "asteroid_number": number,
                    "damit_internal_asteroid_id": internal_id,
                    "name": asteroid["name"],
                    "created": asteroid["created"],
                    "modified": asteroid["modified"],
                    "creation_date_is_after_cutoff": True,
                    "lightcurve_path": lightcurve_path.relative_to(staging).as_posix(),
                    "lightcurve_sha256": _sha256_file(lightcurve_path),
                    "lightcurve_audit": lightcurve_audit,
                    "known_period_hours": period,
                    "period_source": {
                        "selection": "lowest_model_id_with_quality_flag_at_least_3",
                        "model_id": int(period_model["id"]),
                        "quality_flag": float(period_model["quality_flag"]),
                        "created": period_model["created"],
                        "model_table_sha256": _sha256_bytes(model_body),
                        "pole_columns_excluded_from_input_index": True,
                    },
                }
            )
            reference_objects.append(
                {
                    "object_id": f"asteroid_{number}",
                    "asteroid_number": number,
                    "damit_internal_asteroid_id": internal_id,
                    "created": asteroid["created"],
                    "models": reference_models,
                }
            )

        input_index: dict[str, object] = {
            "schema": TEMPORAL_INPUT_INDEX_SCHEMA,
            "study_spec_sha256": _sha256_file(spec_path),
            "cutoff_utc": cutoff.isoformat().replace("+00:00", "Z"),
            "source": "official DAMIT endpoints",
            "source_timestamp_interpretation": (
                "DAMIT table timestamps recorded verbatim; treated as UTC for cutoff validation"
            ),
            "selection_uses_reference_axes": False,
            "identity_count": len(input_objects),
            "objects": input_objects,
        }
        _write_json(staging / "input-index.json", input_index)
        reference_manifest: dict[str, object] = {
            "schema": TEMPORAL_REFERENCE_SCHEMA,
            "study_spec_sha256": _sha256_file(spec_path),
            "cutoff_utc": cutoff.isoformat().replace("+00:00", "Z"),
            "warning": "reference labels; do not expose to prediction preparation or fit selection",
            "model_table_path": "raw/reference/asteroid_models.csv",
            "model_table_sha256": _sha256_bytes(model_body),
            "objects": reference_objects,
        }
        _write_json(staging / "reference" / "reference-manifest.json", reference_manifest)
        receipt: dict[str, object] = {
            "schema": TEMPORAL_FETCH_RECEIPT_SCHEMA,
            "study_spec_sha256": _sha256_file(spec_path),
            "official_endpoints_only": True,
            "request_order": "serial",
            "input_index_path": "input-index.json",
            "input_index_sha256": _sha256_file(staging / "input-index.json"),
            "reference_manifest_path": "reference/reference-manifest.json",
            "reference_manifest_sha256": _sha256_file(
                staging / "reference" / "reference-manifest.json"
            ),
            "fetches": fetch_records,
        }
        _write_json(staging / "fetch-receipt.json", receipt)
        os.replace(staging, destination)
        return receipt
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _safe_snapshot_path(snapshot: Path, relative: object) -> Path:
    if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise TemporalDAMITError("temporal input index contains an unsafe source path")
    resolved = (snapshot / relative).resolve()
    try:
        resolved.relative_to(snapshot.resolve())
    except ValueError as exc:
        raise TemporalDAMITError("temporal input source escapes its snapshot") from exc
    return resolved


def prepare_temporal_damit_inputs(
    *, snapshot_directory: str | Path, output_directory: str | Path
) -> dict[str, object]:
    """Prepare three model inputs without opening the reference directory."""
    snapshot = Path(snapshot_directory)
    destination = Path(output_directory)
    if destination.exists():
        raise TemporalDAMITError("temporal prepared directory must not already exist")
    index_path = snapshot / "input-index.json"
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TemporalDAMITError(f"cannot load temporal input index: {exc}") from exc
    if index.get("schema") != TEMPORAL_INPUT_INDEX_SCHEMA:
        raise TemporalDAMITError("temporal input index schema mismatch")
    objects = index.get("objects")
    if not isinstance(objects, list) or len(objects) != 3:
        raise TemporalDAMITError("temporal input index must contain exactly three objects")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    manifest_rows: list[dict[str, object]] = []
    try:
        for row in objects:
            object_id = str(row["object_id"])
            lightcurve = _safe_snapshot_path(snapshot, row["lightcurve_path"])
            if _sha256_file(lightcurve) != row["lightcurve_sha256"]:
                raise TemporalDAMITError(f"lightcurve hash mismatch for {object_id}")
            try:
                epochs = parse_damit_lightcurve(lightcurve)
                period = float(row["known_period_hours"])
            except (ValueError, KeyError, TypeError) as exc:
                raise TemporalDAMITError(f"cannot prepare {object_id}: {exc}") from exc
            if not math.isfinite(period) or period <= 0:
                raise TemporalDAMITError(f"invalid known period for {object_id}")
            payload: dict[str, object] = {
                "schema": EXTERNAL_PREPARED_SCHEMA,
                "object_id": object_id,
                "external_role": "strict_temporal_external_case_series",
                "known_period_hours": period,
                "period_provenance": (
                    "official DAMIT model-table period; lowest model ID with quality flag >=3; "
                    "axis columns excluded from preparation input"
                ),
                "source": {
                    "repository": "DAMIT",
                    "created": row["created"],
                    "lightcurve_sha256": row["lightcurve_sha256"],
                    "input_index_sha256": _sha256_file(index_path),
                },
                "epochs": [
                    {
                        "epoch_id": epoch.epoch_id,
                        "observations": [
                            {
                                "time_jd": observation.time_jd,
                                "relative_brightness": observation.relative_brightness,
                                "sun_asteroid_ecliptic_j2000_au": list(
                                    observation.sun_asteroid_ecliptic_j2000_au
                                ),
                                "observer_asteroid_ecliptic_j2000_au": list(
                                    observation.observer_asteroid_ecliptic_j2000_au
                                ),
                                "measured_error": observation.measured_error,
                            }
                            for observation in epoch.observations
                        ],
                    }
                    for epoch in epochs
                ],
            }
            serialized = json.dumps(payload, sort_keys=True).lower()
            forbidden = ('"lambda_deg"', '"beta_deg"', '"axis_', '"spin_', '"solutions"')
            if any(token in serialized for token in forbidden):
                raise TemporalDAMITError(f"reference label leaked into prepared {object_id}")
            output_path = staging / "objects" / f"{object_id}.json"
            _write_json(output_path, payload)
            manifest_rows.append(
                {
                    "object_id": object_id,
                    "path": output_path.relative_to(staging).as_posix(),
                    "sha256": _sha256_file(output_path),
                    "epoch_count": len(epochs),
                    "observation_count": sum(len(epoch.observations) for epoch in epochs),
                }
            )
        manifest: dict[str, object] = {
            "schema": TEMPORAL_PREPARED_MANIFEST_SCHEMA,
            "input_index_sha256": _sha256_file(index_path),
            "selection_uses_reference_axes": False,
            "reference_directory_access_required": False,
            "object_count": len(manifest_rows),
            "objects": manifest_rows,
        }
        _write_json(staging / "manifest.json", manifest)
        os.replace(staging, destination)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


__all__ = [
    "DAMIT_ASTEROID_TABLE_URL",
    "DAMIT_MODEL_TABLE_URL",
    "EXTERNAL_PREPARED_SCHEMA",
    "TEMPORAL_FETCH_RECEIPT_SCHEMA",
    "TEMPORAL_INPUT_INDEX_SCHEMA",
    "TEMPORAL_PREPARED_MANIFEST_SCHEMA",
    "TEMPORAL_REFERENCE_SCHEMA",
    "TemporalDAMITError",
    "fetch_temporal_damit_snapshot",
    "prepare_temporal_damit_inputs",
]
