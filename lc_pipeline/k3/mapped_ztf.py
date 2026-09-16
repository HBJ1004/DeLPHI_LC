"""Mapped, label-blind ZTF preparation for internal DAMIT identities.

The old follow-up accidentally used an internal DAMIT database suffix as an
MPC number.  This module makes the two namespaces explicit: data acquisition
uses a mapped MPC number, while every period/fold/prepared-object join retains
the frozen internal ``asteroid_<damit_id>`` identifier.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .survey_identity import BINDING_SCHEMA, IDENTITY_MAP_SCHEMA
from .ztf_external import (
    FINK_NORMALIZED_SCHEMA,
    HorizonsCache,
    ZTFExternalError,
    _parse_horizons_target_identity,
    fetch_horizons_cache,
    fink_rows_to_epoch,
    normalize_fink_rows,
)

MAPPED_FINK_SCHEMA = "delphi.k3-mapped-fink-normalized.v1"
MAPPED_INGEST_SCHEMA = "delphi.k3-mapped-fink-ingest-manifest.v1"
MAPPED_HORIZONS_SCHEMA = "delphi.k3-mapped-horizons-manifest.v1"
MAPPED_PREPARED_SCHEMA = "delphi.k3-mapped-ztf-prepared.v1"
MAPPED_PREPARED_MANIFEST_SCHEMA = "delphi.k3-mapped-ztf-prepared-manifest.v1"
# Verified against the public current-host Swagger schema on 2026-09-11.
# The legacy api.fink-portal.org host is retained only in historical receipts.
FINK_SSO_API_URL = "https://api.ztf.fink-portal.org/api/v1/sso"
_PROVISIONAL_DESIGNATION = re.compile(r"^\d{4}[A-Z]{1,3}\d{0,3}$")


class MappedZTFError(ValueError):
    """Raised when an internal DAMIT identity is not safely bound to MPC data."""


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_new_json(path: Path, value: object) -> None:
    if path.exists():
        raise MappedZTFError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_new_bytes(path: Path, value: bytes) -> None:
    """Create a receipt payload once; never replace an earlier response."""
    if path.exists():
        raise MappedZTFError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def _read_json(path: Path, description: str) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MappedZTFError(f"cannot read {description}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise MappedZTFError(f"{description} must be a JSON object")
    return value


def _load_identity_map(path: Path) -> tuple[Mapping[str, object], dict[str, Mapping[str, object]]]:
    document = _read_json(path, "survey identity map")
    required = {"schema", "identity_table_sha256", "splits_sha256", "objects"}
    if document.get("schema") != IDENTITY_MAP_SCHEMA or not required.issubset(document):
        raise MappedZTFError("survey identity map schema is invalid")
    if not isinstance(document["objects"], list) or not isinstance(document["identity_table_sha256"], str):
        raise MappedZTFError("survey identity map is malformed")
    rows: dict[str, Mapping[str, object]] = {}
    for row in document["objects"]:
        if not isinstance(row, Mapping):
            raise MappedZTFError("survey identity map object row is invalid")
        try:
            object_id, damit_id, number, fold = (
                str(row["object_id"]), int(row["damit_id"]), int(row["mpc_number"]), int(row["held_out_fold"])
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MappedZTFError("survey identity map requires a numbered mapped object") from exc
        if object_id != f"asteroid_{damit_id}" or damit_id <= 0 or number <= 0 or fold not in range(5):
            raise MappedZTFError("survey identity map object has inconsistent namespaces")
        if object_id in rows:
            raise MappedZTFError("survey identity map has duplicate internal object IDs")
        rows[object_id] = row
    if len(rows) != 170:
        raise MappedZTFError("survey identity map must cover all 170 frozen internal objects")
    return document, rows


def _binding(row: Mapping[str, object], *, identity_map_sha256: str, identity_table_sha256: str) -> dict[str, object]:
    return {
        "schema": BINDING_SCHEMA,
        "damit_id": int(row["damit_id"]),
        "mpc_number": int(row["mpc_number"]),
        "object_id": str(row["object_id"]),
        "identity_map_sha256": identity_map_sha256,
        "identity_table_sha256": identity_table_sha256,
        "resolved_mpc_number": int(row["mpc_number"]),
        "held_out_fold": int(row["held_out_fold"]),
    }


def _validate_binding(value: Mapping[str, object], expected: Mapping[str, object], *, map_hash: str, table_hash: str) -> Mapping[str, object]:
    binding = value.get("identity_binding")
    wanted = _binding(expected, identity_map_sha256=map_hash, identity_table_sha256=table_hash)
    if binding != wanted:
        raise MappedZTFError("input lacks the exact mapped DAMIT-to-MPC identity binding")
    if value.get("object_id") != expected["object_id"] or value.get("held_out_fold") != expected["held_out_fold"]:
        raise MappedZTFError("mapped input internal object or held-out fold differs from identity map")
    return binding


def _periods(
    path: Path, identities: Mapping[str, Mapping[str, object]], *, identity_map_sha256: str
) -> tuple[Mapping[str, object], dict[str, Mapping[str, object]]]:
    document = _read_json(path, "ZTF period manifest")
    if document.get("schema") != "delphi.k3-mapped-ztf-period-manifest.v1":
        raise MappedZTFError("ZTF period manifest schema is invalid")
    if document.get("identity_map_sha256") != identity_map_sha256:
        raise MappedZTFError("ZTF period manifest is not bound to the identity map")
    rows: dict[str, Mapping[str, object]] = {}
    values = document.get("objects")
    if not isinstance(values, list):
        raise MappedZTFError("ZTF period manifest object list is invalid")
    for row in values:
        if not isinstance(row, Mapping) or not isinstance(row.get("object_id"), str):
            raise MappedZTFError("ZTF period manifest object row is invalid")
        object_id = str(row["object_id"])
        if object_id not in identities or object_id in rows:
            raise MappedZTFError("ZTF period manifest does not use unique internal object IDs")
        if (
            row.get("held_out_fold") != identities[object_id]["held_out_fold"]
            or row.get("mpc_number") != identities[object_id]["mpc_number"]
        ):
            raise MappedZTFError("ZTF period manifest fold disagrees with identity map")
        try:
            period = float(row["known_period_hours"])
        except (KeyError, TypeError, ValueError) as exc:
            raise MappedZTFError("ZTF period manifest period is invalid") from exc
        if period <= 0:
            raise MappedZTFError("ZTF period manifest period must be positive")
        rows[object_id] = row
    return document, rows


def _decode_fink_records(body: bytes) -> list[Mapping[str, object]]:
    """Accept the documented JSON records response without pandas coercion."""
    try:
        value = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MappedZTFError("Fink response is not JSON") from exc
    if isinstance(value, list) and all(isinstance(row, Mapping) for row in value):
        return list(value)
    # Pandas' column-oriented JSON is also emitted by some Fink deployments.
    if isinstance(value, Mapping) and value and all(isinstance(column, Mapping) for column in value.values()):
        indices = sorted({str(index) for column in value.values() for index in column})
        return [{key: column.get(index) for key, column in value.items()} for index in indices]
    raise MappedZTFError("Fink JSON response must be records or column-oriented rows")


def _canonical_provisional_aliases(
    rows: Sequence[Mapping[str, object]], identity: Mapping[str, object]
) -> tuple[list[Mapping[str, object]], list[dict[str, object]]]:
    """Bind named Fink targets to the map; canonicalize only provisional spacing."""
    designation = identity.get("designation")
    name = identity.get("name")
    expected = designation.strip() if isinstance(designation, str) and designation.strip() else (
        name.strip() if isinstance(name, str) and name.strip() else None
    )
    if expected is None:
        return list(rows), []
    mapped_form = re.sub(r"\s+", "", expected).upper()
    is_provisional = bool(_PROVISIONAL_DESIGNATION.fullmatch(mapped_form))
    normalized: list[Mapping[str, object]] = []
    transformations: list[dict[str, object]] = []
    for index, value in enumerate(rows):
        if not isinstance(value, Mapping):
            normalized.append(value)
            continue
        source_designation, source_name = value.get("i:ssnamenr"), value.get("sso_name")
        if not isinstance(source_designation, str) or source_designation.strip().isdecimal():
            normalized.append(value)
            continue
        if not isinstance(source_name, str):
            raise MappedZTFError("named Fink target lacks sso_name for identity-map validation")
        designation_form = re.sub(r"\s+", "", source_designation).upper()
        name_form = re.sub(r"\s+", "", source_name).upper()
        if designation_form != mapped_form or name_form != mapped_form:
            raise MappedZTFError("Fink named target differs from the identity-map designation")
        if not is_provisional:
            normalized.append(value)
            continue
        # The shared normalizer intentionally compares named aliases exactly.
        # This temporary value is constrained to the identity-map-declared
        # provisional designation; originals remain in raw bytes and audit
        # metadata below.
        replacement = dict(value)
        replacement["sso_name"] = source_designation.strip()
        normalized.append(replacement)
        transformations.append(
            {
                "source_row_index": index,
                "identity_map_designation": expected,
                "canonical_comparison_form": mapped_form,
                "source_i:ssnamenr": source_designation,
                "source_sso_name": source_name,
                "normalizer_sso_name": replacement["sso_name"],
            }
        )
    return normalized, transformations


def _normalize_mapped_fink_rows(
    identity: Mapping[str, object], rows: Sequence[Mapping[str, object]], *, maximum_sigma_magnitude: float = 0.2,
) -> tuple[list[dict[str, object]], dict[str, int], list[dict[str, object]]]:
    number = int(identity["mpc_number"])
    normalized_source, transformations = _canonical_provisional_aliases(rows, identity)
    normalized_rows, rejected = normalize_fink_rows(
        f"asteroid_{number}", normalized_source, maximum_sigma_magnitude=maximum_sigma_magnitude
    )
    return normalized_rows, rejected, transformations


def _validate_fink_acquisition_rows(rows: Sequence[Mapping[str, object]], identity: Mapping[str, object]) -> tuple[bool, list[dict[str, object]]]:
    """Validate target identity; return whether the frozen photometry filter retains a row."""
    required = {"i:ssnamenr", "sso_name", "i:ra", "i:dec", "i:jd", "i:magpsf", "i:sigmapsf", "i:fid"}
    for index, row in enumerate(rows):
        if not required.issubset(row) or not isinstance(row.get("sso_name"), str) or not row["sso_name"].strip():
            raise MappedZTFError(f"Fink row {index} lacks required identity/photometry fields")
    # This is intentionally called with the physical alias, never an internal
    # DAMIT suffix.  It checks numeric designation rows and named aliases.
    number = int(identity["mpc_number"])
    normalized_source, transformations = _canonical_provisional_aliases(rows, identity)
    try:
        normalized, _ = normalize_fink_rows(f"asteroid_{number}", normalized_source)
    except ZTFExternalError as exc:
        # A response can identify the requested physical object correctly but
        # have no r-band, sigma-qualified detections.  That is coverage/data
        # availability, not an endpoint or namespace failure.
        if str(exc) == f"no retained Fink detections for asteroid_{number}":
            return False, transformations
        raise MappedZTFError(f"Fink response failed target/photometry validation: {exc}") from exc
    return bool(normalized), transformations


def fetch_mapped_fink(
    *, identity_map_path: str | Path, output_directory: str | Path, object_ids: Sequence[str] | None = None,
    timeout_seconds: float = 60.0, request_interval_seconds: float = 1.0,
) -> dict[str, object]:
    """Fetch explicit MPC targets with per-target immutable receipts.

    Empty replies and HTTP failures are retained as such.  They are never
    interpreted as a reason to fall back to an internal DAMIT number.
    """
    identity_path, output = Path(identity_map_path), Path(output_directory)
    identity_map, identities = _load_identity_map(identity_path)
    if timeout_seconds <= 0 or request_interval_seconds < 1.0:
        raise MappedZTFError("Fink fetch timeout must be positive and interval at least one second")
    selected = tuple(object_ids) if object_ids is not None else tuple(identities)
    if not selected or len(set(selected)) != len(selected) or any(item not in identities for item in selected):
        raise MappedZTFError("Fink fetch object IDs must be unique mapped internal identities")
    output.mkdir(parents=True, exist_ok=True)
    map_hash = _sha256(identity_path)
    results: list[dict[str, object]] = []
    for position, object_id in enumerate(sorted(selected, key=lambda item: int(identities[item]["damit_id"]))):
        identity = identities[object_id]
        number = int(identity["mpc_number"])
        receipt_path, raw_path, error_path = output / "receipts" / f"{number}.json", output / "raw" / f"{number}.json", output / "errors" / f"{number}.txt"
        if receipt_path.exists():
            receipt = _read_json(receipt_path, "existing Fink fetch receipt")
            if receipt.get("object_id") != object_id or receipt.get("mpc_number") != number or receipt.get("identity_map_sha256") != map_hash:
                raise MappedZTFError("existing Fink receipt has a conflicting identity binding")
            response_relative = receipt.get("response_path")
            if response_relative is not None:
                if not isinstance(response_relative, str) or not isinstance(receipt.get("response_sha256"), str):
                    raise MappedZTFError("existing Fink receipt response body metadata is invalid")
                existing_response = (output / response_relative).resolve()
                if (
                    not existing_response.is_relative_to(output.resolve())
                    or not existing_response.is_file()
                    or _sha256(existing_response) != receipt.get("response_sha256")
                ):
                    raise MappedZTFError("existing Fink receipt response body hash mismatch")
            error_relative = receipt.get("error_path")
            if error_relative is not None:
                if not isinstance(error_relative, str):
                    raise MappedZTFError("existing Fink receipt error body path is invalid")
                error_hash = receipt.get("error_sha256", receipt.get("response_sha256"))
                if not isinstance(error_hash, str):
                    raise MappedZTFError("existing Fink receipt error body hash is invalid")
                existing_error = (output / error_relative).resolve()
                if (
                    not existing_error.is_relative_to(output.resolve())
                    or not existing_error.is_file()
                    or _sha256(existing_error) != error_hash
                ):
                    raise MappedZTFError("existing Fink receipt error body hash mismatch")
            if receipt.get("status") in {"received_validated", "received_no_retained_rows"}:
                raw_relative, raw_hash = receipt.get("raw_path"), receipt.get("raw_sha256")
                if not isinstance(raw_relative, str) or not isinstance(raw_hash, str):
                    raise MappedZTFError("existing successful Fink receipt lacks raw integrity metadata")
                existing_raw = (output / raw_relative).resolve()
                if not existing_raw.is_relative_to(output.resolve()) or not existing_raw.is_file() or _sha256(existing_raw) != raw_hash:
                    raise MappedZTFError("existing successful Fink receipt raw response hash mismatch")
            results.append(dict(receipt))
            continue
        payload = json.dumps({"n_or_d": str(number), "withEphem": True, "output-format": "json"}).encode("utf-8")
        request = Request(FINK_SSO_API_URL, data=payload, headers={"Content-Type": "application/json", "User-Agent": "DeLPHI-K3-mapped-preparation/1.0"}, method="POST")
        status, body, error = "failed_http", b"", None
        http_status: int | None = None
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                body, http_status, status = response.read(), int(response.status), "received"
        except HTTPError as exc:
            body, http_status, error = exc.read(), int(exc.code), str(exc)
        except (URLError, TimeoutError, OSError) as exc:
            error = str(exc)
        common = {"object_id": object_id, "damit_id": int(identity["damit_id"]), "mpc_number": number, "held_out_fold": int(identity["held_out_fold"]), "identity_map_sha256": map_hash, "endpoint": FINK_SSO_API_URL, "request_payload_sha256": hashlib.sha256(payload).hexdigest(), "http_status": http_status, "response_sha256": hashlib.sha256(body).hexdigest()}
        if status == "received":
            response_path = output / "responses" / f"{number}.json"
            _write_new_bytes(response_path, body)
            common = {**common, "response_path": f"responses/{number}.json"}
            try:
                rows = _decode_fink_records(body)
                if not rows:
                    receipt = {**common, "status": "empty_response", "raw_path": None, "row_count": 0}
                else:
                    retained, transformations = _validate_fink_acquisition_rows(rows, identity)
                    _write_new_json(raw_path, rows)
                    receipt = {
                        **common,
                        "status": "received_validated" if retained else "received_no_retained_rows",
                        "raw_path": f"raw/{number}.json",
                        "raw_sha256": _sha256(raw_path),
                        "row_count": len(rows),
                        "identity_alias_normalizations": transformations,
                    }
            except MappedZTFError as exc:
                _write_new_bytes(error_path, body)
                receipt = {**common, "status": "invalid_response", "error": str(exc), "error_path": f"errors/{number}.txt", "error_sha256": _sha256(error_path), "row_count": None}
        else:
            _write_new_bytes(error_path, body)
            receipt = {**common, "status": "failed_http", "error": error, "error_path": f"errors/{number}.txt", "error_sha256": _sha256(error_path), "row_count": None}
        _write_new_json(receipt_path, receipt)
        results.append(receipt)
        if position + 1 < len(selected):
            time.sleep(request_interval_seconds)
    return {"schema": "delphi.k3-mapped-fink-fetch-batch.v1", "identity_map_sha256": map_hash, "requested_object_count": len(selected), "received_validated_count": sum(row["status"] == "received_validated" for row in results), "received_no_retained_rows_count": sum(row["status"] == "received_no_retained_rows" for row in results), "empty_response_count": sum(row["status"] == "empty_response" for row in results), "invalid_response_count": sum(row["status"] == "invalid_response" for row in results), "failed_http_count": sum(row["status"] == "failed_http" for row in results), "objects": results}


def ingest_mapped_fink(
    *, identity_map_path: str | Path, period_manifest_path: str | Path,
    raw_directory: str | Path, output_directory: str | Path, maximum_sigma_magnitude: float = 0.2,
    fetch_receipts_directory: str | Path | None = None,
    fetch_receipt_override_directories: Sequence[str | Path] = (), raw_override_directories: Sequence[str | Path] = (),
) -> dict[str, object]:
    """Normalize local MPC-numbered Fink files while preserving internal join IDs."""
    identity_path, period_path, raw_root, output = map(Path, (identity_map_path, period_manifest_path, raw_directory, output_directory))
    receipt_root = Path(fetch_receipts_directory) if fetch_receipts_directory is not None else None
    override_roots = tuple(Path(path) for path in fetch_receipt_override_directories)
    raw_override_roots = tuple(Path(path) for path in raw_override_directories)
    if output.exists() or not raw_root.is_dir() or (receipt_root is not None and not receipt_root.is_dir()) or any(not root.is_dir() for root in override_roots) or any(not root.is_dir() for root in raw_override_roots):
        raise MappedZTFError("mapped Fink output must be new and raw directory must exist")
    identity_map, identities = _load_identity_map(identity_path)
    map_hash, table_hash = _sha256(identity_path), str(identity_map["identity_table_sha256"])
    _, periods = _periods(
        period_path, identities, identity_map_sha256=map_hash
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    records: list[dict[str, object]] = []
    try:
        for object_id, identity in sorted(identities.items(), key=lambda item: int(item[1]["damit_id"])):
            number = int(identity["mpc_number"])
            raw_candidates = tuple(root / f"{number}.json" for root in (*raw_override_roots, raw_root))
            raw_path = next((path for path in raw_candidates if path.is_file()), raw_root / f"{number}.json")
            common = {"object_id": object_id, "damit_id": int(identity["damit_id"]), "mpc_number": number, "held_out_fold": int(identity["held_out_fold"]), "source_filename": raw_path.name}
            receipt_common: dict[str, object] = {}
            if object_id not in periods:
                records.append({**common, "status": "missing_period_manifest_entry"})
                continue
            if receipt_root is not None:
                receipt_candidates = tuple(root / f"{number}.json" for root in (*override_roots, receipt_root))
                receipt_path = next((path for path in receipt_candidates if path.is_file()), None)
                if receipt_path is None:
                    raise MappedZTFError(f"missing immutable Fink receipt for MPC {number}")
                receipt = _read_json(receipt_path, "Fink fetch receipt")
                if (
                    receipt.get("object_id") != object_id
                    or receipt.get("damit_id") != int(identity["damit_id"])
                    or receipt.get("mpc_number") != number
                    or receipt.get("held_out_fold") != int(identity["held_out_fold"])
                    or receipt.get("identity_map_sha256") != map_hash
                ):
                    raise MappedZTFError("Fink fetch receipt identity differs from mapped object")
                receipt_status = receipt.get("status")
                receipt_common = {
                    "fetch_receipt_path": str(receipt_path),
                    "fetch_receipt_sha256": _sha256(receipt_path),
                }
                if receipt_status != "received_validated":
                    if receipt_status == "received_no_retained_rows":
                        if not raw_path.is_file() or _sha256(raw_path) != receipt.get("raw_sha256"):
                            raise MappedZTFError("no-retained Fink response raw hash mismatch")
                        records.append({**common, **receipt_common, "status": "fetched_no_retained_detections"})
                    elif receipt_status in {"empty_response", "invalid_response", "failed_http"}:
                        records.append({**common, **receipt_common, "status": f"fetched_{receipt_status}"})
                    else:
                        raise MappedZTFError("Fink fetch receipt status is not recognized")
                    continue
                if (
                    receipt.get("raw_path") != f"raw/{number}.json"
                    or not raw_path.is_file()
                    or _sha256(raw_path) != receipt.get("raw_sha256")
                ):
                    raise MappedZTFError("validated Fink response raw hash mismatch")
            if not raw_path.is_file():
                records.append({**common, "status": "missing_local_raw_may_be_fetchable"})
                continue
            try:
                raw_rows = json.loads(raw_path.read_text(encoding="utf-8"))
                if not isinstance(raw_rows, list):
                    raise MappedZTFError("raw Fink response must be a list")
                acquisition_alias = f"asteroid_{number}"
                rows, rejected, transformations = _normalize_mapped_fink_rows(
                    identity, raw_rows, maximum_sigma_magnitude=maximum_sigma_magnitude
                )
            except (OSError, json.JSONDecodeError, ZTFExternalError, MappedZTFError) as exc:
                records.append({**common, "status": "invalid_local_raw", "error": str(exc)})
                continue
            binding = _binding(identity, identity_map_sha256=map_hash, identity_table_sha256=table_hash)
            payload = {
                "schema": MAPPED_FINK_SCHEMA, "object_id": object_id, "held_out_fold": int(identity["held_out_fold"]),
                "identity_binding": binding, "acquisition_alias": acquisition_alias,
                "source_filename": raw_path.name, "source_sha256": _sha256(raw_path),
                "raw_row_count": len(raw_rows), "retained_row_count": len(rows), "rejected_row_counts": rejected,
                "identity_alias_normalizations": transformations,
                **receipt_common,
                "filter": {"fid": 2, "band": "ztf_r", "maximum_sigma_magnitude": maximum_sigma_magnitude}, "rows": rows,
            }
            destination = staging / "objects" / f"{object_id}.json"
            _write_new_json(destination, payload)
            records.append({**common, **receipt_common, "status": "ready", "normalized_path": f"objects/{object_id}.json", "normalized_sha256": _sha256(destination), "retained_row_count": len(rows), "source_sha256": payload["source_sha256"]})
        ready = [row for row in records if row["status"] == "ready"]
        manifest = {
            "schema": MAPPED_INGEST_SCHEMA, "identity_map_sha256": map_hash, "identity_table_sha256": table_hash,
            "period_manifest_sha256": _sha256(period_path), "period_manifest_object_count": len(periods),
            "raw_directory": str(raw_root), "expected_object_count": len(identities), "ready_object_count": len(ready),
            "raw_override_directories": [str(root) for root in raw_override_roots],
            "fetch_receipts_directory": str(receipt_root) if receipt_root is not None else None,
            "fetch_receipt_override_directories": [str(root) for root in override_roots],
            "missing_local_raw_count": sum(row["status"] == "missing_local_raw_may_be_fetchable" for row in records),
            "fetched_no_retained_count": sum(row["status"] == "fetched_no_retained_detections" for row in records),
            "fetched_empty_response_count": sum(row["status"] == "fetched_empty_response" for row in records),
            "fetched_invalid_response_count": sum(row["status"] == "fetched_invalid_response" for row in records),
            "fetched_failed_http_count": sum(row["status"] == "fetched_failed_http" for row in records),
            "missing_period_count": sum(row["status"] == "missing_period_manifest_entry" for row in records),
            "invalid_local_raw_count": sum(row["status"] == "invalid_local_raw" for row in records), "objects": records,
        }
        _write_new_json(staging / "manifest.json", manifest)
        os.replace(staging, output)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _mapped_normalized(path: Path, identity: Mapping[str, object], map_hash: str, table_hash: str) -> Mapping[str, object]:
    value = _read_json(path, "mapped normalized Fink object")
    if value.get("schema") != MAPPED_FINK_SCHEMA:
        raise MappedZTFError("legacy unbound Fink input is not accepted")
    _validate_binding(value, identity, map_hash=map_hash, table_hash=table_hash)
    alias = f"asteroid_{int(identity['mpc_number'])}"
    if value.get("acquisition_alias") != alias or not isinstance(value.get("rows"), list):
        raise MappedZTFError("mapped normalized acquisition alias or rows are invalid")
    return value


def _validate_cache(cache_path: Path, *, mpc_number: int, rows: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    cache = _read_json(cache_path, "mapped Horizons cache")
    parsed = HorizonsCache.from_mapping(cache)
    metadata = parsed.query_metadata
    target = metadata.get("resolved_target_identity")
    if metadata.get("asteroid_number") != mpc_number or not isinstance(target, Mapping) or target.get("asteroid_number") != mpc_number:
        raise MappedZTFError("Horizons cache target differs from mapped MPC number")
    if [float(row["jd"]) for row in parsed.rows] != [float(row["jd"]) for row in rows]:
        raise MappedZTFError("Horizons cache epochs differ from mapped Fink rows")
    return cache


def _validated_reuse_cache(
    roots: Sequence[Path], *, object_id: str, mpc_number: int, rows: Sequence[Mapping[str, object]]
) -> Path | None:
    """Return an old cache only when its raw bytes prove the same MPC target."""
    for root in roots:
        # Old caches use physical acquisition aliases; new mapped caches use
        # the internal prepared-object ID.  Both candidates are accepted only
        # after the same physical-target, epoch, and raw-byte validation.
        candidates = (root / f"asteroid_{mpc_number}" / "cache.json", root / object_id / "cache.json")
        for candidate in candidates:
            if not candidate.is_file():
                continue
            try:
                cache = _validate_cache(candidate, mpc_number=mpc_number, rows=rows)
                responses = cache["query_metadata"].get("raw_responses")
                if not isinstance(responses, list) or not responses:
                    continue
                for response in responses:
                    if not isinstance(response, Mapping) or not isinstance(response.get("path"), str):
                        raise MappedZTFError("reused Horizons cache raw response metadata is invalid")
                    raw_path = (candidate.parent / str(response["path"])).resolve()
                    if not raw_path.is_relative_to(candidate.parent.resolve()) or _sha256(raw_path) != response.get("sha256"):
                        raise MappedZTFError("reused Horizons cache raw response hash mismatch")
                    if _parse_horizons_target_identity(raw_path.read_bytes()).get("asteroid_number") != mpc_number:
                        raise MappedZTFError("reused Horizons raw target differs from mapped MPC number")
                return candidate.parent
            except (MappedZTFError, OSError, ZTFExternalError):
                continue
    return None


def fetch_mapped_horizons(
    *, ingest_manifest_path: str | Path, identity_map_path: str | Path, output_directory: str | Path,
    batch_size: int = 20, timeout_seconds: float = 120.0, request_interval_seconds: float = 1.0,
    reuse_horizons_roots: Sequence[str | Path] = (),
) -> dict[str, object]:
    """Fetch or safely resume MPC-targeted Horizons geometry for ready mapped rows."""
    manifest_path, identity_path, output = map(Path, (ingest_manifest_path, identity_map_path, output_directory))
    source = _read_json(manifest_path, "mapped Fink manifest")
    identity_map, identities = _load_identity_map(identity_path)
    map_hash, table_hash = _sha256(identity_path), str(identity_map["identity_table_sha256"])
    if timeout_seconds <= 0 or request_interval_seconds < 1.0:
        raise MappedZTFError("Horizons timeout must be positive and interval at least one second")
    if source.get("schema") != MAPPED_INGEST_SCHEMA or source.get("identity_map_sha256") != map_hash:
        raise MappedZTFError("mapped Fink manifest is not bound to this identity map")
    final_manifest = output / "manifest.json"
    if final_manifest.exists():
        raise MappedZTFError("refusing to overwrite mapped Horizons manifest")
    output.mkdir(parents=True, exist_ok=True)
    reuse_roots = tuple(Path(value).resolve() for value in reuse_horizons_roots)
    results: list[dict[str, object]] = []
    for record in source.get("objects", []):
        if not isinstance(record, Mapping) or record.get("status") != "ready":
            continue
        object_id = str(record["object_id"])
        identity = identities.get(object_id)
        if identity is None:
            raise MappedZTFError("mapped Fink manifest contains unknown internal object")
        normalized_path = manifest_path.parent / str(record["normalized_path"])
        if _sha256(normalized_path) != record.get("normalized_sha256"):
            raise MappedZTFError("mapped normalized Fink hash mismatch")
        normalized = _mapped_normalized(normalized_path, identity, map_hash, table_hash)
        cache_dir, cache_path = output / object_id, output / object_id / "cache.json"
        common = {"object_id": object_id, "damit_id": int(identity["damit_id"]), "mpc_number": int(identity["mpc_number"]), "held_out_fold": int(identity["held_out_fold"])}
        try:
            if cache_path.is_file():
                cache = _validate_cache(cache_path, mpc_number=int(identity["mpc_number"]), rows=normalized["rows"])
                status = "resumed_validated_cache"
            elif cache_dir.exists():
                raise MappedZTFError("incomplete mapped Horizons cache requires manual audit")
            else:
                reused = _validated_reuse_cache(
                    reuse_roots, object_id=object_id, mpc_number=int(identity["mpc_number"]), rows=normalized["rows"]
                )
                if reused is not None:
                    shutil.copytree(reused, cache_dir)
                    cache = _validate_cache(cache_path, mpc_number=int(identity["mpc_number"]), rows=normalized["rows"])
                    status = "reused_physical_mpc_cache"
                else:
                    # This is the sole suffix-parser call path: the temporary source
                    # has an explicitly named physical MPC acquisition alias.
                    alias = str(normalized["acquisition_alias"])
                    alias_path = output / "acquisition-aliases" / f"{alias}.json"
                    alias_payload = {"schema": FINK_NORMALIZED_SCHEMA, "object_id": alias, "rows": normalized["rows"]}
                    if alias_path.exists():
                        if _read_json(alias_path, "existing acquisition alias") != alias_payload:
                            raise MappedZTFError("existing acquisition alias does not match mapped Fink rows")
                    else:
                        _write_new_json(alias_path, alias_payload)
                    cache = fetch_horizons_cache(normalized_object_path=alias_path, output_directory=cache_dir, batch_size=batch_size, timeout_seconds=timeout_seconds, request_interval_seconds=request_interval_seconds)
                    cache = _validate_cache(cache_path, mpc_number=int(identity["mpc_number"]), rows=normalized["rows"])
                    status = "fetched"
            results.append({**common, "status": status, "cache_path": f"{object_id}/cache.json", "cache_sha256": _sha256(cache_path), "query_metadata_sha256": cache["query_metadata_sha256"], "rows_sha256": cache["rows_sha256"]})
        except (MappedZTFError, ZTFExternalError, OSError) as exc:
            failure_path = output / "failures" / f"{object_id}.json"
            _write_new_json(failure_path, {**common, "status": "failed_horizons", "error": str(exc)})
            results.append({**common, "status": "failed_horizons", "failure_path": f"failures/{object_id}.json", "failure_sha256": _sha256(failure_path)})
    result = {"schema": MAPPED_HORIZONS_SCHEMA, "identity_map_sha256": map_hash, "identity_table_sha256": table_hash, "ingest_manifest_sha256": _sha256(manifest_path), "reuse_roots": [str(path) for path in reuse_roots], "expected_object_count": int(source["expected_object_count"]), "mapped_fink_ready_count": int(source["ready_object_count"]), "missing_local_raw_count": int(source["missing_local_raw_count"]), "missing_period_count": int(source["missing_period_count"]), "ready_object_count": sum(row["status"] != "failed_horizons" for row in results), "failed_horizons_count": sum(row["status"] == "failed_horizons" for row in results), "reused_object_count": sum(row["status"] == "reused_physical_mpc_cache" for row in results), "fetched_object_count": sum(row["status"] == "fetched" for row in results), "objects": results}
    _write_new_json(final_manifest, result)
    return result


def prepare_mapped_ztf(
    *, ingest_manifest_path: str | Path, horizons_manifest_path: str | Path, period_manifest_path: str | Path,
    identity_map_path: str | Path, output_directory: str | Path,
) -> dict[str, object]:
    """Join mapped photometry/geometry to internal periods and emit bound inputs."""
    ingest_path, horizons_path, period_path, identity_path, output = map(Path, (ingest_manifest_path, horizons_manifest_path, period_manifest_path, identity_map_path, output_directory))
    if output.exists():
        raise MappedZTFError("mapped prepared output directory must not already exist")
    identity_map, identities = _load_identity_map(identity_path)
    map_hash, table_hash = _sha256(identity_path), str(identity_map["identity_table_sha256"])
    _, periods = _periods(
        period_path, identities, identity_map_sha256=map_hash
    )
    ingest, horizons = _read_json(ingest_path, "mapped Fink manifest"), _read_json(horizons_path, "mapped Horizons manifest")
    if ingest.get("schema") != MAPPED_INGEST_SCHEMA or horizons.get("schema") != MAPPED_HORIZONS_SCHEMA or ingest.get("identity_map_sha256") != map_hash or horizons.get("identity_map_sha256") != map_hash:
        raise MappedZTFError("mapped preparation inputs are not bound to the identity map")
    ingest_rows = ingest.get("objects")
    if not isinstance(ingest_rows, list):
        raise MappedZTFError("mapped Fink manifest object list is invalid")
    ingest_ids = [row.get("object_id") for row in ingest_rows if isinstance(row, Mapping)]
    if (
        len(ingest_rows) != len(identities)
        or len(ingest_ids) != len(identities)
        or set(ingest_ids) != set(identities)
    ):
        raise MappedZTFError("mapped Fink manifest must retain exactly one row per internal object")
    horizon_rows = {str(row["object_id"]): row for row in horizons.get("objects", []) if isinstance(row, Mapping)}
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    records: list[dict[str, object]] = []
    try:
        for record in ingest_rows:
            if not isinstance(record, Mapping) or not isinstance(record.get("object_id"), str):
                raise MappedZTFError("mapped Fink manifest object row is invalid")
            if record.get("status") != "ready":
                records.append(
                    {
                        "object_id": str(record["object_id"]),
                        "status": f"ingest_{record.get('status')}",
                    }
                )
                continue
            object_id, identity = str(record["object_id"]), identities[str(record["object_id"])]
            if object_id not in periods:
                records.append({"object_id": object_id, "status": "missing_period_manifest_entry"})
                continue
            horizon = horizon_rows.get(object_id)
            if horizon is None or horizon.get("status") not in {
                "fetched",
                "reused_physical_mpc_cache",
                "resumed_validated_cache",
            }:
                records.append({"object_id": object_id, "status": "missing_horizons_cache"})
                continue
            normalized_path = ingest_path.parent / str(record["normalized_path"])
            if _sha256(normalized_path) != record.get("normalized_sha256"):
                raise MappedZTFError("mapped normalized Fink hash mismatch")
            normalized = _mapped_normalized(normalized_path, identity, map_hash, table_hash)
            cache_path = horizons_path.parent / str(horizon["cache_path"])
            if _sha256(cache_path) != horizon.get("cache_sha256"):
                raise MappedZTFError("mapped Horizons cache hash mismatch")
            cache = _validate_cache(cache_path, mpc_number=int(identity["mpc_number"]), rows=normalized["rows"])
            epoch = fink_rows_to_epoch(object_id, normalized["rows"], cache)
            # K3 tokenization deliberately requires at least two points per
            # source epoch.  The lower-level epoch type permits one point, so
            # make the availability exclusion explicit here after all mapped
            # identity, hash, and Horizons-geometry checks have succeeded.
            if len(epoch.observations) < 2:
                records.append(
                    {
                        "object_id": object_id,
                        "status": "insufficient_observations",
                        "retained_observation_count": len(epoch.observations),
                    }
                )
                continue
            target = cache["query_metadata"]["resolved_target_identity"]
            binding = _binding(identity, identity_map_sha256=map_hash, identity_table_sha256=table_hash)
            period = periods[object_id]
            payload = {"schema": MAPPED_PREPARED_SCHEMA, "object_id": object_id, "held_out_fold": int(identity["held_out_fold"]), "identity_binding": binding, "resolved_target_identity": target, "known_period_hours": float(period["known_period_hours"]), "period_provenance": period.get("period_provenance"), "epochs": [{"epoch_id": epoch.epoch_id, "observations": [item.__dict__ for item in epoch.observations]}], "horizons_query_metadata_sha256": cache["query_metadata_sha256"], "horizons_rows_sha256": cache["rows_sha256"]}
            destination = staging / "objects" / f"{object_id}.json"
            _write_new_json(destination, payload)
            records.append({"object_id": object_id, "status": "ready", "path": f"objects/{object_id}.json", "sha256": _sha256(destination), "held_out_fold": int(identity["held_out_fold"]), "mpc_number": int(identity["mpc_number"])})
        ready = [row for row in records if row["status"] == "ready"]
        if len(records) != len(identities) or len({row["object_id"] for row in records}) != len(identities):
            raise MappedZTFError("mapped prepared manifest must retain exactly one row per internal object")
        manifest = {"schema": MAPPED_PREPARED_MANIFEST_SCHEMA, "identity_map_sha256": map_hash, "identity_table_sha256": table_hash, "ingest_manifest_sha256": _sha256(ingest_path), "horizons_manifest_sha256": _sha256(horizons_path), "period_manifest_sha256": _sha256(period_path), "expected_object_count": len(identities), "mapped_fink_ready_count": int(ingest["ready_object_count"]), "missing_local_raw_count": int(ingest["missing_local_raw_count"]), "period_manifest_object_count": len(periods), "ready_object_count": len(ready), "missing_period_count": sum(row["status"] == "missing_period_manifest_entry" for row in records), "missing_horizons_count": sum(row["status"] == "missing_horizons_cache" for row in records), "insufficient_observations_count": sum(row["status"] == "insufficient_observations" for row in records), "objects": records}
        _write_new_json(staging / "manifest.json", manifest)
        os.replace(staging, output)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
