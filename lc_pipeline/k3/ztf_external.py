"""Offline, fail-closed conversion of enriched Fink photometry for K3.

No ephemeris is fabricated here: callers must provide cached, hash-bound JPL
Horizons asteroid-centric ecliptic-J2000 vectors for every retained exposure.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import yaml

from ..v2.preprocessing import Observation, ObservationEpoch


class ZTFExternalError(ValueError):
    """Raised when external survey data are not physically auditable."""


FINK_NORMALIZED_SCHEMA = "delphi.k3-fink-normalized.v1"
FINK_INGEST_MANIFEST_SCHEMA = "delphi.k3-fink-ingest-manifest.v1"
HORIZONS_API_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"
HORIZONS_MANIFEST_SCHEMA = "delphi.k3-horizons-directory-manifest.v1"
HORIZONS_PLAN_SCHEMA = "delphi.k3-horizons-fetch-plan.v1"
HORIZONS_IDENTITY_AUDIT_SCHEMA = "delphi.k3-horizons-identity-audit.v1"
HORIZONS_JD_TOLERANCE_DAYS = 2e-9
HttpFetcher = Callable[[str, float], tuple[bytes, str, str | None, str | None]]


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_new_json(path: Path, value: object) -> None:
    if path.exists():
        raise ZTFExternalError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _number_from_object_id(value: object) -> int:
    if not isinstance(value, str) or not value.startswith("asteroid_"):
        raise ZTFExternalError(f"invalid asteroid object ID: {value!r}")
    suffix = value.removeprefix("asteroid_")
    if not suffix.isdigit() or int(suffix) <= 0:
        raise ZTFExternalError(f"invalid asteroid object ID: {value!r}")
    return int(suffix)


def _test_fold_identities(path: str | Path) -> dict[str, int]:
    """Load identity membership without opening the label-bearing catalog."""
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        folds = sorted(document["folds"], key=lambda row: int(row["fold"]))
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ZTFExternalError(f"cannot load frozen splits: {exc}") from exc
    if [int(row["fold"]) for row in folds] != list(range(5)):
        raise ZTFExternalError("frozen splits must contain folds zero through four")
    identities: dict[str, int] = {}
    for row in folds:
        fold = int(row["fold"])
        values = row.get("test_ids")
        if not isinstance(values, list):
            raise ZTFExternalError(f"fold {fold} has no test identity list")
        for object_id in values:
            _number_from_object_id(object_id)
            if object_id in identities:
                raise ZTFExternalError(f"duplicate held-out identity: {object_id}")
            identities[object_id] = fold
    if len(identities) != 170:
        raise ZTFExternalError(f"expected 170 held-out identities, found {len(identities)}")
    return identities


def _finite_float(row: Mapping[str, object], key: str, *, positive: bool = False) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ZTFExternalError(f"Fink row requires numeric {key}") from exc
    if not math.isfinite(value) or (positive and value <= 0):
        qualifier = "positive " if positive else ""
        raise ZTFExternalError(f"Fink row requires finite {qualifier}{key}")
    return value


def _fink_designation(
    row: Mapping[str, object], source_index: int, asteroid_number: int
) -> tuple[str, bool]:
    """Validate a Fink identity token without assuming it is always numeric.

    Historical Fink responses can switch ``i:ssnamenr`` between the numbered
    designation (for example ``"107"``) and the stable name (``"Camilla"``)
    within one response.  File selection from the frozen split is authoritative;
    every numeric token must still agree with it.  Named tokens are retained and,
    when Fink also supplies ``sso_name``, checked against that alias.
    """
    try:
        raw = row["i:ssnamenr"]
    except KeyError as exc:
        raise ZTFExternalError(
            f"Fink source row {source_index} lacks i:ssnamenr"
        ) from exc
    if isinstance(raw, bool):
        raise ZTFExternalError(f"Fink source row {source_index} has invalid i:ssnamenr")
    if isinstance(raw, int):
        numeric = raw
        designation = str(raw)
    elif isinstance(raw, float) and math.isfinite(raw) and raw.is_integer():
        numeric = int(raw)
        designation = str(numeric)
    elif isinstance(raw, str) and raw.strip():
        designation = raw.strip()
        numeric = int(designation) if designation.isdecimal() else None
    else:
        raise ZTFExternalError(f"Fink source row {source_index} has invalid i:ssnamenr")
    if numeric is not None:
        if numeric != asteroid_number:
            raise ZTFExternalError(
                f"Fink identity mismatch: expected {asteroid_number}, row reports {numeric}"
            )
        return designation, False
    alias = row.get("sso_name")
    if isinstance(alias, str) and alias.strip() and alias.strip().casefold() != designation.casefold():
        raise ZTFExternalError(
            f"Fink named designation mismatch at row {source_index}: "
            f"i:ssnamenr={designation!r}, sso_name={alias!r}"
        )
    return designation, True


def normalize_fink_rows(
    object_id: str,
    rows: Sequence[Mapping[str, object]],
    *,
    maximum_sigma_magnitude: float = 0.2,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Whitelist, normalize, and filter one legacy enriched Fink response.

    The returned rows contain no DAMIT solution, pole, or fold-training data.
    Original row indices are retained so every filtering decision is auditable.
    """
    asteroid_number = _number_from_object_id(object_id)
    if not math.isfinite(maximum_sigma_magnitude) or maximum_sigma_magnitude <= 0:
        raise ZTFExternalError("maximum_sigma_magnitude must be finite and positive")
    retained: list[dict[str, object]] = []
    counts = {
        "wrong_filter": 0,
        "invalid_photometry": 0,
        "sigma_rejected": 0,
        "named_designation_rows": 0,
    }
    named_designations: set[str] = set()
    for source_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ZTFExternalError(f"Fink source row {source_index} is not an object")
        designation, is_named = _fink_designation(row, source_index, asteroid_number)
        if is_named:
            counts["named_designation_rows"] += 1
            named_designations.add(designation.casefold())
        try:
            fid = int(row["i:fid"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ZTFExternalError(f"Fink source row {source_index} lacks integer fid") from exc
        if fid != 2:
            counts["wrong_filter"] += 1
            continue
        try:
            jd = _finite_float(row, "i:jd")
            magnitude = _finite_float(row, "i:magpsf")
            sigma = _finite_float(row, "i:sigmapsf", positive=True)
        except ZTFExternalError:
            counts["invalid_photometry"] += 1
            continue
        if sigma > maximum_sigma_magnitude:
            counts["sigma_rejected"] += 1
            continue
        dhelio = _finite_float(row, "Dhelio", positive=True)
        dobs = _finite_float(row, "Dobs", positive=True)
        phase = _finite_float(row, "Phase")
        if not 0 <= phase <= 180:
            raise ZTFExternalError(f"Fink source row {source_index} has invalid phase angle")
        normalized: dict[str, object] = {
            "source_row_index": source_index,
            "fink_ssnamenr": designation,
            "jd": jd,
            "fid": fid,
            "magpsf": magnitude,
            "sigmapsf": sigma,
            "Dhelio": dhelio,
            "Dobs": dobs,
            "phase": phase,
        }
        if "Date" in row:
            normalized["fink_ephemeris_jd"] = _finite_float(row, "Date")
        if "RA" in row or "DEC" in row:
            if "RA" not in row or "DEC" not in row:
                raise ZTFExternalError("Fink RA and DEC must occur together")
            normalized["ra"] = _finite_float(row, "RA") % 360.0
            normalized["dec"] = _finite_float(row, "DEC")
            if not -90 <= float(normalized["dec"]) <= 90:
                raise ZTFExternalError(f"Fink source row {source_index} has invalid declination")
        if "sso_name" in row and isinstance(row["sso_name"], str):
            normalized["sso_name"] = row["sso_name"]
        retained.append(normalized)
    if len(named_designations) > 1:
        raise ZTFExternalError(
            f"Fink response for {object_id} contains multiple named designations"
        )
    retained.sort(key=lambda row: (float(row["jd"]), int(row["source_row_index"])))
    jds = [float(row["jd"]) for row in retained]
    if len(jds) != len(set(jds)):
        raise ZTFExternalError(f"retained Fink JDs are not unique for {object_id}")
    if not retained:
        raise ZTFExternalError(f"no retained Fink detections for {object_id}")
    return retained, counts


def ingest_fink_directory(
    *,
    raw_directory: str | Path,
    splits_path: str | Path,
    study_spec_path: str | Path,
    output_directory: str | Path,
) -> dict[str, object]:
    """Create a deterministic, label-blind normalized Fink cohort offline."""
    raw_root = Path(raw_directory)
    splits = Path(splits_path)
    spec_path = Path(study_spec_path)
    destination = Path(output_directory)
    if destination.exists():
        raise ZTFExternalError("Fink ingest output directory must not already exist")
    if not raw_root.is_dir():
        raise ZTFExternalError("Fink raw directory does not exist")
    try:
        spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
        photometry = spec["ztf_cross_survey"]["photometry"]
        maximum_sigma = float(photometry["maximum_sigma_magnitude"])
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise ZTFExternalError(f"cannot load follow-up photometry policy: {exc}") from exc
    if photometry.get("filter") != "ztf_r" or photometry.get("epoch_policy") != "one_source_epoch_per_ztf_object":
        raise ZTFExternalError("unsupported frozen Fink filter or epoch policy")
    identities = _test_fold_identities(splits)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    manifest_rows: list[dict[str, object]] = []
    try:
        for object_id in sorted(identities, key=_number_from_object_id):
            number = _number_from_object_id(object_id)
            raw_path = raw_root / f"{number}.json"
            if not raw_path.is_file():
                manifest_rows.append(
                    {
                        "object_id": object_id,
                        "fold": identities[object_id],
                        "status": "missing_raw_file",
                        "source_filename": raw_path.name,
                    }
                )
                continue
            try:
                raw_rows = json.loads(raw_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ZTFExternalError(f"cannot parse {raw_path.name}: {exc}") from exc
            if not isinstance(raw_rows, list):
                raise ZTFExternalError(f"Fink raw file {raw_path.name} must contain a list")
            retained, rejected = normalize_fink_rows(
                object_id, raw_rows, maximum_sigma_magnitude=maximum_sigma
            )
            output_name = f"{object_id}.json"
            payload = {
                "schema": FINK_NORMALIZED_SCHEMA,
                "object_id": object_id,
                "asteroid_number": number,
                "held_out_fold": identities[object_id],
                "source": "Fink broker enriched ZTF response",
                "source_filename": raw_path.name,
                "source_sha256": _sha256_file(raw_path),
                "study_spec_sha256": _sha256_file(spec_path),
                "identity_binding": {
                    "authority": "frozen_split_identity_and_numbered_source_filename",
                    "numeric_i_ssnamenr_must_match": True,
                    "stable_named_i_ssnamenr_permitted_and_retained": True,
                },
                "filter": {
                    "fid": 2,
                    "band": "ztf_r",
                    "maximum_sigma_magnitude": maximum_sigma,
                },
                "raw_row_count": len(raw_rows),
                "retained_row_count": len(retained),
                "rejected_row_counts": rejected,
                "rows": retained,
            }
            output_path = staging / "objects" / output_name
            _write_new_json(output_path, payload)
            manifest_rows.append(
                {
                    "object_id": object_id,
                    "fold": identities[object_id],
                    "status": "ready",
                    "source_filename": raw_path.name,
                    "source_sha256": payload["source_sha256"],
                    "raw_row_count": len(raw_rows),
                    "retained_row_count": len(retained),
                    "normalized_path": f"objects/{output_name}",
                    "normalized_sha256": _sha256_file(output_path),
                }
            )
        ready = [row for row in manifest_rows if row["status"] == "ready"]
        manifest: dict[str, object] = {
            "schema": FINK_INGEST_MANIFEST_SCHEMA,
            "study_spec_sha256": _sha256_file(spec_path),
            "splits_sha256": _sha256_file(splits),
            "source": "existing local raw Fink/ZTF JSON; bytes are not rewritten",
            "selection_uses_labels": False,
            "expected_identity_count": len(identities),
            "ready_object_count": len(ready),
            "missing_object_count": len(identities) - len(ready),
            "retained_observation_count": sum(int(row["retained_row_count"]) for row in ready),
            "objects": manifest_rows,
        }
        _write_new_json(staging / "manifest.json", manifest)
        os.replace(staging, destination)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


@dataclass(frozen=True)
class _HorizonsResponse:
    body: bytes
    final_url: str
    http_date: str | None
    content_type: str | None


def _default_http_fetcher(url: str, timeout_seconds: float) -> tuple[bytes, str, str | None, str | None]:
    request = Request(url, headers={"User-Agent": "DeLPHI-K3-followup/1.0 (scientific cache)"})
    last_error: OSError | None = None
    for attempt in range(3):
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                return (
                    response.read(),
                    response.geturl(),
                    response.headers.get("Date"),
                    response.headers.get("Content-Type"),
                )
        except OSError as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(2**attempt)
    assert last_error is not None
    raise ZTFExternalError(f"JPL Horizons request failed after three attempts: {last_error}")


def horizons_query_parameters(
    asteroid_number: int, center: str, jds: Sequence[float]
) -> dict[str, str]:
    """Return the exact official Horizons vector-query settings."""
    if asteroid_number <= 0 or center not in {"500@10", "500@399"} or not jds:
        raise ZTFExternalError("invalid Horizons target, center, or empty epoch list")
    if len(jds) > 10_000 or not all(math.isfinite(float(jd)) for jd in jds):
        raise ZTFExternalError("Horizons TLIST requires at most 10,000 finite JDs")
    return {
        "format": "json",
        "COMMAND": f"'{asteroid_number};'",
        "OBJ_DATA": "'NO'",
        "MAKE_EPHEM": "'YES'",
        "EPHEM_TYPE": "'VECTORS'",
        "CENTER": f"'{center}'",
        "TLIST": " ".join(f"'{float(jd):.12f}'" for jd in jds),
        "TLIST_TYPE": "'JD'",
        "TIME_TYPE": "'UT'",
        "TIME_DIGITS": "'FRACSEC'",
        "REF_PLANE": "'ECLIPTIC'",
        "REF_SYSTEM": "'ICRF'",
        "OUT_UNITS": "'AU-D'",
        "VEC_TABLE": "'1'",
        "VEC_CORR": "'NONE'",
        "CSV_FORMAT": "'YES'",
        "VEC_LABELS": "'NO'",
    }


def _parse_horizons_vectors(payload: bytes, expected_jds: Sequence[float]) -> tuple[tuple[float, float, float], ...]:
    try:
        document = json.loads(payload)
        signature = document["signature"]
        result = str(document["result"])
    except (UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ZTFExternalError(f"invalid JPL Horizons JSON response: {exc}") from exc
    if document.get("error"):
        raise ZTFExternalError(f"JPL Horizons returned an error: {document['error']}")
    if not isinstance(signature, Mapping) or "NASA/JPL Horizons API" not in str(signature.get("source")) or not signature.get("version"):
        raise ZTFExternalError("JPL Horizons signature is missing or unsupported")
    if "$$SOE" not in result or "$$EOE" not in result:
        raise ZTFExternalError("JPL Horizons result lacks ephemeris delimiters")
    table = result.split("$$SOE", 1)[1].split("$$EOE", 1)[0]
    parsed: list[tuple[float, float, float]] = []
    returned_jds: list[float] = []
    for row in csv.reader(line for line in table.splitlines() if line.strip()):
        values = [field.strip() for field in row if field.strip()]
        if len(values) < 5:
            raise ZTFExternalError("JPL Horizons vector row has fewer than five fields")
        try:
            jd = float(values[0])
            vector = tuple(float(value) for value in values[2:5])
        except ValueError as exc:
            raise ZTFExternalError("JPL Horizons vector row is nonnumeric") from exc
        if not math.isfinite(jd) or not all(math.isfinite(value) for value in vector):
            raise ZTFExternalError("JPL Horizons vector row contains nonfinite values")
        returned_jds.append(jd)
        parsed.append(vector)
    if len(parsed) != len(expected_jds):
        raise ZTFExternalError(
            f"JPL Horizons returned {len(parsed)} epochs; expected {len(expected_jds)}"
        )
    for expected, returned in zip(expected_jds, returned_jds, strict=True):
        # Horizons' fractional-second JD serialization has shown about 1e-9 d
        # quantization at JD ~2.46e6.  Two nanodays (0.173 ms) accepts that
        # documented representation loss while still rejecting a shifted epoch.
        if abs(float(expected) - returned) > HORIZONS_JD_TOLERANCE_DAYS:
            raise ZTFExternalError(
                f"JPL Horizons epoch mismatch: requested {expected}, returned {returned}"
            )
    return tuple(parsed)


def _parse_horizons_target_identity(payload: bytes) -> dict[str, object]:
    """Extract the resolved numbered target from an official Horizons response."""
    try:
        document = json.loads(payload)
        result = str(document["result"])
    except (UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ZTFExternalError(f"invalid JPL Horizons JSON response: {exc}") from exc
    match = re.search(r"^Target body name:\s*(\d+)\s+(.+?)\s+\{source:", result, re.MULTILINE)
    if match is None:
        raise ZTFExternalError("JPL Horizons response lacks a numbered target identity")
    number = int(match.group(1))
    descriptor = match.group(2).strip()
    primary_name = re.sub(r"\s+\([^()]*\)\s*$", "", descriptor).strip()
    if number <= 0 or not primary_name:
        raise ZTFExternalError("JPL Horizons returned an invalid target identity")
    return {
        "asteroid_number": number,
        "primary_name": primary_name,
        "target_descriptor": descriptor,
    }


def _validate_resolved_target(
    normalized: Mapping[str, object], identities: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    object_id = normalized.get("object_id")
    number = _number_from_object_id(object_id)
    if not identities:
        raise ZTFExternalError(f"no Horizons target identities for {object_id}")
    first = dict(identities[0])
    if any(dict(identity) != first for identity in identities[1:]):
        raise ZTFExternalError(f"inconsistent Horizons target identities for {object_id}")
    if first.get("asteroid_number") != number:
        raise ZTFExternalError(
            f"Horizons target mismatch for {object_id}: {first.get('asteroid_number')}"
        )
    primary_name = str(first["primary_name"])
    rows = normalized.get("rows")
    if not isinstance(rows, list):
        raise ZTFExternalError(f"normalized Fink object has no rows: {object_id}")
    named = {
        str(row["fink_ssnamenr"]).strip()
        for row in rows
        if isinstance(row, Mapping)
        and isinstance(row.get("fink_ssnamenr"), str)
        and not str(row["fink_ssnamenr"]).strip().isdecimal()
    }
    if any(value.casefold() != primary_name.casefold() for value in named):
        raise ZTFExternalError(
            f"Fink named designation does not match Horizons for {object_id}: "
            f"{sorted(named)!r} versus {primary_name!r}"
        )
    first["fink_named_designations"] = sorted(named, key=str.casefold)
    first["validation_authority"] = "JPL Horizons resolved Target body name"
    return first


def fetch_horizons_cache(
    *,
    normalized_object_path: str | Path,
    output_directory: str | Path,
    batch_size: int = 20,
    timeout_seconds: float = 120.0,
    request_interval_seconds: float = 0.25,
    fetcher: HttpFetcher | None = None,
) -> dict[str, object]:
    """Fetch and freeze exact Sun/Earth vectors for one normalized ZTF object."""
    source_path = Path(normalized_object_path)
    destination = Path(output_directory)
    if destination.exists():
        raise ZTFExternalError("Horizons cache output directory must not already exist")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ZTFExternalError("batch_size must be a positive integer")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ZTFExternalError("timeout_seconds must be finite and positive")
    if not math.isfinite(request_interval_seconds) or request_interval_seconds < 0:
        raise ZTFExternalError("request_interval_seconds must be finite and nonnegative")
    try:
        normalized = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ZTFExternalError(f"cannot load normalized Fink object: {exc}") from exc
    if normalized.get("schema") != FINK_NORMALIZED_SCHEMA:
        raise ZTFExternalError("normalized Fink object schema mismatch")
    object_id = normalized.get("object_id")
    asteroid_number = _number_from_object_id(object_id)
    rows = normalized.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ZTFExternalError("normalized Fink object has no retained rows")
    jds = [float(row["jd"]) for row in rows]
    if jds != sorted(jds) or len(jds) != len(set(jds)):
        raise ZTFExternalError("normalized Fink JDs must be sorted and unique")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    active_fetcher = fetcher or _default_http_fetcher
    raw_manifest: list[dict[str, object]] = []
    merged_rows: list[dict[str, object]] = []
    resolved_identities: list[dict[str, object]] = []
    try:
        batches = [jds[index : index + batch_size] for index in range(0, len(jds), batch_size)]
        request_index = 0
        for batch_index, batch in enumerate(batches):
            vectors: dict[str, tuple[tuple[float, float, float], ...]] = {}
            for role, center in (("sun", "500@10"), ("earth", "500@399")):
                if request_index and request_interval_seconds:
                    time.sleep(request_interval_seconds)
                parameters = horizons_query_parameters(asteroid_number, center, batch)
                requested_url = f"{HORIZONS_API_URL}?{urlencode(parameters)}"
                body, final_url, http_date, content_type = active_fetcher(
                    requested_url, timeout_seconds
                )
                raw_path = staging / "raw" / f"batch-{batch_index:04d}-{role}.json"
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                raw_path.write_bytes(body)
                vectors[role] = _parse_horizons_vectors(body, batch)
                resolved_identities.append(_parse_horizons_target_identity(body))
                raw_manifest.append(
                    {
                        "batch": batch_index,
                        "role": f"asteroid_to_{role}",
                        "center": center,
                        "request_parameters_sha256": _hash(parameters),
                        "requested_url_sha256": hashlib.sha256(requested_url.encode()).hexdigest(),
                        "final_url": final_url,
                        "http_date": http_date,
                        "content_type": content_type,
                        "path": raw_path.relative_to(staging).as_posix(),
                        "bytes": len(body),
                        "sha256": hashlib.sha256(body).hexdigest(),
                    }
                )
                request_index += 1
            for offset, jd in enumerate(batch):
                # Horizons returns center-to-target. Negation makes both vectors
                # asteroid-centric, matching the DAMIT/Fink preparation contract.
                sun = [-value for value in vectors["sun"][offset]]
                earth = [-value for value in vectors["earth"][offset]]
                merged_rows.append(
                    {
                        "jd": jd,
                        "asteroid_to_sun_ecliptic_j2000_au": sun,
                        "asteroid_to_earth_ecliptic_j2000_au": earth,
                    }
                )
        resolved_target = _validate_resolved_target(normalized, resolved_identities)
        metadata: dict[str, object] = {
            "endpoint": HORIZONS_API_URL,
            "api": "JPL Horizons query-parameter API",
            "asteroid_number": asteroid_number,
            "object_id": object_id,
            "resolved_target_identity": resolved_target,
            "normalized_object_sha256": _sha256_file(source_path),
            "query_policy": {
                "target": f"{asteroid_number};",
                "centers": {"sun": "500@10", "earth": "500@399"},
                "returned_vector_orientation": "center_to_asteroid_then_negated",
                "output_orientation": "asteroid_to_target",
                "time_input": "UT Julian date from i:jd",
                "reference_plane": "ecliptic J2000",
                "reference_system": "ICRF",
                "vector_correction": "NONE",
                "units": "AU-D",
                "batch_size": batch_size,
                "request_order": "serial_sun_then_earth_per_batch",
            },
            "raw_responses": raw_manifest,
        }
        cache: dict[str, object] = {
            "source": "JPL Horizons",
            "query_metadata": metadata,
            "rows": merged_rows,
            "query_metadata_sha256": _hash(metadata),
            "rows_sha256": _hash(merged_rows),
        }
        _write_new_json(staging / "cache.json", cache)
        os.replace(staging, destination)
        return cache
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def fetch_horizons_directory(
    *,
    normalized_manifest_path: str | Path,
    output_directory: str | Path,
    batch_size: int = 20,
    timeout_seconds: float = 120.0,
    request_interval_seconds: float = 0.25,
    fetcher: HttpFetcher | None = None,
) -> dict[str, object]:
    """Resumably fetch one atomic Horizons cache per ready Fink object."""
    manifest_path = Path(normalized_manifest_path)
    output = Path(output_directory)
    try:
        source = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ZTFExternalError(f"cannot load normalized Fink manifest: {exc}") from exc
    if source.get("schema") != FINK_INGEST_MANIFEST_SCHEMA:
        raise ZTFExternalError("normalized Fink manifest schema mismatch")
    final_manifest_path = output / "manifest.json"
    if final_manifest_path.exists():
        raise ZTFExternalError("Horizons directory manifest already exists")
    output.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    for row in source["objects"]:
        if row["status"] != "ready":
            continue
        normalized_path = manifest_path.parent / str(row["normalized_path"])
        if _sha256_file(normalized_path) != row["normalized_sha256"]:
            raise ZTFExternalError(f"normalized object hash mismatch: {row['object_id']}")
        object_output = output / str(row["object_id"])
        cache_path = object_output / "cache.json"
        if object_output.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                parsed = HorizonsCache.from_mapping(cached)
            except (OSError, json.JSONDecodeError, ZTFExternalError) as exc:
                raise ZTFExternalError(
                    f"incomplete existing Horizons cache for {row['object_id']}: {exc}"
                ) from exc
            if parsed.query_metadata.get("normalized_object_sha256") != row["normalized_sha256"]:
                raise ZTFExternalError(f"stale Horizons cache for {row['object_id']}")
        else:
            cached = fetch_horizons_cache(
                normalized_object_path=normalized_path,
                output_directory=object_output,
                batch_size=batch_size,
                timeout_seconds=timeout_seconds,
                request_interval_seconds=request_interval_seconds,
                fetcher=fetcher,
            )
        results.append(
            {
                "object_id": row["object_id"],
                "normalized_sha256": row["normalized_sha256"],
                "cache_path": f"{row['object_id']}/cache.json",
                "cache_sha256": _sha256_file(cache_path),
                "query_metadata_sha256": cached["query_metadata_sha256"],
                "rows_sha256": cached["rows_sha256"],
                "row_count": len(cached["rows"]),
            }
        )
    result: dict[str, object] = {
        "schema": HORIZONS_MANIFEST_SCHEMA,
        "source": "JPL Horizons",
        "study_spec_sha256": source["study_spec_sha256"],
        "normalized_manifest_sha256": _sha256_file(manifest_path),
        "selection_uses_labels": False,
        "object_count": len(results),
        "row_count": sum(int(row["row_count"]) for row in results),
        "objects": results,
    }
    _write_new_json(final_manifest_path, result)
    return result


def _vector(value: object, name: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ZTFExternalError(f"{name} must be a three-vector")
    try:
        vector = tuple(float(component) for component in value)
    except (TypeError, ValueError) as exc:
        raise ZTFExternalError(f"{name} must be numeric") from exc
    if not all(math.isfinite(component) for component in vector) or np.linalg.norm(vector) <= 0:
        raise ZTFExternalError(f"{name} must be finite and nonzero")
    return vector


def _angle(left: Sequence[float], right: Sequence[float]) -> float:
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    return math.degrees(math.acos(float(np.clip(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))


def _equatorial(vector: Sequence[float]) -> np.ndarray:
    # IAU J2000 mean obliquity, sufficient only for an input-frame audit.
    epsilon = math.radians(23.4392911)
    x, y, z = vector
    return np.asarray((x, math.cos(epsilon) * y - math.sin(epsilon) * z, math.sin(epsilon) * y + math.cos(epsilon) * z))


@dataclass(frozen=True)
class HorizonsCache:
    query_metadata: Mapping[str, object]
    rows: tuple[Mapping[str, object], ...]
    query_metadata_sha256: str
    rows_sha256: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "HorizonsCache":
        required = {"source", "query_metadata", "rows", "query_metadata_sha256", "rows_sha256"}
        if set(value) != required or value.get("source") != "JPL Horizons":
            raise ZTFExternalError("Horizons cache requires JPL Horizons source and complete metadata")
        metadata, rows = value["query_metadata"], value["rows"]
        if not isinstance(metadata, Mapping) or not isinstance(rows, list) or not rows:
            raise ZTFExternalError("Horizons cache metadata and rows are required")
        if value["query_metadata_sha256"] != _hash(metadata) or value["rows_sha256"] != _hash(rows):
            raise ZTFExternalError("Horizons cache metadata or vector hash mismatch")
        return cls(metadata, tuple(rows), str(value["query_metadata_sha256"]), str(value["rows_sha256"]))


def plan_horizons_directory(
    *,
    normalized_manifest_path: str | Path,
    output_directory: str | Path,
    batch_size: int = 20,
) -> dict[str, object]:
    """Audit resume state and estimate requests without network access or writes."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ZTFExternalError("batch_size must be a positive integer")
    manifest_path = Path(normalized_manifest_path)
    output = Path(output_directory)
    try:
        source = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ZTFExternalError(f"cannot load normalized Fink manifest: {exc}") from exc
    if source.get("schema") != FINK_INGEST_MANIFEST_SCHEMA:
        raise ZTFExternalError("normalized Fink manifest schema mismatch")
    rows: list[dict[str, object]] = []
    for row in source.get("objects", []):
        if row.get("status") != "ready":
            continue
        object_id = str(row["object_id"])
        normalized_path = manifest_path.parent / str(row["normalized_path"])
        if _sha256_file(normalized_path) != row["normalized_sha256"]:
            raise ZTFExternalError(f"normalized object hash mismatch: {object_id}")
        retained_count = int(row["retained_row_count"])
        estimated_requests = 2 * math.ceil(retained_count / batch_size)
        object_output = output / object_id
        cache_path = object_output / "cache.json"
        status = "pending"
        if object_output.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                parsed = HorizonsCache.from_mapping(cached)
            except (OSError, json.JSONDecodeError, ZTFExternalError) as exc:
                raise ZTFExternalError(
                    f"incomplete existing Horizons cache for {object_id}: {exc}"
                ) from exc
            if parsed.query_metadata.get("normalized_object_sha256") != row["normalized_sha256"]:
                raise ZTFExternalError(f"stale Horizons cache for {object_id}")
            if len(parsed.rows) != retained_count:
                raise ZTFExternalError(f"Horizons row-count mismatch for {object_id}")
            status = "cached"
            estimated_requests = 0
        rows.append(
            {
                "object_id": object_id,
                "status": status,
                "retained_observation_count": retained_count,
                "estimated_request_count": estimated_requests,
            }
        )
    return {
        "schema": HORIZONS_PLAN_SCHEMA,
        "network_accessed": False,
        "filesystem_modified": False,
        "normalized_manifest_sha256": _sha256_file(manifest_path),
        "output_directory": output.as_posix(),
        "batch_size": batch_size,
        "ready_object_count": len(rows),
        "cached_object_count": sum(row["status"] == "cached" for row in rows),
        "pending_object_count": sum(row["status"] == "pending" for row in rows),
        "estimated_request_count": sum(int(row["estimated_request_count"]) for row in rows),
        "objects": rows,
    }


def _safe_child(root: Path, relative: object, role: str) -> Path:
    if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ZTFExternalError(f"unsafe {role} path")
    value = (root / relative).resolve()
    try:
        value.relative_to(root.resolve())
    except ValueError as exc:
        raise ZTFExternalError(f"{role} path escapes its root") from exc
    return value


def audit_horizons_directory_identities(
    *,
    normalized_manifest_path: str | Path,
    horizons_manifest_path: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    """Bind every Fink designation to the identity resolved by raw Horizons bytes."""
    normalized_path = Path(normalized_manifest_path)
    horizons_path = Path(horizons_manifest_path)
    destination = Path(output_path)
    if destination.exists():
        raise ZTFExternalError("Horizons identity-audit output must not already exist")
    try:
        normalized_manifest = json.loads(normalized_path.read_text(encoding="utf-8"))
        horizons_manifest = json.loads(horizons_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ZTFExternalError(f"cannot load Horizons audit input: {exc}") from exc
    if normalized_manifest.get("schema") != FINK_INGEST_MANIFEST_SCHEMA:
        raise ZTFExternalError("normalized Fink manifest schema mismatch")
    if horizons_manifest.get("schema") != HORIZONS_MANIFEST_SCHEMA:
        raise ZTFExternalError("Horizons directory manifest schema mismatch")
    if horizons_manifest.get("normalized_manifest_sha256") != _sha256_file(normalized_path):
        raise ZTFExternalError("Horizons manifest is not bound to the normalized Fink manifest")
    normalized_rows = {
        str(row["object_id"]): row
        for row in normalized_manifest.get("objects", [])
        if row.get("status") == "ready"
    }
    horizons_rows = {
        str(row["object_id"]): row for row in horizons_manifest.get("objects", [])
    }
    if (
        len(normalized_rows) != 169
        or len(horizons_rows) != 169
        or set(normalized_rows) != set(horizons_rows)
    ):
        raise ZTFExternalError("identity audit requires the same complete 169-object cohort")
    horizons_root = horizons_path.parent
    audited: list[dict[str, object]] = []
    for object_id in sorted(normalized_rows, key=_number_from_object_id):
        normalized_row = normalized_rows[object_id]
        horizons_row = horizons_rows[object_id]
        object_path = _safe_child(
            normalized_path.parent, normalized_row["normalized_path"], "normalized object"
        )
        if _sha256_file(object_path) != normalized_row["normalized_sha256"]:
            raise ZTFExternalError(f"normalized object hash mismatch: {object_id}")
        normalized = json.loads(object_path.read_text(encoding="utf-8"))
        cache_path = _safe_child(horizons_root, horizons_row["cache_path"], "Horizons cache")
        if _sha256_file(cache_path) != horizons_row["cache_sha256"]:
            raise ZTFExternalError(f"Horizons cache hash mismatch: {object_id}")
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        parsed = HorizonsCache.from_mapping(cache)
        if parsed.query_metadata.get("normalized_object_sha256") != normalized_row["normalized_sha256"]:
            raise ZTFExternalError(f"Horizons cache source mismatch: {object_id}")
        raw_records = parsed.query_metadata.get("raw_responses")
        if not isinstance(raw_records, list) or not raw_records:
            raise ZTFExternalError(f"Horizons cache lacks raw responses: {object_id}")
        identities: list[dict[str, object]] = []
        evidence: list[dict[str, object]] = []
        for raw_record in raw_records:
            if not isinstance(raw_record, Mapping):
                raise ZTFExternalError(f"invalid raw Horizons record: {object_id}")
            raw_path = _safe_child(cache_path.parent, raw_record.get("path"), "raw response")
            raw_bytes = raw_path.read_bytes()
            raw_hash = hashlib.sha256(raw_bytes).hexdigest()
            if raw_hash != raw_record.get("sha256") or len(raw_bytes) != raw_record.get("bytes"):
                raise ZTFExternalError(f"raw Horizons response mismatch: {object_id}")
            identities.append(_parse_horizons_target_identity(raw_bytes))
            evidence.append(
                {
                    "batch": int(raw_record["batch"]),
                    "role": str(raw_record["role"]),
                    "sha256": raw_hash,
                }
            )
        target = _validate_resolved_target(normalized, identities)
        embedded = parsed.query_metadata.get("resolved_target_identity")
        if embedded is not None and embedded != target:
            raise ZTFExternalError(f"embedded Horizons target identity mismatch: {object_id}")
        audited.append(
            {
                "object_id": object_id,
                "normalized_sha256": normalized_row["normalized_sha256"],
                "cache_path": horizons_row["cache_path"],
                "cache_sha256": horizons_row["cache_sha256"],
                "row_count": int(horizons_row["row_count"]),
                "resolved_target_identity": target,
                "raw_identity_evidence": evidence,
            }
        )
    result: dict[str, object] = {
        "schema": HORIZONS_IDENTITY_AUDIT_SCHEMA,
        "source": "retained official JPL Horizons API response bytes",
        "study_spec_sha256": normalized_manifest["study_spec_sha256"],
        "selection_uses_labels": False,
        "normalized_manifest_sha256": _sha256_file(normalized_path),
        "horizons_manifest_sha256": _sha256_file(horizons_path),
        "object_count": len(audited),
        "objects": audited,
    }
    _write_new_json(destination, result)
    return result


def rebind_horizons_directory(
    *,
    source_normalized_manifest_path: str | Path,
    final_normalized_manifest_path: str | Path,
    source_horizons_manifest_path: str | Path,
    output_directory: str | Path,
) -> dict[str, object]:
    """Rebind complete caches after a metadata-only study-spec refreeze.

    Raw Horizons bytes and vector rows are copied unchanged. Every old/new
    normalized object must be identical after removing only its recorded study
    spec hash; any scientific or selection change fails closed.
    """
    old_path = Path(source_normalized_manifest_path)
    new_path = Path(final_normalized_manifest_path)
    horizons_path = Path(source_horizons_manifest_path)
    destination = Path(output_directory)
    if destination.exists():
        raise ZTFExternalError("rebound Horizons output directory must not already exist")
    try:
        old_manifest = json.loads(old_path.read_text(encoding="utf-8"))
        new_manifest = json.loads(new_path.read_text(encoding="utf-8"))
        horizons_manifest = json.loads(horizons_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ZTFExternalError(f"cannot load Horizons rebind input: {exc}") from exc
    if (
        old_manifest.get("schema") != FINK_INGEST_MANIFEST_SCHEMA
        or new_manifest.get("schema") != FINK_INGEST_MANIFEST_SCHEMA
        or horizons_manifest.get("schema") != HORIZONS_MANIFEST_SCHEMA
        or horizons_manifest.get("normalized_manifest_sha256") != _sha256_file(old_path)
    ):
        raise ZTFExternalError("Horizons rebind input schemas or source binding are invalid")
    old_rows = {
        str(row["object_id"]): row
        for row in old_manifest.get("objects", [])
        if row.get("status") == "ready"
    }
    new_rows = {
        str(row["object_id"]): row
        for row in new_manifest.get("objects", [])
        if row.get("status") == "ready"
    }
    horizon_rows = {
        str(row["object_id"]): row for row in horizons_manifest.get("objects", [])
    }
    if (
        len(old_rows) != 169
        or set(new_rows) != set(old_rows)
        or set(horizon_rows) != set(old_rows)
        or horizons_manifest.get("object_count") != 169
    ):
        raise ZTFExternalError("Horizons rebind requires a complete identical 169-object cohort")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    rebound_rows: list[dict[str, object]] = []
    try:
        for object_id in sorted(old_rows, key=_number_from_object_id):
            old_row = old_rows[object_id]
            new_row = new_rows[object_id]
            old_object_path = _safe_child(
                old_path.parent, old_row["normalized_path"], "old normalized object"
            )
            new_object_path = _safe_child(
                new_path.parent, new_row["normalized_path"], "final normalized object"
            )
            if (
                _sha256_file(old_object_path) != old_row["normalized_sha256"]
                or _sha256_file(new_object_path) != new_row["normalized_sha256"]
            ):
                raise ZTFExternalError(f"normalized object hash mismatch: {object_id}")
            old_object = json.loads(old_object_path.read_text(encoding="utf-8"))
            new_object = json.loads(new_object_path.read_text(encoding="utf-8"))
            old_spec_hash = old_object.pop("study_spec_sha256", None)
            new_spec_hash = new_object.pop("study_spec_sha256", None)
            if (
                old_object != new_object
                or old_spec_hash == new_spec_hash
                or new_spec_hash != new_manifest.get("study_spec_sha256")
            ):
                raise ZTFExternalError(
                    f"normalized science changed; metadata-only rebind forbidden: {object_id}"
                )
            source_cache_path = _safe_child(
                horizons_path.parent, horizon_rows[object_id]["cache_path"], "source cache"
            )
            if _sha256_file(source_cache_path) != horizon_rows[object_id]["cache_sha256"]:
                raise ZTFExternalError(f"source Horizons cache hash mismatch: {object_id}")
            source_cache = json.loads(source_cache_path.read_text(encoding="utf-8"))
            parsed = HorizonsCache.from_mapping(source_cache)
            if parsed.query_metadata.get("normalized_object_sha256") != old_row["normalized_sha256"]:
                raise ZTFExternalError(f"source Horizons cache is stale: {object_id}")
            output_object_root = staging / object_id
            shutil.copytree(source_cache_path.parent, output_object_root)
            rebound_cache = dict(source_cache)
            metadata = dict(parsed.query_metadata)
            metadata["normalized_object_sha256"] = new_row["normalized_sha256"]
            metadata["normalization_rebind"] = {
                "source_normalized_manifest_sha256": _sha256_file(old_path),
                "final_normalized_manifest_sha256": _sha256_file(new_path),
                "source_normalized_object_sha256": old_row["normalized_sha256"],
                "final_normalized_object_sha256": new_row["normalized_sha256"],
                "final_study_spec_sha256": new_manifest["study_spec_sha256"],
                "only_study_spec_hash_changed": True,
                "raw_responses_and_vector_rows_unchanged": True,
            }
            rebound_cache["query_metadata"] = metadata
            rebound_cache["query_metadata_sha256"] = _hash(metadata)
            output_cache_path = output_object_root / "cache.json"
            output_cache_path.write_text(
                json.dumps(rebound_cache, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            HorizonsCache.from_mapping(rebound_cache)
            rebound_rows.append(
                {
                    "object_id": object_id,
                    "normalized_sha256": new_row["normalized_sha256"],
                    "cache_path": f"{object_id}/cache.json",
                    "cache_sha256": _sha256_file(output_cache_path),
                    "query_metadata_sha256": rebound_cache["query_metadata_sha256"],
                    "rows_sha256": rebound_cache["rows_sha256"],
                    "row_count": len(rebound_cache["rows"]),
                }
            )
        result: dict[str, object] = {
            "schema": HORIZONS_MANIFEST_SCHEMA,
            "source": "JPL Horizons",
            "study_spec_sha256": new_manifest["study_spec_sha256"],
            "normalized_manifest_sha256": _sha256_file(new_path),
            "selection_uses_labels": False,
            "object_count": len(rebound_rows),
            "row_count": sum(int(row["row_count"]) for row in rebound_rows),
            "offline_rebind": {
                "source_horizons_manifest_sha256": _sha256_file(horizons_path),
                "source_normalized_manifest_sha256": _sha256_file(old_path),
                "final_normalized_manifest_sha256": _sha256_file(new_path),
                "only_study_spec_hash_changed": True,
            },
            "objects": rebound_rows,
        }
        _write_new_json(staging / "manifest.json", result)
        os.replace(staging, destination)
        return result
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def fink_rows_to_epoch(
    object_id: str, rows: Sequence[Mapping[str, object]], horizons_cache: Mapping[str, object], *,
    norm_relative_tolerance: float = .02, phase_tolerance_deg: float = 2.0,
    radec_tolerance_deg: float = 2.0,
) -> ObservationEpoch:
    """Return one r-band epoch after validating cached vector geometry offline."""
    if not isinstance(object_id, str) or not object_id.strip():
        raise ZTFExternalError("object_id is required")
    if not 0 < norm_relative_tolerance < 1 or phase_tolerance_deg <= 0 or radec_tolerance_deg <= 0:
        raise ZTFExternalError("geometry tolerances must be positive")
    cache = HorizonsCache.from_mapping(horizons_cache)
    indexed: dict[float, Mapping[str, object]] = {}
    for row in cache.rows:
        try:
            jd = float(row["jd"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ZTFExternalError("Horizons vector row requires JD") from exc
        if not math.isfinite(jd) or jd in indexed:
            raise ZTFExternalError("Horizons vector JDs must be finite and unique")
        indexed[jd] = row
    retained: list[Mapping[str, object]] = []
    for row in rows:
        try:
            fid, mag, sigma = int(row["fid"]), float(row["magpsf"]), float(row["sigmapsf"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ZTFExternalError("Fink row requires fid, magpsf, and sigmapsf") from exc
        if fid == 2 and math.isfinite(mag) and math.isfinite(sigma) and 0 < sigma <= .2:
            retained.append(row)
    if not retained:
        raise ZTFExternalError("no finite r-band Fink detections satisfy 0 < sigmapsf <= 0.2")
    magnitudes = np.asarray([float(row["magpsf"]) for row in retained])
    reference_magnitude = float(np.median(magnitudes))
    observations: list[Observation] = []
    for row in retained:
        try:
            jd = float(row["jd"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ZTFExternalError("retained Fink row requires finite jd") from exc
        matching = [key for key in indexed if abs(key - jd) <= 1e-9]
        if len(matching) != 1:
            raise ZTFExternalError("every retained Fink JD requires exactly one cached Horizons vector")
        ephemeris = indexed[matching[0]]
        sun = _vector(ephemeris.get("asteroid_to_sun_ecliptic_j2000_au"), "asteroid_to_sun")
        observer = _vector(ephemeris.get("asteroid_to_earth_ecliptic_j2000_au"), "asteroid_to_earth")
        try:
            dhelio, dobs, phase = float(row["Dhelio"]), float(row["Dobs"]), float(row["phase"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ZTFExternalError("retained Fink row requires Dhelio, Dobs, and phase") from exc
        if not all(math.isfinite(value) and value > 0 for value in (dhelio, dobs)) or not math.isfinite(phase):
            raise ZTFExternalError("Fink geometry scalars must be finite")
        if abs(np.linalg.norm(sun) - dhelio) / dhelio > norm_relative_tolerance or abs(np.linalg.norm(observer) - dobs) / dobs > norm_relative_tolerance:
            raise ZTFExternalError("cached Horizons vector norm disagrees with Fink distance")
        if abs(_angle(sun, observer) - phase) > phase_tolerance_deg:
            raise ZTFExternalError("cached Horizons vector angle disagrees with Fink phase")
        if "ra" in row or "dec" in row:
            if "ra" not in row or "dec" not in row:
                raise ZTFExternalError("RA and DEC must be supplied together")
            ra, dec = float(row["ra"]), float(row["dec"])
            direction = _equatorial(-np.asarray(observer))
            direction /= np.linalg.norm(direction)
            expected = np.asarray((math.cos(math.radians(dec)) * math.cos(math.radians(ra)), math.cos(math.radians(dec)) * math.sin(math.radians(ra)), math.sin(math.radians(dec))))
            if _angle(direction, expected) > radec_tolerance_deg:
                raise ZTFExternalError("cached Horizons observer vector disagrees with Fink RA/DEC")
        flux = 10 ** (-.4 * (float(row["magpsf"]) - reference_magnitude))
        flux_error = flux * math.log(10) / 2.5 * float(row["sigmapsf"])
        observations.append(Observation(time_jd=jd, relative_brightness=flux, measured_error=flux_error, sun_asteroid_ecliptic_j2000_au=sun, observer_asteroid_ecliptic_j2000_au=observer))
    return ObservationEpoch(f"ztf-r-{object_id}", tuple(observations))


def prepared_ztf_object(object_id: str, rows: Sequence[Mapping[str, object]], horizons_cache: Mapping[str, object], *, known_period_hours: float, period_provenance: str) -> dict[str, object]:
    """Create a label-free frozen external object document; never queries a service."""
    epoch = fink_rows_to_epoch(object_id, rows, horizons_cache)
    return {"schema": "delphi.k3-ztf-prepared.v1", "object_id": object_id, "known_period_hours": known_period_hours, "period_provenance": period_provenance, "epochs": [{"epoch_id": epoch.epoch_id, "observations": [observation.__dict__ for observation in epoch.observations]}], "observer_geometry": {"target": "Earth center (Horizons 500@399)", "approximation": "geocenter substitutes for the individual ZTF observatory position"}, "horizons_query_metadata_sha256": HorizonsCache.from_mapping(horizons_cache).query_metadata_sha256, "horizons_rows_sha256": HorizonsCache.from_mapping(horizons_cache).rows_sha256}
