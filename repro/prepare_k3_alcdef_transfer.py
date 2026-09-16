#!/usr/bin/env python3
"""Fetch session geometry and prepare locked ALCDEF transfer inputs.

This module deliberately has two stages. ``fetch-geometry`` consumes the raw
ALCDEF extraction but no Gaia table. ``prepare`` consumes the frozen SBDB
period receipts and cached geometry, still without Gaia axes.  ALCDEF session
midpoint geometry is applied to every point in that session; raw observations
and midpoint selection are retained for inspection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlencode

import numpy as np

from lc_pipeline.k3.ztf_external import (
    HORIZONS_API_URL,
    _default_http_fetcher,
    _parse_horizons_target_identity,
    _parse_horizons_vectors,
    horizons_query_parameters,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path, schema: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema") != schema:
        raise ValueError(f"unexpected schema in {path}")
    return value


def _write_new(path: Path, value: object) -> None:
    if path.exists():
        raise ValueError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _numbers(raw: Mapping[str, Any]) -> list[tuple[str, int, list[dict[str, Any]], list[dict[str, Any]]]]:
    rows = raw.get("objects")
    if not isinstance(rows, list) or not rows:
        raise ValueError("raw extraction has no objects")
    result = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("raw extraction has invalid object")
        identity, number, sessions, points = row.get("object_id"), row.get("source_mpc_number"), row.get("sessions"), row.get("points")
        if not isinstance(identity, str) or not isinstance(number, int) or number <= 0 or not isinstance(sessions, list) or not isinstance(points, list):
            raise ValueError("raw extraction object is malformed")
        result.append((identity, number, sessions, points))
    return result


def _midpoints(sessions: list[dict[str, Any]], points: list[dict[str, Any]]) -> list[tuple[str, float]]:
    by_session: dict[str, list[float]] = {str(row["ID"]): [] for row in sessions if isinstance(row.get("ID"), str)}
    for point in points:
        try:
            identifier, jd = str(point["MDID"]), float(point["JD"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("raw point lacks session or Julian date") from exc
        if identifier in by_session and math.isfinite(jd) and jd > 0:
            by_session[identifier].append(jd)
    result = [(identifier, float(statistics.median(values))) for identifier, values in by_session.items() if values]
    if not result:
        raise ValueError("object has no session points")
    return sorted(result, key=lambda item: item[1])


def fetch_geometry(raw_path: Path, output: Path, *, batch_size: int, timeout_seconds: float, interval_seconds: float) -> dict[str, Any]:
    raw = _read(raw_path, "delphi.k3-locked-alcdef-raw-extract.v1")
    # The public endpoint returned HTTP 502 for a 64-epoch TLIST pilot.  The
    # prior corrected Fink/Horizons acquisition used twenty successfully, so
    # retain that conservative transport limit for this separate cohort.
    if batch_size <= 0 or batch_size > 20 or timeout_seconds <= 0 or interval_seconds < 0:
        raise ValueError("invalid geometry-fetch settings")
    records = []
    for identity, number, sessions, points in _numbers(raw):
        midpoint_rows = _midpoints(sessions, points)
        jds = [value for _, value in midpoint_rows]
        vectors: dict[str, list[float]] = {}
        raw_responses = []
        identities = []
        request_count = 0
        for start in range(0, len(jds), batch_size):
            batch = jds[start : start + batch_size]
            for role, center in (("sun", "500@10"), ("earth", "500@399")):
                if request_count and interval_seconds:
                    time.sleep(interval_seconds)
                parameters = horizons_query_parameters(number, center, batch)
                url = f"{HORIZONS_API_URL}?{urlencode(parameters)}"
                body, final_url, http_date, content_type = _default_http_fetcher(url, timeout_seconds)
                parsed = _parse_horizons_vectors(body, batch)
                identities.append(_parse_horizons_target_identity(body))
                for offset, vector in enumerate(parsed):
                    # Horizons gives center-to-asteroid; K3 uses asteroid-to-target.
                    vectors[f"{role}:{start + offset}"] = [-float(value) for value in vector]
                raw_responses.append({"batch": start // batch_size, "role": role, "center": center, "request_url": url, "final_url": final_url, "http_date": http_date, "content_type": content_type, "response_sha256": hashlib.sha256(body).hexdigest(), "response_bytes": len(body), "raw_response_utf8": body.decode("utf-8")})
                request_count += 1
        first = identities[0]
        if any(item != first for item in identities) or first.get("asteroid_number") != number:
            raise ValueError(f"Horizons identity mismatch for {identity}")
        records.append({"object_id": identity, "mpc_number": number, "resolved_target_identity": first, "session_midpoints": [{"metadata_id": identifier, "jd": jd, "asteroid_to_sun_ecliptic_j2000_au": vectors[f"sun:{index}"], "asteroid_to_earth_ecliptic_j2000_au": vectors[f"earth:{index}"]} for index, (identifier, jd) in enumerate(midpoint_rows)], "raw_responses": raw_responses})
    payload = {"schema": "delphi.k3-alcdef-session-geometry.v1", "purpose": "session-midpoint JPL Horizons geometry before K3 prediction and before Gaia-axis reference access", "raw_extraction_path": str(raw_path), "raw_extraction_sha256": _sha256(raw_path), "endpoint": HORIZONS_API_URL, "objects": records}
    _write_new(output, payload)
    return payload


def prepare(raw_path: Path, periods_path: Path, geometry_path: Path, output: Path) -> dict[str, Any]:
    raw = _read(raw_path, "delphi.k3-locked-alcdef-raw-extract.v1")
    periods = _read(periods_path, "delphi.k3-alcdef-sbdb-periods.v1")
    geometry = _read(geometry_path, "delphi.k3-alcdef-session-geometry.v1")
    if periods.get("cohort_lock_sha256") != raw.get("cohort_lock", {}).get("sha256"):
        raise ValueError("period receipts are not bound to raw cohort lock")
    if geometry.get("raw_extraction_sha256") != _sha256(raw_path):
        raise ValueError("geometry cache is not bound to raw extraction")
    by_period = {row.get("object_id"): row for row in periods.get("objects", []) if isinstance(row, dict)}
    by_geometry = {row.get("object_id"): row for row in geometry.get("objects", []) if isinstance(row, dict)}
    objects = []
    for identity, number, sessions, points in _numbers(raw):
        period = by_period.get(identity)
        cache = by_geometry.get(identity)
        if not isinstance(period, dict) or period.get("status") != "ready" or not isinstance(cache, dict):
            raise ValueError(f"input unavailable for {identity}")
        period_hours = float(period["known_period_hours"])
        if not period_hours > 0:
            raise ValueError(f"invalid period for {identity}")
        vectors = {str(row["metadata_id"]): row for row in cache.get("session_midpoints", []) if isinstance(row, dict)}
        point_groups: dict[str, list[dict[str, Any]]] = {}
        for point in points:
            point_groups.setdefault(str(point.get("MDID")), []).append(point)
        prepared_epochs = []
        for session in sessions:
            session_id = str(session.get("ID"))
            geometry_row, raw_points = vectors.get(session_id), point_groups.get(session_id, [])
            parsed = []
            for row in raw_points:
                try:
                    jd, magnitude = float(row["JD"]), float(row["Mag"])
                except (KeyError, TypeError, ValueError):
                    continue
                if math.isfinite(jd) and math.isfinite(magnitude):
                    parsed.append((jd, magnitude, row.get("MagErr")))
            if geometry_row is None or len(parsed) < 2:
                continue
            reference = float(np.median([item[1] for item in parsed]))
            observations = []
            for jd, magnitude, raw_error in parsed:
                brightness = 10 ** (-0.4 * (magnitude - reference))
                try:
                    sigma = float(raw_error)
                    error = brightness * math.log(10) / 2.5 * sigma if sigma > 0 else None
                except (TypeError, ValueError):
                    error = None
                observations.append({"time_jd": jd, "relative_brightness": brightness, "measured_error": error, "sun_asteroid_ecliptic_j2000_au": geometry_row["asteroid_to_sun_ecliptic_j2000_au"], "observer_asteroid_ecliptic_j2000_au": geometry_row["asteroid_to_earth_ecliptic_j2000_au"]})
            prepared_epochs.append({"epoch_id": f"alcdef-{identity}-{session_id}", "observations": observations})
        if not prepared_epochs:
            raise ValueError(f"no usable epochs for {identity}")
        objects.append({"schema": "delphi.k3-external-prepared.v1", "object_id": identity, "known_period_hours": period_hours, "period_provenance": "JPL SBDB physical-parameter rot_per receipt; source citation retained", "epochs": prepared_epochs, "input_summary": {"mpc_number": number, "native_session_count": len(sessions), "prepared_epoch_count": len(prepared_epochs), "raw_point_count": len(points), "geometry_policy": "one JPL Horizons vector pair at each native-session median JD, assigned to that session's points", "photometry_policy": "session-median magnitude normalized to relative brightness; no distance, phase, or light-time correction"}})
    payload = {"schema": "delphi.k3-alcdef-prepared-manifest.v1", "purpose": "label-free prepared ALCDEF transfer inputs before Gaia-axis reference access", "raw_extraction_sha256": _sha256(raw_path), "period_receipts_sha256": _sha256(periods_path), "geometry_sha256": _sha256(geometry_path), "objects": objects}
    _write_new(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    geometry = commands.add_parser("fetch-geometry")
    geometry.add_argument("--raw", type=Path, required=True)
    geometry.add_argument("--output", type=Path, required=True)
    geometry.add_argument("--batch-size", type=int, default=20)
    geometry.add_argument("--timeout-seconds", type=float, default=120.0)
    geometry.add_argument("--request-interval-seconds", type=float, default=0.2)
    ready = commands.add_parser("prepare")
    ready.add_argument("--raw", type=Path, required=True)
    ready.add_argument("--periods", type=Path, required=True)
    ready.add_argument("--geometry", type=Path, required=True)
    ready.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "fetch-geometry":
            result = fetch_geometry(args.raw, args.output, batch_size=args.batch_size, timeout_seconds=args.timeout_seconds, interval_seconds=args.request_interval_seconds)
        else:
            result = prepare(args.raw, args.periods, args.geometry, args.output)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps({"object_count": len(result["objects"])}, sort_keys=True))


if __name__ == "__main__":
    main()
