#!/usr/bin/env python3
"""Extract raw ALCDEF records for an already locked cohort.

The command reads only the selected MPC identities from a checksum-bound PDS
ZIP.  It preserves source magnitude and correction fields without applying
distance, light-time, phase, or geometry transformations.  Gaia reference
axes are neither accepted nor read.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

_RANGE = re.compile(r"-(\d+)-(\d+)\.csv$")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _range_contains(name: str, values: set[int]) -> bool:
    match = _RANGE.search(name)
    return bool(match and any(int(match.group(1)) <= value <= int(match.group(2)) for value in values))


def _rows(payload: bytes, required: set[str]) -> list[dict[str, str]]:
    decoded = payload.decode("utf-8-sig")
    parsed = list(csv.reader(io.StringIO(decoded)))
    index = next((offset for offset, row in enumerate(parsed) if required.issubset({cell.strip() for cell in row})), None)
    if index is None:
        raise ValueError(f"missing header {sorted(required)}")
    header = [cell.strip() for cell in parsed[index]]
    return [{key: row[position].strip() for position, key in enumerate(header)} for row in parsed[index + 1 :] if len(row) >= len(header) and any(cell.strip() for cell in row)]


def _positive_int(value: str | None) -> int | None:
    try:
        result = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def extract(lock_path: Path, source_zip: Path) -> dict[str, Any]:
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read cohort lock: {exc}") from exc
    if lock.get("schema") != "delphi.k3-alcdef-gaia-cohort-lock.v1":
        raise ValueError("cohort lock schema mismatch")
    selected = lock.get("objects")
    if not isinstance(selected, list) or not selected:
        raise ValueError("cohort lock has no objects")
    identifiers = [row.get("object_id") for row in selected if isinstance(row, dict)]
    if len(identifiers) != len(selected) or any(not isinstance(value, str) or not value.startswith("mpc:") for value in identifiers):
        raise ValueError("cohort lock contains invalid identities")
    numbers = {int(str(value).split(":", 1)[1]) for value in identifiers}
    metadata: dict[int, dict[str, dict[str, str]]] = defaultdict(dict)
    points: dict[int, list[dict[str, str]]] = defaultdict(list)
    inputs: list[dict[str, Any]] = []
    with zipfile.ZipFile(source_zip) as archive:
        names = archive.namelist()
        metadata_names = [name for name in names if "/alcdef_metadata-" in name and _range_contains(name, numbers)]
        lc_names = [name for name in names if "/alcdef_lcdata-" in name and _range_contains(name, numbers)]
        if not metadata_names or not lc_names:
            raise ValueError("locked identities have no matching ALCDEF source tables")
        for name in sorted(metadata_names):
            payload = archive.read(name)
            inputs.append({"path": name, "sha256": _sha256_bytes(payload), "bytes": len(payload)})
            for row in _rows(payload, {"ID", "ObjNumber", "SessionDateTime"}):
                number, mdid = _positive_int(row.get("ObjNumber")), _positive_int(row.get("ID"))
                if number in numbers and mdid is not None:
                    metadata[number][str(mdid)] = row
        for name in sorted(lc_names):
            payload = archive.read(name)
            inputs.append({"path": name, "sha256": _sha256_bytes(payload), "bytes": len(payload)})
            for row in _rows(payload, {"ObjectNumber", "MDID", "JD", "Mag", "MagErr"}):
                number, mdid = _positive_int(row.get("ObjectNumber")), _positive_int(row.get("MDID"))
                if number in numbers and mdid is not None and str(mdid) in metadata[number]:
                    points[number].append(row)
    objects = []
    for object_id in identifiers:
        number = int(str(object_id).split(":", 1)[1])
        sessions = metadata[number]
        objects.append({
            "object_id": object_id,
            "source_mpc_number": number,
            "session_count": len(sessions),
            "raw_point_count": len(points[number]),
            "sessions": [sessions[key] for key in sorted(sessions, key=int)],
            "points": points[number],
        })
    return {
        "schema": "delphi.k3-locked-alcdef-raw-extract.v1",
        "purpose": "raw locked ALCDEF extraction before geometry, transformations, prediction, or Gaia reference access",
        "cohort_lock": {"path": str(lock_path), "sha256": _sha256_file(lock_path)},
        "source_zip": {"path": str(source_zip), "sha256": _sha256_file(source_zip)},
        "input_tables": inputs,
        "objects": objects,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-lock", type=Path, required=True)
    parser.add_argument("--source-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    try:
        payload = extract(args.cohort_lock, args.source_zip)
    except (OSError, ValueError, zipfile.BadZipFile, UnicodeError, csv.Error) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
