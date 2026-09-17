#!/usr/bin/env python3
"""Audit low-quality DAMIT identities used as synthetic-data donors."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

SCHEMA = "delphi.k3-lowq-synthetic-donor-exposure.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def audit(
    index_path: Path,
    synthetic_root: Path,
    lineage_path: Path,
    output: Path,
) -> dict[str, object]:
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    selected_lowq_ids = {
        f"asteroid_{int(row['damit_asteroid_id'])}" for row in index.get("objects", [])
    }
    if not selected_lowq_ids:
        raise ValueError("low-Q index contains no identities")
    status_by_id = {
        f"asteroid_{str(row['object_id']).split(':', 1)[1]}": row["lineage_status"]
        for row in lineage.get("objects", [])
    }
    analyzed_lowq_ids = set(status_by_id)
    if not analyzed_lowq_ids or not analyzed_lowq_ids <= selected_lowq_ids:
        raise ValueError("lineage identities are not a nonempty subset of the low-Q index")
    not_analyzed_ids = selected_lowq_ids - analyzed_lowq_ids

    roles: dict[str, dict[str, object]] = {}
    exposed_by_role: dict[str, set[str]] = {}
    for role in ("train", "validation", "test"):
        manifests = sorted((synthetic_root / role).glob("*.manifest.json"))
        if not manifests:
            raise ValueError(f"synthetic role has no manifests: {role}")
        donors: set[str] = set()
        hashes: dict[str, str] = {}
        for path in manifests:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("split") != role:
                raise ValueError(f"synthetic manifest role mismatch: {path}")
            donors.update(str(value) for value in payload.get("geometry_donor_ids", []))
            donors.update(
                str(value).split("/", 1)[0] for value in payload.get("shape_donor_ids", [])
            )
            hashes[path.name] = _sha256(path)
        exposed = donors & selected_lowq_ids
        exposed_by_role[role] = exposed
        status_counts = {
            status: sum(
                status_by_id.get(object_id, "not_analyzed") == status
                for object_id in exposed
            )
            for status in (
                "structured_reference_disjoint",
                "structured_reference_overlap",
                "missing_structured_reference",
                "not_analyzed",
            )
        }
        roles[role] = {
            "manifest_count": len(manifests),
            "manifest_sha256": hashes,
            "unique_donor_count": len(donors),
            "lowq_exposed_count": len(exposed),
            "lowq_exposed_ids": sorted(exposed),
            "lineage_status_counts": status_counts,
        }
    train_or_validation = exposed_by_role["train"] | exposed_by_role["validation"]
    result: dict[str, object] = {
        "schema": SCHEMA,
        "scope": "synthetic_geometry_and_shape_donor_identity_exposure",
        "input_index_sha256": _sha256(index_path),
        "lineage_analysis_sha256": _sha256(lineage_path),
        "selected_lowq_identity_count": len(selected_lowq_ids),
        "analyzed_lowq_identity_count": len(analyzed_lowq_ids),
        "not_analyzed_lowq_ids": sorted(not_analyzed_ids),
        "roles": roles,
        "train_or_validation_exposed_count": len(train_or_validation),
        "train_or_validation_exposed_ids": sorted(train_or_validation),
        "train_or_validation_lineage_status_counts": {
            status: sum(
                status_by_id.get(object_id, "not_analyzed") == status
                for object_id in train_or_validation
            )
            for status in (
                "structured_reference_disjoint",
                "structured_reference_overlap",
                "missing_structured_reference",
                "not_analyzed",
            )
        },
        "interpretation": (
            "Exposure means that an identity supplied observing geometry or a shape model to "
            "synthetic generation. Synthetic axes and periods were generated independently, so "
            "this is not exposure to the low-quality reference pole labels."
        ),
    }
    _write(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--synthetic-root", type=Path, required=True)
    parser.add_argument("--lineage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = audit(args.index, args.synthetic_root, args.lineage, args.output)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "passed": True,
                "train_or_validation_exposed_count": result[
                    "train_or_validation_exposed_count"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
