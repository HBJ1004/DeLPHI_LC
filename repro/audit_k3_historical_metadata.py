#!/usr/bin/env python3
"""Audit historical identity exposure using only JSON/YAML metadata files.

Binary arrays, checkpoints, generated axes, scores, and light curves are never
opened.  This identifies whether a historical directory has an enumerable
object namespace; it does not turn an incomplete archive into an exhaustive
project-history attestation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

_OBJECT_ID = re.compile(r"\basteroid_[1-9][0-9]*\b")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metadata_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise ValueError(f"metadata root is not a directory: {root}")
    files = sorted([*root.rglob("*.json"), *root.rglob("*.yaml"), *root.rglob("*.yml")])
    if not files:
        raise ValueError(f"no JSON/YAML metadata files under {root}")
    return files


def _extract_ids(path: Path) -> set[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"cannot read metadata {path}: {exc}") from exc
    return set(_OBJECT_ID.findall(text))


def audit(
    phase30_root: Path,
    phase40_metadata: Path,
    original_identity_map: Path,
    full_identity_aliases: Path,
    candidate_inventory: Path,
    phase40_manifest: Path | None = None,
) -> dict[str, Any]:
    try:
        map_doc = json.loads(original_identity_map.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read original identity map: {exc}") from exc
    if map_doc.get("schema") != "delphi.k3-survey-identity-map.v1":
        raise ValueError("original identity map schema mismatch")
    original_ids = {str(row.get("object_id")) for row in map_doc.get("objects", []) if isinstance(row, dict)}
    if len(original_ids) != 170:
        raise ValueError("original identity map does not contain exactly 170 objects")

    try:
        aliases_doc = json.loads(full_identity_aliases.read_text(encoding="utf-8"))
        candidates_doc = json.loads(candidate_inventory.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read full aliases or candidate inventory: {exc}") from exc
    if aliases_doc.get("schema") != "delphi.k3-survey-identity-map.v1":
        raise ValueError("full identity aliases schema mismatch")
    if candidates_doc.get("schema") != "delphi.k3-alcdef-gaia-eligibility.v1":
        raise ValueError("candidate inventory schema mismatch")
    physical_by_object = {}
    for row in aliases_doc.get("objects", []):
        if not isinstance(row, dict) or not isinstance(row.get("object_id"), str):
            continue
        physical = row.get("physical_identity")
        number = row.get("mpc_number")
        if isinstance(physical, str):
            physical_by_object[str(row["object_id"])] = physical
        elif isinstance(number, int) and not isinstance(number, bool) and number > 0:
            physical_by_object[str(row["object_id"])] = f"mpc:{number}"
    candidate_ids = {
        str(row.get("object_id"))
        for row in candidates_doc.get("candidates", [])
        if isinstance(row, dict) and isinstance(row.get("object_id"), str)
    }

    def inventory(root: Path) -> dict[str, Any]:
        files = _metadata_files(root)
        ids: set[str] = set()
        rows = []
        for path in files:
            file_ids = _extract_ids(path)
            ids.update(file_ids)
            rows.append({"relative_path": path.relative_to(root).as_posix(), "sha256": _sha256(path), "object_id_count": len(file_ids)})
        physical_ids = {physical_by_object[item] for item in ids if item in physical_by_object}
        return {
            "root": str(root.resolve()),
            "metadata_file_count": len(rows),
            "metadata_files": rows,
            "object_ids": sorted(ids),
            "object_id_count": len(ids),
            "unknown_ids_relative_to_original": sorted(ids - original_ids),
            "all_enumerated_ids_are_original": ids.issubset(original_ids),
            "mapped_physical_id_count": len(physical_ids),
            "unmapped_object_ids": sorted(ids - set(physical_by_object)),
            "candidate_inventory_overlap": sorted(physical_ids & candidate_ids),
        }

    phase30 = inventory(phase30_root)
    phase40 = inventory(phase40_metadata)
    phase40_enumerable = bool(phase40["object_ids"])
    phase40_manifest_summary: dict[str, Any] | None = None
    if phase40_manifest is not None:
        try:
            rows = [json.loads(line) for line in phase40_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read phase40 manifest: {exc}") from exc
        if not rows or any(not isinstance(row, dict) or not isinstance(row.get("object_id"), str) for row in rows):
            raise ValueError("phase40 manifest lacks object_id records")
        identifiers = {str(row["object_id"]) for row in rows}
        donor_keys = {key for row in rows for key in row if key in {"source_object_id", "donor_object_id", "damit_id", "mpc_number"}}
        phase40_manifest_summary = {
            "path": str(phase40_manifest.resolve()),
            "sha256": _sha256(phase40_manifest),
            "sample_count": len(rows),
            "unique_sample_id_count": len(identifiers),
            "all_ids_are_synthetic_namespace": all(item.startswith(("synth_", "synthetic_")) for item in identifiers),
            "donor_identity_fields_present": sorted(donor_keys),
            "donor_identity_enumerable": bool(donor_keys),
        }
    return {
        "schema": "delphi.k3-historical-metadata-exposure-audit.v1",
        "purpose": "metadata-only historical identity screen; not exhaustive project history",
        "original_identity_map": {"path": str(original_identity_map.resolve()), "sha256": _sha256(original_identity_map), "object_count": len(original_ids)},
        "full_identity_aliases": {"path": str(full_identity_aliases.resolve()), "sha256": _sha256(full_identity_aliases), "object_count": len(physical_by_object)},
        "candidate_inventory": {"path": str(candidate_inventory.resolve()), "sha256": _sha256(candidate_inventory), "object_count": len(candidate_ids)},
        "phase30": phase30,
        "phase40": {**phase40, "identity_enumeration_available": phase40_enumerable, "manifest": phase40_manifest_summary},
        "conclusion": {
            "phase30_candidate_identities_to_mark_exposed": phase30["candidate_inventory_overlap"],
            "phase40_clears_external_candidates": False,
            "remaining_history_status": "unknown: phase40 synthetic samples do not enumerate real donor identities; other project history is not certified exhaustive",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase30-root", type=Path, required=True)
    parser.add_argument("--phase40-metadata", type=Path, required=True)
    parser.add_argument("--original-identity-map", type=Path, required=True)
    parser.add_argument("--full-identity-aliases", type=Path, required=True)
    parser.add_argument("--candidate-inventory", type=Path, required=True)
    parser.add_argument("--phase40-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    try:
        payload = audit(
            args.phase30_root,
            args.phase40_metadata,
            args.original_identity_map,
            args.full_identity_aliases,
            args.candidate_inventory,
            args.phase40_manifest,
        )
    except ValueError as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
