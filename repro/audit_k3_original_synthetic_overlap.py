"""Audit original-cohort identities against declared synthetic donor IDs only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Mapping

from lc_pipeline.k3.generalization_validation import (
    GeneralizationValidationError,
    canonical_json,
    sha256_file,
)

REPORT_SCHEMA = "delphi.k3-original-synthetic-donor-overlap.v1"
OUTPUT_NAME = "original-cohort-synthetic-donor-audit.json"
EXPECTED_MANIFEST_COUNT = 96
EXPECTED_ORIGINAL_COUNT = 170
EXPECTED_DONOR_COUNTS = {"train": 411, "validation": 51, "test": 52}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DAMIT_OBJECT = re.compile(r"^asteroid_([1-9][0-9]*)$")
_SHAPE_DONOR = re.compile(r"^(asteroid_[1-9][0-9]*)/model_[1-9][0-9]*$")


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GeneralizationValidationError(f"cannot read JSON {path}: {exc}") from exc


def _artifact(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise GeneralizationValidationError(f"required source file is missing: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _identity_maps(document: object) -> tuple[dict[str, str], dict[str, str], str]:
    if not isinstance(document, Mapping):
        raise GeneralizationValidationError("identity table must be a JSON object")
    table_hash = document.get("identity_table_sha256")
    if not isinstance(table_hash, str) or not _SHA256.fullmatch(table_hash.lower()):
        raise GeneralizationValidationError("identity table lacks an official-table SHA-256")
    rows = document.get("objects")
    if not isinstance(rows, list):
        raise GeneralizationValidationError("identity table lacks objects")
    to_damit: dict[str, str] = {}
    to_physical: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise GeneralizationValidationError("identity table contains a non-object row")
        object_id = row.get("object_id")
        damit_id = row.get("damit_id")
        mpc_number = row.get("mpc_number")
        if (
            not isinstance(object_id, str)
            or not isinstance(damit_id, int)
            or isinstance(damit_id, bool)
            or damit_id <= 0
            or object_id != f"asteroid_{damit_id}"
        ):
            raise GeneralizationValidationError(
                "identity row does not explicitly bind asteroid_N to DAMIT internal N"
            )
        if object_id in to_damit:
            raise GeneralizationValidationError(f"duplicate identity-table object: {object_id}")
        damit_identity = f"damit:{damit_id}"
        to_damit[object_id] = damit_identity
        if isinstance(mpc_number, int) and not isinstance(mpc_number, bool) and mpc_number > 0:
            to_physical[object_id] = f"mpc:{mpc_number}"
        else:
            to_physical[object_id] = damit_identity
    return to_damit, to_physical, table_hash.lower()


def _original_ids(
    document: object, *, expected_count: int
) -> tuple[set[str], dict[str, list[str]]]:
    if not isinstance(document, Mapping) or not isinstance(document.get("folds"), list):
        raise GeneralizationValidationError("publication splits lack folds")
    by_partition: dict[str, set[str]] = {
        role: set() for role in ("train", "validation", "calibration", "test")
    }
    for fold in document["folds"]:
        if not isinstance(fold, Mapping):
            raise GeneralizationValidationError("publication split contains an invalid fold")
        for role in by_partition:
            values = fold.get(f"{role}_ids", [])
            if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
                raise GeneralizationValidationError(f"publication {role}_ids are invalid")
            by_partition[role].update(values)
    combined = set().union(*by_partition.values())
    if len(combined) != expected_count:
        raise GeneralizationValidationError(
            f"expected {expected_count} original identities, found {len(combined)}"
        )
    return combined, {role: sorted(values) for role, values in by_partition.items()}


def _donor_parent(value: object, *, kind: str) -> str:
    if not isinstance(value, str):
        raise GeneralizationValidationError(f"{kind} donor ID must be a string")
    if kind == "shape":
        match = _SHAPE_DONOR.fullmatch(value)
        if match is None:
            raise GeneralizationValidationError(
                f"shape donor does not use asteroid_N/model_M syntax: {value}"
            )
        return match.group(1)
    if _DAMIT_OBJECT.fullmatch(value) is None:
        raise GeneralizationValidationError(
            f"geometry donor does not use asteroid_N syntax: {value}"
        )
    return value


def build_overlap_report(
    *,
    original_splits: Path,
    identity_aliases: Path,
    published_metadata_root: Path,
    expected_manifest_count: int = EXPECTED_MANIFEST_COUNT,
    expected_original_count: int = EXPECTED_ORIGINAL_COUNT,
    expected_donor_counts: Mapping[str, int] = EXPECTED_DONOR_COUNTS,
) -> dict[str, object]:
    aliases_document = _read_json(identity_aliases)
    to_damit, to_physical, official_hash = _identity_maps(aliases_document)
    original_document = _read_json(original_splits)
    original_legacy, original_partitions = _original_ids(
        original_document, expected_count=expected_original_count
    )
    missing_original = sorted(original_legacy - set(to_damit))
    if missing_original:
        raise GeneralizationValidationError(
            f"original identities lack official mappings: {missing_original}"
        )
    original_damit = {to_damit[value] for value in original_legacy}
    original_physical = {to_physical[value] for value in original_legacy}
    if len(original_damit) != expected_original_count or len(original_physical) != expected_original_count:
        raise GeneralizationValidationError(
            "original cohort does not map one-to-one to DAMIT and physical identities"
        )

    metadata_root = published_metadata_root.resolve()
    if not metadata_root.is_dir():
        raise GeneralizationValidationError(f"published metadata root is missing: {metadata_root}")
    manifest_paths = sorted(metadata_root.rglob("*.manifest.json"))
    if len(manifest_paths) != expected_manifest_count:
        raise GeneralizationValidationError(
            f"expected {expected_manifest_count} manifests, found {len(manifest_paths)}"
        )
    manifest_rows: list[dict[str, object]] = []
    donor_ids: dict[str, dict[str, set[str]]] = {
        split: {"shape": set(), "geometry": set()}
        for split in ("train", "validation", "test")
    }
    for path in manifest_paths:
        document = _read_json(path)
        if not isinstance(document, Mapping) or document.get("schema") != "delphi.k3-synthetic-shard.v1":
            raise GeneralizationValidationError(f"invalid synthetic manifest: {path}")
        split = document.get("split")
        if split not in donor_ids:
            raise GeneralizationValidationError(f"invalid synthetic split in {path}: {split}")
        for kind, key in (("shape", "shape_donor_ids"), ("geometry", "geometry_donor_ids")):
            values = document.get(key)
            if not isinstance(values, list):
                raise GeneralizationValidationError(f"manifest lacks {key}: {path}")
            donor_ids[str(split)][kind].update(
                _donor_parent(value, kind=kind) for value in values
            )
        manifest_rows.append(
            {
                "relative_path": path.relative_to(metadata_root).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "split": split,
            }
        )

    split_reports: dict[str, object] = {}
    every_overlap: set[str] = set()
    count_checks: dict[str, bool] = {}
    for split in ("train", "validation", "test"):
        role_reports: dict[str, object] = {}
        for kind in ("shape", "geometry"):
            legacy = donor_ids[split][kind]
            missing = sorted(legacy - set(to_damit))
            if missing:
                raise GeneralizationValidationError(
                    f"{split} {kind} donors lack official mappings: {missing}"
                )
            damit = {to_damit[value] for value in legacy}
            physical = {to_physical[value] for value in legacy}
            overlap_damit = sorted(damit & original_damit)
            overlap_physical = sorted(physical & original_physical)
            every_overlap.update(overlap_physical)
            expected = expected_donor_counts.get(split)
            count_matches = expected is None or len(damit) == expected
            count_checks[f"{split}_{kind}"] = count_matches
            role_reports[kind] = {
                "unique_declared_parent_ids": len(legacy),
                "unique_damit_identities": len(damit),
                "unique_physical_identities": len(physical),
                "expected_unique_donors": expected,
                "expected_count_matches": count_matches,
                "overlap_count_damit": len(overlap_damit),
                "overlap_count_physical": len(overlap_physical),
                "overlap_damit_identities": overlap_damit,
                "overlap_physical_identities": overlap_physical,
            }
        split_reports[split] = {
            "manifest_count": sum(row["split"] == split for row in manifest_rows),
            "shape_and_geometry_parent_sets_equal": (
                donor_ids[split]["shape"] == donor_ids[split]["geometry"]
            ),
            "roles": role_reports,
        }

    return {
        "schema": REPORT_SCHEMA,
        "purpose": "original_170_vs_declared_synthetic_donor_identity_integrity_check",
        "scope_limit": (
            "verifies_only_shape_and_geometry_donor_ids_declared_in_JSON_manifests; "
            "does_not_verify_retraining_or_NPZ_contents"
        ),
        "retraining_verified": False,
        "npz_contents_verified": False,
        "namespace_resolution": {
            "legacy_asteroid_N": "damit_internal_id_N",
            "comparison_identity": "official_table_mapped_physical_identity",
            "numeric_suffix_collapse_across_namespaces": False,
            "official_table_sha256": official_hash,
        },
        "sources": {
            "publication_splits": _artifact(original_splits),
            "identity_aliases": _artifact(identity_aliases),
            "published_synthetic_manifests": {
                "root": str(metadata_root),
                "manifest_count": len(manifest_rows),
                "manifest_set_sha256": _digest(manifest_rows),
                "manifests": manifest_rows,
            },
        },
        "original_cohort": {
            "unique_legacy_ids": len(original_legacy),
            "unique_damit_identities": len(original_damit),
            "unique_physical_identities": len(original_physical),
            "partition_unique_counts_across_folds": {
                role: len(values) for role, values in original_partitions.items()
            },
        },
        "synthetic_splits": split_reports,
        "checks": {
            "all_role_overlaps_zero": not every_overlap,
            "all_expected_donor_counts_match": all(count_checks.values()),
            "all_shape_geometry_parent_sets_equal": all(
                donor_ids[split]["shape"] == donor_ids[split]["geometry"]
                for split in donor_ids
            ),
        },
        "all_overlap_physical_identities": sorted(every_overlap),
    }


def write_report(path: Path, report: object) -> None:
    target = path.resolve()
    if target.exists():
        raise GeneralizationValidationError(f"refusing to overwrite {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--original-splits", type=Path, required=True)
    root.add_argument("--identity-aliases", type=Path, required=True)
    root.add_argument("--published-metadata-root", type=Path, required=True)
    root.add_argument("--output", type=Path, required=True)
    return root


def main() -> None:
    args = parser().parse_args()
    report = build_overlap_report(
        original_splits=args.original_splits,
        identity_aliases=args.identity_aliases,
        published_metadata_root=args.published_metadata_root,
    )
    write_report(args.output, report)


if __name__ == "__main__":
    main()
