"""Build the metadata-only K3 identity-exposure inventory.

This command never loads reference pole solutions or runs a model.  Every
source location is supplied explicitly, and existing output files are never
overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Mapping

from lc_pipeline.k3.generalization_validation import (
    GeneralizationValidationError,
    audit_identity_exposure,
    canonical_json,
    sha256_file,
)

INVENTORY_SCHEMA = "delphi.k3-exposure-source-inventory.v1"
EXPECTED_MANIFEST_COUNT = 96
EXPECTED_CENSUS_COUNT = 943
EXPECTED_FINK_FILE_COUNT = 179
PRIOR_POST_CUTOFF_IDS = ("mpc:49", "mpc:279", "mpc:366")
_MPC_ID = re.compile(r"^mpc:[1-9][0-9]*$")


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


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _manifest_set(
    root: Path, *, expected_count: int
) -> tuple[list[object], dict[str, object]]:
    resolved = root.resolve()
    if not resolved.is_dir():
        raise GeneralizationValidationError(f"manifest root is missing: {resolved}")
    paths = sorted(resolved.rglob("*.manifest.json"))
    if len(paths) != expected_count:
        raise GeneralizationValidationError(
            f"expected {expected_count} manifests below {resolved}, found {len(paths)}"
        )
    rows = [
        {
            "relative_path": path.relative_to(resolved).as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in paths
    ]
    documents = [_read_json(path) for path in paths]
    for path, document in zip(paths, documents, strict=True):
        if not isinstance(document, Mapping) or document.get("schema") != "delphi.k3-synthetic-shard.v1":
            raise GeneralizationValidationError(f"invalid synthetic shard manifest: {path}")
    return documents, {
        "root": str(resolved),
        "manifest_count": len(rows),
        "manifest_set_sha256": _canonical_digest(rows),
        "manifests": rows,
    }


def _unknown_directory(root: Path) -> dict[str, object]:
    resolved = root.resolve()
    if not resolved.exists():
        return {
            "path": str(resolved),
            "state": "missing",
            "coverage": "unknown",
            "reason": "historical_artifact_path_missing",
        }
    if not resolved.is_dir():
        raise GeneralizationValidationError(f"expected a directory: {resolved}")
    files = sorted(path for path in resolved.rglob("*") if path.is_file())
    manifests = [path for path in files if path.name.endswith(".manifest.json")]
    if manifests:
        raise GeneralizationValidationError(
            "eef6d4ec now contains donor manifests; inventory logic must be reviewed"
        )
    rows = [
        {
            "relative_path": path.relative_to(resolved).as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in files
    ]
    directories = sorted(
        path.relative_to(resolved).as_posix()
        for path in resolved.rglob("*")
        if path.is_dir()
    )
    return {
        "path": str(resolved),
        "state": "empty" if not files else "no_donor_manifests",
        "coverage": "unknown",
        "reason": "no_auditable_donor_manifest",
        "file_count": len(rows),
        "directory_snapshot_sha256": _canonical_digest(
            {"directories": directories, "files": rows}
        ),
        "files": rows,
    }


def _fink_inventory(
    raw_dir: Path, fetch_script: Path, *, expected_count: int
) -> tuple[list[dict[str, int]], dict[str, object]]:
    raw_root = raw_dir.resolve()
    if not raw_root.is_dir():
        raise GeneralizationValidationError(f"historical Fink raw directory missing: {raw_root}")
    script = fetch_script.resolve()
    script_text = script.read_text(encoding="utf-8")
    namespace_checks = {
        "documents_iau_asteroid_number": "IAU asteroid number" in script_text,
        "passes_number_to_fink_n_or_d": bool(
            re.search(r"[\"']n_or_d[\"']\s*:\s*str\(asteroid_number\)", script_text)
        ),
    }
    if not all(namespace_checks.values()):
        raise GeneralizationValidationError(
            "fetch script does not establish that numeric raw filenames are MPC/IAU numbers"
        )
    paths = sorted(raw_root.glob("*.json"), key=lambda path: path.name)
    if len(paths) != expected_count:
        raise GeneralizationValidationError(
            f"expected {expected_count} historical Fink JSON files, found {len(paths)}"
        )
    numbers: list[int] = []
    rows: list[dict[str, object]] = []
    for path in paths:
        if not re.fullmatch(r"[1-9][0-9]*", path.stem):
            raise GeneralizationValidationError(
                f"historical Fink filename is not a positive MPC number: {path.name}"
            )
        number = int(path.stem)
        numbers.append(number)
        rows.append(
            {
                "relative_path": path.name,
                "mpc_identity": f"mpc:{number}",
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    if len(numbers) != len(set(numbers)):
        raise GeneralizationValidationError("historical Fink filenames contain duplicate MPC IDs")
    return ([{"mpc_number": number} for number in numbers], {
        "raw_root": str(raw_root),
        "raw_json_count": len(rows),
        "raw_file_set_sha256": _canonical_digest(rows),
        "files": rows,
        "namespace": "mpc",
        "namespace_evidence": {
            "fetch_script": _artifact(script),
            "checks": namespace_checks,
        },
    })


def _legacy_observed_mpc(document: object) -> list[dict[str, int]]:
    if not isinstance(document, Mapping) or not isinstance(document.get("objects"), list):
        raise GeneralizationValidationError("legacy ZTF identity audit lacks objects")
    rows: list[dict[str, int]] = []
    for row in document["objects"]:
        observed = row.get("observed_mpc_number") if isinstance(row, Mapping) else None
        if not isinstance(observed, int) or isinstance(observed, bool) or observed <= 0:
            raise GeneralizationValidationError(
                "legacy ZTF identity audit has an invalid observed MPC number"
            )
        rows.append({"mpc_number": observed})
    if len(rows) != len({row["mpc_number"] for row in rows}):
        raise GeneralizationValidationError("legacy ZTF actual query identities are duplicated")
    return rows


def _census_ids(document: object, *, expected_count: int) -> list[str]:
    if not isinstance(document, Mapping):
        raise GeneralizationValidationError("census crossmatch must be a JSON object")
    availability = document.get("reference_availability")
    if not isinstance(availability, Mapping):
        raise GeneralizationValidationError("census lacks reference_availability")
    if availability.get("poles_loaded") is not False or availability.get("solutions_loaded") is not False:
        raise GeneralizationValidationError(
            "exposure inventory requires a crossmatch with no pole/solution values loaded"
        )
    records = availability.get("records")
    if not isinstance(records, list) or len(records) != expected_count:
        raise GeneralizationValidationError(
            f"expected {expected_count} census identity records"
        )
    identities = [
        row.get("candidate_identity") if isinstance(row, Mapping) else None
        for row in records
    ]
    if any(not isinstance(value, str) or not _MPC_ID.fullmatch(value) for value in identities):
        raise GeneralizationValidationError("census contains a non-MPC candidate identity")
    if len(identities) != len(set(identities)):
        raise GeneralizationValidationError("census candidate identities are not unique")
    if document.get("crossmatch_count") != len(identities):
        raise GeneralizationValidationError("census crossmatch_count is inconsistent")
    return list(identities)


def _role_counts(audit: Mapping[str, object]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    objects = audit.get("objects")
    if not isinstance(objects, list):
        return {}
    for row in objects:
        if not isinstance(row, Mapping):
            continue
        for reason in row.get("reasons", []):
            if isinstance(reason, str) and reason.startswith("present_in:"):
                counts[reason.removeprefix("present_in:")] += 1
    return dict(sorted(counts.items()))


def _official_physical_map(aliases: object) -> dict[str, str]:
    if not isinstance(aliases, Mapping):
        raise GeneralizationValidationError("official identity aliases must be an object")
    rows = aliases.get("objects")
    if not isinstance(rows, list):
        raise GeneralizationValidationError("official identity aliases lack objects")
    result: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise GeneralizationValidationError("official identity alias row is invalid")
        object_id = row.get("object_id")
        mpc_number = row.get("mpc_number")
        if not isinstance(object_id, str):
            raise GeneralizationValidationError("official identity alias row lacks object_id")
        if isinstance(mpc_number, int) and not isinstance(mpc_number, bool) and mpc_number > 0:
            result[object_id] = f"mpc:{mpc_number}"
    return result


def _outer_test_membership(splits: object, aliases: object) -> dict[str, int]:
    if not isinstance(splits, Mapping):
        raise GeneralizationValidationError("original splits must be an object")
    folds = splits.get("folds")
    if not isinstance(folds, list):
        raise GeneralizationValidationError("original splits lack folds")
    physical = _official_physical_map(aliases)
    membership: dict[str, int] = {}
    for row in folds:
        if not isinstance(row, Mapping) or not isinstance(row.get("fold"), int):
            raise GeneralizationValidationError("original splits contain an invalid fold")
        test_ids = row.get("test_ids")
        if not isinstance(test_ids, list) or any(not isinstance(item, str) for item in test_ids):
            raise GeneralizationValidationError("original split test_ids are invalid")
        for legacy_id in test_ids:
            canonical = physical.get(legacy_id)
            if canonical is None:
                raise GeneralizationValidationError(
                    f"original split identity lacks official mapping: {legacy_id}"
                )
            if canonical in membership:
                raise GeneralizationValidationError(
                    f"original physical identity occurs in multiple outer tests: {canonical}"
                )
            membership[canonical] = int(row["fold"])
    return membership


def build_exposure_inventory(
    *,
    census_crossmatch: Path,
    identity_aliases: Path,
    original_splits: Path,
    published_metadata_root: Path,
    historical_k3_root: Path,
    eef6d4ec_root: Path,
    legacy_ztf_audit: Path,
    historical_fink_raw_dir: Path,
    historical_fink_fetch_script: Path,
    expected_manifest_count: int = EXPECTED_MANIFEST_COUNT,
    expected_census_count: int = EXPECTED_CENSUS_COUNT,
    expected_fink_file_count: int = EXPECTED_FINK_FILE_COUNT,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Return the census audit, prior-three audit, and source receipt."""
    census_document = _read_json(census_crossmatch)
    candidate_ids = _census_ids(census_document, expected_count=expected_census_count)
    aliases = _read_json(identity_aliases)
    splits = _read_json(original_splits)
    legacy = _read_json(legacy_ztf_audit)
    legacy_actual_queries = _legacy_observed_mpc(legacy)
    published, published_inventory = _manifest_set(
        published_metadata_root, expected_count=expected_manifest_count
    )
    historical, historical_inventory = _manifest_set(
        historical_k3_root, expected_count=expected_manifest_count
    )
    published_hashes = {
        row["relative_path"]: row["sha256"] for row in published_inventory["manifests"]
    }
    historical_hashes = {
        row["relative_path"]: row["sha256"] for row in historical_inventory["manifests"]
    }
    if published_hashes != historical_hashes:
        raise GeneralizationValidationError(
            "published metadata and frozen k3-2408c563 manifests are not byte-identical"
        )
    eef_inventory = _unknown_directory(eef6d4ec_root)
    fink_rows, fink_inventory = _fink_inventory(
        historical_fink_raw_dir,
        historical_fink_fetch_script,
        expected_count=expected_fink_file_count,
    )
    common_sources: dict[str, object] = {
        "original_170_publication_splits": splits,
        "published_final_synthetic_96_shards": published,
        "historical_k3_2408c563_synthetic_96_shards": historical,
        "historical_fink_mpc_queries": fink_rows,
        "invalid_legacy_ztf_actual_mpc_queries": legacy_actual_queries,
        "prior_post_cutoff_evaluated_cases": {"object_ids": list(PRIOR_POST_CUTOFF_IDS)},
        "historical_k3_eef6d4ec_donor_registry": {
            "status": "unknown",
            "registry_complete": False,
        },
        "unregistered_project_history": {
            "status": "unknown",
            "registry_complete": False,
        },
    }
    census_audit = audit_identity_exposure(
        candidate_ids, exposure_sources=common_sources, aliases=aliases
    )
    prior_audit = audit_identity_exposure(
        list(PRIOR_POST_CUTOFF_IDS), exposure_sources=common_sources, aliases=aliases
    )
    outer_test_membership = _outer_test_membership(splits, aliases)
    coverage = {
        "exhaustive_project_history_certified": False,
        "confirmatory_unexposed_claim_permitted": False,
        "unknown_status_interpretation": (
            "not_found_in_the_enumerated_sources_but_not_cleared_against_exhaustive_history"
        ),
        "known_absence_is_not_certified_absence": True,
        "coverage_blockers": [
            "historical_k3_eef6d4ec_has_no_auditable_donor_manifest",
            "no_complete_project_wide_identity_exposure_registry",
        ],
    }
    census_audit["coverage_boundary"] = coverage
    census_audit["known_exposed_by_source_nonexclusive"] = _role_counts(census_audit)
    census_audit["source_inventory_schema"] = INVENTORY_SCHEMA
    prior_audit["coverage_boundary"] = coverage
    prior_audit["known_exposed_by_source_nonexclusive"] = _role_counts(prior_audit)
    prior_overlap_counts: Counter[str] = Counter()
    for row in prior_audit["objects"]:
        reasons = set(row["reasons"])
        object_id = str(row["object_id"])
        annotations = {
            "original_170_outer_test_fold": outer_test_membership.get(object_id),
            "original_170_member": object_id in outer_test_membership,
            "published_final_synthetic_donor": (
                "present_in:published_final_synthetic_96_shards" in reasons
            ),
            "historical_k3_2408c563_synthetic_donor": (
                "present_in:historical_k3_2408c563_synthetic_96_shards" in reasons
            ),
            "historical_fink_query": "present_in:historical_fink_mpc_queries" in reasons,
            "invalid_legacy_ztf_query": (
                "present_in:invalid_legacy_ztf_actual_mpc_queries" in reasons
            ),
        }
        row["enumerated_source_overlap"] = annotations
        for key, value in annotations.items():
            if value is True:
                prior_overlap_counts[key] += 1
    prior_audit["interpretation"] = (
        "all_three_are_development_exposed_by_prior_evaluation; no new pole scoring performed"
    )
    prior_audit["source_inventory_schema"] = INVENTORY_SCHEMA

    legacy_objects = legacy.get("objects") if isinstance(legacy, Mapping) else None
    receipt = {
        "schema": INVENTORY_SCHEMA,
        "purpose": "metadata_only_identity_exposure_inventory",
        "pole_values_loaded": False,
        "model_execution_performed": False,
        "inputs": {
            "census_crossmatch": _artifact(census_crossmatch),
            "identity_aliases": _artifact(identity_aliases),
            "original_splits": _artifact(original_splits),
            "legacy_ztf_actual_mpc_identity_audit": {
                **_artifact(legacy_ztf_audit),
                "object_count": len(legacy_objects) if isinstance(legacy_objects, list) else None,
            },
            "published_final_synthetic_metadata": published_inventory,
            "historical_k3_2408c563_synthetic_metadata": historical_inventory,
            "published_and_historical_manifests_byte_identical": True,
            "historical_k3_eef6d4ec": eef_inventory,
            "historical_fink_mpc_queries": fink_inventory,
            "prior_post_cutoff_evaluated_cases": list(PRIOR_POST_CUTOFF_IDS),
            "identity_resolution": {
                "legacy_asteroid_ids_are": "damit_internal_ids",
                "candidate_and_fink_ids_are": "mpc_numbered_designations",
                "numeric_suffix_collapse_across_namespaces": False,
                "official_table_sha256": (
                    aliases.get("identity_table_sha256")
                    if isinstance(aliases, Mapping)
                    else None
                ),
            },
        },
        "counts": {
            "census_candidates": len(candidate_ids),
            "census_known_exposed": census_audit["counts"]["exposed"],
            "census_unknown": census_audit["counts"]["unknown"],
            "census_certified_unexposed": census_audit["counts"]["unexposed"],
            "prior_post_cutoff_cases": len(PRIOR_POST_CUTOFF_IDS),
            "prior_three_enumerated_source_overlaps": dict(
                sorted(prior_overlap_counts.items())
            ),
        },
        "coverage_boundary": coverage,
    }
    return census_audit, prior_audit, receipt


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _write_outputs(
    output_dir: Path,
    census_audit: dict[str, object],
    prior_audit: dict[str, object],
    receipt: dict[str, object],
    *,
    revision: str | None = None,
) -> None:
    output_root = output_dir.resolve()
    if revision is not None and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", revision):
        raise GeneralizationValidationError("output revision contains unsafe characters")
    suffix = "" if revision is None else f"-{revision}"
    names = {
        "census_audit": f"k3-census-exposure-audit{suffix}.json",
        "prior_three_audit": f"k3-prior-three-exposure-audit{suffix}.json",
        "source_inventory": f"k3-exposure-source-inventory{suffix}.json",
    }
    targets = {role: output_root / name for role, name in names.items()}
    existing = [str(path) for path in targets.values() if path.exists()]
    if existing:
        raise GeneralizationValidationError(
            f"refusing to overwrite existing exposure output(s): {existing}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    census_bytes = _json_bytes(census_audit)
    prior_bytes = _json_bytes(prior_audit)
    receipt["derived_outputs"] = {
        "census_audit": {
            "filename": names["census_audit"],
            "sha256": hashlib.sha256(census_bytes).hexdigest(),
            "size_bytes": len(census_bytes),
        },
        "prior_three_audit": {
            "filename": names["prior_three_audit"],
            "sha256": hashlib.sha256(prior_bytes).hexdigest(),
            "size_bytes": len(prior_bytes),
        },
    }
    payloads = {
        targets["census_audit"]: census_bytes,
        targets["prior_three_audit"]: prior_bytes,
        targets["source_inventory"]: _json_bytes(receipt),
    }
    temporary_paths: list[Path] = []
    try:
        for target, payload in payloads.items():
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{target.name}.", dir=output_root
            )
            temporary = Path(temporary_name)
            temporary_paths.append(temporary)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        for target, temporary in zip(payloads, temporary_paths, strict=True):
            os.replace(temporary, target)
    finally:
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--census-crossmatch", type=Path, required=True)
    root.add_argument("--identity-aliases", type=Path, required=True)
    root.add_argument("--original-splits", type=Path, required=True)
    root.add_argument("--published-metadata-root", type=Path, required=True)
    root.add_argument("--historical-k3-root", type=Path, required=True)
    root.add_argument("--eef6d4ec-root", type=Path, required=True)
    root.add_argument("--legacy-ztf-audit", type=Path, required=True)
    root.add_argument("--historical-fink-raw-dir", type=Path, required=True)
    root.add_argument("--historical-fink-fetch-script", type=Path, required=True)
    root.add_argument("--output-dir", type=Path, required=True)
    root.add_argument("--revision")
    return root


def main() -> None:
    args = parser().parse_args()
    census_audit, prior_audit, receipt = build_exposure_inventory(
        census_crossmatch=args.census_crossmatch,
        identity_aliases=args.identity_aliases,
        original_splits=args.original_splits,
        published_metadata_root=args.published_metadata_root,
        historical_k3_root=args.historical_k3_root,
        eef6d4ec_root=args.eef6d4ec_root,
        legacy_ztf_audit=args.legacy_ztf_audit,
        historical_fink_raw_dir=args.historical_fink_raw_dir,
        historical_fink_fetch_script=args.historical_fink_fetch_script,
    )
    _write_outputs(
        args.output_dir, census_audit, prior_audit, receipt, revision=args.revision
    )


if __name__ == "__main__":
    main()
