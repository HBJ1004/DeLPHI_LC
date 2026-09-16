"""Explicit, no-discovery CLI for K3 generalization validation artifacts."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Mapping

from lc_pipeline.k3.generalization_validation import (
    GeneralizationValidationError,
    analyze_generalization,
    audit_identity_exposure,
    classify_reference_records,
    create_prediction_receipt,
    create_protocol_lock,
    locked_resource_contains,
    sha256_file,
    validate_external_exposure_clearance,
    validate_prediction_receipt,
    validate_protocol_lock,
)


def _read(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GeneralizationValidationError(f"cannot read {path}: {exc}") from exc


def _write_new(path: Path, payload: object) -> None:
    if path.exists():
        raise GeneralizationValidationError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _assignments(values: list[str], *, option: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        role, separator, raw_path = value.partition("=")
        if not separator or not role or not raw_path or role in result:
            raise GeneralizationValidationError(
                f"{option} values must be unique ROLE=PATH assignments"
            )
        result[role] = Path(raw_path)
    return result


def _candidate_ids(document: object) -> list[str]:
    if isinstance(document, Mapping) and isinstance(document.get("object_ids"), list):
        values = document["object_ids"]
    elif isinstance(document, Mapping) and isinstance(document.get("objects"), list):
        values = [row.get("object_id") for row in document["objects"] if isinstance(row, Mapping)]
    elif (
        isinstance(document, Mapping)
        and isinstance(document.get("reference_availability"), Mapping)
        and isinstance(document["reference_availability"].get("records"), list)
    ):
        # The public source-census crossmatch deliberately contains identities
        # and availability metadata only; no target pole values are needed for
        # an exposure audit.
        values = [
            row.get("candidate_identity")
            for row in document["reference_availability"]["records"]
            if isinstance(row, Mapping)
        ]
    elif isinstance(document, list):
        values = document
    else:
        raise GeneralizationValidationError(
            "candidate ID JSON must be a list, object_ids list, objects list, "
            "or source-census crossmatch"
        )
    if any(not isinstance(value, str) for value in values):
        raise GeneralizationValidationError("candidate IDs must be strings")
    return list(values)


def _reference_records(document: object) -> list[Mapping[str, object]]:
    raw = document.get("objects") if isinstance(document, Mapping) else document
    if not isinstance(raw, list) or any(not isinstance(row, Mapping) for row in raw):
        raise GeneralizationValidationError("reference records must be an objects list")
    return raw


def _cmd_audit(args: argparse.Namespace) -> None:
    sources = {
        role: _read(path) for role, path in _assignments(args.source, option="--source").items()
    }
    aliases = _read(args.aliases) if args.aliases is not None else None
    result = audit_identity_exposure(
        _candidate_ids(_read(args.candidate_ids)), exposure_sources=sources, aliases=aliases
    )
    result["provenance"] = {
        "candidate_ids_path": str(args.candidate_ids.resolve()),
        "candidate_ids_sha256": sha256_file(args.candidate_ids),
        "sources": {
            role: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for role, path in sorted(_assignments(args.source, option="--source").items())
        },
        "aliases": None if args.aliases is None else {
            "path": str(args.aliases.resolve()), "sha256": sha256_file(args.aliases)
        },
    }
    _write_new(args.output, result)


def _cmd_classify(args: argparse.Namespace) -> None:
    result = classify_reference_records(_reference_records(_read(args.records)))
    result["source"] = {
        "path": str(args.records.resolve()), "sha256": sha256_file(args.records)
    }
    _write_new(args.output, result)


def _cmd_lock(args: argparse.Namespace) -> None:
    result = create_protocol_lock(
        spec_path=args.spec,
        code_paths=args.code,
        model_paths=args.model,
        data_paths=args.data,
        reference_paths=args.reference,
    )
    _write_new(args.output, result)


def _cmd_seal(args: argparse.Namespace) -> None:
    result = create_prediction_receipt(
        lock_path=args.lock,
        prediction_paths=_assignments(args.prediction, option="--prediction"),
    )
    _write_new(args.output, result)


def _cmd_analyze(args: argparse.Namespace) -> None:
    lock = _read(args.lock)
    receipt = _read(args.receipt)
    if not isinstance(lock, Mapping) or not isinstance(receipt, Mapping):
        raise GeneralizationValidationError("lock and receipt must be JSON objects")
    validate_protocol_lock(lock)
    predictions = {
        "candidate": args.candidate_predictions,
        "atlas": args.atlas_predictions,
    }
    validate_prediction_receipt(
        receipt, lock_path=args.lock, expected_predictions=predictions
    )
    for bound_path, role in (
        (args.cohort, "cohort"),
        (args.exposure_audit, "exposure audit"),
    ):
        if locked_resource_contains(lock, bound_path):
            continue
        raise GeneralizationValidationError(
            f"{role} path is not explicitly bound in the lock data/reference resources"
        )
    cohort = _read(args.cohort)
    exposure_audit = _read(args.exposure_audit)
    if not isinstance(exposure_audit, Mapping):
        raise GeneralizationValidationError("exposure audit must be a JSON object")
    validate_external_exposure_clearance(cohort, exposure_audit)
    result = analyze_generalization(
        cohort=cohort,
        candidate_predictions=_read(args.candidate_predictions),
        atlas_predictions=_read(args.atlas_predictions),
        uniform_resamples=args.uniform_resamples,
        uniform_seed=args.uniform_seed,
    )
    result["provenance"] = {
        "protocol_lock_path": str(args.lock.resolve()),
        "protocol_lock_sha256": sha256_file(args.lock),
        "prediction_receipt_path": str(args.receipt.resolve()),
        "prediction_receipt_sha256": sha256_file(args.receipt),
        "cohort_path": str(args.cohort.resolve()),
        "cohort_sha256": sha256_file(args.cohort),
        "exposure_audit_path": str(args.exposure_audit.resolve()),
        "exposure_audit_sha256": sha256_file(args.exposure_audit),
        "candidate_predictions_sha256": sha256_file(args.candidate_predictions),
        "atlas_predictions_sha256": sha256_file(args.atlas_predictions),
    }
    _write_new(args.output, result)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    audit = commands.add_parser("audit-exposure")
    audit.add_argument("--candidate-ids", type=Path, required=True)
    audit.add_argument("--source", action="append", default=[], metavar="ROLE=PATH", required=True)
    audit.add_argument("--aliases", type=Path)
    audit.add_argument("--output", type=Path, required=True)
    audit.set_defaults(run=_cmd_audit)

    classify = commands.add_parser("classify-references")
    classify.add_argument("--records", type=Path, required=True)
    classify.add_argument("--output", type=Path, required=True)
    classify.set_defaults(run=_cmd_classify)

    lock = commands.add_parser("lock")
    lock.add_argument("--spec", type=Path, required=True)
    lock.add_argument("--code", type=Path, action="append", required=True)
    lock.add_argument("--model", type=Path, action="append", required=True)
    lock.add_argument("--data", type=Path, action="append", required=True)
    lock.add_argument("--reference", type=Path, action="append", required=True)
    lock.add_argument("--output", type=Path, required=True)
    lock.set_defaults(run=_cmd_lock)

    seal = commands.add_parser("seal")
    seal.add_argument("--lock", type=Path, required=True)
    seal.add_argument(
        "--prediction", action="append", default=[], metavar="ROLE=PATH", required=True
    )
    seal.add_argument("--output", type=Path, required=True)
    seal.set_defaults(run=_cmd_seal)

    analyze = commands.add_parser("analyze")
    analyze.add_argument("--cohort", type=Path, required=True)
    analyze.add_argument("--exposure-audit", type=Path, required=True)
    analyze.add_argument("--candidate-predictions", type=Path, required=True)
    analyze.add_argument("--atlas-predictions", type=Path, required=True)
    analyze.add_argument("--lock", type=Path, required=True)
    analyze.add_argument("--receipt", type=Path, required=True)
    analyze.add_argument("--uniform-resamples", type=int, default=10_000)
    analyze.add_argument("--uniform-seed", type=int, default=20260910)
    analyze.add_argument("--output", type=Path, required=True)
    analyze.set_defaults(run=_cmd_analyze)
    return root


def main() -> None:
    args = parser().parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
