"""Acquire and prepare label-blind inputs for the frozen K3 follow-up study.

There is deliberately no prediction or reference-scoring command here.
Network access occurs only in subcommands whose names start with ``fetch-``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Sequence

from lc_pipeline.k3.external_provenance import (
    build_external_model_manifest,
    build_ztf_period_manifest,
    prepare_ztf_followup_directory,
)
from lc_pipeline.k3.temporal_external import (
    fetch_temporal_damit_snapshot,
    prepare_temporal_damit_inputs,
)
from lc_pipeline.k3.ztf_external import (
    audit_horizons_directory_identities,
    fetch_horizons_directory,
    ingest_fink_directory,
    plan_horizons_directory,
    rebind_horizons_directory,
)
from repro.network_root import LOCAL_SCHEMA, SEEDS, guard_output, sha256


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    ingest = commands.add_parser("ingest-fink", help="normalize raw Fink JSON offline")
    ingest.add_argument("--raw-directory", type=Path, required=True)
    ingest.add_argument("--splits", type=Path, required=True)
    ingest.add_argument("--spec", type=Path, required=True)
    ingest.add_argument("--output-directory", type=Path, required=True)

    plan = commands.add_parser(
        "plan-horizons", help="audit resume state without network access or writes"
    )
    plan.add_argument("--normalized-manifest", type=Path, required=True)
    plan.add_argument("--output-directory", type=Path, required=True)
    plan.add_argument("--batch-size", type=int, default=20)

    audit = commands.add_parser(
        "audit-horizons", help="bind retained responses to resolved target identities offline"
    )
    audit.add_argument("--normalized-manifest", type=Path, required=True)
    audit.add_argument("--horizons-manifest", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)

    rebind = commands.add_parser(
        "rebind-horizons", help="rebind complete caches after a metadata-only spec refreeze"
    )
    rebind.add_argument("--source-normalized-manifest", type=Path, required=True)
    rebind.add_argument("--final-normalized-manifest", type=Path, required=True)
    rebind.add_argument("--source-horizons-manifest", type=Path, required=True)
    rebind.add_argument("--output-directory", type=Path, required=True)

    horizons = commands.add_parser(
        "fetch-horizons", help="fetch missing vectors from the official JPL API"
    )
    horizons.add_argument("--normalized-manifest", type=Path, required=True)
    horizons.add_argument("--output-directory", type=Path, required=True)
    horizons.add_argument("--batch-size", type=int, default=20)
    horizons.add_argument("--timeout-seconds", type=float, default=120.0)
    horizons.add_argument("--request-interval-seconds", type=float, default=0.25)

    periods = commands.add_parser(
        "build-ztf-period-manifest", help="export fixed periods without pole labels"
    )
    periods.add_argument("--normalized-manifest", type=Path, required=True)
    periods.add_argument("--catalog", type=Path, required=True)
    periods.add_argument("--spec", type=Path, required=True)
    periods.add_argument("--output", type=Path, required=True)

    ztf_prepare = commands.add_parser(
        "prepare-ztf", help="bulk prepare only after the complete Horizons identity audit"
    )
    ztf_prepare.add_argument("--normalized-manifest", type=Path, required=True)
    ztf_prepare.add_argument("--horizons-manifest", type=Path, required=True)
    ztf_prepare.add_argument("--horizons-identity-audit", type=Path, required=True)
    ztf_prepare.add_argument("--period-manifest", type=Path, required=True)
    ztf_prepare.add_argument("--spec", type=Path, required=True)
    ztf_prepare.add_argument("--output-directory", type=Path, required=True)

    models = commands.add_parser(
        "bind-models", help="verify and bind all 25 publication checkpoints"
    )
    models.add_argument("--artifact-root", type=Path, required=True)
    models.add_argument("--archive-index", type=Path, required=True)
    models.add_argument("--publication-package-manifest", type=Path, required=True)
    models.add_argument("--output", type=Path, required=True)

    local = commands.add_parser(
        "bind-local-rerun", help="bind local checkpoints/evaluations without claiming publication promotion"
    )
    local.add_argument("--artifact-root", type=Path, required=True)
    local.add_argument("--output", type=Path, required=True)
    local.add_argument("--allow-unpublished-local-rerun", action="store_true", required=True)

    bundles = commands.add_parser("export-local-rerun-bundles", help="export freshly calibrated local ensembles")
    for name in ("artifact-root", "splits", "output", "repository"):
        bundles.add_argument("--" + name, type=Path, required=True)
    bundles.add_argument("--allow-unpublished-local-rerun", action="store_true", required=True)

    step = commands.add_parser(
        "local-rerun-step", help="call an unchanged evaluation function with a recorded dirty-checkout exception"
    )
    step.add_argument("--allow-unpublished-local-rerun", action="store_true", required=True)
    step.add_argument("argv", nargs=argparse.REMAINDER)

    temporal = commands.add_parser(
        "fetch-temporal-damit", help="freeze official DAMIT inputs and references"
    )
    temporal.add_argument("--spec", type=Path, required=True)
    temporal.add_argument("--output-directory", type=Path, required=True)
    temporal.add_argument("--timeout-seconds", type=float, default=120.0)
    temporal.add_argument("--request-interval-seconds", type=float, default=0.25)

    prepare = commands.add_parser(
        "prepare-temporal-damit", help="prepare inputs without opening references"
    )
    prepare.add_argument("--snapshot-directory", type=Path, required=True)
    prepare.add_argument("--output-directory", type=Path, required=True)
    return parser


def local_rerun_step(argv: list[str]) -> dict:
    """Only the publication clean-Git gate is skipped; no numerical code changes."""
    from lc_pipeline.k3 import cli
    from lc_pipeline.k3 import publication_run as run

    if argv and argv[0] == "--":
        argv = argv[1:]
    args = cli._parser().parse_args(argv)
    allowed = {
        "evaluate-publication-real-oof", "aggregate-publication-real-oof",
        "evaluate-publication-real-oof-ensemble", "evaluate-publication-synthetic",
        "evaluate-publication-real-calibration", "evaluate-publication-real-calibration-ensemble",
        "calibrate-publication-real-oof-ensemble",
    }
    if args.command not in allowed:
        raise ValueError("local rerun adapter permits only headline prerequisite evaluations")
    guard_output(args.output)
    if args.command == "evaluate-publication-real-oof":
        result = run.evaluate_real_oof_checkpoint(
            args.checkpoint, args.catalog, args.dump_root, args.splits, args.output,
            fold=args.fold, seed=args.seed, device=args.device, batch_size=args.batch_size)
    elif args.command == "evaluate-publication-real-calibration":
        result = run.evaluate_real_calibration_checkpoint(
            args.checkpoint, args.catalog, args.dump_root, args.splits, args.output,
            fold=args.fold, seed=args.seed, device=args.device, batch_size=args.batch_size)
    elif args.command == "evaluate-publication-real-calibration-ensemble":
        result = run.evaluate_real_calibration_ensemble(
            args.evaluation_directory, args.model_directory, args.catalog, args.dump_root,
            args.splits, args.output, seeds=args.seeds, device=args.device)
    elif args.command == "calibrate-publication-real-oof-ensemble":
        result = run.calibrate_real_oof_ensemble_artifacts(
            args.calibration_ensemble, args.splits, args.oof_ensemble, args.output)
    elif args.command == "aggregate-publication-real-oof":
        result = run.aggregate_real_oof_artifacts(
            args.evaluation_directory, args.splits, args.output, seed=args.seed)
    elif args.command == "evaluate-publication-real-oof-ensemble":
        result = run.evaluate_real_oof_ensemble(
            args.evaluation_directory, args.model_directory, args.catalog, args.dump_root,
            args.splits, args.output, seeds=args.seeds, device=args.device)
    else:
        result = run.evaluate_synthetic_checkpoint(
            args.checkpoint, args.test_directory, args.output,
            device=args.device, batch_size=args.batch_size)
    receipt = {
        "schema": "delphi.k3-local-rerun-step.v1", "argv": argv,
        "skipped_publication_gates": ["clean committed export/evaluation checkout"],
        "publication_promotion_claimed": False, "result": result,
        "artifact_sha256": sha256(args.output),
    }
    if hasattr(args, "checkpoint"):
        receipt["checkpoint_sha256"] = sha256(args.checkpoint)
    with Path(str(args.output) + ".receipt.json").open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    return receipt


def local_evaluation_receipt(root: Path, path: Path, checkpoint_sha256: str | None = None) -> dict:
    """Read a sidecar or the native CLI's hash receipt in the training log."""
    digest = sha256(path)
    sidecar = Path(str(path) + ".receipt.json")
    if sidecar.exists():
        value = json.loads(sidecar.read_text())
        if value.get("artifact_sha256") != digest:
            raise ValueError(f"evaluation receipt hash mismatch: {path}")
        return {**value, "receipt_source": str(sidecar)}
    for log in sorted(root.parent.glob("train-queue*.log")):
        with log.open(errors="replace") as stream:
            for line in stream:
                if not line.startswith("{"):
                    continue
                try:
                    value = json.loads(line).get("summary", {})
                except json.JSONDecodeError:
                    continue
                if (value.get("artifact_sha256") == digest
                        and (checkpoint_sha256 is None or value.get("checkpoint_sha256") == checkpoint_sha256)):
                    return {**value, "receipt_source": str(log)}
    raise ValueError(f"no native CLI or local hash receipt for {path}")


def bind_local_rerun(root: Path, output: Path) -> dict:
    """A distinct local binding; no invented archive or promotion evidence."""
    import numpy as np
    import torch

    from lc_pipeline.k3.publication_run import audit_publication_checkpoint_set

    guard_output(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    audit_path = output.parent / "checkpoint-audit.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        if audit.get("passed") is not True or any(
            sha256(root / "models" / name) != digest
            for name, digest in audit.get("checkpoint_sha256", {}).items()
        ) or len(audit.get("checkpoint_sha256", {})) != 31:
            raise ValueError("existing checkpoint audit differs from this run")
    else:
        audit = audit_publication_checkpoint_set(root / "models", audit_path)
    records, hashes, receipts = [], {}, {}
    for fold in range(5):
        for seed in SEEDS:
            filename = f"real-fold-{fold}-seed-{seed}.pt"
            path = root / "models" / filename
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            records.append({
                "filename": filename, "fold": fold, "seed": seed,
                "sha256": audit["checkpoint_sha256"][filename],
                "size_bytes": path.stat().st_size,
                "run_provenance": checkpoint["run_provenance"],
            })
            evaluation = root / "evaluations" / f"real-fold-{fold}-seed-{seed}.npz"
            receipt = local_evaluation_receipt(root, evaluation, audit["checkpoint_sha256"][filename])
            if (receipt.get("checkpoint_sha256") != audit["checkpoint_sha256"][filename]
                    or receipt.get("artifact_sha256") != sha256(evaluation)):
                raise ValueError(f"evaluation/checkpoint receipt mismatch: {evaluation}")
    with np.load(root / "evaluations/real-oof-ensemble.npz", allow_pickle=False) as ensemble:
        bound = dict(zip(ensemble["checkpoint_names"].astype(str),
                         ensemble["checkpoint_sha256"].astype(str), strict=True))
        if bound != {r["filename"]: r["sha256"] for r in records}:
            raise ValueError("OOF ensemble does not bind the chosen 25 models")
    needed = [root / "evaluations" / f"real-fold-{fold}-seed-{seed}.npz"
              for fold in range(5) for seed in SEEDS]
    needed += [root / "evaluations" / f"real-oof-seed-{seed}.npz" for seed in SEEDS]
    needed += [root / "evaluations/real-oof-ensemble.npz"]
    needed += [root / "evaluations" / f"synthetic-test-seed-{seed}.npz" for seed in SEEDS]
    for path in needed:
        checkpoint_name = path.name.replace("synthetic-test-", "synthetic-").replace(".npz", ".pt")
        checkpoint_hash = audit["checkpoint_sha256"].get(checkpoint_name)
        receipt = local_evaluation_receipt(root, path, checkpoint_hash)
        if receipt.get("artifact_sha256") != sha256(path):
            raise ValueError(f"evaluation receipt hash mismatch: {path}")
        if path.name.startswith("synthetic-test-seed-"):
            checkpoint_name = path.name.replace("synthetic-test-", "synthetic-").replace(".npz", ".pt")
            if receipt.get("checkpoint_sha256") != audit["checkpoint_sha256"][checkpoint_name]:
                raise ValueError(f"synthetic evaluation checkpoint mismatch: {path}")
        hashes[path.relative_to(root).as_posix()] = sha256(path)
        receipts[path.relative_to(root).as_posix()] = receipt
    for seed in SEEDS:
        aggregate = root / "evaluations" / f"real-oof-seed-{seed}.npz"
        with np.load(aggregate, allow_pickle=False) as arrays:
            expected = [sha256(root / "evaluations" / f"real-fold-{fold}-seed-{seed}.npz") for fold in range(5)]
            if arrays["input_artifact_sha256"].astype(str).tolist() != expected:
                raise ValueError(f"seed aggregate binds other fold evaluations: {aggregate}")
    with np.load(root / "evaluations/real-oof-ensemble.npz", allow_pickle=False) as arrays:
        expected = [sha256(root / "evaluations" / f"real-oof-seed-{seed}.npz") for seed in SEEDS]
        if arrays["input_aggregate_sha256"].astype(str).tolist() != expected:
            raise ValueError("ensemble binds other seed aggregates")
    repo = Path(__file__).resolve().parents[1]
    value = {
        "schema": LOCAL_SCHEMA, "artifact_root": str(root.resolve()),
        "protocol_sha256": audit["protocol_sha256"],
        "analysis_commit": audit["implementation_commit"],
        "model_count": 25, "models": records, "artifact_sha256": hashes,
        "evaluation_receipts": receipts,
        "checkpoint_audit_sha256": sha256(audit_path),
        "publication_promotion_claimed": False,
        "skipped_publication_gates": [
            "clean committed analysis checkout", "publication package/archive release",
            "fixed-period downstream and control promotion results",
            "calibration and safe-bundle export (headline uses trusted local checkpoints)",
        ],
        "analysis_checkout_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "analysis_diff_sha256": hashlib.sha256(subprocess.check_output(
            ["git", "diff", "HEAD"], cwd=repo)).hexdigest(),
        "source_sha256": {name: sha256(repo / name) for name in (
            "repro/network_root.py", "repro/prepare_k3_followup_inputs.py")},
    }
    with output.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return value


def _summary(value: dict[str, object]) -> dict[str, object]:
    keys = (
        "schema",
        "expected_identity_count",
        "ready_object_count",
        "missing_object_count",
        "retained_observation_count",
        "cached_object_count",
        "pending_object_count",
        "estimated_request_count",
        "object_count",
        "model_count",
        "row_count",
        "network_accessed",
        "filesystem_modified",
    )
    result = {key: value[key] for key in keys if key in value}
    if "fetches" in value:
        result["fetch_count"] = len(value["fetches"])
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "ingest-fink":
        result = ingest_fink_directory(
            raw_directory=args.raw_directory,
            splits_path=args.splits,
            study_spec_path=args.spec,
            output_directory=args.output_directory,
        )
    elif args.command == "plan-horizons":
        result = plan_horizons_directory(
            normalized_manifest_path=args.normalized_manifest,
            output_directory=args.output_directory,
            batch_size=args.batch_size,
        )
    elif args.command == "audit-horizons":
        result = audit_horizons_directory_identities(
            normalized_manifest_path=args.normalized_manifest,
            horizons_manifest_path=args.horizons_manifest,
            output_path=args.output,
        )
    elif args.command == "rebind-horizons":
        result = rebind_horizons_directory(
            source_normalized_manifest_path=args.source_normalized_manifest,
            final_normalized_manifest_path=args.final_normalized_manifest,
            source_horizons_manifest_path=args.source_horizons_manifest,
            output_directory=args.output_directory,
        )
    elif args.command == "fetch-horizons":
        result = fetch_horizons_directory(
            normalized_manifest_path=args.normalized_manifest,
            output_directory=args.output_directory,
            batch_size=args.batch_size,
            timeout_seconds=args.timeout_seconds,
            request_interval_seconds=args.request_interval_seconds,
        )
    elif args.command == "build-ztf-period-manifest":
        result = build_ztf_period_manifest(
            normalized_manifest_path=args.normalized_manifest,
            catalog_path=args.catalog,
            study_spec_path=args.spec,
            output_path=args.output,
        )
    elif args.command == "prepare-ztf":
        result = prepare_ztf_followup_directory(
            normalized_manifest_path=args.normalized_manifest,
            horizons_manifest_path=args.horizons_manifest,
            horizons_identity_audit_path=args.horizons_identity_audit,
            period_manifest_path=args.period_manifest,
            study_spec_path=args.spec,
            output_directory=args.output_directory,
        )
    elif args.command == "bind-models":
        result = build_external_model_manifest(
            artifact_root=args.artifact_root,
            archive_index_path=args.archive_index,
            publication_package_manifest_path=args.publication_package_manifest,
            output_path=args.output,
        )
    elif args.command == "bind-local-rerun":
        result = bind_local_rerun(args.artifact_root, args.output)
    elif args.command == "export-local-rerun-bundles":
        from repro.export_k3_bundles import export

        result = export(args.artifact_root, args.splits, args.output, args.repository, local_rerun=True)
    elif args.command == "local-rerun-step":
        result = local_rerun_step(args.argv)
    elif args.command == "fetch-temporal-damit":
        result = fetch_temporal_damit_snapshot(
            study_spec_path=args.spec,
            output_directory=args.output_directory,
            timeout_seconds=args.timeout_seconds,
            request_interval_seconds=args.request_interval_seconds,
        )
    elif args.command == "prepare-temporal-damit":
        result = prepare_temporal_damit_inputs(
            snapshot_directory=args.snapshot_directory,
            output_directory=args.output_directory,
        )
    else:  # pragma: no cover - argparse enforces the command set
        raise AssertionError(args.command)
    print(json.dumps(_summary(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
