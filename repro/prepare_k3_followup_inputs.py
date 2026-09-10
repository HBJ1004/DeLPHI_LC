"""Acquire and prepare label-blind inputs for the frozen K3 follow-up study.

There is deliberately no prediction or reference-scoring command here.
Network access occurs only in subcommands whose names start with ``fetch-``.
"""

from __future__ import annotations

import argparse
import json
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
)


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
