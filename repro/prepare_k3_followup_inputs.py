"""Acquire and prepare label-blind inputs for the frozen K3 follow-up study.

There is deliberately no prediction or reference-scoring command here.
Network access occurs only in subcommands whose names start with ``fetch-``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from lc_pipeline.k3.temporal_external import (
    fetch_temporal_damit_snapshot,
    prepare_temporal_damit_inputs,
)
from lc_pipeline.k3.ztf_external import (
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

    horizons = commands.add_parser(
        "fetch-horizons", help="fetch missing vectors from the official JPL API"
    )
    horizons.add_argument("--normalized-manifest", type=Path, required=True)
    horizons.add_argument("--output-directory", type=Path, required=True)
    horizons.add_argument("--batch-size", type=int, default=20)
    horizons.add_argument("--timeout-seconds", type=float, default=120.0)
    horizons.add_argument("--request-interval-seconds", type=float, default=0.25)

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
    elif args.command == "fetch-horizons":
        result = fetch_horizons_directory(
            normalized_manifest_path=args.normalized_manifest,
            output_directory=args.output_directory,
            batch_size=args.batch_size,
            timeout_seconds=args.timeout_seconds,
            request_interval_seconds=args.request_interval_seconds,
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
