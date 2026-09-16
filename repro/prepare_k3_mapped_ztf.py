"""Stage label-blind, DAMIT-to-MPC-mapped ZTF preparation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lc_pipeline.k3.mapped_ztf import (
    fetch_mapped_fink,
    fetch_mapped_horizons,
    ingest_mapped_fink,
    prepare_mapped_ztf,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest", help="normalize local MPC-numbered Fink files")
    fetch_fink = commands.add_parser("fetch-fink", help="fetch explicit MPC-numbered public Fink rows")
    fetch = commands.add_parser("fetch-horizons", help="fetch/resume mapped MPC Horizons caches")
    prepare = commands.add_parser("prepare", help="build internal-ID prepared objects")
    for item in (ingest, fetch_fink, fetch, prepare):
        item.add_argument("--identity-map", type=Path, required=True)
        item.add_argument("--output", type=Path, required=True)
    ingest.add_argument("--period-manifest", type=Path, required=True)
    ingest.add_argument("--raw-directory", type=Path, required=True)
    ingest.add_argument("--fetch-receipts-directory", type=Path)
    ingest.add_argument("--fetch-receipt-override-directory", type=Path, action="append", default=[])
    ingest.add_argument("--raw-override-directory", type=Path, action="append", default=[])
    ingest.add_argument("--maximum-sigma-magnitude", type=float, default=0.2)
    fetch_fink.add_argument("--object-id", action="append")
    fetch_fink.add_argument("--timeout-seconds", type=float, default=60.0)
    fetch_fink.add_argument("--request-interval-seconds", type=float, default=1.0)
    fetch.add_argument("--ingest-manifest", type=Path, required=True)
    fetch.add_argument("--batch-size", type=int, default=20)
    fetch.add_argument("--timeout-seconds", type=float, default=120.0)
    fetch.add_argument("--request-interval-seconds", type=float, default=1.0)
    fetch.add_argument("--reuse-horizons-root", action="append", default=[])
    prepare.add_argument("--ingest-manifest", type=Path, required=True)
    prepare.add_argument("--horizons-manifest", type=Path, required=True)
    prepare.add_argument("--period-manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "ingest":
        result = ingest_mapped_fink(identity_map_path=args.identity_map, period_manifest_path=args.period_manifest, raw_directory=args.raw_directory, output_directory=args.output, maximum_sigma_magnitude=args.maximum_sigma_magnitude, fetch_receipts_directory=args.fetch_receipts_directory, fetch_receipt_override_directories=args.fetch_receipt_override_directory, raw_override_directories=args.raw_override_directory)
    elif args.command == "fetch-fink":
        result = fetch_mapped_fink(identity_map_path=args.identity_map, output_directory=args.output, object_ids=args.object_id, timeout_seconds=args.timeout_seconds, request_interval_seconds=args.request_interval_seconds)
    elif args.command == "fetch-horizons":
        result = fetch_mapped_horizons(ingest_manifest_path=args.ingest_manifest, identity_map_path=args.identity_map, output_directory=args.output, batch_size=args.batch_size, timeout_seconds=args.timeout_seconds, request_interval_seconds=args.request_interval_seconds, reuse_horizons_roots=args.reuse_horizons_root)
    else:
        result = prepare_mapped_ztf(ingest_manifest_path=args.ingest_manifest, horizons_manifest_path=args.horizons_manifest, period_manifest_path=args.period_manifest, identity_map_path=args.identity_map, output_directory=args.output)
    print(json.dumps({key: value for key, value in result.items() if key != "objects"}, sort_keys=True))


if __name__ == "__main__":
    main()
