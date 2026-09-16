#!/usr/bin/env python3
"""Create a small, label-blind census of extracted public source products."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lc_pipeline.k3.external_sources import (
    ExternalSourceError,
    census_external_source,
    crossmatch_damit_metadata,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Census local extracted ALCDEF PDS or TSSYS-DR1 files. "
            "No downloads, pole values, geometry, or error computations are performed."
        )
    )
    parser.add_argument("path", type=Path, nargs="?", help="local extracted source directory or source file")
    parser.add_argument(
        "--kind",
        choices=("auto", "alcdef", "tssys"),
        default="auto",
        help="source format (default: infer release.merge versus ALCDEF CSVs)",
    )
    parser.add_argument("--source-uri", help="canonical source URI to record in the JSON")
    parser.add_argument("--identity-map", type=Path, help="optional DAMIT asteroids.csv for MPC-number to internal-ID mapping")
    parser.add_argument("--crossmatch", type=Path, help="crossmatch an existing census JSON to DAMIT identities (metadata only)")
    parser.add_argument("-o", "--output", type=Path, help="write JSON here (default: stdout)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.crossmatch:
            if not args.identity_map:
                raise ExternalSourceError("--crossmatch requires --identity-map")
            result = crossmatch_damit_metadata(args.crossmatch, args.identity_map)
        else:
            if not args.path:
                raise ExternalSourceError("a source path is required unless --crossmatch is used")
            result = census_external_source(args.path, kind=args.kind, source_uri=args.source_uri, identity_map_path=args.identity_map)
    except ExternalSourceError as exc:
        raise SystemExit(f"census failed: {exc}") from exc
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
