#!/usr/bin/env python3
"""Build a deterministic metadata-only external eligibility inventory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lc_pipeline.k3.external_eligibility import (
    ExternalEligibilityError,
    build_eligibility_inventory,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSON list of metadata rows")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        rows = json.loads(args.input.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ExternalEligibilityError("input must be a JSON list")
        result = build_eligibility_inventory(rows)
    except (OSError, UnicodeError, json.JSONDecodeError, ExternalEligibilityError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
