"""Create a label-blind, hash-verified interpretation of archived development runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# This script is a documented source-tree entry point as well as a module.
# Running its file path makes Python place ``repro/`` (not the checkout root)
# on sys.path, so bootstrap only that local root.  Installed users can use the
# same CLI through ``python -m repro...`` without relying on this branch.
_SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from lc_pipeline.k3.convergence_reinterpretation import (  # noqa: E402
    reinterpret_development_execution_graph,
    score_reinterpreted_development,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    reinterpret = commands.add_parser(
        "reinterpret", help="create a label-blind interpretation receipt"
    )
    reinterpret.add_argument("--parent-root", type=Path, required=True)
    reinterpret.add_argument("--output", type=Path, required=True)
    reinterpret.add_argument("--study-spec", type=Path, required=True)
    reinterpret.add_argument("--revised-study-lock", type=Path, required=True)
    reinterpret.add_argument("--development-manifest", type=Path, required=True)
    score = commands.add_parser("score", help="score an already sealed interpretation receipt")
    score.add_argument("--receipt", type=Path, required=True)
    score.add_argument("--receipt-sha256", required=True)
    score.add_argument("--reference-catalog", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    if arguments.command == "reinterpret":
        result = reinterpret_development_execution_graph(
            parent_root=arguments.parent_root,
            output_directory=arguments.output,
            study_spec_path=arguments.study_spec,
            revised_lock_path=arguments.revised_study_lock,
            development_manifest_path=arguments.development_manifest,
        )
    else:
        result = score_reinterpreted_development(
            receipt_path=arguments.receipt,
            expected_receipt_sha256=arguments.receipt_sha256,
            reference_catalog_path=arguments.reference_catalog,
            output_path=arguments.output,
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
