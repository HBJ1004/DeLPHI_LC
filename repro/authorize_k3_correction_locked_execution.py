"""Create or verify a correction-bound authorization for the locked cohort.

This command does not execute a solver.  It validates the sealed corrected
development decision and writes an immutable authorization for a later,
separately approved locked-cohort command.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from lc_pipeline.k3.convergence_correction_bridge import (  # noqa: E402
    create_correction_locked_authorization,
    materialize_frozen_selection_inputs,
    validate_correction_locked_authorization,
)


def _shared(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--score", type=Path, required=True)
    parser.add_argument("--revised-lock", type=Path, required=True)
    parser.add_argument("--locked-manifest", type=Path, required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    authorize = commands.add_parser(
        "authorize", help="create a one-shot authorization without running"
    )
    _shared(authorize)
    authorize.add_argument("--receipt-sha256", required=True)
    authorize.add_argument("--score-sha256", required=True)
    authorize.add_argument("--locked-execution-directory", type=Path, required=True)
    authorize.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify", help="revalidate a previously created authorization")
    _shared(verify)
    verify.add_argument("--authorization", type=Path, required=True)
    materialize = commands.add_parser(
        "materialize-selection", help="write frozen-runner compatibility scores and selection"
    )
    materialize.add_argument("--receipt", type=Path, required=True)
    materialize.add_argument("--score", type=Path, required=True)
    materialize.add_argument("--revised-lock", type=Path, required=True)
    materialize.add_argument("--spec", type=Path, required=True)
    materialize.add_argument("--spec-checksum", type=Path, required=True)
    materialize.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    if arguments.command == "materialize-selection":
        result = materialize_frozen_selection_inputs(
            receipt_path=arguments.receipt,
            score_path=arguments.score,
            revised_lock_path=arguments.revised_lock,
            spec_path=arguments.spec,
            spec_checksum_path=arguments.spec_checksum,
            output_directory=arguments.output,
        )
    elif arguments.command == "authorize":
        result = create_correction_locked_authorization(
            receipt_path=arguments.receipt,
            receipt_sha256=arguments.receipt_sha256,
            score_path=arguments.score,
            score_sha256=arguments.score_sha256,
            revised_lock_path=arguments.revised_lock,
            locked_manifest_path=arguments.locked_manifest,
            locked_execution_directory=arguments.locked_execution_directory,
            authorization_path=arguments.output,
        )
    else:
        result = validate_correction_locked_authorization(
            authorization_path=arguments.authorization,
            receipt_path=arguments.receipt,
            score_path=arguments.score,
            revised_lock_path=arguments.revised_lock,
            locked_manifest_path=arguments.locked_manifest,
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
