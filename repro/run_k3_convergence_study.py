"""Run the frozen, phase-separated K3 convergence follow-up.

This command keeps solver execution label blind.  Reference axes enter only
through the explicit ``score-*`` commands, after an execution artifact has
been written and hash bound.  In particular, the locked cohort cannot be run
until a valid development-selection artifact and all five development score
artifacts are supplied.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SELECTION_SCHEMA = "delphi.k3-convergence-development-selection.v1"


def _add_spec_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--spec-checksum", type=Path, required=True)


def _add_frozen_resource_arguments(parser: argparse.ArgumentParser) -> None:
    _add_spec_arguments(parser)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--locked-manifest", type=Path, required=True)
    parser.add_argument("--blind-inputs", type=Path, required=True)
    parser.add_argument("--ensemble", type=Path, required=True)
    parser.add_argument("--neural-timing", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)


def _resource_kwargs(arguments: argparse.Namespace) -> dict[str, Path]:
    return {
        "spec_path": arguments.spec,
        "spec_checksum_path": arguments.spec_checksum,
        "split_path": arguments.splits,
        "development_manifest_path": arguments.development_manifest,
        "locked_manifest_path": arguments.locked_manifest,
        "blind_inputs_path": arguments.blind_inputs,
        "ensemble_path": arguments.ensemble,
        "neural_timing_path": arguments.neural_timing,
        "source_archive": arguments.source_archive,
        "source_root": arguments.source_root,
        "executable": arguments.executable,
    }


def _require_five(paths: list[Path], description: str) -> tuple[Path, ...]:
    identities = {path.resolve() for path in paths}
    if len(paths) != 5 or len(identities) != 5:
        raise ValueError(f"{description} requires five distinct artifacts")
    return tuple(paths)


def _selected_execution(path: Path) -> tuple[float, Path]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        if (
            document.get("schema") != SELECTION_SCHEMA
            or document.get("status") != "selected"
        ):
            raise ValueError("development selection status is not selected")
        tolerance = float(document["selected_tolerance"])
        output_directory = Path(document["locked_execution_directory"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError("cannot read a selected execution plan") from exc
    if tolerance not in (0.01, 0.003, 0.001, 0.0003, 0.0001):
        raise ValueError("development selection contains a non-protocol tolerance")
    return tolerance, output_directory


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    lock = commands.add_parser(
        "lock", help="freeze and verify all execution identities before any solver run"
    )
    _add_frozen_resource_arguments(lock)
    lock.add_argument("--output", type=Path, required=True)

    development = commands.add_parser(
        "execute-development-grid",
        help="run all five development tolerances without opening reference labels",
    )
    _add_frozen_resource_arguments(development)
    development.add_argument("--lock", type=Path, required=True)
    development.add_argument("--dump-root", type=Path, required=True)
    development.add_argument("--output-root", type=Path, required=True)

    score_development = commands.add_parser(
        "score-development",
        help="score one already-frozen development execution artifact",
    )
    _add_spec_arguments(score_development)
    score_development.add_argument("--splits", type=Path, required=True)
    score_development.add_argument("--development-manifest", type=Path, required=True)
    score_development.add_argument("--execution", type=Path, required=True)
    score_development.add_argument("--reference-catalog", type=Path, required=True)
    score_development.add_argument("--lock", type=Path, required=True)
    score_development.add_argument("--output", type=Path, required=True)

    select = commands.add_parser(
        "select-development",
        help="apply the frozen rule to exactly five development score artifacts",
    )
    _add_spec_arguments(select)
    select.add_argument("--lock", type=Path, required=True)
    select.add_argument(
        "--score",
        dest="scores",
        type=Path,
        action="append",
        required=True,
        help="development score artifact; provide exactly five times",
    )
    select.add_argument("--output", type=Path, required=True)

    locked = commands.add_parser(
        "execute-locked",
        help="run the selected tolerance once on the locked cohort, still label blind",
    )
    _add_frozen_resource_arguments(locked)
    locked.add_argument("--lock", type=Path, required=True)
    locked.add_argument("--dump-root", type=Path, required=True)
    locked.add_argument("--development-selection", type=Path, required=True)
    locked.add_argument(
        "--development-score",
        dest="development_scores",
        type=Path,
        action="append",
        required=True,
        help="development score artifact; provide exactly five times",
    )

    score_locked = commands.add_parser(
        "score-locked",
        help="score the one frozen locked-cohort execution and apply final gates",
    )
    _add_spec_arguments(score_locked)
    score_locked.add_argument("--splits", type=Path, required=True)
    score_locked.add_argument("--locked-manifest", type=Path, required=True)
    score_locked.add_argument("--execution", type=Path, required=True)
    score_locked.add_argument("--reference-catalog", type=Path, required=True)
    score_locked.add_argument("--lock", type=Path, required=True)
    score_locked.add_argument("--development-selection", type=Path, required=True)
    score_locked.add_argument(
        "--development-score",
        dest="development_scores",
        type=Path,
        action="append",
        required=True,
        help="development score artifact; provide exactly five times",
    )
    score_locked.add_argument("--output", type=Path, required=True)
    return parser


def _run(arguments: argparse.Namespace) -> dict[str, Any]:
    from lc_pipeline.k3.convergence_study import (
        create_development_selection,
        create_study_lock,
        execute_blind_cohort,
        execute_development_grid,
        score_blind_execution,
    )

    if arguments.command == "lock":
        return create_study_lock(
            **_resource_kwargs(arguments), output_path=arguments.output
        )

    if arguments.command == "execute-development-grid":
        return execute_development_grid(
            **_resource_kwargs(arguments),
            lock_path=arguments.lock,
            dump_root=arguments.dump_root,
            output_root=arguments.output_root,
        )

    if arguments.command == "score-development":
        return score_blind_execution(
            execution_path=arguments.execution,
            reference_catalog_path=arguments.reference_catalog,
            lock_path=arguments.lock,
            spec_path=arguments.spec,
            spec_checksum_path=arguments.spec_checksum,
            split_path=arguments.splits,
            cohort_manifest_path=arguments.development_manifest,
            output_path=arguments.output,
            expected_role="development",
        )

    if arguments.command == "select-development":
        scores = _require_five(arguments.scores, "development selection")
        return create_development_selection(
            score_paths=scores,
            lock_path=arguments.lock,
            spec_path=arguments.spec,
            spec_checksum_path=arguments.spec_checksum,
            output_path=arguments.output,
        )

    scores = _require_five(arguments.development_scores, "locked evaluation")
    if arguments.command == "execute-locked":
        tolerance, output_directory = _selected_execution(
            arguments.development_selection
        )
        return execute_blind_cohort(
            **_resource_kwargs(arguments),
            role="locked_evaluation",
            lock_path=arguments.lock,
            dump_root=arguments.dump_root,
            output_directory=output_directory,
            convergence_tolerance=tolerance,
            development_selection_path=arguments.development_selection,
            development_score_paths=scores,
        )

    if arguments.command == "score-locked":
        return score_blind_execution(
            execution_path=arguments.execution,
            reference_catalog_path=arguments.reference_catalog,
            lock_path=arguments.lock,
            spec_path=arguments.spec,
            spec_checksum_path=arguments.spec_checksum,
            split_path=arguments.splits,
            cohort_manifest_path=arguments.locked_manifest,
            output_path=arguments.output,
            expected_role="locked_evaluation",
            development_selection_path=arguments.development_selection,
            development_score_paths=scores,
        )
    raise AssertionError(f"unhandled command: {arguments.command}")


def main() -> None:
    arguments = _build_parser().parse_args()
    print(json.dumps(_run(arguments), sort_keys=True))


if __name__ == "__main__":
    main()
