"""Tune, freeze, execute, and score the adaptive DeLPHI workflow benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    tuning = commands.add_parser("tune")
    tuning.add_argument("--parent-lock", type=Path, required=True)
    tuning.add_argument("--archive", type=Path, required=True)
    tuning.add_argument("--output", type=Path, required=True)

    freezing = commands.add_parser("freeze")
    freezing.add_argument("--parent-lock", type=Path, required=True)
    freezing.add_argument("--tuning-report", type=Path, required=True)
    freezing.add_argument("--output", type=Path, required=True)
    freezing.add_argument("--scoring-output", type=Path, required=True)

    execution = commands.add_parser("execute")
    execution.add_argument("--lock", type=Path, required=True)
    execution.add_argument("--output", type=Path, required=True)

    scoring = commands.add_parser("score")
    scoring.add_argument("--lock", type=Path, required=True)
    scoring.add_argument("--execution", type=Path, required=True)
    scoring.add_argument("--scoring-lock", type=Path, required=True)
    scoring.add_argument("--output", type=Path, required=True)

    verification = commands.add_parser("verify")
    verification.add_argument("--lock", type=Path, required=True)
    verification.add_argument("--execution", type=Path, required=True)

    worker = commands.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("--lock", type=Path, required=True)
    worker.add_argument("--job", type=Path, required=True)

    args = parser.parse_args()
    from lc_pipeline.workflow_benchmark import (
        WorkflowBenchmarkError,
        execute,
        freeze,
        score,
        tune,
        verify_execution,
        worker_main,
    )

    try:
        if args.command == "tune":
            result = tune(parent_lock=args.parent_lock, archive=args.archive, output=args.output)
            print(json.dumps({"status": "tuned", "rule": result["selected_rule"]}))
        elif args.command == "freeze":
            lock, scoring_lock = freeze(
                parent_lock=args.parent_lock,
                tuning_report=args.tuning_report,
                output=args.output,
                scoring_output=args.scoring_output,
            )
            print(
                json.dumps(
                    {
                        "status": "frozen",
                        "evaluation_objects": len(lock["objects"]),
                        "scoring_phase": scoring_lock["phase"],
                    }
                )
            )
        elif args.command == "execute":
            result = execute(lock_path=args.lock, output=args.output)
            print(json.dumps({"status": "executed", "cases": len(result["rows"])}))
        elif args.command == "score":
            result = score(
                lock_path=args.lock,
                execution_path=args.execution,
                scoring_lock=args.scoring_lock,
                output=args.output,
            )
            print(
                json.dumps(
                    {
                        "status": "scored",
                        "established": result[
                            "acceleration_at_comparable_accuracy_established"
                        ],
                        "point_estimates": result["point_estimates"],
                    }
                )
            )
        elif args.command == "verify":
            result = verify_execution(lock_path=args.lock, execution_path=args.execution)
            print(json.dumps(result))
        else:
            worker_main(args.lock, args.job)
    except (WorkflowBenchmarkError, OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
