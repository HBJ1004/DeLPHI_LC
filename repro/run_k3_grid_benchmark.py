"""Freeze, execute and score the separate known-period pole-grid experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    lock = commands.add_parser("freeze")
    for flag in ("blind-inputs", "splits", "development", "dump-root", "bundle-root",
                 "capacity-report", "reference-catalog", "output"):
        lock.add_argument(f"--{flag}", type=Path, required=True)
    lock.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    execution = commands.add_parser("execute")
    execution.add_argument("--lock", type=Path, required=True)
    execution.add_argument("--output", type=Path, required=True)
    execution.add_argument("--role", choices=("pilot", "full"), required=True)
    execution.add_argument("--pilot", type=Path)
    scoring = commands.add_parser("score")
    scoring.add_argument("--lock", type=Path, required=True)
    scoring.add_argument("--execution", type=Path, required=True)
    scoring.add_argument("--output", type=Path, required=True)
    continuation = commands.add_parser("continue")
    continuation.add_argument("--lock", type=Path, required=True)
    continuation.add_argument("--pilot", type=Path, required=True)
    continuation.add_argument("--root", type=Path, required=True)
    worker = commands.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("--lock", type=Path, required=True)
    group = worker.add_mutually_exclusive_group(required=True)
    group.add_argument("--job", type=Path)
    group.add_argument("--fold", type=int)
    args = parser.parse_args()
    from lc_pipeline.grid_benchmark import (
        GridBenchmarkError,
        continue_after_pilot,
        execute,
        freeze,
        score,
        worker_main,
    )
    try:
        if args.command == "freeze":
            result = freeze(**{k: v for k, v in vars(args).items() if k != "command"})
            print(json.dumps({"status": "frozen", "objects": len(result["objects"]),
                              "pilot_ids": result["pilot_ids"], "output": str(args.output)}))
        elif args.command == "execute":
            result = execute(lock_path=args.lock, output=args.output, role=args.role, pilot_path=args.pilot)
            print(json.dumps({"status": "executed", "role": result["role"],
                              "cases": len(result["rows"]), "budget": result.get("budget")}))
        elif args.command == "score":
            result = score(lock_path=args.lock, execution_path=args.execution, output=args.output)
            print(json.dumps({"status": "scored", "role": result["role"], "output": str(args.output)}))
        elif args.command == "continue":
            result = continue_after_pilot(
                lock_path=args.lock, pilot_path=args.pilot, root=args.root
            )
            print(json.dumps({"status": result["phase"], "root": str(args.root)}))
        else:
            worker_main(args.lock, job_path=args.job, fold=args.fold)
    except (GridBenchmarkError, OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
