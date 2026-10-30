"""Prepare, execute/resume, or inspect the isolated retrospective reliability study."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from lc_pipeline.k3.reliability_study import (
    execute, prepare, start_background, status, prepare_period_rerun, run_period_rerun, score_period_rerun,
)
from repro import network_root as network


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("prepare")
    build.add_argument("--root", type=Path, required=True)
    build.add_argument("--previous-lock", type=Path, required=True)
    build.add_argument("--exposure-audit", type=Path, required=True)
    build.add_argument("--solver-report", type=Path, required=True)
    build.add_argument("--protocol", type=Path, default=Path("repro/k3_reliability_protocol.yaml"))
    build.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    for name in ("run", "status", "start"):
        command = commands.add_parser(name)
        command.add_argument("--root", type=Path, required=True)
        if name == "run":
            command.add_argument("--pilot-only", action="store_true")
    local = commands.add_parser('prepare-period-rerun')
    local.add_argument('--root', type=Path, required=True)
    local.add_argument('--published-root', type=Path, required=True)
    local = commands.add_parser('run-period-rerun')
    local.add_argument('--root', type=Path, required=True)
    local.add_argument('--device', default='cuda')
    local.add_argument('--max-seconds', type=float, default=3300)
    local = commands.add_parser('score-period-rerun')
    local.add_argument('--root', type=Path, required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    try:
        if network.switched() and command in ('prepare', 'run', 'start'):
            raise ValueError('use the period-rerun commands with the fixed published schedule')
        if command == 'prepare-period-rerun':
            result = prepare_period_rerun(**args)
        elif command == 'run-period-rerun':
            result = run_period_rerun(**args)
        elif command == 'score-period-rerun':
            result = score_period_rerun(**args)
        elif command == "prepare":
            result = prepare(**args)
        elif command == "run":
            result = execute(**args)
        elif command == "start":
            result = start_background(args["root"])
        else:
            result = status(args["root"])
    except (OSError, ValueError, TimeoutError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True), flush=True)
    if result.get('remaining', 0):
        raise SystemExit(3)


if __name__ == "__main__":
    main()
