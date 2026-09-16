"""Prepare, execute/resume, or inspect the isolated retrospective reliability study."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from lc_pipeline.k3.reliability_study import execute, prepare, start_background, status


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
    args = vars(parser.parse_args())
    command = args.pop("command")
    try:
        if command == "prepare":
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


if __name__ == "__main__":
    main()
