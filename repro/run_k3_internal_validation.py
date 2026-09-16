"""Prepare/run/status/export the sealed K3 internal-role diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lc_pipeline.k3 import internal_validation as study


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("prepare")
    a.add_argument("--root", type=Path, required=True)
    a.add_argument("--original-study", type=Path, required=True)
    for name in ("run", "status", "export"):
        a = sub.add_parser(name)
        a.add_argument("--root", type=Path, required=True)
        if name == "run":
            a.add_argument("--parity-only", action="store_true")
    x = vars(p.parse_args())
    command = x.pop("command")
    try:
        result = getattr(study, command)(**x)
    except (OSError, ValueError, TimeoutError) as e:
        p.error(str(e))
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
