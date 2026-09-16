"""Export read-only publication counterparts from a sealed K3 reliability run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lc_pipeline.k3.reliability_publication import ReliabilityPublicationError, render_publication


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    try:
        result = render_publication(args.run_root, args.output)
    except (OSError, ReliabilityPublicationError) as exc:
        p.error(str(exc))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
