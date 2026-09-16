"""Score all eight sealed variants on the original, already-exposed 170 objects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lc_pipeline.k3.generalization_diagnostic_scoring import score_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("diagnostic-manifest", "prediction-receipt", "identity-map", "splits", "model-manifest", "reference-catalog", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = score_diagnostics(**vars(args))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps({
        "role": report["role"], "n_intended_objects": report["n_intended_objects"],
        "n_available_inputs": report["n_available_inputs"], "output": str(args.output),
        "variants": {name: row["all_intended_objects"] for name, row in report["variants"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()
