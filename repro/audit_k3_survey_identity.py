"""Audit the physical identity join independently of photometry and pole scoring."""

import argparse
import json
from pathlib import Path

from lc_pipeline.k3.survey_identity import write_identity_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asteroid-table", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = write_identity_audit(args.asteroid_table, args.splits, args.prepared_manifest, args.output_directory)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps({key: value for key, value in result.items() if key != "objects"}, indent=2))


if __name__ == "__main__":
    main()
