"""Export known periods, with an explicit database-ID/physical-number mapping."""

import argparse
import json
from pathlib import Path

from lc_pipeline.k3.survey_identity import export_mapped_periods


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-map", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = export_mapped_periods(args.identity_map, args.catalog, args.output)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps({key: value for key, value in result.items() if key != "objects"}))


if __name__ == "__main__":
    main()
