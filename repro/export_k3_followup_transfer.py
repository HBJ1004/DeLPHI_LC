"""Export sealed follow-up transfer records without rerunning any computation."""

from __future__ import annotations

import argparse
from pathlib import Path

from lc_pipeline.k3.followup_transfer_reporting import export_followup_transfer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, default=Path(__file__).resolve().parents[2] / "data"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data/followup-transfer-publication-20260916",
    )
    parser.add_argument(
        "--comparator",
        type=Path,
        required=True,
        help="Released comparator NPZ containing object_ids and atlas_errors_deg.",
    )
    args = parser.parse_args()
    print(export_followup_transfer(args.data_root, args.output, args.comparator)["schema"])


if __name__ == "__main__":
    main()
