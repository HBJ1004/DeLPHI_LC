"""Create the frozen development and locked-evaluation object manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def prepare(spec_path: Path, splits_path: Path, output_directory: Path) -> None:
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    split = json.loads(splits_path.read_text(encoding="utf-8"))
    split_spec = spec["convergence_acceleration"]["split"]
    salt = str(split_spec["salt"])
    development: list[str] = []
    evaluation: list[str] = []
    for fold in sorted(split["folds"], key=lambda row: int(row["fold"])):
        ranked = sorted(
            (str(value) for value in fold["test_ids"]),
            key=lambda value: hashlib.sha256(f"{salt}:{value}".encode("ascii")).hexdigest(),
        )
        development.extend(ranked[:6])
        evaluation.extend(ranked[6:])
    if len(development) != 30 or len(evaluation) != 140:
        raise ValueError("follow-up split must contain 30 development and 140 evaluation objects")
    if set(development) & set(evaluation) or len(set(development + evaluation)) != 170:
        raise ValueError("follow-up object roles must be disjoint and cover all 170 objects")
    common = {
        "schema": "delphi.k3-convergence-object-subset.v1",
        "study_spec_sha256": _sha256(spec_path),
        "source_full_split_sha256": _sha256(splits_path),
        "selection": str(split_spec["method"]),
        "salt": salt,
    }
    _write(
        output_directory / "development-objects.json",
        {**common, "role": "development", "object_ids": development},
    )
    _write(
        output_directory / "locked-evaluation-objects.json",
        {**common, "role": "locked_evaluation", "object_ids": evaluation},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.spec, args.splits, args.output_directory)


if __name__ == "__main__":
    main()
