"""Export the label-blind, frozen inputs for the K3 convergence study.

The resulting JSON deliberately contains no reference pole/vector or derived
accuracy column.  It is the only catalog-like artifact the solver-execution
phase is allowed to open.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

SCHEMA = "delphi.k3-convergence-blind-inputs.v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(catalog_path: Path, splits_path: Path, ensemble_path: Path, output: Path) -> None:
    split = json.loads(splits_path.read_text(encoding="utf-8"))
    folds = sorted(split["folds"], key=lambda row: int(row["fold"]))
    if [int(row["fold"]) for row in folds] != list(range(5)):
        raise ValueError("the frozen split must contain folds zero through four")
    expected = [(str(object_id), int(row["fold"])) for row in folds for object_id in row["test_ids"]]
    if len(expected) != 170 or len({object_id for object_id, _ in expected}) != 170:
        raise ValueError("the convergence cohort must contain 170 unique outer-test objects")

    catalog = {
        row["object_id"]: row
        for row in (
            json.loads(line)
            for line in catalog_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    with np.load(ensemble_path, allow_pickle=False) as artifact:
        schema = str(artifact["schema"].item())
        object_ids = tuple(artifact["object_ids"].astype(str))
        axes = np.asarray(artifact["refined_axes"], dtype=np.float64)
    if (
        schema != "delphi.k3-real-oof-ensemble.v1"
        or object_ids != tuple(object_id for object_id, _ in expected)
        or axes.shape != (170, 3, 3)
        or not np.all(np.isfinite(axes))
    ):
        raise ValueError("the source ensemble is not the aligned 170-object OOF ensemble")

    objects = []
    for index, (object_id, fold) in enumerate(expected):
        row = catalog[object_id]
        # Intentionally access only the permitted fixed period and lightcurve
        # provenance.  Reference directions are never copied into this file.
        period = float(row["solutions"][0]["period_hours"])
        lightcurve = row["lightcurve"]
        if not np.isfinite(period) or period <= 0:
            raise ValueError(f"invalid fixed period for {object_id}")
        objects.append(
            {
                "fold": fold,
                "guided_axes": axes[index].tolist(),
                "lightcurve": {
                    "source_path": str(lightcurve["source_path"]),
                    "source_sha256": str(lightcurve["source_sha256"]),
                },
                "object_id": object_id,
                "period_hours": period,
            }
        )
    payload = {
        "schema": SCHEMA,
        "source_catalog_sha256": _sha256(catalog_path),
        "source_ensemble_sha256": _sha256(ensemble_path),
        "source_full_split_sha256": _sha256(splits_path),
        "objects": objects,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--ensemble", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    prepare(arguments.catalog, arguments.splits, arguments.ensemble, arguments.output)


if __name__ == "__main__":
    main()
