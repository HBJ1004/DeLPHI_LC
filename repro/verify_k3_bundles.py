"""Reevaluate all 170 held-out objects through the public ensemble interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lc_pipeline.k3.bundle import K3EnsemblePredictor, sha256
from lc_pipeline.k3.damit import load_damit_object
from lc_pipeline.k3.evaluation import oracle_at_k_error_deg
from lc_pipeline.v2.preprocessing import KnownPeriod


def verify(bundles: Path, archive: Path, catalog: Path, dump: Path, output: Path, device: str) -> dict:
    """Require unchanged mode indices and oracle errors within 0.001 degree."""
    if output.exists():
        raise ValueError("output already exists")
    records = {row["object_id"]: row for row in map(json.loads, catalog.read_text().splitlines())}
    with np.load(archive, allow_pickle=False) as data:
        ids, folds = data["object_ids"].astype(str), data["folds"].astype(int)
        expected_errors = data["oracle_errors_deg"].copy()
        expected_indices = data["deployed_mode_indices"].copy()
        expected_axes = data["refined_axes"].copy()
    if len(ids) != 170 or len(set(ids)) != 170:
        raise ValueError("verification requires the full 170-object OOF artifact")
    rows = []
    for fold in range(5):
        predictor = K3EnsemblePredictor(bundles / f"k3-oof-fold-{fold}", device=device)
        for index in np.flatnonzero(folds == fold):
            oid = ids[index]
            solutions = records[oid]["solutions"]
            source = load_damit_object(dump, oid)
            prediction = predictor.predict(source.epochs, object_id=oid, known_period=KnownPeriod(
                solutions[0]["period_hours"], "frozen-DAMIT-primary-solution-fixed-period"))
            axes = np.asarray([row["axis_xyz"] for row in prediction["axes"]])
            indices = [row["grid_index"] for row in prediction["axes"]]
            targets = np.asarray([row["vector"] for row in solutions], dtype=np.float32)
            error = oracle_at_k_error_deg(axes, targets)
            difference = abs(error - float(expected_errors[index]))
            axis_difference = float(np.max(np.abs(axes - expected_axes[index])))
            passed = (indices == expected_indices[index].tolist()
                      and difference <= .001 and axis_difference <= 1e-4)
            rows.append({"object_id": oid, "fold": fold, "oracle_error_deg": error,
                         "error_difference_deg": difference,
                         "max_axis_component_difference": axis_difference, "passed": passed})
        print(f"verified fold {fold}: {len(rows)}/170 objects", flush=True)
        del predictor
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    report = {"schema": "delphi.k3-bundle-parity.v1", "passed": all(row["passed"] for row in rows),
              "n_objects": len(rows), "source_sha256": sha256(archive),
              "catalog_sha256": sha256(catalog), "device": device, "rows": rows}
    output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise ValueError("export parity failed; inspect report and do not publish")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundles", "ensemble", "catalog", "dump-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    verify(args.bundles, args.ensemble, args.catalog, args.dump_root, args.output, args.device)


if __name__ == "__main__":
    main()
