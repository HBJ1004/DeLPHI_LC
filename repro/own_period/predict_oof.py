"""Out-of-fold DeLPHI candidates for the 170 asteroids with a chosen period.

    python -m repro.own_period.predict_oof --periods damit --output X.npz
    python -m repro.own_period.predict_oof --periods own-periods-170.json --output Y.npz

The prediction follows ``publication_run.evaluate_real_oof_ensemble`` step by
step (same examples, networks, score maps, modes and adjustment), except that
the period of every asteroid is read from ``--periods``.  With ``damit`` it
uses the published period (first catalogue solution) and must reproduce the
published axes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from lc_pipeline.k3.damit import load_damit_object
from lc_pipeline.k3.evaluation import (
    ensemble_score_grids,
    modes_from_score_grid,
    oracle_at_k_error_deg,
)
from lc_pipeline.k3.inference import refine_ensemble_axes, score_axial_grid
from lc_pipeline.k3.publication_run import K3_OOF_SEEDS
from lc_pipeline.k3.tokenizer import tokenize_epochs
from lc_pipeline.k3.training import K3TrainingExample, collate_examples
from lc_pipeline.v2.preprocessing import KnownPeriod
from repro.network_root import NetworkPath, guard_output, network_metadata
from repro.network_root import load_models as _load_models
from repro.own_period import DataPath

ROOT = DataPath("../../..")
FOLLOWUP = DataPath("../..")
DATA = FOLLOWUP / "inputs/training-run/repro/data/damit-20250610T000301Z"
DUMP = ROOT / "damit-20250610T000301Z"
MODELS = NetworkPath("models")
MODEL_MANIFEST = NetworkPath("binding")
INPUT_KEYS = {"phase_features", "phase_mask", "geometry_features", "epoch_features", "epoch_mask"}


def load_periods(spec: str, catalog: dict[str, dict]) -> tuple[dict[str, float | None], str, str]:
    if spec == "damit":
        periods = {
            oid: float(row["solutions"][0]["period_hours"])
            for oid, row in catalog.items()
            if row.get("solutions")
        }
        return periods, "frozen-DAMIT-primary-solution-fixed-period", "damit-catalog"
    path = Path(spec)
    value = json.loads(path.read_text())
    return (
        value["periods_hours"],
        value["provenance"],
        hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--periods", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--objects", type=lambda value: value.split(","), help="CPU smoke subset")
    args = parser.parse_args()
    guard_output(args.output)
    if args.output.exists():
        raise FileExistsError(args.output)
    catalog = {
        row["object_id"]: row
        for row in map(json.loads, (DATA / "catalog.jsonl").read_text().splitlines())
    }
    folds = sorted(
        json.loads((DATA / "publication-splits-v2.3.json").read_text())["folds"],
        key=lambda r: int(r["fold"]),
    )
    periods, provenance, period_source = load_periods(args.periods, catalog)
    ids, fold_of, used, axes_out, starts_out, scores_out, errors, seconds = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for row in folds:
        fold = int(row["fold"])
        selected = [oid for oid in row["test_ids"] if not args.objects or oid in args.objects]
        if not selected:
            continue
        models, _, _ = _load_models(MODELS, MODEL_MANIFEST, (fold,), args.device)
        for oid in selected:
            entry = catalog[oid]
            targets = np.asarray([s["vector"] for s in entry["solutions"]], dtype=np.float32)
            period = periods.get(oid)
            ids.append(oid)
            fold_of.append(fold)
            used.append(np.nan if period is None else float(period))
            if period is None:
                axes_out.append(np.full((3, 3), np.nan))
                starts_out.append(np.full((3, 3), np.nan))
                scores_out.append(np.full(3, np.nan))
                errors.append(np.nan)
                seconds.append(np.nan)
                continue
            started = time.perf_counter()
            source = load_damit_object(DUMP, oid)
            example = K3TrainingExample(
                object_id=oid,
                tokenized=tokenize_epochs(
                    source.epochs, known_period=KnownPeriod(float(period), provenance)
                ),
                target_axes=targets,
            )
            inputs = {
                k: v.to(args.device)
                for k, v in collate_examples((example,)).items()
                if k in INPUT_KEYS
            }
            grids = np.stack(
                [
                    np.asarray(score_axial_grid(m, inputs, chunk_size=1024), dtype=np.float64)
                    for m in models
                ]
            )
            modes = modes_from_score_grid(ensemble_score_grids(grids[:, None, :])[0])
            starts = np.asarray([mode.axis_xyz for mode in modes], dtype=np.float64)
            axes, scores = refine_ensemble_axes(models, inputs, starts)
            seconds.append(time.perf_counter() - started)
            axes_out.append(axes)
            starts_out.append(starts)
            scores_out.append(scores)
            errors.append(oracle_at_k_error_deg(axes, targets, k=3))
            print(oid, fold, f"{period:.5f}", f"{errors[-1]:.2f}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        np.savez_compressed(
            handle,
            schema=np.asarray("delphi.own-period-oof.v1"),
            seeds=np.asarray(K3_OOF_SEEDS, dtype=np.int64),
            object_ids=np.asarray(ids),
            folds=np.asarray(fold_of, dtype=np.int8),
            period_hours=np.asarray(used),
            period_provenance=np.asarray(provenance),
            period_source=np.asarray(period_source),
            initial_axes=np.asarray(starts_out),
            refined_axes=np.asarray(axes_out),
            refined_scores=np.asarray(scores_out),
            errors_deg=np.asarray(errors),
            inference_seconds=np.asarray(seconds),
            **network_metadata(),
        )
    valid = np.asarray(errors)[np.isfinite(errors)]
    print(
        json.dumps(
            {
                "output": str(args.output),
                "n": len(ids),
                "n_valid": int(valid.size),
                "mean_deg": float(valid.mean()),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
