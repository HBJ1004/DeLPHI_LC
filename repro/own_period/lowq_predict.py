"""DeLPHI candidates for the lower-quality DAMIT census with a chosen period.

    python -m repro.own_period.lowq_predict --periods published --limit 500 --output V.jsonl
    python -m repro.own_period.lowq_predict --periods OP/lowq/own-periods-lowq.json --output Y.jsonl

This follows ``repro/run_k3_lowq_census.py predict`` line by line (the fixed
fold-0 five-seed ensemble of the published census, raw ``lc.txt`` parsed with
``parse_damit_lightcurve``, score maps averaged, three modes refined).  Only
the period differs.  With ``published`` the DAMIT model-table period of the
census index is used, and the published axes must be reproduced.  With a
period file, an asteroid without a period gets ``status: no_period``.

The run is resumable (rows are appended, finished asteroids are skipped) and
``--max-seconds`` bounds one chunk.  ``--finalize`` checks completeness and
writes a manifest once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from lc_pipeline.k3.bundle import _model_inputs
from lc_pipeline.k3.evaluation import ensemble_score_grids, modes_from_score_grid
from lc_pipeline.k3.inference import refine_ensemble_axes, score_axial_grid
from lc_pipeline.k3.ztf_prediction import _load_models
from lc_pipeline.v2.data import parse_damit_lightcurve
from lc_pipeline.v2.preprocessing import KnownPeriod

from .lowq_search import FOLLOWUP, INDEX, INDEX_SHA256, SNAPSHOT, sha256

MODELS = FOLLOWUP / "inputs/frozen-artifacts/k3-definitive-7874092/models"
MODEL_MANIFEST = FOLLOWUP / "data/external-models-final.json"
MODEL_MANIFEST_SHA256 = "6d041bcbedd9e7ecacc9c4768d0eeb81827065f016dd4db9e982373db3ebfefa"
FOLD = 0  # the published census policy: fixed fold-0 five-seed ensemble


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def predict_one(models, row: dict, period: KnownPeriod, target: torch.device) -> dict:
    """The body of the published census loop, unchanged except for the period."""
    object_id = str(row["object_id"])
    source = (SNAPSHOT / str(row["lightcurve_path"])).resolve()
    source.relative_to(SNAPSHOT.resolve())
    before = time.perf_counter()
    try:
        epochs = parse_damit_lightcurve(source)
        inputs = _model_inputs(epochs, period, target)
        _sync(target)
        with torch.inference_mode():
            grids = np.stack([score_axial_grid(model, inputs, chunk_size=1024) for model in models])
            mean = ensemble_score_grids(grids[:, None, :])[0]
            modes = modes_from_score_grid(mean)
        with torch.enable_grad():
            axes, scores = refine_ensemble_axes(
                models, inputs, np.asarray([mode.axis_xyz for mode in modes])
            )
        _sync(target)
        return {
            "object_id": object_id,
            "status": "ok",
            "quality_flag": int(row["quality_flag"]),
            "lightcurve_sha256": sha256(source),
            "epoch_count": len(epochs),
            "observation_count": sum(len(epoch.observations) for epoch in epochs),
            "wall_seconds": time.perf_counter() - before,
            "period_hours": period.hours,
            "period_provenance": period.provenance,
            "axes": [
                {"axis_xyz": axis.tolist(), "score": float(score), "grid_index": int(m.grid_index)}
                for axis, score, m in zip(axes, scores, modes, strict=True)
            ],
        }
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        return {
            "object_id": object_id,
            "status": "input_rejected",
            "quality_flag": int(row["quality_flag"]),
            "wall_seconds": time.perf_counter() - before,
            "period_hours": period.hours,
            "reason": f"{type(exc).__name__}:{exc}",
        }


def done_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    with path.open() as handle:
        return {json.loads(line)["object_id"] for line in handle if line.strip()}


def selected_rows(limit: int | None) -> list[dict]:
    if sha256(INDEX) != INDEX_SHA256:
        raise ValueError("census input index differs from the published one")
    rows = json.loads(INDEX.read_text())["objects"]
    return rows[:limit] if limit is not None else rows


def run(args) -> int:
    rows = selected_rows(args.limit)
    own = None
    if args.periods != "published":
        own = json.loads(Path(args.periods).read_text())
        if own.get("cohort") != "lowq" or own.get("census_index_sha256") != INDEX_SHA256:
            raise ValueError("period file belongs to another cohort")
    if sha256(MODEL_MANIFEST) != MODEL_MANIFEST_SHA256:
        raise ValueError("model manifest differs from the published census")
    target = torch.device(args.device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    finished = done_ids(args.output)
    todo = [r for r in rows if r["object_id"] not in finished]
    print(f"{len(finished)} finished, {len(todo)} to do", flush=True)
    if not todo:
        return 0
    models, _names, _bindings = _load_models(MODELS, MODEL_MANIFEST, (FOLD,), str(target))
    if len(models) != 5:
        raise ValueError("census policy requires five seeds from one fixed fold")
    started, count = time.perf_counter(), 0
    with args.output.open("a", buffering=1) as stream:
        for row in todo:
            if time.perf_counter() - started > args.max_seconds:
                break
            object_id = str(row["object_id"])
            if own is None:
                period = KnownPeriod(float(row["known_period_hours"]), str(row["period_provenance"]))
            elif own["periods_hours"].get(object_id) is None:
                status = own["details"][object_id]["status"]
                item = {"object_id": object_id, "status": "no_period",
                        "quality_flag": int(row["quality_flag"]), "search_status": status,
                        "reason": f"no_period:{status}", "wall_seconds": 0.0}
                stream.write(json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n")
                count += 1
                continue
            else:
                period = KnownPeriod(float(own["periods_hours"][object_id]), own["provenance"])
            item = predict_one(models, row, period, target)
            stream.write(json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n")
            count += 1
            if count % 100 == 0:
                print(f"{count} this chunk, {time.perf_counter() - started:.0f} s", flush=True)
    remaining = len(todo) - count
    print(f"chunk finished: {count} new, {remaining} remaining", flush=True)
    return 3 if remaining else 0


def finalize(args) -> None:
    rows = selected_rows(args.limit)
    items = {}
    with args.output.open() as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                if item["object_id"] in items:
                    raise ValueError(f"duplicate row {item['object_id']}")
                items[item["object_id"]] = item
    missing = [r["object_id"] for r in rows if r["object_id"] not in items]
    if missing:
        raise ValueError(f"{len(missing)} asteroids missing")
    counts: dict[str, int] = {}
    for item in items.values():
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    manifest = {
        "schema": "delphi.own-period-lowq-predictions.v1",
        "inference_policy": "fixed_fold_five_seed_ensemble (as published census)",
        "fold": FOLD,
        "periods": args.periods if args.periods == "published" else str(Path(args.periods).name),
        "periods_sha256": None if args.periods == "published" else sha256(Path(args.periods)),
        "census_index_sha256": INDEX_SHA256,
        "model_manifest_sha256": MODEL_MANIFEST_SHA256,
        "device": args.device,
        "limit": args.limit,
        "rows": len(items),
        "status_counts": dict(sorted(counts.items())),
        "predictions_sha256": sha256(args.output),
    }
    with args.finalize.open("x") as stream:
        stream.write(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--periods", required=True, help="'published' or an own-period file")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-seconds", type=float, default=3300.0)
    parser.add_argument("--finalize", type=Path, default=None, help="write manifest once")
    args = parser.parse_args()
    if args.finalize is not None:
        finalize(args)
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
