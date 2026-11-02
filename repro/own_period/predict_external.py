"""DeLPHI candidates for the ZTF and ALCDEF comparisons with a chosen period.

    python -m repro.own_period.predict_external --cohort ztf --periods published --output X.jsonl
    python -m repro.own_period.predict_external --cohort ztf --periods own-periods-ztf.json --output Y.jsonl

ZTF follows ``ztf_prediction.predict_prepared_ztf_object`` (five networks of
the run in which the asteroid is a test asteroid) and ALCDEF follows
``predict_k3_alcdef_transfer.predict`` (all 25 networks).  Only the period
differs.  With ``published`` the period of the prepared object is used and
the published axes must be reproduced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from lc_pipeline.k3.bundle import _model_inputs
from lc_pipeline.k3.evaluation import ensemble_score_grids, modes_from_score_grid
from lc_pipeline.k3.inference import refine_ensemble_axes, score_axial_grid
from lc_pipeline.k3.ztf_prediction import _fold_policy, _load_prepared
from lc_pipeline.v2.preprocessing import KnownPeriod
from repro.network_root import NetworkPath, guard_output, json_metadata
from repro.network_root import load_historical_models as _load_models

from .search_external import FOLLOWUP, PREPARED, ready_objects

MODELS = NetworkPath("models")
MODEL_MANIFEST = NetworkPath("binding")
SPLITS = (
    FOLLOWUP / "inputs/training-run/repro/data/damit-20250610T000301Z/publication-splits-v2.3.json"
)


def predict(models, epochs, period, device, cohort):
    inputs = _model_inputs(epochs, period, torch.device(device))
    if cohort == "ztf":
        grids = np.stack([score_axial_grid(m, inputs, chunk_size=1024) for m in models])
        modes = modes_from_score_grid(ensemble_score_grids(grids[:, None, :])[0])
        return refine_ensemble_axes(models, inputs, np.asarray([mode.axis_xyz for mode in modes]))
    with torch.inference_mode():
        grids = np.stack([score_axial_grid(m, inputs, chunk_size=1024) for m in models])
        modes = modes_from_score_grid(ensemble_score_grids(grids[:, None, :])[0])
    with torch.enable_grad():
        return refine_ensemble_axes(models, inputs, np.asarray([mode.axis_xyz for mode in modes]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=sorted(PREPARED), required=True)
    parser.add_argument("--periods", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--objects", help="comma-separated identities for a CPU switch check")
    args = parser.parse_args()
    guard_output(args.output)
    own = None if args.periods == "published" else json.loads(Path(args.periods).read_text())
    if own is not None and own.get("cohort") != args.cohort:
        raise ValueError("period file belongs to another cohort")
    cache = {}
    items = [
        (
            _fold_policy(oid, SPLITS, "existing_identity")
            if args.cohort == "ztf"
            else tuple(range(5)),
            oid,
            path,
            digest,
        )
        for oid, path, digest in ready_objects(args.cohort)
        if args.objects is None or oid in args.objects.split(",")
    ]
    with args.output.open("x", buffering=1) as stream:
        for folds, object_id, path, digest in sorted(items):
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f"{object_id}: prepared-object hash mismatch")
            _, published, epochs = _load_prepared(path)
            if own is None:
                period = published
            elif own["periods_hours"].get(object_id) is None:
                stream.write(json.dumps({"object_id": object_id, "status": "no_period", **json_metadata()}) + "\n")
                continue
            else:
                period = KnownPeriod(float(own["periods_hours"][object_id]), own["provenance"])
            if folds not in cache:
                cache.clear()
                cache[folds] = _load_models(MODELS, MODEL_MANIFEST, folds, args.device)[0]
            axes, scores = predict(cache[folds], epochs, period, args.device, args.cohort)
            stream.write(
                json.dumps(
                    {
                        "object_id": object_id,
                        "status": "ok",
                        "period_hours": period.hours,
                        "period_provenance": period.provenance,
                        "folds": list(folds),
                        "axes": np.asarray(axes).tolist(),
                        "scores": np.asarray(scores).tolist(),
                        **json_metadata(),
                    }
                )
                + "\n"
            )
            print(object_id, f"{period.hours:.5f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
