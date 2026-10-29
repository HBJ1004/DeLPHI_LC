"""Full out-of-fold DeLPHI run with a chosen period: score maps and per-seed results.

    python -m repro.own_period.predict_oof_full --periods damit --output-dir D predict
    python -m repro.own_period.predict_oof_full --periods own-periods-170.json --output-dir D predict
    python -m repro.own_period.predict_oof_full --output-dir D score

``predict`` follows ``publication_run._evaluate_examples`` (one network at a
time, batches of four, as for the published ``real-fold-<f>-seed-<s>.npz``)
and ``publication_run.evaluate_real_oof_ensemble`` (five averaged networks,
one asteroid at a time, as for the published ``real-oof-ensemble.npz``)
statement by statement.  The only change is that every asteroid is tokenized
with the period read from ``--periods``.  No reference pole enters this stage:
the examples carry a placeholder target axis, and only predictions (score
maps, starting axes, adjusted axes, scores, times) are written, to
``predictions/``.

``score`` runs afterwards.  It reads the DAMIT reference axes and writes, from
the saved predictions, files with exactly the keys, dtypes and layout of the
published files: ``real-fold-<f>-seed-<s>.npz``, ``real-oof-seed-<s>.npz`` and
``real-oof-ensemble.npz``.

``predict`` can be split into chunks (``--stage seeds|ensemble`` and
``--folds``) so that each chunk holds the GPU lock separately.  Every output
file is written once and never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from lc_pipeline.k3.config import K3ScoreModelConfig
from lc_pipeline.k3.damit import load_damit_object
from lc_pipeline.k3.evaluation import (
    ensemble_score_grids,
    modes_from_score_grid,
    oracle_at_k_error_deg,
)
from lc_pipeline.k3.inference import refine_axes, refine_ensemble_axes, score_axial_grid
from lc_pipeline.k3.model import CandidateConditionedScorer
from lc_pipeline.k3.publication_run import (
    K3_DEPLOYED_GRID_MAX_NORMALIZED_RMS,
    K3_OOF_SEEDS,
)
from lc_pipeline.k3.tokenizer import tokenize_epochs
from lc_pipeline.k3.training import K3TrainingExample, collate_examples
from lc_pipeline.v2.preprocessing import KnownPeriod

from repro.own_period import DataPath

ROOT = DataPath("../../..")
FOLLOWUP = DataPath("../..")
DATA = FOLLOWUP / "inputs/training-run/repro/data/damit-20250610T000301Z"
CATALOG = DATA / "catalog.jsonl"
SPLITS = DATA / "publication-splits-v2.3.json"
DUMP = ROOT / "damit-20250610T000301Z"
MODELS = FOLLOWUP / "inputs/frozen-artifacts/k3-definitive-7874092/models"
PUBLISHED = FOLLOWUP / "inputs/frozen-artifacts/k3-definitive-7874092/evaluations"
INPUT_KEYS = {"phase_features", "phase_mask", "geometry_features", "epoch_features", "epoch_mask"}
PLACEHOLDER_AXIS = np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32)


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_npz_once(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)


def write_json_once(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=1, sort_keys=True) + "\n")


def fold_rows() -> list[dict]:
    rows = sorted(json.loads(SPLITS.read_text())["folds"], key=lambda r: int(r["fold"]))
    if [int(r["fold"]) for r in rows] != list(range(5)):
        raise SystemExit("splits must hold folds 0-4")
    return rows


def catalog_rows() -> dict[str, dict]:
    return {
        row["object_id"]: row
        for row in (json.loads(line) for line in CATALOG.read_text().splitlines() if line.strip())
    }


def load_periods(spec: str) -> tuple[dict[str, float], str, str]:
    """Periods for the prediction stage.  ``damit`` reads the catalogue periods
    (the published choice: the first catalogue solution)."""
    if spec == "damit":
        periods = {}
        for oid, row in catalog_rows().items():
            if row.get("solutions"):
                periods[oid] = float(row["solutions"][0]["period_hours"])
        return periods, "frozen-DAMIT-primary-solution-fixed-period", "damit-catalog"
    path = Path(spec)
    value = json.loads(path.read_text())
    return value["periods_hours"], value["provenance"], sha256(path)


def build_examples(ids, periods, provenance) -> tuple[K3TrainingExample, ...]:
    """As ``publication_run.load_real_examples``, with the chosen period and
    without reference axes (a placeholder axis is carried instead)."""
    examples = []
    for oid in ids:
        period = periods.get(oid)
        if period is None or not np.isfinite(period) or period <= 0:
            raise SystemExit(f"{oid}: no usable period; failures are not supported in this run")
        source = load_damit_object(DUMP, oid)
        tokenized = tokenize_epochs(
            source.epochs, known_period=KnownPeriod(float(period), provenance)
        )
        examples.append(
            K3TrainingExample(object_id=oid, tokenized=tokenized, target_axes=PLACEHOLDER_AXIS)
        )
    return tuple(examples)


def load_model(fold: int, seed: int, device: str):
    path = MODELS / f"real-fold-{fold}-seed-{seed}.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("stage") != "real-oof" or int(checkpoint.get("config", {}).get("seed", -1)) != seed:
        raise SystemExit(f"unexpected checkpoint {path.name}")
    config = K3ScoreModelConfig(**checkpoint["model_config"])
    model = CandidateConditionedScorer(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, config, path


def predict_seed(fold, seed, examples, out: Path, device: str, batch_size: int = 4) -> None:
    """Statement-for-statement copy of ``_evaluate_examples`` without scoring."""
    destination = out / f"pred-fold-{fold}-seed-{seed}.npz"
    if destination.exists():
        print("exists", destination.name, flush=True)
        return
    model, _, path = load_model(fold, seed, device)
    grids = np.empty((len(examples), 6144), dtype=np.float32)
    initial_out = np.empty((len(examples), 3, 3), dtype=np.float64)
    refined_out = np.empty((len(examples), 3, 3), dtype=np.float64)
    identifiers: list[str] = []
    started = time.time()
    for start in range(0, len(examples), batch_size):
        values = examples[start : start + batch_size]
        raw = collate_examples(values)
        inputs = {key: value for key, value in raw.items() if key in INPUT_KEYS}
        inputs = {key: value.to(device) for key, value in inputs.items()}
        scores = np.asarray(score_axial_grid(model, inputs, chunk_size=1024))
        if scores.ndim == 1:
            scores = scores[None, :]
        grids[start : start + len(values)] = scores
        for index, example in enumerate(values):
            modes = modes_from_score_grid(scores[index])
            initial = np.asarray([mode.axis_xyz for mode in modes])
            one = {key: value[index : index + 1] for key, value in inputs.items()}
            axes, _ = refine_axes(model, one, initial)
            initial_out[start + index] = initial
            refined_out[start + index] = axes
            identifiers.append(example.object_id)
    write_npz_once(
        destination,
        object_ids=np.asarray(identifiers),
        score_grids=grids,
        initial_axes=initial_out,
        refined_axes=refined_out,
        checkpoint_sha256=np.asarray(sha256(path)),
    )
    print("seed", fold, seed, f"{time.time() - started:.1f}s", flush=True)


def predict_ensemble(fold, examples, out: Path, device: str) -> None:
    """Statement-for-statement copy of the per-fold loop of
    ``evaluate_real_oof_ensemble`` without scoring.  The cached map is the mean
    of this run's own per-seed maps (as published, where it is the mean of the
    published per-seed maps)."""
    destination = out / f"pred-ensemble-fold-{fold}.npz"
    if destination.exists():
        print("exists", destination.name, flush=True)
        return
    seed_grids = []
    for seed in K3_OOF_SEEDS:
        with np.load(out / f"pred-fold-{fold}-seed-{seed}.npz", allow_pickle=False) as artifact:
            if tuple(artifact["object_ids"].astype(str)) != tuple(e.object_id for e in examples):
                raise SystemExit(f"fold {fold} seed {seed} predictions do not match the split")
            seed_grids.append(np.asarray(artifact["score_grids"], dtype=np.float32))
    mean_grids = ensemble_score_grids(seed_grids)
    models, names, hashes = [], [], []
    for seed in K3_OOF_SEEDS:
        model, _, path = load_model(fold, seed, device)
        models.append(model)
        names.append(path.name)
        hashes.append(sha256(path))
    n = len(examples)
    result = {
        "score_grids": np.empty((n, 6144), dtype=np.float64),
        "initial_axes": np.empty((n, 3, 3), dtype=np.float64),
        "refined_axes": np.empty((n, 3, 3), dtype=np.float64),
        "refined_scores": np.empty((n, 3), dtype=np.float64),
        "inference_wall_seconds": np.empty(n, dtype=np.float64),
        "cached_grid_max_abs_differences": np.empty(n, dtype=np.float64),
        "cached_grid_mean_abs_differences": np.empty(n, dtype=np.float64),
        "cached_grid_centered_rms_differences": np.empty(n, dtype=np.float64),
        "cached_grid_normalized_rms_differences": np.empty(n, dtype=np.float64),
        "cached_mode_indices": np.empty((n, 3), dtype=np.int64),
        "deployed_mode_indices": np.empty((n, 3), dtype=np.int64),
    }
    for index, example in enumerate(examples):
        inference_started = time.perf_counter()
        raw = collate_examples((example,))
        inputs = {key: value.to(device) for key, value in raw.items() if key in INPUT_KEYS}
        deployed_grids = np.stack(
            [
                np.asarray(score_axial_grid(model, inputs, chunk_size=1024), dtype=np.float64)
                for model in models
            ]
        )
        deployed_mean_grid = ensemble_score_grids(deployed_grids[:, None, :])[0]
        cached_mean_grid = mean_grids[index]
        absolute_difference = np.abs(deployed_mean_grid - cached_mean_grid)
        result["cached_grid_max_abs_differences"][index] = float(np.max(absolute_difference))
        result["cached_grid_mean_abs_differences"][index] = float(np.mean(absolute_difference))
        centered_difference = (
            deployed_mean_grid
            - float(np.mean(deployed_mean_grid))
            - cached_mean_grid
            + float(np.mean(cached_mean_grid))
        )
        centered_rms = float(np.sqrt(np.mean(np.square(centered_difference))))
        normalized_rms = centered_rms / max(float(np.std(cached_mean_grid)), 1.0)
        result["cached_grid_centered_rms_differences"][index] = centered_rms
        result["cached_grid_normalized_rms_differences"][index] = normalized_rms
        if normalized_rms > K3_DEPLOYED_GRID_MAX_NORMALIZED_RMS:
            raise SystemExit(f"{example.object_id}: deployed map differs from cached seed maps")
        cached_modes = modes_from_score_grid(cached_mean_grid)
        modes = modes_from_score_grid(deployed_mean_grid)
        result["cached_mode_indices"][index] = tuple(int(m.grid_index) for m in cached_modes)
        result["deployed_mode_indices"][index] = tuple(int(m.grid_index) for m in modes)
        starts = np.asarray([mode.axis_xyz for mode in modes], dtype=np.float64)
        axes, scores = refine_ensemble_axes(models, inputs, starts)
        result["inference_wall_seconds"][index] = time.perf_counter() - inference_started
        result["score_grids"][index] = deployed_mean_grid
        result["initial_axes"][index] = starts
        result["refined_axes"][index] = axes
        result["refined_scores"][index] = scores
    result["score_grids"] = result["score_grids"].astype(np.float32)
    write_npz_once(
        destination,
        object_ids=np.asarray([e.object_id for e in examples]),
        checkpoint_names=np.asarray(names),
        checkpoint_sha256=np.asarray(hashes),
        **result,
    )
    print("ensemble", fold, f"median {np.median(result['inference_wall_seconds']):.3f}s", flush=True)


def cmd_predict(args) -> int:
    out = args.output_dir / "predictions"
    periods, provenance, source = load_periods(args.periods)
    record = {
        "periods": args.periods,
        "period_provenance": provenance,
        "period_source": source,
        "seeds": list(K3_OOF_SEEDS),
        "device": args.device,
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None,
        "code_sha256": sha256(Path(__file__)),
    }
    manifest = out / "prediction-run.json"
    if manifest.exists():
        if json.loads(manifest.read_text())["period_source"] != source:
            raise SystemExit("this output directory was started with other periods")
    else:
        write_json_once(manifest, record)
    for row in fold_rows():
        fold = int(row["fold"])
        if args.folds and fold not in args.folds:
            continue
        ids = list(row["test_ids"])
        if args.objects:
            ids = [oid for oid in ids if oid in args.objects]
            if not ids:
                continue
        examples = build_examples(ids, periods, provenance)
        if args.stage in ("seeds", "all"):
            for seed in K3_OOF_SEEDS:
                predict_seed(fold, seed, examples, out, args.device)
        if args.stage in ("ensemble", "all"):
            predict_ensemble(fold, examples, out, args.device)
    return 0


def cmd_score(args) -> int:
    """Score the saved predictions and write the published file layouts."""
    pred = args.output_dir / "predictions"
    out = args.output_dir / "evaluations"
    catalog = catalog_rows()

    def targets(oid):
        return np.asarray([s["vector"] for s in catalog[oid]["solutions"]], dtype=np.float32)

    def unit_targets(oid):
        # K3TrainingExample normalizes the float32 targets the same way.
        t = targets(oid)
        return t / np.linalg.norm(t, axis=1)[:, None]

    rows = fold_rows()
    expected_ids = [oid for row in rows for oid in row["test_ids"]]
    expected_folds = np.concatenate(
        [np.full(len(r["test_ids"]), int(r["fold"]), dtype=np.int8) for r in rows]
    )
    if args.objects:
        keep = np.isin(expected_ids, args.objects)
        expected_ids = [oid for oid, k in zip(expected_ids, keep) if k]
        expected_folds = expected_folds[keep]
    aggregate_hashes = []
    for seed in K3_OOF_SEEDS:
        ids_all, errors_all, grids_all, folds_all, hashes = [], [], [], [], []
        for row in rows:
            fold = int(row["fold"])
            source = pred / f"pred-fold-{fold}-seed-{seed}.npz"
            if not source.exists():
                continue
            with np.load(source, allow_pickle=False) as a:
                ids = a["object_ids"].astype(str)
                grids = np.asarray(a["score_grids"], dtype=np.float32)
                axes = np.asarray(a["refined_axes"], dtype=np.float64)
            errors = np.asarray(
                [oracle_at_k_error_deg(axes[i], unit_targets(oid), k=3) for i, oid in enumerate(ids)],
                dtype=np.float64,
            )
            target = out / f"real-fold-{fold}-seed-{seed}.npz"
            write_npz_once(target, object_ids=np.asarray(ids), score_grids=grids, oracle_errors_deg=errors)
            ids_all.extend(ids.tolist())
            errors_all.append(errors)
            grids_all.append(grids)
            folds_all.append(np.full(len(ids), fold, dtype=np.int8))
            hashes.append(sha256(target))
        aggregate = out / f"real-oof-seed-{seed}.npz"
        write_npz_once(
            aggregate,
            object_ids=np.asarray(ids_all),
            folds=np.concatenate(folds_all),
            score_grids=np.concatenate(grids_all),
            oracle_errors_deg=np.concatenate(errors_all),
            input_artifact_sha256=np.asarray(hashes),
        )
        aggregate_hashes.append(sha256(aggregate))
    parts = [np.load(pred / f"pred-ensemble-fold-{int(r['fold'])}.npz", allow_pickle=False)
             for r in rows if (pred / f"pred-ensemble-fold-{int(r['fold'])}.npz").exists()]
    ids = np.concatenate([p["object_ids"].astype(str) for p in parts])
    if ids.tolist() != list(expected_ids):
        raise SystemExit("ensemble predictions do not cover the expected asteroids in order")
    cat = {key: np.concatenate([p[key] for p in parts]) for key in (
        "score_grids", "initial_axes", "refined_axes", "refined_scores", "inference_wall_seconds",
        "cached_grid_max_abs_differences", "cached_grid_mean_abs_differences",
        "cached_grid_centered_rms_differences", "cached_grid_normalized_rms_differences",
        "cached_mode_indices", "deployed_mode_indices", "checkpoint_names", "checkpoint_sha256")}
    grid_errors = np.asarray(
        [oracle_at_k_error_deg(cat["initial_axes"][i], unit_targets(oid), k=3) for i, oid in enumerate(ids)])
    errors = np.asarray(
        [oracle_at_k_error_deg(cat["refined_axes"][i], unit_targets(oid), k=3) for i, oid in enumerate(ids)])
    write_npz_once(
        out / "real-oof-ensemble.npz",
        schema=np.asarray("delphi.k3-real-oof-ensemble.v1"),
        seeds=np.asarray(K3_OOF_SEEDS, dtype=np.int64),
        object_ids=np.asarray(list(ids)),
        folds=expected_folds,
        score_grids=cat["score_grids"].astype(np.float32),
        initial_axes=cat["initial_axes"],
        refined_axes=cat["refined_axes"],
        refined_scores=cat["refined_scores"],
        inference_wall_seconds=cat["inference_wall_seconds"],
        cached_grid_max_abs_differences=cat["cached_grid_max_abs_differences"],
        cached_grid_mean_abs_differences=cat["cached_grid_mean_abs_differences"],
        cached_grid_centered_rms_differences=cat["cached_grid_centered_rms_differences"],
        cached_grid_normalized_rms_differences=cat["cached_grid_normalized_rms_differences"],
        cached_mode_indices=cat["cached_mode_indices"],
        deployed_mode_indices=cat["deployed_mode_indices"],
        grid_oracle_errors_deg=grid_errors,
        oracle_errors_deg=errors,
        input_aggregate_sha256=np.asarray(aggregate_hashes),
        checkpoint_names=cat["checkpoint_names"],
        checkpoint_sha256=cat["checkpoint_sha256"],
    )
    summary = {
        "n": int(len(ids)),
        "mean_deg": float(errors.mean()),
        "grid_mean_deg": float(grid_errors.mean()),
        "per_seed_mean_deg": {
            str(s): float(np.load(out / f"real-oof-seed-{s}.npz")["oracle_errors_deg"].mean())
            for s in K3_OOF_SEEDS
        },
    }
    write_json_once(out / "score-summary.json", summary)
    print(json.dumps(summary, indent=1))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--objects", type=lambda s: s.split(","), help="smoke tests only: comma-separated asteroids")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--periods", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--stage", choices=("seeds", "ensemble", "all"), default="all")
    p.add_argument("--folds", type=int, nargs="*")
    sub.add_parser("score")
    args = parser.parse_args()
    return cmd_predict(args) if args.command == "predict" else cmd_score(args)


if __name__ == "__main__":
    sys.exit(main())
