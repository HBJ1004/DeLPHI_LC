"""Additional, explicitly retrospective K3 short-session and training-gap checks.

The registered reliability run is read-only input. This separate experiment
uses one fixed thinning realization, no tuning, and a common >=20 short-session
cohort. It never treats in-sample predictions as out-of-fold performance.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from lc_pipeline.k3.evidence_companion import angular_error, session_metadata, short_session_subset
from lc_pipeline.k3.reliability_sampling import read_lc, to_epochs
from lc_pipeline.k3.reliability_study import (
    binding,
    canonical,
    digest,
    exclusive,
    input_hash,
    progress,
    runtime_sources,
    seal,
    unseal,
    verify,
)

COUNTS = (1, 3, 5, 10, 20, None)
CAPS = (10, 30, None)
SEED = 20260916


def prepare(root, previous):
    root, previous = Path(root).resolve(), Path(previous).resolve()
    old = unseal(previous / "study.json")
    unseal(previous / "report.json")
    if root.exists():
        raise FileExistsError("output must be new")
    metadata = []
    for row in old["objects"]:
        curve = read_lc(verify(row["lightcurve"]), object_id=row["object_id"], period_hours=row["period_hours"])
        value = session_metadata(curve)
        value.pop("session_span_hours")
        metadata.append(value)
    splits = next(b for b in old["bindings"] if Path(b["path"]).name == "publication-splits-v2.3.json")
    for bundle in old["bundles"]:
        for member in bundle["files"]:
            verify(member)
    root.mkdir(parents=True)
    source = Path(__file__).resolve().parents[1]
    snapshot = root / "runtime-source"
    snapshot.mkdir()
    for name in ("lc_pipeline", "repro"):
        shutil.copytree(source / name, snapshot / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(source / "pyproject.toml", snapshot / "pyproject.toml")
    lock = {
        "schema": "delphi.k3-evidence-companion-checks.v1", "role": "post_hoc_diagnostic",
        "previous": str(previous), "parent_study": binding(previous / "study.json"),
        "parent_report": binding(previous / "report.json"), "splits": splits,
        "source_snapshot": str(snapshot), "runtime_sources": runtime_sources(snapshot),
        "python": binding(sys.executable), "objects": old["objects"], "bundles": old["bundles"],
        "eligible_ids": [r["object_id"] for r in metadata if r["short_sessions_le12h"] >= 20],
        "metadata": metadata, "session_counts": COUNTS, "point_caps": CAPS,
        "seed": SEED, "repeats": 1,
        "sampling": "native sessions lasting <=12h; seeded nested subsets; no epoch regrouping",
        "night_limitation": "session is a visit proxy; local night and observatory are not identified",
        "training_gap": "five-model ensemble on each fold's 108 training objects vs its 34 test objects",
        "statistical_unit": "asteroid; per-fold means for training gap, not 540 independent test objects",
        "max_execution_seconds": 7200,
    }
    seal(root / "lock.json", lock)
    return {"eligible": len(lock["eligible_ids"]), "night_predictions": len(lock["eligible_ids"]) * 18,
            "in_sample_predictions": 540, "root": str(root)}


def diagnostic_prediction(predictor, curve):
    """Explicit in-sample diagnostic kernel; do not disguise object identifiers.

    The public predictor rejects known training identities for ordinary use.
    This diagnostic calls its numerical kernels and labels every row in-sample.
    """
    from lc_pipeline.k3.bundle import _model_inputs
    from lc_pipeline.k3.evaluation import ensemble_score_grids, modes_from_score_grid
    from lc_pipeline.k3.inference import refine_ensemble_axes, score_axial_grid
    from lc_pipeline.v2.preprocessing import KnownPeriod
    inputs = _model_inputs(to_epochs(curve), KnownPeriod(hours=curve.period_hours,
                           provenance="supplied-period-retrospective-training-gap"), predictor.device)
    maps = [score_axial_grid(model, inputs, chunk_size=1024) for model in predictor.models]
    score = ensemble_score_grids(np.stack(maps)[:, None, :])[0]
    modes = modes_from_score_grid(score)
    axes, _ = refine_ensemble_axes(predictor.models, inputs, np.asarray([m.axis_xyz for m in modes]))
    return axes


def run(root):
    import torch

    from lc_pipeline.k3.bundle import K3EnsemblePredictor
    from lc_pipeline.v2.preprocessing import KnownPeriod
    root = Path(root).resolve()
    with exclusive(root):
        lock = unseal(root / "lock.json")
        if runtime_sources(Path(__file__).resolve().parents[1]) != lock["runtime_sources"]:
            raise ValueError("execute the frozen source snapshot")
        for key in ("parent_study", "parent_report", "python", "splits"):
            verify(lock[key])
        budget_path = root / "budget.json"
        if not budget_path.exists():
            seal(budget_path, {"deadline": time.time() + lock["max_execution_seconds"]})
        deadline = unseal(budget_path)["deadline"]
        splits = json.loads(Path(lock["splits"]["path"]).read_text())["folds"]
        curves = {r["object_id"]: read_lc(verify(r["lightcurve"]), object_id=r["object_id"],
                  period_hours=r["period_hours"]) for r in lock["objects"]}
        torch.set_num_threads(6)
        count = 0
        expected = []
        try:
            for fold in range(5):
                predictor = K3EnsemblePredictor(lock["bundles"][fold]["directory"], device="cuda")
                jobs = [(oid, "in_sample", None, None) for oid in splits[fold]["train_ids"]]
                jobs += [(oid, "short_sessions", n, cap) for oid in splits[fold]["test_ids"]
                         if oid in lock["eligible_ids"] for n in lock["session_counts"] for cap in lock["point_caps"]]
                for oid, role, n, cap in jobs:
                    name = "in_sample" if role == "in_sample" else f"sessions-{n}-cap-{cap}"
                    path = root / "predictions" / f"fold-{fold}" / oid / f"{name}.json"
                    expected.append(str(path.relative_to(root)))
                    curve = curves[oid]
                    if role == "short_sessions":
                        curve = short_session_subset(curve, session_count=n, point_cap=cap, seed=lock["seed"])
                    fingerprint = input_hash(curve)
                    if path.exists():
                        cached = unseal(path)
                        if (cached["input_sha256"] != fingerprint or cached["object_id"] != oid
                                or cached["role"] != role or cached["fold"] != fold):
                            raise ValueError("cached prediction binding mismatch")
                    else:
                        if time.time() >= deadline:
                            raise TimeoutError("companion check budget exhausted; no partial scientific report")
                        progress(root, stage="prediction", fold=fold, object_id=oid, role=role,
                                 condition=name, completed=count, total=540 + 18 * len(lock["eligible_ids"]))
                        start = time.monotonic()
                        row = {"object_id": oid, "fold": fold, "role": role, "condition": name,
                               "session_count_cap": n, "point_cap": cap, "input_sha256": fingerprint,
                               "seed": lock["seed"], "actual_sessions": len(curve.valid_sessions),
                               "actual_points": sum(len(s.observations) for s in curve.valid_sessions)}
                        try:
                            if role == "in_sample":
                                assert oid in splits[fold]["train_ids"]
                                axes = diagnostic_prediction(predictor, curve)
                            else:
                                assert oid in splits[fold]["test_ids"]
                                result = predictor.predict(to_epochs(curve), known_period=KnownPeriod(
                                    hours=curve.period_hours, provenance="supplied-period-short-session-study"), object_id=oid)
                                axes = np.asarray([v["axis_xyz"] for v in result["axes"]])
                            if axes.shape != (3, 3) or not np.isfinite(axes).all():
                                raise ValueError("malformed prediction")
                            row.update(status="ok", axes=axes.tolist())
                        except (ValueError, RuntimeError) as exc:
                            row.update(status="failed", axes=None, reason=str(exc))
                            torch.cuda.empty_cache()
                        row["runtime_seconds"] = time.monotonic() - start
                        seal(path, row)
                    count += 1
                del predictor
                torch.cuda.empty_cache()
            if not (root / "predictions-complete.json").exists():
                seal(root / "predictions-complete.json", {"files": [{"path": p, "sha256": digest(root / p)} for p in expected]})
            score(root, lock)
            progress(root, stage="complete", completed=count)
        except Exception as exc:
            progress(root, stage="stopped", reason=f"{type(exc).__name__}: {exc}")
            raise
    return {"root": str(root), "stage": "complete", "n_predictions": count}


def score(root, lock):
    if (root / "results.json").exists():
        return unseal(root / "results.json")
    old = unseal(Path(lock["previous"]) / "study.json")
    catalog = {r["object_id"]: r for r in map(json.loads, verify(old["catalog"]).read_text().splitlines())}
    rows = []
    receipt = unseal(root / "predictions-complete.json")
    expected_n = 540 + 18 * len(lock["eligible_ids"])
    if len(receipt["files"]) != expected_n:
        raise ValueError("incomplete prediction schedule")
    for entry in receipt["files"]:
        path = root / entry["path"]
        if digest(path) != entry["sha256"]:
            raise ValueError("changed prediction")
        row = unseal(path)
        row["oracle_error_deg"] = angular_error(row["axes"], [s["vector"] for s in catalog[row["object_id"]]["solutions"]]) if row["status"] == "ok" else 90.0
        rows.append(row)
    result = {"role": "post_hoc_diagnostic", "rows": rows, "one_sampling_realization": True,
              "limitations": ["training rows are in-sample", "short sessions are not confirmed local nights",
                              "no new model or candidate ranking", "no model selection from this report"]}
    seal(root / "results.json", result)
    return result


def start(root):
    root = Path(root).resolve()
    lock = unseal(root / "lock.json")
    if (root / "launch.json").exists():
        raise ValueError("existing launch: inspect process and use run explicitly to resume")
    env = {key: value for key, value in os.environ.items() if key in ("PATH", "HOME", "LANG", "LC_ALL", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES")}
    env.update(PYTHONPATH=lock["source_snapshot"], OMP_NUM_THREADS="6", MKL_NUM_THREADS="6", OPENBLAS_NUM_THREADS="6")
    with (root / "run.log").open("xb") as log:
        p = subprocess.Popen([str(verify(lock["python"])), "-m", "repro.run_k3_companion_checks", "run", "--root", str(root)],
                             cwd=lock["source_snapshot"], env=env, stdin=subprocess.DEVNULL,
                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    receipt = {"pid": p.pid, "launched_epoch": time.time()}
    seal(root / "launch.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "start"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--previous", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.root, args.previous)
    elif args.command == "run":
        result = run(args.root)
    else:
        result = start(args.root)
    print(canonical(result), flush=True)


if __name__ == "__main__":
    main()
