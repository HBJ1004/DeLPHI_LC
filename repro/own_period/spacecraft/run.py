#!/usr/bin/env python3
"""Frozen spacecraft experiment, one exclusive output stage per subcommand.

Run with PYTHONPATH=<worktree> python -I repro/own_period/spacecraft/run.py
<command> --output-dir <spacecraft>.  -I ignores PYTHONPATH, so only this
trusted checkout is explicitly added below. Downloaded inputs are JSON/text.

Order: validate (CPU), lock, geometry (network), sbdb (network), prepare,
search (CUDA, under lock_run.sh), predict (CUDA, under lock_run.sh), score.
Every stage refuses an existing destination, including incomplete stages.
Source extractions are never edited: lock writes bound derivatives under lock/.
There are deliberately no alternate search, geometry, or rescue policies.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
WORKTREE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORKTREE))
FOLLOWUP = WORKTREE.parent
OWN = FOLLOWUP / "data/own-period-20261021"
OUTPUT = OWN / "spacecraft"
SNAPSHOT = FOLLOWUP.parent / "damit-20250610T000301Z"
TRAINING = FOLLOWUP / "inputs/training-run/repro/data/damit-20250610T000301Z"
MODELS = FOLLOWUP / "inputs/frozen-artifacts/k3-definitive-7874092/models"
MODEL_MANIFEST = FOLLOWUP / "data/external-models-final.json"
PUBLISHED_PREPARED = FOLLOWUP / "data/generalization-20260910/locked-cohort/alcdef-prepared-30-20260911"
PUBLISHED_AXES = OWN / "validation-alcdef-published.jsonl"
NEW = {"itokawa": (25143, ("alcdef", "lowry2014", "combined")),
       "ryugu": (162173, ("alcdef", "kim2013", "combined")),
       "bennu": (101955, ("hergenrother2020", "combined"))}
DAMIT = {"lutetia": 113, "ida": 478, "gaspra": 477}
RESCORE = {"eros": 1714, "ceres": 3414, "vesta": 3415}
OOF_FILES = {"own": "own-period-oof.npz", "catalogue": "validation-damit-period-oof.npz"}
# CPU float32 versus the archived CUDA prediction is not bit-exact. This
# component bound is below 0.01 degree for these unit axes; report actual drift.
AXIS_TOLERANCE = 1e-4


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    from repro.own_period.search_external import sha256
    return sha256(path)


def binding(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha(path)}


def checked(row: dict, root: Path | None = None) -> Path:
    path = Path(row["path"]) if root is None else root / row["path"]
    if root is not None and not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"path escapes stage: {path}")
    if sha(path) != row["sha256"]:
        raise ValueError(f"hash mismatch: {path}")
    return path


def write(path: Path, value: object) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)


def stage(out: Path, name: str) -> Path:
    path = out / name
    path.mkdir(parents=True, exist_ok=False)
    return path


def available(out: Path, name: str) -> None:
    path = out / name
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite stage: {path}")


def relative_binding(path: Path, root: Path) -> dict:
    return {"path": path.relative_to(root).as_posix(), "sha256": sha(path)}


def raw_path(out: Path, row: dict) -> Path:
    return checked(row, out / "lock" if row["kind"] == "external" else None)


def locked(out: Path) -> tuple[dict, list[dict]]:
    root = out / "lock"
    manifest = read(root / "manifest.json")
    lock_path = checked(manifest["cohort_lock"], root)
    lock = read(lock_path)
    for entry in lock["files"].values():
        checked(entry)
    rows = manifest["inputs"]
    if [r["run_id"] for r in rows] != [r["run_id"] for r in lock["inputs"]]:
        raise ValueError("locked input order differs")
    for row, original in zip(rows, lock["inputs"], strict=True):
        source = checked(original["source"])
        path = raw_path(out, row)
        if row["kind"] == "external":
            expected = read(source)
            expected["cohort_lock"] = {"path": str(lock_path.resolve()), "sha256": sha(lock_path)}
            if read(path) != expected:
                raise ValueError(f"bound raw extraction differs: {row['run_id']}")
        elif path != source:
            raise ValueError("DAMIT lightcurve path differs")
        for key in ("run_id", "kind", "object_id", "name", "source_name", "primary"):
            if row[key] != original[key]:
                raise ValueError(f"locked metadata differs: {key}")
    return lock, rows


def cmd_lock(args) -> dict:
    from repro import prepare_k3_alcdef_transfer as adapter
    from repro.own_period.spacecraft.convert import training_check

    out = args.output_dir
    available(out, "lock")
    raw_dir = args.raw_dir or out
    snapshot = args.snapshot_dir
    rows = []
    for name, (number, sources) in NEW.items():
        for source in sources:
            path = raw_dir / f"raw-extraction-{name}-{source}.json"
            raw = adapter._read(path, "delphi.k3-locked-alcdef-raw-extract.v1")
            objects = adapter._numbers(raw)
            if len(objects) != 1 or objects[0][:2] != (f"mpc:{number}", number):
                raise ValueError(f"unexpected raw identity: {path}")
            adapter._midpoints(objects[0][2], objects[0][3])
            rows.append({"run_id": f"{name}-{source}", "kind": "external",
                         "name": name, "source_name": source, "primary": source == "combined",
                         "object_id": f"mpc:{number}", "source": binding(path)})
    for name, number in DAMIT.items():
        rows.append({"run_id": f"{name}-damit", "kind": "damit", "name": name,
                     "source_name": "damit", "primary": True,
                     "object_id": f"asteroid_{number}",
                     "source": binding(snapshot / f"files/asteroid_{number}/lc.txt")})
    files = {"protocol": binding(out / "PROTOCOL.md"),
             "reference_poles": binding(out / "reference-poles.json"),
             "script": binding(Path(__file__)), "model_manifest": binding(MODEL_MANIFEST),
             "splits": binding(TRAINING / "publication-splits-v2.3.json"),
             "model_table": binding(snapshot / "tables/asteroid_models.csv")}
    for mode, filename in OOF_FILES.items():
        files[f"oof_{mode}"] = binding(args.oof_dir / filename)
    # Freeze implementations as well as observations, without reading pole columns.
    for i, relative in enumerate(("repro/prepare_k3_alcdef_transfer.py",
                                 "repro/fetch_k3_alcdef_periods.py",
                                 "repro/own_period/search_external.py",
                                 "repro/own_period/predict_external.py",
                                 "repro/own_period/predict_oof.py",
                                 "repro/own_period/score.py", "repro/period_search/search.py",
                                 "lc_pipeline/v2/data.py")):
        files[f"implementation_{i}"] = binding(WORKTREE / relative)
    lock = {"schema": "delphi.k3-alcdef-gaia-cohort-lock.v1",
            "purpose": "exploratory spacecraft cohort; Mathilde excluded by PROTOCOL.md",
            "objects": [{"object_id": f"mpc:{number}", "mpc_number": number, "name": name}
                        for name, (number, _) in NEW.items()],
            "inputs": rows, "files": files, "training_check": training_check(),
            "catalogue_period_rule": "DAMIT model table, lowest numeric model id; SBDB rot_per for new data",
            "rescore_only": RESCORE}
    root = stage(out, "lock")
    path = root / "cohort-lock.json"
    write(path, lock)
    bound = []
    for row in rows:
        metadata = {k: v for k, v in row.items() if k != "source"}
        if row["kind"] == "external":
            value = read(Path(row["source"]["path"]))
            value["cohort_lock"] = {"path": str(path.resolve()), "sha256": sha(path)}
            target = root / "raw" / f"{row['run_id']}.json"
            write(target, value)
            metadata.update(relative_binding(target, root))
        else:
            metadata.update(row["source"])
        bound.append(metadata)
    write(root / "manifest.json", {"schema": "delphi.spacecraft-lock.v1",
          "cohort_lock": relative_binding(path, root), "inputs": bound})
    return {"stage": "lock", "raw_extractions": 8, "damit_objects": 3, "sha256": sha(path)}


def cmd_geometry(args) -> dict:
    from repro.prepare_k3_alcdef_transfer import fetch_geometry
    out = args.output_dir
    available(out, "geometry")
    _, rows = locked(out)
    root = stage(out, "geometry")
    entries = []
    for row in rows:
        if row["kind"] != "external":
            continue
        raw = checked(row, out / "lock")
        path = root / f"{row['run_id']}.json"
        fetch_geometry(raw, path, batch_size=args.batch_size,
                       timeout_seconds=args.timeout_seconds, interval_seconds=args.request_interval_seconds)
        entries.append({"run_id": row["run_id"], **relative_binding(path, root)})
        print("geometry", row["run_id"], flush=True)
    write(root / "manifest.json", {"cohort_lock_sha256": sha(out / "lock/cohort-lock.json"),
                                   "objects": entries})
    return {"stage": "geometry", "count": len(entries)}


def cmd_sbdb(args) -> dict:
    from repro.fetch_k3_alcdef_periods import fetch
    out = args.output_dir
    available(out, "sbdb")
    locked(out)
    root = stage(out, "sbdb")
    value = fetch(out / "lock/cohort-lock.json", root / "periods.json",
                  timeout_seconds=args.timeout_seconds)
    return {"stage": "sbdb", "statuses": {r["object_id"]: r["status"] for r in value["objects"]}}


def geometry_rows(out: Path) -> dict:
    manifest = read(out / "geometry/manifest.json")
    if manifest["cohort_lock_sha256"] != sha(out / "lock/cohort-lock.json"):
        raise ValueError("geometry manifest belongs to another lock")
    return {r["run_id"]: checked(r, out / "geometry") for r in manifest["objects"]}


def cmd_prepare(args) -> dict:
    from repro.prepare_k3_alcdef_transfer import prepare
    out = args.output_dir
    available(out, "prepared")
    _, rows = locked(out)
    geometry = geometry_rows(out)
    root = stage(out, "prepared")
    entries = []
    for row in rows:
        if row["kind"] != "external":
            continue
        manifest_path = root / "source-manifests" / f"{row['run_id']}.json"
        result = prepare(checked(row, out / "lock"), out / "sbdb/periods.json",
                         geometry[row["run_id"]], manifest_path)
        if len(result["objects"]) != 1 or result["objects"][0]["object_id"] != row["object_id"]:
            raise ValueError("prepared identity differs")
        value = result["objects"][0]
        target = root / "objects" / f"{row['run_id']}.json"
        write(target, value)
        entries.append({"run_id": row["run_id"], "object_id": row["object_id"],
                        **relative_binding(target, root),
                        "source_manifest": relative_binding(manifest_path, root),
                        "epoch_count": len(value["epochs"]),
                        "observation_count": sum(len(e["observations"]) for e in value["epochs"])})
    write(root / "manifest.json", {"schema": "delphi.spacecraft-prepared.v1",
          "cohort_lock_sha256": sha(out / "lock/cohort-lock.json"),
          "period_receipts_sha256": sha(out / "sbdb/periods.json"),
          "geometry_manifest_sha256": sha(out / "geometry/manifest.json"), "objects": entries})
    return {"stage": "prepare", "count": len(entries)}


def prepared(out: Path) -> dict:
    manifest = read(out / "prepared/manifest.json")
    for key, path in (("cohort_lock_sha256", out / "lock/cohort-lock.json"),
                      ("period_receipts_sha256", out / "sbdb/periods.json"),
                      ("geometry_manifest_sha256", out / "geometry/manifest.json")):
        if manifest[key] != sha(path):
            raise ValueError(f"prepared dependency differs: {path}")
    geometry = geometry_rows(out)
    result = {}
    for row in manifest["objects"]:
        path = checked(row, out / "prepared")
        source = read(checked(row["source_manifest"], out / "prepared"))
        if (source["geometry_sha256"] != sha(geometry[row["run_id"]])
                or source["period_receipts_sha256"] != manifest["period_receipts_sha256"]
                or source["raw_extraction_sha256"] != sha(out / "lock/raw" / f"{row['run_id']}.json")):
            raise ValueError("prepared geometry, raw extraction, or period receipt differs")
        if source["objects"] != [read(path)]:
            raise ValueError("materialized prepared input differs")
        result[row["run_id"]] = path
    expected = {f"{name}-{s}" for name, (_, sources) in NEW.items() for s in sources}
    if set(result) != expected:
        raise ValueError("prepared inputs differ from the eight registered runs")
    return result


def cmd_search(args) -> dict:
    from repro.own_period.search_external import PROVENANCE, curves_of
    from repro.period_search.search import estimate, read_curves
    out = args.output_dir
    available(out, "search")
    _, rows = locked(out)
    inputs = prepared(out)
    root = stage(out, "search")
    periods, details = {}, {}
    for row in rows:
        path = inputs[row["run_id"]] if row["kind"] == "external" else raw_path(out, row)
        curves = curves_of(path) if row["kind"] == "external" else read_curves(path)
        started = time.perf_counter()
        try:
            result = estimate(curves, "fourier_gpu", resolution=1, timeout=180, sparse_fallback=True)
        except TimeoutError as error:
            result = {"status": "timeout", "reason": str(error), "ranked_periods": []}
        ranked = [float(p) for p in (result.get("ranked_periods") or [])]
        period = ranked[0] if result.get("status") == "ok" and ranked else None
        if period is not None and (not math.isfinite(period) or period <= 0):
            raise ValueError("search returned an invalid period")
        run_id = row["run_id"]
        periods[run_id] = period
        details[run_id] = {"status": result.get("status"), "reason": result.get("reason"),
                           "input_regime": result.get("input_regime"), "ranked_periods": ranked[:5],
                           "seconds": time.perf_counter() - started, "input_sha256": sha(path),
                           "curves": len(curves), "observations": sum(len(c.time) for c in curves)}
        print("search", run_id, details[run_id]["status"], period, flush=True)
    write(root / "periods.json", {"schema": "delphi.own-periods.v1", "cohort": "spacecraft",
          "provenance": PROVENANCE, "cohort_lock_sha256": sha(out / "lock/cohort-lock.json"),
          "prepared_manifest_sha256": sha(out / "prepared/manifest.json"),
          "search_source_sha256": sha(WORKTREE / "repro/period_search/search.py"),
          "search_call": 'estimate(curves, "fourier_gpu", resolution=1, timeout=180, sparse_fallback=True)',
          "periods_hours": periods, "details": details})
    return {"stage": "search", "found": sum(p is not None for p in periods.values()), "count": len(rows)}


def ensemble(device: str, threads: int):
    import torch
    from lc_pipeline.k3.ztf_prediction import _load_models
    torch.set_num_threads(threads)
    models, names, bindings = _load_models(MODELS, MODEL_MANIFEST, tuple(range(5)), device)
    if len(models) != 25:
        raise ValueError("the spacecraft experiment requires all 25 frozen networks")
    return models, names, bindings


def prediction(models, epochs, hours: float, provenance: str, device: str) -> dict:
    import numpy as np
    from lc_pipeline.v2.preprocessing import KnownPeriod
    from repro.own_period.predict_external import predict
    started = time.perf_counter()
    axes, scores = predict(models, epochs, KnownPeriod(hours, provenance), device, "alcdef")
    axes, scores = np.asarray(axes), np.asarray(scores)
    if axes.shape != (3, 3) or scores.shape != (3,) or not np.isfinite(axes).all() or not np.isfinite(scores).all():
        raise ValueError("prediction must contain three finite candidates and scores")
    return {"status": "ok", "period_hours": hours, "period_provenance": provenance,
            "axes": axes.tolist(), "scores": scores.tolist(), "folds": list(range(5)),
            "inference_seconds": time.perf_counter() - started}


def validation(out: Path) -> dict:
    report = read(out / "validation/result.json")
    if (report["status"] != "PASS" or report["script_sha256"] != sha(Path(__file__))
            or report["model_manifest_sha256"] != sha(MODEL_MANIFEST)
            or report["max_axis_difference"] > AXIS_TOLERANCE
            or report["published_axes_sha256"] != sha(PUBLISHED_AXES)):
        raise ValueError("successful validation of this script and frozen ensemble is required first")
    return report


def model_rows(lock: dict) -> dict:
    with checked(lock["files"]["model_table"]).open(encoding="utf-8-sig", newline="") as stream:
        grouped = {}
        for row in csv.DictReader(stream):
            grouped.setdefault(int(row["asteroid_id"]), []).append(row)
    return {key: sorted(rows, key=lambda r: int(r["id"])) for key, rows in grouped.items()}


def cmd_predict(args) -> dict:
    from lc_pipeline.k3.ztf_prediction import _load_prepared
    from lc_pipeline.v2.data import parse_damit_lightcurve
    out = args.output_dir
    available(out, "predictions")
    validation(out)
    lock, rows = locked(out)
    inputs = prepared(out)
    own = read(out / "search/periods.json")
    if (own["cohort_lock_sha256"] != sha(out / "lock/cohort-lock.json")
            or own["prepared_manifest_sha256"] != sha(out / "prepared/manifest.json")
            or set(own["periods_hours"]) != {r["run_id"] for r in rows}):
        raise ValueError("period file belongs to different inputs")
    table = model_rows(lock)
    models, names, bindings = ensemble(args.device, args.threads)
    root = stage(out, "predictions")
    outputs = {mode: [] for mode in OOF_FILES}
    for row in rows:
        run_id = row["run_id"]
        path = inputs[run_id] if row["kind"] == "external" else raw_path(out, row)
        if own["details"][run_id]["input_sha256"] != sha(path):
            raise ValueError("search and prediction observations differ")
        if row["kind"] == "external":
            identity, catalogue, epochs = _load_prepared(path)
            if identity != row["object_id"]:
                raise ValueError("prepared object identity differs")
            catalogue_hours, catalogue_provenance = catalogue.hours, catalogue.provenance
        else:
            epochs = parse_damit_lightcurve(path)  # The published low-Q census loader.
            selected = table[DAMIT[row["name"]]][0]
            catalogue_hours = float(selected["period"])
            catalogue_provenance = f"DAMIT primary model-table period; model {selected['id']}"
        for mode in OOF_FILES:
            period = own["periods_hours"][run_id] if mode == "own" else catalogue_hours
            provenance = own["provenance"] if mode == "own" else catalogue_provenance
            result = ({"status": "no_period", "period_hours": None} if period is None else
                      prediction(models, epochs, float(period), provenance, args.device))
            outputs[mode].append({"run_id": run_id, "object_id": row["object_id"],
                                  "input_sha256": sha(path), **result})
            print("predict", mode, run_id, result["status"], flush=True)
    products = {}
    for mode, values in outputs.items():
        path = root / f"{mode}.jsonl"
        with path.open("x", encoding="utf-8") as stream:
            for row in values:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        products[mode] = relative_binding(path, root)
    write(root / "manifest.json", {"schema": "delphi.spacecraft-predictions.v1",
          "cohort_lock_sha256": sha(out / "lock/cohort-lock.json"), "device": args.device,
          "search_sha256": sha(out / "search/periods.json"),
          "prepared_manifest_sha256": sha(out / "prepared/manifest.json"),
          "validation_sha256": sha(out / "validation/result.json"),
          "models": names, "model_bindings": bindings, "outputs": products})
    return {"stage": "predict", "inputs": len(rows), "models": len(models), "modes": list(outputs)}


def cmd_validate(args) -> dict:
    import numpy as np
    import torch
    from lc_pipeline.k3.ztf_prediction import _load_prepared
    out = args.output_dir
    available(out, "validation")
    rows = [r for r in read(PUBLISHED_PREPARED / "manifest.json")["objects"]
            if r["object_id"] == args.object_id]
    reference = [json.loads(line) for line in PUBLISHED_AXES.read_text().splitlines() if line.strip()]
    reference = [r for r in reference if r["object_id"] == args.object_id and r["status"] == "ok"]
    if len(rows) != 1 or len(reference) != 1:
        raise ValueError("validation requires one published ready ALCDEF cohort object")
    path = checked(rows[0], PUBLISHED_PREPARED)
    identity, period, epochs = _load_prepared(path)
    if period.hours != reference[0]["period_hours"]:
        raise ValueError("published validation periods differ")
    models, names, bindings = ensemble(args.device, args.threads)
    result = prediction(models, epochs, period.hours, period.provenance, args.device)
    actual, expected = np.asarray(result["axes"]), np.asarray(reference[0]["axes"])
    difference = float(np.max(np.abs(actual - expected)))
    a = actual / np.linalg.norm(actual, axis=1, keepdims=True)
    b = expected / np.linalg.norm(expected, axis=1, keepdims=True)
    angles = np.degrees(np.arccos(np.clip(np.abs(np.sum(a * b, axis=1)), 0, 1)))
    report = {"status": "PASS" if difference <= AXIS_TOLERANCE else "FAIL",
              "object_id": identity, "max_axis_difference": difference,
              "max_axis_difference_deg": float(angles.max()), "component_tolerance": AXIS_TOLERANCE,
              "comparison": "same ordered candidates, no permutation or sign alignment",
              "max_score_difference": float(np.max(np.abs(np.asarray(result["scores"]) - reference[0]["scores"]))),
              "script_sha256": sha(Path(__file__)), "model_manifest_sha256": sha(MODEL_MANIFEST),
              "prepared_sha256": sha(path), "published_axes_sha256": sha(PUBLISHED_AXES),
              "device": args.device, "threads": args.threads, "torch": torch.__version__,
              "python": platform.python_version(), "platform": platform.platform(),
              "models": names, "model_bindings": bindings, "prediction": result}
    root = stage(out, "validation")
    write(root / "result.json", report)
    print(json.dumps({k: report[k] for k in ("status", "object_id", "max_axis_difference", "max_axis_difference_deg")}), flush=True)
    if report["status"] != "PASS":
        raise ValueError("published ALCDEF axes were not reproduced; see validation/result.json")
    return {k: report[k] for k in ("status", "object_id", "max_axis_difference", "max_axis_difference_deg")}


def errors(result: dict, spacecraft, damit) -> dict:
    import numpy as np
    from repro.own_period.score import FAILED_DEG, axial_error
    if result["status"] == "no_period":
        return {"status": "no_period", "period_hours": None, "oracle_deg": FAILED_DEG,
                "best_scored_deg": FAILED_DEG, "oracle_excluding_failures_deg": None,
                "best_scored_excluding_failures_deg": None,
                "damit_oracle_deg": FAILED_DEG if damit is not None else None,
                "damit_best_scored_deg": FAILED_DEG if damit is not None else None}
    if result["status"] != "ok":
        raise ValueError(f"unexpected prediction status: {result['status']}")
    axes, scores = np.asarray(result["axes"], float), np.asarray(result["scores"], float)
    if (axes.shape != (3, 3) or scores.shape != (3,) or not np.isfinite(axes).all()
            or not np.isfinite(scores).all() or (np.linalg.norm(axes, axis=1) == 0).any()):
        raise ValueError("invalid candidates or scores")
    best = int(np.argmax(scores))
    oracle = axial_error(axes, spacecraft)
    top = axial_error(axes[best:best + 1], spacecraft)
    return {"status": "ok", "period_hours": float(result["period_hours"]),
            "oracle_deg": oracle, "best_scored_deg": top, "best_scored_candidate": best,
            "oracle_excluding_failures_deg": oracle, "best_scored_excluding_failures_deg": top,
            "damit_oracle_deg": axial_error(axes, damit) if damit is not None else None,
            "damit_best_scored_deg": axial_error(axes[best:best + 1], damit) if damit is not None else None}


def cmd_score(args) -> dict:
    import numpy as np
    from repro.compute_reference_baselines import unit_from_degrees
    from repro.own_period.score import period_class
    out = args.output_dir
    available(out, "scores.json")
    available(out, "RESULTS.md")
    lock, rows = locked(out)
    inputs = prepared(out)
    own = read(out / "search/periods.json")
    manifest = read(out / "predictions/manifest.json")
    for key, path in (("cohort_lock_sha256", out / "lock/cohort-lock.json"),
                      ("search_sha256", out / "search/periods.json"),
                      ("prepared_manifest_sha256", out / "prepared/manifest.json"),
                      ("validation_sha256", out / "validation/result.json")):
        if manifest[key] != sha(path):
            raise ValueError(f"prediction dependency differs: {path}")
    validation(out)
    table = model_rows(lock)
    predictions = {}
    for mode in OOF_FILES:
        path = checked(manifest["outputs"][mode], out / "predictions")
        values = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        predictions[mode] = {r["run_id"]: r for r in values}
        if len(values) != len(rows) or set(predictions[mode]) != {r["run_id"] for r in rows}:
            raise ValueError("missing or duplicate prediction rows")
        for row in rows:
            value = predictions[mode][row["run_id"]]
            source = inputs[row["run_id"]] if row["kind"] == "external" else raw_path(out, row)
            if mode == "own":
                expected_period = own["periods_hours"][row["run_id"]]
            elif row["kind"] == "external":
                expected_period = read(source)["known_period_hours"]
            else:
                expected_period = float(table[DAMIT[row["name"]]][0]["period"])
            if (value["object_id"] != row["object_id"] or value["input_sha256"] != sha(source)
                    or value["period_hours"] != expected_period
                    or (value["status"] == "no_period") != (expected_period is None)):
                raise ValueError("prediction input or period differs")
    reference_path = checked(lock["files"]["reference_poles"])
    reference = read(reference_path)
    if reference["schema"] != "delphi.spacecraft-reference-poles.v1":
        raise ValueError("reference pole schema differs")
    references = {r["object"].lower(): r for r in reference["objects"]}
    for name, number in RESCORE.items():
        rows.append({"run_id": f"{name}-oof", "name": name, "source_name": "existing-oof",
                     "object_id": f"asteroid_{number}", "primary": True, "kind": "rescore"})
    for mode in OOF_FILES:
        with np.load(checked(lock["files"][f"oof_{mode}"]), allow_pickle=False) as data:
            ids = [str(x) for x in data["object_ids"]]
            for name, number in RESCORE.items():
                identity = f"asteroid_{number}"
                if ids.count(identity) != 1:
                    raise ValueError(f"OOF identity missing or repeated: {identity}")
                i = ids.index(identity)
                p = float(data["period_hours"][i])
                result = ({"status": "no_period", "period_hours": None} if not math.isfinite(p) else
                          {"status": "ok", "period_hours": p,
                           "axes": data["refined_axes"][i].tolist(),
                           "scores": data["refined_scores"][i].tolist()})
                predictions[mode][f"{name}-oof"] = result
    objects = []
    for row in rows:
        ref = references[row["name"]]
        vector = np.asarray(ref["unit_vector"], float).reshape(1, 3)
        if not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
            raise ValueError("invalid spacecraft reference axis")
        vector /= np.linalg.norm(vector, axis=1, keepdims=True)
        number = ref["damit_id"]
        poles = (np.asarray([unit_from_degrees(float(r["lambda"]), float(r["beta"]))
                             for r in table[int(number)]]) if number is not None else None)
        values = {mode: errors(predictions[mode][row["run_id"]], vector, poles) for mode in OOF_FILES}
        objects.append({k: row[k] for k in ("run_id", "object_id", "name", "source_name", "primary", "kind")} |
                       {"period_class": period_class(values["own"]["period_hours"], values["catalogue"]["period_hours"]),
                        "spacecraft_reference": ref, **values})
    medians = {}
    primary = [r for r in objects if r["primary"]]
    for mode in OOF_FILES:
        good = [r for r in primary if r[mode]["status"] == "ok"]
        medians[mode] = {"n_primary": len(primary), "n_failed": len(primary) - len(good)}
        for key in ("oracle_deg", "best_scored_deg"):
            medians[mode][f"median_{key}_failed_at_90"] = float(np.median([r[mode][key] for r in primary]))
            medians[mode][f"median_{key}_failed_excluded"] = (float(np.median([r[mode][key] for r in good])) if good else None)
    result = {"schema": "delphi.spacecraft-scores.v1", "scope": "exploratory; no post-score variants",
              "cohort_lock_sha256": sha(out / "lock/cohort-lock.json"),
              "reference_poles_sha256": sha(reference_path), "predictions_manifest_sha256": sha(out / "predictions/manifest.json"),
              "primary_policy": "three combined new-data inputs, three unseen DAMIT objects, three existing OOF objects",
              "damit_reference_policy": "all frozen model-table poles; oracle over poles, ignoring sign",
              "objects": objects, "primary_medians": medians}
    lines = ["# Spacecraft poles: exploratory results", "",
             "Primary: combined new-data inputs, unseen DAMIT inputs, and existing OOF rescoring. Mathilde excluded.",
             "All new predictions use 25 frozen networks. Eros/Ceres/Vesta retain their original five-model OOF predictions.",
             "Errors are axial degrees. Oracle uses all three candidates; best uses the largest network score.",
             "Failures count as 90 degrees and are also excluded in the primary medians. No rescue or post-score changes.",
             "", "| Run | Primary | Period class | Own P (h) | Catalogue P (h) | Own oracle | Own best | Catalogue oracle | Catalogue best | DAMIT own oracle / best | DAMIT catalogue oracle / best |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    def fmt(value):
        return "n/a" if value is None else f"{value:.6f}"
    for row in objects:
        a, b = row["own"], row["catalogue"]
        cells = [row["run_id"], "yes" if row["primary"] else "secondary", row["period_class"],
                 fmt(a["period_hours"]), fmt(b["period_hours"]), fmt(a["oracle_deg"]), fmt(a["best_scored_deg"]),
                 fmt(b["oracle_deg"]), fmt(b["best_scored_deg"]),
                 f"{fmt(a['damit_oracle_deg'])} / {fmt(a['damit_best_scored_deg'])}",
                 f"{fmt(b['damit_oracle_deg'])} / {fmt(b['damit_best_scored_deg'])}"]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "Primary medians (degrees; no other averages):", "", "```json",
              json.dumps(medians, indent=2, allow_nan=False), "```", "",
              f"Reference SHA-256: `{sha(reference_path)}`.",
              f"Cohort lock SHA-256: `{sha(out / 'lock/cohort-lock.json')}`.",
              "Kernel URLs and hashes and all per-object statuses are retained in scores.json."]
    write(out / "scores.json", result)
    with (out / "RESULTS.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    return {"stage": "score", "objects": len(objects), "primary_objects": len(primary)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    handlers = {"lock": cmd_lock, "geometry": cmd_geometry, "sbdb": cmd_sbdb,
                "prepare": cmd_prepare, "search": cmd_search, "predict": cmd_predict,
                "score": cmd_score, "validate": cmd_validate}
    for name, handler in handlers.items():
        command = commands.add_parser(name)
        command.add_argument("--output-dir", type=Path, default=OUTPUT)
        command.set_defaults(handler=handler)
        if name == "lock":
            command.add_argument("--raw-dir", type=Path, help="original extractions; default --output-dir")
            command.add_argument("--snapshot-dir", type=Path, default=SNAPSHOT)
            command.add_argument("--oof-dir", type=Path, default=OWN)
        if name in ("predict", "validate"):
            command.add_argument("--device", choices=("cpu", "cuda"), default="cpu" if name == "validate" else "cuda")
            command.add_argument("--threads", type=int, default=4)
        if name == "validate":
            command.add_argument("--object-id", default="mpc:26628",
                                 help="published adapter-check object (CPU-stable candidates)")
        if name in ("geometry", "sbdb"):
            command.add_argument("--timeout-seconds", type=float, default=120 if name == "geometry" else 60)
        if name == "geometry":
            command.add_argument("--batch-size", type=int, default=20)
            command.add_argument("--request-interval-seconds", type=float, default=0.2)
    args = parser.parse_args(argv)
    args.output_dir = args.output_dir.resolve()
    if hasattr(args, "threads") and args.threads < 1:
        parser.error("--threads must be positive")
    result = args.handler(args)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
