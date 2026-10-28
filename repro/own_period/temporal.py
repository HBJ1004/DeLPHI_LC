"""Own-period version of the three published temporal cases.

Run search, predict --damit-periods in a scratch directory, predict --check-dir
SCRATCH, then score. GPU commands must use runners/lock_run.sh. All outputs
are new; interrupted output directories must be inspected before retrying.

Published preparation: prepare_temporal_damit_inputs (prepare-temporal-damit).
Published prediction: predict_prepared_ztf_object (predict-ztf-external), policy
post_cutoff_damit_record, all 25 frozen networks. Own-period ZTF uses
predict_external: _load_prepared -> KnownPeriod override -> _model_inputs ->
score_axial_grid -> ensemble_score_grids -> modes_from_score_grid ->
refine_ensemble_axes. We use the published predictor unchanged, applying the
period manifest to the already-frozen temporal preparation. Only period and
its provenance change; all epochs, uncertainties and geometry remain exact.
The mapped-ZTF period-row schema is retained, with temporal MPC identities
and no held-out fold (these cases use all folds). Fink preparation itself is
inapplicable to these DAMIT lightcurves.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import shlex
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from lc_pipeline.physics.axial import axial_angular_error_deg

REPO = Path(__file__).resolve().parents[2]
DATA = REPO.parent / "data"
OUTPUT = DATA / "own-period-20261021/temporal"
PREPARED = DATA / "temporal-prepared-final"
PUBLISHED = DATA / "temporal-predictions-final"
PUBLISHED_SCORES = DATA / "temporal-scores-final"
REFERENCES = DATA / "temporal-references-final.json"
MODELS = REPO.parent / "inputs/frozen-artifacts/k3-definitive-7874092/models"
MODEL_MANIFEST = DATA / "external-models-final.json"
SPLITS = REPO.parent / "inputs/training-run/repro/data/damit-20250610T000301Z/publication-splits-v2.3.json"
OBJECT_IDS = ("asteroid_49", "asteroid_279", "asteroid_366")
PROVENANCE = "own-fourier-gpu-top1"
POLICY = "post_cutoff_damit_record"
PREDICTION_SCHEMA = "delphi.k3-ztf-external-prediction.v1"
PERIOD_SCHEMA = "delphi.k3-mapped-ztf-period-manifest.v1"
CHECK_TOL_DEG = 1e-2  # same inputs and candidates; GPU and CPU here differ from the published run by up to 0.007 deg (floating point)
TOL = 1e-8


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preflight(paths) -> None:
    for path in paths:
        if os.path.lexists(path):
            raise FileExistsError(f"refusing to overwrite {path}")


def save_new(outputs: dict[Path, object]) -> None:
    preflight(outputs)
    texts = {path: value if isinstance(value, str) else
             json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
             for path, value in outputs.items()}
    for path, value in texts.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            stream.write(value)


def require_gpu_lock() -> None:
    """Same inherited-fd/conflicting-flock check as refine_chain_post."""
    path = Path(os.environ.get("LOCK", "/home/hangbin9/MERCUDA-paper/.gpu.lock"))
    try:
        inherited, actual = os.fstat(9), path.stat()
        if (inherited.st_dev, inherited.st_ino) != (actual.st_dev, actual.st_ino):
            raise ValueError("fd 9 does not refer to the shared GPU lock")
        with path.open("rb") as probe:
            try:
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return
            fcntl.flock(probe, fcntl.LOCK_UN)
    except (OSError, ValueError):
        pass
    command = [str(DATA / "own-period-20261021/runners/lock_run.sh"), "--",
               sys.executable, "-m", "repro.own_period.temporal", *sys.argv[1:]]
    raise RuntimeError("GPU work requires runners/lock_run.sh: " + shlex.join(command))


def source_objects(directory: Path) -> tuple[dict, dict[str, Path]]:
    manifest = read(directory / "manifest.json")
    rows = manifest["objects"]
    if (manifest.get("schema") != "delphi.k3-temporal-prepared-manifest.v1"
            or len(rows) != 3 or {r["object_id"] for r in rows} != set(OBJECT_IDS)):
        raise ValueError("expected exactly the three published temporal objects")
    paths = {}
    for row in rows:
        relative = Path(row["path"])
        path = (directory / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(directory.resolve()):
            raise ValueError("prepared path escapes input directory")
        if sha256(path) != row["sha256"]:
            raise ValueError(f"prepared-object hash mismatch: {row['object_id']}")
        paths[row["object_id"]] = path
    return manifest, paths


def valid_period(value) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0)


def load_own_periods(path: Path, prepared_dir: Path) -> dict:
    value = read(path)
    if (value.get("schema") != "delphi.own-periods.v1"
            or value.get("cohort") != "temporal" or value.get("provenance") != PROVENANCE
            or value.get("prepared_manifest_sha256") != sha256(prepared_dir / "manifest.json")
            or value.get("search_source_sha256") != sha256(REPO / "repro/period_search/search.py")
            or set(value.get("periods_hours", {})) != set(OBJECT_IDS)):
        raise ValueError("own periods do not match the temporal search inputs")
    for period in value["periods_hours"].values():
        if period is not None and not valid_period(period):
            raise ValueError("own period must be finite and positive, or null for failure")
    return value


def command_search(args) -> int:
    destination = args.output_dir / "own-periods-temporal.json"
    preflight([destination])
    require_gpu_lock()
    # This adapter reads only epochs; it never accesses known_period_hours.
    from repro.own_period.search_external import curves_of
    from repro.period_search.search import estimate

    _, paths = source_objects(args.prepared_dir)
    periods, details = {}, {}
    for oid in OBJECT_IDS:
        started = time.perf_counter()
        try:
            result = estimate(curves_of(paths[oid]), "fourier_gpu", resolution=1,
                              timeout=180, sparse_fallback=True)
        except TimeoutError as error:
            result = {"status": "timeout", "reason": str(error), "ranked_periods": []}
        ranked = result.get("ranked_periods") or []
        period = float(ranked[0]) if result.get("status") == "ok" and ranked else None
        if period is not None and not valid_period(period):
            raise ValueError(f"invalid top search period: {oid}")
        periods[oid] = period
        details[oid] = {"status": result.get("status"),
                        "input_regime": result.get("input_regime"),
                        "ranked_periods": ranked[:5], "seconds": time.perf_counter() - started}
        print(oid, details[oid]["status"], period, flush=True)
    save_new({destination: {"schema": "delphi.own-periods.v1", "cohort": "temporal",
                           "provenance": PROVENANCE,
                           "search_source_sha256": sha256(REPO / "repro/period_search/search.py"),
                           "prepared_manifest_sha256": sha256(args.prepared_dir / "manifest.json"),
                           "periods_hours": periods, "details": details}})
    return 0


def period_manifest(periods: dict, source_hash: str, provenance: str) -> dict:
    if set(periods) != set(OBJECT_IDS) or any(
            p is not None and not valid_period(p) for p in periods.values()):
        raise ValueError("period manifest requires three valid periods or explicit failures")
    # Failed searches have no row, matching mapped preparation's missing-period policy.
    return {"schema": PERIOD_SCHEMA, "cohort": "temporal", "policy": POLICY,
            "source_sha256": source_hash, "pole_coordinates_exported": False,
            "failed_object_ids": [oid for oid in OBJECT_IDS if periods[oid] is None],
            "objects": [{"object_id": oid, "mpc_number": int(oid.removeprefix("asteroid_")),
                         "held_out_fold": None, "known_period_hours": periods[oid],
                         "period_provenance": provenance}
                        for oid in OBJECT_IDS if periods[oid] is not None]}


def prepare_inputs(directory: Path, paths: dict[str, Path], manifest: dict) -> dict:
    """Apply a period manifest to the exact published preparation, without retokenizing."""
    preflight([directory])
    directory.mkdir(parents=True, exist_ok=False)
    rows = []
    for row in manifest["objects"]:
        oid = row["object_id"]
        value = read(paths[oid])
        if value.get("schema") != "delphi.k3-external-prepared.v1" or value["object_id"] != oid:
            raise ValueError("invalid temporal prepared object")
        value["known_period_hours"] = row["known_period_hours"]
        value["period_provenance"] = row["period_provenance"]
        path = directory / "objects" / f"{oid}.json"
        save_new({path: value})
        rows.append({"object_id": oid, "path": f"objects/{oid}.json", "sha256": sha256(path)})
    result = {"schema": "delphi.k3-temporal-prepared-manifest.v1", "objects": rows,
              "period_manifest_sha256": sha256(directory.parent / "period-manifest.json"),
              "failed_object_ids": manifest["failed_object_ids"]}
    save_new({directory / "manifest.json": result})
    return result


def published_predictions(directory: Path) -> dict:
    values = {oid: read(directory / f"{oid}.json") for oid in OBJECT_IDS}
    for oid, row in values.items():
        if row.get("object_id") != oid or row.get("schema") != PREDICTION_SCHEMA:
            raise ValueError("published prediction identity/schema mismatch")
        if row.get("status", "ok") == "ok" and (
                row.get("policy") != POLICY or row.get("model_count") != 25
                or row.get("folds") != list(range(5))):
            raise ValueError("temporal predictions require all 25 frozen models")
    return values


def context(args) -> dict:
    # No pole file or reference period is opened here.
    paths = [Path(__file__), REPO / "lc_pipeline/k3/ztf_prediction.py",
             REPO / "repro/own_period/predict_external.py", args.model_manifest, args.splits,
             args.prepared_dir / "manifest.json"]
    paths += [args.published_dir / f"{oid}.json" for oid in OBJECT_IDS]
    return {"sources": {str(p.resolve()): sha256(p) for p in paths}, "device": args.device,
            "policy": POLICY, "model_count": 25}


def checked_receipt(path: Path, expected_context: dict) -> dict:
    value = read(path)
    if (value.get("schema") != "delphi.own-period-temporal-damit-check.v1"
            or value.get("passed") is not True or value.get("context") != expected_context
            or set(value.get("errors_deg", {})) != set(OBJECT_IDS)
            or set(value.get("published_errors_deg", {})) != set(OBJECT_IDS)):
        raise ValueError("run the DAMIT-period prediction check with these inputs first")
    for oid in OBJECT_IDS:
        actual, expected = value["errors_deg"][oid], value["published_errors_deg"][oid]
        if (not all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 90
                    for v in (actual, expected)) or abs(actual - expected) > CHECK_TOL_DEG):
            raise ValueError("DAMIT-period reproduction check failed")
    return value


def oracle_error(prediction: dict, references) -> float | None:
    if prediction.get("status", "ok") != "ok":
        return None
    if prediction.get("schema") != PREDICTION_SCHEMA:
        raise ValueError("unexpected prediction schema")
    axes = np.asarray([r["axis_xyz"] for r in prediction["axes"]], dtype=float)
    targets = np.asarray(references, dtype=float)
    if (axes.shape != (3, 3) or targets.ndim != 2 or targets.shape[0] < 1
            or targets.shape[1] != 3 or not np.isfinite(axes).all()
            or not np.isfinite(targets).all() or np.any(np.linalg.norm(axes, axis=1) == 0)
            or np.any(np.linalg.norm(targets, axis=1) == 0)):
        raise ValueError("invalid prediction/reference axes")
    return float(np.min(axial_angular_error_deg(axes[:, None, :], targets[None, :, :])))


def check_predictions(args, predictions: dict, expected_context: dict) -> dict:
    references = read(args.references)
    actual, expected = {}, {}
    for oid in OBJECT_IDS:
        actual[oid] = oracle_error(predictions[oid], references[oid])
        published_path = args.published_dir / f"{oid}.json"
        score = read(args.published_scores_dir / f"{oid}.json")
        if (score.get("object_id") != oid
                or score.get("schema") != "delphi.k3-ztf-external-score.v1"
                or score.get("prediction_sha256") != sha256(published_path)):
            raise ValueError("published score is not bound to the published prediction")
        expected[oid] = score["oracle_at_3_error_deg"]
        if actual[oid] is None or abs(actual[oid] - expected[oid]) > CHECK_TOL_DEG:
            raise ValueError(f"DAMIT reproduction failed for {oid}: {actual[oid]} != {expected[oid]}")
    return {"schema": "delphi.own-period-temporal-damit-check.v1", "passed": True,
            "context": expected_context, "errors_deg": actual, "published_errors_deg": expected,
            "tolerance_deg": CHECK_TOL_DEG, "references_sha256": sha256(args.references)}


def command_predict(args) -> int:
    require_gpu_lock()
    names = ("period-manifest.json", "prepared", "predictions", "prediction-run.json",
             "damit-check.json", "scores.json", "RESULTS.md")
    preflight(args.output_dir / name for name in names)
    _, paths = source_objects(args.prepared_dir)
    baseline = published_predictions(args.published_dir)
    expected_context = context(args)
    if args.damit_periods:
        periods = {oid: read(paths[oid])["known_period_hours"] for oid in OBJECT_IDS}
        provenance = "published-DAMIT-period-reproduction-check"
        source_hash = sha256(args.prepared_dir / "manifest.json")
        receipt = None
    else:
        if args.check_dir is None:
            raise ValueError("predict requires --check-dir from a successful --damit-periods run")
        receipt = checked_receipt(args.check_dir / "damit-check.json", expected_context)
        own_path = args.output_dir / "own-periods-temporal.json"
        periods = load_own_periods(own_path, args.prepared_dir)["periods_hours"]
        provenance, source_hash = PROVENANCE, sha256(own_path)
    manifest = period_manifest(periods, source_hash, provenance)
    # The actual published routine writes in a private temporary directory;
    # promotion to user outputs always uses exclusive create.
    from lc_pipeline.k3.ztf_prediction import predict_prepared_ztf_object

    save_new({args.output_dir / "period-manifest.json": manifest})
    prepared = prepare_inputs(args.output_dir / "prepared", paths, manifest)
    prepared_paths = {r["object_id"]: args.output_dir / "prepared" / r["path"]
                      for r in prepared["objects"]}
    predictions = {}
    for oid in OBJECT_IDS:
        status = baseline[oid].get("status", "ok")
        if status != "ok" or periods[oid] is None:
            row = {"schema": PREDICTION_SCHEMA, "object_id": oid,
                   "status": status if status != "ok" else "no_period"}
        else:
            with tempfile.TemporaryDirectory(prefix="delphi-temporal-") as scratch:
                row = predict_prepared_ztf_object(
                    prepared_paths[oid], args.models, args.splits, Path(scratch) / "prediction.json",
                    model_manifest_path=args.model_manifest, policy=POLICY, device=args.device)
            if (row["checkpoint_bindings"] != baseline[oid]["checkpoint_bindings"]
                    or row["model_manifest_sha256"] != baseline[oid]["model_manifest_sha256"]):
                raise ValueError("temporal checkpoint selection differs from published predictions")
        save_new({args.output_dir / "predictions" / f"{oid}.json": row})
        predictions[oid] = row
        print(oid, row.get("status", "ok"), flush=True)
    if args.damit_periods:
        receipt = check_predictions(args, predictions, expected_context)
    save_new({args.output_dir / "damit-check.json": receipt,
              args.output_dir / "prediction-run.json": {
                  "schema": "delphi.own-period-temporal-prediction-run.v1",
                  "damit_periods": args.damit_periods, "context": expected_context,
                  "period_manifest_sha256": sha256(args.output_dir / "period-manifest.json"),
                  "prepared_manifest_sha256": sha256(args.output_dir / "prepared/manifest.json"),
                  "prediction_sha256": {oid: sha256(args.output_dir / "predictions" / f"{oid}.json")
                                        for oid in OBJECT_IDS}}})
    return 0


def summary(errors) -> dict:
    values = np.asarray(errors, dtype=float)
    return {"n": len(values), "mean_deg": float(values.mean()) if len(values) else None,
            "median_deg": float(np.median(values)) if len(values) else None,
            "within_20_percent": float(100 * np.mean(values <= 20 + TOL)) if len(values) else None,
            "within_30_percent": float(100 * np.mean(values <= 30 + TOL)) if len(values) else None}


def make_scores(periods: dict, predictions: dict, references: dict, damit_periods: dict) -> dict:
    rows, all_errors, valid = [], [], []
    for oid in OBJECT_IDS:
        if not valid_period(damit_periods[oid]):
            raise ValueError("invalid DAMIT scoring period")
        period = periods[oid]
        error = oracle_error(predictions[oid], references[oid]) if period is not None else None
        if error is not None:
            valid.append(error)
        all_errors.append(90.0 if error is None else error)
        rows.append({"object_id": oid, "own_period_hours": period,
                     "damit_period_hours": damit_periods[oid],
                     "period_ratio_to_damit": period / damit_periods[oid] if period is not None else None,
                     "status": (predictions[oid].get("status", "ok") if period is not None else "no_period"),
                     "oracle_at_3_error_deg": error,
                     "error_deg_failures_at_90": all_errors[-1]})
    return {"schema": "delphi.own-period-temporal-scores.v1",
            "scope": "exploratory revision analysis; three published temporal cases",
            "failure_rule": "no period or published failure: 90 degrees and also excluded",
            "objects": rows, "failed": len(all_errors) - len(valid),
            "failures_at_90": summary(all_errors), "failures_excluded": summary(valid)}


def results_markdown(scores: dict) -> str:
    def fmt(x):
        return "NA" if x is None else f"{x:.6f}"

    lines = ["# Own-period temporal cases", "", scores["scope"], "", scores["failure_rule"], "",
             "| Object | Own period (h) | Own / DAMIT | Oracle error (deg) | Status |",
             "|---|---:|---:|---:|---|"]
    for row in scores["objects"]:
        lines.append(f"| {row['object_id']} | {fmt(row['own_period_hours'])} | "
                     f"{fmt(row['period_ratio_to_damit'])} | {fmt(row['oracle_at_3_error_deg'])} | {row['status']} |")
    for mode in ("failures_at_90", "failures_excluded"):
        value = scores[mode]
        lines += ["", f"{mode}: n={value['n']}, mean={fmt(value['mean_deg'])} deg, "
                  f"median={fmt(value['median_deg'])} deg; <=20: {fmt(value['within_20_percent'])}%; "
                  f"<=30: {fmt(value['within_30_percent'])}%."]
    return "\n".join(lines) + "\n"


def command_score(args) -> int:
    preflight([args.output_dir / "scores.json", args.output_dir / "RESULTS.md"])
    run = read(args.output_dir / "prediction-run.json")
    if (run.get("schema") != "delphi.own-period-temporal-prediction-run.v1"
            or run.get("damit_periods") is not False or run.get("context") != context(args)):
        raise ValueError("score requires a completed own-period prediction run with these inputs")
    receipt = checked_receipt(args.output_dir / "damit-check.json", run["context"])
    _, paths = source_objects(args.prepared_dir)
    periods = load_own_periods(args.output_dir / "own-periods-temporal.json", args.prepared_dir)["periods_hours"]
    manifest_path = args.output_dir / "period-manifest.json"
    if (sha256(manifest_path) != run["period_manifest_sha256"]
            or read(manifest_path) != period_manifest(periods, sha256(args.output_dir / "own-periods-temporal.json"), PROVENANCE)
            or sha256(args.output_dir / "prepared/manifest.json") != run["prepared_manifest_sha256"]):
        raise ValueError("prediction periods/preparation changed")
    predictions = {}
    for oid in OBJECT_IDS:
        path = args.output_dir / "predictions" / f"{oid}.json"
        if sha256(path) != run["prediction_sha256"][oid]:
            raise ValueError("prediction changed after completion")
        predictions[oid] = read(path)
        if predictions[oid].get("object_id") != oid:
            raise ValueError("prediction identity mismatch")
    # Only scoring opens reference axes and reads the published periods.
    if sha256(args.references) != receipt["references_sha256"]:
        raise ValueError("scoring references differ from DAMIT check")
    scores = make_scores(periods, predictions, read(args.references),
                         {oid: read(paths[oid])["known_period_hours"] for oid in OBJECT_IDS})
    scores["inputs"] = {"prediction_run_sha256": sha256(args.output_dir / "prediction-run.json"),
                        "references_sha256": sha256(args.references), "damit_check": receipt}
    save_new({args.output_dir / "scores.json": scores,
              args.output_dir / "RESULTS.md": results_markdown(scores)})
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, function in (("search", command_search), ("predict", command_predict), ("score", command_score)):
        command = commands.add_parser(name)
        command.set_defaults(function=function)
        command.add_argument("--output-dir", type=Path, default=OUTPUT)
        command.add_argument("--prepared-dir", type=Path, default=PREPARED)
        if name != "search":
            command.add_argument("--published-dir", type=Path, default=PUBLISHED)
            command.add_argument("--model-manifest", type=Path, default=MODEL_MANIFEST)
            command.add_argument("--splits", type=Path, default=SPLITS)
            command.add_argument("--device", default="cuda")
            command.add_argument("--references", type=Path, default=REFERENCES)
        if name == "predict":
            command.add_argument("--models", type=Path, default=MODELS)
            command.add_argument("--published-scores-dir", type=Path, default=PUBLISHED_SCORES)
            mode = command.add_mutually_exclusive_group()
            mode.add_argument("--damit-periods", action="store_true")
            mode.add_argument("--check-dir", type=Path)
    args = parser.parse_args(argv)
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
