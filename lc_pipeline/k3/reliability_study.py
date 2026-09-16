"""Isolated, resumable retrospective sampling and withheld-observation study.

Prediction and fit stages never open evaluation reference axes. The supplied
period is historical metadata, so this is not independent period validation.
"""
from __future__ import annotations

import concurrent.futures
import contextlib
import fcntl
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

SCHEMA = "delphi.k3-reliability-study.v1"
SEEDS = (20260915, 20260916, 20260917)
PERIOD_MULTIPLIERS = (0.9999, 1.0001, 0.999, 1.001, 0.99, 1.01, 0.5, 2.0)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def binding(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": digest(path)}


def verify(value):
    path = Path(value["path"])
    if digest(path) != value["sha256"]:
        raise ValueError(f"bound input changed: {path}")
    return path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"


def write_json(path, value, *, replace=False):
    """Atomic write; never silently overwrite a completed scientific record."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(path)
    fd, name = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(canonical(value))
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(name, path)
        else:
            os.link(name, path)
            os.unlink(name)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def seal(path, value):
    checksum = hashlib.sha256(canonical(value).encode()).hexdigest()
    write_json(path, {"seal_schema": "sha256-canonical-json-v1", "payload": value,
                      "payload_sha256": checksum})


def unseal(path):
    document = read_json(path)
    if (document.get("seal_schema") != "sha256-canonical-json-v1" or
            hashlib.sha256(canonical(document["payload"]).encode()).hexdigest() != document["payload_sha256"]):
        raise ValueError(f"completed record checksum changed: {path}")
    return document["payload"]


@contextlib.contextmanager
def exclusive(root):
    """Process lock survives a dead terminal and releases if the process dies."""
    path = Path(root) / "run.lock"
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("another reliability process holds the run lock") from exc
        yield


class Budget:
    """Persisted wall deadlines never reset after interruption.

    Unattended downtime is conservatively charged to the interrupted phase.
    An operator cannot accidentally receive a fresh budget by restarting.
    """
    def __init__(self, root, phase, seconds):
        self.path = Path(root) / f"budget-{phase}.json"
        if self.path.exists():
            self.state = unseal(self.path)
            if self.state["limit_seconds"] != seconds:
                raise ValueError("cannot change an existing phase budget")
        else:
            now = time.time()
            self.state = {"phase": phase, "start_epoch": now,
                          "deadline_epoch": now + seconds, "limit_seconds": seconds}
            seal(self.path, self.state)

    def remaining(self):
        deadline = self.state["deadline_epoch"]
        total = self.path.parent / "budget-total.json"
        if total.exists() and total != self.path:
            deadline = min(deadline, unseal(total)["deadline_epoch"])
        return max(0.0, deadline - time.time())

    def check(self, reserve=0):
        if self.remaining() <= reserve:
            raise TimeoutError(f"{self.state['phase']} computation budget exhausted")


def stable_key(oid, salt="reliability-20260915"):
    return hashlib.sha256(f"{salt}:{oid}".encode()).hexdigest()


def balanced_order(rows):
    groups = {fold: sorted((r for r in rows if r["fold"] == fold),
                           key=lambda r: stable_key(r["object_id"])) for fold in range(5)}
    return [groups[fold][index]["object_id"] for index in range(max(map(len, groups.values())))
            for fold in range(5) if index < len(groups[fold])]


def pilot_ids(rows, development_ids, *, holdout=False):
    selected = []
    for fold in range(5):
        candidates = [r for r in rows if r["fold"] == fold and r["object_id"] in development_ids
                      and (not holdout or r["holdout_eligible"])]
        ordered = sorted(candidates, key=lambda r: (r["n_points"], r["object_id"]))
        if len(ordered) < 2:
            raise ValueError("each pilot fold needs two eligible development objects")
        selected.extend([ordered[(len(ordered) - 1) // 2]["object_id"], ordered[-1]["object_id"]])
    return selected


def choose_schedule(inference_seconds_by_object, inference_rows, holdout_seconds, eligible_order,
                    inference_projection_by_repeat=None):
    """Choose sample size from runtimes only; no scores are accepted here."""
    if len(inference_seconds_by_object) != 10 or len(holdout_seconds) != 10:
        raise ValueError("schedule requires all ten technical pilot objects per stage")
    maximum = max(inference_seconds_by_object.values())
    # Pilot measures every condition at one repeat, including fixed period checks.
    estimates = ({r: 2 * maximum * len(inference_rows) * r for r in (3, 1)}
                 if inference_projection_by_repeat is None else
                 {r: float(inference_projection_by_repeat[str(r)]) for r in (3, 1)})
    repeats = next((r for r in (3, 1) if estimates[r] <= 10 * 3600), None)
    max_fit = 2 * max(holdout_seconds.values())
    choices = list(dict.fromkeys([len(eligible_order), 125, 100, 75, 50, 30]))
    count = next((n for n in choices if n <= len(eligible_order) and n * max_fit <= 11 * 3600), None)
    if repeats is None or count is None:
        raise ValueError("complete registered study is infeasible under the computation budget")
    return {"repeats": repeats, "inference_estimate_seconds": estimates[repeats],
            "holdout_count": count, "holdout_ids": eligible_order[:count],
            "holdout_estimate_seconds": count * max_fit,
            "selection_uses": "technical runtime only; no reference or withheld scores"}


def project_inference(root, lock):
    """Per-condition pilot maxima, with a global maximum for unmeasured cells."""
    pilot_cells = [unseal(p) for p in (Path(root) / "pilot" / "predictions").glob("*/*/repeat-0.json")]
    if not pilot_cells or any(r["status"] != "ok" for r in pilot_cells):
        raise ValueError("technical inference pilot contains a failure; fix before scientific execution")
    maxima = {}
    for cell in pilot_cells:
        maxima[cell["condition"]] = max(maxima.get(cell["condition"], 0), cell["runtime_s"])
    global_maximum = max(maxima.values())
    estimates = {}
    for repeats in (3, 1):
        total = 0.0
        for config in lock["conditions"]:
            n = len(lock["grid_ids"]) if config["family"] == "two_d" else len(lock["objects"])
            total += n * (repeats if config["repeatable"] else 1) * maxima.get(config["condition"], global_maximum)
        # Include a conservative allowance for loading five folds and all file writes.
        estimates[str(repeats)] = 2 * total + 300
    return {"estimates_seconds": estimates, "condition_max_seconds": maxima,
            "unmeasured_condition_fallback_seconds": global_maximum}


def runtime_sources(root):
    root = Path(root)
    return {str(p.relative_to(root)): digest(p)
            for folder in ("lc_pipeline", "repro") for p in sorted((root / folder).rglob("*.py"))}


def verify_study(root):
    lock = unseal(Path(root) / "study.json")
    for item in lock["bindings"]:
        verify(item)
    for row in lock["objects"]:
        verify(row["lightcurve"])
    for b in lock["bundles"]:
        for file in b["files"]:
            verify(file)
    if runtime_sources(Path(__file__).resolve().parents[2]) != lock["runtime_sources"]:
        raise ValueError("runtime source changed; resume the sealed source snapshot")
    import importlib.metadata
    if {k: importlib.metadata.version(k) for k in lock["environment"]} != lock["environment"]:
        raise ValueError("installed runtime package versions changed")
    return lock


def checkpoint_read(path):
    path = Path(path)
    if path.exists():
        return unseal(path)
    return None


def status(root):
    root = Path(root)
    result = {"root": str(root), "pid": None}
    if (root / "progress.json").exists():
        result.update(read_json(root / "progress.json"))
    if (root / "pilot" / "progress.json").exists():
        pilot = read_json(root / "pilot" / "progress.json")
        if pilot["updated_epoch"] > result.get("updated_epoch", 0):
            result.update(pilot)
    if result.get("pid"):
        try:
            os.kill(result["pid"], 0)
            result["process_alive"] = True
        except ProcessLookupError:
            result["process_alive"] = False
    result["completed_prediction_files"] = sum(1 for _ in (root / "predictions").glob("*/*/*.json")
                                                if not _.name.endswith("sha256.json"))
    result["completed_fit_cells"] = sum(1 for _ in (root / "fits").glob("*/*/*/result.json"))
    return result


def start_background(root):
    """Launch the sealed source outside a terminal's process session."""
    root = Path(root).resolve()
    lock = unseal(root / "study.json")
    executable = verify(lock["python"])
    with (root / "launch.lock").open("a+") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "launch.json").exists():
            prior = read_json(root / "launch.json")
            try:
                os.kill(prior["pid"], 0)
                return {**prior, "already_running": True}
            except ProcessLookupError:
                pass
        command = [str(executable), "-m", "repro.run_k3_reliability_study", "run", "--root", str(root)]
        inherited = {k: v for k, v in os.environ.items() if k in
                     ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "LD_LIBRARY_PATH",
                      "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "CUBLAS_WORKSPACE_CONFIG")}
        environment = {**inherited, "PYTHONPATH": lock["source_snapshot"],
                       "OMP_NUM_THREADS": "6", "MKL_NUM_THREADS": "6", "OPENBLAS_NUM_THREADS": "6"}
        with (root / "run.log").open("ab", buffering=0) as log:
            process = subprocess.Popen(command, cwd=lock["source_snapshot"], env=environment,
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        receipt = {"pid": process.pid, "launched_epoch": time.time(), "command": command,
                   "log": str(root / "run.log"), "detached_session": True}
        write_json(root / "launch.json", receipt, replace=True)
        progress(root, stage="starting", pid=process.pid)
        return receipt


def progress(root, **values):
    write_json(Path(root) / "progress.json", {"pid": os.getpid(), "updated_epoch": time.time(),
                                             **values}, replace=True)


def conditions():
    from .reliability_sampling import condition_configs
    result = []
    for config in condition_configs():
        kind = config["kind"]
        name = kind
        for key in ("fraction", "cap", "block_count", "per_block_cap", "noise_sigma"):
            if key in config:
                name += f"-{key}-{config[key]}"
        result.append({"condition": name, "family": kind, "sampling": config,
                       "period_multiplier": 1.0, "repeatable": kind != "full" or "noise_sigma" in config})
    result.extend({"condition": f"period-{p}", "family": "period", "sampling": {"kind": "full"},
                   "period_multiplier": p, "repeatable": False} for p in PERIOD_MULTIPLIERS)
    return result


def baselines_by_condition(configurations):
    result = {}
    for config in configurations:
        if config["condition"] == "full":
            continue
        sampling = config["sampling"]
        baseline = "full"
        if config["family"] == "two_d":
            baseline = "grid-full"
        elif sampling.get("noise_sigma") is not None and sampling["kind"] == "point":
            baseline = next(c["condition"] for c in configurations if c["sampling"]["kind"] == "point"
                            and c["sampling"].get("fraction") == sampling["fraction"]
                            and c["sampling"].get("noise_sigma") is None)
        elif sampling["kind"] == "observation_cap" and sampling["cap"] is not None:
            baseline = next(c["condition"] for c in configurations if c["sampling"]["kind"] == "observation_cap"
                            and c["sampling"]["cap"] is None)
        result[config["condition"]] = baseline
    return result


def input_metadata(curve, period):
    from .reliability_sampling import sampling_metadata
    value = sampling_metadata(curve)
    obs = curve.observations
    times = np.asarray([o.time_jd for o in obs])
    flux = np.asarray([o.flux for o in obs])
    if not len(times):
        raise ValueError("sampling left no usable observations")
    phases = ((times - times.min()) * 24 / period) % 1
    geometry = np.asarray([o.observer for o in obs])
    geometry /= np.linalg.norm(geometry, axis=1, keepdims=True)
    # Directional second moment is sign/rotation independent; no pole labels enter.
    moments = np.linalg.eigvalsh(geometry.T @ geometry / len(geometry))
    value.update(n_points=len(obs), n_sessions=curve.native_sessions,
                 time_span_days=float(np.ptp(times)),
                 occupied_phase_bins=int(len(np.unique(np.minimum(63, (phases * 64).astype(int))))),
                 phase_coverage_fraction=float(len(np.unique((phases * 64).astype(int))) / 64),
                 observer_second_moment_eigenvalues=moments.tolist(),
                 flux_cv=float(np.std(flux) / np.mean(flux)))
    return value


def prepare(*, root, previous_lock, exposure_audit, solver_report, protocol, source_root):
    """Freeze inputs and training-only atlases, then snapshot code for execution."""
    import torch

    from repro.audit_k3_original_synthetic_overlap import build_overlap_report
    from repro.build_k3_external_atlas import _placeholder

    from ..v2.atlas import fit_axis_atlas, write_axis_atlas
    from ..v2.data import load_catalog
    from .reliability_sampling import ReliabilitySamplingError, read_lc, select_holdout
    torch.set_num_threads(1)

    root, source_root = Path(root).resolve(), Path(source_root).resolve()
    if root.exists():
        raise FileExistsError("study output must be a new directory")
    old = read_json(previous_lock)
    splits_path = verify(old["inputs"]["splits"])
    folds = read_json(splits_path)["folds"]
    catalog_path = verify(old["inputs"]["reference_catalog"])
    development = set(read_json(verify(old["inputs"]["development"]))["object_ids"])
    exposure = read_json(exposure_audit)
    sources = exposure["sources"]
    aliases_path = verify(sources["identity_aliases"])
    overlap = build_overlap_report(original_splits=splits_path, identity_aliases=aliases_path,
                                  published_metadata_root=Path(sources["published_synthetic_manifests"]["root"]))
    if not all(overlap["checks"].values()) or overlap["all_overlap_physical_identities"]:
        raise ValueError("synthetic donor identity exclusion failed")
    report = read_json(solver_report)
    if not all(report.get("native_parity", {}).get(k) for k in
               ("stdout_identical", "stderr_identical", "native_modelled_identical")):
        raise ValueError("solver output-only patch requires technical trajectory parity evidence")
    if digest(report["binary"]) != report["binary_sha256"]:
        raise ValueError("prepared solver binary changed")
    seen = set()
    for fold in folds:
        roles = [set(fold[role]) for role in ("train_ids", "validation_ids", "calibration_ids", "test_ids")]
        if list(map(len, roles)) != [108, 14, 14, 34] or sum(map(len, roles)) != len(set.union(*roles)):
            raise ValueError("split roles are not disjoint with registered sizes")
        if seen & roles[-1]:
            raise ValueError("duplicate out-of-fold object")
        seen |= roles[-1]
    if len(seen) != 170 or {r["object_id"] for r in old["objects"]} != seen:
        raise ValueError("cohort differs from original170")
    for b in old["bundles"]:
        for file in b["files"]:
            verify(file)
        manifest = read_json(Path(b["directory"]) / "bundle.json")
        if manifest["object_roles"] != {k: folds[manifest["fold"]][k] for k in
                                        ("train_ids", "validation_ids", "calibration_ids", "test_ids")}:
            raise ValueError("ensemble roles differ from splits")
    rows = []
    for old_row in old["objects"]:
        row = {k: old_row[k] for k in ("object_id", "fold", "period_hours", "lightcurve")}
        if row["object_id"] not in folds[row["fold"]]["test_ids"]:
            raise ValueError("wrong fold bound to input object")
        curve = read_lc(verify(row["lightcurve"]), object_id=row["object_id"], period_hours=row["period_hours"])
        meta = input_metadata(curve, row["period_hours"])
        if meta["n_points"] != old_row["structure"]["total_observations"]:
            raise ValueError("native adapter lost original observations")
        # Technical parity without reading labels or making any new prediction.
        import torch

        from ..v2.preprocessing import KnownPeriod
        from .bundle import _model_inputs
        from .damit import _read_lc
        from .reliability_sampling import to_epochs
        period = KnownPeriod(hours=row["period_hours"], provenance="adapter-parity-only")
        expected = _model_inputs(_read_lc(Path(row["lightcurve"]["path"])), period, torch.device("cpu"))
        actual = _model_inputs(to_epochs(curve), period, torch.device("cpu"))
        if any(not torch.equal(actual[key], expected[key]) for key in expected):
            raise ValueError("unmodified native-adapter feature parity failed")
        row.update(meta)
        try:
            holdout = select_holdout(curve)
            row.update(holdout_eligible=True, retained_points=holdout.train.total_observations,
                       withheld_points=holdout.withheld.total_observations)
        except ReliabilitySamplingError as exc:
            row.update(holdout_eligible=False, holdout_reason=str(exc))
        rows.append(row)
    root.mkdir(parents=True)
    Budget(root, "total", 24 * 3600)
    seal(root / "donor-check.json", overlap)
    catalog = load_catalog(catalog_path)
    atlases = []
    for fold in folds:
        values = [_placeholder(oid, catalog[oid].solution_vectors, catalog[oid].solution_periods_hours)
                  for oid in fold["train_ids"]]
        atlas = fit_axis_atlas(values, partition_role="train", forbidden_object_ids=fold["test_ids"])
        out = root / "atlases" / f"fold-{fold['fold']}.json"
        write_axis_atlas(out, atlas)
        atlases.append({"fold": fold["fold"], "file": binding(out), "axes": atlas.axes,
                        "source_ids": fold["train_ids"]})
    bindings = [binding(p) for p in (previous_lock, exposure_audit, solver_report, protocol, aliases_path)]
    bindings += [old["inputs"][k] for k in ("splits", "reference_catalog", "development")]
    bindings += [binding(report["binary"])] + [a["file"] for a in atlases]
    for m in sources["published_synthetic_manifests"]["manifests"]:
        bindings.append(binding(Path(sources["published_synthetic_manifests"]["root"]) / m["relative_path"]))
    snapshot = root / "runtime-source"
    snapshot.mkdir()
    for name in ("lc_pipeline", "repro"):
        shutil.copytree(source_root / name, snapshot / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(source_root / "pyproject.toml", snapshot / "pyproject.toml")
    import importlib.metadata
    environment = {k: importlib.metadata.version(k) for k in
                   ("numpy", "torch", "scipy", "astropy", "safetensors")}
    lock = {"schema": SCHEMA, "role": "retrospective_reliability_not_new_external_validation",
            "objects": rows, "bundles": old["bundles"], "atlases": atlases, "bindings": bindings,
            "catalog": binding(catalog_path), "source_snapshot": str(snapshot),
            "runtime_sources": runtime_sources(snapshot), "environment": environment,
            "python": binding(sys.executable), "device": "cuda", "conditions": conditions(),
            "pilot_ids": pilot_ids(rows, development),
            "holdout_pilot_ids": pilot_ids(rows, development, holdout=True),
            "holdout_order": balanced_order([r for r in rows if r["holdout_eligible"]]),
            "grid_ids": [r["object_id"] for r in rows if r["merged_blocks"] >= 10],
            "solver": {"executable": binding(report["binary"])},
            "solver_settings": {k: old["settings"][k] for k in
                                ("workers", "convergence_tolerance", "timeout_seconds_per_start", "iteration_cap")},
            "registered_compute_hours": {"pilot": 1, "inference": 10, "fits": 11, "score": 2}}
    seal(root / "study.json", lock)
    return {"root": str(root), "object_count": len(rows), "grid_count": len(lock["grid_ids"]),
            "holdout_eligible": len(lock["holdout_order"]), "condition_count": len(lock["conditions"])}


def selected_curve(row, config, repeat):
    from dataclasses import replace

    from .reliability_sampling import read_lc, transform_curve
    source = read_lc(verify(row["lightcurve"]), object_id=row["object_id"], period_hours=row["period_hours"])
    selection = {**config["sampling"], "seed": SEEDS[repeat], "repeat": 0}
    result = transform_curve(source, selection)
    return replace(result, period_hours=row["period_hours"] * config["period_multiplier"])


def input_hash(curve):
    payload = [{"id": s.session_id, "flag": s.calibrated,
                "points": [[o.time_jd, o.flux, *o.sun, *o.observer] for o in s.observations]}
               for s in curve.sessions]
    return hashlib.sha256(canonical({"period_hours": curve.period_hours, "sessions": payload}).encode()).hexdigest()


def predict_curve(predictor, curve, object_id):
    from ..v2.preprocessing import KnownPeriod
    from .reliability_sampling import to_epochs
    result = predictor.predict(to_epochs(curve), known_period=KnownPeriod(
        hours=curve.period_hours, provenance="supplied-historical-period-reliability-study"), object_id=object_id)
    axes = np.asarray([a["axis_xyz"] for a in result["axes"]])
    if axes.shape != (3, 3) or not np.all(np.isfinite(axes)):
        raise ValueError("prediction must contain exactly three finite axes")
    return axes.tolist()


def prediction_cell(root, row, config, repeat, predictor):
    target = Path(root) / "predictions" / row["object_id"] / config["condition"] / f"repeat-{repeat}.json"
    cached = checkpoint_read(target)
    if cached is not None:
        expected = {"object_id": row["object_id"], "condition": config["condition"], "repeat": repeat,
                    "fold": row["fold"], "seed": SEEDS[repeat], "source_sha256": row["lightcurve"]["sha256"],
                    "model_count": 5, "model_id": "k3"}
        if any(cached.get(k) != v for k, v in expected.items()):
            raise ValueError("prediction checkpoint identity mismatch")
        if cached.get("input_sha256") is not None:
            if input_hash(selected_curve(row, config, repeat)) != cached["input_sha256"]:
                raise ValueError("prediction checkpoint transformed input changed")
        return cached
    started = time.monotonic()
    value = {"object_id": row["object_id"], "fold": row["fold"], "condition": config["condition"],
             "condition_family": config["family"], "repeat": repeat, "seed": SEEDS[repeat],
             "source_sha256": row["lightcurve"]["sha256"], "model_id": "k3", "model_count": 5}
    try:
        curve = selected_curve(row, config, repeat)
        value.update(input_sha256=input_hash(curve), metadata=input_metadata(curve, curve.period_hours),
                     accumulated_phase_drift_cycles=float(
                         (max(o.time_jd for o in curve.observations) - min(o.time_jd for o in curve.observations)) *
                         24 * abs(1 / curve.period_hours - 1 / row["period_hours"])))
        axes = predict_curve(predictor, curve, row["object_id"])
        value.update(status="ok", axes=axes)
    except (ValueError, RuntimeError) as exc:
        value.update(status="failed", axes=None, failure_code=f"{type(exc).__name__}: {exc}")
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    value["runtime_s"] = time.monotonic() - started
    seal(target, value)
    return value


def inference(root, lock, ids, repeats, budget):
    import torch

    from .bundle import K3EnsemblePredictor
    torch.set_num_threads(6)
    by_id = {r["object_id"]: r for r in lock["objects"]}
    durations = {}
    for fold in range(5):
        objects = [by_id[o] for o in ids if by_id[o]["fold"] == fold]
        if not objects:
            continue
        budget.check()
        predictor = K3EnsemblePredictor(lock["bundles"][fold]["directory"], device=lock["device"])
        for row in objects:
            started = time.monotonic()
            cells = []
            for config in lock["conditions"]:
                if config["family"] == "two_d" and row["object_id"] not in lock["grid_ids"]:
                    continue
                for repeat in range(repeats if config["repeatable"] else 1):
                    budget.check()
                    progress(root, stage=budget.state["phase"], operation="inference", object_id=row["object_id"],
                             condition=config["condition"], repeat=repeat, remaining_seconds=budget.remaining())
                    cells.append(prediction_cell(root, row, config, repeat, predictor))
            # Reused pilot cells contribute their original runtime, not zero.
            durations[row["object_id"]] = max(time.monotonic() - started, sum(r["runtime_s"] for r in cells))
        del predictor
        torch.cuda.empty_cache()
    return durations


def holdout_input(root, row, sampling):
    """Return only fit input; withheld flux is never passed to candidate/fit code."""
    from .reliability_sampling import read_lc, select_holdout, transform_curve, write_lc
    curve = read_lc(verify(row["lightcurve"]), object_id=row["object_id"], period_hours=row["period_hours"])
    divided = select_holdout(curve)
    retained = divided.train
    if sampling == "thin10":
        retained = transform_curve(retained, {"kind": "point", "fraction": 0.1, "seed": SEEDS[0]})
    target = Path(root) / "fit-inputs" / row["object_id"] / sampling / "lc.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        check = read_lc(target, object_id=row["object_id"], period_hours=row["period_hours"])
        # Serialized files number retained sessions anew; compare measurements and flags.
        def measurements(c):
            return [(s.calibrated, s.observations) for s in c.sessions]
        if measurements(check) != measurements(retained):
            raise ValueError("cached fit input differs from label-blind heldout split")
    else:
        fd, temporary = tempfile.mkstemp(prefix=".pending-input-", dir=target.parent)
        os.close(fd)
        try:
            # The native writer requires a new destination.
            os.unlink(temporary)
            write_lc(retained, temporary)
            os.link(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return retained, target


def validate_start(path):
    record = read_json(path / "record.json")
    for name, checksum in record["files"].items():
        if Path(name).name != name or digest(path / name) != checksum:
            raise ValueError("native solver checkpoint product changed")
    return record


def native_start_resumable(job, index, axis, observed, lock, directory, budget):
    from ..grid_benchmark import _native_start
    target = directory / f"start-{index:03d}"
    if (target / "record.json").exists():
        return validate_start(target)
    if target.exists():
        interrupted = directory / "interrupted"
        interrupted.mkdir(exist_ok=True)
        # Preserve incomplete solver products for diagnosis; never reuse as a fit.
        target.rename(interrupted / f"{target.name}-{time.time_ns()}")
    budget.check(reserve=lock["solver_settings"]["timeout_seconds_per_start"] + 5)
    return _native_start(job, index, axis, observed, lock["solver"], lock["solver_settings"], directory)


def forward_parity(selected_folder, native_path):
    from .reliability_forward import load_solution, normalize_mean, read_lightcurves
    from .reliability_forward import predict_curve as forward
    solution = load_solution(selected_folder / "solution-parameters.txt", selected_folder / "solution-areas.txt")
    curves = read_lightcurves(native_path)
    native = np.asarray((selected_folder / "modelled-lightcurve.txt").read_text().split(), dtype=float)
    offset, maximum = 0, 0.0
    for curve in curves:
        n = len(curve.observations)
        predicted = forward(solution, curve.observations, include_phase=not curve.relative)
        difference = np.abs(normalize_mean(native[offset:offset + n]) - normalize_mean(predicted))
        maximum = max(maximum, float(difference.max()))
        offset += n
    if offset != len(native) or maximum > 1e-5:
        raise ValueError(f"forward/native normalized parity failed: {maximum}")
    return maximum


def fit_cell(root, row, sampling, arm, lock, predictor, budget):
    from ..grid_benchmark import guided_starts, pole_grid
    directory = Path(root) / "fits" / row["object_id"] / sampling / arm
    result_path = directory / "result.json"
    cached = checkpoint_read(result_path)
    if cached is not None:
        expected = {"object_id": row["object_id"], "fold": row["fold"], "sampling": sampling, "arm": arm}
        if any(cached.get(k) != v for k, v in expected.items()):
            raise ValueError("fit checkpoint identity mismatch")
        curve, native_path = holdout_input(root, row, sampling)
        if verify(cached["fit_input"]) != native_path.resolve() or cached["input_sha256"] != input_hash(curve):
            raise ValueError("fit checkpoint input changed")
        for name, checksum in cached["start_records"].items():
            if digest(directory / name) != checksum:
                raise ValueError("fit receipt changed after completion")
            validate_start((directory / name).parent)
        return cached
    directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    curve, native_path = holdout_input(root, row, sampling)
    axes = None
    if arm == "k3":
        prediction_file = directory / "prediction.json"
        prediction = checkpoint_read(prediction_file)
        if prediction is None:
            axes = predict_curve(predictor, curve, row["object_id"])
            prediction = {"axes": axes, "input_sha256": input_hash(curve), "fold": row["fold"]}
            seal(prediction_file, prediction)
        if prediction["input_sha256"] != input_hash(curve) or prediction["fold"] != row["fold"]:
            raise ValueError("holdout prediction input/fold changed")
        axes = prediction["axes"]
    elif arm == "atlas":
        axes = lock["atlases"][row["fold"]]["axes"]
    elif arm != "classical20":
        raise ValueError("unregistered fit arm")
    starts = pole_grid(20) if axes is None else guided_starts(axes, 20)
    observed = tuple(np.asarray([o.flux for o in s.observations]) for s in curve.sessions)
    job = {"period_hours": row["period_hours"], "lightcurve": binding(native_path)}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(native_start_resumable, job, index, axis, observed, lock, directory, budget)
                   for index, axis in enumerate(starts)]
        records = [future.result() for future in futures]
    fits = [r["selected_fit"] for r in records if r["selected_fit"] is not None]
    selected = min(fits, key=lambda r: (r["final_relative_rms"], r["start_index"])) if fits else None
    parity = None
    if selected:
        parity = forward_parity(directory / f"start-{selected['start_index']:03d}", native_path)
    result = {"object_id": row["object_id"], "fold": row["fold"], "sampling": sampling,
              "arm": arm, "status": "ok" if selected is not None else "failed",
              "selected_fit": selected, "starts_requested": len(starts), "valid_starts": len(fits),
              "start_records": {f"start-{r['start_index']:03d}/record.json":
                                digest(directory / f"start-{r['start_index']:03d}" / "record.json") for r in records},
              "fit_input": binding(native_path), "input_sha256": input_hash(curve),
              "metadata": input_metadata(curve, row["period_hours"]),
              "forward_native_max_difference": parity, "runtime_s": time.monotonic() - started,
              "withheld_scores_opened": False}
    seal(result_path, result)
    return result


def fitting(root, lock, ids, budget):
    import torch

    from .bundle import K3EnsemblePredictor
    torch.set_num_threads(6)
    by_id = {r["object_id"]: r for r in lock["objects"]}
    durations = {}
    for fold in range(5):
        objects = [by_id[o] for o in ids if by_id[o]["fold"] == fold]
        if not objects:
            continue
        predictor = K3EnsemblePredictor(lock["bundles"][fold]["directory"], device=lock["device"])
        for row in objects:
            started = time.monotonic()
            values = []
            cells = [(sampling, arm) for sampling in ("full", "thin10") for arm in ("classical20", "k3", "atlas")]
            cells.sort(key=lambda x: stable_key(row["object_id"] + ":" + ":".join(x)))
            for sampling, arm in cells:
                budget.check()
                progress(root, stage=budget.state["phase"], operation="fitting", object_id=row["object_id"],
                         sampling=sampling, arm=arm, remaining_seconds=budget.remaining())
                values.append(fit_cell(root, row, sampling, arm, lock, predictor, budget))
            durations[row["object_id"]] = max(time.monotonic() - started, sum(v["runtime_s"] for v in values))
        del predictor
        torch.cuda.empty_cache()
    return durations


def execute(root, *, pilot_only=False):
    root = Path(root).resolve()
    with exclusive(root):
        lock = verify_study(root)
        if binding(sys.executable) != lock["python"]:
            raise ValueError("run with the frozen Python interpreter")
        try:
            pilot = checkpoint_read(root / "pilot.json")
            if pilot is None:
                budget = Budget(root, "pilot", 3600)
                inference_times = inference(root / "pilot", lock, lock["pilot_ids"], 1, budget)
                fit_times = fitting(root / "pilot", lock, lock["holdout_pilot_ids"], budget)
                pilot = {"inference_seconds_by_object": inference_times, "holdout_seconds_by_object": fit_times,
                         "inference_projection": project_inference(root, lock),
                         "scientific_scores_opened": False}
                seal(root / "pilot.json", pilot)
            if pilot_only:
                progress(root, stage="pilot_complete", scientific_scores_opened=False)
                return pilot
            schedule = checkpoint_read(root / "schedule.json")
            if schedule is None:
                schedule = choose_schedule(pilot["inference_seconds_by_object"], lock["objects"],
                                           pilot["holdout_seconds_by_object"], lock["holdout_order"],
                                           pilot["inference_projection"]["estimates_seconds"])
                seal(root / "schedule.json", schedule)
            # Full execution keeps pilot products separate, so no development-only cell
            # is silently counted as a new result or used for condition selection.
            if not (root / "inference-complete.json").exists():
                durations = inference(root, lock, [r["object_id"] for r in lock["objects"]], schedule["repeats"],
                                      Budget(root, "inference", 10 * 3600))
                seal(root / "inference-complete.json", {"durations": durations})
            if not (root / "fits-complete.json").exists():
                durations = fitting(root, lock, schedule["holdout_ids"], Budget(root, "fits", 11 * 3600))
                seal(root / "fits-complete.json", {"durations": durations})
            score(root, lock, schedule)
            progress(root, stage="complete")
            return {"status": "complete", "root": str(root)}
        except Exception as exc:
            progress(root, stage="stopped", reason=f"{type(exc).__name__}: {exc}")
            raise


def oracle_error(axes, references):
    axes, references = np.asarray(axes, float), np.asarray(references, float)
    if axes.shape != (3, 3) or references.ndim != 2 or references.shape[1] != 3:
        raise ValueError("oracle@3 requires three axes and at least one catalog vector")
    axes = axes / np.linalg.norm(axes, axis=1, keepdims=True)
    references = references / np.linalg.norm(references, axis=1, keepdims=True)
    value = float(np.degrees(np.arccos(np.clip(np.abs(axes @ references.T).max(), 0, 1))))
    if not math.isfinite(value):
        raise ValueError("nonfinite oracle error")
    return value


def clustering(axes):
    """All candidates, equally weighted; not an oracle-selected sky map."""
    axes = np.asarray(axes, float).reshape(-1, 3)
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    eigenvalues = np.linalg.eigvalsh(axes.T @ axes / len(axes))
    return {"axis_count": len(axes), "second_moment_eigenvalues": eigenvalues.tolist(),
            "largest_second_moment_eigenvalue": float(eigenvalues[-1]),
            "interpretation": "descriptive directional concentration, not proof of memorization"}


def _write_text_new(path, text):
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(name, path)
    finally:
        os.unlink(name)


def score(root, lock, schedule):
    from ..v2.data import load_catalog
    from .reliability_forward import (
        ForwardLightcurve,
        ForwardObservation,
        holdout_shape_rmse,
        load_solution,
    )
    from .reliability_sampling import read_lc, select_holdout
    from .reliability_scoring import (
        render_reliability_csv,
        render_reliability_tex,
        score_reliability,
        stratified_asteroid_bootstrap,
    )
    root = Path(root)
    if (root / "report.json").exists():
        report = unseal(root / "report.json")
        for file in report["report_files"]:
            if digest(root / file["path"]) != file["sha256"]:
                raise ValueError("completed report export changed")
        return report
    unseal(root / "inference-complete.json")
    unseal(root / "fits-complete.json")
    budget = Budget(root, "score", 2 * 3600)
    verify_study(root)
    catalog = load_catalog(verify(lock["catalog"]))
    rows = []
    repeats_by_condition = {c["condition"]: list(range(schedule["repeats"] if c["repeatable"] else 1))
                            for c in lock["conditions"]}
    repeats_by_condition["grid-full"] = [0]
    random_axes = {}
    for entry in lock["objects"]:
        oid = entry["object_id"]
        references = catalog[oid].solution_vectors
        for config in lock["conditions"]:
            for repeat in repeats_by_condition[config["condition"]]:
                budget.check()
                if config["family"] == "two_d" and oid not in lock["grid_ids"]:
                    value = {"object_id": oid, "fold": entry["fold"], "condition": config["condition"],
                             "repeat": repeat, "seed": SEEDS[repeat], "status": "ineligible",
                             "axes": None, "error_deg": None, "reason": "fewer_than_ten_observing_blocks"}
                else:
                    path = root / "predictions" / oid / config["condition"] / f"repeat-{repeat}.json"
                    value = unseal(path)
                    if (value["object_id"], value["condition"], value["repeat"]) != (oid, config["condition"], repeat):
                        raise ValueError("prediction identity mismatch at scoring")
                    value["error_deg"] = oracle_error(value["axes"], references) if value["status"] == "ok" else 90.0
                    value["reference_solution_count"] = len(references)
                    value["reference_primary_abs_latitude_deg"] = float(np.degrees(np.arcsin(abs(references[0][2]))))
                rows.append(value)
        base = next(r for r in rows if r["object_id"] == oid and r["condition"] == "full")
        if oid in lock["grid_ids"]:
            gridbase = {**base, "condition": "grid-full", "reused_condition": "full"}
        else:
            gridbase = {"object_id": oid, "fold": entry["fold"], "condition": "grid-full", "repeat": 0,
                        "seed": SEEDS[0], "status": "ineligible", "axes": None, "error_deg": None}
        rows.append(gridbase)
        rng = np.random.default_rng(int(stable_key(oid, "random-three-20260915")[:16], 16))
        random = rng.normal(size=(1000, 3, 3))
        random /= np.linalg.norm(random, axis=2, keepdims=True)
        ref = np.asarray(references)
        ref /= np.linalg.norm(ref, axis=1, keepdims=True)
        errors = np.degrees(np.arccos(np.clip(np.max(np.abs(random @ ref.T), axis=(1, 2)), 0, 1)))
        random_axes[oid] = {"object_id": oid, "fold": entry["fold"], "mean_error_deg": float(errors.mean()),
                            "first_draw_error_deg": float(errors[0]), "within20_fraction": float(np.mean(errors <= 20)),
                            "first_draw_axes": random[0].tolist(),
                            "atlas_error_deg": oracle_error(lock["atlases"][entry["fold"]]["axes"], references)}
    summary = score_reliability(rows, expected_objects={r["object_id"]: r["fold"] for r in lock["objects"]},
                               expected_conditions=list(repeats_by_condition),
                               expected_repeats_by_condition=repeats_by_condition,
                               baseline_by_condition=baselines_by_condition(lock["conditions"]))
    holdout_rows = []
    for oid in schedule["holdout_ids"]:
        entry = next(r for r in lock["objects"] if r["object_id"] == oid)
        curve = read_lc(verify(entry["lightcurve"]), object_id=oid, period_hours=entry["period_hours"])
        withheld = select_holdout(curve).withheld
        curves = tuple(ForwardLightcurve(tuple(ForwardObservation(o.time_jd, o.flux, o.sun, o.observer)
                                              for o in s.observations), relative=s.calibrated == 0)
                       for s in withheld.sessions)
        constant = float(np.sqrt(np.mean([np.mean(np.square(np.asarray([o.flux for o in s.observations]) /
                                                           np.mean([o.flux for o in s.observations]) - 1))
                                         for s in withheld.sessions])))
        for sampling in ("full", "thin10"):
            for arm in ("classical20", "k3", "atlas"):
                budget.check()
                directory = root / "fits" / oid / sampling / arm
                fitted = unseal(directory / "result.json")
                if (fitted["object_id"], fitted["fold"], fitted["sampling"], fitted["arm"]) != (oid, entry["fold"], sampling, arm):
                    raise ValueError("fit result identity mismatch at scoring")
                verify(fitted["fit_input"])
                row = {k: fitted[k] for k in ("object_id", "fold", "arm", "sampling", "status")}
                row.update(withheld_normalized_curve_rms=None, constant_brightness_rms=constant,
                           fit_completed=fitted["selected_fit"] is not None)
                if fitted["selected_fit"] is not None:
                    selected = directory / f"start-{fitted['selected_fit']['start_index']:03d}"
                    validate_start(selected)
                    solution = load_solution(selected / "solution-parameters.txt", selected / "solution-areas.txt")
                    try:
                        row["withheld_normalized_curve_rms"] = holdout_shape_rmse(solution, curves)
                    except ValueError as exc:
                        row.update(status="failed", failure_code=f"forward_scoring: {exc}")
                    row["training_rms"] = fitted["selected_fit"]["final_relative_rms"]
                    lon, lat = np.radians([solution.longitude_deg, solution.latitude_deg])
                    direction = np.asarray([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
                    reference = np.asarray(catalog[oid].solution_vectors)
                    reference /= np.linalg.norm(reference, axis=1, keepdims=True)
                    row["catalog_axis_disagreement_deg"] = float(np.degrees(np.arccos(np.clip(
                        np.abs(reference @ direction).max(), 0, 1))))
                holdout_rows.append(row)
    holdout_summary = {}
    for sampling in ("full", "thin10"):
        for arm in ("classical20", "k3", "atlas"):
            subset = [r for r in holdout_rows if r["sampling"] == sampling and r["arm"] == arm]
            valid = [r for r in subset if r["withheld_normalized_curve_rms"] is not None]
            value = {"n_scheduled": len(subset), "n_fit_completed": sum(r["fit_completed"] for r in subset),
                     "n_forward_valid": len(valid)}
            if valid:
                value["withheld_rms"] = stratified_asteroid_bootstrap(valid, value_key="withheld_normalized_curve_rms")
                value["constant_brightness_rms"] = stratified_asteroid_bootstrap(valid, value_key="constant_brightness_rms")
            if arm != "classical20":
                base = {r["object_id"]: r for r in holdout_rows if r["sampling"] == sampling and
                        r["arm"] == "classical20" and r["withheld_normalized_curve_rms"] is not None}
                matched = [{"object_id": r["object_id"], "fold": r["fold"], "difference":
                            r["withheld_normalized_curve_rms"] - base[r["object_id"]]["withheld_normalized_curve_rms"]}
                           for r in valid if r["object_id"] in base]
                value["n_completed_pairs"] = len(matched)
                value["candidate_minus_classical_rms"] = stratified_asteroid_bootstrap(matched, value_key="difference") if matched else None
            holdout_summary[f"{sampling}-{arm}"] = value
    for arm in ("classical20", "k3", "atlas"):
        full = {r["object_id"]: r for r in holdout_rows if r["sampling"] == "full" and r["arm"] == arm
                and r["withheld_normalized_curve_rms"] is not None}
        thin = [r for r in holdout_rows if r["sampling"] == "thin10" and r["arm"] == arm
                and r["withheld_normalized_curve_rms"] is not None and r["object_id"] in full]
        matched = [{"object_id": r["object_id"], "fold": r["fold"], "difference":
                    r["withheld_normalized_curve_rms"] - full[r["object_id"]]["withheld_normalized_curve_rms"]}
                   for r in thin]
        holdout_summary[f"thin10-minus-full-{arm}"] = {
            "n_completed_pairs": len(matched),
            "difference": stratified_asteroid_bootstrap(matched, value_key="difference") if matched else None}
    bases = [r for r in rows if r["condition"] == "full" and r["status"] == "ok"]
    sky = {"k3": clustering([r["axes"] for r in bases]),
           "atlas": clustering([lock["atlases"][r["fold"]]["axes"] for r in bases]),
           "uniform_random_three": clustering([random_axes[r["object_id"]]["first_draw_axes"] for r in bases])}
    report = {"schema": SCHEMA, "sampling": summary, "withheld": holdout_summary, "clustering": sky,
              "random_control": stratified_asteroid_bootstrap(list(random_axes.values()), value_key="first_draw_error_deg"),
              "random_control_expected_1000_draws": stratified_asteroid_bootstrap(list(random_axes.values()), value_key="mean_error_deg"),
              "atlas_control": stratified_asteroid_bootstrap(list(random_axes.values()), value_key="atlas_error_deg"),
              "claims": "retrospective sensitivity and withheld-observation checks, not proof of absence of overfitting"}
    # Reports are generated only after all registered prediction and fit cells validate.
    for name, payload in (("scored-rows.json", rows), ("withheld-rows.json", holdout_rows),
                          ("random-atlas-rows.json", list(random_axes.values()))):
        target = root / name
        if target.exists():
            if unseal(target) != payload:
                raise ValueError("partial scoring products differ on resume")
        else:
            seal(target, payload)
    for name, payload in (("sampling-table.csv", render_reliability_csv(summary)),
                          ("sampling-table.tex", render_reliability_tex(summary))):
        path = root / name
        if path.exists():
            if path.read_text() != payload:
                raise ValueError("scoring text differs on resume")
        else:
            _write_text_new(path, payload)
    from .reliability_reporting import write_reports
    report["report_files"] = [{"path": str(p.relative_to(root)), "sha256": digest(p)}
                              for p in write_reports(root, rows, report, lock)]
    seal(root / "report.json", report)
    return report
