"""Reference-blind, directly timed known-period pole-grid benchmark.

This module deliberately lives outside ``k3`` so classical cold workers do not
import the neural stack. Published weights, APIs and archived runs are unchanged.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

RADIUS_DEG = 15.772555008039914
ARMS = ("classical20", "guided20", "classical15", "guided15", "standard6")
MODES = ("cold", "warm")
STANDARD = ((0, 0), (180, 0), (90, 60), (240, 60), (90, -60), (240, -60))
SCHEMA = "delphi.k3-pole-grid-benchmark.v3"
ID_PATTERN = re.compile(r"asteroid_[0-9]+\Z")
SETTINGS = {
    "radius_deg": RADIUS_DEG,
    "radius_provenance": "previously_reported_170_object_mean_oracle_error_not_a_confidence_radius",
    "spacing_degrees": [20, 15],
    "primary_comparison": "classical20_vs_guided20_cold",
    "workers": 6,
    "repeats": 3,
    "order_seed": 20260911,
    "execution_order": "deterministically_shuffled_fold_mode_blocks_then_shuffled_arm_object_repeat_cases",
    "convergence_tolerance": 0.0003,
    "iteration_cap": 1000,
    "timeout_seconds_per_start": 300,
    "period": "supplied_fixed_same_in_both_arms",
    "search_region": "initializations_only_optimizer_free_to_leave_region",
    "selection": "minimum_valid_final_relative_rms_then_start_index",
    "completion": "all_starts_attempted_and_at_least_one_valid_fit",
    "warmup": "one_synthetic_object_per_persistent_worker_never_evaluation_input",
    "cold": "fresh_python_process_including_imports_model_loading_input_output_not_os_cache_cold",
    "warm": "persistent_worker_input_read_to_written_result_including_ipc_excludes_worker_setup",
    "full_run_budget_seconds": 604800,
    "budget_revision": (
        "user_authorized_seven_day_ceiling_after_the_v1_engineering_pilot_"
        "exceeded_24_hours_no_scientific_endpoint_or_criterion_changed"
    ),
    "parent_v1_lock_sha256": (
        "bbc8bc676f25db49576f4cf4f3a2e4d16acdcbc72c921655e9452551cd2c50ca"
    ),
    "environment_relock": (
        "v2_pilot_retained_after_the_frozen_system_python_executable_changed_"
        "before_full_execution_no_scientific_endpoint_or_criterion_changed"
    ),
    "parent_v2_lock_sha256": (
        "7c871cc249f4c461b4eb1210bda47d18d1769a226431f01c23963e30e3e396e9"
    ),
    "bootstrap_resamples": 10000,
    "bootstrap_seed": 20260911,
    "bootstrap_percentiles": [2.5, 97.5],
    "binary_noninferiority_margin": 0.05,
    "rms_noninferiority_ratio": 1.01,
}


class GridBenchmarkError(ValueError):
    """An incomplete or unbound benchmark must not produce a result."""


def read_json(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise GridBenchmarkError("expected JSON object")
    return value


def digest(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_once(path: str | Path, value: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _binding(path: str | Path) -> dict:
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": digest(path)}


def _verify_binding(binding: dict) -> Path:
    path = Path(binding["path"])
    if digest(path) != binding["sha256"]:
        raise GridBenchmarkError(f"input checksum changed: {path.name}")
    return path


def _runtime_sources(root: Path) -> dict:
    paths = list((root / "lc_pipeline").rglob("*.py"))
    paths.append(root / "repro/run_k3_grid_benchmark.py")
    return {str(p.relative_to(root)): digest(p) for p in sorted(paths)}


def _environment(device: str) -> dict:
    """Capture stable timing prerequisites without initializing CUDA in the parent."""
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) < SETTINGS["workers"]:
        raise GridBenchmarkError("benchmark requires at least six available affinity CPUs")
    packages = {name: importlib.metadata.version(name) for name in
                ("numpy", "torch", "scipy", "astropy", "astropy-healpix", "safetensors")}
    cpuinfo = Path("/proc/cpuinfo").read_text()
    models = sorted(set(re.findall(r"^model name\s*:\s*(.+)$", cpuinfo, re.M)))
    gpu = None
    if device == "cuda":
        result = subprocess.run([
            "nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv,noheader"], capture_output=True, text=True, timeout=30, check=True)
        gpu = result.stdout.strip().splitlines()
        if not gpu:
            raise GridBenchmarkError("CUDA benchmark has no visible GPU")
    return {"python": sys.version, "interpreter": _binding(sys.executable),
            "packages": packages, "platform": platform.platform(), "cpu_count": os.cpu_count(),
            "cpu_affinity": affinity, "cpu_models": models, "device": device, "gpu": gpu,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "worker_thread_environment": {key: "1" for key in
                ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
            "neural_torch_threads": 6}


def unit_vectors(axes: object) -> np.ndarray:
    value = np.asarray(axes, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 3 or not np.all(np.isfinite(value)):
        raise GridBenchmarkError("axes must be finite N by 3 vectors")
    scale = np.max(np.abs(value), axis=1)
    if np.any(scale <= 0):
        raise GridBenchmarkError("zero axis")
    scaled = value / scale[:, None]
    return scaled / np.linalg.norm(scaled, axis=1)[:, None]


def _xyz(longitude: float, latitude: float) -> list[float]:
    lon, lat = math.radians(longitude), math.radians(latitude)
    return [math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)]


def pole_grid(spacing: int) -> np.ndarray:
    if type(spacing) is not int or spacing not in (15, 20):
        raise GridBenchmarkError("grid spacing must be 15 or 20 degrees")
    coordinates = [(0, -90), (0, 90)] + [
        (lon, lat) for lat in range(-90 + spacing, 90, spacing)
        for lon in range(0, 360, spacing)
    ]
    return unit_vectors([_xyz(lon, lat) for lon, lat in coordinates])


def guided_starts(axes: object, spacing: int, radius: float = RADIUS_DEG) -> np.ndarray:
    candidates = unit_vectors(axes)
    if candidates.shape != (3, 3) or not math.isfinite(radius) or not 0 <= radius <= 90:
        raise GridBenchmarkError("exactly three axes and a radius in [0,90] are required")
    grid = pole_grid(spacing)
    inside = np.max(np.abs(grid @ candidates.T), axis=1) >= math.cos(math.radians(radius)) - 1e-12
    # Signed directions remain distinct for physical inversion. The mask is axial.
    proposed = np.concatenate((grid[inside], candidates, -candidates))
    unique: list[np.ndarray] = []
    for vector in proposed:
        if not any(np.linalg.norm(vector - prior) <= 1e-10 for prior in unique):
            unique.append(vector)
    # Canonical order is invariant to candidate ordering and sign conventions.
    unique.sort(key=lambda p: tuple(np.round(p, 12)))
    return np.stack(unique)


def starts_for_arm(arm: str, axes: object | None = None) -> np.ndarray:
    if arm == "standard6":
        return unit_vectors([_xyz(lon, lat) for lon, lat in STANDARD])
    if arm not in ARMS:
        raise GridBenchmarkError("unknown benchmark arm")
    spacing = 20 if arm.endswith("20") else 15
    return guided_starts(axes, spacing) if arm.startswith("guided") else pole_grid(spacing)


def _validated_capacity(path: Path, objects: list[dict]) -> dict:
    from .k3.solver_capacity import (
        CAPACITY_FIELDS,
        LightcurveStructure,
        SolverCapacityError,
        capacity_violations,
        internal_capacity_padding,
        internal_capacity_requirements,
        read_static_capacities,
        require_current_internal_capacity_receipt,
        tree_sha256,
    )
    report = read_json(path)
    try:
        require_current_internal_capacity_receipt(report)
    except SolverCapacityError as exc:
        raise GridBenchmarkError(str(exc)) from exc
    if report.get("internal_capacity_padding") != internal_capacity_padding():
        raise GridBenchmarkError("a corrected internal-capacity v2 report is required")
    try:
        expanded = report["expanded_solver"]
        preflight = report["expanded_solver_preflight"]
        sanitizer = report["sanitizer_boundary_checks"]
    except KeyError as exc:
        raise GridBenchmarkError("capacity receipt is incomplete") from exc
    if not all(isinstance(value, dict) for value in (expanded, preflight, sanitizer)):
        raise GridBenchmarkError("capacity receipt has malformed sections")
    parity = report.get("development_supported_input_parity_smoke", {})
    if (parity.get("status") != "passed" or parity.get("output_hashes_match") is not True
            or parity.get("object_id") != "asteroid_227"
            or parity.get("official", {}).get("return_code") != 0
            or parity.get("expanded", {}).get("return_code") != 0
            or parity.get("official") != parity.get("expanded")):
        raise GridBenchmarkError("capacity receipt lacks supported-input parity evidence")
    root = Path(str(expanded.get("source_root", ""))).resolve()
    binary = Path(str(expanded.get("binary", ""))).resolve()
    period_scan = Path(str(expanded.get("period_scan_binary", ""))).resolve()
    if (
        not root.is_dir()
        or tree_sha256(root) != expanded.get("source_tree_sha256")
        or not binary.is_file()
        or binary.parent != root
        or digest(binary) != expanded.get("binary_sha256")
        or not period_scan.is_file()
        or period_scan.parent != root
        or digest(period_scan) != expanded.get("period_scan_binary_sha256")
    ):
        raise GridBenchmarkError("corrected solver tree changed")
    runs = sanitizer.get("runs")
    if sanitizer.get("status") != "passed" or not isinstance(runs, dict) or set(runs) != {
        "asteroid_109", "asteroid_2512"
    }:
        raise GridBenchmarkError("capacity receipt lacks required boundary sanitizer evidence")
    for object_id in sorted(runs):
        row = runs[object_id]
        if not isinstance(row, dict) or any(
            not isinstance(row.get(binary_name), dict)
            or row[binary_name].get("return_code") != 0
            for binary_name in ("convexinv", "period_scan")
        ):
            raise GridBenchmarkError("capacity receipt has failed boundary sanitizer evidence")
    capacities = read_static_capacities(root)
    if preflight.get("capacities") != capacities:
        raise GridBenchmarkError("capacity receipt static capacities differ from its solver")
    reported: dict[str, dict] = {}
    for role, expected_count in (("development", 30), ("locked_evaluation", 140)):
        cohort = preflight.get(role)
        rows = cohort.get("objects") if isinstance(cohort, dict) else None
        if not isinstance(rows, list) or len(rows) != expected_count:
            raise GridBenchmarkError("capacity receipt lacks full declared cohort support")
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("object_id"), str):
                raise GridBenchmarkError("capacity receipt has an invalid cohort row")
            object_id = row["object_id"]
            if object_id in reported:
                raise GridBenchmarkError("capacity receipt repeats an object")
            native = row.get("structure")
            effective = row.get("internal_required_capacities")
            if not isinstance(native, dict) or not isinstance(effective, dict):
                raise GridBenchmarkError("capacity receipt lacks effective internal requirements")
            try:
                expected = internal_capacity_requirements(LightcurveStructure(**native))
            except (TypeError, ValueError) as exc:
                raise GridBenchmarkError("capacity receipt has an invalid native structure") from exc
            if effective != expected or row.get("violations") != [] or any(
                expected[name] > capacities[name] for name in CAPACITY_FIELDS
            ):
                raise GridBenchmarkError("capacity receipt does not reserve internal solver writes")
            reported[object_id] = row
    for row in objects:
        object_id = row["object_id"]
        if (
            object_id not in reported
            or reported[object_id].get("structure") != row["structure"]
            or capacity_violations(LightcurveStructure(**row["structure"]), capacities)
        ):
            raise GridBenchmarkError(f"unsafe solver capacity for {row['object_id']}")
    if set(reported) != {row["object_id"] for row in objects}:
        raise GridBenchmarkError("capacity receipt cohort differs from frozen benchmark objects")
    return {"executable": _binding(binary), "capacity_report": _binding(path),
            "source_root": str(root), "source_tree_sha256": expanded["source_tree_sha256"],
            "capacities": capacities}


def freeze(*, blind_inputs: Path, splits: Path, development: Path, dump_root: Path,
           bundle_root: Path, capacity_report: Path, reference_catalog: Path,
           output: Path, device: str = "cuda") -> dict:
    from .k3.solver_capacity import parse_lightcurve_structure
    blind = read_json(blind_inputs)
    split_document = read_json(splits)
    folds = split_document["folds"]
    if split_document.get("schema") != "delphi.grouped-splits.v2":
        raise GridBenchmarkError("full split schema is invalid")
    if blind.get("source_full_split_sha256") != digest(splits):
        raise GridBenchmarkError("blind inputs and split identity disagree")
    if blind.get("source_catalog_sha256") != digest(reference_catalog):
        raise GridBenchmarkError("reference catalog hash differs from frozen inputs")
    rows, seen = [], set()
    root = dump_root.resolve(strict=True)
    for entry in blind["objects"]:
        oid, fold = entry["object_id"], entry["fold"]
        if not ID_PATTERN.fullmatch(oid) or oid in seen or type(fold) is not int or fold not in range(5):
            raise GridBenchmarkError("invalid or duplicate object identity")
        seen.add(oid)
        if oid not in folds[fold]["test_ids"]:
            raise GridBenchmarkError("object not held out in declared fold")
        source = entry["lightcurve"]
        if source["source_path"] != f"files/{oid}/lc.txt":
            raise GridBenchmarkError("internal DAMIT ID and native lightcurve path disagree")
        lightcurve = (root / source["source_path"]).resolve(strict=True)
        if not lightcurve.is_relative_to(root) or digest(lightcurve) != source["source_sha256"]:
            raise GridBenchmarkError("lightcurve escaped dump or changed")
        period = entry["period_hours"]
        if isinstance(period, bool) or not math.isfinite(period) or period <= 0:
            raise GridBenchmarkError("invalid supplied period")
        structure = parse_lightcurve_structure(lightcurve)
        rows.append({"object_id": oid, "fold": fold, "period_hours": period,
                     "lightcurve": _binding(lightcurve), "structure": structure.as_dict()})
    if len(rows) != 170 or set(seen) != {o for f in folds for o in f["test_ids"]}:
        raise GridBenchmarkError("benchmark must contain all 170 OOF objects")
    if not isinstance(folds, list) or len(folds) != 5:
        raise GridBenchmarkError("full split must contain five outer folds")
    expected_development: list[str] = []
    salt = "delphi-k3-convergence-followup-20260910"
    for fold in range(5):
        split_fold = folds[fold]
        if not isinstance(split_fold, dict) or split_fold.get("fold") != fold:
            raise GridBenchmarkError("full split fold order is invalid")
        ranked = sorted(
            (str(value) for value in split_fold.get("test_ids", [])),
            key=lambda value: hashlib.sha256(f"{salt}:{value}".encode("ascii")).hexdigest(),
        )
        if len(ranked) != 34:
            raise GridBenchmarkError("full split test role is invalid")
        expected_development.extend(ranked[:6])
    development_document = read_json(development)
    if (
        development_document.get("schema") != "delphi.k3-convergence-object-subset.v1"
        or development_document.get("role") != "development"
        or development_document.get("salt") != salt
        or development_document.get("selection")
        != "six_development_objects_per_original_outer_fold_by_sha256_rank"
        or development_document.get("source_full_split_sha256") != digest(splits)
        or development_document.get("object_ids") != expected_development
        or len(expected_development) != 30
    ):
        raise GridBenchmarkError("development manifest is not the exact frozen 30-object cohort")
    bundles = []
    for fold in range(5):
        directory = (bundle_root / f"k3-oof-fold-{fold}").resolve(strict=True)
        manifest = read_json(directory / "bundle.json")
        if (
            manifest.get("schema") != "delphi.k3-ensemble-bundle.v1"
            or manifest.get("bundle_id") != f"k3-oof-fold-{fold}"
            or manifest.get("fold") != fold
            or manifest.get("splits_sha256") != digest(splits)
        ):
            raise GridBenchmarkError("bundle split/fold mismatch")
        expected_roles = {
            key: folds[fold][key]
            for key in ("train_ids", "validation_ids", "calibration_ids", "test_ids")
        }
        if manifest.get("object_roles") != expected_roles:
            raise GridBenchmarkError("bundle roles mismatch")
        files = [_binding(directory / "bundle.json")]
        members = manifest.get("members")
        if not isinstance(members, list) or [member.get("seed") if isinstance(member, dict) else None
                                             for member in members] != [17, 42, 137, 777, 2027]:
            raise GridBenchmarkError("bundle must contain five ordered evaluated seeds")
        for member in members:
            if set(member) != {"seed", "file", "sha256", "source_checkpoint_sha256"}:
                raise GridBenchmarkError("bundle member metadata is incomplete")
            file = (directory / member["file"]).resolve(strict=True)
            if file.parent != directory or digest(file) != member["sha256"]:
                raise GridBenchmarkError("unsafe or changed weight")
            files.append(_binding(file))
        bundles.append({"directory": str(directory), "files": files})
    dev = set(expected_development)
    pilot_ids = []
    for fold in range(5):
        eligible = [r["object_id"] for r in rows if r["fold"] == fold and r["object_id"] in dev]
        if not eligible:
            raise GridBenchmarkError("development pilot lacks an outer fold")
        pilot_ids.append(min(eligible, key=lambda oid: hashlib.sha256(
            f"grid-pilot-20260911:{oid}".encode()).hexdigest()))
    code_root = Path(__file__).resolve().parents[1]
    payload = {"schema": SCHEMA, "phase": "frozen_before_grid_execution",
               "scope": "retrospective_known_period_not_untouched_confirmation",
               "settings": SETTINGS, "objects": rows, "pilot_ids": pilot_ids,
               "bundles": bundles, "device": device, "code_root": str(code_root),
               "runtime_sources": _runtime_sources(code_root),
               "inputs": {"blind_inputs": _binding(blind_inputs), "splits": _binding(splits),
                          "development": _binding(development),
                          "reference_catalog": _binding(reference_catalog)},
               "solver": _validated_capacity(capacity_report, rows),
               "environment": _environment(device)}
    write_once(output, payload)
    return payload


def validate_lock(path: Path, *, require_environment: bool = True) -> dict:
    lock = read_json(path)
    if lock.get("schema") != SCHEMA or lock.get("settings") != SETTINGS:
        raise GridBenchmarkError("incompatible or edited benchmark settings")
    if require_environment and lock.get("environment") != _environment(lock["device"]):
        raise GridBenchmarkError("timing environment changed since lock")
    root = Path(lock["code_root"])
    if _runtime_sources(root) != lock["runtime_sources"]:
        raise GridBenchmarkError("runtime source changed since lock")
    for binding in lock["inputs"].values():
        # Hash bytes only; reference records never enter execution.
        _verify_binding(binding)
    for row in lock["objects"]:
        _verify_binding(row["lightcurve"])
    for bundle in lock["bundles"]:
        for binding in bundle["files"]:
            _verify_binding(binding)
    if _validated_capacity(Path(lock["solver"]["capacity_report"]["path"]), lock["objects"]) != lock["solver"]:
        raise GridBenchmarkError("solver binding changed")
    return lock


def _native_start(job: dict, index: int, vector: np.ndarray, observed: tuple,
                  solver: dict, settings: dict, directory: Path) -> dict:
    from .v2.convexinv import _FINAL, _ITERATION, ConvexinvParameters
    target = directory / f"start-{index:03d}"
    target.mkdir()
    lon = math.degrees(math.atan2(float(vector[1]), float(vector[0]))) % 360
    lat = math.degrees(math.asin(float(np.clip(vector[2], -1, 1))))
    params = ConvexinvParameters(lambda_deg=lon, beta_deg=lat, period_hours=job["period_hours"],
                                free_period=False, iteration_stop_condition=settings["convergence_tolerance"])
    parameter_path = target / "parameters.txt"
    with parameter_path.open("x", encoding="ascii") as stream:
        stream.write(params.render())
    products = {"modelled": target / "modelled-lightcurve.txt",
                "parameters": target / "solution-parameters.txt", "areas": target / "solution-areas.txt"}
    command = [solver["executable"]["path"], "-v", "-o", str(products["areas"]),
               "-p", str(products["parameters"]), str(parameter_path), str(products["modelled"])]
    started = time.perf_counter()
    with Path(job["lightcurve"]["path"]).open("rb") as stream:
        try:
            result = subprocess.run(command, stdin=stream, capture_output=True, cwd=target,
                                    timeout=settings["timeout_seconds_per_start"], check=False)
            code, stdout, stderr, timed_out = result.returncode, result.stdout, result.stderr, False
        except subprocess.TimeoutExpired as exc:
            code, stdout, stderr, timed_out = None, exc.stdout or b"", exc.stderr or b"", True
    elapsed = time.perf_counter() - started
    for name, content in (("stdout.txt", stdout), ("stderr.txt", stderr)):
        with (target / name).open("xb") as stream:
            stream.write(content)
    text = stdout.decode("utf-8", errors="replace")
    iterations = list(_ITERATION.finditer(text))
    count = int(iterations[-1].group(1)) if iterations else None
    final = _FINAL.search(text)
    selected, error = None, None
    try:
        if timed_out or code != 0 or count is None or not 0 < count < settings["iteration_cap"]:
            raise GridBenchmarkError("timeout_exit_or_iteration_cap")
        diagnostics = [float(iterations[-1].group(i)) for i in (2, 3)]
        if any(not math.isfinite(x) or x < 0 for x in diagnostics):
            raise GridBenchmarkError("invalid_final_fit_diagnostics")
        if final is None or not all(p.is_file() and p.stat().st_size for p in products.values()):
            raise GridBenchmarkError("missing_solution_product")
        final_lon, final_lat, period = (float(final.group(i)) for i in (1, 2, 3))
        if not all(math.isfinite(x) for x in (final_lon, final_lat, period)):
            raise GridBenchmarkError("nonfinite_solution")
        if not math.isclose(period, job["period_hours"], rel_tol=1e-7, abs_tol=1e-9):
            raise GridBenchmarkError("fixed_period_changed")
        predicted = np.asarray(products["modelled"].read_text().split(), dtype=np.float64)
        if len(predicted) != sum(len(curve) for curve in observed) or not np.all(np.isfinite(predicted)):
            raise GridBenchmarkError("invalid_modelled_lightcurve")
        offset, residuals = 0, []
        for curve in observed:
            part = predicted[offset:offset + len(curve)]
            offset += len(curve)
            if float(curve.mean()) == 0 or float(part.mean()) == 0:
                raise GridBenchmarkError("zero_mean_lightcurve")
            residuals.extend(np.square(curve / curve.mean() - part / part.mean()).tolist())
        rms = math.sqrt(sum(residuals) / len(residuals))
        if not math.isfinite(rms) or rms <= 0:
            raise GridBenchmarkError("invalid_rms")
        selected = {"axis": _xyz(final_lon, final_lat), "final_relative_rms": rms,
                    "start_index": index, "raw_longitude": final_lon, "raw_latitude": final_lat}
    except (GridBenchmarkError, ValueError, OSError) as exc:
        error = str(exc)
    files = {p.name: digest(p) for p in target.iterdir() if p.is_file()}
    receipt = {"start_index": index, "initial_axis": vector.tolist(), "return_code": code,
               "timed_out": timed_out, "iterations": count, "solver_wall_seconds": elapsed,
               "selected_fit": selected, "failure_reason": error, "files": files}
    write_once(target / "record.json", receipt)
    return receipt


def _predictor(lock: dict, fold: int):
    import torch

    from .k3.bundle import K3EnsemblePredictor
    torch.set_num_threads(6)
    return K3EnsemblePredictor(lock["bundles"][fold]["directory"], device=lock["device"])


def _warmup(predictor) -> None:
    from .v2.preprocessing import KnownPeriod, Observation, ObservationEpoch
    observations = tuple(Observation(
        time_jd=2450000.0 + index / 256,
        relative_brightness=1 + 0.1 * math.sin(index * math.pi / 8),
        sun_asteroid_ecliptic_j2000_au=(1.0, 0.0, 0.0),
        observer_asteroid_ecliptic_j2000_au=(0.0, 1.0, 0.0),
    ) for index in range(64))
    predictor.predict((ObservationEpoch(epoch_id="synthetic-warmup", observations=observations),),
                      known_period=KnownPeriod(hours=6.0, provenance="synthetic-worker-warmup"),
                      object_id="synthetic-worker-warmup")


def worker_case(lock: dict, job: dict, directory: Path, predictor=None) -> dict:
    from .v2.convexinv import _parse_lightcurve_brightness
    start = time.perf_counter()
    directory.mkdir(parents=True, exist_ok=False)
    _verify_binding(job["lightcurve"])
    observed = _parse_lightcurve_brightness(Path(job["lightcurve"]["path"]))
    prediction, axes, inference = None, None, 0.0
    if job["arm"].startswith("guided"):
        from .k3.damit import _read_lc
        from .v2.preprocessing import KnownPeriod
        neural_started = time.perf_counter()
        predictor = predictor or _predictor(lock, job["fold"])
        epochs = _read_lc(Path(job["lightcurve"]["path"]))
        prediction = predictor.predict(epochs, known_period=KnownPeriod(
            hours=job["period_hours"], provenance="frozen-supplied-known-period"), object_id=job["object_id"])
        if predictor.device.type == "cuda":
            import torch
            torch.cuda.synchronize(predictor.device)
        inference = time.perf_counter() - neural_started
        axes = [item["axis_xyz"] for item in prediction["axes"]]
        write_once(directory / "prediction.json", prediction)
    starts = starts_for_arm(job["arm"], axes)
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    with concurrent.futures.ThreadPoolExecutor(max_workers=lock["settings"]["workers"]) as pool:
        futures = [pool.submit(_native_start, job, index, vector, observed, lock["solver"],
                               lock["settings"], directory) for index, vector in enumerate(starts)]
        records = [future.result() for future in futures]
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    fits = [r["selected_fit"] for r in records if r["selected_fit"] is not None]
    selected = min(fits, key=lambda f: (f["final_relative_rms"], f["start_index"])) if fits else None
    result = {k: job[k] for k in ("object_id", "fold", "repeat_index", "arm", "timing_mode")}
    result.update({"completed": selected is not None, "selected_fit": selected,
                   "prediction_sha256": digest(directory / "prediction.json") if prediction else None,
                   "starts_requested": len(starts), "valid_starts": len(fits),
                   "inference_and_optional_load_seconds": inference,
                   "solver_cpu_seconds": after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime,
                   "native_solver_wall_seconds_sum": sum(r["solver_wall_seconds"] for r in records),
                   "iterations_sum": sum(r["iterations"] or 0 for r in records),
                   "cell_files": {f"start-{i:03d}/record.json": digest(directory / f"start-{i:03d}/record.json")
                                  for i in range(len(records))}})
    write_once(directory / "selected-result.json", result)
    result["worker_wall_seconds"] = time.perf_counter() - start
    return result


def worker_main(lock_path: Path, *, job_path: Path | None = None, fold: int | None = None) -> None:
    # The parent verifies all identities once before execution, outside latency
    # measurement. Workers read only their pole-free jobs and deployment inputs.
    lock = read_json(lock_path)
    if job_path is not None:
        job = read_json(job_path)
        result = worker_case(lock, job, Path(job["directory"]))
        print(json.dumps(result, allow_nan=False), flush=True)
        return
    if fold not in range(5):
        raise GridBenchmarkError("persistent worker requires a fold")
    setup = time.perf_counter()
    predictor = _predictor(lock, fold)
    _warmup(predictor)
    print(json.dumps({"ready": True, "setup_seconds": time.perf_counter() - setup}), flush=True)
    for line in sys.stdin:
        job = json.loads(line)
        if job.get("stop"):
            return
        if job["fold"] != fold:
            raise GridBenchmarkError("worker fold mismatch")
        print(json.dumps(worker_case(lock, job, Path(job["directory"]), predictor), allow_nan=False), flush=True)


def _jobs(lock: dict, role: str) -> list[dict]:
    if role not in ("pilot", "full"):
        raise GridBenchmarkError("execution role must be pilot or full")
    rows = sorted(
        (
            row
            for row in lock["objects"]
            if role == "full" or row["object_id"] in lock["pilot_ids"]
        ),
        key=lambda row: row["object_id"],
    )
    rng = random.Random(SETTINGS["order_seed"])
    jobs = []
    # Randomize all fold/mode blocks to avoid a systematic cold-first temporal
    # trend, then randomize all arm/object/repeat cases within each block.  The
    # executor tears down a warm worker before the next block, so a later cold
    # block still runs without a resident neural process.
    blocks = [
        (fold, mode)
        for fold in sorted({int(row["fold"]) for row in rows})
        for mode in MODES
    ]
    rng.shuffle(blocks)
    for fold, mode in blocks:
        cases = [
            {**row, "repeat_index": repeat, "arm": arm, "timing_mode": mode}
            for row in rows
            if row["fold"] == fold
            for repeat in range(3)
            for arm in ARMS
        ]
        rng.shuffle(cases)
        jobs.extend(cases)
    return jobs


def _pilot_budget(lock: dict, rows: list[dict], elapsed_seconds: float) -> dict:
    """Recompute the conservative full-run projection from validated pilot rows."""
    if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0:
        raise GridBenchmarkError("pilot elapsed time must be positive and finite")
    pilot_ids = set(lock["pilot_ids"])
    if len(pilot_ids) != 5:
        raise GridBenchmarkError("pilot budget requires five unique pilot objects")
    keys = []
    wall = []
    for row in rows:
        try:
            key = tuple(
                row[name]
                for name in ("object_id", "repeat_index", "arm", "timing_mode")
            )
            value = row["wall_seconds"]
        except (KeyError, TypeError) as exc:
            raise GridBenchmarkError("pilot timing row is incomplete") from exc
        if (
            key[0] not in pilot_ids
            or key[1] not in range(3)
            or key[2] not in ARMS
            or key[3] not in MODES
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0
        ):
            raise GridBenchmarkError("pilot timing row is invalid")
        keys.append(key)
        wall.append(float(value))
    expected = {
        (object_id, repeat, arm, mode)
        for object_id in pilot_ids
        for repeat in range(3)
        for arm in ARMS
        for mode in MODES
    }
    if len(keys) != len(expected) or len(set(keys)) != len(keys) or set(keys) != expected:
        raise GridBenchmarkError("pilot budget requires the complete validated 150-cell grid")
    observed_cells = sum(wall)
    if elapsed_seconds + 1e-9 < observed_cells:
        raise GridBenchmarkError("pilot elapsed time is shorter than its serial cell timings")
    pilot_meta = [row for row in lock["objects"] if row["object_id"] in pilot_ids]
    if len(pilot_meta) != len(pilot_ids):
        raise GridBenchmarkError("pilot metadata is incomplete")
    count_scale = len(lock["objects"]) / len(pilot_meta)
    pilot_points = sum(row["structure"]["total_observations"] for row in pilot_meta)
    full_points = sum(row["structure"]["total_observations"] for row in lock["objects"])
    if pilot_points <= 0 or full_points <= 0:
        raise GridBenchmarkError("pilot observation-count scale is invalid")
    observation_scale = full_points / pilot_points
    scale = max(count_scale, observation_scale)
    estimate = 2.0 * elapsed_seconds * scale
    limit = float(SETTINGS["full_run_budget_seconds"])
    return {
        "observed_pilot_cell_wall_seconds": observed_cells,
        "observed_pilot_elapsed_seconds": float(elapsed_seconds),
        "object_count_scale": float(count_scale),
        "observation_count_scale": float(observation_scale),
        "projection_scale": float(scale),
        "projected_full_elapsed_seconds": estimate,
        "method": "2x_pilot_elapsed_x_max_object_count_or_observation_count_scale",
        "statistical_upper_bound": False,
        "full_run_budget_seconds": limit,
        "within_full_run_budget": estimate <= limit,
    }


def _remaining_seconds(deadline: float | None, maximum: float) -> float:
    if deadline is None:
        return maximum
    remaining = deadline - time.perf_counter()
    if remaining <= 0:
        raise GridBenchmarkError("configured elapsed budget reached; no complete result")
    return min(maximum, remaining)


def _terminate_process_group(process: subprocess.Popen) -> None:
    """Terminate a worker and every native solver it currently owns."""
    import signal

    # start_new_session=True makes the worker PID the process-group ID.  Keep
    # using that known ID even if the Python group leader has already exited:
    # its native solver children may still be alive in the orphaned group.
    process_group = process.pid
    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError:
        if process.poll() is None:
            process.terminate()
    grace_deadline = time.perf_counter() + 10.0
    while time.perf_counter() < grace_deadline:
        try:
            os.killpg(process_group, 0)
        except ProcessLookupError:
            break
        except OSError:
            if process.poll() is not None:
                break
        time.sleep(0.05)
    try:
        os.killpg(process_group, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError:
        if process.poll() is None:
            process.kill()
    if process.poll() is None:
        process.wait(timeout=10)


def _readline_before(
    process: subprocess.Popen, deadline: float | None, *, description: str
) -> str:
    """Read one worker response with both a cell cap and the full-run deadline."""
    import select

    timeout = _remaining_seconds(deadline, 15000.0)
    ready, _, _ = select.select([process.stdout], [], [], timeout)
    if not ready:
        _terminate_process_group(process)
        if deadline is not None and time.perf_counter() >= deadline:
            raise GridBenchmarkError("configured elapsed budget reached; no complete result")
        raise GridBenchmarkError(f"{description} exceeded the bounded worker timeout")
    line = process.stdout.readline()
    if not line:
        raise GridBenchmarkError(f"{description} worker exited without a response")
    return line


def _stop_persistent_worker(process: subprocess.Popen, deadline: float | None) -> None:
    if process.poll() is not None:
        return
    try:
        process.stdin.write('{"stop":true}\n')
        process.stdin.flush()
        process.wait(timeout=_remaining_seconds(deadline, 60.0))
    except (BrokenPipeError, OSError, subprocess.TimeoutExpired, GridBenchmarkError):
        _terminate_process_group(process)


def _worker_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    return environment


def execute(*, lock_path: Path, output: Path, role: str, pilot_path: Path | None = None) -> dict:
    lock = validate_lock(lock_path)
    pilot_authorization = None
    pilot_execution_binding = None
    if role == "full":
        if pilot_path is None:
            raise GridBenchmarkError("full execution requires a completed pilot")
        pilot = validate_execution(pilot_path, lock_path)
        if pilot["role"] != "pilot":
            raise GridBenchmarkError("full execution requires a validated pilot execution")
        pilot_authorization = _pilot_budget(
            lock, pilot["validated_rows"], float(pilot["execution_elapsed_seconds"])
        )
        if pilot_authorization["within_full_run_budget"] is not True:
            raise GridBenchmarkError("pilot does not authorize full execution within the configured budget")
        pilot_execution_binding = _binding(pilot_path)
    elif role != "pilot":
        raise GridBenchmarkError("execution role must be pilot or full")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    start_record = {
        "schema": SCHEMA,
        "lock_sha256": digest(lock_path),
        "role": role,
        "started_unix": time.time(),
    }
    if pilot_execution_binding is not None:
        start_record["pilot_execution"] = pilot_execution_binding
    write_once(output / "execution-start.json", start_record)
    jobs = _jobs(lock, role)
    rows, setups = [], []
    environment = _worker_environment()
    worker = None
    worker_log = None
    current_block = None
    current_name = None
    overall_started = time.perf_counter()
    deadline = (
        overall_started + float(SETTINGS["full_run_budget_seconds"])
        if role == "full"
        else None
    )
    try:
        try:
            for position, job in enumerate(jobs):
                _remaining_seconds(deadline, 15000.0)
                block = (job["timing_mode"], job["fold"])
                if block != current_block:
                    if worker is not None:
                        _stop_persistent_worker(worker, deadline)
                        worker = None
                        if worker_log is not None:
                            worker_log.close()
                            worker_log = None
                    current_block = block
                    if job["timing_mode"] == "cold":
                        setups.append(
                            {
                                "fold": job["fold"],
                                "timing_mode": "cold",
                                "ready": True,
                                "persistent_worker": False,
                                "fresh_process_per_case": True,
                                "setup_seconds": 0.0,
                                "setup_in_cell_timing": True,
                            }
                        )
                    else:
                        worker_log = (
                            output / f"warm-worker-fold-{job['fold']}.log"
                        ).open("xb")
                        worker = subprocess.Popen(
                            [
                                sys.executable,
                                "-m",
                                "repro.run_k3_grid_benchmark",
                                "_worker",
                                "--lock",
                                str(lock_path.resolve()),
                                "--fold",
                                str(job["fold"]),
                            ],
                            cwd=lock["code_root"],
                            env=environment,
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=worker_log,
                            text=True,
                            start_new_session=True,
                        )
                        ready = json.loads(
                            _readline_before(
                                worker, deadline, description="persistent setup"
                            )
                        )
                        if ready.get("ready") is not True:
                            raise GridBenchmarkError("persistent worker did not report ready")
                        setups.append(
                            {
                                "fold": job["fold"],
                                "timing_mode": "warm",
                                "ready": True,
                                "persistent_worker": True,
                                "fresh_process_per_case": False,
                                "setup_seconds": ready["setup_seconds"],
                                "setup_in_cell_timing": False,
                            }
                        )
                name = (
                    f"{job['object_id']}-r{job['repeat_index']}-"
                    f"{job['arm']}-{job['timing_mode']}"
                )
                current_name = name
                directory = output / "cases" / name
                job["directory"] = str(directory)
                if job["timing_mode"] == "cold":
                    job_path = output / "jobs" / f"{name}.json"
                    write_once(job_path, job)
                    started = time.perf_counter()
                    child = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "repro.run_k3_grid_benchmark",
                            "_worker",
                            "--lock",
                            str(lock_path.resolve()),
                            "--job",
                            str(job_path),
                        ],
                        cwd=lock["code_root"],
                        env=environment,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        start_new_session=True,
                    )
                    try:
                        stdout, stderr = child.communicate(
                            timeout=_remaining_seconds(deadline, 15000.0)
                        )
                    except subprocess.TimeoutExpired:
                        _terminate_process_group(child)
                        stdout, stderr = child.communicate()
                        with (output / f"{name}.log").open("x") as stream:
                            stream.write(stderr)
                        if deadline is not None and time.perf_counter() >= deadline:
                            raise GridBenchmarkError(
                                "configured elapsed budget reached; no complete result"
                            )
                        raise GridBenchmarkError(
                            f"cold worker exceeded the bounded timeout for {name}"
                        )
                    except BaseException:
                        _terminate_process_group(child)
                        raise
                    elapsed = time.perf_counter() - started
                    with (output / f"{name}.log").open("x") as stream:
                        stream.write(stderr)
                    if child.returncode != 0:
                        raise GridBenchmarkError(
                            f"cold worker failed for {name}; retained log"
                        )
                    record = json.loads(stdout)
                else:
                    if worker is None:  # pragma: no cover - guarded by block setup
                        raise GridBenchmarkError("warm worker is unavailable")
                    started = time.perf_counter()
                    worker.stdin.write(json.dumps(job) + "\n")
                    worker.stdin.flush()
                    response = _readline_before(
                        worker, deadline, description=f"warm case {name}"
                    )
                    elapsed = time.perf_counter() - started
                    record = json.loads(response)
                if deadline is not None and time.perf_counter() > deadline:
                    raise GridBenchmarkError(
                        "configured elapsed budget reached; no complete result"
                    )
                record["wall_seconds"] = elapsed
                record["case_path"] = str(directory.relative_to(output))
                write_once(directory / "timed-result.json", record)
                rows.append(
                    {
                        "path": f"{record['case_path']}/timed-result.json",
                        "sha256": digest(directory / "timed-result.json"),
                    }
                )
                print(
                    f"grid {role}: {position + 1}/{len(jobs)} {name} "
                    f"{elapsed:.2f}s valid={record['valid_starts']}/"
                    f"{record['starts_requested']}",
                    flush=True,
                )
        finally:
            if worker is not None:
                _stop_persistent_worker(worker, deadline)
            if worker_log is not None:
                worker_log.close()

        elapsed_total = time.perf_counter() - overall_started
        if role == "full" and elapsed_total > SETTINGS["full_run_budget_seconds"]:
            raise GridBenchmarkError("configured elapsed budget exceeded; no complete result")
        expected_setups = {
            (mode, fold)
            for mode in MODES
            for fold in {int(job["fold"]) for job in jobs}
        }
        if {(row["timing_mode"], row["fold"]) for row in setups} != expected_setups:
            raise GridBenchmarkError("execution did not complete every fold-mode setup block")
        payload = {
            "schema": SCHEMA,
            "phase": "blind_execution_and_fit_selection_complete",
            "execution_complete": True,
            "role": role,
            "lock_sha256": digest(lock_path),
            "reference_records_opened": False,
            "rows": rows,
            "worker_setup": setups,
            "warm_worker_setup": [
                row for row in setups if row["timing_mode"] == "warm"
            ],
            "execution_elapsed_seconds": elapsed_total,
        }
        if role == "pilot":
            pilot_rows = [read_json(output / item["path"]) for item in rows]
            payload["budget"] = _pilot_budget(lock, pilot_rows, elapsed_total)
        else:
            payload["pilot_budget_authorization"] = pilot_authorization
            payload["pilot_execution"] = pilot_execution_binding
        write_once(output / "execution.json", payload)
        return payload
    except BaseException as exc:
        diagnostic = {
            "schema": SCHEMA,
            "phase": "execution_interrupted_no_scoreable_execution",
            "execution_complete": False,
            "role": role,
            "lock_sha256": digest(lock_path),
            "reference_records_opened": False,
            "reason": f"{type(exc).__name__}: {exc}",
            "current_case": current_name,
            "completed_case_count": len(rows),
            "expected_case_count": len(jobs),
            "rows": rows,
            "worker_setup": setups,
            "execution_elapsed_seconds": time.perf_counter() - overall_started,
            "full_run_budget_seconds": SETTINGS["full_run_budget_seconds"],
        }
        if pilot_execution_binding is not None:
            diagnostic["pilot_execution"] = pilot_execution_binding
            diagnostic["pilot_budget_authorization"] = pilot_authorization
        write_once(output / "execution-interrupted.json", diagnostic)
        raise


def _contained(root: Path, relative: str) -> Path:
    path = (root / relative).resolve(strict=True)
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise GridBenchmarkError("artifact path escapes execution root")
    return path


def validate_execution(
    path: Path, lock_path: Path, *, require_environment: bool = True
) -> dict:
    lock = (
        validate_lock(lock_path)
        if require_environment
        else validate_lock(lock_path, require_environment=False)
    )
    document = read_json(path)
    if (document.get("schema") != SCHEMA or document.get("lock_sha256") != digest(lock_path)
            or document.get("phase") != "blind_execution_and_fit_selection_complete"
            or document.get("reference_records_opened") is not False
            or document.get("execution_complete") is not True):
        raise GridBenchmarkError("execution identity/phase mismatch")
    root = path.resolve().parent
    expected = {(j["object_id"], j["repeat_index"], j["arm"], j["timing_mode"]): j
                for j in _jobs(lock, document["role"])}
    seen, values = set(), []
    for binding in document["rows"]:
        case_path = _contained(root, binding["path"])
        if digest(case_path) != binding["sha256"]:
            raise GridBenchmarkError("timed record checksum changed")
        row = read_json(case_path)
        key = tuple(row[k] for k in ("object_id", "repeat_index", "arm", "timing_mode"))
        if key not in expected or key in seen or row["fold"] != expected[key]["fold"]:
            raise GridBenchmarkError("unexpected or duplicate execution case")
        seen.add(key)
        selected_result = read_json(case_path.parent / "selected-result.json")
        if any(row[k] != v for k, v in selected_result.items()):
            raise GridBenchmarkError("timed result differs from written selection")
        axes = None
        if row["arm"].startswith("guided"):
            if digest(case_path.parent / "prediction.json") != row["prediction_sha256"]:
                raise GridBenchmarkError("prediction checksum changed")
            prediction = read_json(case_path.parent / "prediction.json")
            if prediction["object_id"] != row["object_id"] or prediction["fold"] != row["fold"]:
                raise GridBenchmarkError("prediction identity changed")
            axes = [a["axis_xyz"] for a in prediction["axes"]]
        starts = starts_for_arm(row["arm"], axes)
        if len(starts) != row["starts_requested"] or len(row["cell_files"]) != len(starts):
            raise GridBenchmarkError("missing requested native starts")
        fits = []
        for index, vector in enumerate(starts):
            relative = f"start-{index:03d}/record.json"
            cell_path = _contained(case_path.parent, relative)
            if digest(cell_path) != row["cell_files"].get(relative):
                raise GridBenchmarkError("native record checksum changed")
            cell = read_json(cell_path)
            if cell["start_index"] != index or not np.allclose(cell["initial_axis"], vector, rtol=0, atol=1e-12):
                raise GridBenchmarkError("initial pole differs from frozen search rule")
            for file, checksum in cell["files"].items():
                if digest(_contained(cell_path.parent, file)) != checksum:
                    raise GridBenchmarkError("native product/log checksum changed")
            if cell["selected_fit"] is not None:
                fits.append(cell["selected_fit"])
        best = min(fits, key=lambda f: (f["final_relative_rms"], f["start_index"])) if fits else None
        if row["selected_fit"] != best or row["completed"] != bool(fits) or row["valid_starts"] != len(fits):
            raise GridBenchmarkError("minimum-RMS selection changed")
        values.append(row)
    if seen != set(expected):
        raise GridBenchmarkError("incomplete execution cannot be scored")
    elapsed = document.get("execution_elapsed_seconds")
    if (isinstance(elapsed, bool) or not isinstance(elapsed, (int, float))
            or not math.isfinite(elapsed) or elapsed <= 0):
        raise GridBenchmarkError("invalid execution elapsed time")
    for row in values:
        seconds = row["wall_seconds"]
        if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
                or not math.isfinite(seconds) or seconds <= 0):
            raise GridBenchmarkError("invalid cell wall time")
    if elapsed + 1e-9 < sum(row["wall_seconds"] for row in values):
        raise GridBenchmarkError("execution elapsed time is shorter than serial cells")
    if document["role"] == "full" and elapsed > SETTINGS["full_run_budget_seconds"]:
        raise GridBenchmarkError("full execution exceeded the configured budget")
    expected_setups = {(j["timing_mode"], j["fold"]) for j in expected.values()}
    setups = document.get("worker_setup", [])
    keys = [(row.get("timing_mode"), row.get("fold")) for row in setups]
    if len(keys) != len(set(keys)) or set(keys) != expected_setups:
        raise GridBenchmarkError("missing or duplicated worker setup evidence")
    for setup in setups:
        warm = setup["timing_mode"] == "warm"
        seconds = setup.get("setup_seconds")
        if (setup.get("ready") is not True or setup.get("persistent_worker") is not warm
                or setup.get("fresh_process_per_case") is not (not warm)
                or setup.get("setup_in_cell_timing") is not (not warm)
                or isinstance(seconds, bool) or not isinstance(seconds, (int, float))
                or not math.isfinite(seconds) or (seconds <= 0 if warm else seconds != 0)):
            raise GridBenchmarkError("invalid worker setup evidence")
    if document.get("warm_worker_setup") != [s for s in setups if s["timing_mode"] == "warm"]:
        raise GridBenchmarkError("warm setup evidence disagrees with execution blocks")
    if document["role"] == "pilot":
        if document.get("budget") != _pilot_budget(lock, values, elapsed):
            raise GridBenchmarkError("pilot budget differs from recomputed timing projection")
    else:
        pilot_path = _verify_binding(document["pilot_execution"])
        if read_json(pilot_path).get("role") != "pilot":
            raise GridBenchmarkError("full execution must bind an engineering pilot")
        pilot = validate_execution(
            pilot_path, lock_path, require_environment=require_environment
        )
        authorization = _pilot_budget(lock, pilot["validated_rows"], pilot["execution_elapsed_seconds"])
        if (authorization["within_full_run_budget"] is not True
                or document.get("pilot_budget_authorization") != authorization):
            raise GridBenchmarkError("full execution has invalid pilot authorization")
    return {**document, "validated_rows": values}


def score(*, lock_path: Path, execution_path: Path, output: Path) -> dict:
    from .k3.grid_benchmark_scoring import score_rows
    execution = validate_execution(
        execution_path, lock_path, require_environment=False
    )
    lock = read_json(lock_path)
    ids = {r["object_id"] for r in execution["validated_rows"]}
    references = {}
    with _verify_binding(lock["inputs"]["reference_catalog"]).open() as stream:
        for line in stream:
            entry = json.loads(line)
            if entry["object_id"] in ids:
                references[entry["object_id"]] = [s["vector"] for s in entry["solutions"]]
    normalized = []
    for row in execution["validated_rows"]:
        item = {k: row[k] for k in ("object_id", "fold", "repeat_index", "arm", "timing_mode",
                                   "wall_seconds", "completed")}
        fit = row["selected_fit"]
        item["selected_fit"] = ({k: fit[k] for k in ("axis", "final_relative_rms", "start_index")}
                                if fit is not None else None)
        normalized.append(item)
    results = score_rows(normalized, references,
                         resamples=SETTINGS["bootstrap_resamples"], seed=SETTINGS["bootstrap_seed"])
    payload = {"schema": SCHEMA, "phase": "reference_scoring_after_sealed_selection",
               "role": execution["role"], "execution_sha256": digest(execution_path),
               "lock_sha256": digest(lock_path), "results": results}
    if execution["role"] == "pilot":
        payload["claim_scope"] = "engineering_pilot_only_not_cohort_acceleration_evidence"
        payload["budget"] = execution["budget"]
        # Preserve numerical diagnostics, but never publish a cohort decision
        # from the five development objects used only for engineering timing.
        results["decision"] = {"primary_verdict": "not_applicable_engineering_pilot",
                               "positive_verdict_authorized": False}
        for analysis in results["analyses"].values():
            analysis["decision"] = {"positive_verdict_authorized": False,
                                    "scope": "engineering_pilot_only"}
    write_once(output, payload)
    return payload


def continue_after_pilot(*, lock_path: Path, pilot_path: Path, root: Path) -> dict:
    """Run a budget-authorized full execution before opening any references."""
    lock = validate_lock(lock_path)
    pilot = validate_execution(pilot_path, lock_path)
    if pilot["role"] != "pilot":
        raise GridBenchmarkError("continuation requires a validated pilot execution")
    authorization = _pilot_budget(
        lock, pilot["validated_rows"], float(pilot["execution_elapsed_seconds"])
    )
    root = root.resolve(strict=True)
    start = {
        "schema": SCHEMA,
        "phase": "continuation_authorized_before_full_execution",
        "lock_sha256": digest(lock_path),
        "pilot_execution": _binding(pilot_path),
        "pilot_budget_authorization": authorization,
        "reference_records_opened": False,
    }
    write_once(root / "continuation-start.json", start)
    if authorization["within_full_run_budget"] is not True:
        pilot_score = root / "pilot-score.json"
        score(lock_path=lock_path, execution_path=pilot_path, output=pilot_score)
        stopped = {
            **start,
            "phase": "continuation_stopped_by_configured_budget",
            "pilot_score": _binding(pilot_score),
        }
        write_once(root / "continuation-stop.json", stopped)
        return stopped
    full_directory = root / "full"
    full = execute(
        lock_path=lock_path,
        output=full_directory,
        role="full",
        pilot_path=pilot_path,
    )
    full_execution = full_directory / "execution.json"
    full_score = root / "full-score.json"
    pilot_score = root / "pilot-score.json"
    score(lock_path=lock_path, execution_path=full_execution, output=full_score)
    score(lock_path=lock_path, execution_path=pilot_path, output=pilot_score)
    complete = {
        **start,
        "phase": "full_execution_and_post_selection_scoring_complete",
        "full_execution": _binding(full_execution),
        "full_score": _binding(full_score),
        "pilot_score": _binding(pilot_score),
        "full_case_count": len(full["rows"]),
    }
    write_once(root / "continuation-complete.json", complete)
    return complete
