#!/usr/bin/env python3
"""POST HOC spacecraft classical inversion, PROTOCOL.md Addendum 2.

Order: validate, export, run, score. All products live under --output-dir.
Run with PYTHONPATH=<worktree> <pinned python> -I <this file> <command>.
-I ignores PYTHONPATH; this script explicitly adds only its own checkout.

export writes DAMIT's count / (point-count, calibrated-flag) / eight-column
format. Flag 0 means relative, 1 absolute. Prepared session fluxes are relative.
The columns are JD, brightness, asteroid-to-Sun XYZ, asteroid-to-observer XYZ,
in ecliptic J2000 AU. No signs, frames, normalization, times or geometry change.
The adapter already negated Horizons center-to-asteroid vectors. MapCam's
observer is OSIRIS-REx, even though the C source calls that column Earth.

run calls grid_benchmark._native_start, without opening reference poles or
DeLPHI scores. Native record.json files and products retain the benchmark
layout; diagnostics.json adds explicit convergence, chi2 and final-pole fields.
Completed starts are hash-verified and reused. Interrupted starts are preserved
under interrupted/ before retrying; no file is overwritten. An advisory lock
prevents simultaneous runs in one output directory. Workers may change on resume.

--runs, --periods and --max-starts on run permit CPU smoke tests. score requires
the complete grid unless --allow-partial is explicit; partial reports are marked
SMOKE TEST throughout and cannot be mistaken for Addendum 2 results.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import fcntl
import hashlib
import json
import math
import os
import resource
import struct
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
WORKTREE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORKTREE))
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np  # noqa: E402

from lc_pipeline import grid_benchmark as grid  # noqa: E402
from lc_pipeline.k3.solver_capacity import (  # noqa: E402
    capacity_violations, parse_lightcurve_structure, read_static_capacities, tree_sha256,
)
from lc_pipeline.v2.convexinv import (  # noqa: E402
    _FINAL, _ITERATION, _parse_lightcurve_brightness,
)
from lc_pipeline.v2.preprocessing import Observation, ObservationEpoch  # noqa: E402

from repro.own_period import DataPath
from repro.network_root import (NetworkPath, check_json, check_jsonl, guard_output,
                                json_metadata, switched)
from repro.own_period.spacecraft import run as network_run

OWN = DataPath()
SPACECRAFT = NetworkPath("results", "spacecraft", legacy_path=OWN / "spacecraft")
BENCHMARK = OWN / "benchmark"
OUTPUT = SPACECRAFT / "classical"
ARMS = ("standard6", "classical20")
MODES = ("own", "catalogue")
PRIMARY = ("itokawa-combined", "ryugu-combined", "bennu-combined",
           "lutetia-damit", "ida-damit", "gaspra-damit")
POSTHOC = ("ryugu-combined-grond", "bennu-ground-all", "bennu-mapcam", "bennu-all")
RUNS = PRIMARY + POSTHOC
SOLVER_SHA256 = "4df011a50493fc09033177f504370d85773644062fa4ee955ecef626f05275b5"
IMPLEMENTATIONS = ("lc_pipeline/grid_benchmark.py", "lc_pipeline/v2/convexinv.py",
                   "lc_pipeline/v2/data.py", "lc_pipeline/v2/preprocessing.py",
                   "lc_pipeline/k3/solver_capacity.py")
CONVENTION = {
    "columns": ["JD", "brightness", "asteroid_to_Sun_x", "asteroid_to_Sun_y",
                "asteroid_to_Sun_z", "asteroid_to_observer_x", "asteroid_to_observer_y",
                "asteroid_to_observer_z"],
    "frame": "ecliptic J2000", "vector_units": "AU", "origin": "asteroid",
    "prepared_flag": 0, "flag_meaning": "0 relative; 1 calibrated/absolute",
    "geometry": "prepared vectors copied verbatim; no negation, rotation or unit normalization",
    "photometry": "prepared brightness and native sessions unchanged; no extra corrections",
    "uncertainties": "not columns in DAMIT format; benchmark uses per-curve mean brightness as sigma",
    "evidence": ["convexinv.c stdin fscanf comments and Inrel = 1 - i_temp",
                 "v2/data.py parse_damit_lightcurve maps columns 2:5 and 5:8 directly",
                 "prepare_k3_alcdef_transfer.py negates Horizons vectors before preparing fields",
                 "ztf_external.py Horizons REF_PLANE=ECLIPTIC, REF_SYSTEM=ICRF, OUT_UNITS=AU-D"],
}


def read(path: Path) -> dict:
    return grid.read_json(path)


def binding(path: Path) -> dict:
    return {"path": str(path.resolve(strict=True)), "sha256": grid.digest(path)}


def checked(entry: dict, root: Path | None = None) -> Path:
    path = Path(entry["path"]) if root is None else root / entry["path"]
    path = path.resolve(strict=True)
    if root is not None and not path.is_relative_to(root.resolve()):
        raise ValueError(f"bound path escapes output: {path}")
    if grid.digest(path) != entry["sha256"]:
        raise ValueError(f"bound file changed: {path}")
    return path


def relative_binding(path: Path, root: Path) -> dict:
    return {"path": path.relative_to(root).as_posix(), "sha256": grid.digest(path)}


def available(path: Path) -> None:
    guard_output(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite: {path}")


def write(path: Path, value: dict) -> None:
    guard_output(path)
    grid.write_once(path, {**value, **json_metadata()})


def write_or_verify(path: Path, value: dict) -> None:
    if path.exists():
        if read(path) != value:
            raise ValueError(f"existing receipt differs: {path}")
    else:
        write(path, value)


def unique(rows: list[dict], key: str, value: str) -> dict:
    matches = [row for row in rows if row[key] == value]
    if len(matches) != 1:
        raise ValueError(f"expected one {key}={value}, got {len(matches)}")
    return matches[0]


def period(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError(f"invalid period: {value}")
    return float(value)


def benchmark(path: Path) -> dict:
    """Read only the pole-free execution lock, solver and numerical sources."""
    lock = read(path)
    for name in IMPLEMENTATIONS:
        if grid.digest(WORKTREE / name) != lock["runtime_sources"][name]:
            raise ValueError(f"numerical source differs from published benchmark: {name}")
    settings = lock["solver_settings"]
    for key in ("convergence_tolerance", "iteration_cap", "timeout_seconds_per_start"):
        if settings[key] != grid.SETTINGS[key]:
            raise ValueError(f"benchmark setting differs: {key}")
    solver = lock["solver"]
    executable = checked(solver["executable"])
    if grid.digest(executable) != SOLVER_SHA256:
        raise ValueError("Addendum 2 requires expanded convexinv sha 4df011a5")
    checked(solver["capacity_report"])
    if (tree_sha256(solver["source_root"]) != solver["source_tree_sha256"]
            or read_static_capacities(solver["source_root"]) != solver["capacities"]):
        raise ValueError("published solver tree/capacities changed")
    # _native_start rejects capped fits; the bound binary itself enforces this
    # limit during optimization. Check both parts of the benchmark contract.
    header = (Path(solver["source_root"]) / "constants.h").read_text(encoding="ascii")
    cap = [int(line.split()[2]) for line in header.splitlines()
           if line.split()[:2] == ["#define", "MAX_N_ITER"]]
    if cap != [settings["iteration_cap"]]:
        raise ValueError("compiled iteration cap differs from benchmark acceptance cap")
    if len(grid.starts_for_arm("standard6")) != 6 or len(grid.pole_grid(20)) != 146:
        raise ValueError("published start grid changed")
    return lock


def lc_payload(path: Path) -> tuple[list[int], list[list[list[float]]]]:
    """Independent token framing for flags and exact eight-column comparisons."""
    tokens = path.read_text(encoding="ascii").split()
    count, cursor, flags, curves = int(tokens[0]), 1, [], []
    if count <= 0:
        raise ValueError("empty lightcurve file")
    for _ in range(count):
        n, flag = int(tokens[cursor]), int(tokens[cursor + 1])
        cursor += 2
        if n <= 0 or flag not in (0, 1):
            raise ValueError("invalid lightcurve header")
        rows = []
        for _ in range(n):
            row = [float(x) for x in tokens[cursor:cursor + 8]]
            if len(row) != 8 or not all(math.isfinite(x) for x in row):
                raise ValueError("invalid lightcurve observation")
            cursor += 8
            rows.append(row)
        curves.append(rows)
        flags.append(flag)
    if cursor != len(tokens):
        raise ValueError("trailing lightcurve tokens")
    return flags, curves


def observation_rows(epochs) -> list[list[list[float]]]:
    return [[[o.time_jd, o.relative_brightness, *o.sun_asteroid_ecliptic_j2000_au,
              *o.observer_asteroid_ecliptic_j2000_au] for o in e.observations] for e in epochs]


def write_lightcurves(path: Path, epochs, flags: list[int]) -> dict:
    """One lossless writer for prepared, DAMIT and validation observations."""
    curves = observation_rows(epochs)
    if not curves or len(curves) != len(flags) or any(f not in (0, 1) for f in flags):
        raise ValueError("invalid epoch/flag framing")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="ascii") as stream:
        stream.write(f"{len(curves)}\n")
        for rows, flag in zip(curves, flags, strict=True):
            if not rows:
                raise ValueError("empty observation epoch")
            stream.write(f"{len(rows)} {flag}\n")
            for row in rows:
                if not all(math.isfinite(x) for x in row):
                    raise ValueError("nonfinite observation")
                stream.write(" ".join(format(x, ".17g") for x in row) + "\n")
    if lc_payload(path) != (flags, curves):
        raise ValueError("writer changed an observation or header")
    return {"epochs": len(curves), "observations": sum(map(len, curves)),
            "exact_float64_round_trip": True, "calibrated_flags": flags}


def round_trip(row: dict, target: Path) -> dict:
    # Exactly the DAMIT loader used by spacecraft/run.py. Imports torch on CPU
    # through v2.data, but never constructs a model or initializes CUDA.
    from lc_pipeline.v2.data import parse_damit_lightcurve

    native = checked(row["lightcurve"])
    flags, curves = lc_payload(native)
    epochs = parse_damit_lightcurve(native)
    if observation_rows(epochs) != curves:
        raise ValueError("DAMIT loader changed native observation columns")
    result = write_lightcurves(target, epochs, flags)
    if lc_payload(target) != (flags, curves):
        raise ValueError("export differs from benchmark's native solver input")
    def numeric_digest(path):
        return hashlib.sha256(b"".join(struct.pack("<d", float(t))
                                      for t in path.read_text().split())).hexdigest()
    a, b = numeric_digest(native), numeric_digest(target)
    if a != b:
        raise ValueError("solver numeric input differs")
    return {"status": "PASS", "object_id": row["object_id"], **result,
            "benchmark_input": binding(native), "exported_input": binding(target),
            "native_numeric_sha256": a, "exported_numeric_sha256": b,
            "byte_identical": native.read_bytes() == target.read_bytes(),
            "equality": "every count, flag and float64 column identical; whitespace/decimal spelling may differ",
            "max_absolute_column_difference": 0.0, "convention": CONVENTION}


def source_inputs(spacecraft: Path) -> list[dict]:
    """Bind only the used observations/periods, without opening pole records.

    The primary protocol was extended after its lock. Check actual data and
    period bindings, rather than requiring that obsolete protocol hash.
    """
    result = []
    for run_id in RUNS:
        base = spacecraft if run_id in PRIMARY else spacecraft / "posthoc"
        lock_path, search_path = base / "lock/cohort-lock.json", base / "search/periods.json"
        lock, search = read(lock_path), read(search_path)
        prepared_path = base / "prepared/manifest.json"
        prepared = read(prepared_path)
        if (search["cohort_lock_sha256"] != grid.digest(lock_path)
                or search["prepared_manifest_sha256"] != grid.digest(prepared_path)
                or prepared["cohort_lock_sha256"] != grid.digest(lock_path)):
            raise ValueError(f"search/prepared lock differs: {run_id}")
        meta = unique(lock["inputs"], "run_id", run_id)
        dependencies = [binding(lock_path), binding(search_path), binding(prepared_path)]
        if meta["kind"] == "external":
            entry = unique(prepared["objects"], "run_id", run_id)
            path = checked(entry, base / "prepared")
            value = read(path)
            if value["schema"] != "delphi.k3-external-prepared.v1" or value["object_id"] != meta["object_id"]:
                raise ValueError("prepared identity/schema differs")
            manifest_path = checked(entry["source_manifest"], base / "prepared")
            source = read(manifest_path)
            receipt_path = base / "sbdb/periods.json"
            receipt = read(receipt_path)
            if (source["objects"] != [value]
                    or source["period_receipts_sha256"] != grid.digest(receipt_path)
                    or receipt["cohort_lock_sha256"] != grid.digest(lock_path)):
                raise ValueError("prepared object or period receipt changed")
            cat = unique(receipt["objects"], "object_id", meta["object_id"])
            catalogue = period(cat["known_period_hours"])
            if cat["status"] != "ready" or catalogue != period(value["known_period_hours"]):
                raise ValueError("prepared catalogue period differs from SBDB receipt")
            catalogue_source = {"rule": "SBDB rot_per receipt (as prepared for DeLPHI)",
                                "receipt": binding(receipt_path)}
            epochs = tuple(ObservationEpoch(e["epoch_id"], tuple(Observation(**o) for o in e["observations"]))
                           for e in value["epochs"])
            flags = [0] * len(epochs)
            dependencies.extend([binding(manifest_path), binding(receipt_path)])
        else:
            from lc_pipeline.v2.data import parse_damit_lightcurve
            path = checked(meta["source"])
            epochs = parse_damit_lightcurve(path)
            flags, _ = lc_payload(path)
            table_path = checked(lock["files"]["model_table"])
            # Retain only IDs and periods; no lambda/beta fields are accessed.
            candidates = []
            with table_path.open(encoding="utf-8-sig", newline="") as stream:
                for r in csv.DictReader(stream):
                    if f"asteroid_{r['asteroid_id']}" == meta["object_id"]:
                        candidates.append((int(r["id"]), period(r["period"])))
            model_id, catalogue = min(candidates)
            catalogue_source = {"rule": "DAMIT lowest numeric model id, as run.py",
                                "model_id": model_id, "table": binding(table_path)}
            dependencies.append(binding(table_path))
        if search["details"][run_id]["input_sha256"] != grid.digest(path):
            raise ValueError(f"observations differ from DeLPHI period search: {run_id}")
        result.append({"run_id": run_id, "name": meta["name"], "object_id": meta["object_id"],
                       "primary_set": run_id in PRIMARY, "kind": meta["kind"],
                       "input": binding(path), "dependencies": dependencies,
                       "periods_hours": {"own": period(search["periods_hours"][run_id]), "catalogue": catalogue},
                       "catalogue_source": catalogue_source, "epochs": epochs, "flags": flags})
    return result


def cmd_export(args) -> dict:
    root = args.output_dir / "export"
    available(root)
    lock = benchmark(args.benchmark_dir / "evaluation-lock.json")
    rows = source_inputs(args.spacecraft_dir)
    root.mkdir(parents=True)
    check_row = unique(lock["objects"], "object_id", "asteroid_1733")
    proof = round_trip(check_row, root / "round-trip/lc.txt")
    write(root / "round-trip.json", proof)
    entries = []
    for row in rows:
        path = root / "inputs" / f"{row['run_id']}.txt"
        stats = write_lightcurves(path, row["epochs"], row["flags"])
        if row["kind"] == "damit" and lc_payload(path) != lc_payload(checked(row["input"])):
            raise ValueError("DAMIT export differs from native benchmark input")
        structure = parse_lightcurve_structure(path)
        violations = capacity_violations(structure, lock["solver"]["capacities"])
        if violations:
            raise ValueError(f"solver capacity exceeded for {row['run_id']}: {violations}")
        entries.append({k: v for k, v in row.items() if k not in ("epochs", "flags")} |
                       {"lightcurve": relative_binding(path, args.output_dir), **stats,
                        "structure": structure.as_dict()})
    write(root / "manifest.json", {"schema": "delphi.spacecraft-classical-export.v1", "posthoc": True,
          "protocol": binding(args.spacecraft_dir / "PROTOCOL.md"), "script": binding(Path(__file__)),
          "benchmark_lock": binding(args.benchmark_dir / "evaluation-lock.json"),
          "round_trip": relative_binding(root / "round-trip.json", args.output_dir),
          "convention_sources": [binding(WORKTREE / p) for p in (
              "lc_pipeline/v2/data.py", "lc_pipeline/k3/ztf_external.py",
              "repro/prepare_k3_alcdef_transfer.py", "repro/own_period/spacecraft/posthoc.py")] +
              [binding(Path(lock["solver"]["source_root"]) / "convexinv.c")],
          "convention": CONVENTION, "inputs": entries})
    return {"stage": "export", "inputs": len(entries), "round_trip": proof["status"],
            "observations": sum(r["observations"] for r in entries)}


def start_diagnostics(record: dict, target: Path) -> dict:
    stdout = (target / "stdout.txt").read_text(encoding="utf-8", errors="replace")
    iterations, final = list(_ITERATION.finditer(stdout)), _FINAL.search(stdout)
    def finite(value):
        number = float(value)
        return number if math.isfinite(number) else None
    fit = record["selected_fit"]
    return {"converged": fit is not None, "convergence_definition": "benchmark valid selected_fit",
            "chi2": finite(iterations[-1].group(2)) if iterations else None,
            "deviation": finite(iterations[-1].group(3)) if iterations else None,
            "final_longitude_deg": finite(final.group(1)) if final else None,
            "final_latitude_deg": finite(final.group(2)) if final else None,
            "final_period_hours": finite(final.group(3)) if final else None,
            "final_relative_rms": fit["final_relative_rms"] if fit else None,
            "pole": fit["axis"] if fit else None, "failure_reason": record["failure_reason"]}


def verify_record(target: Path, index: int, vector) -> dict:
    record = read(target / "record.json")
    if record["start_index"] != index or record["initial_axis"] != vector.tolist():
        raise ValueError(f"start identity differs: {target}")
    for name, digest in record["files"].items():
        checked({"path": name, "sha256": digest}, target)
    return record


def execute_start(job: dict, index: int, vector, lock: dict, case: Path) -> dict:
    """Shared path for validation and the full run, including native receipts."""
    case.mkdir(parents=True, exist_ok=True)
    target = case / f"start-{index:03d}"
    if (target / "record.json").exists():
        record = verify_record(target, index, vector)
    else:
        if target.exists():
            # Preserve an interrupted process's bytes before a fresh attempt.
            archive = case / "interrupted"
            archive.mkdir(exist_ok=True)
            attempt = 0
            while (archive / f"{target.name}-{attempt:03d}").exists():
                attempt += 1
            target.rename(archive / f"{target.name}-{attempt:03d}")
        observed = _parse_lightcurve_brightness(checked(job["lightcurve"]))
        record = grid._native_start(job, index, vector, observed, lock["solver"], lock["solver_settings"], case)
    write_or_verify(target / "diagnostics.json", start_diagnostics(record, target))
    return record


def cmd_validate(args) -> dict:
    root = args.output_dir / "validation"
    available(root)
    lock_path = args.benchmark_dir / "evaluation-lock.json"
    lock = benchmark(lock_path)
    row = unique(lock["objects"], "object_id", args.object_id)
    vectors = grid.starts_for_arm(args.arm)
    if not 0 <= args.start_index < len(vectors):
        raise ValueError("validation start index outside chosen arm")
    published = (args.benchmark_dir / "full-grid/cases" /
                 f"{args.object_id}-r0-{args.arm}-cold" / f"start-{args.start_index:03d}")
    expected = verify_record(published, args.start_index, vectors[args.start_index])
    root.mkdir(parents=True)
    proof = round_trip(row, root / "lc.txt")
    write(root / "round-trip.json", proof)
    job = {**row, "lightcurve": binding(root / "lc.txt")}
    actual = execute_start(job, args.start_index, vectors[args.start_index], lock,
                           root / "cases" / f"{args.object_id}-r0-{args.arm}-cold")
    # Timing is not deterministic. Require every other field, every product
    # byte, all printed diagnostics and the computed RMS to be identical.
    stable_actual = {k: v for k, v in actual.items() if k != "solver_wall_seconds"}
    stable_expected = {k: v for k, v in expected.items() if k != "solver_wall_seconds"}
    passed = actual["selected_fit"] is not None and stable_actual == stable_expected
    result = {"schema": "delphi.spacecraft-classical-validation.v1", "status": "PASS" if passed else "FAIL",
              "script": binding(Path(__file__)), "benchmark_lock": binding(lock_path),
              "published_record": binding(published / "record.json"), "object_id": args.object_id,
              "arm": args.arm, "start_index": args.start_index, "round_trip": proof,
              "bit_exact_except_wall_time": stable_actual == stable_expected,
              "comparison": "exact record equality excluding solver_wall_seconds; all product hashes exact",
              "expected_fit": expected["selected_fit"], "actual_fit": actual["selected_fit"],
              "matching_product_hashes": actual["files"] == expected["files"],
              "differing_fields": [k for k in stable_expected if stable_actual.get(k) != stable_expected[k]],
              "observed_wall_seconds": actual["solver_wall_seconds"]}
    write(root / "result.json", result)
    if not passed:
        raise ValueError(f"published full-grid fit not reproduced; see {root / 'result.json'}")
    return {k: result[k] for k in ("status", "object_id", "arm", "start_index",
                                  "bit_exact_except_wall_time", "matching_product_hashes", "observed_wall_seconds")}


def load_export(out: Path) -> dict:
    manifest = read(out / "export/manifest.json")
    if checked(manifest["script"]) != Path(__file__).resolve() and not switched():
        raise ValueError("export used another script")
    network_run.checked_append_only(manifest["protocol"])
    for entry in manifest["convention_sources"]:
        checked(entry)
    proof = read(checked(manifest["round_trip"], out))
    if proof["status"] != "PASS":
        raise ValueError("conversion round trip required")
    checked(proof["benchmark_input"])
    checked(proof["exported_input"])
    if [r["run_id"] for r in manifest["inputs"]] != list(RUNS):
        raise ValueError("export does not contain exactly the Addendum 2 inputs")
    for row in manifest["inputs"]:
        checked(row["input"])
        checked(row["lightcurve"], out)
        for dep in row["dependencies"]:
            checked(dep)
    return manifest


def case_path(out: Path, run_id: str, mode: str, arm: str) -> Path:
    return out / mode / "cases" / f"{run_id}-r0-{arm}-cold"


def run_plan(args, manifest: dict, lock: dict) -> dict:
    validation_path = args.output_dir / "validation/result.json"
    validation = read(validation_path)
    if (validation["status"] != "PASS" or validation["script"] != binding(Path(__file__))
            or validation["benchmark_lock"] != manifest["benchmark_lock"]
            or manifest["benchmark_lock"] != binding(args.benchmark_dir / "evaluation-lock.json")):
        raise ValueError("validate and export must use this script and the same benchmark lock")
    cases, rows = [], []
    for row in manifest["inputs"]:
        if row["run_id"] not in args.runs:
            continue
        rows.append(row["run_id"])
        for mode in MODES:
            if mode not in args.periods:
                continue
            remaining = args.max_starts if args.max_starts is not None else 152
            for arm in ARMS:
                count = min(len(grid.starts_for_arm(arm)), remaining)
                remaining -= count
                if count:
                    cases.append({"run_id": row["run_id"], "mode": mode, "arm": arm,
                                  "period_hours": row["periods_hours"][mode], "starts": count})
    return {"schema": "delphi.spacecraft-classical-run.v1", "posthoc": True,
            "export": binding(args.output_dir / "export/manifest.json"),
            "validation": binding(validation_path), "script": binding(Path(__file__)),
            "implementations": {name: binding(WORKTREE / name) for name in IMPLEMENTATIONS},
            "benchmark_lock": manifest["benchmark_lock"], "solver": lock["solver"],
            "solver_settings": lock["solver_settings"], "runs": rows,
            "periods": [m for m in MODES if m in args.periods], "cases": cases,
            "full_grid": args.max_starts is None or args.max_starts == 152,
            "reference_poles_opened": False,
            "period_policy": "exact DeLPHI periods; no rounding or period refitting",
            "tie_break": "RMS, standard6 before classical20, start index"}


def cmd_run(args) -> dict:
    out = args.output_dir
    manifest = load_export(out)
    lock = benchmark(args.benchmark_dir / "evaluation-lock.json")
    plan = run_plan(args, manifest, lock)
    plan_path = out / "run.json"
    write_or_verify(plan_path, plan)
    # flock modifies no file bytes; it is released even after SIGKILL.
    with plan_path.open("rb") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("another run is active in this output directory") from exc
        receipts = []
        for spec in plan["cases"]:
            row = unique(manifest["inputs"], "run_id", spec["run_id"])
            case = case_path(out, spec["run_id"], spec["mode"], spec["arm"])
            job = {"object_id": row["object_id"], "period_hours": spec["period_hours"],
                   "lightcurve": binding(checked(row["lightcurve"], out))}
            case.mkdir(parents=True, exist_ok=True)
            write_or_verify(case / "job.json", {**job, **spec, "run_plan": binding(plan_path)})
            records = []
            if spec["period_hours"] is not None:
                vectors = grid.starts_for_arm(spec["arm"])[:spec["starts"]]
                with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
                    futures = [pool.submit(execute_start, job, i, v, lock, case) for i, v in enumerate(vectors)]
                    # Emit progress at each completion, including long cases.
                    for future in concurrent.futures.as_completed(futures):
                        records.append(future.result())
                        r = records[-1]
                        print(f"{spec['run_id']} {spec['mode']} {spec['arm']} start-{r['start_index']:03d} "
                              f"converged={r['selected_fit'] is not None}", flush=True)
                records.sort(key=lambda r: r["start_index"])
            fits = [r["selected_fit"] for r in records if r["selected_fit"] is not None]
            result = {**spec, "object_id": row["object_id"], "valid": len(fits),
                      "status": "no_period" if spec["period_hours"] is None else "attempted",
                      "selected_fit": min(fits, key=lambda f: (f["final_relative_rms"], f["start_index"])) if fits else None,
                      "solver_wall_seconds": sum(r["solver_wall_seconds"] for r in records),
                      "records": [{"record": relative_binding(case / f"start-{r['start_index']:03d}/record.json", out),
                                   "diagnostics": relative_binding(case / f"start-{r['start_index']:03d}/diagnostics.json", out)}
                                  for r in records]}
            write_or_verify(case / "case.json", result)
            receipts.append(relative_binding(case / "case.json", out))
        execution = {"schema": "delphi.spacecraft-classical-execution.v1", "posthoc": True,
                     "run_plan": binding(plan_path), "cases": receipts, "full_grid": plan["full_grid"],
                     "reference_poles_opened": False, "status": "all_planned_starts_attempted"}
        write_or_verify(out / "execution.json", execution)
    return {"stage": "run", "cases": len(receipts), "full_grid": plan["full_grid"],
            "workers": args.workers, "reference_poles_opened": False}


def angle(a, b, axial: bool = False) -> float:
    av, bv = grid.unit_vectors([a, b])
    dot = float(np.dot(av, bv))
    return math.degrees(math.acos(float(np.clip(abs(dot) if axial else dot, -1, 1))))


def fit_key(fit: dict) -> tuple:
    return fit["final_relative_rms"], ARMS.index(fit["arm"]), fit["start_index"]


def outcomes(fits: list[dict], reference, axial: bool) -> dict:
    """RMS-ordered greedy representatives are pairwise more than 20 deg apart.

    This follows workflow_benchmark.cluster_diagnostic's representative rule,
    separately for directed poles and axial poles. Representatives never use
    the reference; only the subsequent error/oracle calculation does.
    """
    ordered = sorted(fits, key=fit_key)
    if not ordered:
        return {"a_selected": None, "b_ambiguity": {"distinct_minima": 0, "oracle": None,
                "representatives": [], "near_best_fits": 0}, "c_near_reference": None}
    best = ordered[0]
    def measured(fit):
        return {**fit, "error_deg": angle(fit["axis"], reference, axial),
                "spin_error_deg": angle(fit["axis"], reference),
                "axial_error_deg": angle(fit["axis"], reference, True),
                "rms_relative_to_best": fit["final_relative_rms"] / best["final_relative_rms"]}
    near = [f for f in ordered if f["final_relative_rms"] <= 1.05 * best["final_relative_rms"]]
    representatives = []
    for fit in near:
        if all(angle(fit["axis"], r["axis"], axial) > 20 for r in representatives):
            representatives.append(fit)
    close = [f for f in ordered if angle(f["axis"], reference, axial) <= 20]
    return {"a_selected": measured(best),
            "b_ambiguity": {"distinct_minima": len(representatives), "near_best_fits": len(near),
                            "representatives": [measured(f) for f in representatives],
                            "oracle": measured(min(representatives, key=lambda f: (angle(f["axis"], reference, axial), fit_key(f)))),
                            "all_near_best_oracle": measured(min(near, key=lambda f: (angle(f["axis"], reference, axial), fit_key(f))))},
            "c_near_reference": ({**measured(close[0]), "qualifying_fits": len(close)} if close else None)}


def cmd_score(args) -> dict:
    destination = args.output_dir
    out = getattr(args, "fit_dir", None) or destination
    available(destination / "scores.json")
    available(destination / "RESULTS.md")
    destination.mkdir(parents=True, exist_ok=True)
    execution = read(out / "execution.json")
    plan = read(checked(execution["run_plan"]))
    if (execution["status"] != "all_planned_starts_attempted"
            or execution["reference_poles_opened"] is not False or plan["reference_poles_opened"] is not False):
        raise ValueError("complete reference-blind execution required")
    if not plan["full_grid"] and not args.allow_partial:
        raise ValueError("partial grids require --allow-partial; only for a labelled smoke test")
    manifest = load_export(out)
    if plan["export"] != binding(out / "export/manifest.json") or (not switched() and plan["script"] != binding(Path(__file__))):
        raise ValueError("execution inputs/script changed")
    checked(plan["script"])
    for entry in plan["implementations"].values():
        checked(entry)
    checked(plan["validation"])
    cases = [read(checked(e, out)) for e in execution["cases"]]
    if len(cases) != len(plan["cases"]):
        raise ValueError("missing or duplicate cases")
    grouped = {}
    for spec, case in zip(plan["cases"], cases, strict=True):
        if any(case[k] != v for k, v in spec.items()):
            raise ValueError("case differs from run plan")
        expected_n = 0 if spec["period_hours"] is None else spec["starts"]
        if len(case["records"]) != expected_n:
            raise ValueError("not every planned start has a record")
        fits = grouped.setdefault((spec["run_id"], spec["mode"]), [])
        vectors = grid.starts_for_arm(spec["arm"])
        for i, entry in enumerate(case["records"]):
            record_path = checked(entry["record"], out)
            diagnostics_path = checked(entry["diagnostics"], out)
            record = verify_record(record_path.parent, i, vectors[i])
            if read(diagnostics_path) != start_diagnostics(record, record_path.parent):
                raise ValueError("start diagnostics changed")
            if record["selected_fit"] is not None:
                fits.append({**record["selected_fit"], "arm": spec["arm"],
                             "chi2": read(diagnostics_path)["chi2"], "record": entry["record"]})
    # Reference and DeLPHI score files are first opened here, after fitting.
    reference_path = network_run.input_root(args.spacecraft_dir) / "reference-poles.json"
    reference = read(reference_path)
    if reference["schema"] != "delphi.spacecraft-reference-poles.v1" or reference["ecliptic_frame"] != "ECLIPJ2000":
        raise ValueError("unexpected reference schema/frame")
    refs = {r["object"].lower(): r for r in reference["objects"]}
    score_paths = [args.spacecraft_dir / "scores.json", args.spacecraft_dir / "posthoc/scores.json"]
    for path in score_paths:
        check_json(path)
    delphi = [read(p) for p in score_paths]
    predictions = {}
    for base, document in zip((args.spacecraft_dir, args.spacecraft_dir / "posthoc"), delphi, strict=True):
        if document["reference_poles_sha256"] != grid.digest(reference_path):
            raise ValueError("DeLPHI used different spacecraft reference poles")
        prediction_manifest_path = base / "predictions/manifest.json"
        check_json(prediction_manifest_path)
        if document["predictions_manifest_sha256"] != grid.digest(prediction_manifest_path):
            raise ValueError("DeLPHI score/prediction binding differs")
        prediction_manifest = read(prediction_manifest_path)
        for mode in plan["periods"]:
            path = checked(prediction_manifest["outputs"][mode], base / "predictions")
            check_jsonl(path)
            predictions[base, mode] = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    objects = []
    for (run_id, mode), fits in grouped.items():
        source = unique(manifest["inputs"], "run_id", run_id)
        expected_period = source["periods_hours"][mode]
        comparison = unique(delphi[0 if run_id in PRIMARY else 1]["objects"], "run_id", run_id)
        d = comparison[mode]
        base = args.spacecraft_dir if run_id in PRIMARY else args.spacecraft_dir / "posthoc"
        prediction = unique(predictions[base, mode], "run_id", run_id)
        if (prediction["input_sha256"] != source["input"]["sha256"]
                or prediction["period_hours"] != expected_period):
            raise ValueError(f"DeLPHI prediction used different observations/period: {run_id} {mode}")
        if d["period_hours"] != expected_period or comparison["object_id"] != source["object_id"]:
            raise ValueError(f"DeLPHI input/period differs: {run_id} {mode}")
        if comparison["spacecraft_reference"] != refs[source["name"]]:
            raise ValueError("DeLPHI reference row differs")
        counts = [c for c in cases if c["run_id"] == run_id and c["mode"] == mode]
        objects.append({"run_id": run_id, "mode": mode, "object_id": source["object_id"],
                        "primary_set": source["primary_set"], "period_hours": expected_period,
                        "status": "no_period" if expected_period is None else "ok" if fits else "no_valid_fit",
                        "attempted_starts": sum(len(c["records"]) for c in counts), "valid_starts": len(fits),
                        "requested_starts": sum(c["starts"] for c in counts),
                        "selected_axial_error_deg": (angle(min(fits, key=fit_key)["axis"],
                            refs[source["name"]]["unit_vector"], True) if fits else 90.0),
                        "spin_vector": outcomes(fits, refs[source["name"]]["unit_vector"], False),
                        "axial": outcomes(fits, refs[source["name"]]["unit_vector"], True),
                        "delphi_axial": {k: d[k] for k in ("status", "period_hours", "oracle_deg", "best_scored_deg")},
                        "reference": refs[source["name"]]})
    label = "POST HOC: Addendum 2" if plan["full_grid"] else "POST HOC: SMOKE TEST, PARTIAL GRID"
    result = {"schema": "delphi.spacecraft-classical-scores.v1", "scope": label, "posthoc": True,
              "execution": binding(out / "execution.json"), "reference_poles": binding(reference_path),
              "delphi_scores": [binding(p) for p in score_paths], "full_grid": plan["full_grid"],
              "comparable_metric": "axial angle in degrees (absolute spin-vector dot product)",
              "ambiguity_rule": "RMS <= 1.05 best; ascending RMS/arm/index greedy representatives >20 deg apart; oracle over representatives",
              "c_rule": "lowest RMS among valid final poles within 20 deg of reference; initial pole irrelevant",
              "failure_policy": "no period or valid fit: selected axial error 90 deg, signed angle undefined",
              "aggregation": "none; each input and period listed", "objects": objects}
    grid_description = ("Six standard poles plus the 146-pole 20-degree grid, at the exact DeLPHI periods."
                        if plan["full_grid"] else "SMOKE TEST ONLY: incomplete grid; these are software checks.")
    lines = [f"# Spacecraft classical inversion — {label}", "", grid_description,
             "convexinv is a signed spin-vector fit; the axial table is comparable with DeLPHI's axial errors.",
             "Converged means a valid benchmark fit: exit 0, finite products, 0 < iterations < 1000, fixed period check, no timeout (300 s/start).",
             "Ambiguity: RMS within 5% of best, greedy lowest-RMS representatives more than 20 degrees apart.",
             "The ambiguity oracle uses those representatives. scores.json also gives the oracle over every near-best fit.",
             "Outcome (c) selects the lowest-RMS final pole within 20 degrees; n/a means no qualifying fit.",
             "No-period/no-valid-fit selected errors count as 90 axial degrees; signed errors are undefined.", ""]
    def fmt(v):
        return "n/a" if v is None else f"{v:.6f}"
    for metric, title in (("axial", "Axial angles (comparable)"), ("spin_vector", "Signed spin-vector angles")):
        lines += [f"## {title}", "",
                  "| Input | Set | Period | P (h) | Valid/attempted | (a) selected error | (b) minima | (b) oracle error | (c) error | (c) RMS/best | DeLPHI axial oracle | DeLPHI axial best |",
                  "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for row in objects:
            m, d = row[metric], row["delphi_axial"]
            a, b, c = m["a_selected"], m["b_ambiguity"], m["c_near_reference"]
            cells = [row["run_id"], "primary" if row["primary_set"] else "post hoc", row["mode"],
                     fmt(row["period_hours"]), f"{row['valid_starts']}/{row['attempted_starts']}",
                     fmt(a["error_deg"] if a else 90.0 if metric == "axial" else None), str(b["distinct_minima"]),
                     fmt(b["oracle"]["error_deg"] if b["oracle"] else None),
                     fmt(c["error_deg"] if c else None), fmt(c["rms_relative_to_best"] if c else None),
                     fmt(d["oracle_deg"]), fmt(d["best_scored_deg"])]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    write(destination / "scores.json", result)
    with (destination / "RESULTS.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    return {"stage": "score", "input_period_pairs": len(objects), "scope": label}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, handler in (("export", cmd_export), ("run", cmd_run), ("score", cmd_score), ("validate", cmd_validate)):
        p = commands.add_parser(name)
        p.add_argument("--output-dir", type=Path, default=OUTPUT)
        p.add_argument("--spacecraft-dir", type=Path, default=SPACECRAFT, help="read-only spacecraft inputs")
        p.add_argument("--benchmark-dir", type=Path, default=BENCHMARK, help="read-only published own-period benchmark")
        p.set_defaults(handler=handler)
        if name == "run":
            p.add_argument("--workers", type=int, default=6)
            p.add_argument("--runs", nargs="+", choices=RUNS, default=list(RUNS))
            p.add_argument("--periods", nargs="+", choices=MODES, default=list(MODES))
            p.add_argument("--max-starts", type=int, help="SMOKE TEST only: first N of 152 starts per input/period")
        if name == "score":
            p.add_argument("--fit-dir", type=Path, help="read-only published standard-start fits for a network rerun")
            p.add_argument("--allow-partial", action="store_true", help="permit explicitly labelled smoke-test reports")
        if name == "validate":
            p.add_argument("--object-id", default="asteroid_1733")
            p.add_argument("--arm", choices=ARMS, default="standard6")
            p.add_argument("--start-index", type=int, default=0)
    args = parser.parse_args(argv)
    if switched() and (args.command != "score" or args.fit_dir is None):
        parser.error("switched classical runs score --fit-dir only; reuse the published standard-start fits")
    for key in ("output_dir", "spacecraft_dir", "benchmark_dir"):
        setattr(args, key, getattr(args, key).resolve())
    if args.command == "run":
        if args.workers < 1 or (args.max_starts is not None and not 1 <= args.max_starts <= 152):
            parser.error("workers must be positive; max-starts must be in 1..152")
        if len(set(args.runs)) != len(args.runs) or len(set(args.periods)) != len(args.periods):
            parser.error("duplicate run/period selections")
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.perf_counter()
    try:
        result = args.handler(args)
    except (ValueError, OSError, KeyError, IndexError) as error:
        parser.exit(1, f"classical {args.command}: {error}\n")
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    result.update(wall_seconds=time.perf_counter() - started,
                  solver_cpu_seconds=after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
