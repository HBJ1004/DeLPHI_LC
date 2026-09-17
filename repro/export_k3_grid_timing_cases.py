#!/usr/bin/env python3
"""Export de-identified per-case pole-grid timing evidence for publication."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

SCHEMA = "delphi.k3-pole-grid-timing-public-export.v1"
ARMS = ("classical15", "classical20", "guided15", "guided20", "standard6")
MODES = ("cold", "warm")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _write(path: Path, value: object) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _gpu_model(raw: object) -> str:
    if not isinstance(raw, list) or len(raw) != 1 or not isinstance(raw[0], str):
        raise ValueError("benchmark lock has an invalid GPU description")
    return raw[0].split(",", 1)[0].strip()


def export(root: Path, output: Path) -> dict[str, object]:
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    lock_path = root / "lock.json"
    execution_path = root / "full" / "execution.json"
    score_path = root / "full-score.json"
    lock = _read(lock_path)
    execution = _read(execution_path)
    score = _read(score_path)
    if (
        lock.get("schema") != "delphi.k3-pole-grid-benchmark.v3"
        or execution.get("schema") != lock.get("schema")
        or score.get("schema") != lock.get("schema")
        or execution.get("execution_complete") is not True
        or score.get("phase") != "reference_scoring_after_sealed_selection"
    ):
        raise ValueError("pole-grid benchmark documents do not form a completed v3 result")
    if execution.get("lock_sha256") != _sha256(lock_path):
        raise ValueError("execution is not bound to the supplied lock")
    if score.get("lock_sha256") != _sha256(lock_path) or score.get("execution_sha256") != _sha256(
        execution_path
    ):
        raise ValueError("score is not bound to the supplied lock and execution")

    results = score.get("results")
    if not isinstance(results, dict):
        raise ValueError("score has no results")
    scored = results.get("selected_axial_errors_by_object")
    if not isinstance(scored, dict) or set(scored) != set(ARMS):
        raise ValueError("score has an incomplete arm set")
    errors: dict[tuple[str, str, str, int], float] = {}
    for arm in ARMS:
        modes = scored[arm]
        if not isinstance(modes, dict) or set(modes) != set(MODES):
            raise ValueError(f"score has incomplete timing modes for {arm}")
        for mode in MODES:
            rows = modes[mode]
            if not isinstance(rows, list) or len(rows) != 170:
                raise ValueError(f"score has an invalid object count for {arm}/{mode}")
            for row in rows:
                repeat_errors = row.get("repeat_error_degrees")
                if not isinstance(repeat_errors, list) or len(repeat_errors) != 3:
                    raise ValueError("score has an invalid repeat-error vector")
                for repeat, value in enumerate(repeat_errors):
                    errors[(str(row["object_id"]), arm, mode, repeat)] = float(value)

    public_rows = []
    execution_rows = execution.get("rows")
    if not isinstance(execution_rows, list) or len(execution_rows) != 5100:
        raise ValueError("execution must contain the full 5 x 2 x 3 x 170 factorial")
    seen: set[tuple[str, str, str, int]] = set()
    for binding in execution_rows:
        if not isinstance(binding, dict) or not isinstance(binding.get("path"), str):
            raise ValueError("execution row is invalid")
        case_path = root / "full" / str(binding["path"])
        if _sha256(case_path) != binding.get("sha256"):
            raise ValueError(f"timed-result hash mismatch: {case_path.name}")
        case = _read(case_path)
        key = (
            str(case.get("object_id")),
            str(case.get("arm")),
            str(case.get("timing_mode")),
            int(case.get("repeat_index", -1)),
        )
        if key in seen or key not in errors:
            raise ValueError(f"duplicate or unscored benchmark case: {key}")
        seen.add(key)
        selected = case.get("selected_fit")
        if not isinstance(selected, dict):
            raise ValueError(f"case has no selected fit: {key}")
        axis = selected.get("axis")
        if not isinstance(axis, list) or len(axis) != 3:
            raise ValueError(f"case has an invalid selected axis: {key}")
        public_rows.append(
            {
                "object_id": key[0],
                "fold": int(case["fold"]),
                "arm": key[1],
                "timing_mode": key[2],
                "repeat_index": key[3],
                "completed": bool(case["completed"]),
                "starts_requested": int(case["starts_requested"]),
                "valid_starts": int(case["valid_starts"]),
                "iterations_sum": int(case["iterations_sum"]),
                "wall_seconds": float(case["wall_seconds"]),
                "worker_wall_seconds": float(case["worker_wall_seconds"]),
                "solver_cpu_seconds": float(case["solver_cpu_seconds"]),
                "native_solver_wall_seconds_sum": float(case["native_solver_wall_seconds_sum"]),
                "inference_and_optional_load_seconds": float(
                    case["inference_and_optional_load_seconds"]
                ),
                "selected_final_relative_rms": float(selected["final_relative_rms"]),
                "selected_axis_x": float(axis[0]),
                "selected_axis_y": float(axis[1]),
                "selected_axis_z": float(axis[2]),
                "selected_axial_error_deg": errors[key],
                "recovered_within_20_deg": errors[key] <= 20.0,
            }
        )
    expected = {
        (object_id, arm, mode, repeat)
        for object_id in results["cohort"]["object_ids"]
        for arm in ARMS
        for mode in MODES
        for repeat in range(3)
    }
    if seen != expected:
        raise ValueError("execution does not cover the expected factorial")
    public_rows.sort(
        key=lambda row: (
            str(row["object_id"]),
            str(row["timing_mode"]),
            str(row["arm"]),
            int(row["repeat_index"]),
        )
    )

    output.mkdir(parents=True)
    csv_path = output / "timing-cases.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(public_rows[0]))
        writer.writeheader()
        writer.writerows(public_rows)
    settings = lock["settings"]
    environment = lock["environment"]
    primary = results["analyses"]["cold_step20_primary"]
    manifest = {
        "schema": SCHEMA,
        "scope": lock["scope"],
        "primary_comparison": settings["primary_comparison"],
        "case_count": len(public_rows),
        "cold_case_count": sum(row["timing_mode"] == "cold" for row in public_rows),
        "warm_case_count": sum(row["timing_mode"] == "warm" for row in public_rows),
        "warm_timing_status": (
            "reported_for_provenance_but_not_interpreted_because_classical_warm_was_"
            "slower_than_process_cold_on_the_same_host"
        ),
        "timing_cases_csv_sha256": _sha256(csv_path),
        "source_hashes": {
            "lock": _sha256(lock_path),
            "execution": _sha256(execution_path),
            "score": _sha256(score_path),
        },
        "host": {
            "cpu": environment["cpu_models"],
            "logical_cpu_count_available": environment["cpu_count"],
            "cpu_affinity": environment["cpu_affinity"],
            "gpu_model": _gpu_model(environment["gpu"]),
            "platform": environment["platform"],
            "python": environment["python"],
            "packages": environment["packages"],
            "solver_workers": settings["workers"],
            "worker_thread_environment": environment["worker_thread_environment"],
        },
        "settings": {
            key: settings[key]
            for key in (
                "spacing_degrees",
                "radius_deg",
                "radius_provenance",
                "repeats",
                "workers",
                "convergence_tolerance",
                "iteration_cap",
                "timeout_seconds_per_start",
                "selection",
                "period",
                "search_region",
                "cold",
                "warm",
            )
        },
        "primary_result": {
            "baseline_arm": primary["baseline_arm"],
            "candidate_arm": primary["candidate_arm"],
            "timing_mode": primary["timing_mode"],
            "runtime": primary["runtime"],
            "completion": primary["completion"],
            "recovery": primary["recovery"],
            "rms": primary["rms"],
            "decision": primary["decision"],
        },
        "thresholds": results["thresholds"],
        "path_privacy": "absolute paths, GPU UUID, usernames, and executable paths omitted",
    }
    _write(output / "manifest.json", manifest)
    payload = (output / "manifest.json").read_text(encoding="utf-8") + csv_path.read_text(
        encoding="utf-8"
    )
    if re.search(r"/mnt/|/home/|[A-Za-z]:\\\\|GPU-[0-9a-f-]{8,}", payload):
        raise ValueError("public timing export contains a private path or GPU UUID")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest = export(args.benchmark_root, args.output)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "case_count": manifest["case_count"],
                "timing_cases_csv_sha256": manifest["timing_cases_csv_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
