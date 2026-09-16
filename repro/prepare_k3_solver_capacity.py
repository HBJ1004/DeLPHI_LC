"""Preflight frozen blind lightcurve capacity and build a bounded solver copy.

This is intentionally separate from the convergence protocol runner.  It does
not open a reference catalog, K3 axes, predictions, or score artifacts, and it
never executes a locked-cohort solver run.  The original GPL solver tree is
read-only: compilation happens only in a newly-created expanded copy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

from lc_pipeline.k3.solver_capacity import (
    CAPACITY_FIELDS,
    INTERNAL_CAPACITY_RECEIPT_SCHEMA,
    SolverCapacityError,
    capacity_violations,
    feasible_support,
    internal_capacity_padding,
    internal_capacity_requirements,
    load_blind_lightcurve_rows,
    read_static_capacities,
    require_declared_support,
    required_capacities,
    sha256_file,
    solver_directory,
    tree_sha256,
)

SCHEMA = INTERNAL_CAPACITY_RECEIPT_SCHEMA


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _read_manifest(path: Path, expected_role: str) -> tuple[str, ...]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        object_ids = document["object_ids"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise SolverCapacityError(f"cannot read {expected_role} cohort manifest: {exc}") from exc
    if document.get("role") != expected_role or not isinstance(object_ids, list):
        raise SolverCapacityError(f"{expected_role} cohort manifest has the wrong role or object list")
    if not object_ids or any(not isinstance(item, str) or not item for item in object_ids):
        raise SolverCapacityError(f"{expected_role} cohort manifest has invalid object identifiers")
    if len(object_ids) != len(set(object_ids)):
        raise SolverCapacityError(f"{expected_role} cohort manifest has duplicate object identifiers")
    return tuple(object_ids)


def _rows_for_capacity(rows: Sequence[Mapping[str, object]], capacities: Mapping[str, int]) -> list[dict[str, object]]:
    rendered: list[dict[str, object]] = []
    for row in rows:
        structure = row["structure"]
        assert hasattr(structure, "as_dict")
        violations = capacity_violations(structure, capacities)
        rendered.append(
            {
                "object_id": row["object_id"],
                "lightcurve_sha256": row["source_sha256"],
                "structure": structure.as_dict(),
                "internal_required_capacities": internal_capacity_requirements(structure),
                "violations": list(violations),
            }
        )
    return rendered


def _cohort_report(
    name: str, rows: Sequence[Mapping[str, object]], capacities: Mapping[str, int]
) -> dict[str, object]:
    objects = _rows_for_capacity(rows, capacities)
    structures = [row["structure"] for row in rows]
    return {
        "role": name,
        "required_capacities": required_capacities(structures),
        "support": feasible_support(objects),
        "objects": objects,
    }


def _replace_capacity_defines(header: Path, capacities: Mapping[str, int]) -> str:
    original = header.read_text(encoding="ascii")
    lines: list[str] = []
    changed: dict[str, int] = {}
    for line in original.splitlines(keepends=True):
        stripped = line.lstrip()
        replacement = line
        for name in CAPACITY_FIELDS:
            prefix = f"#define {name}"
            if stripped.startswith(prefix):
                newline = "\n" if line.endswith("\n") else ""
                comment = ""
                if "/*" in line:
                    comment = " " + line[line.index("/*") :].rstrip("\n")
                replacement = f"#define {name:<18} {int(capacities[name])}{comment}{newline}"
                changed[name] = int(capacities[name])
                break
        lines.append(replacement)
    if set(changed) != set(CAPACITY_FIELDS):
        raise SolverCapacityError("expanded copy header lacks an expected static capacity define")
    header.write_text("".join(lines), encoding="ascii", newline="")
    return hashlib.sha256(_canonical({"changed_defines": changed})).hexdigest()


def _compiler_identity(compiler: str) -> dict[str, str]:
    resolved = shutil.which(compiler)
    if resolved is None:
        raise SolverCapacityError(f"cannot resolve compiler: {compiler}")
    binary = Path(resolved).resolve()
    completed = subprocess.run([str(binary), "--version"], text=True, capture_output=True, check=False)
    version = (completed.stdout or completed.stderr).splitlines()
    if completed.returncode != 0 or not version:
        raise SolverCapacityError("compiler --version did not succeed")
    return {"path": str(binary), "sha256": sha256_file(binary), "version": version[0]}


def _size(binary: Path) -> dict[str, int] | None:
    executable = shutil.which("size")
    if executable is None:
        return None
    completed = subprocess.run([executable, "-B", str(binary)], text=True, capture_output=True, check=False)
    fields = completed.stdout.splitlines()
    if completed.returncode or len(fields) < 2:
        return None
    try:
        text, data, bss = (int(value) for value in fields[-1].split()[:3])
    except ValueError:
        return None
    return {"text_bytes": text, "data_bytes": data, "bss_bytes": bss}


def _stack_limit_bytes() -> int | None:
    try:
        import resource

        value, _ = resource.getrlimit(resource.RLIMIT_STACK)
        return None if value == resource.RLIM_INFINITY else int(value)
    except (ImportError, OSError):
        return None


def _build_copy(
    *, original_solver: Path, output_root: Path, capacities: Mapping[str, int], compiler: str
) -> dict[str, object]:
    """Copy, patch exactly three bounds, and compile without touching the original."""
    output = output_root.resolve()
    original = original_solver.resolve()
    if output.exists():
        raise SolverCapacityError(f"refusing to overwrite existing capacity output: {output}")
    if output == original or output.is_relative_to(original):
        raise SolverCapacityError("expanded solver output must not be the original source or its child")
    output.parent.mkdir(parents=True, exist_ok=True)
    before = tree_sha256(original)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        copied = temporary / "expanded-solver" / "convexinv"
        # Copy generated files as well: this lets the upstream ``make clean``
        # target run unchanged in the copy (it errors when no object files
        # exist).  Nothing under ``original`` is ever compiled or cleaned.
        shutil.copytree(original, copied)
        patch_hash = _replace_capacity_defines(copied / "constants.h", capacities)
        subprocess.run(["make", "clean"], cwd=copied, text=True, capture_output=True, check=True)
        subprocess.run(
            ["make", "convexinv", "period_scan", f"CC={compiler}"],
            cwd=copied,
            text=True,
            capture_output=True,
            check=True,
        )
        expanded_binary = copied / "convexinv"
        period_scan = copied / "period_scan"
        if not expanded_binary.is_file() or not period_scan.is_file():
            raise SolverCapacityError("expanded solver build did not produce convexinv and period_scan")
        if tree_sha256(original) != before:
            raise SolverCapacityError("original solver source changed while building expanded copy")
        os.replace(temporary, output)
    except subprocess.CalledProcessError as exc:
        shutil.rmtree(temporary, ignore_errors=True)
        raise SolverCapacityError(f"expanded solver compilation failed: {exc.stderr or exc.stdout}") from exc
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    expanded = output / "expanded-solver" / "convexinv"
    return {
        "source_root": str(expanded),
        "source_tree_sha256": tree_sha256(expanded),
        "binary": str(expanded / "convexinv"),
        "binary_sha256": sha256_file(expanded / "convexinv"),
        "period_scan_binary": str(expanded / "period_scan"),
        "period_scan_binary_sha256": sha256_file(expanded / "period_scan"),
        "patch_sha256": patch_hash,
        "static_size": _size(expanded / "convexinv"),
        "original_source_tree_sha256": {"before": before, "after": tree_sha256(original)},
    }


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _run_dev_parity_smoke(
    *, original_solver: Path, expanded_binary: Path, development_rows: Sequence[Mapping[str, object]],
    original_capacities: Mapping[str, int], output_root: Path
) -> dict[str, object]:
    """Compare one original-supported development input without retaining outputs.

    The parameter file is a fixed, two-iteration official convexinv input.  It
    does not use a K3 start, an input period, labels, or fit selection.
    """
    candidate = next(
        (row for row in development_rows if not capacity_violations(row["structure"], original_capacities)),
        None,
    )
    if candidate is None:
        raise SolverCapacityError("no original-supported development input is available for parity smoke")
    # ``source_path`` was validated under dump_root by load_blind_lightcurve_rows;
    # the absolute input is reconstructed from its resolved parent held below.
    input_path = Path(str(candidate["resolved_path"]))
    parameters = (
        "0 1\n0 1\n5.76198 0\n0\n0\n0.1\n6 6\n8\n"
        "0.5 0\n0.1 0\n-0.5 0\n0.1 0\n2\n"
    ).encode("ascii")
    original_binary = original_solver / "convexinv"
    if not original_binary.is_file():
        raise SolverCapacityError("official convexinv binary is required for development parity smoke")
    before = tree_sha256(original_solver)
    with tempfile.TemporaryDirectory(prefix=".dev-parity.", dir=output_root) as directory_name:
        directory = Path(directory_name)

        def run(binary: Path, name: str) -> dict[str, object]:
            parameter_path = directory / f"{name}.parameters"
            lightcurve_output = directory / f"{name}.lightcurves"
            parameter_output = directory / f"{name}.final-parameters"
            area_output = directory / f"{name}.areas"
            parameter_path.write_bytes(parameters)
            completed = subprocess.run(
                [
                    str(binary), "-v", "-o", str(area_output), "-p", str(parameter_output),
                    str(parameter_path), str(lightcurve_output),
                ],
                input=input_path.read_bytes(),
                capture_output=True,
                timeout=120,
                check=False,
            )
            payload = {
                "return_code": completed.returncode,
                "stdout_sha256": _sha256_bytes(completed.stdout),
                "stderr_sha256": _sha256_bytes(completed.stderr),
                "lightcurve_output_sha256": sha256_file(lightcurve_output) if lightcurve_output.is_file() else None,
                "parameter_output_sha256": sha256_file(parameter_output) if parameter_output.is_file() else None,
                "area_output_sha256": sha256_file(area_output) if area_output.is_file() else None,
            }
            if completed.returncode != 0 or None in payload.values():
                raise SolverCapacityError("development parity smoke did not produce a complete successful solver output")
            return payload

        official = run(original_binary, "official")
        expanded = run(expanded_binary, "expanded")
    if tree_sha256(original_solver) != before:
        raise SolverCapacityError("original solver source changed during development parity smoke")
    parity = official == expanded
    if not parity:
        raise SolverCapacityError("expanded solver output differs from official solver on supported development smoke")
    return {
        "status": "passed",
        "cohort_role": "development",
        "object_id": candidate["object_id"],
        "input_lightcurve_sha256": candidate["source_sha256"],
        "fixed_parameter_sha256": _sha256_bytes(parameters),
        "output_hashes_match": True,
        "official": official,
        "expanded": expanded,
    }


def _build_sanitizer_copy(
    *, expanded_solver: Path, compiler: str, output_root: Path
) -> dict[str, object]:
    """Build disposable ASAN/UBSAN binaries from the corrected source copy."""
    sanitizer_root = output_root / "sanitizer-build" / "convexinv"
    if sanitizer_root.exists():
        raise SolverCapacityError(f"refusing to overwrite sanitizer build: {sanitizer_root}")
    sanitizer_root.parent.mkdir(parents=True, exist_ok=False)
    shutil.copytree(expanded_solver, sanitizer_root)
    flags = "-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer"
    try:
        subprocess.run(["make", "clean"], cwd=sanitizer_root, text=True, capture_output=True, check=True)
        subprocess.run(
            ["make", "convexinv", "period_scan", f"CC={compiler}", f"OPTFLAGS={flags}"],
            cwd=sanitizer_root,
            text=True,
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise SolverCapacityError(f"sanitizer solver compilation failed: {exc.stderr or exc.stdout}") from exc
    binaries = {name: sanitizer_root / name for name in ("convexinv", "period_scan")}
    if not all(path.is_file() for path in binaries.values()):
        raise SolverCapacityError("sanitizer build did not produce both solver binaries")
    return {
        "source_root": str(sanitizer_root),
        "compiler_flags": flags,
        "convexinv_sha256": sha256_file(binaries["convexinv"]),
        "period_scan_sha256": sha256_file(binaries["period_scan"]),
    }


def _run_sanitizer_boundary_checks(
    *, sanitizer_root: Path, rows_by_id: Mapping[str, Mapping[str, object]], output_root: Path
) -> dict[str, object]:
    """Exercise both known boundary inputs without retaining solver outputs."""
    expected = {"asteroid_2512", "asteroid_109"}
    if not expected.issubset(rows_by_id):
        raise SolverCapacityError("boundary objects are absent from blind lightcurve inputs")
    environment = {
        **os.environ,
        "ASAN_OPTIONS": "detect_leaks=0:halt_on_error=1",
        "UBSAN_OPTIONS": "halt_on_error=1:print_stacktrace=1",
    }
    convex_parameters = (
        "0 1\n0 1\n5.76198 0\n0\n0\n0.1\n6 6\n8\n"
        "0.5 0\n0.1 0\n-0.5 0\n0.1 0\n2\n"
    ).encode("ascii")
    scan_parameters = (
        "5.76198 5.76198 1\n0.1\n6 6\n8\n"
        "0.5 0\n0.1 0\n-0.5 0\n0.1 0\n1\n1\n"
    ).encode("ascii")
    results: dict[str, object] = {}
    with tempfile.TemporaryDirectory(prefix=".sanitizer-boundary.", dir=output_root) as directory_name:
        directory = Path(directory_name)
        for object_id in sorted(expected):
            input_path = Path(str(rows_by_id[object_id]["resolved_path"]))
            lightcurve = input_path.read_bytes()
            object_result: dict[str, object] = {
                "input_lightcurve_sha256": str(rows_by_id[object_id]["source_sha256"]),
                "native_structure": rows_by_id[object_id]["structure"].as_dict(),
            }
            for executable, parameters in (("convexinv", convex_parameters), ("period_scan", scan_parameters)):
                parameter_path = directory / f"{object_id}-{executable}.parameters"
                parameter_path.write_bytes(parameters)
                if executable == "convexinv":
                    command = [
                        str(sanitizer_root / executable), "-v", "-o",
                        str(directory / f"{object_id}.areas"), "-p",
                        str(directory / f"{object_id}.parameters-out"), str(parameter_path),
                        str(directory / f"{object_id}.lightcurves"),
                    ]
                else:
                    command = [
                        str(sanitizer_root / executable), "-v", str(parameter_path),
                        str(directory / f"{object_id}.periods"),
                    ]
                completed = subprocess.run(
                    command,
                    input=lightcurve,
                    capture_output=True,
                    timeout=180,
                    check=False,
                    env=environment,
                )
                result = {
                    "return_code": completed.returncode,
                    "stdout_sha256": _sha256_bytes(completed.stdout),
                    "stderr_sha256": _sha256_bytes(completed.stderr),
                }
                if completed.returncode != 0:
                    message = completed.stderr.decode("utf-8", errors="replace")[-1000:]
                    raise SolverCapacityError(
                        f"sanitizer {executable} check failed for {object_id}: {message}"
                    )
                object_result[executable] = result
            results[object_id] = object_result
    return {
        "status": "passed",
        "boundary_object_ids": sorted(expected),
        "asan_options": environment["ASAN_OPTIONS"],
        "ubsan_options": environment["UBSAN_OPTIONS"],
        "runs": results,
    }


def prepare(
    *,
    blind_inputs: Path,
    development_manifest: Path,
    locked_manifest: Path,
    dump_root: Path,
    solver_root: Path,
    output_root: Path,
    compiler: str = "cc",
    run_dev_parity_smoke: bool = True,
    run_sanitizer_checks: bool = True,
) -> dict[str, object]:
    """Create a fail-closed capacity report and an exact-bounds convexinv copy."""
    if output_root.exists():
        raise SolverCapacityError(f"refusing to overwrite existing capacity output: {output_root}")
    development_ids = _read_manifest(development_manifest, "development")
    locked_ids = _read_manifest(locked_manifest, "locked_evaluation")
    if set(development_ids) & set(locked_ids):
        raise SolverCapacityError("development and locked cohorts overlap")
    original_solver = solver_directory(solver_root, "convexinv")
    resolved_output = output_root.resolve()
    if resolved_output == original_solver or resolved_output.is_relative_to(original_solver):
        raise SolverCapacityError("expanded solver output must not be the original source or its child")
    original_binary = original_solver / "convexinv"
    if not original_binary.is_file():
        raise SolverCapacityError("official convexinv binary is required for capacity provenance")
    original_capacities = read_static_capacities(original_solver)
    all_rows = load_blind_lightcurve_rows(blind_inputs, dump_root)
    for row in all_rows:
        row["resolved_path"] = str((dump_root.resolve() / Path(str(row["source_path"]))).resolve())
    internal_id_path_consistent = all(
        Path(str(row["source_path"])).parts == ("files", str(row["object_id"]), "lc.txt")
        for row in all_rows
    )
    if not internal_id_path_consistent:
        raise SolverCapacityError(
            "blind lightcurve paths do not follow the frozen internal object-id mapping"
        )
    by_id = {str(row["object_id"]): row for row in all_rows}
    if len(by_id) != len(all_rows):
        raise SolverCapacityError("blind lightcurve metadata has duplicate object identifiers")
    try:
        development_rows = [by_id[object_id] for object_id in development_ids]
        locked_rows = [by_id[object_id] for object_id in locked_ids]
    except KeyError as exc:
        raise SolverCapacityError("cohort object is absent from blind execution inputs") from exc
    development_report = _cohort_report("development", development_rows, original_capacities)
    locked_report = _cohort_report("locked_evaluation", locked_rows, original_capacities)
    original_support_met = bool(
        development_report["support"]["required_support_met"]
        and locked_report["support"]["required_support_met"]
    )
    required = required_capacities([row["structure"] for row in all_rows])
    # A capacity rebuild is additive.  Reducing a supported input dimension
    # merely because this cohort happens not to use it would create a needless
    # compatibility regression and weaken the parity smoke's meaning.
    expanded_target = {
        name: max(int(original_capacities[name]), int(required[name])) for name in CAPACITY_FIELDS
    }
    compiler_info = _compiler_identity(compiler)
    expanded = _build_copy(
        original_solver=original_solver,
        output_root=output_root,
        capacities=expanded_target,
        compiler=str(compiler_info["path"]),
    )
    expanded_capacities = read_static_capacities(Path(str(expanded["source_root"])))
    expanded_development = _cohort_report("development", development_rows, expanded_capacities)
    expanded_locked = _cohort_report("locked_evaluation", locked_rows, expanded_capacities)
    require_declared_support(expanded_development["objects"], role="development")
    require_declared_support(expanded_locked["objects"], role="locked_evaluation")
    parity_smoke = (
        _run_dev_parity_smoke(
            original_solver=original_solver,
            expanded_binary=Path(str(expanded["binary"])),
            development_rows=development_rows,
            original_capacities=original_capacities,
            output_root=output_root,
        )
        if run_dev_parity_smoke
        else {"status": "not_run", "reason": "explicitly disabled by caller"}
    )
    if run_sanitizer_checks:
        sanitizer_build = _build_sanitizer_copy(
            expanded_solver=Path(str(expanded["source_root"])),
            compiler=str(compiler_info["path"]),
            output_root=output_root,
        )
        sanitizer = _run_sanitizer_boundary_checks(
            sanitizer_root=Path(str(sanitizer_build["source_root"])),
            rows_by_id=by_id,
            output_root=output_root,
        )
        sanitizer["build"] = sanitizer_build
    else:
        sanitizer = {"status": "not_run", "reason": "explicitly disabled by caller"}
    report: dict[str, object] = {
        "schema": SCHEMA,
        "scope": {
            "capacity_measurement_metadata_only": True,
            "reference_catalog_opened": False,
            "prediction_outcomes_opened": False,
            "locked_solver_execution": False,
            "original_solver_execution": bool(run_dev_parity_smoke),
            "parity_smoke_development_only": bool(run_dev_parity_smoke),
            "sanitizer_boundary_checks": bool(run_sanitizer_checks),
        },
        "inputs": {
            "blind_inputs_sha256": sha256_file(blind_inputs),
            "blind_internal_object_id_to_lightcurve_path_verified": internal_id_path_consistent,
            "development_manifest_sha256": sha256_file(development_manifest),
            "locked_manifest_sha256": sha256_file(locked_manifest),
            "original_solver_root": str(original_solver),
            "original_solver_source_tree_sha256": tree_sha256(original_solver),
            "original_solver_binary": str(original_binary),
            "original_solver_binary_sha256": sha256_file(original_binary),
            "original_static_capacities": original_capacities,
            "conjgradinv_static_capacities_inspected": read_static_capacities(solver_root, "conjgradinv"),
        },
        "original_solver_preflight": {
            "development": development_report,
            "locked_evaluation": locked_report,
            "declared_cohort_support_met": original_support_met,
            "solver_execution_authorized": original_support_met,
            "reason": (
                "all declared objects fit the original static capacities"
                if original_support_met
                else "original static capacity excludes declared cohort members; do not silently drop them"
            ),
        },
        "expanded_solver_preflight": {
            "capacities": expanded_capacities,
            "development": expanded_development,
            "locked_evaluation": expanded_locked,
            "declared_cohort_support_met": True,
            "same_binary_required_for_baseline_and_guided_arms": str(expanded["binary"]),
        },
        "expanded_solver": expanded,
        "internal_capacity_padding": internal_capacity_padding(),
        "development_supported_input_parity_smoke": parity_smoke,
        "sanitizer_boundary_checks": sanitizer,
        "compiler": compiler_info,
        "memory_and_stack_risk": {
            "expanded_static_size": expanded["static_size"],
            "process_stack_soft_limit_bytes": _stack_limit_bytes(),
            "assessment": "Static BSS is measured from the compiled copy; dynamic observation arrays scale with MAX_N_OBS. Run solver processes serially and retain the recorded size/stack values in any execution lock.",
        },
        "structural_worst_case": {
            "native_structure": {
                "lightcurve_count": expanded_target["MAX_LC"] - 1,
                "total_observations": expanded_target["MAX_N_OBS"] - 3,
                "max_points_per_lightcurve": expanded_target["POINTS_MAX"],
            },
            "internal_required_capacities": expanded_target,
            "capacity_violations": [],
            "solver_executed": False,
            "reason": "structural-only check; no synthetic photometry or locked solver execution was run",
        },
        "conclusion": {
            "original_design_infeasible": not original_support_met,
            "interpretation": (
                "This is a static-input-capacity failure, not evidence that no convergence tolerance works."
                if not original_support_met
                else "The original static capacities cover every declared cohort input."
            ),
            "required_support_policy": "fail closed: all 30 development and all 140 locked objects must fit before either cohort is executed",
        },
    }
    report_path = output_root / "solver-capacity-preflight.json"
    report_path.write_bytes(_canonical(report) + b"\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blind-inputs", type=Path, required=True)
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--locked-manifest", type=Path, required=True)
    parser.add_argument("--dump-root", type=Path, required=True)
    parser.add_argument("--solver-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--compiler", default="cc")
    parser.add_argument("--skip-dev-parity-smoke", action="store_true")
    parser.add_argument("--skip-sanitizer-checks", action="store_true")
    arguments = parser.parse_args()
    report = prepare(
        blind_inputs=arguments.blind_inputs,
        development_manifest=arguments.development_manifest,
        locked_manifest=arguments.locked_manifest,
        dump_root=arguments.dump_root,
        solver_root=arguments.solver_root,
        output_root=arguments.output_root,
        compiler=arguments.compiler,
        run_dev_parity_smoke=not arguments.skip_dev_parity_smoke,
        run_sanitizer_checks=not arguments.skip_sanitizer_checks,
    )
    print(json.dumps({"output_root": str(arguments.output_root), "schema": report["schema"]}))


if __name__ == "__main__":
    main()
