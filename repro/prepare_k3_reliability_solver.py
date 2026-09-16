"""Make a full-precision, output-only convexinv copy for reliability studies.

The solver is copied before it is patched.  The patch changes only ``-o`` and
``-p`` serialization; its optimizer, trajectory logging, and native modelled
lightcurve writer are byte-for-byte untouched.  A small supplied parity input
can verify that claim without opening a cohort, prediction, or score artifact.
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


class ReliabilitySolverPreparationError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def patch_full_precision_export(source: Path) -> str:
    """Patch only convexinv's area and parameter file format strings."""
    path = source / "convexinv.c"
    original = path.read_text(encoding="ascii")
    replacements = {
        'fprintf(f_areas, "%14.12e\\n", Area[i]);':
            'fprintf(f_areas, "%.17g\\n", Area[i]);',
        'fprintf(f_areas, "%14.12e %14.12e %14.12e\\n", Nor[i][1], Nor[i][2], Nor[i][3]);':
            'fprintf(f_areas, "%.17g %.17g %.17g\\n", Nor[i][1], Nor[i][2], Nor[i][3]);',
        'fprintf(f_areas, "%14.12e\\n", dark);':
            'fprintf(f_areas, "%.17g\\n", dark);',
        'fprintf(f_areas, "%14.12e %14.12e %14.12e\\n", -chck[1]/dark,-chck[2]/dark,-chck[3]/dark);':
            'fprintf(f_areas, "%.17g %.17g %.17g\\n", -chck[1]/dark,-chck[2]/dark,-chck[3]/dark);',
        'fprintf(f_par, "%5.1f %+5.1f %12.8f\\n", cg[Ncoef+2] * RAD2DEG , 90 - (cg[Ncoef+1] * RAD2DEG), 24 * prd);':
            'fprintf(f_par, "%.17g %.17g %.17g\\n", cg[Ncoef+2] * RAD2DEG , 90 - (cg[Ncoef+1] * RAD2DEG), 24 * prd);',
        'fprintf(f_par, "%f %g\\n", jd_0, Phi_0 * RAD2DEG);':
            'fprintf(f_par, "%.17g %.17g\\n", jd_0, Phi_0 * RAD2DEG);',
        'fprintf(f_par, "%g ", cg[Ncoef+3+i]);':
            'fprintf(f_par, "%.17g ", cg[Ncoef+3+i]);',
        'fprintf(f_par, "\\n%g\\n", exp(cg[Ncoef+Nphpar+4]));':
            'fprintf(f_par, "\\n%.17g\\n", exp(cg[Ncoef+Nphpar+4]));',
    }
    patched = original
    for old, new in replacements.items():
        if patched.count(old) != 1:
            raise ReliabilitySolverPreparationError(f"expected exactly one export statement: {old}")
        patched = patched.replace(old, new)
    # The normal native-lightcurve line is our primary guard against an
    # accidental broad format replacement.
    native_line = 'fprintf(f_lc_out, "%g\\n", Yout[k] * Sclnw[i] /'
    if native_line not in patched or 'fprintf(f_lc_out, "%.17g' in patched:
        raise ReliabilitySolverPreparationError("patch would alter native modelled-lightcurve output")
    path.write_text(patched, encoding="ascii", newline="")
    return hashlib.sha256(patched.encode("ascii")).hexdigest()


def prepare_solver_copy(*, original_solver: Path, output: Path, compiler: str = "cc") -> dict[str, object]:
    """Copy, patch, and build a standalone convexinv source tree once."""
    original, output = original_solver.resolve(), output.resolve()
    if not (original / "convexinv.c").is_file() or not (original / "Makefile").is_file():
        raise ReliabilitySolverPreparationError("original_solver is not a convexinv source directory")
    if output.exists():
        raise ReliabilitySolverPreparationError(f"refusing to overwrite existing output: {output}")
    if output == original or output.is_relative_to(original):
        raise ReliabilitySolverPreparationError("output must be outside the original source tree")
    before = _tree_sha256(original)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        copied = temporary / "convexinv"
        shutil.copytree(original, copied)
        patch_sha = patch_full_precision_export(copied)
        clean = subprocess.run(["make", "clean"], cwd=copied, text=True, capture_output=True, check=False)
        build = subprocess.run(["make", "convexinv", f"CC={compiler}"], cwd=copied,
                               text=True, capture_output=True, check=False)
        if build.returncode or not (copied / "convexinv").is_file():
            raise ReliabilitySolverPreparationError(f"build failed: {build.stderr or build.stdout}")
        if _tree_sha256(original) != before:
            raise ReliabilitySolverPreparationError("original solver changed during preparation")
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    copied = output / "convexinv"
    return {
        "schema": "k3-reliability-solver-copy-v1",
        "original_source": str(original),
        "original_tree_sha256_before": before,
        "original_tree_sha256_after": _tree_sha256(original),
        "copy_source": str(copied),
        "copy_tree_sha256": _tree_sha256(copied),
        "binary": str(copied / "convexinv"),
        "binary_sha256": sha256(copied / "convexinv"),
        "export_patch_sha256": patch_sha,
        "make_clean_returncode": clean.returncode,
    }


def receipt_for_existing_copy(*, original_solver: Path, output: Path) -> dict[str, object]:
    """Re-open an interrupted preparation without rebuilding or overwriting it."""
    original, output = original_solver.resolve(), output.resolve()
    copied, binary = output / "convexinv", output / "convexinv" / "convexinv"
    if not (original / "convexinv").is_file() or not binary.is_file():
        raise ReliabilitySolverPreparationError("--resume requires an existing prepared convexinv binary")
    text = (copied / "convexinv.c").read_text(encoding="ascii")
    if 'fprintf(f_lc_out, "%.17g' in text or 'fprintf(f_par, "%.17g %.17g %.17g\\n"' not in text:
        raise ReliabilitySolverPreparationError("--resume output is not the expected output-only precision patch")
    original_hash = _tree_sha256(original)
    return {
        "schema": "k3-reliability-solver-copy-v1",
        "original_source": str(original),
        "original_tree_sha256_before": original_hash,
        "original_tree_sha256_after": original_hash,
        "copy_source": str(copied),
        "copy_tree_sha256": _tree_sha256(copied),
        "binary": str(binary),
        "binary_sha256": sha256(binary),
        "export_patch_sha256": hashlib.sha256(text.encode("ascii")).hexdigest(),
        "make_clean_returncode": None,
        "resumed_after_interrupted_preparation": True,
    }


def verify_native_parity(*, original_binary: Path, copied_binary: Path,
                         parameters: Path, lightcurve: Path, workdir: Path) -> dict[str, object]:
    """Verify trajectories and native modelled output on one technical fixture."""
    # Each run deliberately has its own cwd, so fixture paths must not be
    # interpreted relative to either temporary solver-output directory.
    original_binary, copied_binary = original_binary.resolve(), copied_binary.resolve()
    parameters, lightcurve = parameters.resolve(), lightcurve.resolve()
    workdir = workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=False)
    products: dict[str, dict[str, Path]] = {}
    runs: dict[str, subprocess.CompletedProcess[bytes]] = {}
    for name, binary in (("original", original_binary), ("copy", copied_binary)):
        folder = workdir / name
        folder.mkdir()
        product = {"modelled": folder / "modelled.txt", "areas": folder / "areas.txt", "parameters": folder / "parameters.txt"}
        with lightcurve.open("rb") as stream:
            run = subprocess.run([str(binary), "-v", "-o", str(product["areas"]), "-p", str(product["parameters"]),
                                  str(parameters), str(product["modelled"])], stdin=stream, cwd=folder,
                                 capture_output=True, check=False)
        if run.returncode:
            raise ReliabilitySolverPreparationError(f"{name} parity run failed: {run.stderr.decode(errors='replace')}")
        products[name], runs[name] = product, run
    comparisons = {
        "stdout_identical": runs["original"].stdout == runs["copy"].stdout,
        "stderr_identical": runs["original"].stderr == runs["copy"].stderr,
        "native_modelled_identical": products["original"]["modelled"].read_bytes() == products["copy"]["modelled"].read_bytes(),
    }
    if not all(comparisons.values()):
        raise ReliabilitySolverPreparationError(f"output-only patch parity failed: {comparisons}")
    return {**comparisons,
            "original_modelled_sha256": sha256(products["original"]["modelled"]),
            "copy_modelled_sha256": sha256(products["copy"]["modelled"]),
            "copy_parameters_sha256": sha256(products["copy"]["parameters"]),
            "copy_areas_sha256": sha256(products["copy"]["areas"])}


def _defaults() -> tuple[Path, Path]:
    data = Path(__file__).resolve().parents[2] / "data"
    return (data / "solver-internal-capacity-corrected" / "20260911" / "expanded-solver" / "convexinv",
            data / "reliability-20260915" / "solver")


def main(argv: list[str] | None = None) -> int:
    default_source, default_output = _defaults()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-solver", type=Path, default=default_source)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--compiler", default="cc")
    parser.add_argument("--parity-parameters", type=Path)
    parser.add_argument("--parity-lightcurve", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--resume", action="store_true", help="complete parity/receipt for an existing untouched copy")
    args = parser.parse_args(argv)
    if bool(args.parity_parameters) != bool(args.parity_lightcurve):
        parser.error("--parity-parameters and --parity-lightcurve must be supplied together")
    receipt = (receipt_for_existing_copy(original_solver=args.original_solver, output=args.output)
               if args.resume else prepare_solver_copy(original_solver=args.original_solver, output=args.output,
                                                        compiler=args.compiler))
    if args.parity_parameters:
        parity_dir = args.output / "technical-parity"
        suffix = 1
        while parity_dir.exists():
            parity_dir = args.output / f"technical-parity-resume-{suffix}"
            suffix += 1
        receipt["native_parity"] = verify_native_parity(
            original_binary=args.original_solver / "convexinv", copied_binary=args.output / "convexinv" / "convexinv",
            parameters=args.parity_parameters, lightcurve=args.parity_lightcurve,
            workdir=parity_dir)
        receipt["native_parity"]["directory"] = str(parity_dir.resolve())
    destination = args.receipt or args.output / "receipt.json"
    destination.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
