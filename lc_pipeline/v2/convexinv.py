"""Auditable adapters for external GPL DAMIT inversion programs.

The DAMIT source and binaries stay outside this MIT-licensed repository.  This
module creates documented input files, invokes supplied executables, and records
the input, output, binary, compiler, and timing provenance of each run.
"""

from __future__ import annotations

import hashlib
import math
import re
import resource
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from ..physics.axial import axial_angular_error_deg
from .data import canonical_json, sha256_file


class ConvexinvError(ValueError):
    """Raised when an external inversion cannot be executed or audited."""


@dataclass(frozen=True)
class ConvexinvParameters:
    """The documented 13-line ``convexinv`` parameter file."""

    lambda_deg: float
    beta_deg: float
    period_hours: float
    free_lambda: bool = True
    free_beta: bool = True
    free_period: bool = True
    zero_time_jd: float = 0.0
    rotation_phase_deg: float = 0.0
    convexity_weight: float = 0.1
    harmonic_degree: int = 6
    harmonic_order: int = 6
    triangulation_rows: int = 8
    phase_a: float = 0.5
    phase_d: float = 0.1
    phase_k: float = -0.5
    lambert_coefficient: float = 0.1
    iteration_stop_condition: float = 50.0

    def __post_init__(self) -> None:
        values = (
            self.lambda_deg,
            self.beta_deg,
            self.period_hours,
            self.zero_time_jd,
            self.rotation_phase_deg,
            self.convexity_weight,
            self.phase_a,
            self.phase_d,
            self.phase_k,
            self.lambert_coefficient,
            self.iteration_stop_condition,
        )
        if not all(math.isfinite(value) for value in values) or self.period_hours <= 0:
            raise ConvexinvError("convexinv parameters must be finite and period positive")
        if self.convexity_weight < 0 or self.iteration_stop_condition <= 0:
            raise ConvexinvError("convexity weight and stopping condition must be positive")
        if min(self.harmonic_degree, self.harmonic_order, self.triangulation_rows) <= 0:
            raise ConvexinvError("harmonic degree/order and triangulation rows must be positive")

    def render(self) -> str:
        def flag(value: bool) -> int:
            return 1 if value else 0

        return "\n".join(
            (
                f"{self.lambda_deg:.12g} {flag(self.free_lambda)}",
                f"{self.beta_deg:.12g} {flag(self.free_beta)}",
                f"{self.period_hours:.12g} {flag(self.free_period)}",
                f"{self.zero_time_jd:.12g}",
                f"{self.rotation_phase_deg:.12g}",
                f"{self.convexity_weight:.12g}",
                f"{self.harmonic_degree} {self.harmonic_order}",
                str(self.triangulation_rows),
                f"{self.phase_a:.12g} 0",
                f"{self.phase_d:.12g} 0",
                f"{self.phase_k:.12g} 0",
                f"{self.lambert_coefficient:.12g} 0",
                f"{self.iteration_stop_condition:.12g}",
            )
        ) + "\n"


@dataclass(frozen=True)
class PeriodScanParameters:
    """The documented 10-line ``period_scan`` parameter file."""

    period_start_hours: float
    period_end_hours: float
    step_coefficient: float = 0.8
    convexity_weight: float = 0.1
    harmonic_degree: int = 6
    harmonic_order: int = 6
    triangulation_rows: int = 8
    phase_a: float = 0.5
    phase_d: float = 0.1
    phase_k: float = -0.5
    lambert_coefficient: float = 0.1
    iteration_stop_condition: float = 50.0
    minimum_iterations: int = 10

    def __post_init__(self) -> None:
        values = (
            self.period_start_hours,
            self.period_end_hours,
            self.step_coefficient,
            self.convexity_weight,
            self.phase_a,
            self.phase_d,
            self.phase_k,
            self.lambert_coefficient,
            self.iteration_stop_condition,
        )
        if not all(math.isfinite(value) for value in values):
            raise ConvexinvError("period-scan parameters must be finite")
        if not 0 < self.period_start_hours < self.period_end_hours:
            raise ConvexinvError("period scan must have positive increasing bounds")
        if not 0 < self.step_coefficient < 1:
            raise ConvexinvError("period scan step coefficient must lie in (0, 1)")
        if min(self.harmonic_degree, self.harmonic_order, self.triangulation_rows) <= 0:
            raise ConvexinvError("harmonic degree/order and triangulation rows must be positive")
        if self.iteration_stop_condition <= 0 or self.minimum_iterations <= 0:
            raise ConvexinvError("period scan iteration settings must be positive")

    def render(self) -> str:
        return "\n".join(
            (
                f"{self.period_start_hours:.12g} {self.period_end_hours:.12g} {self.step_coefficient:.12g}",
                f"{self.convexity_weight:.12g}",
                f"{self.harmonic_degree} {self.harmonic_order}",
                str(self.triangulation_rows),
                f"{self.phase_a:.12g} 0",
                f"{self.phase_d:.12g} 0",
                f"{self.phase_k:.12g} 0",
                f"{self.lambert_coefficient:.12g} 0",
                f"{self.iteration_stop_condition:.12g}",
                str(self.minimum_iterations),
            )
        ) + "\n"


def write_convexinv_parameters(path: str | Path, parameters: ConvexinvParameters) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(parameters.render(), encoding="ascii")
    return destination


def write_period_scan_parameters(path: str | Path, parameters: PeriodScanParameters) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(parameters.render(), encoding="ascii")
    return destination


@dataclass(frozen=True)
class ConvexinvProvenance:
    executable_sha256: str
    source_tree_sha256: str
    input_sha256: str
    compiler_version: str
    command: tuple[str, ...]
    parameter_sha256: str | None = None


@dataclass(frozen=True)
class ConvexinvResult:
    return_code: int | None
    timed_out: bool
    wall_time_seconds: float
    iterations: int | None
    chi2: float | None
    deviation: float | None
    final_lambda_deg: float | None
    final_beta_deg: float | None
    stdout_sha256: str
    stderr_sha256: str
    provenance: ConvexinvProvenance
    process_cpu_time_seconds: float | None = None
    final_period_hours: float | None = None
    relative_rms_from_output: float | None = None
    output_lightcurve_sha256: str | None = None
    output_parameter_sha256: str | None = None
    output_area_sha256: str | None = None


@dataclass(frozen=True)
class PeriodScanRow:
    period_hours: float
    rms: float
    chi2: float
    iterations: int
    dark_area_percent: float


@dataclass(frozen=True)
class PeriodScanResult:
    return_code: int | None
    timed_out: bool
    wall_time_seconds: float
    process_cpu_time_seconds: float | None
    rows: tuple[PeriodScanRow, ...]
    stdout_sha256: str
    stderr_sha256: str
    output_periods_sha256: str | None
    provenance: ConvexinvProvenance


_ITERATION = re.compile(
    r"^\s*(\d+)\s+chi2\s+([-+0-9.eE]+)\s+dev\s+([-+0-9.eE]+)", re.M
)
_FINAL = re.compile(
    r"lambda, beta and period \(hrs\):\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)",
    re.I,
)


def _tree_sha256(root: str | Path) -> str:
    source = Path(root)
    if not source.is_dir():
        raise ConvexinvError("convexinv source root is not a directory")
    entries = [
        {"path": path.relative_to(source).as_posix(), "sha256": sha256_file(path)}
        for path in sorted(item for item in source.rglob("*") if item.is_file())
    ]
    if not entries:
        raise ConvexinvError("convexinv source root is empty")
    return hashlib.sha256(canonical_json(entries).encode("utf-8")).hexdigest()


def _compiler_version(command: Sequence[str]) -> str:
    try:
        completed = subprocess.run(
            list(command), capture_output=True, text=True, timeout=30, check=False
        )
        return (completed.stdout or completed.stderr).splitlines()[0]
    except (IndexError, OSError, subprocess.TimeoutExpired):
        return "unavailable"


def _file_hash_or_none(path: Path) -> str | None:
    return sha256_file(path) if path.is_file() else None


def _require_file(path: str | Path, description: str) -> Path:
    value = Path(path)
    if not value.is_file():
        raise ConvexinvError(f"{description} must be an existing file: {value}")
    return value.resolve()
def _run_external(
    command: Sequence[str], *, stdin_path: Path, cwd: Path, timeout_seconds: float
) -> tuple[int | None, bool, float, float, bytes, bytes]:
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ConvexinvError("timeout_seconds must be finite and positive")
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.monotonic()
    try:
        with stdin_path.open("rb") as handle:
            completed = subprocess.run(
                list(command), cwd=cwd, stdin=handle, capture_output=True,
                timeout=timeout_seconds, check=False,
            )
        return_code, timed_out, stdout, stderr = (
            completed.returncode, False, completed.stdout, completed.stderr
        )
    except subprocess.TimeoutExpired as exc:
        return_code, timed_out = None, True
        stdout = exc.stdout if isinstance(exc.stdout, bytes) else (exc.stdout or "").encode()
        stderr = exc.stderr if isinstance(exc.stderr, bytes) else (exc.stderr or "").encode()
    elapsed = time.monotonic() - started
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = (after.ru_utime + after.ru_stime) - (before.ru_utime + before.ru_stime)
    return return_code, timed_out, elapsed, cpu, stdout, stderr


def _parse_lightcurve_brightness(path: Path) -> tuple[np.ndarray, ...]:
    try:
        tokens = path.read_text(encoding="ascii").split()
    except (OSError, UnicodeDecodeError) as exc:
        raise ConvexinvError(f"cannot parse DAMIT lightcurve input: {exc}") from exc
    try:
        cursor, curve_count = 0, int(tokens[0])
        cursor += 1
        curves: list[np.ndarray] = []
        for _ in range(curve_count):
            count = int(tokens[cursor])
            cursor += 2  # count and relative/calibrated flag
            brightness = []
            for _ in range(count):
                cursor += 1  # JD
                brightness.append(float(tokens[cursor]))
                cursor += 7  # brightness plus six geometry values
            curves.append(np.asarray(brightness, dtype=np.float64))
    except (IndexError, ValueError) as exc:
        raise ConvexinvError(f"malformed DAMIT lightcurve input: {exc}") from exc
    if cursor != len(tokens) or not curves or any(not curve.size for curve in curves):
        raise ConvexinvError("malformed DAMIT lightcurve input")
    return tuple(curves)


def relative_rms_from_modelled_lightcurve(
    lightcurve_path: str | Path, modelled_path: str | Path
) -> float:
    """Compute DAMIT's per-lightcurve relative residual RMS from model output."""
    observed = _parse_lightcurve_brightness(Path(lightcurve_path))
    try:
        modelled = np.asarray(Path(modelled_path).read_text(encoding="ascii").split(), dtype=np.float64)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ConvexinvError(f"cannot parse modeled lightcurve output: {exc}") from exc
    expected = sum(len(curve) for curve in observed)
    if modelled.shape != (expected,) or not np.all(np.isfinite(modelled)):
        raise ConvexinvError("modeled lightcurve output does not align with the input")
    squared_sum, offset = 0.0, 0
    for curve in observed:
        predicted = modelled[offset : offset + len(curve)]
        offset += len(curve)
        observed_mean, predicted_mean = float(np.mean(curve)), float(np.mean(predicted))
        if observed_mean == 0 or predicted_mean == 0:
            raise ConvexinvError("cannot normalize zero-mean lightcurve")
        squared_sum += float(np.square(curve / observed_mean - predicted / predicted_mean).sum())
    return math.sqrt(squared_sum / expected)


def run_convexinv(
    *, executable: str | Path, source_root: str | Path, lightcurve_file: str | Path,
    parameter_file: str | Path, output_directory: str | Path, timeout_seconds: float = 3600.0,
    compiler_command: Sequence[str] = ("cc", "--version"),
) -> ConvexinvResult:
    """Run documented ``convexinv`` once with auditable inputs and outputs."""
    binary = _require_file(executable, "convexinv executable")
    lightcurve = _require_file(lightcurve_file, "DAMIT lightcurve")
    parameters = _require_file(parameter_file, "convexinv parameter file")
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    out_lcs, out_par, out_areas = (
        output / "modelled-lightcurve.txt", output / "solution-parameters.txt", output / "solution-areas.txt"
    )
    if any(path.exists() for path in (out_lcs, out_par, out_areas)):
        raise ConvexinvError("convexinv output directory already contains a run result")
    command = (
        str(binary), "-v", "-o", str(out_areas), "-p", str(out_par), str(parameters), str(out_lcs)
    )
    provenance = ConvexinvProvenance(
        executable_sha256=sha256_file(binary), source_tree_sha256=_tree_sha256(source_root),
        input_sha256=sha256_file(lightcurve), parameter_sha256=sha256_file(parameters),
        compiler_version=_compiler_version(compiler_command), command=command,
    )
    return_code, timed_out, elapsed, cpu, stdout, stderr = _run_external(
        command, stdin_path=lightcurve, cwd=output, timeout_seconds=timeout_seconds
    )
    decoded_stdout = stdout.decode("utf-8", errors="replace")
    iterations, final = list(_ITERATION.finditer(decoded_stdout)), _FINAL.search(decoded_stdout)
    modelled_hash = _file_hash_or_none(out_lcs)
    return ConvexinvResult(
        return_code=return_code, timed_out=timed_out, wall_time_seconds=elapsed,
        process_cpu_time_seconds=cpu,
        iterations=int(iterations[-1].group(1)) if iterations else None,
        chi2=float(iterations[-1].group(2)) if iterations else None,
        deviation=float(iterations[-1].group(3)) if iterations else None,
        final_lambda_deg=float(final.group(1)) if final else None,
        final_beta_deg=float(final.group(2)) if final else None,
        final_period_hours=float(final.group(3)) if final else None,
        relative_rms_from_output=(
            relative_rms_from_modelled_lightcurve(lightcurve, out_lcs)
            if return_code == 0 and not timed_out and modelled_hash else None
        ),
        stdout_sha256=hashlib.sha256(stdout).hexdigest(), stderr_sha256=hashlib.sha256(stderr).hexdigest(),
        output_lightcurve_sha256=modelled_hash, output_parameter_sha256=_file_hash_or_none(out_par),
        output_area_sha256=_file_hash_or_none(out_areas), provenance=provenance,
    )


def parse_period_scan_rows(path: str | Path) -> tuple[PeriodScanRow, ...]:
    """Parse the documented five-column ``period_scan`` output file."""
    try:
        lines = Path(path).read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ConvexinvError(f"cannot read period scan output: {exc}") from exc
    rows: list[PeriodScanRow] = []
    for line in lines:
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 5:
            raise ConvexinvError("period scan output must have five columns")
        try:
            row = PeriodScanRow(float(fields[0]), float(fields[1]), float(fields[2]), int(fields[3]), float(fields[4]))
        except ValueError as exc:
            raise ConvexinvError(f"invalid period scan output row: {line}") from exc
        if not all(math.isfinite(value) for value in (row.period_hours, row.rms, row.chi2, row.dark_area_percent)):
            raise ConvexinvError("period scan output contains a non-finite value")
        rows.append(row)
    if not rows:
        raise ConvexinvError("period scan output is empty")
    return tuple(rows)


def run_period_scan(
    *, executable: str | Path, source_root: str | Path, lightcurve_file: str | Path,
    parameter_file: str | Path, output_directory: str | Path, timeout_seconds: float = 3600.0,
    compiler_command: Sequence[str] = ("cc", "--version"),
) -> PeriodScanResult:
    """Run documented ``period_scan`` once and retain all trial-period rows."""
    binary = _require_file(executable, "period_scan executable")
    lightcurve = _require_file(lightcurve_file, "DAMIT lightcurve")
    parameters = _require_file(parameter_file, "period_scan parameter file")
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    out_periods = output / "period-scan.txt"
    if out_periods.exists():
        raise ConvexinvError("period-scan output directory already contains a run result")
    command = (str(binary), "-v", str(parameters), str(out_periods))
    provenance = ConvexinvProvenance(
        executable_sha256=sha256_file(binary), source_tree_sha256=_tree_sha256(source_root),
        input_sha256=sha256_file(lightcurve), parameter_sha256=sha256_file(parameters),
        compiler_version=_compiler_version(compiler_command), command=command,
    )
    return_code, timed_out, elapsed, cpu, stdout, stderr = _run_external(
        command, stdin_path=lightcurve, cwd=output, timeout_seconds=timeout_seconds
    )
    output_hash = _file_hash_or_none(out_periods)
    return PeriodScanResult(
        return_code=return_code, timed_out=timed_out, wall_time_seconds=elapsed,
        process_cpu_time_seconds=cpu,
        rows=parse_period_scan_rows(out_periods) if return_code == 0 and output_hash else (),
        stdout_sha256=hashlib.sha256(stdout).hexdigest(), stderr_sha256=hashlib.sha256(stderr).hexdigest(),
        output_periods_sha256=output_hash, provenance=provenance,
    )


def axis_to_six_pole_starts(axes: Sequence[Sequence[float]]) -> tuple[tuple[float, float], ...]:
    """Convert three axes to six signed ecliptic pole initializations."""
    array = np.asarray(axes, dtype=np.float64)
    if array.shape != (3, 3) or not np.all(np.isfinite(array)):
        raise ConvexinvError("axis starts require exactly three finite vectors")
    starts: list[tuple[float, float]] = []
    for axis in array:
        norm = np.linalg.norm(axis)
        if norm == 0:
            raise ConvexinvError("axis starts cannot contain zero vectors")
        for signed in (axis / norm, -axis / norm):
            longitude = math.degrees(math.atan2(signed[1], signed[0])) % 360.0
            latitude = math.degrees(math.asin(float(np.clip(signed[2], -1.0, 1.0))))
            starts.append((longitude, latitude))
    return tuple(starts)


def inversion_success(
    result: ConvexinvResult, *, target_axis: Sequence[float], best_rms: float, final_rms: float
) -> bool:
    """Apply the locked finite/pole/RMS success definition when RMS is supplied."""
    if (
        result.timed_out or result.return_code != 0 or result.final_lambda_deg is None
        or result.final_beta_deg is None or not all(math.isfinite(value) for value in (best_rms, final_rms))
        or best_rms < 0 or final_rms < 0
    ):
        return False
    longitude, latitude = math.radians(result.final_lambda_deg), math.radians(result.final_beta_deg)
    pole = (math.cos(latitude) * math.cos(longitude), math.cos(latitude) * math.sin(longitude), math.sin(latitude))
    return axial_angular_error_deg(pole, target_axis) <= 20.0 and final_rms <= 1.05 * best_rms


__all__ = [
    "ConvexinvError", "ConvexinvParameters", "ConvexinvProvenance", "ConvexinvResult",
    "PeriodScanParameters", "PeriodScanResult", "PeriodScanRow", "axis_to_six_pole_starts",
    "inversion_success", "parse_period_scan_rows", "relative_rms_from_modelled_lightcurve",
    "run_convexinv", "run_period_scan", "write_convexinv_parameters", "write_period_scan_parameters",
]
