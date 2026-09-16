"""Bounded, label-blind capacity checks for the DAMIT convergence solver.

The convergence study consumes a frozen blind-input artifact.  This module is
deliberately narrower: it reads only the object identifier, lightcurve path,
and lightcurve bytes needed to count the three static C limits.  In
particular, it never reads K3 axes, catalog solutions, fit results, or any
reference pole.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


class SolverCapacityError(ValueError):
    """Raised when a capacity preflight cannot establish safe support."""


CAPACITY_FIELDS = ("POINTS_MAX", "MAX_N_OBS", "MAX_LC")
INTERNAL_CAPACITY_PADDING_VERSION = "convexinv-regularization-append-v1"
INTERNAL_REGULARIZATION_OBSERVATIONS = 3
INTERNAL_REGULARIZATION_LIGHTCURVES = 1
INTERNAL_MIN_REGULARIZATION_POINTS = 3
INTERNAL_CAPACITY_RECEIPT_SCHEMA = "delphi.k3-solver-internal-capacity-preflight.v2"
_DEFINE = re.compile(r"^\s*#define\s+(POINTS_MAX|MAX_N_OBS|MAX_LC)\s+(\d+)\b")


@dataclass(frozen=True)
class LightcurveStructure:
    """Counts parsed from the lightcurve framing, without retaining photometry."""

    lightcurve_count: int
    total_observations: int
    max_points_per_lightcurve: int

    def as_dict(self) -> dict[str, int]:
        return {
            "lightcurve_count": self.lightcurve_count,
            "total_observations": self.total_observations,
            "max_points_per_lightcurve": self.max_points_per_lightcurve,
        }


def internal_capacity_requirements(structure: LightcurveStructure) -> dict[str, int]:
    """Return allocations required after convexinv appends regularization data.

    ``convexinv.c`` appends one three-point regularization lightcurve after
    parsing every native input.  The native parser checks only the input
    dimensions, so admission must reserve those later writes as well.
    """
    return {
        "POINTS_MAX": max(structure.max_points_per_lightcurve, INTERNAL_MIN_REGULARIZATION_POINTS),
        "MAX_N_OBS": structure.total_observations + INTERNAL_REGULARIZATION_OBSERVATIONS,
        "MAX_LC": structure.lightcurve_count + INTERNAL_REGULARIZATION_LIGHTCURVES,
    }


def internal_capacity_padding() -> dict[str, int | str]:
    """Describe the solver-internal allocation appended after native input parsing."""
    return {
        "version": INTERNAL_CAPACITY_PADDING_VERSION,
        "regularization_observations_added": INTERNAL_REGULARIZATION_OBSERVATIONS,
        "regularization_lightcurves_added": INTERNAL_REGULARIZATION_LIGHTCURVES,
        "minimum_regularization_points_per_lightcurve": INTERNAL_MIN_REGULARIZATION_POINTS,
    }


def require_current_internal_capacity_receipt(receipt: Mapping[str, object]) -> None:
    """Reject pre-correction receipts before a new locked solver execution.

    This checks only the receipt's correction contract.  Callers still must
    verify their own source, executable, cohort, and artifact hashes.
    """
    if receipt.get("schema") != INTERNAL_CAPACITY_RECEIPT_SCHEMA:
        raise SolverCapacityError("capacity receipt does not bind the current internal-padding schema")
    if receipt.get("internal_capacity_padding") != internal_capacity_padding():
        raise SolverCapacityError("capacity receipt does not bind the current internal-padding version")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(root: str | Path) -> str:
    """Hash regular source files deterministically, including their relative names."""
    source = Path(root)
    if not source.is_dir():
        raise SolverCapacityError(f"source root is not a directory: {source}")
    digest = hashlib.sha256()
    for path in sorted(candidate for candidate in source.rglob("*") if candidate.is_file()):
        relative = path.relative_to(source).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def solver_directory(solver_root: str | Path, solver: str = "convexinv") -> Path:
    """Accept either the version root or the solver's own source directory."""
    root = Path(solver_root).resolve()
    candidate = root if (root / "constants.h").is_file() else root / solver
    if not candidate.is_dir() or not (candidate / "constants.h").is_file():
        raise SolverCapacityError(f"cannot locate {solver} constants.h below {root}")
    return candidate


def read_static_capacities(solver_root: str | Path, solver: str = "convexinv") -> dict[str, int]:
    """Read the exact three checked C limits from an unmodified constants header."""
    header = solver_directory(solver_root, solver) / "constants.h"
    values: dict[str, int] = {}
    for line in header.read_text(encoding="ascii").splitlines():
        match = _DEFINE.match(line)
        if match:
            values[match.group(1)] = int(match.group(2))
    if set(values) != set(CAPACITY_FIELDS) or any(value <= 0 for value in values.values()):
        raise SolverCapacityError(f"{header} does not define all positive solver capacity limits")
    return {name: values[name] for name in CAPACITY_FIELDS}


def parse_lightcurve_structure(path: str | Path) -> LightcurveStructure:
    """Parse only the line framing of a DAMIT lightcurve input.

    A solver observation is one nonempty physical line in the frozen input
    format.  We intentionally do not parse its brightness or geometry values.
    """
    try:
        lines = [line for line in Path(path).read_text(encoding="ascii").splitlines() if line.strip()]
    except (OSError, UnicodeDecodeError) as exc:
        raise SolverCapacityError(f"cannot read lightcurve structure from {path}: {exc}") from exc
    if not lines:
        raise SolverCapacityError(f"lightcurve is empty: {path}")
    try:
        lightcurve_count = int(lines[0].strip())
    except ValueError as exc:
        raise SolverCapacityError(f"lightcurve count is invalid: {path}") from exc
    if lightcurve_count < 1:
        raise SolverCapacityError(f"lightcurve count must be positive: {path}")
    cursor = 1
    point_counts: list[int] = []
    for _ in range(lightcurve_count):
        if cursor >= len(lines):
            raise SolverCapacityError(f"lightcurve header is truncated: {path}")
        fields = lines[cursor].split()
        if len(fields) != 2:
            raise SolverCapacityError(f"lightcurve block header is invalid: {path}")
        try:
            points = int(fields[0])
            int(fields[1])
        except ValueError as exc:
            raise SolverCapacityError(f"lightcurve block header is invalid: {path}") from exc
        if points < 1 or cursor + points >= len(lines) + 1:
            raise SolverCapacityError(f"lightcurve block is truncated or empty: {path}")
        point_counts.append(points)
        cursor += points + 1
    if cursor != len(lines):
        raise SolverCapacityError(f"lightcurve has unexpected trailing records: {path}")
    return LightcurveStructure(
        lightcurve_count=lightcurve_count,
        total_observations=sum(point_counts),
        max_points_per_lightcurve=max(point_counts),
    )


def capacity_violations(
    structure: LightcurveStructure, capacities: Mapping[str, int]
) -> tuple[str, ...]:
    """Return the C macro names that would reject this input."""
    required = internal_capacity_requirements(structure)
    if set(capacities) != set(CAPACITY_FIELDS):
        raise SolverCapacityError("capacity mapping must contain exactly the three static C limits")
    return tuple(name for name in CAPACITY_FIELDS if required[name] > int(capacities[name]))


def required_capacities(structures: Iterable[LightcurveStructure]) -> dict[str, int]:
    values = list(structures)
    if not values:
        raise SolverCapacityError("cannot derive capacities from an empty cohort")
    requirements = [internal_capacity_requirements(value) for value in values]
    return {name: max(int(value[name]) for value in requirements) for name in CAPACITY_FIELDS}


def feasible_support(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Summarize how many declared objects fit, without changing the cohort."""
    violations = [row for row in rows if row["violations"]]
    return {
        "declared_object_count": len(rows),
        "feasible_object_count": len(rows) - len(violations),
        "infeasible_object_count": len(violations),
        "required_support_met": len(violations) == 0,
    }


def require_declared_support(rows: Sequence[Mapping[str, object]], *, role: str) -> None:
    """Fail closed rather than quietly dropping capacity-excluded objects."""
    summary = feasible_support(rows)
    if not summary["required_support_met"]:
        raise SolverCapacityError(
            f"{role} requires all {summary['declared_object_count']} declared objects, "
            f"but only {summary['feasible_object_count']} fit solver capacity"
        )


def load_blind_lightcurve_rows(
    blind_inputs_path: str | Path,
    dump_root: str | Path,
    *,
    object_ids: Sequence[str] | None = None,
) -> list[dict[str, object]]:
    """Load and hash only lightcurve metadata from the blind execution artifact."""
    try:
        document = json.loads(Path(blind_inputs_path).read_text(encoding="utf-8"))
        objects = document["objects"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise SolverCapacityError(f"cannot read blind lightcurve metadata: {exc}") from exc
    if not isinstance(objects, list):
        raise SolverCapacityError("blind execution inputs must contain an object list")
    requested = None if object_ids is None else set(object_ids)
    root = Path(dump_root).resolve()
    result: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in objects:
        if not isinstance(row, Mapping):
            raise SolverCapacityError("blind execution object row is invalid")
        object_id = row.get("object_id")
        metadata = row.get("lightcurve")
        if not isinstance(object_id, str) or not isinstance(metadata, Mapping):
            raise SolverCapacityError("blind execution object lacks lightcurve metadata")
        if object_id in seen:
            raise SolverCapacityError(f"duplicate blind object identifier: {object_id}")
        seen.add(object_id)
        if requested is not None and object_id not in requested:
            continue
        source_path = metadata.get("source_path")
        expected_hash = metadata.get("source_sha256")
        if not isinstance(source_path, str) or not isinstance(expected_hash, str):
            raise SolverCapacityError(f"invalid lightcurve metadata for {object_id}")
        relative = Path(source_path)
        resolved = (root / relative).resolve()
        if relative.is_absolute() or not resolved.is_relative_to(root):
            raise SolverCapacityError(f"lightcurve path escapes dump root for {object_id}")
        if not resolved.is_file() or sha256_file(resolved) != expected_hash:
            raise SolverCapacityError(f"lightcurve does not match blind metadata for {object_id}")
        result.append(
            {
                "object_id": object_id,
                "source_path": relative.as_posix(),
                "source_sha256": expected_hash,
                "resolved_path": str(resolved),
                "structure": parse_lightcurve_structure(resolved),
            }
        )
    if requested is not None:
        missing = requested - {str(row["object_id"]) for row in result}
        if missing:
            raise SolverCapacityError("cohort object is absent from blind execution inputs")
    return result
