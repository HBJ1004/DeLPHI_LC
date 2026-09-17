"""Contracts for bounded, label-blind solver-capacity preflight."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from lc_pipeline.k3.convergence_study import (
    _capacity_revision_payload,
    _verify_source_matches_archive,
)
from lc_pipeline.k3.solver_capacity import (
    INTERNAL_CAPACITY_PADDING_VERSION,
    LightcurveStructure,
    SolverCapacityError,
    capacity_violations,
    internal_capacity_padding,
    internal_capacity_requirements,
    require_current_internal_capacity_receipt,
    require_declared_support,
    tree_sha256,
)
from repro.prepare_k3_solver_capacity import prepare


@pytest.mark.parametrize(
    ("structure", "expected"),
    [
        (LightcurveStructure(1, 7, 3), ()),
        (LightcurveStructure(2, 6, 3), ("MAX_LC",)),
        (LightcurveStructure(1, 8, 3), ("MAX_N_OBS",)),
        (LightcurveStructure(1, 4, 4), ("POINTS_MAX",)),
    ],
)
def test_capacity_edges_reject_only_the_strictly_exceeded_macro(structure, expected):
    capacities = {"MAX_LC": 2, "MAX_N_OBS": 10, "POINTS_MAX": 3}
    assert capacity_violations(structure, capacities) == expected


def test_required_support_fails_closed_instead_of_changing_the_population():
    rows = [{"violations": []}, {"violations": ["MAX_N_OBS"]}]
    with pytest.raises(SolverCapacityError, match="all 2 declared objects, but only 1"):
        require_declared_support(rows, role="development")


def _write_lightcurve(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("1\n2 1\n0 0 0 0 0 0 0 0\n0 0 0 0 0 0 0 0\n", encoding="ascii")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_solver(version_root: Path) -> None:
    for name in ("convexinv", "conjgradinv"):
        directory = version_root / name
        directory.mkdir(parents=True)
        (directory / "constants.h").write_text(
            "#define POINTS_MAX 3\n#define MAX_N_OBS 10\n#define MAX_LC 2\n", encoding="ascii"
        )
    convex = version_root / "convexinv"
    (convex / "convexinv.c").write_text("int main(void) { return 0; }\n", encoding="ascii")
    (convex / "period_scan.c").write_text("int main(void) { return 0; }\n", encoding="ascii")
    (convex / "Makefile").write_text(
        "convexinv: convexinv.c\n\t$(CC) -o $@ $<\n"
        "period_scan: period_scan.c\n\t$(CC) -o $@ $<\n"
        "clean:\n\trm -f convexinv period_scan *.o\n",
        encoding="ascii",
    )
    # The preflight records the official binary even when this fixture skips
    # the real development parity smoke.
    (convex / "convexinv").write_bytes(b"fixture-official-binary")


def _fixture_inputs(tmp_path: Path) -> dict[str, Path]:
    dump = tmp_path / "dump"
    digest = _write_lightcurve(dump / "files" / "dev" / "lc.txt")
    locked_digest = _write_lightcurve(dump / "files" / "locked" / "lc.txt")
    blind = tmp_path / "blind.json"
    blind.write_text(
        json.dumps(
            {
                "objects": [
                    {"object_id": "dev", "lightcurve": {"source_path": "files/dev/lc.txt", "source_sha256": digest}},
                    {"object_id": "locked", "lightcurve": {"source_path": "files/locked/lc.txt", "source_sha256": locked_digest}},
                ]
            }
        ),
        encoding="utf-8",
    )
    development = tmp_path / "development.json"
    development.write_text(json.dumps({"role": "development", "object_ids": ["dev"]}), encoding="utf-8")
    locked = tmp_path / "locked.json"
    locked.write_text(json.dumps({"role": "locked_evaluation", "object_ids": ["locked"]}), encoding="utf-8")
    solver = tmp_path / "solver"
    _write_solver(solver)
    return {"blind": blind, "development": development, "locked": locked, "dump": dump, "solver": solver}


def test_expanded_copy_preserves_original_source_and_expands_only_copy(tmp_path: Path):
    if shutil.which("cc") is None or shutil.which("make") is None:
        pytest.skip("fixture compilation requires a POSIX C compiler and make")
    values = _fixture_inputs(tmp_path)
    before = tree_sha256(values["solver"] / "convexinv")
    output = tmp_path / "capacity-output"

    report = prepare(
        blind_inputs=values["blind"],
        development_manifest=values["development"],
        locked_manifest=values["locked"],
        dump_root=values["dump"],
        solver_root=values["solver"],
        output_root=output,
        run_dev_parity_smoke=False,
        run_sanitizer_checks=False,
    )

    assert tree_sha256(values["solver"] / "convexinv") == before
    assert report["expanded_solver_preflight"]["capacities"] == {
        "POINTS_MAX": 3,
        "MAX_N_OBS": 10,
        "MAX_LC": 2,
    }
    assert report["schema"] == "delphi.k3-solver-internal-capacity-preflight.v2"
    assert report["internal_capacity_padding"]["version"] == INTERNAL_CAPACITY_PADDING_VERSION
    assert (output / "expanded-solver" / "convexinv" / "period_scan").is_file()
    assert (output / "expanded-solver" / "convexinv" / "convexinv").is_file()
    assert report["original_solver_preflight"]["solver_execution_authorized"] is True


def test_preflight_refuses_existing_output_or_an_output_inside_original(tmp_path: Path):
    values = _fixture_inputs(tmp_path)
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    common = dict(
        blind_inputs=values["blind"],
        development_manifest=values["development"],
        locked_manifest=values["locked"],
        dump_root=values["dump"],
        solver_root=values["solver"],
        run_dev_parity_smoke=False,
        run_sanitizer_checks=False,
    )
    with pytest.raises(SolverCapacityError, match="refusing to overwrite"):
        prepare(**common, output_root=occupied)
    with pytest.raises(SolverCapacityError, match="must not be the original"):
        prepare(**common, output_root=values["solver"] / "convexinv" / "capacity-copy")


def test_boundary_capacity_reserves_solver_internal_regularization_append():
    structure = LightcurveStructure(209, 26609, 1549)
    assert internal_capacity_requirements(structure) == {
        "POINTS_MAX": 1549,
        "MAX_N_OBS": 26612,
        "MAX_LC": 210,
    }
    assert capacity_violations(
        structure, {"POINTS_MAX": 1549, "MAX_N_OBS": 26609, "MAX_LC": 209}
    ) == ("MAX_N_OBS", "MAX_LC")


def test_old_capacity_receipt_is_rejected_without_current_padding_contract():
    with pytest.raises(SolverCapacityError, match="current internal-padding schema"):
        require_current_internal_capacity_receipt(
            {"schema": "delphi.k3-solver-capacity-preflight.v1"}
        )
    require_current_internal_capacity_receipt(
        {
            "schema": "delphi.k3-solver-internal-capacity-preflight.v2",
            "internal_capacity_padding": internal_capacity_padding(),
        }
    )


def test_unsafe_pre_padding_capacity_revision_is_rejected(tmp_path: Path):
    """The prior receipt must not authorize a solver that appends past its limits."""
    root = Path(__file__).resolve().parents[1]
    data = root.parent / "data" / "solver-capacity-revision"
    report = data / "solver-capacity-preflight.json"
    source = data / "expanded-solver" / "convexinv"
    binary = source / "convexinv"
    archive = root.parent / "inputs" / "solver" / "damit-version_0.2.1.tar.gz"
    blind = root / "repro/data/k3-followup-20260910/convergence-blind-inputs.json"
    required = (report, source, binary, archive, blind)
    if not all(path.exists() for path in required):
        pytest.skip("local sealed solver-capacity evidence is not present in this checkout")
    with pytest.raises(Exception, match="differs from the official archive"):
        _verify_source_matches_archive(source, archive)
    with pytest.raises(Exception, match="current internal-padding schema"):
        _capacity_revision_payload(
            revision_path=report, source_archive=archive, source_root=source,
            executable=binary, blind_inputs_path=blind,
        )

    tampered = tmp_path / "tampered-capacity-report.json"
    payload = json.loads(report.read_text())
    payload["expanded_solver"]["binary_sha256"] = "0" * 64
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(Exception, match="current internal-padding schema"):
        _capacity_revision_payload(
            revision_path=tampered, source_archive=archive, source_root=source,
            executable=binary, blind_inputs_path=blind,
        )
