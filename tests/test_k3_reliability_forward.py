"""Independent checks for the convexinv forward-only reliability renderer."""

from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from lc_pipeline.k3.reliability_forward import (
    ConvexinvSolution,
    ForwardLightcurve,
    ForwardObservation,
    ReliabilityForwardError,
    holdout_shape_rmse,
    load_solution,
    native_output_shape,
    native_relative_shape,
    predict_curve,
    read_lightcurves,
)
from repro.prepare_k3_reliability_solver import patch_full_precision_export


def _solution() -> ConvexinvSolution:
    # The fourth entry is deliberately a dark closure facet: it must not alter
    # forward brightness, even though its normal faces the observer.
    return ConvexinvSolution(
        longitude_deg=371.25, latitude_deg=123.5, period_hours=7.25,
        epoch_jd=2450000.0, phase_deg=-27.125, phase_a=0.5, phase_d=0.1,
        phase_k=-0.5, lambert=0.1,
        areas=np.array([0.7, 0.2, 0.1]),
        normals=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    )


def _observations() -> tuple[ForwardObservation, ...]:
    return tuple(ForwardObservation(
        2450000.0 + index * 0.071,
        1.0 + 0.1 * index,
        (1.0, 0.12 * (index + 1), 0.07),
        (0.6 - 0.05 * index, 0.8, 0.11),
    ) for index in range(5))


def test_synthetic_forward_is_shape_only_and_does_not_use_withheld_flux() -> None:
    solution, observations = _solution(), _observations()
    prediction = native_relative_shape(solution, observations)
    changed_flux = tuple(ForwardObservation(item.time_jd, 1000.0 - item.flux, item.sun, item.observer)
                         for item in observations)
    assert np.allclose(prediction, native_relative_shape(solution, changed_flux), rtol=0, atol=0)
    exact = ForwardLightcurve(tuple(ForwardObservation(item.time_jd, float(value), item.sun, item.observer)
                                    for item, value in zip(observations, prediction, strict=True)), relative=True)
    assert holdout_shape_rmse(solution, (exact,)) == pytest.approx(0.0, abs=1e-14)
    # A calibrated native curve retains phase while a relative one has it
    # divided before curve scaling.  Both branches are forward-only.
    absolute_prediction = native_output_shape(solution, observations, relative=False)
    assert not np.allclose(prediction, absolute_prediction)
    absolute = ForwardLightcurve(tuple(ForwardObservation(item.time_jd, float(value), item.sun, item.observer)
                                       for item, value in zip(observations, absolute_prediction, strict=True)), relative=False)
    assert holdout_shape_rmse(solution, (exact, absolute)) == pytest.approx(0.0, abs=1e-14)
    # Metric is mean(per-curve MSE), not point-weighted MSE.
    perturbed = ForwardLightcurve((exact.observations[0],) + tuple(
        ForwardObservation(item.time_jd, item.flux * 1.1, item.sun, item.observer)
        for item in exact.observations[1:]), relative=True)
    manual = math.sqrt(np.mean([
        np.mean((native_relative_shape(solution, exact.observations) -
                 np.asarray([item.flux for item in exact.observations]) / np.mean([item.flux for item in exact.observations])) ** 2),
        np.mean((native_relative_shape(solution, perturbed.observations) -
                 np.asarray([item.flux for item in perturbed.observations]) / np.mean([item.flux for item in perturbed.observations])) ** 2),
    ]))
    assert holdout_shape_rmse(solution, (exact, perturbed)) == pytest.approx(manual)


@pytest.mark.parametrize("bad", [np.nan, np.inf, 0.0])
def test_invalid_fluxes_and_predictions_are_rejected(bad: float) -> None:
    solution, observations = _solution(), _observations()
    invalid = ForwardLightcurve(tuple(ForwardObservation(item.time_jd, bad, item.sun, item.observer)
                                      for item in observations), relative=True)
    with pytest.raises(ReliabilityForwardError):
        holdout_shape_rmse(solution, (invalid,))
    if bad != 0:
        bad_time = ForwardObservation(bad, 1.0, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        with pytest.raises(ReliabilityForwardError):
            predict_curve(solution, (bad_time,))


def test_export_reader_preserves_raw_euler_values_and_excludes_dark_facet(tmp_path: Path) -> None:
    params = tmp_path / "parameters.txt"
    areas = tmp_path / "areas.txt"
    params.write_text("371.125 123.5 7.25\n2450000.125 -27.125\n0.5 0.1 -0.5\n0.1\n", encoding="ascii")
    # A perfectly closed native solution can emit nan for the deliberately
    # discarded dark normal (division by a zero dark area); it is not a facet.
    areas.write_text("4\n0.7\n1 0 0\n0.2\n0 1 0\n0.1\n0 0 1\n999\nnan nan nan\n", encoding="ascii")
    solution = load_solution(params, areas)
    assert solution.longitude_deg == 371.125
    assert solution.latitude_deg == 123.5
    assert solution.phase_deg == -27.125
    assert solution.areas.tolist() == [0.7, 0.2, 0.1]
    assert solution.normals.shape == (3, 3)


def test_native_input_reader_rejects_nonpositive_flux(tmp_path: Path) -> None:
    source = tmp_path / "bad.lcs"
    source.write_text("1\n1 0\n2450000 0 1 0 0 0 1 0 0\n", encoding="ascii")
    with pytest.raises(ReliabilityForwardError, match="positive"):
        read_lightcurves(source)


def _technical_parameters() -> str:
    return "37.125 0\n123.5 0\n7.25 0\n2450000\n-27.125\n0.1\n0 0\n1\n0.5 0\n0.1 0\n-0.5 0\n0.1 0\n2\n"


def _technical_lightcurve() -> str:
    rows = [
        "2450000.000 1.00 1 .12 .07 .60 .80 .11",
        "2450000.071 1.10 1 .24 .07 .55 .80 .11",
        "2450000.142 0.95 1 .36 .07 .50 .80 .11",
        "2450000.213 1.05 1 .48 .07 .45 .80 .11",
        "2450000.284 1.02 1 .60 .07 .40 .80 .11",
    ]
    return "1\n5 0\n" + "\n".join(rows) + "\n"


def test_patched_solver_native_relative_parity_on_independent_tiny_fixture(tmp_path: Path) -> None:
    """The source translation matches native normalized relative output <=1e-5."""
    root = Path(__file__).resolve().parents[2]
    original = root / "data" / "solver-internal-capacity-corrected" / "20260911" / "expanded-solver" / "convexinv"
    if not (original / "convexinv").is_file() or shutil.which("make") is None:
        pytest.skip("technical convexinv fixture is unavailable")
    copy = tmp_path / "convexinv"
    shutil.copytree(original, copy)
    patch_full_precision_export(copy)
    build = subprocess.run(["make", "convexinv"], cwd=copy, text=True, capture_output=True, check=False)
    assert build.returncode == 0, build.stderr or build.stdout
    parameter_path, input_path = tmp_path / "input.parameters", tmp_path / "input.lcs"
    parameter_path.write_text(_technical_parameters(), encoding="ascii")
    input_path.write_text(_technical_lightcurve(), encoding="ascii")
    modelled, exported_parameters, exported_areas = tmp_path / "modelled.txt", tmp_path / "solution-parameters.txt", tmp_path / "solution-areas.txt"
    with input_path.open("rb") as stream:
        run = subprocess.run([str(copy / "convexinv"), "-v", "-o", str(exported_areas), "-p", str(exported_parameters),
                              str(parameter_path), str(modelled)], stdin=stream, cwd=tmp_path,
                             capture_output=True, check=False)
    assert run.returncode == 0, run.stderr.decode(errors="replace")
    solution = load_solution(exported_parameters, exported_areas)
    curve, = read_lightcurves(input_path)
    forward = native_relative_shape(solution, curve.observations)
    native = np.loadtxt(modelled, dtype=np.float64)
    residual = np.sqrt(np.mean(np.square(forward - native / np.mean(native))))
    assert residual <= 1e-5
