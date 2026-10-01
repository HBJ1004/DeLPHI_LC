"""Offline tests for the novice example scripts under examples/."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

from lc_pipeline.k3.predict import read_observations

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"example_{name}", EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


converter = _load("damit_to_observations")
poles = _load("prediction_to_poles")

# Two tiny DAMIT lightcurves (3 and 2 points) in the lc.txt layout:
# JD, linear intensity, asteroid->Sun xyz, asteroid->Earth xyz (ecliptic J2000, au).
LC_TXT = """2
3 0
2450000.10 1.010  1.5 -0.2 0.05   0.6 -0.9 0.03
2450000.12 0.980  1.5 -0.2 0.05   0.6 -0.9 0.03
2450000.14 1.020  1.5 -0.2 0.05   0.6 -0.9 0.03
2 1
2450400.30 0.950  -1.1 1.7 -0.04  -0.3 0.8 -0.02
2450400.31 1.050  -1.1 1.7 -0.04  -0.3 0.8 -0.02
"""

# Minimal DAMIT asteroid_models table: asteroid 777 has a model without a
# quality flag (id 50) and two accepted models (ids 90 and 60).
MODELS_CSV = """id,asteroid_id,lambda,beta,period,quality_flag
50,777,10,20,5.1,
90,777,30,40,5.3,3
60,777,50,60,5.2,4
70,888,1,2,9.9,2
"""
ASTEROIDS_CSV = "id,number,name,designation\n777,1234,Testname,\n"


@pytest.fixture()
def lc_file(tmp_path: Path) -> Path:
    path = tmp_path / "lc.txt"
    path.write_text(LC_TXT, encoding="ascii")
    return path


def test_local_lc_with_supplied_period(tmp_path, lc_file) -> None:
    output = tmp_path / "observations.json"
    assert converter.main([
        "--lc", str(lc_file), "--object-id", "my-asteroid",
        "--period-hours", "7.25", "--period-provenance", "test period source",
        "--output", str(output),
    ]) == 0
    object_id, period, epochs = read_observations(output)
    assert object_id == "my-asteroid"
    assert (period.hours, period.provenance) == (7.25, "test period source")
    # One DAMIT lightcurve becomes one epoch; numbers are copied unchanged.
    assert [epoch.epoch_id for epoch in epochs] == ["damit-lc-0000", "damit-lc-0001"]
    assert [len(epoch.observations) for epoch in epochs] == [3, 2]
    first = epochs[1].observations[0]
    assert first.time_jd == 2450400.30
    assert first.relative_brightness == 0.95
    assert first.sun_asteroid_ecliptic_j2000_au == (-1.1, 1.7, -0.04)
    assert first.observer_asteroid_ecliptic_j2000_au == (-0.3, 0.8, -0.02)
    assert first.measured_error is None  # DAMIT has no per-point errors


def test_period_from_local_model_table_follows_paper_rule(tmp_path, lc_file) -> None:
    models = tmp_path / "asteroid_models.csv"
    models.write_text(MODELS_CSV, encoding="utf-8")
    asteroids = tmp_path / "asteroids.csv"
    asteroids.write_text(ASTEROIDS_CSV, encoding="utf-8")
    output = tmp_path / "observations.json"
    converter.main([
        "--damit-id", "777", "--lc", str(lc_file), "--models-table", str(models),
        "--asteroids-table", str(asteroids), "--output", str(output),
    ])
    object_id, period, _ = read_observations(output)
    assert object_id == "asteroid_777"  # the exact benchmark-style ID
    # Lowest model id among quality >= 3 (model 60), not the unflagged model 50.
    assert period.hours == 5.2
    assert "DAMIT model 60 of (1234) Testname (DAMIT asteroid 777)" in period.provenance
    assert "quality flag 4" in period.provenance


def test_model_choice_override_and_error(tmp_path, lc_file) -> None:
    models = tmp_path / "asteroid_models.csv"
    models.write_text(MODELS_CSV, encoding="utf-8")
    output = tmp_path / "observations.json"
    converter.main([
        "--damit-id", "777", "--lc", str(lc_file), "--models-table", str(models),
        "--model-id", "50", "--output", str(output),
    ])
    assert read_observations(output)[1].hours == 5.1
    with pytest.raises(SystemExit, match="quality flag >= 3"):
        converter.main([
            "--damit-id", "888", "--lc", str(lc_file), "--models-table", str(models),
            "--output", str(tmp_path / "other.json"),
        ])


def test_download_mode_uses_damit_exports(tmp_path, monkeypatch) -> None:
    requested: list[str] = []
    # The live DAMIT CSV exports begin with a UTF-8 byte-order mark.
    responses = {
        converter.LIGHTCURVE_URL.format(damit_id=777): LC_TXT.encode("ascii"),
        converter.MODEL_TABLE_URL: b"\xef\xbb\xbf" + MODELS_CSV.encode("utf-8"),
        converter.ASTEROID_TABLE_URL: b"\xef\xbb\xbf" + ASTEROIDS_CSV.encode("utf-8"),
    }

    def fake_fetch(url: str) -> bytes:
        requested.append(url)
        return responses[url]

    monkeypatch.setattr(converter, "fetch", fake_fetch)
    output = tmp_path / "observations.json"
    converter.main(["--damit-id", "777", "--output", str(output)])
    assert requested[0] == (
        "https://damit.cuni.cz/projects/damit/light_curves/exportAllForAsteroid/777/plaintext"
    )
    assert (tmp_path / "observations.lc.txt").read_text(encoding="ascii") == LC_TXT
    object_id, period, epochs = read_observations(output)
    assert (object_id, period.hours, len(epochs)) == ("asteroid_777", 5.2, 2)
    assert "exports/table/asteroid_models, retrieved" in period.provenance


def test_refuses_to_overwrite_and_needs_a_period(tmp_path, lc_file) -> None:
    output = tmp_path / "observations.json"
    output.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        converter.main(["--lc", str(lc_file), "--object-id", "x", "--period-hours", "5",
                        "--period-provenance", "p", "--output", str(output)])
    with pytest.raises(SystemExit):
        converter.main(["--lc", str(lc_file), "--object-id", "x",
                        "--output", str(tmp_path / "new.json")])


def test_axis_to_both_ends_in_ecliptic_coordinates() -> None:
    (lam, beta), (lam2, beta2) = poles.axis_to_poles([-1.0, 0.0, 0.0])
    assert (round(lam, 9), round(beta, 9), round(lam2, 9) % 360, round(beta2, 9)) == (180, 0, 0, 0)
    (lam, beta), (lam2, beta2) = poles.axis_to_poles(poles.pole_to_axis(124.0, 39.0))
    assert math.isclose(lam, 124.0) and math.isclose(beta, 39.0)
    assert math.isclose(lam2, 304.0) and math.isclose(beta2, -39.0)
    assert math.isclose(poles.axial_angle_deg([0, 0, 1], [0, 0, -1]), 0.0, abs_tol=1e-9)


def test_expected_astraea_prediction_gives_six_starting_poles() -> None:
    expected = json.loads((EXAMPLES / "astraea" / "expected_prediction.json").read_text())
    rows = poles.starting_poles(expected)
    assert len(rows) == 6 and {row["candidate"] for row in rows} == {1, 2, 3}
    assert poles.max_difference_deg(expected, expected) == pytest.approx(0.0, abs=1e-4)
    # The reference pole of (5) Astraea (DAMIT model 1816: 124, +39) is 11.9 deg
    # from the nearest candidate axis in the run-0 bundle.
    reference = poles.pole_to_axis(124.0, 39.0)
    oracle = min(poles.axial_angle_deg(axis["axis_xyz"], reference) for axis in expected["axes"])
    assert oracle == pytest.approx(11.92, abs=0.01)


def test_astraea_example_input_is_valid() -> None:
    object_id, period, epochs = read_observations(EXAMPLES / "astraea" / "observations.json")
    assert object_id == "asteroid_103"
    assert period.hours == 16.80059
    assert len(epochs) == 25 and sum(len(epoch.observations) for epoch in epochs) == 881


def test_horizons_helper_converts_magnitudes_to_linear_flux(tmp_path) -> None:
    horizons = _load("horizons_geometry")  # astroquery is imported only when querying
    path = tmp_path / "photometry.csv"
    path.write_text("epoch_id,jd,mag,mag_err\nn1,2460000.1,12.0,0.01\nn1,2460000.2,12.5,0.02\n",
                    encoding="utf-8")
    epoch_ids, jd, flux, error = horizons.read_photometry(path)
    assert epoch_ids == ["n1", "n1"]
    # Brighter (smaller magnitude) means larger flux; 0.5 mag is a factor 10**0.2.
    assert flux[0] / flux[1] == pytest.approx(10 ** 0.2)
    assert error[0] == pytest.approx(0.4 * math.log(10) * flux[0] * 0.01)
