from pathlib import Path

import numpy as np

from lc_pipeline.k3.reliability_sampling import (
    CurveObservation,
    LightCurve,
    NativeSession,
    add_multiplicative_noise,
    merge_observing_blocks,
    read_lc,
    select_holdout,
    select_observations,
    write_lc,
)


def _obs(t: float, flux: float = 1.0) -> CurveObservation:
    return CurveObservation(t, flux, (1, 0, 0), (0, 1, 0))


def _curve() -> LightCurve:
    sessions = tuple(
        NativeSession(
            f"s{i}", tuple(_obs(1 + i * 40 + j, 1 + j / 10) for j in range(5)), i % 2, 5, i
        )
        for i in range(4)
    )
    return LightCurve(sessions, "stable-object", 10.0)


def test_native_roundtrip_keeps_session_headers_and_immutability(tmp_path: Path):
    source = tmp_path / "lc.txt"
    write_lc(_curve(), source)
    curve = read_lc(source, object_id="stable-object", period_hours=10)
    assert [session.calibrated for session in curve.sessions] == [0, 1, 0, 1]
    before = curve.sessions[0].observations
    selected = select_observations(curve, {"kind": "point", "fraction": 0.5, "seed": 20260915})
    assert curve.sessions[0].observations == before
    assert all(len(session.observations) == 2 for session in selected.sessions)


def test_blocks_merge_only_gaps_less_than_thirty_days():
    curve = LightCurve(
        (
            NativeSession("a", (_obs(1), _obs(2)), 0),
            NativeSession("b", (_obs(20), _obs(21)), 1),
            NativeSession("c", (_obs(52), _obs(53)), 0),
        )
    )
    merged = merge_observing_blocks(curve)
    assert [[s.session_id for s in block.sessions] for block in merged.blocks] == [
        ["a", "b"],
        ["c"],
    ]


def test_cap_is_deterministic_and_noise_is_positive_mean_preserving():
    curve = _curve()
    config = {"kind": "observation_cap", "cap": 10, "seed": 20260916}
    one = select_observations(curve, config)
    two = select_observations(curve, config)
    assert one == two
    noisy = add_multiplicative_noise(curve, 0.1, seed=20260917)
    assert all(item.flux > 0 for item in noisy.observations)
    assert np.isclose(
        np.mean([a.flux for a in curve.observations]), np.mean([a.flux for a in noisy.observations])
    )


def test_nested_point_levels_and_two_d_accounting():
    # Deliberately separated sessions give one merged block per session.
    curve = LightCurve(
        tuple(
            NativeSession(f"s{i}", tuple(_obs(1 + i * 100 + j) for j in range(40)), 0, 40, i)
            for i in range(12)
        ),
        "nested",
    )
    low = select_observations(curve, {"kind": "point", "fraction": 0.1, "seed": 20260915})
    high = select_observations(curve, {"kind": "point", "fraction": 0.5, "seed": 20260915})
    assert set(item.time_jd for item in low.observations) <= set(
        item.time_jd for item in high.observations
    )
    two_d = select_observations(
        curve, {"kind": "two_d", "block_count": 3, "per_block_cap": 5, "seed": 20260915}
    )
    assert len(two_d.blocks) == 3
    assert len(two_d.observations) == 15
    assert all(len(session.observations) >= 2 for session in two_d.sessions)


def test_atomic_cap_prefixes_are_nested_and_do_not_pad():
    curve = LightCurve(
        tuple(
            NativeSession(f"s{i}", tuple(_obs(1 + i * 100 + j) for j in range(7)), 0, 7, i)
            for i in range(4)
        ),
        "caps",
    )
    previous: set[float] = set()
    for cap in (3, 4, 5, 6, 10):
        value = select_observations(
            curve, {"kind": "observation_cap", "cap": cap, "seed": 20260915}
        )
        current = {item.time_jd for item in value.observations}
        assert previous <= current
        assert len(current) <= cap
        assert all(len(session.observations) >= 2 for session in value.sessions)
        previous = current


def test_latest_block_holdout_is_strict_and_has_no_session_leakage():
    curve = LightCurve(
        tuple(
            NativeSession(f"s{i}", tuple(_obs(1 + i * 100 + j) for j in range(40)), 0, 40, i)
            for i in range(12)
        )
    )
    result = select_holdout(curve)
    train_ids = {session.session_id for session in result.train.sessions}
    withheld_ids = {session.session_id for session in result.withheld.sessions}
    assert result.block.block_id == "block-0011"
    assert train_ids.isdisjoint(withheld_ids)
    assert len(result.withheld.observations) == 40
