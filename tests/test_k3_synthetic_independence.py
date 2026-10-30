"""Independent record streams and a strict training boundary for legacy shards."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch
from scipy.stats import kstest, spearmanr

from lc_pipeline.k3.publication_run import (
    load_synthetic_examples,
    load_synthetic_subset,
    run_synthetic_label_shuffle_control,
    run_synthetic_pretraining,
)
from lc_pipeline.k3.renderer import ConvexFacets
from lc_pipeline.k3.synthetic import (
    LEGACY_SYNTHETIC_SHARD_SCHEMA,
    SYNTHETIC_RECIPE_SCHEMA,
    SYNTHETIC_SHARD_SCHEMA,
    EmpiricalNoiseRecipe,
    ResidualNoiseBank,
    SyntheticDataError,
    SyntheticRecipe,
    SyntheticRecord,
    canonicalize_axes,
    generate_synthetic_record,
    load_synthetic_shard,
    sample_synthetic_draws,
    verify_synthetic_shard,
    write_synthetic_shard,
)
from lc_pipeline.v2.preprocessing import Observation, ObservationEpoch


@pytest.fixture
def recipe():
    return SyntheticRecipe(
        split="smoke", object_count=1, master_seed=20260911,
        noise=EmpiricalNoiseRecipe("unit-test-train-only"),
    )


@pytest.fixture(scope="module")
def population():
    recipe = SyntheticRecipe(
        split="smoke", object_count=10000, master_seed=20260911,
        noise=EmpiricalNoiseRecipe("unit-test-train-only"),
    )
    # The production record helper is exercised with the unchanged seed rule;
    # none of these 10,000 objects renders a lightcurve.
    draws = [sample_synthetic_draws(recipe, seed=recipe.master_seed + i) for i in range(recipe.object_count)]
    return (
        np.asarray([draw.directed_axis for draw in draws]),
        np.log([draw.period_hours for draw in draws]),
        np.asarray([draw.initial_phase for draw in draws]),
    )


def _circular_correlation(first, second):
    """Jammalamadaka-Sarma correlation, with each circular mean removed."""
    first_center = np.angle(np.mean(np.exp(1j * first)))
    second_center = np.angle(np.mean(np.exp(1j * second)))
    first_sine = np.sin(first - first_center)
    second_sine = np.sin(second - second_center)
    return float(np.dot(first_sine, second_sine) / np.sqrt(
        np.dot(first_sine, first_sine) * np.dot(second_sine, second_sine)
    ))


def test_axis_and_normalized_log_period_are_independent(population):
    axes, log_period, _phase = population
    q = 2 * (log_period - math.log(2)) / math.log(100) - 1
    assert abs(spearmanr(np.abs(axes[:, 2]), np.abs(q)).statistic) < 0.05


def test_longitude_and_initial_phase_are_independent(population):
    axes, _log_period, phase = population
    # Check the directed pole to catch the old exact equality, then the actual
    # canonical pole (also modulo pi, so antipodal flips cannot hide coupling).
    for values in (axes, canonicalize_axes(axes)):
        longitude = np.arctan2(values[:, 1], values[:, 0])
        assert abs(_circular_correlation(longitude, phase)) < 0.05
        assert abs(_circular_correlation(2 * longitude, 2 * phase)) < 0.05


def test_axis_and_period_marginals_remain_uniform(population):
    axes, log_period, _phase = population
    np.testing.assert_allclose(np.linalg.norm(axes, axis=1), 1, atol=1e-14)
    assert kstest(axes[:, 2], "uniform", args=(-1, 2)).pvalue > 0.01
    assert kstest(np.abs(axes[:, 2]), "uniform").pvalue > 0.01
    assert kstest(log_period, "uniform", args=(math.log(2), math.log(100))).pvalue > 0.01
    longitude_fraction = np.mod(np.arctan2(axes[:, 1], axes[:, 0]), 2 * math.pi) / (2 * math.pi)
    assert kstest(longitude_fraction, "uniform").pvalue > 0.01


def test_fixed_seed_draws_are_reproducible(recipe):
    first = sample_synthetic_draws(recipe, seed=recipe.master_seed)
    assert first == sample_synthetic_draws(recipe, seed=recipe.master_seed)
    assert first != sample_synthetic_draws(recipe, seed=recipe.master_seed + 1)


@pytest.fixture
def epochs():
    return (ObservationEpoch("donor-epoch", tuple(
        Observation(
            time_jd=2450000 + i / 24, relative_brightness=1, measured_error=0.01,
            sun_asteroid_ecliptic_j2000_au=(1, 0, 0.1),
            observer_asteroid_ecliptic_j2000_au=(1, 0.1, 0),
        ) for i in range(4)
    )),)


@pytest.fixture
def record_inputs(recipe, epochs):
    return dict(
        object_id="synthetic-test", geometry_donor_id="geometry-train",
        geometry_epochs=epochs,
        shape=ConvexFacets(np.concatenate((np.eye(3), -np.eye(3))), np.ones(6), "shape-train"),
        recipe=recipe, seed=recipe.master_seed,
    )


def test_fixed_seed_record_including_noise_is_reproducible(record_inputs):
    first = generate_synthetic_record(**record_inputs)
    assert first == generate_synthetic_record(**record_inputs)
    assert first.seed == record_inputs["seed"]
    draws = sample_synthetic_draws(record_inputs["recipe"], seed=first.seed)
    np.testing.assert_allclose(first.target_axis, canonicalize_axes(np.asarray(draws.directed_axis))[0])
    assert first.period_hours == draws.period_hours


@pytest.mark.parametrize("residual_bank", [False, True])
def test_overrides_preserve_phase_and_noise_streams(record_inputs, monkeypatch, residual_bank):
    rendered = []

    def constant_brightness(_normals, _areas, axes, phases, *_geometry, **_kwargs):
        rendered.append((axes.clone(), phases.clone()))
        return torch.ones_like(phases)

    monkeypatch.setattr("lc_pipeline.k3.synthetic.render_brightness", constant_brightness)
    if residual_bank:
        record_inputs["residual_noise_bank"] = ResidualNoiseBank(
            (np.asarray([-0.01, 0, 0.01]), np.asarray([-0.02, 0, 0.02])),
            np.asarray([0.01, 0.02]), "unit-test-train-only", "a" * 64,
        )
    natural = generate_synthetic_record(**record_inputs)
    overridden = generate_synthetic_record(
        **record_inputs, target_axis=(0, -1, -1), period_hours=8,
    )
    assert overridden.period_hours == 8
    np.testing.assert_allclose(overridden.target_axis, (0, 2 ** -0.5, 2 ** -0.5))
    assert natural.epochs == overridden.epochs  # Noise cannot move when overrides skip draws.
    assert rendered[0][1][0, 0] == rendered[1][1][0, 0]  # Same initial phase.
    assert not torch.equal(rendered[0][1], rendered[1][1])  # Supplied period controls folding.
    assert not torch.equal(rendered[0][0], rendered[1][0])


@pytest.fixture
def shard(tmp_path, recipe, epochs):
    path = tmp_path / "split" / "shard-0000.npz"
    record = SyntheticRecord(
        object_id="synthetic-test", epochs=epochs, target_axis=(0, 0, 1),
        period_hours=8, shape_donor_id="shape-train", geometry_donor_id="geometry-train",
        seed=recipe.master_seed,
    )
    manifest = write_synthetic_shard(path, [record], recipe=recipe, shard_index=0)
    assert manifest["schema"] == SYNTHETIC_SHARD_SCHEMA
    assert manifest["recipe_schema"] == SYNTHETIC_RECIPE_SCHEMA
    return path


def _make_legacy(shard):
    manifest_path = shard.with_suffix(".npz.manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["schema"] = LEGACY_SYNTHETIC_SHARD_SCHEMA
    manifest.pop("recipe_schema")
    manifest_path.write_text(json.dumps(manifest))


@pytest.mark.parametrize("loader", ["complete", "subset"])
def test_training_loaders_refuse_old_version_shards(shard, loader):
    _make_legacy(shard)
    with pytest.raises(SyntheticDataError, match="training requires delphi.k3-synthetic-shard.v2"):
        if loader == "complete":
            load_synthetic_examples(shard.parent)
        else:
            load_synthetic_subset(shard.parent, count=1, seed=17)


@pytest.mark.parametrize("run", [run_synthetic_pretraining, run_synthetic_label_shuffle_control])
def test_training_entry_points_refuse_old_version_before_fitting(shard, tmp_path, monkeypatch, run):
    _make_legacy(shard)

    def forbidden_fit(*_args, **_kwargs):
        pytest.fail("training must reject the old shard before fitting")

    monkeypatch.setattr("lc_pipeline.k3.publication_run.fit_k3", forbidden_fit)
    with pytest.raises(SyntheticDataError, match="coupled random streams"):
        run(shard.parent, shard.parent, tmp_path / "output", seed=17, device="cpu")


def test_current_shards_are_accepted_by_both_training_loaders(shard):
    assert len(load_synthetic_examples(shard.parent)) == 1
    assert len(load_synthetic_subset(shard.parent, count=1, seed=17)) == 1


def test_legacy_analysis_reads_still_verify_the_hash(shard):
    _make_legacy(shard)
    with pytest.raises(SyntheticDataError, match="legacy synthetic shard"):
        load_synthetic_shard(shard)
    assert len(load_synthetic_shard(shard, allow_legacy=True)) == 1
    assert verify_synthetic_shard(shard, allow_legacy=True)["objects"] == 1
    with shard.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(SyntheticDataError, match="hash mismatch"):
        load_synthetic_shard(shard, allow_legacy=True)


def test_current_shard_cannot_claim_a_legacy_recipe(shard):
    manifest_path = shard.with_suffix(".npz.manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["recipe_schema"] = "delphi.k3-synthetic-recipe.v1"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(SyntheticDataError, match="requires recipe schema"):
        load_synthetic_examples(shard.parent)
