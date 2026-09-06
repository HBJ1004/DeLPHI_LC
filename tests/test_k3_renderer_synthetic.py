"""Numerical and provenance tests for K3 simulation."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from lc_pipeline.k3.damit_synthetic import partition_damit_donors
from lc_pipeline.k3.renderer import (
    ConvexFacets,
    normalize_by_epoch,
    render_brightness,
    sample_convex_ellipsoids,
    validate_random_agreement,
)
from lc_pipeline.k3.synthetic import (
    EmpiricalNoiseRecipe,
    ResidualNoiseBank,
    SyntheticDataError,
    SyntheticRecipe,
    generate_synthetic_record,
    load_synthetic_shard,
    sample_uniform_axial_axes,
    verify_synthetic_shard,
    write_synthetic_shard,
)
from lc_pipeline.v2.preprocessing import Observation, ObservationEpoch


def _epochs() -> tuple[ObservationEpoch, ...]:
    epochs = []
    for epoch_index, start in enumerate((2450000.0, 2450100.0)):
        observations = []
        for index in range(7):
            angle = 0.1 * (index + epoch_index)
            observations.append(
                Observation(
                    time_jd=start + index / 24.0,
                    relative_brightness=1.0,
                    sun_asteroid_ecliptic_j2000_au=(math.cos(angle), math.sin(angle), 0.2),
                    observer_asteroid_ecliptic_j2000_au=(math.cos(angle + 0.3), math.sin(angle + 0.3), -0.1),
                )
            )
        epochs.append(ObservationEpoch(f"epoch-{epoch_index}", tuple(observations)))
    return tuple(epochs)


def _shape() -> ConvexFacets:
    generator = torch.Generator().manual_seed(9)
    normals, areas, _ = sample_convex_ellipsoids(
        1, generator=generator, facet_count=64, dtype=torch.float64
    )
    return ConvexFacets(normals[0].numpy(), areas[0].numpy(), "shape-train-1")


def test_torch_renderer_matches_independent_numpy_and_has_pole_gradient() -> None:
    agreement = validate_random_agreement(cases=3, tolerance=1e-10)
    assert agreement.passed
    generator = torch.Generator().manual_seed(12)
    normals, areas, _ = sample_convex_ellipsoids(
        1, generator=generator, facet_count=64, dtype=torch.float64
    )
    pole = torch.tensor([[0.4, -0.3, 0.8]], dtype=torch.float64, requires_grad=True)
    phases = torch.linspace(0, 4 * math.pi, 20, dtype=torch.float64)[None, :]
    sun = torch.randn(1, 20, 3, generator=generator, dtype=torch.float64)
    observer = torch.randn(1, 20, 3, generator=generator, dtype=torch.float64)
    brightness = render_brightness(normals, areas, pole, phases, sun, observer)
    brightness.square().mean().backward()
    assert pole.grad is not None and torch.isfinite(pole.grad).all()
    assert float(torch.linalg.vector_norm(pole.grad)) > 0


def test_epoch_normalization_is_mask_aware() -> None:
    values = torch.tensor([[2.0, 4.0, 9.0, 3.0, 99.0]])
    indices = torch.tensor([[0, 0, 1, 1, 7]])
    mask = torch.tensor([[True, True, True, True, False]])
    normalized = normalize_by_epoch(values, indices, mask)
    assert torch.allclose(normalized[0, :2].mean(), torch.tensor(1.0))
    assert torch.allclose(normalized[0, 2:4].mean(), torch.tensor(1.0))
    assert normalized[0, 4] == 0


def test_uniform_axes_are_unit_and_canonical() -> None:
    axes = sample_uniform_axial_axes(2000, seed=44)
    assert np.allclose(np.linalg.norm(axes, axis=1), 1.0)
    assert np.all(axes[:, 2] >= -1e-14)
    # Uniform spherical z becomes uniform |z| after quotienting.
    assert abs(float(np.mean(axes[:, 2])) - 0.5) < 0.03


def test_damit_donor_partitions_are_deterministic_and_disjoint() -> None:
    donors = [f"asteroid_{index}" for index in range(100)]
    first = partition_damit_donors(donors, master_seed=20260901)
    second = partition_damit_donors(reversed(donors), master_seed=20260901)
    assert first == second
    assert tuple(map(len, first.values())) == (80, 10, 10)
    assert not (set(first["train"]) & set(first["validation"]))
    assert not (set(first["train"]) & set(first["test"]))
    assert not (set(first["validation"]) & set(first["test"]))


def test_synthetic_round_trip_preserves_every_epoch_and_hash(tmp_path) -> None:
    recipe = SyntheticRecipe(
        split="smoke",
        object_count=1,
        master_seed=17,
        noise=EmpiricalNoiseRecipe("unit-test-train-only"),
    )
    record = generate_synthetic_record(
        object_id="synthetic-0001",
        geometry_donor_id="geometry-train-1",
        geometry_epochs=_epochs(),
        shape=_shape(),
        recipe=recipe,
        seed=118,
        period_hours=8.0,
    )
    assert [len(epoch.observations) for epoch in record.epochs] == [7, 7]
    assert all(observation.relative_brightness > 0 for epoch in record.epochs for observation in epoch.observations)
    shard = tmp_path / "shard-0000.npz"
    manifest = write_synthetic_shard(shard, [record], recipe=recipe, shard_index=0)
    restored = load_synthetic_shard(shard)
    selected = load_synthetic_shard(shard, selected_object_ids={"synthetic-0001"})
    verified = verify_synthetic_shard(shard)
    assert manifest["object_count"] == 1 and restored[0].object_id == record.object_id
    assert len(selected) == 1
    assert verified["objects"] == 1 and verified["observations"] == 14
    assert [epoch.epoch_id for epoch in restored[0].epochs] == ["epoch-0", "epoch-1"]
    assert np.allclose(restored[0].target_axis, record.target_axis)
    with pytest.raises(SyntheticDataError, match="overwrite"):
        write_synthetic_shard(shard, [record], recipe=recipe, shard_index=0)
    with shard.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(SyntheticDataError, match="hash"):
        load_synthetic_shard(shard)


def test_publication_generation_requires_real_residual_bank(tmp_path) -> None:
    recipe = SyntheticRecipe(
        split="validation",
        object_count=2000,
        master_seed=18,
        noise=EmpiricalNoiseRecipe("train-only-residual-bank"),
    )
    with pytest.raises(SyntheticDataError, match="residual noise bank"):
        generate_synthetic_record(
            object_id="synthetic-validation-000000",
            geometry_donor_id="geometry-train-1",
            geometry_epochs=_epochs(),
            shape=_shape(),
            recipe=recipe,
            seed=18,
            period_hours=8.0,
        )
    bank_path = tmp_path / "bank.npz"
    np.savez_compressed(
        bank_path,
        n=np.asarray(2),
        mad=np.asarray([0.01, 0.02]),
        r_0=np.asarray([-0.01, 0.0, 0.01, 0.0]),
        r_1=np.asarray([-0.02, 0.02, -0.01, 0.01]),
    )
    bank = ResidualNoiseBank.load(bank_path, provenance="train-only-test-bank")
    record = generate_synthetic_record(
        object_id="synthetic-validation-000000",
        geometry_donor_id="geometry-train-1",
        geometry_epochs=_epochs(),
        shape=_shape(),
        recipe=recipe,
        seed=18,
        period_hours=8.0,
        residual_noise_bank=bank,
    )
    assert len(record.epochs) == 2
