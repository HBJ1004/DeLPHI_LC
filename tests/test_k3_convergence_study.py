"""Focused contracts for the frozen, phase-separated convergence study."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from lc_pipeline.k3.convergence_study import (
    SPEC_SCHEMA,
    SUBSET_SCHEMA,
    _expected_cohorts,
    _paired_binary_noninferiority,
    _paired_rms_ratio,
    _read_spec,
    _stratified_ratio_bootstrap,
    _study_section,
    _validate_subset_manifest,
    execute_blind_cohort,
)
from lc_pipeline.k3.downstream import DownstreamBenchmarkError
from lc_pipeline.v2.data import sha256_file


def _frozen_spec() -> dict[str, object]:
    return {
        "schema": SPEC_SCHEMA,
        "convergence_acceleration": {
            "frozen_inputs": {},
            "split": {
                "method": "six_development_objects_per_original_outer_fold_by_sha256_rank",
                "salt": "synthetic-convergence-test-salt",
                "development_count": 30,
                "locked_evaluation_count": 140,
            },
            "shared_conditions": {
                "convergence_tolerance_grid": [0.01, 0.003, 0.001, 0.0003, 0.0001],
                "iteration_cap": 1000,
                "timeout_seconds_per_start": 300,
                "timing_repeats": 3,
                "repeat_order_seed": 20260910,
                "bootstrap_resamples": 10000,
                "bootstrap_seed": 20260911,
            },
            "analysis": {
                "binary_repeat_aggregation": (
                    "object_success_requires_at_least_two_of_three_repeats"
                ),
                "rms_required_object_support": {
                    "development": 30,
                    "locked_evaluation": 140,
                },
            },
        },
    }


def _write_spec(path: Path) -> tuple[Path, Path, str]:
    path.write_text(yaml.safe_dump(_frozen_spec(), sort_keys=True), encoding="utf-8")
    digest = sha256_file(path)
    checksum = path.with_suffix(".sha256")
    checksum.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    return path, checksum, digest


def _write_split(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "folds": [
                    {
                        "fold": fold,
                        "test_ids": [
                            f"synthetic-{fold}-{index:02d}" for index in range(34)
                        ],
                    }
                    for fold in range(5)
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


def test_paired_binary_bound_is_exact_simultaneous_and_object_level():
    baseline = np.asarray([0] * 3 + [1] + [1] * 26, dtype=np.int8)
    guided = np.asarray([1] * 3 + [0] + [1] * 26, dtype=np.int8)

    result = _paired_binary_noninferiority(baseline, guided)

    assert result["object_count"] == 30
    assert result["favorable_discordances"] == 3
    assert result["adverse_discordances"] == 1
    assert result["point_difference"] == pytest.approx(2 / 30)
    # One-sided 97.5% CP lower for 3/30 minus one-sided 97.5% CP upper for 1/30.
    assert result["simultaneous_exact_lower"] == pytest.approx(
        -0.15105231860369012, abs=1e-14
    )
    assert result["tail_probability"] == pytest.approx(0.025)

    # With no discordances, the conservative lower bound is still negative.
    # This is the boundary that makes a two-percentage-point margin infeasible
    # even for 140 objects, and motivates the frozen five-point margin.
    tied = _paired_binary_noninferiority(np.ones(140), np.ones(140))
    assert tied["point_difference"] == 0.0
    assert tied["simultaneous_exact_lower"] == pytest.approx(
        -0.026005029351292668, abs=1e-14
    )
    assert tied["simultaneous_exact_lower"] >= -0.05
    assert tied["simultaneous_exact_lower"] < -0.02


def test_stratified_ratio_bootstrap_is_deterministic_and_preserves_fold_counts():
    # Every object within a fold is identical, but the folds have different
    # timing ratios. A correctly stratified bootstrap must therefore reproduce
    # the observed aggregate ratio in every resample. An unstratified bootstrap
    # would vary the relative number of objects drawn from the two folds.
    baseline = np.asarray([[1.0, 1.0]] * 2 + [[9.0, 9.0]] * 3)
    guided = np.asarray([[1.0, 1.0]] * 2 + [[3.0, 3.0]] * 3)
    folds = np.asarray([0, 0, 1, 1, 1])

    first = _stratified_ratio_bootstrap(
        baseline, guided, folds, resamples=257, seed=20260911
    )
    second = _stratified_ratio_bootstrap(
        baseline, guided, folds, resamples=257, seed=20260911
    )

    expected = 58.0 / 22.0
    assert first == second
    assert first["point_ratio"] == pytest.approx(expected)
    assert first["acceptance_lower"] == pytest.approx(expected)
    assert first["acceptance_upper"] == pytest.approx(expected)
    assert first["resamples"] == 257
    assert first["seed"] == 20260911
    assert first["stratum_sizes"] == {"0": 2, "1": 3}


@pytest.mark.parametrize("required_support", [30, 140])
def test_paired_rms_ratio_requires_every_predeclared_object(required_support: int):
    baseline = np.ones((required_support, 3), dtype=np.float64)
    guided = np.full((required_support, 3), 1.005, dtype=np.float64)
    jointly_selectable = np.ones((required_support, 3), dtype=bool)

    result = _paired_rms_ratio(
        baseline,
        guided,
        jointly_selectable,
        required_support=required_support,
    )
    assert result["object_support"] == required_support
    assert result["minimum_joint_repeats_per_object"] == 2
    assert result["point_ratio"] == pytest.approx(1.005)

    # A single object with only one jointly completed/selectable repeat makes
    # support n-1. The implementation must fail closed rather than silently
    # changing the RMS population or reference-conditioning the sample.
    jointly_selectable[-1] = [True, False, False]
    with pytest.raises(DownstreamBenchmarkError, match=rf"support.*{required_support}"):
        _paired_rms_ratio(
            baseline,
            guided,
            jointly_selectable,
            required_support=required_support,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong-spec-hash",
        "wrong-role",
        "wrong-object-order",
        "extra-field",
    ],
)
def test_subset_manifest_rejects_every_nonexact_cohort_or_spec_binding(
    tmp_path: Path, mutation: str
):
    spec_path, checksum_path, spec_hash = _write_spec(tmp_path / "study.yaml")
    spec, verified_hash = _read_spec(spec_path, checksum_path)
    study = _study_section(spec)
    split_path = _write_split(tmp_path / "split.json")
    split_hash = sha256_file(split_path)
    expected = _expected_cohorts(split_path, study)["development"]
    manifest: dict[str, object] = {
        "schema": SUBSET_SCHEMA,
        "role": "development",
        "study_spec_sha256": spec_hash,
        "source_full_split_sha256": split_hash,
        "selection": study["split"]["method"],
        "salt": study["split"]["salt"],
        "object_ids": list(expected),
    }
    if mutation == "wrong-spec-hash":
        manifest["study_spec_sha256"] = "0" * 64
    elif mutation == "wrong-role":
        manifest["role"] = "locked_evaluation"
    elif mutation == "wrong-object-order":
        manifest["object_ids"] = list(reversed(expected))
    else:
        manifest["unfrozen_note"] = "must not be accepted"
    manifest_path = tmp_path / "development.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert verified_hash == spec_hash
    with pytest.raises(DownstreamBenchmarkError, match="exact frozen cohort"):
        _validate_subset_manifest(
            manifest_path,
            role="development",
            expected_ids=expected,
            spec_sha256=spec_hash,
            split_sha256=split_hash,
            study=study,
        )


def test_spec_checksum_detects_post_lock_edit(tmp_path: Path):
    spec_path, checksum_path, _ = _write_spec(tmp_path / "study.yaml")
    spec_path.write_text(spec_path.read_text() + "post_lock_edit: true\n", encoding="utf-8")

    with pytest.raises(DownstreamBenchmarkError, match="frozen checksum"):
        _read_spec(spec_path, checksum_path)


def _locked_call_kwargs(tmp_path: Path) -> dict[str, object]:
    """Supply inert values for the evolving keyword-only orchestration API."""
    kwargs: dict[str, object] = {
        "role": "locked_evaluation",
        "development_selection_path": None,
    }
    for name, parameter in inspect.signature(execute_blind_cohort).parameters.items():
        if name in kwargs or parameter.default is not inspect.Parameter.empty:
            continue
        if "tolerance" in name:
            kwargs[name] = 0.001
        elif name in {"resume", "overwrite"}:
            kwargs[name] = False
        else:
            kwargs[name] = tmp_path / name.replace("_path", "").replace("_directory", "")
    return kwargs


def test_locked_execution_refuses_missing_development_selection_before_any_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    def forbidden_run(**_kwargs: object) -> object:
        raise AssertionError("solver execution must not begin before development selection")

    monkeypatch.setattr(
        "lc_pipeline.k3.convergence_study._run_convergence_record", forbidden_run
    )

    with pytest.raises(DownstreamBenchmarkError, match="development selection"):
        execute_blind_cohort(**_locked_call_kwargs(tmp_path))
