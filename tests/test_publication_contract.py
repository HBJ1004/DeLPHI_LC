"""Tests for the frozen V1 publication-run boundary."""

from __future__ import annotations

import json

import pytest

from lc_pipeline.publication.artifacts import (
    ArtifactReference,
    PhaseJob,
    PublicationPhaseManifest,
    atomic_write_json,
    read_phase_manifest,
    verify_artifacts,
)
from lc_pipeline.publication.contract import (
    PUBLICATION_SPEC_SHA256,
    PublicationContractError,
    load_publication_spec,
    repository_root,
    sha256_file,
    validate_publication_inputs,
)
from lc_pipeline.publication.results import manuscript_macros, summarize_oof_rows
from lc_pipeline.publication.runner import OOF_ARMS, OOF_SEEDS, build_oof_plan, create_run_manifest


def test_frozen_spec_binds_the_canonical_publication_inputs() -> None:
    spec_path = repository_root() / "repro" / "v1_publication_spec.yaml"
    assert sha256_file(spec_path) == PUBLICATION_SPEC_SHA256
    spec = load_publication_spec()
    inputs = validate_publication_inputs()
    assert spec["scope"]["eligible_object_count"] == 170
    assert inputs.object_count == 170
    assert len(inputs.quarantined_object_ids) == 4
    assert len(inputs.fold_test_ids) == 5
    assert all(len(identifiers) == 34 for identifiers in inputs.fold_test_ids)


def test_oof_plan_is_complete_and_has_portable_atlas_references() -> None:
    inputs = validate_publication_inputs()
    root = repository_root()
    plan = build_oof_plan(
        inputs,
        output_root=root / "runs" / "pytest-v1-publication",
        dump_root=root.parent / "Colab Notebooks" / "damit-20250610T000301Z",
    )
    assert len(plan.jobs) == len(OOF_ARMS) * 5 * len(OOF_SEEDS)
    assert len({job.job_id for job in plan.jobs}) == len(plan.jobs)
    assert all(job.config.atlas_path and not job.config.atlas_path.startswith("/") for job in plan.jobs)
    assert all(job.config.fold == job.fold and job.config.training.seed == job.seed for job in plan.jobs)
    assert len(plan.plan_sha256) == 64


def test_publication_output_root_must_be_inside_repository() -> None:
    inputs = validate_publication_inputs()
    with pytest.raises(PublicationContractError, match="inside the repository"):
        build_oof_plan(
            inputs,
            output_root="/tmp/not-a-publication-run",
            dump_root=repository_root().parent / "Colab Notebooks" / "damit-20250610T000301Z",
        )


def test_phase_manifest_and_artifact_hashes_round_trip(tmp_path) -> None:
    artifact_path = tmp_path / "summary.json"
    atomic_write_json(artifact_path, {"result": "ok"}, overwrite=False)
    reference = ArtifactReference.from_file(
        artifact_path,
        root=tmp_path,
        artifact_id="summary",
        media_type="application/json",
        role="metrics",
    )
    phase = PublicationPhaseManifest(
        run_id="v1-publication-test",
        phase="oof",
        status="complete",
        jobs=(
            PhaseJob(
                "job-1",
                "complete",
                "0" * 64,
                artifacts=(reference,),
            ),
        ),
    )
    phase_path = tmp_path / "phase.json"
    phase_path.write_text(json.dumps(phase.as_mapping()), encoding="utf-8")
    restored = read_phase_manifest(phase_path)
    assert restored == phase
    verify_artifacts(restored.jobs[0].artifacts, tmp_path)
    with pytest.raises(PublicationContractError, match="refusing to overwrite"):
        atomic_write_json(artifact_path, {"result": "changed"}, overwrite=False)


def test_oof_summary_requires_five_seed_object_votes_and_renders_macros() -> None:
    rows = []
    for model_kind, offset in (
        ("v1_faithful", 0.0),
        ("v1_corrected", 1.0),
        ("amplitude_mlp", 2.0),
    ):
        for condition, penalty in (
            ("raw", 0.0),
            ("zero", 3.0),
            ("atlas", 4.0),
            ("exhaustive_deranged", 5.0),
        ):
            for object_id, base in (("asteroid_a", 10.0), ("asteroid_b", 20.0)):
                for seed in OOF_SEEDS:
                    rows.append(
                        {
                            "object_id": object_id,
                            "model_kind": model_kind,
                            "condition": condition,
                            "seed": seed,
                            "axis_oracle_at3_error_deg": base + offset + penalty,
                        }
                    )
    summary = summarize_oof_rows(rows)
    assert summary["metrics"]["v1_faithful.raw"]["mean_error_deg"] == 15.0
    assert "\\renewcommand{\\PubOracleMean}{15.00" in manuscript_macros(summary)


def test_existing_run_cannot_mix_implementation_commits(tmp_path) -> None:
    inputs = validate_publication_inputs()
    create_run_manifest(inputs, output_root=tmp_path, implementation_commit="a" * 40)
    with pytest.raises(PublicationContractError, match="another implementation commit"):
        create_run_manifest(inputs, output_root=tmp_path, implementation_commit="b" * 40)
