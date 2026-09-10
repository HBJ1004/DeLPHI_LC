"""Focused tests for external period, identity, and checkpoint bindings."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from lc_pipeline.k3.config import K3ScoreModelConfig
from lc_pipeline.k3.external_provenance import (
    EXTERNAL_MODEL_MANIFEST_SCHEMA,
    build_ztf_period_manifest,
)
from lc_pipeline.k3.publication_run import K3_OOF_SEEDS
from lc_pipeline.k3.ztf_external import (
    ZTFExternalError,
    _parse_horizons_target_identity,
    _validate_resolved_target,
)
from lc_pipeline.k3.ztf_prediction import ZTFPredictionError, _load_models, _mapping_hash


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_period_manifest_is_deterministic_and_contains_no_pole_fields(tmp_path: Path):
    objects = [
        {
            "object_id": f"asteroid_{number}",
            "status": "ready",
            "fold": number % 5,
        }
        for number in range(1, 170)
    ]
    normalized = tmp_path / "normalized.json"
    normalized.write_text(
        json.dumps(
            {"schema": "delphi.k3-fink-ingest-manifest.v1", "objects": objects}
        )
    )
    catalog = tmp_path / "catalog.jsonl"
    catalog.write_text(
        "\n".join(
            json.dumps(
                {
                    "object_id": f"asteroid_{number}",
                    "eligible": True,
                    "solutions": [
                        {
                            "model_id": 10_000 + number,
                            "period_hours": 2.0 + number / 100,
                            "source_sha256": f"{number:064x}",
                            "lambda_deg": 123,
                            "beta_deg": 45,
                            "vector": [1, 0, 0],
                        }
                    ],
                },
                sort_keys=True,
            )
            for number in range(1, 170)
        )
        + "\n"
    )
    spec = tmp_path / "spec.yaml"
    spec.write_text("frozen: true\n")
    output = tmp_path / "periods.json"

    result = build_ztf_period_manifest(
        normalized_manifest_path=normalized,
        catalog_path=catalog,
        study_spec_path=spec,
        output_path=output,
    )

    assert result["object_count"] == 169
    assert result["objects"][0]["period_provenance"]["externally_supplied"] is True
    serialized = output.read_text().lower()
    assert all(
        token not in serialized
        for token in ('"solutions"', '"lambda_deg"', '"beta_deg"', '"vector"', '"axis')
    )
    assert result["study_spec_sha256"] == _sha(spec)


def test_horizons_identity_is_authoritative_for_numeric_and_named_designations():
    payload = json.dumps(
        {
            "result": (
                "Target body name: 107 Camilla (A868 WA) {source: JPL#140}\n"
                "$$SOE\n$$EOE"
            )
        }
    ).encode()
    identity = _parse_horizons_target_identity(payload)

    assert identity == {
        "asteroid_number": 107,
        "primary_name": "Camilla",
        "target_descriptor": "Camilla (A868 WA)",
    }
    validated = _validate_resolved_target(
        {
            "object_id": "asteroid_107",
            "rows": [
                {"fink_ssnamenr": "107"},
                {"fink_ssnamenr": "Camilla"},
            ],
        },
        [identity, identity],
    )
    assert validated["fink_named_designations"] == ["Camilla"]
    with pytest.raises(ZTFExternalError, match="does not match Horizons"):
        _validate_resolved_target(
            {"object_id": "asteroid_107", "rows": [{"fink_ssnamenr": "NotCamilla"}]},
            [identity],
        )


class _FakeModel:
    def load_state_dict(self, _state):
        return None

    def to(self, _device):
        return self

    def eval(self):
        return self


def test_model_loader_requires_complete_manifest_and_selects_exact_fold_sets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    model_config = asdict(K3ScoreModelConfig())
    model_hash = _mapping_hash(model_config)
    analysis_commit = "a" * 40
    protocol_hash = "b" * 64
    run_provenance = {
        "implementation_commit": analysis_commit,
        "protocol_sha256": protocol_hash,
        "pretrained_checkpoint_sha256": "c" * 64,
    }
    models = tmp_path / "models"
    models.mkdir()
    records = []
    checkpoint_by_name = {}
    for fold in range(5):
        for seed in K3_OOF_SEEDS:
            filename = f"real-fold-{fold}-seed-{seed}.pt"
            path = models / filename
            path.write_bytes(b"x")
            record = {
                "filename": filename,
                "logical_path": f"models/{filename}",
                "fold": fold,
                "seed": seed,
                "sha256": _sha(path),
                "size_bytes": 1,
                "checkpoint_schema": "delphi.k3-checkpoint.v2",
                "stage": "real-oof",
                "configuration_sha256": "d" * 64,
                "run_provenance": run_provenance,
            }
            records.append(record)
            checkpoint_by_name[filename] = {
                "schema": "delphi.k3-checkpoint.v2",
                "stage": "real-oof",
                "config": {"seed": seed},
                "model_config": model_config,
                "model_state_dict": {},
                "configuration_sha256": "d" * 64,
                "run_provenance": run_provenance,
            }
    manifest = tmp_path / "models.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": EXTERNAL_MODEL_MANIFEST_SCHEMA,
                "publication_package_manifest_sha256": "1" * 64,
                "artifact_archive_sha256": "2" * 64,
                "archive_index_sha256": "3" * 64,
                "checkpoint_audit_sha256": "4" * 64,
                "analysis_commit": analysis_commit,
                "protocol_sha256": protocol_hash,
                "model_config_sha256": model_hash,
                "folds": list(range(5)),
                "seeds": list(K3_OOF_SEEDS),
                "model_count": 25,
                "models": records,
            }
        )
    )
    monkeypatch.setattr(
        "lc_pipeline.k3.ztf_prediction.torch.load",
        lambda path, **_kwargs: checkpoint_by_name[Path(path).name],
    )
    monkeypatch.setattr(
        "lc_pipeline.k3.ztf_prediction.CandidateConditionedScorer",
        lambda _config: _FakeModel(),
    )

    selected, _, bindings = _load_models(models, manifest, (2,), "cpu")
    assert len(selected) == 5
    assert {row["fold"] for row in bindings} == {2}
    all_models, _, all_bindings = _load_models(models, manifest, tuple(range(5)), "cpu")
    assert len(all_models) == 25
    assert {(row["fold"], row["seed"]) for row in all_bindings} == {
        (fold, seed) for fold in range(5) for seed in K3_OOF_SEEDS
    }

    broken = json.loads(manifest.read_text())
    broken["models"].pop()
    manifest.write_text(json.dumps(broken))
    with pytest.raises(ZTFPredictionError, match="complete 25-model set"):
        _load_models(models, manifest, (2,), "cpu")
