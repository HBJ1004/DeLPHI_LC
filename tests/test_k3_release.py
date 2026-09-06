"""Integration tests for the fail-closed K3 publication summary."""

from __future__ import annotations

import hashlib
import json

import numpy as np

from lc_pipeline.k3.protocol import K3_PROTOCOL_SHA256
from lc_pipeline.k3.publication_run import K3_DIAGNOSTIC_CONTROLS, K3_OOF_SEEDS
from lc_pipeline.k3.release import build_publication_release


def _hash(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_release_builder_applies_all_gates_and_writes_macros(tmp_path) -> None:
    root = tmp_path / "artifacts"
    evaluations = root / "evaluations"
    evaluations.mkdir(parents=True)
    (root / "audits").mkdir()
    downstream_directory = root / "downstream" / "fixed-period"
    downstream_directory.mkdir(parents=True)
    object_ids = np.asarray([f"object-{index:03d}" for index in range(170)])
    synthetic_ids = np.asarray([f"synthetic-{index:04d}" for index in range(2000)])

    (root / "audits" / "checkpoint-audit.json").write_text(
        json.dumps(
            {
                "schema": "delphi.k3-checkpoint-audit.v1",
                "passed": True,
                "protocol_sha256": K3_PROTOCOL_SHA256,
                "implementation_commit": "a" * 40,
            }
        ),
        encoding="utf-8",
    )
    downstream = {
        "schema": "delphi.k3-fixed-period-summary.v1",
        "verdict": {
            "passed": True,
            "paired_successful_objects": 170,
            "speed_ratio": 2.0,
            "speed_ratio_ci95_low": 1.8,
            "recovery_rate_inferiority": 0.0,
            "geometric_mean_rms_ratio": 1.0,
            "geometric_mean_rms_ratio_ci95_high": 1.001,
            "failures": [],
        },
    }
    (downstream_directory / "fixed-period-summary.json").write_text(
        json.dumps(downstream), encoding="utf-8"
    )
    np.savez_compressed(
        evaluations / "real-oof-ensemble.npz",
        schema=np.asarray("delphi.k3-real-oof-ensemble.v1"),
        object_ids=object_ids,
        oracle_errors_deg=np.full(170, 12.0),
        seeds=np.asarray(K3_OOF_SEEDS),
    )
    comparator = tmp_path / "v1-comparators.npz"
    np.savez_compressed(
        comparator,
        schema=np.asarray("delphi.k3-v1-comparators.v1"),
        object_ids=object_ids,
        v1_errors_deg=np.full(170, 20.0),
        atlas_errors_deg=np.full(170, 21.0),
        deranged_errors_deg=np.full(170, 32.0),
    )
    for seed in K3_OOF_SEEDS:
        raw_path = evaluations / f"real-oof-seed-{seed}.npz"
        np.savez_compressed(
            raw_path,
            object_ids=object_ids,
            oracle_errors_deg=np.full(170, 12.0),
        )
        np.savez_compressed(
            evaluations / f"real-oof-seed-{seed}-input-swap.npz",
            schema=np.asarray("delphi.k3-real-oof-input-swap.v1"),
            object_ids=object_ids,
            oracle_errors_deg=np.full(170, 30.0),
            raw_artifact_sha256=np.asarray(_hash(raw_path)),
        )
        for control in K3_DIAGNOSTIC_CONTROLS:
            np.savez_compressed(
                evaluations / f"real-oof-seed-{seed}-{control}.npz",
                schema=np.asarray("delphi.k3-real-oof-diagnostic.v1"),
                object_ids=object_ids,
                control=np.asarray(control),
                degradation_deg=np.full(170, 5.0),
            )
        synthetic_raw = evaluations / f"synthetic-test-seed-{seed}.npz"
        np.savez_compressed(
            synthetic_raw,
            object_ids=synthetic_ids,
            oracle_errors_deg=np.full(2000, 12.0),
        )
        np.savez_compressed(
            evaluations / f"synthetic-test-seed-{seed}-input-swap.npz",
            object_ids=synthetic_ids,
            oracle_errors_deg=np.full(2000, 30.0),
            raw_artifact_sha256=np.full(2000, _hash(synthetic_raw)),
        )
    np.savez_compressed(
        evaluations / "synthetic-label-shuffle-test-seed-17.npz",
        object_ids=synthetic_ids,
        oracle_errors_deg=np.full(2000, 25.0),
    )
    np.savez_compressed(
        evaluations / "real-oof-ensemble-calibrated.npz",
        schema=np.asarray("delphi.k3-real-oof-ensemble-calibration.v1"),
        object_ids=object_ids,
        fold_cone90_deg=np.full(5, 15.0),
        fold_cone95_deg=np.full(5, 90.0),
        covered90=np.ones(170, dtype=bool),
        covered95=np.ones(170, dtype=bool),
    )

    output_json = tmp_path / "publication-summary.json"
    output_tex = tmp_path / "k3-results.tex"
    result = build_publication_release(
        artifact_root=root,
        comparator_path=comparator,
        output_json=output_json,
        output_tex=output_tex,
    )
    assert result["promotion_passed"] is True
    assert result["metrics"]["oof"]["mean_error_deg"] == 12.0
    macros = output_tex.read_text(encoding="utf-8")
    assert "\\KThreeOofMean}{12.00\\ensuremath{^\\circ}}" in macros
    assert "\\KThreePromotionVerdict}{\\textsc{promote K3}}" in macros
