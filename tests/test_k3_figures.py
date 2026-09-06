"""Tests for provenance-bound K3 publication figures."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from lc_pipeline.k3.figures import K3FigureError, build_publication_figures
from lc_pipeline.k3.publication_run import K3_OOF_SEEDS


def _hash(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_publication_figures_are_bound_to_release_artifacts(tmp_path) -> None:
    root = tmp_path / "artifacts"
    evaluations = root / "evaluations"
    downstream = root / "downstream" / "fixed-period"
    evaluations.mkdir(parents=True)
    downstream.mkdir(parents=True)
    object_ids = np.asarray([f"object-{index:03d}" for index in range(170)])
    errors = np.linspace(5.0, 40.0, 170)
    ensemble = evaluations / "real-oof-ensemble.npz"
    np.savez_compressed(
        ensemble,
        schema=np.asarray("delphi.k3-real-oof-ensemble.v1"),
        object_ids=object_ids,
        oracle_errors_deg=errors,
    )
    comparator = tmp_path / "comparators.npz"
    np.savez_compressed(
        comparator,
        schema=np.asarray("delphi.k3-v1-comparators.v1"),
        object_ids=object_ids,
        v1_errors_deg=errors + 5,
        atlas_errors_deg=errors + 7,
        deranged_errors_deg=errors + 15,
    )
    for seed_index, seed in enumerate(K3_OOF_SEEDS):
        np.savez_compressed(
            evaluations / f"real-oof-seed-{seed}.npz",
            object_ids=object_ids,
            oracle_errors_deg=errors + seed_index,
        )
    rows_path = downstream / "fixed-period-rows.json"
    rows_path.write_text(
        json.dumps(
            {
                "schema": "delphi.k3-fixed-period-rows.v1",
                "object_ids": object_ids.tolist(),
                "rows": [
                    {
                        "baseline": {
                            "wall_seconds": 2.0 + index / 100,
                            "final_rms": 0.02 + index / 100_000,
                            "success": True,
                        },
                        "candidate": {
                            "wall_seconds": 1.0 + index / 200,
                            "final_rms": 0.0201 + index / 100_000,
                            "success": True,
                        },
                    }
                    for index in range(170)
                ],
            }
        ),
        encoding="utf-8",
    )
    downstream_summary = downstream / "fixed-period-summary.json"
    downstream_summary.write_text(
        json.dumps(
            {
                "schema": "delphi.k3-fixed-period-summary.v1",
                "rows_sha256": _hash(rows_path),
            }
        ),
        encoding="utf-8",
    )
    release = tmp_path / "publication-summary.json"
    release.write_text(
        json.dumps(
            {
                "schema": "delphi.k3-publication-summary.v1",
                "source_sha256": {
                    "oof_ensemble": _hash(ensemble),
                    "v1_comparator": _hash(comparator),
                    "downstream_summary": _hash(downstream_summary),
                },
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "figures"
    result = build_publication_figures(
        artifact_root=root,
        comparator_path=comparator,
        release_summary_path=release,
        output_directory=output,
    )
    assert set(result["figure_sha256"]) == {
        "k3_oracle_error_cdf.pdf",
        "k3_seed_stability.pdf",
        "k3_downstream_benchmark.pdf",
    }
    assert all((output / name).stat().st_size > 1_000 for name in result["figure_sha256"])

    with pytest.raises(K3FigureError, match="refusing to overwrite"):
        build_publication_figures(
            artifact_root=root,
            comparator_path=comparator,
            release_summary_path=release,
            output_directory=output,
        )


def test_publication_figures_allow_nan_rms_for_failed_fits(tmp_path) -> None:
    """Failed fits stay in timing/recovery but are excluded from paired RMS."""
    root = tmp_path / "artifacts"
    evaluations = root / "evaluations"
    downstream = root / "downstream" / "fixed-period"
    evaluations.mkdir(parents=True)
    downstream.mkdir(parents=True)
    object_ids = np.asarray([f"object-{index:03d}" for index in range(170)])
    errors = np.linspace(5.0, 40.0, 170)
    ensemble = evaluations / "real-oof-ensemble.npz"
    np.savez_compressed(
        ensemble,
        schema=np.asarray("delphi.k3-real-oof-ensemble.v1"),
        object_ids=object_ids,
        oracle_errors_deg=errors,
    )
    comparator = tmp_path / "comparators.npz"
    np.savez_compressed(
        comparator,
        schema=np.asarray("delphi.k3-v1-comparators.v1"),
        object_ids=object_ids,
        v1_errors_deg=errors + 5,
        atlas_errors_deg=errors + 7,
        deranged_errors_deg=errors + 15,
    )
    for seed in K3_OOF_SEEDS:
        np.savez_compressed(
            evaluations / f"real-oof-seed-{seed}.npz",
            object_ids=object_ids,
            oracle_errors_deg=errors,
        )
    rows = []
    for index in range(170):
        failed = index == 0
        rows.append(
            {
                "baseline": {
                    "wall_seconds": 2.0,
                    "final_rms": float("nan") if failed else 0.02,
                    "success": not failed,
                },
                "candidate": {
                    "wall_seconds": 1.0,
                    "final_rms": float("nan") if failed else 0.0201,
                    "success": not failed,
                },
            }
        )
    rows_path = downstream / "fixed-period-rows.json"
    rows_path.write_text(
        json.dumps(
            {
                "schema": "delphi.k3-fixed-period-rows.v1",
                "object_ids": object_ids.tolist(),
                "rows": rows,
            }
        ),
        encoding="utf-8",
    )
    summary = downstream / "fixed-period-summary.json"
    summary.write_text(
        json.dumps(
            {
                "schema": "delphi.k3-fixed-period-summary.v1",
                "rows_sha256": _hash(rows_path),
            }
        ),
        encoding="utf-8",
    )
    release = tmp_path / "release.json"
    release.write_text(
        json.dumps(
            {
                "schema": "delphi.k3-publication-summary.v1",
                "source_sha256": {
                    "oof_ensemble": _hash(ensemble),
                    "v1_comparator": _hash(comparator),
                    "downstream_summary": _hash(summary),
                },
            }
        ),
        encoding="utf-8",
    )

    result = build_publication_figures(
        artifact_root=root,
        comparator_path=comparator,
        release_summary_path=release,
        output_directory=tmp_path / "figures",
    )
    assert "k3_downstream_benchmark.pdf" in result["figure_sha256"]


def test_publication_figures_reject_tampered_comparator(tmp_path) -> None:
    release = tmp_path / "release.json"
    release.write_text(
        json.dumps(
            {
                "schema": "delphi.k3-publication-summary.v1",
                "source_sha256": {
                    "oof_ensemble": "0" * 64,
                    "v1_comparator": "0" * 64,
                    "downstream_summary": "0" * 64,
                },
            }
        ),
        encoding="utf-8",
    )
    root = tmp_path / "artifacts"
    (root / "evaluations").mkdir(parents=True)
    comparator = tmp_path / "comparators.npz"
    comparator.write_bytes(b"tampered")
    with pytest.raises(K3FigureError, match="missing release-bound artifact"):
        build_publication_figures(
            artifact_root=root,
            comparator_path=comparator,
            release_summary_path=release,
            output_directory=tmp_path / "figures",
        )
