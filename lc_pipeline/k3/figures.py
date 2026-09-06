"""Provenance-bound figures for the definitive K3 publication release."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from .publication_run import K3_OOF_SEEDS


class K3FigureError(ValueError):
    """Raised when release artifacts cannot support publication figures."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, schema: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise K3FigureError(f"cannot load {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != schema:
        raise K3FigureError(f"{path.name} has the wrong schema")
    return payload


def _cdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    if ordered.shape != (170,) or not np.all(np.isfinite(ordered)):
        raise K3FigureError("CDF input must contain 170 finite values")
    return ordered, np.arange(1, 171, dtype=np.float64) / 170.0


def _save_pdf(figure, destination: Path) -> str:
    if destination.exists():
        raise K3FigureError(f"refusing to overwrite figure {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=".pdf", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        figure.savefig(
            temporary,
            format="pdf",
            bbox_inches="tight",
            metadata={"CreationDate": None, "ModDate": None, "Creator": "DeLPHI"},
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return _sha256(destination)


def build_publication_figures(
    *,
    artifact_root: str | Path,
    comparator_path: str | Path,
    release_summary_path: str | Path,
    output_directory: str | Path,
) -> dict[str, object]:
    """Generate immutable figures only from artifacts bound by the release."""
    root = Path(artifact_root)
    evaluations = root / "evaluations"
    ensemble_path = evaluations / "real-oof-ensemble.npz"
    comparator = Path(comparator_path)
    downstream_summary_path = root / "downstream" / "fixed-period" / "fixed-period-summary.json"
    downstream_rows_path = root / "downstream" / "fixed-period" / "fixed-period-rows.json"
    release = _load_json(Path(release_summary_path), "delphi.k3-publication-summary.v1")
    source_hashes = release.get("source_sha256")
    if not isinstance(source_hashes, dict):
        raise K3FigureError("release summary lacks source hashes")
    expected = {
        "oof_ensemble": ensemble_path,
        "v1_comparator": comparator,
        "downstream_summary": downstream_summary_path,
    }
    for key, path in expected.items():
        if not path.is_file():
            raise K3FigureError(f"missing release-bound artifact {path}")
        if source_hashes.get(key) != _sha256(path):
            raise K3FigureError(f"{key} differs from the release-bound artifact")

    downstream_summary = _load_json(downstream_summary_path, "delphi.k3-fixed-period-summary.v1")
    downstream_rows = _load_json(downstream_rows_path, "delphi.k3-fixed-period-rows.v1")
    if downstream_summary.get("rows_sha256") != _sha256(downstream_rows_path):
        raise K3FigureError("downstream rows differ from the benchmark summary")

    try:
        with np.load(ensemble_path, allow_pickle=False) as artifact:
            if artifact["schema"].item() != "delphi.k3-real-oof-ensemble.v1":
                raise K3FigureError("OOF ensemble has the wrong schema")
            object_ids = tuple(artifact["object_ids"].astype(str))
            k3_errors = np.asarray(artifact["oracle_errors_deg"], dtype=np.float64)
        with np.load(comparator, allow_pickle=False) as artifact:
            if artifact["schema"].item() != "delphi.k3-v1-comparators.v1":
                raise K3FigureError("comparator has the wrong schema")
            if tuple(artifact["object_ids"].astype(str)) != object_ids:
                raise K3FigureError("comparator object order differs from K3")
            comparator_errors = {
                "Frozen V1": np.asarray(artifact["v1_errors_deg"], dtype=np.float64),
                "Train-only atlas": np.asarray(artifact["atlas_errors_deg"], dtype=np.float64),
                "Deranged control": np.asarray(artifact["deranged_errors_deg"], dtype=np.float64),
            }
        seed_errors = []
        for seed in K3_OOF_SEEDS:
            with np.load(evaluations / f"real-oof-seed-{seed}.npz", allow_pickle=False) as artifact:
                if tuple(artifact["object_ids"].astype(str)) != object_ids:
                    raise K3FigureError(f"seed {seed} object order differs")
                seed_errors.append(np.asarray(artifact["oracle_errors_deg"], dtype=np.float64))
    except (OSError, ValueError, KeyError) as exc:
        raise K3FigureError(f"cannot load figure inputs: {exc}") from exc
    _cdf(k3_errors)
    for values in comparator_errors.values():
        _cdf(values)
    if any(values.shape != (170,) or not np.all(np.isfinite(values)) for values in seed_errors):
        raise K3FigureError("per-seed errors are invalid")

    rows = downstream_rows.get("rows")
    if (
        downstream_rows.get("object_ids") != list(object_ids)
        or not isinstance(rows, list)
        or len(rows) != 170
    ):
        raise K3FigureError("downstream rows are not aligned to the OOF ensemble")
    try:
        baseline_time = np.asarray(
            [row["baseline"]["wall_seconds"] for row in rows], dtype=np.float64
        )
        candidate_time = np.asarray(
            [row["candidate"]["wall_seconds"] for row in rows], dtype=np.float64
        )
        baseline_rms = np.asarray([row["baseline"]["final_rms"] for row in rows], dtype=np.float64)
        candidate_rms = np.asarray(
            [row["candidate"]["final_rms"] for row in rows], dtype=np.float64
        )
        baseline_success = np.asarray(
            [row["baseline"]["success"] for row in rows], dtype=bool
        )
        candidate_success = np.asarray(
            [row["candidate"]["success"] for row in rows], dtype=bool
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise K3FigureError(f"downstream row values are invalid: {exc}") from exc
    if any(
        values.shape != (170,) or not np.all(np.isfinite(values)) or np.any(values <= 0)
        for values in (baseline_time, candidate_time)
    ):
        raise K3FigureError("downstream timing values must be finite and positive")
    if baseline_success.shape != (170,) or candidate_success.shape != (170,):
        raise K3FigureError("downstream success masks must contain 170 values")
    paired_success = baseline_success & candidate_success
    if int(np.sum(paired_success)) < 2:
        raise K3FigureError("downstream RMS figure requires two paired successful objects")
    for values, success in (
        (baseline_rms, baseline_success),
        (candidate_rms, candidate_success),
    ):
        if (
            values.shape != (170,)
            or np.any(~np.isfinite(values[success]))
            or np.any(values[success] <= 0)
        ):
            raise K3FigureError(
                "successful downstream RMS values must be finite and positive"
            )

    try:
        import matplotlib
    except ModuleNotFoundError as exc:
        raise K3FigureError("publication figures require the optional 'plot' dependencies") from exc

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
        }
    )
    destination = Path(output_directory)
    hashes: dict[str, str] = {}

    figure, axis = plt.subplots(figsize=(3.4, 2.6))
    for label, values, style in (
        ("DeLPHI K3 ensemble", k3_errors, {"linewidth": 2.0}),
        ("Frozen V1", comparator_errors["Frozen V1"], {"linestyle": "--"}),
        ("Train-only atlas", comparator_errors["Train-only atlas"], {"linestyle": ":"}),
        ("Deranged control", comparator_errors["Deranged control"], {"linestyle": "-."}),
    ):
        x, y = _cdf(values)
        axis.step(x, y, where="post", label=label, **style)
    axis.axvline(20.0, color="0.5", linewidth=0.8)
    axis.set(
        xlabel="Antipode-aware oracle error (deg)",
        ylabel="Cumulative fraction",
        xlim=(0, 90),
        ylim=(0, 1),
    )
    axis.grid(alpha=0.2)
    axis.legend(loc="lower right")
    hashes["k3_oracle_error_cdf.pdf"] = _save_pdf(figure, destination / "k3_oracle_error_cdf.pdf")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(3.4, 2.6))
    positions = np.arange(len(K3_OOF_SEEDS))
    means = np.asarray([np.mean(values) for values in seed_errors])
    medians = np.asarray([np.median(values) for values in seed_errors])
    axis.plot(positions, means, "o-", label="Mean")
    axis.plot(positions, medians, "s--", label="Median")
    axis.axhline(np.mean(k3_errors), color="0.25", linewidth=0.8, label="Ensemble mean")
    axis.set_xticks(positions, [str(seed) for seed in K3_OOF_SEEDS])
    axis.set(xlabel="Frozen seed", ylabel="Oracle error (deg)")
    axis.grid(alpha=0.2)
    axis.legend()
    hashes["k3_seed_stability.pdf"] = _save_pdf(figure, destination / "k3_seed_stability.pdf")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(6.8, 2.6))
    axes[0].scatter(baseline_time, candidate_time, s=8, alpha=0.55)
    limit = float(max(np.max(baseline_time), np.max(candidate_time)))
    axes[0].plot([0, limit], [0, limit], color="0.3", linewidth=0.8)
    axes[0].set(xlabel="Standard starts wall time (s)", ylabel="K3 starts wall time (s)")
    paired_baseline_rms = baseline_rms[paired_success]
    paired_candidate_rms = candidate_rms[paired_success]
    axes[1].scatter(paired_baseline_rms, paired_candidate_rms, s=8, alpha=0.55)
    low = float(min(np.min(paired_baseline_rms), np.min(paired_candidate_rms)))
    high = float(max(np.max(paired_baseline_rms), np.max(paired_candidate_rms)))
    axes[1].plot([low, high], [low, high], color="0.3", linewidth=0.8)
    axes[1].set(xlabel="Standard starts final RMS", ylabel="K3 starts final RMS")
    for axis in axes:
        axis.grid(alpha=0.2)
    hashes["k3_downstream_benchmark.pdf"] = _save_pdf(
        figure, destination / "k3_downstream_benchmark.pdf"
    )
    plt.close(figure)

    manifest = {
        "schema": "delphi.k3-publication-figures.v1",
        "release_summary_sha256": _sha256(Path(release_summary_path)),
        "source_sha256": {
            **{key: _sha256(path) for key, path in expected.items()},
            "downstream_rows": _sha256(downstream_rows_path),
        },
        "figure_sha256": hashes,
    }
    manifest_path = destination / "publication-figures.json"
    if manifest_path.exists():
        raise K3FigureError(f"refusing to overwrite figure manifest {manifest_path}")
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return {**manifest, "manifest_sha256": _sha256(manifest_path)}


__all__ = ["K3FigureError", "build_publication_figures"]
