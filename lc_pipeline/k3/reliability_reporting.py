"""Report materialisation for the preregistered K3 reliability study.

This module is deliberately downstream of :mod:`reliability_scoring`: it only
formats sealed score rows and frozen input metadata.  It does not select an
operating threshold, rank candidates, or inspect reference axes while making
plots.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .reliability_scoring import (
    reliability_table_rows,
    render_reliability_tex,
)


class ReliabilityReportingError(ValueError):
    """Raised when a sealed report cannot be resumed consistently."""


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ReliabilityReportingError(f"report text differs on resume: {path}")
    else:
        from .reliability_study import _write_text_new
        _write_text_new(path, text)
    return path


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    if not rows:
        return _write_text(path, "")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output, fieldnames=list(rows[0]), extrasaction="ignore", lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    return _write_text(path, output.getvalue())


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _family(condition: str, lock: Mapping[str, Any]) -> str:
    for entry in lock.get("conditions", []):
        if entry.get("condition") == condition:
            return str(entry.get("family", "unknown"))
    if condition.startswith("period-"):
        return "period"
    if condition.startswith("two_d-"):
        return "two_d"
    return condition.split("-", 1)[0]


def _summary_rows(report: Mapping[str, Any], lock: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Flatten scorer output and attach family labels and count columns."""
    sampling = report.get("sampling", report)
    rows = reliability_table_rows(sampling)
    condition_config = {
        str(entry.get("condition")): entry.get("sampling", {})
        for entry in lock.get("conditions", [])
        if isinstance(entry, Mapping)
    }
    result = []
    for row in rows:
        value = dict(row)
        condition = str(row["condition"])
        value["family"] = _family(condition, lock)
        model_summary = sampling.get("models", {}).get(str(row["model_id"]), {}).get(condition, {})
        value["mean_error_ci95_low"] = model_summary.get("mean_error_bootstrap_lower_95")
        value["mean_error_ci95_high"] = model_summary.get("mean_error_bootstrap_upper_95")
        config = condition_config.get(condition, {})
        for key in ("cap", "block_count", "per_block_cap", "fraction", "noise_sigma"):
            if key in config:
                value[key] = config[key]
        value["n_label"] = value.get("n_eligible_objects")
        result.append(value)
    return result


def _metadata_objects(lock: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    objects = lock.get("objects", [])
    return {
        str(row["object_id"]): row
        for row in objects
        if isinstance(row, Mapping) and "object_id" in row
    }


def _prediction_metadata(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if isinstance(row.get("metadata"), Mapping):
            result.setdefault(str(row.get("object_id")), row["metadata"])
    return result


def _requirements_rows() -> list[dict[str, str]]:
    return [
        {
            "category": "software/input requirement",
            "requirement": "supplied period",
            "contract": "period_hours is finite and positive",
            "evidence": "native input validation",
            "interpretation": "required input contract",
        },
        {
            "category": "software/input requirement",
            "requirement": "native session",
            "contract": "at least 2 valid positive-flux observations per retained session",
            "evidence": "sampling validator",
            "interpretation": "required input contract",
        },
        {
            "category": "software/input requirement",
            "requirement": "observing geometry",
            "contract": "finite nonzero Sun and observer vectors for every retained observation",
            "evidence": "native input validation",
            "interpretation": "required input contract",
        },
        {
            "category": "empirical interpretation",
            "requirement": "observation count",
            "contract": "no universal N threshold is asserted",
            "evidence": "matched degradation curves and strata",
            "interpretation": "cohort- and condition-conditional only",
        },
    ]


def _descriptive_rows(
    rows: Sequence[Mapping[str, Any]], lock: Mapping[str, Any]
) -> list[dict[str, Any]]:
    objects = _metadata_objects(lock)
    prediction_meta = _prediction_metadata(rows)
    full = [
        row
        for row in rows
        if str(row.get("condition")) == "full" and row.get("status") in {"ok", "failed"}
    ]
    by_object: dict[str, list[float]] = {}
    direct_meta: dict[str, dict[str, Any]] = {}
    for row in full:
        direct_meta.setdefault(str(row.get("object_id")), {}).update(row)
        error = _number(row.get("error_deg"))
        if error is not None:
            by_object.setdefault(str(row.get("object_id")), []).append(error)
    fields = (
        "n_points",
        "native_observations",
        "period_hours",
        "time_span_days",
        "n_sessions",
        "native_sessions",
        "merged_blocks",
        "flux_cv",
        "occupied_phase_bins",
        "phase_coverage_fraction",
        "reference_count",
        "refcount",
        "reference_solution_count",
        "beta",
        "pole_beta",
        "reference_primary_abs_latitude_deg",
    )
    output: list[dict[str, Any]] = []
    for field in fields:
        values: list[tuple[float, float]] = []
        for oid, errors in by_object.items():
            source = objects.get(oid, {})
            nested = prediction_meta.get(oid, {})
            direct = direct_meta.get(oid, {})
            value = source.get(field, nested.get(field, direct.get(field)))
            number = _number(value)
            if number is not None and errors:
                values.append((number, float(np.mean(errors))))
        if not values:
            continue
        data = np.asarray([v[0] for v in values], dtype=float)
        # Quantile bins are descriptive and have no decision interpretation.
        edges = np.unique(np.quantile(data, np.linspace(0, 1, min(5, len(data) + 1))))
        if len(edges) < 2:
            edges = np.asarray([data[0] - 0.5, data[0] + 0.5])
        groups = np.digitize(data, edges[1:-1], right=True)
        for index in range(len(edges) - 1):
            selected = [values[pos][1] for pos, group in enumerate(groups) if group == index]
            if selected:
                output.append(
                    {
                        "field": field,
                        "bin": index + 1,
                        "lower": float(edges[index]),
                        "upper": float(edges[index + 1]),
                        "n_objects": len(selected),
                        "mean_error_deg": float(np.mean(selected)),
                        "caution": "descriptive stratum; not a threshold",
                    }
                )
    return output


def _plot_path(path: Path, title: str, draw: Any) -> Path:
    import os
    import tempfile

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8.5, 5.5))
    try:
        draw(axis)
        axis.set_title(title)
        figure.tight_layout()
        fd, temporary = tempfile.mkstemp(prefix=".pending-figure-", dir=path.parent)
        os.close(fd)
        try:
            figure.savefig(temporary, format="pdf", metadata={"Creator": "DeLPHI K3 reliability reporting"})
            os.link(temporary, path)
        finally:
            os.unlink(temporary)
    finally:
        plt.close(figure)
    return path


def _caps_plot(path: Path, table: Sequence[Mapping[str, Any]]) -> Path:
    values = [
        row
        for row in table
        if row.get("family") == "observation_cap" and _number(row.get("mean_error_deg")) is not None
    ]

    def draw(axis: Any) -> None:
        if not values:
            axis.text(0.5, 0.5, "No scored sampling rows", ha="center", va="center")
            axis.set_axis_off()
            return
        labels = [str(row.get("cap", "all")) for row in values]
        means = [float(row["mean_error_deg"]) for row in values]
        low = [
            float(_number(row.get("mean_error_ci95_low")) or mean)
            for row, mean in zip(values, means, strict=True)
        ]
        high = [
            float(_number(row.get("mean_error_ci95_high")) or mean)
            for row, mean in zip(values, means, strict=True)
        ]
        x = np.arange(len(labels))
        y = np.asarray(means, dtype=np.float64)
        axis.errorbar(
            x,
            y,
            yerr=np.asarray(
                [y - np.asarray(low, dtype=np.float64), np.asarray(high, dtype=np.float64) - y],
                dtype=np.float64,
            ),
            fmt="o",
            capsize=3,
        )
        axis.set_xticks(x, labels, rotation=70, ha="right", fontsize=7)
        axis.set_ylabel("mean oracle@3 error (deg)")
        axis.set_xlabel("condition (n label = eligible-object count)")
        for index, row in enumerate(values):
            axis.annotate(str(row.get("n_label", "")), (index, means[index]), fontsize=6)

    return _plot_path(path, "Matched sampling degradation (descriptive)", draw)


def _block_plot(path: Path, table: Sequence[Mapping[str, Any]]) -> Path:
    values = [row for row in table if row.get("family") == "two_d"]

    def draw(axis: Any) -> None:
        if not values:
            axis.text(0.5, 0.5, "No eligible block-grid rows", ha="center", va="center")
            axis.set_axis_off()
            return
        block_levels = (1, 3, 5, 10, None)
        cap_levels = (5, 10, 20, 50, None)
        lookup = {(row.get("block_count"), row.get("per_block_cap")): row for row in values}
        matrix = np.full((len(block_levels), len(cap_levels)), np.nan)
        for i, block_count in enumerate(block_levels):
            for j, cap in enumerate(cap_levels):
                number = _number(lookup.get((block_count, cap), {}).get("mean_degradation"))
                if number is not None:
                    matrix[i, j] = number
        image = axis.imshow(matrix, aspect="auto", cmap="coolwarm")
        axis.set_xticks(
            np.arange(len(cap_levels)), [str(x or "all") for x in cap_levels], fontsize=8
        )
        axis.set_yticks(
            np.arange(len(block_levels)), [str(x or "all") for x in block_levels], fontsize=8
        )
        axis.set_xlabel("points per block; cell = mean change (deg), n=matched objects")
        axis.set_ylabel("retained merged blocks")
        for i, block_count in enumerate(block_levels):
            for j, cap in enumerate(cap_levels):
                row = lookup.get((block_count, cap))
                if row is not None:
                    mean = _number(row.get("mean_degradation"))
                    label = "—" if mean is None else f"{mean:.2g}\nn={row.get('n_matched', '—')}"
                    axis.text(j, i, label, ha="center", va="center", fontsize=8)
        axis.figure.colorbar(image, ax=axis)

    return _plot_path(path, "Observing-block grid: matched mean oracle@3 change", draw)


def _period_plot(path: Path, table: Sequence[Mapping[str, Any]]) -> Path:
    values = [
        row
        for row in table
        if row.get("family") == "period" and _number(row.get("mean_error_deg")) is not None
    ]

    def draw(axis: Any) -> None:
        labels = [str(row["condition"]).removeprefix("period-") for row in values]
        means = [float(row["mean_error_deg"]) for row in values]
        axis.plot(np.arange(len(labels)), means, "o-")
        axis.set_xticks(np.arange(len(labels)), labels, rotation=65, ha="right", fontsize=7)
        axis.set_ylabel("mean oracle@3 error (deg)")
        axis.set_xlabel("period multiplier; supplied P remains the reference input")

    return _plot_path(path, "Period sensitivity (not period recovery)", draw)


def _descriptive_plot(path: Path, descriptive: Sequence[Mapping[str, Any]]) -> Path:
    def draw(axis: Any) -> None:
        fields = sorted({str(row["field"]) for row in descriptive})
        for index, field in enumerate(fields):
            values = [row for row in descriptive if row["field"] == field]
            axis.plot(
                [index] * len(values), [row["mean_error_deg"] for row in values], "o", label=field
            )
        if fields:
            axis.set_xticks(range(len(fields)), fields, rotation=70, ha="right", fontsize=7)
            axis.legend(fontsize=7)
        else:
            axis.text(0.5, 0.5, "No metadata strata available", ha="center", va="center")
        axis.set_ylabel("full-input mean error (deg)")

    return _plot_path(path, "Full-input descriptive strata", draw)


def _sky_plot(
    path: Path, scored_rows: Sequence[Mapping[str, Any]], lock: Mapping[str, Any]
) -> Path:
    methods: dict[str, np.ndarray] = {}
    axes = [
        row.get("axes")
        for row in scored_rows
        if row.get("condition") == "full"
        and row.get("status") == "ok"
        and row.get("axes") is not None
    ]
    if axes:
        methods["K3 candidates (all axes)"] = np.asarray(axes, dtype=float).reshape(-1, 3)
    atlas = [
        entry.get("axes") for entry in lock.get("atlases", []) if entry.get("axes") is not None
    ]
    if atlas:
        methods["train-only atlas axes"] = np.asarray(atlas, dtype=float).reshape(-1, 3)
    from .reliability_study import stable_key
    random = []
    for row in scored_rows:
        if row.get("condition") == "full" and row.get("status") == "ok":
            seed = int(stable_key(row["object_id"], "random-three-20260915")[:16], 16)
            draw = np.random.default_rng(seed).normal(size=(3, 3))
            random.extend(draw / np.linalg.norm(draw, axis=1, keepdims=True))
    if random:
        methods["uniform random axes (first draw)"] = np.asarray(random)
    for name, vectors in methods.items():
        methods[name] = vectors * np.where(vectors[:, 2:3] < 0, -1, 1)

    def draw(axis: Any) -> None:
        if not methods:
            axis.text(0.5, 0.5, "No candidate axes available", ha="center", va="center")
            axis.set_axis_off()
            return
        for label, values in methods.items():
            normalized = values / np.linalg.norm(values, axis=1, keepdims=True)
            axis.scatter(
                np.degrees(np.arctan2(normalized[:, 1], normalized[:, 0])),
                np.degrees(np.arcsin(np.clip(normalized[:, 2], -1, 1))),
                s=10,
                alpha=0.55,
                label=f"{label} (n={len(values)})",
            )
        axis.set_xlabel("longitude (deg)")
        axis.set_ylabel("latitude (deg)")
        axis.legend(fontsize=7)
        axis.set_xlim(-180, 180)
        axis.set_ylim(-90, 90)

    return _plot_path(path, "Candidate-axis sky map (all returned axes; descriptive)", draw)


def _guidance(table: Sequence[Mapping[str, Any]]) -> str:
    families = sorted({str(row.get("family")) for row in table})
    return (
        """# Reliability-study output guidance

These outputs describe matched, retrospective same-identity conditions under
the supplied-period and observing-geometry assumptions. They are not a
prospective validation, a model ranking, or a universal observation-count
recommendation.

Interpret sampling and block curves together with their eligible-object and
matched-pair denominators, failure rates, and bootstrap intervals. A visible
change is conditional on this cohort and tested regime; sparse strata or wide
intervals are inconclusive. No first threshold is promoted, and untested
temporal coverage, geometry, noise, session structure, and period errors
remain uncharacterized.

Families present in this report: """
        + (", ".join(families) if families else "none")
        + ".\n"
    )


def write_reports(
    root: str | Path,
    scored_rows: Sequence[Mapping[str, Any]],
    report: Mapping[str, Any],
    study_lock: Mapping[str, Any],
) -> list[Path]:
    """Write denominator-complete tables, figures, and conditional guidance.

    A manifest binds the numeric inputs. On resume it must match exactly; text
    artifacts are byte-compared, while existing PDFs are retained after the
    numeric manifest check (PDF metadata can vary between Matplotlib versions).
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    numeric = {
        "rows": list(scored_rows),
        "report": report,
        "objects": study_lock.get("objects", []),
        "conditions": study_lock.get("conditions", []),
    }
    fingerprint = hashlib.sha256(_canonical(numeric).encode("utf-8")).hexdigest()
    manifest_path = root / "report-manifest.json"
    manifest = {"schema": "delphi.k3-reliability-report.v1", "numeric_sha256": fingerprint}
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise ReliabilityReportingError("numeric report inputs differ on resume")
    table = _summary_rows(report, study_lock)
    paths: list[Path] = []
    paths.append(_write_csv(root / "reliability-report.csv", table))
    paths.append(
        _write_text(
            root / "reliability-report.tex", render_reliability_tex(report.get("sampling", report))
        )
    )
    paths.append(_write_csv(root / "software-requirements.csv", _requirements_rows()))
    requirements = _requirements_rows()
    req_lines = [
        r"\begin{tabular}{lllll}",
        r"Category & Requirement & Contract & Evidence & Interpretation \\",
        r"\hline",
    ]
    for row in requirements:
        cells = [
            str(row[key]).replace("&", r"\&")
            for key in ("category", "requirement", "contract", "evidence", "interpretation")
        ]
        req_lines.append(" & ".join(cells) + r" \\")
    req_lines.append(r"\end{tabular}")
    req_tex = "\n".join(req_lines) + "\n"
    paths.append(_write_text(root / "software-requirements.tex", req_tex))
    descriptive = _descriptive_rows(scored_rows, study_lock)
    paths.append(_write_csv(root / "full-input-descriptive.csv", descriptive))

    # Existing PDFs are retained after the numeric manifest check. This makes
    # resume independent of backend PDF metadata while still binding every
    # plotted number to the manifest above.
    def plot_once(path: Path, builder: Any) -> Path:
        if not path.exists():
            builder()
        return path

    paths.append(
        plot_once(root / "sampling-caps.pdf", lambda: _caps_plot(root / "sampling-caps.pdf", table))
    )
    paths.append(
        plot_once(root / "block-matrix.pdf", lambda: _block_plot(root / "block-matrix.pdf", table))
    )
    paths.append(
        plot_once(
            root / "period-sensitivity.pdf",
            lambda: _period_plot(root / "period-sensitivity.pdf", table),
        )
    )
    paths.append(
        plot_once(
            root / "full-input-descriptive.pdf",
            lambda: _descriptive_plot(root / "full-input-descriptive.pdf", descriptive),
        )
    )
    paths.append(
        plot_once(
            root / "candidate-sky.pdf",
            lambda: _sky_plot(root / "candidate-sky.pdf", scored_rows, study_lock),
        )
    )
    paths.append(_write_text(root / "README-guidance.md", _guidance(table)))
    if manifest_path.exists():
        paths.append(manifest_path)
    else:
        paths.append(_write_text(manifest_path, _canonical(manifest) + "\n"))
    return paths


__all__ = ["ReliabilityReportingError", "write_reports"]
