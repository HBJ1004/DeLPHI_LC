"""Evidence-only supplementary exports from sealed reliability artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


class ReliabilitySupplementError(ValueError):
    pass


def _canon(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"


def _hash(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _unseal(p):
    d = json.loads(Path(p).read_text())
    x = d.get("payload")
    if d.get("seal_schema") != "sha256-canonical-json-v1" or hashlib.sha256(
        _canon(x).encode()
    ).hexdigest() != d.get("payload_sha256"):
        raise ReliabilitySupplementError(f"bad seal: {p}")
    return x


def _new(p, text):
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and p.read_text() != text:
        raise ReliabilitySupplementError(f"differing output: {p}")
    if not p.exists():
        p.write_text(text)


def _csv(p, rows):
    import io

    o = io.StringIO(newline="")
    w = csv.DictWriter(o, fieldnames=list(rows[0]), lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    _new(p, o.getvalue())


def _tex(p, rows):
    keys = list(rows[0])
    labels = {
        "n": "N",
        "min": "Minimum",
        "median": "Median",
        "max": "Maximum",
        "property": "Input property",
        "fold": "Fold",
        "mean_error_deg": "Mean error (deg)",
        "median_error_deg": "Median error (deg)",
        "baseline_years": "Baseline (yr)",
        "period_hours": "Period (h)",
        "merged_blocks": "Merged blocks",
        "observations": "Observations",
        "sessions": "Native sessions",
    }

    def value(x):
        if isinstance(x, (int, np.integer)):
            return str(int(x))
        if isinstance(x, (float, np.floating)):
            if float(x).is_integer():
                return str(int(x))
            return f"{float(x):.3f}"
        return str(x).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")

    lines = [
        "\\begin{tabular}{" + "l" * len(keys) + "}",
        " & ".join(labels.get(k, k.replace("_", " ")) for k in keys) + r" \\",
        r"\hline",
    ]
    lines += [
        " & ".join(value(labels.get(str(r[k]), r[k]) if k == "property" else r[k]) for k in keys)
        + r" \\"
        for r in rows
    ]
    lines += ["\\end{tabular}", ""]
    _new(p, "\n".join(lines))


def axial_cap_fraction(axes, sphere, radius_deg):
    """Uniform quadrature estimate of the union of antipodal axis caps."""
    axes = np.asarray(axes, dtype=float)
    norms = np.linalg.norm(axes, axis=1)
    if axes.ndim != 2 or axes.shape[1] != 3 or np.any(norms <= 0) or not np.all(np.isfinite(axes)):
        raise ReliabilitySupplementError("invalid cap axes")
    axes = axes / norms[:, None]
    return float(
        np.any(np.abs(np.asarray(sphere) @ axes.T) >= np.cos(np.deg2rad(radius_deg)), axis=1).mean()
    )


def export(run_root, output, mpc_table=None):
    root = Path(run_root)
    out = Path(output)
    study = _unseal(root / "study.json")
    report = _unseal(root / "report.json")
    scored = _unseal(root / "scored-rows.json")
    _unseal(root / "withheld-rows.json")
    inputs = {
        x: _hash(root / x)
        for x in (
            "study.json",
            "report.json",
            "scored-rows.json",
            "withheld-rows.json",
            "sampling-table.csv",
            "candidate-sky.pdf",
            "period-sensitivity.pdf",
            "sampling-caps.pdf",
        )
    }
    inputs["exporter_source"] = _hash(Path(__file__))
    objects = study["objects"]
    mpc_table = (
        Path(mpc_table)
        if mpc_table is not None
        else Path(objects[0]["lightcurve"]["path"]).parents[2] / "tables" / "asteroids.csv"
    )
    if not mpc_table.is_file():
        raise ReliabilitySupplementError("MPC identity table unavailable")
    inputs["asteroids.csv"] = _hash(mpc_table)
    objects = study["objects"]
    scheduled = [r for r in scored if r["condition"] == "full"]
    full = [r for r in scheduled if r["status"] == "ok"]
    if len({r["object_id"] for r in scheduled}) != 170 or len(scheduled) != 170 or len(full) != 170:
        raise ReliabilitySupplementError(
            "full OOF receipt must be exactly 170 unique successful rows"
        )
    # One object-repeat row makes transparent CDF and fold summaries without a model comparison.
    errors = np.array([r["error_deg"] for r in full])
    cdf = np.sort(errors)
    rows = [{"error_deg": float(v), "cdf": float((i + 1) / len(cdf))} for i, v in enumerate(cdf)]
    _csv(out / "k3_oof_cdf.csv", rows)
    folds = []
    for fold in range(5):
        v = [r["error_deg"] for r in full if r["fold"] == fold]
        folds.append(
            {
                "fold": fold,
                "n": len(v),
                "mean_error_deg": float(np.mean(v)),
                "median_error_deg": float(np.median(v)),
            }
        )
    _csv(out / "k3_fold_metrics.csv", folds)
    _tex(out / "k3_fold_metrics.tex", folds)
    desc = [
        {
            "object_id": o["object_id"],
            "observations": o["native_observations"],
            "sessions": o["native_sessions"],
            "merged_blocks": o["merged_blocks"],
            "baseline_years": o["time_span_days"] / 365.25,
            "period_hours": o["period_hours"],
        }
        for o in objects
    ]
    _csv(out / "k3_observing_structure.csv", desc)
    summary_rows = []
    for key in ("observations", "sessions", "merged_blocks", "baseline_years", "period_hours"):
        values = np.array([float(r[key]) for r in desc])
        summary_rows.append(
            {
                "property": key,
                "n": len(values),
                "min": float(values.min()),
                "median": float(np.median(values)),
                "max": float(values.max()),
            }
        )
    _csv(out / "k3_data_summary.csv", summary_rows)
    _tex(out / "k3_data_summary.tex", summary_rows)
    rms = []
    for key, v in report["withheld"].items():
        if not isinstance(v, dict) or "withheld_rms" not in v:
            continue
        rms.append(
            {
                "arm": key,
                "n": v["n_scheduled"],
                "mean_rms": v["withheld_rms"]["estimate"],
                "ci95_low": v["withheld_rms"]["bootstrap_lower_95"],
                "ci95_high": v["withheld_rms"]["bootstrap_upper_95"],
            }
        )
    _csv(out / "k3_withheld_rms.csv", rms)
    # 170-object descriptive associations: no threshold is selected from these.
    by_id = {r["object_id"]: r for r in full}
    association = []
    for field, label in [
        ("reference_primary_abs_latitude_deg", "primary_reference_abs_latitude_deg"),
        ("reference_solution_count", "reference_solution_count"),
    ]:
        vals = np.array([float(by_id[o["object_id"]][field]) for o in objects])
        err = np.array([float(by_id[o["object_id"]]["error_deg"]) for o in objects])

        def avgrank(x):
            order = np.argsort(x, kind="stable")
            result = np.empty(len(x))
            pos = 0
            while pos < len(x):
                end = pos + 1
                while end < len(x) and x[order[end]] == x[order[pos]]:
                    end += 1
                result[order[pos:end]] = (pos + end - 1) / 2 + 1
                pos = end
            return result

        rho = float(np.corrcoef(avgrank(vals), avgrank(err))[0, 1])
        if field == "reference_solution_count":
            bins = [(1, 1), (2, 2), (3, 3)]
        else:
            bins = [(0, 30), (30, 60), (60, 90)]
        for i, (low, high) in enumerate(bins):
            if field == "reference_solution_count":
                mask = (vals >= low) & (vals <= high)
            else:
                mask = (vals >= low) & (
                    (vals < high) if i < len(bins) - 1 else (vals <= high)
                )
            association.append(
                {
                    "property": label,
                    "spearman_rho": rho,
                    "stratum": f"{low}-{high}",
                    "lower": low,
                    "upper": high,
                    "n": int(mask.sum()),
                    "mean_error_deg": float(err[mask].mean()) if mask.any() else None,
                    "interpretation": "descriptive fixed bins; no threshold",
                }
            )
    _csv(out / "k3_reference_strata.csv", association)
    # Descriptive Spearman matrix. Average ranks preserve ties; no p-values or
    # selection claims are made from these associations.
    matrix_fields = {
        "oracle_error_deg": np.array([by_id[o["object_id"]]["error_deg"] for o in objects]),
        "log_observations": np.log(np.array([o["native_observations"] for o in objects])),
        "native_sessions": np.array([o["native_sessions"] for o in objects]),
        "merged_blocks": np.array([o["merged_blocks"] for o in objects]),
        "baseline_years": np.array([o["time_span_days"] / 365.25 for o in objects]),
        "period_hours": np.array([o["period_hours"] for o in objects]),
        "primary_abs_latitude_deg": np.array(
            [by_id[o["object_id"]]["reference_primary_abs_latitude_deg"] for o in objects]
        ),
        "reference_count": np.array(
            [by_id[o["object_id"]]["reference_solution_count"] for o in objects]
        ),
    }

    def avgrank_matrix(x):
        order = np.argsort(x, kind="stable")
        r = np.empty(len(x))
        start = 0
        while start < len(x):
            end = start + 1
            while end < len(x) and x[order[end]] == x[order[start]]:
                end += 1
            r[order[start:end]] = (start + end - 1) / 2 + 1
            start = end
        return r

    names = list(matrix_fields)
    corr = np.array(
        [
            [
                np.corrcoef(avgrank_matrix(matrix_fields[a]), avgrank_matrix(matrix_fields[b]))[
                    0, 1
                ]
                for b in names
            ]
            for a in names
        ]
    )
    matrix_rows = [
        {
            "property": a,
            **{b: float(corr[i, j]) for j, b in enumerate(names)},
            "interpretation": "descriptive association, not causation",
        }
        for i, a in enumerate(names)
    ]
    _csv(out / "k3_descriptive_spearman_matrix.csv", matrix_rows)
    # Deterministic uniform-sphere quadrature.  It reports geometric cap union
    # area, not a confidence interval or timing measurement.
    radii = [5, 10, 15, 20, 30, 45]

    def sphere_grid(n):
        i = np.arange(n)
        z = 1 - 2 * (i + 0.5) / n
        phi = np.pi * (3 - np.sqrt(5)) * i
        return np.c_[np.sqrt(1 - z * z) * np.cos(phi), np.sqrt(1 - z * z) * np.sin(phi), z]

    sphere = sphere_grid(16384)
    sphere_hi = sphere_grid(32768)
    caprows = []
    for radius in radii:
        unions = []
        contain = []
        for row in full:
            unions.append(axial_cap_fraction(row["axes"], sphere, radius))
            contain.append(float(row["error_deg"] <= radius))
        # convergence on the same axial, antipode-aware cap union
        hi = []
        for row in full:
            hi.append(axial_cap_fraction(row["axes"], sphere_hi, radius))
        caprows.append(
            {
                "radius_deg": radius,
                "quadrature_points": 16384,
                "mean_candidate_cap_union_fraction": float(np.mean(unions)),
                "resolution_32768_fraction": float(np.mean(hi)),
                "resolution_difference": float(np.mean(hi) - np.mean(unions)),
                "empirical_reference_containment_fraction": float(np.mean(contain)),
                "n_objects": len(full),
                "disclosure": "Fibonacci uniform-sphere midpoint quadrature; axial antipodes included; deterministic area approximation, not confidence/timing.",
            }
        )
    _csv(out / "k3_candidate_cap_containment.csv", caprows)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def save(name, draw):
        p = out / name
        if p.exists():
            return
        fig, ax = plt.subplots(figsize=(7.1, 4.3))
        draw(ax)
        fig.tight_layout()
        fig.savefig(
            p,
            format="pdf",
            metadata={
                "Creator": "DeLPHI evidence-only supplement",
                "CreationDate": None,
                "ModDate": None,
            },
        )
        plt.close(fig)

    save(
        "k3_oof_cdf.pdf",
        lambda a: (
            a.plot(cdf, np.arange(1, len(cdf) + 1) / len(cdf), label="K3 OOF"),
            a.set(xlabel="Angular error (degrees)", ylabel="Empirical CDF"),
            a.legend(fontsize=9),
        ),
    )
    save(
        "k3_observing_structure.pdf",
        lambda a: (
            a.scatter([x["observations"] for x in desc], [x["baseline_years"] for x in desc], s=10),
            a.set(xlabel="Native observations", ylabel="Time baseline (years)"),
        ),
    )
    save(
        "k3_withheld_rms.pdf",
        lambda a: (
            a.errorbar(
                range(len(rms)),
                [x["mean_rms"] for x in rms],
                yerr=[
                    [x["mean_rms"] - x["ci95_low"] for x in rms],
                    [x["ci95_high"] - x["mean_rms"] for x in rms],
                ],
                fmt="o",
            ),
            a.set(xlabel="Prespecified arm / sampling", ylabel="Withheld normalized RMS"),
            a.set_xticks(range(len(rms)), [x["arm"] for x in rms], rotation=55, ha="right"),
            a.tick_params(axis="x", labelsize=8),
        ),
    )
    save(
        "k3_reference_strata.pdf",
        lambda a: (
            a.scatter(
                [
                    x["lower"]
                    for x in association
                    if x["property"] == "primary_reference_abs_latitude_deg"
                ],
                [
                    x["mean_error_deg"]
                    for x in association
                    if x["property"] == "primary_reference_abs_latitude_deg"
                ],
            ),
            a.set(
                xlabel="Primary reference |latitude| stratum lower bound (deg)",
                ylabel="Mean K3 OOF error (deg)",
            ),
        ),
    )

    def draw_matrix(ax):
        labels = [
            "Oracle@3 (deg)",
            "Log observations",
            "Native sessions",
            "Observing blocks",
            "Time span (yr)",
            "Period (h)",
            "|Ref. latitude| (deg)",
            "Reference count",
        ]
        image = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(len(names)), labels, rotation=55, ha="right", fontsize=9)
        ax.set_yticks(range(len(names)), labels, fontsize=9)
        ax.set_title("Descriptive Spearman correlations", fontsize=10)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(
                    j,
                    i,
                    f"{corr[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="white" if abs(corr[i, j]) > 0.75 else "black",
                )
        bar = ax.figure.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
        bar.set_label("Spearman correlation", fontsize=9)
        bar.ax.tick_params(labelsize=9)

    save("k3_descriptive_spearman_matrix.pdf", draw_matrix)
    save(
        "k3_candidate_cap_containment.pdf",
        lambda a: (
            a.plot(
                radii,
                [x["mean_candidate_cap_union_fraction"] for x in caprows],
                marker="o",
                label="Candidate-cap union",
            ),
            a.plot(
                radii,
                [x["empirical_reference_containment_fraction"] for x in caprows],
                marker="s",
                label="Reference containment",
            ),
            a.set(xlabel="Angular radius (deg)", ylabel="Fraction"),
            a.legend(fontsize=9),
        ),
    )
    # Correctly mapped multi-epoch example: internal asteroid_101 is MPC 2 Pallas.
    mapping = list(csv.DictReader(mpc_table.open()))
    match = next(
        (r for r in mapping if r.get("internal_id", "") == "101" or r.get("id", "") == "101"), None
    )
    if match is None or match.get("number") != "2" or match.get("name") != "Pallas":
        raise ReliabilitySupplementError("internal asteroid_101 does not map to MPC 2 Pallas")
    pallas = next(o for o in objects if o["object_id"] == "asteroid_101")
    lc = Path(pallas["lightcurve"]["path"])
    if _hash(lc) != pallas["lightcurve"]["sha256"]:
        raise ReliabilitySupplementError("Pallas lightcurve hash mismatch")
    inputs["asteroid_101_lc.txt"] = _hash(lc)
    data = []
    lines = lc.read_text().splitlines()
    pos = 1
    for _ in range(int(lines[0])):
        n = int(lines[pos].split()[0])
        pos += 1
        for line in lines[pos : pos + n]:
            q = line.split()
            data.append((float(q[0]), float(q[1])))
        pos += n
    arr = np.asarray(data)
    save(
        "k3_pallas_multiepoch_example.pdf",
        lambda a: (
            a.scatter(arr[:, 0] - arr[:, 0].min(), arr[:, 1], s=4),
            a.set(
                xlabel="Days since first observation",
                ylabel="Relative flux",
                title="MPC 2 Pallas (internal asteroid_101): native multi-epoch input",
            ),
        ),
    )
    # Existing candidate/period renderings are immutable evidence: copy once and bind their hashes.
    for name in ("candidate-sky.pdf", "period-sensitivity.pdf", "sampling-caps.pdf"):
        target = out / f"k3_{name}"
        if not target.exists():
            shutil.copyfile(root / name, target)
    summary = {
        "schema": "delphi.k3-reliability-supplement-numbers.v1",
        "n_objects": len(objects),
        "n_oof": len(full),
        "full_failure_count": len(scheduled) - len(full),
        "atlas_cdf": "not generated: no atlas_errors_deg comparator located under searched release data paths",
        "withheld_arms": rms,
        "withheld_statistics": report["withheld"],
        "sampling_summary": report["sampling"],
        "reference_strata": association,
        "cap_containment": caprows,
        "example": {"internal_id": "asteroid_101", "mpc_number": 2, "name": "Pallas"},
        "atlas_control": report["atlas_control"],
        "random_control": report["random_control"],
    }
    _new(out / "k3_supplement_numbers.json", json.dumps(summary, sort_keys=True, indent=2) + "\n")
    manifest = {
        "schema": "delphi.k3-reliability-supplement.v1",
        "source_sha256": inputs,
        "files": {
            p.name: _hash(p) for p in out.iterdir() if p.is_file() and p.name != "manifest.json"
        },
        "reproduction_command": "python-pinned repro/export_k3_reliability_supplement.py --run-root <sealed-run> --output <output>",
    }
    _new(out / "manifest.json", json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return manifest
