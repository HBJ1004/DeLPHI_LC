"""Own-period copy of release/paper-v1/staging/part-b-analyses/figures-code/render_main_figures.py.

    python -m repro.own_period.maps.render_figures --analyses A --run R --periods P --out O

Only the eight figures whose DeLPHI content depends on the period are drawn
(Figs 3, 4, 5, 6, 9, 11, 12 and the right panel of 15).  Every drawing
statement is the published one; only the input files (run ensemble, mirror
symmetry, amplitude-aspect comparison, candidate count, reference
reassignment), the folding period of the worked example and the output
folder are changed.  Reference-only parts (standard poles, random axes,
lower-quality curve, amplitude-aspect map) keep the published inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_ARGS = argparse.ArgumentParser(description=__doc__)
_ARGS.add_argument("--analyses", type=Path, required=True)
_ARGS.add_argument("--run", type=Path, required=True, help="directory holding evaluations/")
_ARGS.add_argument("--periods", required=True, help="damit or a periods JSON (worked-example folding)")
_ARGS.add_argument("--out", type=Path, required=True)
_ARGS.add_argument("--figures", nargs="+", choices=("worked-example", "oracle-error", "candidate-sky", "primary-coverage", "interpretation", "candidate-count", "reference-reassignment", "latitude"))
ARGS = _ARGS.parse_args()
from repro import network_root as network
from repro.own_period.reporting_inputs import DataPath, load_npz

ROOT = DataPath("../../../paper")
FOLLOWUP = DataPath("../..")
DAMIT_DUMP = ROOT.parent / "damit-20250610T000301Z"
from repro.own_period.reporting_inputs import InputPath

ENSEMBLE = InputPath(ARGS.run) / "evaluations" / "real-oof-ensemble.npz"
ANALYSES = InputPath(ARGS.analyses)
FIG_OUT = ARGS.out
METHOD_EXAMPLE_OBJECT = "asteroid_107"
ORACLE_EXAMPLE_OBJECT = "asteroid_2514"  # DAMIT asteroid 2514, (3200) Phaethon
INTERPRETATION_EXAMPLE_OBJECT = "asteroid_121"  # DAMIT asteroid 121, (41) Daphne
CANDIDATE_SKY_SAMPLE = 25
CANDIDATE_SKY_SEED = 20260922
OUT = FIG_OUT

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
    }
)


def render_worked_example() -> None:
    """Show the method end to end on the real lightcurves of (9) Metis."""
    from matplotlib.patches import FancyArrowPatch

    catalog_path = FOLLOWUP / "source/repro/data/damit-20250610T000301Z/catalog.jsonl"
    record = next(
        json.loads(line)
        for line in catalog_path.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["object_id"] == METHOD_EXAMPLE_OBJECT
    )
    period_days = record["solutions"][0]["period_hours"] / 24.0
    if ARGS.periods != "damit":
        period_days = json.loads(Path(ARGS.periods).read_text())["periods_hours"][METHOD_EXAMPLE_OBJECT] / 24.0
    epochs = read_damit_lightcurves(
        DAMIT_DUMP / "files" / METHOD_EXAMPLE_OBJECT / "lc.txt"
    )
    reference_time = min(epoch[:, 0].min() for epoch in epochs)

    ensemble = load_npz(
        ENSEMBLE
    )
    position = int(np.flatnonzero(ensemble["object_ids"] == METHOD_EXAMPLE_OBJECT)[0])
    scores = ensemble["score_grids"][position].astype(float)
    candidates = np.asarray(ensemble["refined_axes"][position], dtype=float)
    axes_xyz = axial_grid_vectors()

    fig = plt.figure(figsize=(7.2, 5.3))
    layout = fig.add_gridspec(
        2, 2, hspace=0.62, wspace=0.42,
        left=0.085, right=0.955, top=0.915, bottom=0.085,
    )
    panel_a = fig.add_subplot(layout[0, 0])
    panel_b = fig.add_subplot(layout[0, 1])
    panel_d = fig.add_subplot(layout[1, 0])
    panel_c = fig.add_subplot(layout[1, 1], projection="mollweide")

    # (a) every lightcurve of the asteroid, on its true time axis.
    colours = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, len(epochs)))
    for epoch, colour in zip(epochs, colours, strict=True):
        years = (epoch[:, 0] - reference_time) / 365.25
        panel_a.scatter(years, epoch[:, 1] / np.mean(epoch[:, 1]), s=1.4,
                        color=colour, edgecolors="none")
    panel_a.set_xlabel("Years since the first observation")
    panel_a.set_ylabel("Relative brightness")
    panel_a.set_title(f"(a) {len(epochs)} lightcurves", fontsize=9.5)
    panel_a.tick_params(labelsize=8)

    # (b) one of them folded with the supplied period and binned.
    chosen, best_coverage = epochs[0], -1
    for epoch in epochs:
        phase = ((epoch[:, 0] - reference_time) / period_days) % 1.0
        coverage = len(np.unique((phase * 64).astype(int)))
        if 40 <= len(epoch) <= 150 and coverage > best_coverage:
            chosen, best_coverage = epoch, coverage
    phase = ((chosen[:, 0] - reference_time) / period_days) % 1.0
    flux = chosen[:, 1] / np.mean(chosen[:, 1])
    assignment = (phase * 64).astype(int)
    centres = (np.arange(64) + 0.5) / 64.0
    means = np.array([
        flux[assignment == index].mean() if np.any(assignment == index) else np.nan
        for index in range(64)
    ])
    panel_b.plot(phase, flux, "o", ms=2.6, color="#2878B5", alpha=0.8,
                 label="measurements")
    panel_b.step(centres, means, where="mid", color="#D95F02", lw=1.5,
                 label="64 phase bins")
    panel_b.set_xlabel("Rotational phase")
    panel_b.set_ylabel("Relative brightness")
    panel_b.set_title("(b) One lightcurve folded", fontsize=9.5)
    panel_b.tick_params(labelsize=8)
    panel_b.legend(loc="lower left", frameon=False, fontsize=7.5)

    # (c) the score of every direction, from the five averaged networks.
    sphere = np.vstack([axes_xyz, -axes_xyz])
    sphere_scores = np.concatenate([scores, scores])
    longitude, latitude = ecliptic_angles(sphere)
    order = np.argsort(sphere_scores)
    mapped = panel_c.scatter(longitude[order], latitude[order],
                             c=sphere_scores[order], s=2.2, cmap="viridis",
                             linewidths=0)
    candidate_lon, candidate_lat = ecliptic_angles(np.vstack([candidates,
                                                              -candidates]))
    panel_c.scatter(candidate_lon, candidate_lat, s=70, marker="*",
                    facecolor="none", edgecolor="#D95F02", linewidths=1.2,
                    zorder=5)
    panel_c.grid(True, lw=0.3, color="#999999", alpha=0.3)
    panel_c.set_xticklabels([])
    panel_c.set_yticks(np.radians((-60, -30, 0, 30, 60)))
    panel_c.tick_params(labelsize=6.5)
    panel_c.set_title("(c) Score of 6,144 directions", fontsize=9.5, pad=10)
    bar = fig.colorbar(mapped, ax=panel_c, fraction=0.028, pad=0.03)
    bar.set_label("Score", fontsize=8)
    bar.ax.tick_params(labelsize=7)

    # (d) what the method hands to the classical inversion.
    pole_colours = ("#CC3311", "#0072B2", "#009988")
    for index, axis in enumerate(candidates):
        for sign, marker in ((1.0, "o"), (-1.0, "s")):
            vector = sign * axis
            longitude_deg = np.degrees(np.arctan2(vector[1], vector[0])) % 360.0
            latitude_deg = np.degrees(np.arcsin(np.clip(vector[2], -1.0, 1.0)))
            panel_d.scatter(longitude_deg, latitude_deg, s=46, marker=marker,
                            color=pole_colours[index], zorder=4,
                            edgecolors="white", linewidths=0.6)
            align = "center"
            nudge = 0.0
            if longitude_deg < 40.0:
                align, nudge = "left", 8.0
            elif longitude_deg > 320.0:
                align, nudge = "right", -8.0
            panel_d.annotate(
                f"{longitude_deg:.0f}, {latitude_deg:+.0f}",
                (longitude_deg, latitude_deg), textcoords="offset points",
                xytext=(nudge, 10 if sign > 0 else -15), ha=align, fontsize=7,
                color="#444444",
            )
    panel_d.set_xlim(-55, 415)
    panel_d.set_ylim(-90, 90)
    panel_d.set_xticks((0, 90, 180, 270, 360))
    panel_d.set_yticks((-90, -45, 0, 45, 90))
    panel_d.set_xlabel("Ecliptic longitude (degrees)")
    panel_d.set_ylabel("Ecliptic latitude (degrees)")
    panel_d.set_title("(d) Three axes, six starting poles", fontsize=9.5)
    panel_d.tick_params(labelsize=8)
    panel_d.grid(alpha=0.2)

    # Arrows follow the panels clockwise, so no arrow crosses the figure.
    def connect(start_axes, end_axes, start_point, end_point, label, offset):
        start = start_axes.get_position()
        end = end_axes.get_position()
        from_xy = (start.x0 + start_point[0] * start.width,
                   start.y0 + start_point[1] * start.height)
        to_xy = (end.x0 + end_point[0] * end.width,
                 end.y0 + end_point[1] * end.height)
        fig.add_artist(FancyArrowPatch(
            from_xy, to_xy, transform=fig.transFigure, arrowstyle="-|>",
            mutation_scale=14, lw=1.6, color="#555555",
        ))
        fig.text(0.5 * (from_xy[0] + to_xy[0]) + offset[0],
                 0.5 * (from_xy[1] + to_xy[1]) + offset[1], label,
                 ha="center", va="center", fontsize=8, fontstyle="italic",
                 color="#555555")

    connect(panel_a, panel_b, (1.05, 0.62), (-0.26, 0.62), "fold", (0.0, 0.026))
    connect(panel_b, panel_c, (0.5, -0.30), (0.5, 1.22), "score", (0.040, 0.0))
    connect(panel_c, panel_d, (-0.14, 0.62), (1.16, 0.62), "select", (0.0, 0.026))

    fig.savefig(FIG_OUT / "worked-example.pdf", dpi=300,
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def render_oracle_error() -> None:
    """Show, on one real asteroid, how the oracle error is defined."""
    from matplotlib.patches import Circle as _Circle

    catalog_path = FOLLOWUP / "source/repro/data/damit-20250610T000301Z/catalog.jsonl"
    record = next(
        json.loads(line)
        for line in catalog_path.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["object_id"] == ORACLE_EXAMPLE_OBJECT
    )
    reference = np.asarray(record["solutions"][0]["vector"], dtype=float)
    reference /= np.linalg.norm(reference)

    ensemble = load_npz(
        ENSEMBLE
    )
    position = int(np.flatnonzero(ensemble["object_ids"] == ORACLE_EXAMPLE_OBJECT)[0])
    candidates = np.asarray(ensemble["refined_axes"][position], dtype=float)
    candidates /= np.linalg.norm(candidates, axis=1, keepdims=True)

    # Orthographic view, chosen so that the interesting directions face us.
    near_side = np.array([axis if np.dot(axis, reference) > 0 else -axis
                          for axis in candidates])
    centre = reference + near_side.sum(axis=0)
    centre /= np.linalg.norm(centre)
    view_lon = np.arctan2(centre[1], centre[0])
    view = centre
    east = np.array([-np.sin(view_lon), np.cos(view_lon), 0.0])
    north = np.cross(view, east)

    def project(vectors):
        vectors = np.atleast_2d(vectors)
        return (vectors @ east, vectors @ north, vectors @ view)

    def great_circle(start, end, steps=90):
        angle = np.arccos(np.clip(np.dot(start, end), -1.0, 1.0))
        fractions = np.linspace(0.0, 1.0, steps)[:, None]
        points = (np.sin((1.0 - fractions) * angle) * start
                  + np.sin(fractions * angle) * end) / np.sin(angle)
        return points / np.linalg.norm(points, axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(_Circle((0, 0), 1.0, facecolor="#FCFCFD", edgecolor="#AAAAAA",
                         lw=1.0, zorder=0))

    # Graticule.
    for latitude in np.radians(np.arange(-60, 61, 30)):
        longitudes = np.radians(np.linspace(0, 360, 361))
        line = np.column_stack([np.cos(latitude) * np.cos(longitudes),
                                np.cos(latitude) * np.sin(longitudes),
                                np.full_like(longitudes, np.sin(latitude))])
        x, y, front = project(line)
        x, y = np.where(front > 0, x, np.nan), np.where(front > 0, y, np.nan)
        ax.plot(x, y, color="#CCCCCC", lw=0.5, zorder=1)
    for longitude in np.radians(np.arange(0, 360, 30)):
        latitudes = np.radians(np.linspace(-90, 90, 181))
        line = np.column_stack([np.cos(latitudes) * np.cos(longitude),
                                np.cos(latitudes) * np.sin(longitude),
                                np.sin(latitudes)])
        x, y, front = project(line)
        x, y = np.where(front > 0, x, np.nan), np.where(front > 0, y, np.nan)
        ax.plot(x, y, color="#CCCCCC", lw=0.5, zorder=1)

    # The grid of trial directions, both ends of every axis.
    grid = axial_grid_vectors()
    grid = np.vstack([grid, -grid])
    x, y, front = project(grid)
    visible = front > 0.02
    ax.scatter(x[visible][::8], y[visible][::8], s=2.2, color="#7F8B98",
               alpha=0.55, linewidths=0, zorder=2)

    # The three candidate axes, drawn through the centre of the sphere.
    colours = ("#CC3311", "#0072B2", "#009988")
    arcs = []
    for index, axis in enumerate(candidates):
        front_end = axis if np.dot(axis, reference) > 0 else -axis
        angle = np.degrees(np.arccos(np.clip(np.dot(front_end, reference),
                                             -1.0, 1.0)))
        arcs.append((angle, front_end, index))
        x_end, y_end, _ = project(front_end)
        ax.plot([-x_end[0], x_end[0]], [-y_end[0], y_end[0]],
                color=colours[index], lw=0.8, ls=(0, (1.2, 2.2)), alpha=0.40,
                zorder=3)
        for sign, filled in ((1.0, True), (-1.0, False)):
            x_p, y_p, _ = project(sign * front_end)
            ax.scatter(x_p, y_p, s=76, marker="*", zorder=6,
                       color=colours[index] if filled else "white",
                       edgecolors=colours[index], linewidths=1.0)

    # The reference pole and the angle from each candidate to it.
    best = min(arcs, key=lambda item: item[0])
    for angle, end, index in arcs:
        arc = great_circle(end, reference)
        x_arc, y_arc, _ = project(arc)
        is_best = index == best[2]
        ax.plot(x_arc, y_arc, color="#111111" if is_best else colours[index],
                lw=2.0 if is_best else 1.2, ls="-",
                alpha=1.0 if is_best else 0.75, zorder=5)
        middle = int(len(arc) * (0.5 if is_best else 0.34))
        x_mid, y_mid = x_arc[middle], y_arc[middle]
        step_x = x_arc[middle + 1] - x_arc[middle - 1]
        step_y = y_arc[middle + 1] - y_arc[middle - 1]
        length = np.hypot(step_x, step_y) or 1.0
        reach = 13.0 if is_best else 16.0
        offset = (-step_y / length * reach, step_x / length * reach)
        text = f"{angle:.1f}$^\\circ$" if is_best else f"{angle:.0f}$^\\circ$"
        ax.annotate(text, (x_mid, y_mid),
                    textcoords="offset points", xytext=offset,
                    ha="center", va="center",
                    fontsize=9 if is_best else 8,
                    color="#111111" if is_best else colours[index],
                    weight="bold" if is_best else "normal", zorder=8,
                    bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                              edgecolor="none", alpha=0.85))

    x_ref, y_ref, _ = project(reference)
    ax.scatter(x_ref, y_ref, s=58, marker="o", color="black", zorder=7,
               edgecolors="white", linewidths=0.9)
    ax.text(0.0, -1.19, "Oracle error", ha="center", fontsize=9.5,
            color="#111111")

    handles = [
        plt.Line2D([], [], marker="*", color="none", markerfacecolor="#CC3311",
                   markeredgecolor="#CC3311", markersize=11,
                   label="Candidate axis"),
        plt.Line2D([], [], marker="o", color="none", markerfacecolor="black",
                   markeredgecolor="black", markersize=6,
                   label="Reference pole"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.04, 1.00),
              frameon=False, fontsize=8.5, handletextpad=0.4,
              labelspacing=0.35)
    ax.set_xlim(-1.22, 1.22)
    ax.set_ylim(-1.27, 1.13)
    fig.savefig(FIG_OUT / "oracle-error.pdf", dpi=300,
                bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)


def read_damit_lightcurves(path: Path) -> list[np.ndarray]:
    """Read the epochs of one DAMIT lightcurve file."""
    lines = path.read_text(encoding="utf-8").split("\n")
    index = 0
    epoch_count = int(lines[index].split()[0])
    index += 1
    epochs: list[np.ndarray] = []
    for _ in range(epoch_count):
        point_count = int(lines[index].split()[0])
        index += 1
        rows = []
        for _ in range(point_count):
            parts = lines[index].split()
            index += 1
            rows.append([float(value) for value in parts[:8]])
        epochs.append(np.asarray(rows, dtype=float))
    return epochs


def axial_grid_vectors() -> np.ndarray:
    """Rebuild the 6,144 representative axes of the frozen HEALPix grid."""
    import astropy.units as u
    from astropy_healpix import HEALPix

    healpix = HEALPix(nside=32, order="nested")
    longitude, latitude = healpix.healpix_to_lonlat(np.arange(12 * 32 * 32))
    longitude = longitude.to_value(u.rad)
    latitude = latitude.to_value(u.rad)
    centers = np.stack(
        [
            np.cos(latitude) * np.cos(longitude),
            np.cos(latitude) * np.sin(longitude),
            np.sin(latitude),
        ],
        axis=1,
    )
    tolerance = 1e-14
    keep = (centers[:, 2] > tolerance) | (
        (np.abs(centers[:, 2]) <= tolerance)
        & (
            (centers[:, 1] > tolerance)
            | ((np.abs(centers[:, 1]) <= tolerance) & (centers[:, 0] >= 0))
        )
    )
    return centers[keep]


def ecliptic_angles(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vectors = np.asarray(vectors, dtype=float)
    return (
        np.arctan2(vectors[:, 1], vectors[:, 0]),
        np.arcsin(np.clip(vectors[:, 2], -1.0, 1.0)),
    )


def empirical_cdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(values, dtype=float))
    y = np.arange(1, x.size + 1) / x.size
    return x, y


def render_candidate_sky() -> None:
    """Each reference pole and the closest of the three candidates."""
    import sys

    sys.path.insert(0, str(FOLLOWUP / "source/repro"))
    from compute_reference_baselines import reference_poles

    ensemble = load_npz(
        ENSEMBLE
    )
    poles = reference_poles()
    identifiers = [str(value) for value in ensemble["object_ids"]]

    def angles(vector):
        longitude = np.degrees(np.arctan2(vector[1], vector[0]))
        latitude = np.degrees(np.arcsin(np.clip(vector[2], -1.0, 1.0)))
        return (longitude + 180.0) % 360.0 - 180.0, latitude

    reference_longitude, reference_latitude = [], []
    candidate_longitude, candidate_latitude, errors = [], [], []
    for position, name in enumerate(identifiers):
        candidates = np.asarray(ensemble["refined_axes"][position], dtype=float)
        candidates /= np.linalg.norm(candidates, axis=1, keepdims=True)
        best = None
        for candidate in candidates:
            for sign in (1.0, -1.0):
                for pole in poles[name]:
                    angle = np.degrees(
                        np.arccos(np.clip(np.dot(sign * candidate, pole), -1.0, 1.0))
                    )
                    if best is None or angle < best[0]:
                        best = (angle, sign * candidate, pole)
        angle, candidate, pole = best
        pole_longitude, pole_latitude = angles(pole)
        axis_longitude, axis_latitude = angles(candidate)
        # Keep the drawn line on the short path across the map.
        if axis_longitude - pole_longitude > 180.0:
            axis_longitude -= 360.0
        elif axis_longitude - pole_longitude < -180.0:
            axis_longitude += 360.0
        reference_longitude.append(pole_longitude)
        reference_latitude.append(pole_latitude)
        candidate_longitude.append(axis_longitude)
        candidate_latitude.append(axis_latitude)
        errors.append(angle)

    errors = np.array(errors)
    reference_longitude = np.radians(np.array(reference_longitude))
    reference_latitude = np.radians(np.array(reference_latitude))
    candidate_longitude = np.radians(np.array(candidate_longitude))
    candidate_latitude = np.radians(np.array(candidate_latitude))

    # Drawing all 170 pairs crowds the map, so a fixed random subset is shown.
    drawn = np.sort(
        np.random.default_rng(CANDIDATE_SKY_SEED).choice(
            errors.size, size=CANDIDATE_SKY_SAMPLE, replace=False
        )
    )
    errors = errors[drawn]
    reference_longitude = reference_longitude[drawn]
    reference_latitude = reference_latitude[drawn]
    candidate_longitude = candidate_longitude[drawn]
    candidate_latitude = candidate_latitude[drawn]

    colours = plt.cm.viridis_r
    norm = plt.Normalize(vmin=0.0, vmax=40.0)
    fig = plt.figure(figsize=(7.2, 4.4))
    ax = fig.add_subplot(111, projection="mollweide")
    for index in range(errors.size):
        ax.plot(
            [candidate_longitude[index], reference_longitude[index]],
            [candidate_latitude[index], reference_latitude[index]],
            color=colours(norm(errors[index])), lw=1.2, alpha=0.8, zorder=3,
        )
    ax.scatter(reference_longitude, reference_latitude, color="black", s=26,
               alpha=0.9, edgecolors="none", zorder=4)
    mapped = ax.scatter(candidate_longitude, candidate_latitude, c=errors,
                        cmap=colours, norm=norm, s=28, marker="D", alpha=0.95,
                        edgecolors="k", linewidths=0.4, zorder=5)
    ax.grid(True, alpha=0.25, lw=0.4)
    ax.set_xticklabels([])
    ax.tick_params(labelsize=7.5)
    bar = fig.colorbar(mapped, ax=ax, orientation="horizontal", fraction=0.045,
                       pad=0.06, aspect=38)
    bar.set_label("Oracle error (degrees)", fontsize=9)
    bar.ax.tick_params(labelsize=8)
    handles = [
        plt.Line2D([], [], marker="o", color="none", markerfacecolor="black",
                   markeredgecolor="none", markersize=6, label="DAMIT reference pole"),
        plt.Line2D([], [], marker="D", color="none", markerfacecolor="#4C9A6B",
                   markeredgecolor="k", markeredgewidth=0.3, markersize=6,
                   label="Closest candidate"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.16),
              ncol=2, frameon=False, fontsize=9)
    fig.savefig(FIG_OUT / "candidate-sky.pdf", dpi=300,
                bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.close(fig)


def render_interpretation() -> None:
    """Score maps against the amplitude-aspect relation and the mirror symmetry."""
    import json

    interpretation = ANALYSES
    amplitude = ANALYSES
    ensemble = load_npz(
        ENSEMBLE
    )
    physical = load_npz(amplitude / "amplitude-aspect.npz")
    comparison = json.loads((amplitude / "amplitude-aspect.json").read_text())
    symmetry = json.loads((interpretation / "mirror-symmetry.json").read_text())["objects"]
    names = [str(value) for value in ensemble["object_ids"]]
    grid = axial_grid_vectors()

    example = INTERPRETATION_EXAMPLE_OBJECT
    delphi_map = ensemble["score_grids"][names.index(example)].astype(float)
    physical_names = [str(value) for value in physical["object_ids"]]
    physical_map = physical["amplitude_aspect_maps"][physical_names.index(example)].astype(float)
    reference = next(row for row in symmetry if row["object_id"] == example)
    pole_longitude = np.radians((reference["reference_longitude_deg"] + 180.0) % 360.0 - 180.0)
    pole_latitude = np.radians(reference["reference_latitude_deg"])

    fig = plt.figure(figsize=(7.1, 4.7))
    outer = fig.add_gridspec(2, 2, height_ratios=(0.66, 1.05), hspace=0.16, wspace=0.28)
    both = np.r_[grid, -grid]
    longitude, latitude = ecliptic_angles(both)
    for column, (values, title) in enumerate(
        ((delphi_map, "(a) DeLPHI score"), (physical_map, "(b) Amplitude-aspect fit"))
    ):
        ax = fig.add_subplot(outer[0, column], projection="mollweide")
        z = (values - values.mean()) / values.std()
        mapped = ax.scatter(longitude, latitude, c=np.r_[z, z], cmap="viridis", s=4.5,
                            marker="o", edgecolors="none", rasterized=True,
                            vmin=-2.5, vmax=2.5)
        for sign in (1.0, -1.0):
            lon = pole_longitude if sign > 0 else (pole_longitude + 2 * np.pi) % (2 * np.pi) - np.pi
            ax.scatter([lon], [sign * pole_latitude], s=46, facecolors="none",
                       edgecolors="white", linewidths=1.6, zorder=5)
            ax.scatter([lon], [sign * pole_latitude], s=46, facecolors="none",
                       edgecolors="black", linewidths=0.8, zorder=6)
        ax.set_xticklabels([])
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3, lw=0.4)
        ax.set_title(title, fontsize=9, pad=6)
    bar = fig.colorbar(mapped, ax=fig.axes[:2], orientation="vertical", fraction=0.025,
                       pad=0.02, shrink=0.8)
    bar.set_label("Standardized score", fontsize=7.5)
    bar.ax.tick_params(labelsize=7)

    ax = fig.add_subplot(outer[1, 0])
    latitude_values = np.array([row["observer_abs_latitude_deg"] for row in symmetry])
    correlation = np.array([row["mirror_correlation"] for row in symmetry])
    ax.scatter(latitude_values, correlation, s=14, color="#2878B5", alpha=0.75,
               edgecolors="white", linewidths=0.4)
    ax.set_xscale("log")
    ax.set_xlim(0.15, 60)
    ax.set_ylim(-0.3, 1.02)
    ax.set_xticks([0.3, 1, 3, 10, 30])
    ax.set_xticklabels(["0.3", "1", "3", "10", "30"])
    ax.set_xlabel("Mean ecliptic latitude of observer (degrees)", fontsize=8)
    ax.set_ylabel("Mirror correlation of score map", fontsize=8)
    ax.tick_params(labelsize=7.5)
    ax.grid(alpha=0.2)
    ax.set_title("(c)", fontsize=9, loc="left")

    ax = fig.add_subplot(outer[1, 1])
    agreement = np.array([row["map_correlation"] for row in comparison])
    delphi = np.array([row["delphi_oracle_deg"] for row in comparison])
    classical = np.array([row["amplitude_aspect_oracle_deg"] for row in comparison])
    edges = (-1.0, 0.5, 0.7, 0.8, 1.0)
    labels = ("< 0.5", "0.5-0.7", "0.7-0.8", "> 0.8")
    centres = np.arange(len(labels))
    for values, name, colour, marker, shift in (
        (delphi, "DeLPHI", "#2878B5", "o", -0.08),
        (classical, "Amplitude-aspect", "#D95F02", "s", 0.08),
    ):
        means, lows, highs = [], [], []
        generator = np.random.default_rng(20260923)
        for low, high in zip(edges[:-1], edges[1:]):
            chosen = values[(agreement >= low) & (agreement < high)]
            draws = [generator.choice(chosen, chosen.size).mean() for _ in range(2000)]
            means.append(chosen.mean())
            lows.append(chosen.mean() - np.percentile(draws, 2.5))
            highs.append(np.percentile(draws, 97.5) - chosen.mean())
        ax.errorbar(centres + shift, means, yerr=[lows, highs], fmt=marker, ms=5,
                    color=colour, lw=1.0, capsize=2, label=name)
    ax.set_xticks(centres)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_xlabel("Correlation between the two score maps", fontsize=8)
    ax.set_ylabel("Mean oracle error (degrees)", fontsize=8)
    ax.set_ylim(0, 45)
    ax.tick_params(labelsize=7.5)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax.set_title("(d)", fontsize=9, loc="left")
    fig.savefig(FIG_OUT / "interpretation.pdf", dpi=300,
                bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.close(fig)


def render_candidate_count() -> None:
    """Oracle error and searched area against the number of candidates."""
    import json

    summary = json.loads(
        (ANALYSES / "candidate-count.json").read_text()
    )
    rows = summary["k"]
    k = np.array([row["k"] for row in rows])
    mean = np.array([row["mean_error_deg"] for row in rows])
    median = np.array([row["median_error_deg"] for row in rows])
    area_mean = 100 * np.array([row["search_area_mean_radius"] for row in rows])
    area_p90 = 100 * np.array([row["search_area_p90_radius"] for row in rows])
    standard = summary["standard_poles"]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9))
    ax = axes[0]
    ax.plot(k, mean, "o-", color="#2878B5", lw=1.6, ms=5, label="Mean")
    ax.plot(k, median, "s--", color="#2878B5", lw=1.2, ms=4, alpha=0.8, label="Median")
    ax.axhline(standard["mean_error_deg"], color="#D95F02", lw=1.2, ls=":",
               label="Standard poles, mean")
    ax.set_xlabel("Number of candidate axes")
    ax.set_ylabel("Oracle error (degrees)")
    ax.set_xticks(k)
    ax.set_ylim(0, 32)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    ax.set_title("(a)", fontsize=9, loc="left")
    ax = axes[1]
    ax.plot(k, area_mean, "o-", color="#2878B5", lw=1.6, ms=5, label="Radius = mean error")
    ax.plot(k, area_p90, "s--", color="#2878B5", lw=1.2, ms=4, alpha=0.8,
            label="Radius = 90th percentile")
    ax.axhline(100 * standard["search_area_mean_radius"], color="#D95F02", lw=1.2, ls=":",
               label="Standard poles, mean")
    ax.axhline(100 * standard["search_area_p90_radius"], color="#D95F02", lw=1.0, ls="-.",
               label="Standard poles, 90th pct.")
    ax.set_xlabel("Number of candidate axes")
    ax.set_ylabel("Axis orientations covered (%)")
    ax.set_xticks(k)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=7, loc="center right", bbox_to_anchor=(1.0, 0.6))
    ax.set_title("(b)", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(FIG_OUT / "candidate-count.pdf", dpi=300, facecolor="white")
    plt.close(fig)


def render_reference_reassignment() -> None:
    """Right panel of Figure 15, from this run's reference reassignment."""
    summary = json.loads((ANALYSES / "reference-reassignment.json").read_text())
    fig, ax = plt.subplots(figsize=(3.3, 3.2))
    ax.hist(summary["null_draws_mean_error_deg"], bins=30, color="#AAAAAA")
    observed = summary["observed_mean_error_deg"]
    ax.axvline(observed, color="black", lw=1.6, ls="--")
    ax.set_xlim(10, max(summary["null_draws_mean_error_deg"]) + 2)
    ax.text(observed + 0.8, ax.get_ylim()[1] * 0.9, "Actual\nreference\npoles",
            fontsize=7, color="black", va="top")
    ax.set_xlabel("Mean oracle error (degrees)", fontsize=8)
    ax.set_ylabel("Number of reassignments", fontsize=8)
    ax.tick_params(labelsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG_OUT / "reference-reassignment.pdf", dpi=300, facecolor="white")
    plt.close(fig)


def render_latitude() -> None:
    """Mean oracle error against the absolute ecliptic latitude of the reference pole."""
    import csv
    import json
    import sys
    from collections import defaultdict

    sys.path.insert(0, str(FOLLOWUP / "source/repro"))
    import compute_reference_baselines as base

    main = json.loads((ANALYSES / "mirror-symmetry.json").read_text())["objects"]
    main_lat = np.array([abs(row["reference_latitude_deg"]) for row in main])
    main_err = np.array([row["oracle_error_deg"] for row in main])
    solutions = defaultdict(list)
    with (ROOT.parent / "damit-20250610T000301Z/tables/asteroid_models.csv").open() as stream:
        for row in csv.DictReader(stream):
            try:
                solutions[int(row["asteroid_id"])].append((float(row["lambda"]), float(row["beta"])))
            except ValueError:
                continue
    standard = base.standard_axes()
    lowq = json.loads((FOLLOWUP / "data/publication-revision-20260917-r2/lowq-analysis.json").read_text())["objects"]
    low_lat, low_err, low_std = [], [], []
    for entry in lowq:
        listed = solutions[int(entry["object_id"].split(":")[1])]
        references = np.array([base.unit_from_degrees(longitude, b) for longitude, b in listed])
        low_lat.append(abs(listed[0][1]))
        low_err.append(float(entry["oracle_at_3_error_deg"]))
        low_std.append(base.axial_error_deg(standard, references))
    low_lat, low_err, low_std = map(np.array, (low_lat, low_err, low_std))
    edges = np.arange(0, 91, 15)
    centres = edges[:-1] + 7.5
    generator = np.random.default_rng(20260929)

    def binned(lat, err):
        means, lows, highs = [], [], []
        for low, high in zip(edges[:-1], edges[1:]):
            chosen = err[(lat >= low) & ((lat < high) if high < 90 else (lat <= high))]
            draws = generator.choice(chosen, (2000, chosen.size)).mean(axis=1)
            means.append(chosen.mean())
            lows.append(chosen.mean() - np.percentile(draws, 2.5))
            highs.append(np.percentile(draws, 97.5) - chosen.mean())
        return np.array(means), np.array([lows, highs])

    fig, ax = plt.subplots(figsize=(5.3, 3.4))
    # Okabe-Ito colours plus distinct line styles and markers, so the series stay
    # distinguishable under colour-vision deficiency.
    for lat, err, label, colour, marker, style, shift in (
        (main_lat, main_err, "DeLPHI, 170 test asteroids (quality 3 or higher)", "#0072B2", "o", "-", -1.2),
        (low_lat, low_err, "DeLPHI, 9,783 lower-quality models", "#333333", "s", "--", 1.2),
    ):
        means, yerr = binned(lat, err)
        ax.errorbar(centres + shift, means, yerr=yerr, fmt=marker, ls=style, color=colour, ms=5,
                    lw=1.3, capsize=2, label=label)
    means, _ = binned(low_lat, low_std)
    ax.plot(centres, means, ":", color="#E69F00", lw=1.8, label="Standard poles, lower-quality models")
    ax.set_xticks(edges)
    ax.set_xlim(0, 90)
    ax.set_ylim(0, 45)
    ax.set_xlabel("Absolute ecliptic latitude of the reference pole (degrees)")
    ax.set_ylabel("Mean oracle error (degrees)")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_OUT / "latitude.pdf", dpi=300, facecolor="white")
    plt.close(fig)


def _sampling_rows():
    """Rows of the frozen reliability table (the evidence release, part A)."""
    import csv

    path = FOLLOWUP / "data/reliability-20260915/run-1/sampling-table.csv"
    with path.open() as stream:
        return {row["condition"]: row for row in csv.DictReader(stream)}


def _standard_mean():
    """Mean oracle error of the six standard starting poles for the 170 asteroids."""
    baselines = load_npz(FOLLOWUP / "source/repro/baselines_out/reference-baselines-per-object.npz")
    return float(baselines["standard_errors_deg"].mean())


def render_period_sensitivity() -> None:
    """Mean oracle error when the supplied period is scaled (Appendix C)."""
    rows = _sampling_rows()
    factors = ("0.5", "0.99", "0.999", "0.9999", None, "1.0001", "1.001", "1.01", "2.0")
    labels = ["0.5", "0.99", "0.999", "0.9999", "1", "1.0001", "1.001", "1.01", "2"]
    keys = ["full" if f is None else f"period-{f}" for f in factors]
    means = np.array([float(rows[k]["mean_error_deg"]) for k in keys])
    low = np.array([float(rows[k]["mean_error_bootstrap_lower_95"]) for k in keys])
    high = np.array([float(rows[k]["mean_error_bootstrap_upper_95"]) for k in keys])
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    ax.axhline(_standard_mean(), color="#E69F00", ls=":", lw=1.8, label="Six standard starting poles")
    ax.errorbar(x, means, yerr=[means - low, high - means], fmt="o", color="#0072B2", ms=5,
                capsize=3, label="DeLPHI, scaled period")
    centre = keys.index("full")
    ax.errorbar([centre], [means[centre]], yerr=[[means[centre] - low[centre]], [high[centre] - means[centre]]],
                fmt="s", color="black", ms=6, capsize=3, label="DeLPHI, DAMIT period")
    ax.set_xticks(x, labels)
    ax.set_xlabel("Supplied period divided by the DAMIT period")
    ax.set_ylabel("Mean oracle error (degrees)")
    ax.set_ylim(0, 30)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_OUT / "period-sensitivity.pdf", dpi=300, facecolor="white")
    plt.close(fig)


def render_coverage() -> None:
    baselines = load_npz(
        FOLLOWUP / "source/repro/baselines_out/reference-baselines-per-object.npz"
    )
    delphi_x, delphi_y = empirical_cdf(load_npz(ENSEMBLE)["oracle_errors_deg"])
    standard_x, standard_y = empirical_cdf(baselines["standard_errors_deg"])
    random_cdf = load_npz(FOLLOWUP / "data/interpretation-20260926/random-axes-cdf.npz")
    fig, ax = plt.subplots(figsize=(5.3, 3.4))
    ax.step(delphi_x, delphi_y, where="post", lw=2.0, label="DeLPHI",
            color="#2878B5")
    ax.step(standard_x, standard_y, where="post", lw=1.8,
            label="Standard poles", color="#D95F02")
    ax.fill_between(random_cdf["grid_deg"], random_cdf["low_cdf"], random_cdf["high_cdf"],
                    color="#BBBBBB", alpha=0.5, lw=0)
    ax.plot(random_cdf["grid_deg"], random_cdf["mean_cdf"], lw=1.5, ls=(0, (4, 2)),
            label="Random axes", color="#777777")
    for threshold in (20, 30):
        ax.axvline(threshold, color="#777777", lw=0.8, ls="--")
        ax.text(threshold + 0.8, 0.06, f"{threshold}°", color="#555555")
    ax.set_xlim(0, 90)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("Oracle error with respect to a DAMIT reference (degrees)")
    ax.set_ylabel("Fraction of asteroids")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG_OUT / "primary-coverage.pdf", dpi=300)
    plt.close(fig)


def main() -> None:
    network.guard_output(OUT)
    if "paper-rev-20261021" in OUT.resolve().parts:
        raise ValueError("figure output must be outside the manuscript directory")
    OUT.mkdir(parents=True, exist_ok=True)
    producers = {"worked-example": render_worked_example, "oracle-error": render_oracle_error,
                 "candidate-sky": render_candidate_sky, "interpretation": render_interpretation,
                 "candidate-count": render_candidate_count, "reference-reassignment": render_reference_reassignment,
                 "latitude": render_latitude, "primary-coverage": render_coverage}
    for name in ARGS.figures or producers:
        producers[name]()


if __name__ == "__main__":
    from repro.own_period.figure_data import capture_figures
    with capture_figures():
        main()
