"""Render the reader-facing figures used in the main DeLPHI paper.

The script reads only frozen publication evidence.  It does not recompute or
change any scientific result.

Own-period copy (lower-quality census): only ``render_latitude`` (Figure 11)
and ``render_transfer`` (Figure 7) are run.  Their lower-quality series come
from ``data/own-period-20261021/lowq/scores-own`` (variant ``at90`` or
``excluded``); every other series is the published one and is labelled so.
Output goes to ``data/own-period-20261021/lowq/figures/``.

    PYTHONPATH=<worktree> python repro/own_period/lowq_figures/render_main_figures.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, Rectangle

from repro import network_root as network

# Own-period copy: the paths of the published layout, resolved from this worktree.
from repro.own_period.reporting_inputs import DataPath, load_npz

FOLLOWUP = DataPath("../..")
ROOT = DataPath('lowq')
DAMIT_DUMP = FOLLOWUP.parent / "damit-20250610T000301Z"
SCORES = ROOT / "scores-own"
VARIANTS = {"at90": "no period counted at 90 deg", "excluded": "no period excluded"}
METHOD_EXAMPLE_OBJECT = "asteroid_107"
ORACLE_EXAMPLE_OBJECT = "asteroid_2514"  # DAMIT asteroid 2514, (3200) Phaethon
INTERPRETATION_EXAMPLE_OBJECT = "asteroid_121"  # DAMIT asteroid 121, (41) Daphne
CANDIDATE_SKY_SAMPLE = 25
CANDIDATE_SKY_SEED = 20260922
OUT = network.NetworkPath("results", "figures", "lowq", legacy_path=ROOT / "figures")

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


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / name, dpi=300)
    plt.close(fig)


def render_data_summary() -> None:
    path = ROOT / "k3-followup" / "supplement" / "k3_observing_structure.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    observations = np.array([float(row["observations"]) for row in rows])
    blocks = np.array([float(row["merged_blocks"]) for row in rows])
    years = np.array([float(row["baseline_years"]) for row in rows])

    # This figure is printed at full text width in the manuscript.  Use larger
    # labels than the other compact summary figures so that both panels remain
    # readable in the final PDF.
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.5))
    axes[0].scatter(observations, years, s=20, alpha=0.75, color="#2878B5")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Number of observations", fontsize=11)
    axes[0].set_ylabel("Time span (years)", fontsize=11)
    axes[0].set_title("Observations and time span", fontsize=12)
    bins = np.arange(0.5, blocks.max() + 1.5, 2)
    axes[1].hist(blocks, bins=bins, color="#5AAE61", edgecolor="white")
    axes[1].set_xlabel("30-day lightcurve groups", fontsize=11)
    axes[1].set_ylabel("Number of asteroids", fontsize=11)
    axes[1].set_title("Lightcurve groups per asteroid", fontsize=12)
    for ax in axes:
        ax.tick_params(labelsize=10)
    fig.tight_layout()
    save(fig, "delphi_data_summary.pdf")


def draw_rounded_box(ax, x, y, width, height, facecolor, edgecolor, linewidth=1.2,
                     zorder=2):
    """Rounded stage box centred on (x, y), following the earlier draft."""
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2), width, height,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth, zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def draw_arrow(ax, start, end, color="#555555", linewidth=1.8):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="->", color=color, lw=linewidth,
                                mutation_scale=12))


def draw_lightcurve_mini(ax, x, y, size=0.065):
    """Three short lightcurves seen under different viewing geometries."""
    colors = ("#0077BB", "#EE7733", "#009988")
    phase = np.linspace(0, 1, 60)
    for index, color in enumerate(colors):
        offset = size * (0.52 - 0.52 * index)
        curve = y + offset + size * 0.16 * np.sin(
            4 * np.pi * (phase + 0.18 * index)
        )
        ax.plot(x - size * 0.55 + size * 1.45 * phase, curve, color=color, lw=0.9,
                zorder=5)
        ax.scatter(x - size * 0.55 + size * 1.45 * phase[::9], curve[::9],
                   color=color, s=3.4, zorder=6, edgecolors="none")
        ax.scatter([x - size * 0.95], [y + offset], s=12, color="#AAAAAA", zorder=6)
        for angle, arrow_color in ((150 - 30 * index, "#E8A33D"),
                                   (-20 + 25 * index, "#2878B5")):
            theta = np.deg2rad(angle)
            ax.annotate(
                "",
                xy=(x - size * 0.95 + size * 0.34 * np.cos(theta),
                    y + offset + size * 0.34 * np.sin(theta)),
                xytext=(x - size * 0.95, y + offset),
                arrowprops=dict(arrowstyle="->", color=arrow_color, lw=0.7,
                                mutation_scale=6),
            )


def draw_folded_mini(ax, x, y, size=0.065):
    """A folded lightcurve above a strip of filled and empty phase bins."""
    phase = np.linspace(0, 1, 120)
    curve = (y + size * 0.42
             + size * 0.20 * np.sin(4 * np.pi * phase)
             + size * 0.06 * np.sin(8 * np.pi * phase))
    ax.plot(x - size * 0.75 + size * 1.5 * phase, curve, color="#BBBBBB", lw=0.8,
            zorder=4)
    sample = np.arange(2, 120, 5)
    ax.scatter(x - size * 0.75 + size * 1.5 * phase[sample], curve[sample],
               color="#2E8B57", s=3.4, zorder=5, edgecolors="none")
    occupied = np.zeros(32, dtype=bool)
    occupied[np.unique((phase[sample] * 32).astype(int))] = True
    for index in range(32):
        ax.add_patch(Rectangle(
            (x - size * 0.75 + size * 1.5 * index / 32.0, y - size * 0.62),
            size * 1.5 / 32.0 * 0.82, size * 0.30,
            facecolor="#2E8B57" if occupied[index] else "#DDDDDD",
            edgecolor="none", zorder=5,
        ))


def draw_scoremap_mini(ax, x, y, size=0.065):
    """A sky map of trial axes with three bright regions."""
    mesh_x, mesh_y = np.meshgrid(np.linspace(-1, 1, 180), np.linspace(-1, 1, 110))
    inside = (mesh_x ** 2 + (mesh_y / 0.52) ** 2) <= 1.0
    field = 0.25 * np.exp(-(mesh_y ** 2) / 0.30)
    for peak_x, peak_y in ((-0.55, 0.14), (0.02, -0.20), (0.58, 0.08)):
        field = field + np.exp(
            -(((mesh_x - peak_x) ** 2) / 0.040 + ((mesh_y - peak_y) ** 2) / 0.020)
        )
    ax.imshow(np.where(inside, field, np.nan), origin="lower", cmap="viridis",
              extent=(x - size * 0.95, x + size * 0.95,
                      y - size * 0.50, y + size * 0.50),
              aspect="auto", zorder=4)
    ax.add_patch(Ellipse((x, y), size * 1.90, size * 1.00, facecolor="none",
                         edgecolor="#888888", lw=0.6, zorder=5))


def draw_sphere_mini(ax, x, y, radius=0.042):
    """A sphere carrying three axes, each ending in two poles."""
    ax.add_patch(Circle((x, y), radius, facecolor="#FAFAFA", edgecolor="#666666",
                        linewidth=0.8, zorder=3))
    ax.add_patch(Ellipse((x, y), radius * 1.9, radius * 0.5, facecolor="none",
                         edgecolor="#CCCCCC", linewidth=0.5, linestyle="--",
                         zorder=4))
    for angle, color in ((28, "#CC3311"), (100, "#0077BB"), (156, "#009988")):
        theta = np.deg2rad(angle)
        dx, dy = radius * 0.86 * np.cos(theta), radius * 0.86 * np.sin(theta)
        ax.plot([x - dx, x + dx], [y - dy, y + dy], color=color, lw=1.0, zorder=5)
        ax.scatter([x - dx, x + dx], [y - dy, y + dy], color=color, s=9, zorder=6,
                   edgecolors="white", linewidths=0.3)


def draw_dome(ax, x, y, radius, dots=0, dot_seed=1, dot_color="#8899AA",
              face="#F7F9FB", edge="#888888", lw=0.9, zorder=3):
    """Half of the sphere of directions, drawn as a dome on its equator."""
    from matplotlib.patches import Wedge

    ax.add_patch(Wedge((x, y), radius, 0, 180, facecolor=face, edgecolor=edge,
                       linewidth=lw, zorder=zorder))
    ax.add_patch(Ellipse((x, y), 2 * radius, 0.46 * radius, facecolor="none",
                         edgecolor=edge, linewidth=lw, zorder=zorder + 1))
    if dots:
        rng = np.random.default_rng(dot_seed)
        angle = rng.uniform(0, np.pi, dots)
        spread = np.sqrt(rng.uniform(0, 1, dots))
        ax.scatter(x + radius * 0.93 * spread * np.cos(angle),
                   y + radius * 0.93 * spread * np.sin(angle) * 0.92,
                   s=0.9, color=dot_color, zorder=zorder + 2, edgecolors="none")


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
    epochs = read_damit_lightcurves(
        DAMIT_DUMP / "files" / METHOD_EXAMPLE_OBJECT / "lc.txt"
    )
    reference_time = min(epoch[:, 0].min() for epoch in epochs)

    ensemble = load_npz(
        FOLLOWUP
        / "inputs/frozen-artifacts/k3-definitive-7874092/evaluations/real-oof-ensemble.npz"
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

    fig.savefig(ROOT / "figures" / "worked-example.pdf", dpi=300,
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def render_architecture() -> None:
    """Draw the structure of DeLPHI as four lettered panels."""

    def box(ax, x, y, width, height, fill, edge, lw=1.0, zorder=2):
        patch = FancyBboxPatch(
            (x - width / 2, y - height / 2), width, height,
            boxstyle="round,pad=0.008,rounding_size=0.015",
            facecolor=fill, edgecolor=edge, linewidth=lw, zorder=zorder,
        )
        ax.add_patch(patch)
        return patch

    def arrow_down(ax, x, y_from, y_to, color="#555555", lw=1.2):
        ax.annotate("", xy=(x, y_to), xytext=(x, y_from),
                    arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                                    mutation_scale=10))

    def arrow_right(ax, x_from, x_to, y, color="#555555", lw=1.2):
        ax.annotate("", xy=(x_to, y), xytext=(x_from, y),
                    arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                                    mutation_scale=10))

    def panel_header(ax, letter, title):
        ax.text(0.01, 0.975, letter, fontsize=12.5, fontweight="bold", va="top")
        ax.text(0.062, 0.970, title, fontsize=9.5, va="top", color="#333333")

    def network_glyph(ax, x, y, size, color):
        """A small three-layer network."""
        columns = (x - size, x, x + size)
        counts = (2, 3, 2)
        nodes = []
        for centre_x, count in zip(columns, counts, strict=True):
            column = [(centre_x, y + size * (1.1 - 2.2 * index / (count - 1)))
                      for index in range(count)]
            nodes.append(column)
            for node_x, node_y in column:
                ax.add_patch(Circle((node_x, node_y), size * 0.30,
                                    facecolor="white", edgecolor=color, lw=0.6,
                                    zorder=5))
        for left, right in zip(nodes[:-1], nodes[1:], strict=True):
            for index, (x1, y1) in enumerate(left):
                for offset, (x2, y2) in enumerate(right):
                    ax.plot([x1, x2], [y1, y2], color=color, alpha=0.45,
                            lw=0.45, zorder=4)

    def globe(ax, x, y, radius, axes_angles=((28, "#CC3311"), (100, "#0072B2"),
                                             (156, "#009988"))):
        """A sphere with latitude and longitude lines and axes through it."""
        ax.add_patch(Circle((x, y), radius, facecolor="#F7F9FB",
                            edgecolor="#777777", lw=1.0, zorder=3))
        for fraction in (-0.62, -0.32, 0.0, 0.32, 0.62):
            half = radius * np.sqrt(max(1.0 - fraction ** 2, 0.0))
            ax.add_patch(Ellipse((x, y + radius * fraction), 2 * half,
                                 0.34 * radius, facecolor="none",
                                 edgecolor="#C8D2DA", lw=0.5, zorder=4))
        for width_fraction in (0.35, 0.75, 1.0):
            ax.add_patch(Ellipse((x, y), 2 * radius * width_fraction, 2 * radius,
                                 facecolor="none", edgecolor="#C8D2DA", lw=0.5,
                                 zorder=4))
        for angle, colour in axes_angles:
            theta = np.deg2rad(angle)
            dx, dy = radius * 0.92 * np.cos(theta), radius * 0.92 * np.sin(theta)
            ax.plot([x - dx, x + dx], [y - dy, y + dy], color=colour, lw=1.3,
                    ls=(0, (2.2, 1.6)), alpha=0.7, zorder=6)
            ax.scatter([x + dx], [y + dy], color=colour, s=17, zorder=7,
                       edgecolors="white", linewidths=0.4)
            ax.scatter([x - dx], [y - dy], color=colour, s=17, zorder=5,
                       alpha=0.55, edgecolors="white", linewidths=0.4)

    def score_map(ax, x, y, half_width, half_height, seed=0, stars=False,
                  alpha=1.0, zorder=4):
        mesh_x, mesh_y = np.meshgrid(np.linspace(-1, 1, 200),
                                     np.linspace(-1, 1, 120))
        inside = (mesh_x ** 2 + (mesh_y / 0.55) ** 2) <= 1.0
        rng = np.random.default_rng(seed)
        field = 0.30 * np.exp(-(mesh_y ** 2) / 0.28)
        peaks = ((-0.56, 0.14), (0.0, -0.20), (0.58, 0.10))
        for peak_x, peak_y in peaks:
            jitter_x = peak_x + rng.normal(0, 0.05)
            jitter_y = peak_y + rng.normal(0, 0.04)
            field = field + np.exp(
                -(((mesh_x - jitter_x) ** 2) / 0.05
                  + ((mesh_y - jitter_y) ** 2) / 0.022)
            )
        ax.imshow(np.where(inside, field, np.nan), origin="lower", cmap="viridis",
                  extent=(x - half_width, x + half_width,
                          y - half_height, y + half_height),
                  aspect="auto", zorder=zorder, alpha=alpha)
        ax.add_patch(Ellipse((x, y), 2 * half_width, 2 * half_height,
                             facecolor="none", edgecolor="#888888", lw=0.6,
                             zorder=zorder + 1))
        if stars:
            for peak_x, peak_y in peaks:
                ax.plot(x + half_width * peak_x, y + half_height * peak_y / 0.55,
                        marker="*", ms=8, markerfacecolor="none",
                        markeredgecolor="white", markeredgewidth=1.1,
                        zorder=zorder + 2)

    blue_edge, blue_fill = "#1976D2", "#E6F1FA"
    green_edge, green_fill = "#388E3C", "#E7F3EA"
    orange_edge, orange_fill = "#E65100", "#FFF3E0"
    purple_edge, purple_fill = "#7B1FA2", "#F3E5F5"

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 7.0))
    for ax in axes.ravel():
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    # Panel A: what one lightcurve supplies.
    ax = axes[0, 0]
    panel_header(ax, "A", "Input from one lightcurve")
    phase = np.linspace(0, 1, 110)
    folded = 0.826 + 0.032 * np.sin(4 * np.pi * phase)
    ax.plot(0.17 + 0.66 * phase, folded, color="#BBBBBB", lw=0.9, zorder=3)
    ax.scatter(0.17 + 0.66 * phase[::4], folded[::4], s=3.4, color="#2E8B57",
               zorder=4, edgecolors="none")
    for edge_position in np.linspace(0, 1, 13):
        ax.plot([0.17 + 0.66 * edge_position] * 2, [0.776, 0.876],
                color="#CCCCCC", lw=0.4, zorder=2)
    ax.text(0.5, 0.902, "Folded and cut into 64 phase bins", ha="center",
            fontsize=8, color="#666666")

    ax.text(0.5, 0.730, "Each occupied bin carries", ha="center", fontsize=8,
            color="#666666")
    box(ax, 0.26, 0.655, 0.44, 0.105, blue_fill, blue_edge)
    ax.text(0.26, 0.686, "Photometry  6D", ha="center", va="center", fontsize=8.5,
            fontweight="bold", color=blue_edge)
    ax.text(0.26, 0.633, "Brightness, scatter, count,\nphase span, error", ha="center",
            va="center", fontsize=6.8, color="#444444", linespacing=1.2)
    box(ax, 0.74, 0.655, 0.44, 0.105, orange_fill, orange_edge)
    ax.text(0.74, 0.686, "Geometry  8D", ha="center", va="center", fontsize=8.5,
            fontweight="bold", color=orange_edge)
    ax.text(0.74, 0.633, "Sun and observer directions\nand distances", ha="center",
            va="center", fontsize=6.8, color="#444444", linespacing=1.2)

    ax.text(0.5, 0.478, "The lightcurve as a whole carries", ha="center",
            fontsize=8, color="#666666")
    box(ax, 0.5, 0.405, 0.72, 0.105, green_fill, green_edge)
    ax.text(0.5, 0.436, "Summary  22D", ha="center", va="center", fontsize=8.5,
            fontweight="bold", color=green_edge)
    ax.text(0.5, 0.383, "Fourier harmonics 1 to 8  16D\nRange, coverage, count, span, period, error  6D",
            ha="center", va="center", fontsize=6.8, color="#444444",
            linespacing=1.2)

    arrow_down(ax, 0.26, 0.596, 0.556)
    arrow_down(ax, 0.74, 0.596, 0.556)
    ax.text(0.26, 0.532, "To panel B", ha="center", fontsize=7.5, color=blue_edge)
    ax.text(0.74, 0.532, "To panel C", ha="center", fontsize=7.5,
            color=orange_edge)
    arrow_down(ax, 0.5, 0.346, 0.296)
    ax.text(0.5, 0.272, "To panel B", ha="center", fontsize=7.5, color=green_edge)
    ax.text(0.5, 0.155, "Bins without observations are masked", ha="center",
            fontsize=7.5, color="#666666")

    # Panel B: the encoder, which ends in one vector per lightcurve.
    ax = axes[0, 1]
    panel_header(ax, "B", "Encoder")
    for index in range(14):
        box(ax, 0.20 + 0.042 * index, 0.880, 0.034, 0.046, blue_fill, blue_edge,
            lw=0.5)
    ax.text(0.15, 0.880, "64 bins", ha="right", va="center", fontsize=7.5,
            color="#666666")
    ax.text(0.83, 0.880, "6D each", ha="left", va="center", fontsize=7.5,
            color="#666666")
    arrow_down(ax, 0.5, 0.848, 0.798)
    box(ax, 0.5, 0.735, 0.64, 0.090, orange_fill, orange_edge, lw=1.1)
    ax.text(0.5, 0.735, "Convolutional network", ha="center", va="center",
            fontsize=9)
    ax.text(0.845, 0.735, "3 blocks,\n6D to 128D", ha="left", va="center",
            fontsize=7.5, color="#666666", linespacing=1.2)
    arrow_down(ax, 0.5, 0.688, 0.640)
    ax.text(0.53, 0.664, "128D per bin", ha="left", va="center", fontsize=7.5,
            color="#666666")
    ax.text(0.47, 0.664, "8D per bin to panel C", ha="right", va="center",
            fontsize=7.2, color=blue_edge)
    box(ax, 0.5, 0.590, 0.64, 0.080, "#FAFAFA", "#888888", lw=0.9)
    ax.text(0.5, 0.590, "Average over the occupied bins", ha="center",
            va="center", fontsize=8.5)
    arrow_down(ax, 0.5, 0.548, 0.505)
    box(ax, 0.34, 0.462, 0.24, 0.070, "#FAFAFA", "#888888", lw=0.9)
    ax.text(0.34, 0.462, "128D", ha="center", va="center", fontsize=8.5)
    ax.text(0.50, 0.462, "+", ha="center", va="center", fontsize=11,
            color="#555555")
    box(ax, 0.66, 0.462, 0.24, 0.070, green_fill, green_edge, lw=0.9)
    ax.text(0.66, 0.462, "22D", ha="center", va="center", fontsize=8.5)
    ax.text(0.5, 0.393, "One token per lightcurve", ha="center", va="center",
            fontsize=7.5, color="#666666")
    arrow_down(ax, 0.5, 0.368, 0.325)
    box(ax, 0.5, 0.258, 0.72, 0.105, orange_fill, orange_edge, lw=1.1)
    ax.text(0.5, 0.280, "Transformer over the lightcurves", ha="center",
            va="center", fontsize=9)
    ax.text(0.5, 0.236, "2 layers, 4 heads, width 128", ha="center", va="center",
            fontsize=7, color="#666666")
    arrow_down(ax, 0.5, 0.200, 0.155)
    for index, colour in enumerate(("#0072B2", "#D55E00", "#009E73", "#8E7CC3")):
        box(ax, 0.245 + 0.170 * index, 0.108, 0.135, 0.062, "#FFFFFF", colour,
            lw=1.0)
        label = r"$\mathbf{h}_%d$" % (index + 1) if index < 3 else r"$\mathbf{h}_N$"
        ax.text(0.245 + 0.170 * index, 0.108, label, ha="center", va="center",
                fontsize=8.5)
    ax.text(0.5, 0.040, "Output, one 128D vector per lightcurve", ha="center",
            fontsize=7.8, color="#333333")

    # Panel C: the trial axis, and the score it receives.
    ax = axes[1, 0]
    panel_header(ax, "C", "Trial axis and its score")
    centre = (0.22, 0.772)
    ax.add_patch(Circle(centre, 0.062, facecolor="#C9C9C9", edgecolor="none",
                        zorder=3))
    ax.plot([centre[0] - 0.036, centre[0] + 0.036],
            [centre[1] - 0.118, centre[1] + 0.118], color="#333333", lw=1.7,
            ls=(0, (4, 2)), zorder=4)
    ax.text(centre[0] + 0.054, centre[1] + 0.112, r"$\mathbf{p}$", fontsize=10)
    for angle, colour, name in ((152, "#E8A33D", r"$\mathbf{s}$"),
                                (-20, "#2878B5", r"$\mathbf{o}$")):
        theta = np.deg2rad(angle)
        ax.annotate("", xy=(centre[0] + 0.165 * np.cos(theta),
                            centre[1] + 0.165 * np.sin(theta)),
                    xytext=centre,
                    arrowprops=dict(arrowstyle="->", color=colour, lw=1.3,
                                    mutation_scale=8))
        ax.text(centre[0] + 0.205 * np.cos(theta),
                centre[1] + 0.205 * np.sin(theta), name, fontsize=10,
                color=colour, ha="center", va="center")
    ax.text(0.70, 0.820, "One direction tested\nagainst the geometry\nof every bin",
            ha="center", va="center", fontsize=7.8, color="#444444",
            linespacing=1.3)

    box(ax, 0.25, 0.594, 0.44, 0.115, purple_fill, purple_edge, lw=1.1)
    ax.text(0.25, 0.634, "Axis terms  3D", ha="center", va="center", fontsize=8.5,
            fontweight="bold", color=purple_edge)
    ax.text(0.25, 0.567,
            r"$(\mathbf{p}\cdot\mathbf{s})^2$, $(\mathbf{p}\cdot\mathbf{o})^2$," "\n"
            r"$(\mathbf{p}\cdot\mathbf{s})(\mathbf{p}\cdot\mathbf{o})$",
            ha="center", va="center", fontsize=7.2, color="#444444",
            linespacing=1.2)
    box(ax, 0.75, 0.594, 0.44, 0.115, orange_fill, orange_edge, lw=1.1)
    ax.text(0.75, 0.629, "Shared terms  4D", ha="center", va="center",
            fontsize=8.5, fontweight="bold", color=orange_edge)
    ax.text(0.75, 0.574, "Sun to observer angle  2D,\nlog distances  2D",
            ha="center", va="center", fontsize=7.0, color="#444444",
            linespacing=1.2)
    arrow_down(ax, 0.25, 0.534, 0.500)
    arrow_down(ax, 0.75, 0.534, 0.500)
    box(ax, 0.5, 0.455, 0.58, 0.080, "#FAFAFA", "#888888", lw=0.9)
    ax.text(0.5, 0.455, "7D for each occupied bin", ha="center", va="center",
            fontsize=8.5)
    arrow_down(ax, 0.40, 0.412, 0.372)
    arrow_down(ax, 0.60, 0.412, 0.372)
    box(ax, 0.25, 0.322, 0.42, 0.090, "#FAFAFA", "#888888", lw=0.9)
    ax.text(0.25, 0.322, "Mean and scatter\nover the bins  14D", ha="center",
            va="center", fontsize=7.6, linespacing=1.2)
    box(ax, 0.75, 0.322, 0.42, 0.090, blue_fill, blue_edge, lw=0.9)
    ax.text(0.75, 0.322, "Times 8D photometry of\neach bin, averaged  56D",
            ha="center", va="center", fontsize=7.6, linespacing=1.2)
    arrow_down(ax, 0.25, 0.275, 0.238)
    arrow_down(ax, 0.75, 0.275, 0.238)
    box(ax, 0.30, 0.185, 0.48, 0.090, "#FFFFFF", "#0072B2", lw=1.0)
    ax.text(0.30, 0.185, r"With $\mathbf{h}_n$, averaged" + "\nover the lightcurves",
            ha="center", va="center", fontsize=7.8, linespacing=1.2)
    arrow_right(ax, 0.555, 0.615, 0.185)
    box(ax, 0.78, 0.185, 0.30, 0.080, green_fill, green_edge, lw=1.2)
    ax.text(0.78, 0.185, "Score", ha="center", va="center", fontsize=10,
            fontweight="bold", color=green_edge)
    ax.text(0.5, 0.080, "One number per direction, higher values match better",
            ha="center", fontsize=7.8, color="#444444")

    # Panel D: five networks, their averaged map, and the output.
    ax = axes[1, 1]
    panel_header(ax, "D", "Five networks and the output")
    for index, seed in enumerate((17, 42, 137, 777, 2027)):
        centre_x = 0.120 + 0.082 * index
        network_glyph(ax, centre_x, 0.812, 0.024, "#E65100")
        score_map(ax, centre_x, 0.722, 0.034, 0.019, seed=seed, zorder=4)
        ax.text(centre_x, 0.672, str(seed), ha="center", fontsize=5.8,
                color="#666666")
    ax.text(0.28, 0.898, "Five trained networks, one map each", ha="center",
            fontsize=8, color="#666666")
    ax.text(0.71, 0.722, "6,144\ndirections\neach", ha="center", va="center",
            fontsize=7.2, color="#666666", linespacing=1.2)
    arrow_down(ax, 0.28, 0.640, 0.592)
    ax.text(0.335, 0.616, "Average", ha="left", va="center", fontsize=7.8,
            color="#333333")
    score_map(ax, 0.30, 0.510, 0.190, 0.082, seed=3, stars=True, zorder=4)
    ax.text(0.30, 0.398, "Three peaks, at least $15^\\circ$ apart", ha="center",
            fontsize=7.8, color="#444444")
    ax.text(0.71, 0.510, "Stars mark\nthe three\nhighest peaks", ha="center",
            va="center", fontsize=7.2, color="#666666", linespacing=1.25)
    arrow_down(ax, 0.30, 0.368, 0.318)
    globe(ax, 0.30, 0.190, 0.115)
    ax.text(0.71, 0.190, "Three axes,\nsix starting\npoles", ha="center",
            va="center", fontsize=8, color="#333333", linespacing=1.25)
    ax.text(0.5, 0.040, "Each axis gives two poles for classical inversion",
            ha="center", fontsize=7.8, color="#444444")

    fig.tight_layout(pad=0.5)
    fig.savefig(ROOT / "figures" / "architecture.pdf", dpi=300,
                bbox_inches="tight", pad_inches=0.02, facecolor="white")
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
        FOLLOWUP
        / "inputs/frozen-artifacts/k3-definitive-7874092/evaluations/real-oof-ensemble.npz"
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
    fig.savefig(ROOT / "figures" / "oracle-error.pdf", dpi=300,
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
        FOLLOWUP
        / "inputs/frozen-artifacts/k3-definitive-7874092/evaluations/real-oof-ensemble.npz"
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
    fig.savefig(ROOT / "figures" / "candidate-sky.pdf", dpi=300,
                bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.close(fig)


def render_interpretation() -> None:
    """Score maps against the amplitude-aspect relation and the mirror symmetry."""
    import json

    interpretation = FOLLOWUP / "data/interpretation-20260923"
    amplitude = FOLLOWUP / "data/interpretation-20260930"
    ensemble = load_npz(
        FOLLOWUP
        / "inputs/frozen-artifacts/k3-definitive-7874092/evaluations/real-oof-ensemble.npz"
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
    fig.savefig(ROOT / "figures" / "interpretation.pdf", dpi=300,
                bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.close(fig)


def render_candidate_count() -> None:
    """Oracle error and searched area against the number of candidates."""
    import json

    summary = json.loads(
        (FOLLOWUP / "data/interpretation-20260926/candidate-count.json").read_text()
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
    fig.savefig(ROOT / "figures" / "candidate-count.pdf", dpi=300, facecolor="white")
    plt.close(fig)


def render_appendix_checks() -> None:
    """Input removals, training against test error, and reference reassignment."""
    import csv
    import json

    evaluations = FOLLOWUP / "inputs/frozen-artifacts/k3-definitive-7874092/evaluations"
    finer = FOLLOWUP / "data/interpretation-20260923/ablations"
    seeds = (17, 42, 137, 777, 2027)
    rows = [
        ("All numerical inputs", evaluations, "zero-numeric"),
        ("Sun and observer directions", evaluations, "geometry"),
        ("All brightness information", evaluations, "brightness"),
        ("Brightness in the 64 phase bins", finer, "binned-brightness"),
        ("Fourier coefficients", finer, "fourier-all"),
        ("Peak-to-peak amplitude", finer, "amplitude"),
        ("Observation counts and coverage", evaluations, "sampling-summary"),
        ("Period as a number", evaluations, "period-scalar"),
        ("Sun term and product term", finer, "observer-aspect-only"),
        ("Observer term and product term", finer, "solar-aspect-only"),
        ("Product term only", finer, "no-cross-term"),
    ]
    # Okabe-Ito colours with one marker per seed, for colour-vision deficiency.
    colours = ("#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7")
    markers = ("o", "s", "^", "D", "v")
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    y = np.arange(len(rows))[::-1]
    for position, (label, folder, control) in zip(y, rows):
        gaps = []
        for seed in seeds:
            ablated, base = [], []
            for fold in range(5):
                ablated.append(load_npz(folder / f"real-fold-{fold}-seed-{seed}-{control}.npz")["oracle_errors_deg"])
                base.append(load_npz(evaluations / f"real-fold-{fold}-seed-{seed}.npz")["oracle_errors_deg"])
            gaps.append(np.concatenate(ablated).mean() - np.concatenate(base).mean())
        for colour, marker, gap in zip(colours, markers, gaps):
            ax.scatter(gap, position, s=18, color=colour, marker=marker, zorder=3,
                       edgecolors="white", linewidths=0.3)
        ax.plot([np.mean(gaps)] * 2, [position - 0.3, position + 0.3], color="black", lw=1.4)
    ax.axvline(0, color="#777777", lw=0.8)
    ax.set_yticks(y, [row[0] for row in rows], fontsize=8)
    ax.set_xlabel("Change in mean oracle error after removal (degrees)")
    ax.grid(axis="x", alpha=0.2)
    handles = [plt.Line2D([], [], marker=m, ls="", color=c, label=f"Seed {s}")
               for c, m, s in zip(colours, markers, seeds)]
    handles.append(plt.Line2D([], [], color="black", lw=1.4, label="Mean"))
    ax.legend(handles=handles, fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "input-removal.pdf", dpi=300, facecolor="white")
    plt.close(fig)

    report = FOLLOWUP / "data/internal-validation-20260916/run-1/report-export-publication"
    with (report / "train-oof-pairs.csv").open() as stream:
        pairs = [(float(r["oof_error_deg"]), float(r["mean_train_error_deg"])) for r in csv.DictReader(stream)]
    pairs = np.array(pairs)
    fig, ax = plt.subplots(figsize=(3.3, 3.2))
    ax.scatter(pairs[:, 0], pairs[:, 1], s=9, color="#2878B5", alpha=0.8, edgecolors="none")
    ax.plot([0, 90], [0, 90], color="#777777", lw=0.9)
    ax.set_xlim(0, 90)
    ax.set_ylim(0, 90)
    ax.set_xlabel("Oracle error as a test asteroid (degrees)", fontsize=8)
    ax.set_ylabel("Oracle error as a training asteroid (degrees)", fontsize=8)
    ax.tick_params(labelsize=7.5)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "training-test.pdf", dpi=300, facecolor="white")
    plt.close(fig)

    summary = json.loads((report / "summary.json").read_text())["reference_reassignment"]
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
    fig.savefig(ROOT / "figures" / "reference-reassignment.pdf", dpi=300, facecolor="white")
    plt.close(fig)


def render_data_degradation() -> None:
    """Mean oracle error against retained 30-day groups and observations per group."""
    import csv

    path = FOLLOWUP / "data/reliability-publication-20260916-final/k3_data_degradation.csv"
    blocks = ["1", "3", "5", "10", "all"]
    caps = ["5", "10", "20", "50", "all"]
    grid = np.full((5, 5), np.nan)
    with path.open() as stream:
        for row in csv.DictReader(stream):
            if not row["condition"].startswith("two_d-"):
                continue
            b = row["block_count"] if row["block_count"] not in ("", "None") else "all"
            c = row["observation_cap_per_block"] if row["observation_cap_per_block"] not in ("", "None") else "all"
            grid[blocks.index(b), caps.index(c)] = float(row["mean_error_deg"])
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    image = ax.imshow(grid, cmap="viridis_r", aspect="auto")
    for i in range(5):
        for j in range(5):
            value = grid[i, j]
            ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=9,
                    color="white" if value > 28 else "black")
    ax.set_xticks(range(5), caps)
    ax.set_yticks(range(5), blocks)
    ax.set_xlabel("Maximum observations per 30-day group (all = no limit)")
    ax.set_ylabel("30-day groups retained (all = complete record)")
    bar = fig.colorbar(image, ax=ax)
    bar.set_label("Mean oracle error (degrees)")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "data-degradation.pdf", dpi=300, facecolor="white")
    plt.close(fig)


def render_shapley() -> None:
    """Shapley values of the eight input groups: mean share and per-asteroid spread."""
    import glob
    import sys

    sys.path.insert(0, str(FOLLOWUP / "source/repro"))
    sys.path.insert(0, str(FOLLOWUP / "source"))
    import group_shapley as gs
    import group_shapley_summary as su

    masks = su.flat_map_masks()
    per_seed = {}
    per_object = {}
    per_object_p = {}
    for path in sorted(glob.glob(str(FOLLOWUP / "data/interpretation-20260926/shapley/*.npz"))):
        data = load_npz(path)
        seed = int(path.split("seed-")[1].split(".")[0])
        percentile = data["percentile"].copy()
        percentile[:, masks] = 0.5
        error = data["grid_error"].copy()
        values_p = gs.shapley(percentile)
        per_seed.setdefault(seed, []).append(values_p)
        for index, name in enumerate(data["object_ids"]):
            per_object_p.setdefault(str(name), []).append(values_p[index])
        # grid-error values: flat-map combinations need the flat-map peaks, as in the summary script
        grid = gs.axial_healpix_grid(int(data["nside"]))
        modes = gs.FastModes(grid)
        flat_axes = modes.vectors[modes(np.zeros(len(grid.vectors)))]
        catalog = {}
        import json
        for line in gs.CATALOG.read_text().splitlines():
            row = json.loads(line)
            catalog[row["object_id"]] = np.array([s["vector"] for s in row["solutions"]], dtype=float)
        for index, name in enumerate(data["object_ids"]):
            flat_error = gs.axial_error(flat_axes, catalog[str(name)])
            error[index, masks] = flat_error
        values_e = gs.shapley(error)
        for index, name in enumerate(data["object_ids"]):
            per_object.setdefault(str(name), []).append(values_e[index])
    labels = {
        "binned_brightness": "Brightness in phase bins",
        "sun_term": "Sun term",
        "product_term": "Product term",
        "sampling_summaries": "Observation counts and coverage",
        "phase_angle_and_distances": "Phase angle and distances",
        "lightcurve_shape_summaries": "Fourier and amplitude",
        "period": "Period as a number",
        "observer_term": "Observer term",
    }
    groups = list(gs.GROUPS)
    shares = []
    for seed, values in sorted(per_seed.items()):
        values = np.concatenate(values)
        shares.append(100 * values.mean(axis=0) / values.sum(axis=1).mean())
    shares = np.array(shares)
    pooled = np.array([np.mean(v, axis=0) for v in per_object_p.values()])
    mean_share = 100 * pooled.mean(axis=0) / pooled.sum(axis=1).mean()
    order = np.argsort(mean_share)
    errors = np.array([np.mean(v, axis=0) for v in per_object.values()])

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.3), sharey=True,
                             gridspec_kw={"width_ratios": (1, 1.25)})
    y = np.arange(len(groups))
    ax = axes[0]
    ax.barh(y, mean_share[order], color="#2878B5", height=0.6)
    ax.errorbar(mean_share[order], y,
                xerr=[mean_share[order] - shares.min(0)[order], shares.max(0)[order] - mean_share[order]],
                fmt="none", ecolor="#333333", elinewidth=0.9, capsize=2)
    ax.axvline(0, color="#777777", lw=0.8)
    ax.set_yticks(y, [labels[groups[i]] for i in order], fontsize=8)
    ax.set_xlabel("Share of total (%)", fontsize=8)
    ax.tick_params(labelsize=7.5)
    ax.grid(axis="x", alpha=0.2)
    ax.set_title("(a)", fontsize=9, loc="left")
    ax = axes[1]
    generator = np.random.default_rng(20260928)
    for row, index in enumerate(order):
        values = errors[:, index]
        ax.scatter(values, row + generator.uniform(-0.22, 0.22, values.size), s=4,
                   color="#2878B5", alpha=0.45, edgecolors="none")
        ax.plot([values.mean()] * 2, [row - 0.3, row + 0.3], color="black", lw=1.4)
    ax.axvline(0, color="#777777", lw=0.8)
    ax.set_xlabel("Change of oracle error (degrees)", fontsize=8)
    ax.tick_params(labelsize=7.5)
    ax.grid(axis="x", alpha=0.2)
    ax.set_title("(b)", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "shapley.pdf", dpi=300, facecolor="white")
    plt.close(fig)
    print("mean shares", dict(zip([groups[i] for i in order], np.round(mean_share[order], 1))))
    print("mean errors", dict(zip([groups[i] for i in order], np.round(errors.mean(0)[order], 2))))


def render_latitude(variant: str) -> None:
    """Mean oracle error against the absolute ecliptic latitude of the reference pole.

    Own-period copy: the lower-quality DeLPHI curve and the standard-pole line
    use the own-period census (``variant``); the main-sample curve is the
    published one (another package redraws it).  The published lower-quality
    curve is added in grey for comparison.  The binned series are saved as JSON.
    """
    import json

    main = json.loads((FOLLOWUP / "data/interpretation-20260926/mirror-symmetry.json").read_text())["objects"]
    main_lat = np.array([abs(row["reference_latitude_deg"]) for row in main])
    main_err = np.array([row["oracle_error_deg"] for row in main])
    series = json.loads((SCORES / f"direction-{variant}.json").read_text())["series"]
    low_lat = np.array(series["abs_latitude_deg"])
    low_err = np.array(series["delphi_error_deg"])
    low_std = np.array(series["standard_error_deg"])
    published = json.loads((SCORES.parent / "scores-published-check" / "direction-excluded.json").read_text())["series"]
    pub_lat = np.array(published["abs_latitude_deg"])
    pub_err = np.array(published["delphi_error_deg"])
    edges = np.arange(0, 91, 15)
    centres = edges[:-1] + 7.5
    generator = np.random.default_rng(20260929)
    saved = {"variant": variant, "variant_meaning": VARIANTS[variant], "bin_edges_deg": edges.tolist(),
             "bootstrap": "2,000 resamples, generator seed 20260929 shared in drawing order"}

    def binned(lat, err):
        means, lows, highs = [], [], []
        for low, high in zip(edges[:-1], edges[1:]):
            chosen = err[(lat >= low) & ((lat < high) if high < 90 else (lat <= high))]
            draws = generator.choice(chosen, (2000, chosen.size)).mean(axis=1)
            means.append(chosen.mean())
            lows.append(chosen.mean() - np.percentile(draws, 2.5))
            highs.append(np.percentile(draws, 97.5) - chosen.mean())
        return np.array(means), np.array([lows, highs])

    def counts(lat):
        return [int(((lat >= lo) & ((lat < hi) if hi < 90 else (lat <= hi))).sum())
                for lo, hi in zip(edges[:-1], edges[1:])]

    fig, ax = plt.subplots(figsize=(5.3, 3.4))
    # Okabe-Ito colours plus distinct line styles and markers, so the series stay
    # distinguishable under colour-vision deficiency.
    for key, lat, err, label, colour, marker, style, shift in (
        ("main_published", main_lat, main_err, "DeLPHI, 170 test asteroids (DAMIT period, published)", "#0072B2", "o", "-", -1.2),
        ("lowq_own", low_lat, low_err, f"DeLPHI, {low_err.size:,} lower-quality models (own period, {VARIANTS[variant]})", "#333333", "s", "--", 1.2),
    ):
        means, yerr = binned(lat, err)
        saved[key] = {"n": int(err.size), "bin_n": counts(lat), "mean_deg": means.tolist(),
                      "ci95_low_deg": (means - yerr[0]).tolist(), "ci95_high_deg": (means + yerr[1]).tolist()}
        ax.errorbar(centres + shift, means, yerr=yerr, fmt=marker, ls=style, color=colour, ms=5,
                    lw=1.3, capsize=2, label=label)
    means, _ = binned(low_lat, low_std)
    saved["lowq_standard"] = {"n": int(low_std.size), "bin_n": counts(low_lat), "mean_deg": means.tolist()}
    ax.plot(centres, means, ":", color="#E69F00", lw=1.8, label="Standard poles, same lower-quality models")
    means, _ = binned(pub_lat, pub_err)
    saved["lowq_published"] = {"n": int(pub_err.size), "bin_n": counts(pub_lat), "mean_deg": means.tolist()}
    ax.plot(centres + 1.2, means, "-", color="#999999", lw=1.0, alpha=0.8,
            label=f"DeLPHI, {pub_err.size:,} lower-quality models (DAMIT period, published)")
    ax.set_xticks(edges)
    ax.set_xlim(0, 90)
    ax.set_ylim(0, 90 if variant == "at90" else 45)
    ax.set_xlabel("Absolute ecliptic latitude of the reference pole (degrees)")
    ax.set_ylabel("Mean oracle error (degrees)")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=6.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / f"latitude-{variant}.pdf", dpi=300, facecolor="white")
    fig.savefig(OUT / f"latitude-{variant}.png", dpi=200, facecolor="white")
    plt.close(fig)
    with (OUT / f"latitude-{variant}-series.json").open("x") as stream:
        stream.write(json.dumps(saved, indent=1) + "\n")


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


def render_sampling_caps() -> None:
    """Mean oracle error when the total number of observations is limited (Appendix B)."""
    rows = _sampling_rows()
    caps = ("10", "20", "50", "100", "200", "500", "None")
    labels = ["10", "20", "50", "100", "200", "500", "All"]
    means = np.array([float(rows[f"observation_cap-cap-{c}"]["mean_error_deg"]) for c in caps])
    low = np.array([float(rows[f"observation_cap-cap-{c}"]["mean_error_bootstrap_lower_95"]) for c in caps])
    high = np.array([float(rows[f"observation_cap-cap-{c}"]["mean_error_bootstrap_upper_95"]) for c in caps])
    x = np.arange(len(caps))
    fig, ax = plt.subplots(figsize=(5.3, 3.3))
    ax.axhline(_standard_mean(), color="#E69F00", ls=":", lw=1.8, label="Six standard starting poles")
    ax.errorbar(x, means, yerr=[means - low, high - means], fmt="o-", color="#0072B2", ms=5,
                lw=1.3, capsize=3, label="DeLPHI")
    ax.set_xticks(x, labels)
    ax.set_xlabel("Maximum total number of observations per asteroid")
    ax.set_ylabel("Mean oracle error (degrees)")
    ax.set_ylim(0, 45)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "sampling-caps.pdf", dpi=300, facecolor="white")
    plt.close(fig)


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
    fig.savefig(ROOT / "figures" / "period-sensitivity.pdf", dpi=300, facecolor="white")
    plt.close(fig)


def render_coverage() -> None:
    baselines = load_npz(
        FOLLOWUP / "source/repro/baselines_out/reference-baselines-per-object.npz"
    )
    if network.switched():
        errors = load_npz(
            DataPath("maps/own-period/evaluations/real-oof-ensemble.npz")
        )["oracle_errors_deg"]
    else:
        errors = baselines["delphi_errors_deg"]
    delphi_x, delphi_y = empirical_cdf(errors)
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
    fig.savefig(ROOT / "figures" / "primary-coverage.pdf", dpi=300)
    plt.close(fig)


def render_transfer(variant: str) -> None:
    """Figure 7 with the three lower-quality rows at the own period (``variant``).

    The main-sample, ZTF and ALCDEF rows are the published values (other
    packages recompute them) and are marked "(published)".
    """
    labels = [
        "Main DAMIT test asteroids (published)",
        "All lower-quality DAMIT (own period)",
        "Gaia DR3 lower-quality subset (own period)",
        "ASAS-SN lower-quality subset (own period)",
        "Mapped ZTF (published)",
        "ALCDEF–Gaia (published)",
    ]
    import json

    summary = json.loads(
        (FOLLOWUP / "source/repro/baselines_out/reference-baselines.json").read_text()
    )
    own = json.loads((SCORES / f"baselines-{variant}.json").read_text())
    rows = [
        summary["standard_minus_delphi"],
        own["lowq"]["standard_minus_delphi"],
        own["lowq_gaia"]["standard_minus_delphi"],
        own["lowq_asas"]["standard_minus_delphi"],
        summary["primary_variants"]["ztf_primary"]["standard_minus_delphi"],
        summary["primary_variants"]["alcdef_primary"]["standard_minus_delphi"],
    ]
    difference = np.array([row["mean_difference_deg"] for row in rows])
    low = np.array([row["ci95_low_deg"] for row in rows])
    high = np.array([row["ci95_high_deg"] for row in rows])
    print("generalization differences", variant, np.round(difference, 2), np.round(low, 2), np.round(high, 2))
    with (OUT / f"generalization-checks-{variant}-rows.json").open("x") as stream:
        stream.write(json.dumps({"variant": variant, "variant_meaning": VARIANTS[variant],
                                 "rows": [dict(label=label, **r) for label, r in zip(labels, rows)]}, indent=1) + "\n")
    y = np.arange(len(labels))[::-1]
    colors = np.where(difference >= 0, "#2878B5", "#C44E52")
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    ax.axvline(0, color="#333333", lw=0.9)
    ax.errorbar(
        difference,
        y,
        xerr=np.vstack((difference - low, high - difference)),
        fmt="none",
        ecolor="#555555",
        elinewidth=1.2,
        capsize=3,
    )
    ax.scatter(difference, y, c=colors, s=38, zorder=3)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Standard-pole error minus DeLPHI error (degrees)")
    ax.set_title(f"Lower-quality rows at the own period, {VARIANTS[variant]}", fontsize=8)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / f"generalization-checks-{variant}.pdf", dpi=300)
    fig.savefig(OUT / f"generalization-checks-{variant}.png", dpi=200)
    plt.close(fig)


def render_inversion() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.8))
    methods = ["146-start\ngrid", "DeLPHI\nguided", "Six standard\nstarts"]
    times = [40.67, 18.07, 2.79]
    agreements = [135, 134, 108]
    colors = ["#777777", "#2878B5", "#D95F02"]
    axes[0].bar(methods, times, color=colors)
    axes[0].tick_params(axis="x", labelsize=7)
    axes[0].set_ylabel("Mean time per object (s)")
    axes[0].set_title("End-to-end time")
    axes[0].text(0.5, 0.92, "2.25×", transform=axes[0].transAxes, ha="center")
    axes[1].bar(methods, agreements, color=colors)
    axes[1].tick_params(axis="x", labelsize=7)
    axes[1].set_ylim(0, 170)
    axes[1].set_ylabel("Objects within 20°")
    axes[1].set_title("Reference agreement")
    for index, value in enumerate(agreements):
        axes[1].text(index, value + 3, str(value), ha="center", va="bottom")
    axes[2].axhline(1.01, color="#777777", ls="--", lw=1.0, label="Fixed limit")
    axes[2].errorbar(
        [0],
        [1.019],
        yerr=[[1.019 - 1.013], [1.025 - 1.019]],
        fmt="o",
        color="#C44E52",
        capsize=4,
    )
    axes[2].set_xlim(-0.6, 0.6)
    axes[2].set_xticks([0], ["Guided / grid"])
    axes[2].set_ylim(0.995, 1.03)
    axes[2].set_ylabel("Residual RMS ratio")
    axes[2].set_title("Fit residual")
    axes[2].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    save(fig, "delphi_inversion_benchmark.pdf")


def main() -> None:
    global OUT
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.out
    network.guard_output(OUT)
    if "paper-rev-20261021" in OUT.resolve().parts:
        raise ValueError("figure output must be outside the manuscript directory")
    OUT.mkdir(parents=True, exist_ok=True)
    # Own-period copy: only Figures 11 and 7, for both failure variants.
    for variant in VARIANTS:
        render_latitude(variant)
        render_transfer(variant)


if __name__ == "__main__":
    from repro.own_period.figure_data import capture_figures
    with capture_figures():
        main()
