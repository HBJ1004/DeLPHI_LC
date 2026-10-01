"""Mirror symmetry of the DeLPHI score maps and dependence on pole direction.

Outputs data/interpretation-20260923/mirror-symmetry.json with
  * per-asteroid correlation between score(lambda, beta) and
    score(lambda + 180 deg, beta), and a lambda + 90 deg control;
  * mean absolute ecliptic latitude of the observer and Sun directions;
  * the fraction of asteroids whose three candidates contain a mirror pair,
    with a null that keeps each candidate latitude and draws its longitude;
  * the score of the reference pole minus that of its mirror;
  * oracle error against reference latitude and longitude, for the 170
    test asteroids and for the 9,783 lower-quality DAMIT asteroids.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import kruskal, spearmanr, wilcoxon

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import compute_reference_baselines as base  # noqa: E402

from lc_pipeline.k3.grid import axial_healpix_grid  # noqa: E402

FOLLOWUP = HERE.parents[1]
ENSEMBLE = (
    FOLLOWUP / "inputs/frozen-artifacts/k3-definitive-7874092/evaluations/real-oof-ensemble.npz"
)
DUMP = FOLLOWUP.parent / "damit-20250610T000301Z"
LOWQ = FOLLOWUP / "data/publication-revision-20260917-r2/lowq-analysis.json"
OUTPUT = FOLLOWUP / "data/interpretation-20260926/mirror-symmetry.json"
SEED = 20260923


def mirror(vectors):
    vectors = np.asarray(vectors, dtype=float)
    return np.stack((-vectors[..., 0], -vectors[..., 1], vectors[..., 2]), axis=-1)


def axial_angle(first, second):
    return float(np.degrees(np.arccos(np.clip(abs(first @ second), 0.0, 1.0))))


def viewing_latitudes(object_id):
    tokens = (DUMP / "files" / object_id / "lc.txt").read_text().split()
    count, cursor = int(tokens[0]), 1
    observer, sun = [], []
    for _ in range(count):
        n = int(tokens[cursor])
        cursor += 2
        rows = np.asarray(tokens[cursor : cursor + 8 * n], dtype=float).reshape(n, 8)
        cursor += 8 * n
        s = rows[:, 2:5] / np.linalg.norm(rows[:, 2:5], axis=1, keepdims=True)
        e = rows[:, 5:8] / np.linalg.norm(rows[:, 5:8], axis=1, keepdims=True)
        sun.append(np.abs(s[:, 2]).mean())
        observer.append(np.abs(e[:, 2]).mean())
    return (
        float(np.degrees(np.arcsin(np.mean(observer)))),
        float(np.degrees(np.arcsin(np.mean(sun)))),
    )


def closest_mirror_pair(axes):
    return min(axial_angle(axes[i], mirror(axes[j])) for i in range(3) for j in range(3) if i != j)


def rotate_z(vector, angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([c * vector[0] - s * vector[1], s * vector[0] + c * vector[1], vector[2]])


def binned(values, errors, edges):
    out = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (values >= low) & ((values < high) if high < edges[-1] else (values <= high))
        out.append(
            {
                "low": low,
                "high": high,
                "n": int(mask.sum()),
                "mean_error_deg": float(errors[mask].mean()),
                "within_20": float(np.mean(errors[mask] < 20.0)),
            }
        )
    return out


def main():
    ensemble = np.load(ENSEMBLE)
    names = [str(value) for value in ensemble["object_ids"]]
    maps = ensemble["score_grids"].astype(float)
    errors = ensemble["oracle_errors_deg"]
    axes = ensemble["refined_axes"] / np.linalg.norm(
        ensemble["refined_axes"], axis=-1, keepdims=True
    )
    grid = axial_healpix_grid(32).vectors
    tree = cKDTree(np.r_[grid, -grid])

    def index_of(vectors):
        return tree.query(vectors)[1] % len(grid)

    mirror_index = index_of(mirror(grid))
    quarter_index = index_of(np.c_[-grid[:, 1], grid[:, 0], grid[:, 2]])
    poles = base.reference_poles()
    generator = np.random.default_rng(SEED)

    per_object = []
    for k, name in enumerate(names):
        observer_lat, sun_lat = viewing_latitudes(name)
        z = (maps[k] - maps[k].mean()) / maps[k].std()
        pole = poles[name][0]
        separation = axial_angle(pole, mirror(pole))
        per_object.append(
            {
                "object_id": name,
                "mirror_correlation": float(np.corrcoef(maps[k], maps[k][mirror_index])[0, 1]),
                "quarter_turn_correlation": float(
                    np.corrcoef(maps[k], maps[k][quarter_index])[0, 1]
                ),
                "observer_abs_latitude_deg": observer_lat,
                "sun_abs_latitude_deg": sun_lat,
                "closest_mirror_pair_deg": closest_mirror_pair(axes[k]),
                "pole_mirror_separation_deg": separation,
                "reference_minus_mirror_score_z": float(
                    z[index_of(pole[None])[0]] - z[index_of(mirror(pole)[None])[0]]
                ),
                "reference_longitude_deg": float(np.degrees(np.arctan2(pole[1], pole[0])) % 360.0),
                "reference_latitude_deg": float(np.degrees(np.arcsin(np.clip(pole[2], -1, 1)))),
                "oracle_error_deg": float(errors[k]),
            }
        )
    table = {
        key: np.array([row[key] for row in per_object])
        for key in per_object[0]
        if key != "object_id"
    }

    null = []
    for _ in range(300):
        rotated = np.array(
            [[rotate_z(v, generator.uniform(0, 2 * np.pi)) for v in triple] for triple in axes]
        )
        null.append([closest_mirror_pair(triple) for triple in rotated])
    null = np.array(null)
    pairs = {}
    for threshold in (10.0, 15.0):
        fractions = (null < threshold).mean(axis=1)
        pairs[f"{threshold:.0f}"] = {
            "observed": float(np.mean(table["closest_mirror_pair_deg"] < threshold)),
            "null_mean": float(fractions.mean()),
            "null_97_5": float(np.percentile(fractions, 97.5)),
            "null_max": float(fractions.max()),
        }

    symmetry_bins = []
    for low, high in ((0, 3), (3, 6), (6, 10), (10, 90)):
        mask = (table["observer_abs_latitude_deg"] >= low) & (
            table["observer_abs_latitude_deg"] < high
        )
        symmetry_bins.append(
            {
                "low": low,
                "high": high,
                "n": int(mask.sum()),
                "median_mirror_correlation": float(np.median(table["mirror_correlation"][mask])),
            }
        )
    preference = []
    usable = table["pole_mirror_separation_deg"] >= 30.0
    for low, high in ((0, 6), (6, 10), (10, 90)):
        mask = (
            usable
            & (table["observer_abs_latitude_deg"] >= low)
            & (table["observer_abs_latitude_deg"] < high)
        )
        d = table["reference_minus_mirror_score_z"][mask]
        preference.append(
            {
                "low": low,
                "high": high,
                "n": int(mask.sum()),
                "fraction_reference_higher": float(np.mean(d > 0)),
                "median_difference_z": float(np.median(d)),
                "wilcoxon_p": float(wilcoxon(d).pvalue),
            }
        )

    longitude = table["reference_longitude_deg"]
    latitude = table["reference_latitude_deg"]
    main_sample = {
        "spearman_abs_latitude": float(spearmanr(np.abs(latitude), errors)[0]),
        "spearman_abs_latitude_p": float(spearmanr(np.abs(latitude), errors)[1]),
        "latitude_bins": binned(np.abs(latitude), errors, [0, 30, 60, 90]),
        "longitude_bins": binned(longitude, errors, list(range(0, 361, 60))),
        "longitude_kruskal_p": float(
            kruskal(
                *[errors[(longitude >= lo) & (longitude < lo + 60)] for lo in range(0, 360, 60)]
            ).pvalue
        ),
    }

    solutions = defaultdict(list)
    with (DUMP / "tables/asteroid_models.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                solutions[int(row["asteroid_id"])].append(
                    (float(row["lambda"]), float(row["beta"]))
                )
            except ValueError:
                continue
    standard = base.standard_axes()
    low_rows = []
    for entry in json.loads(LOWQ.read_text())["objects"]:
        listed = solutions[int(entry["object_id"].split(":")[1])]
        references = np.array([base.unit_from_degrees(lam, b) for lam, b in listed])
        low_rows.append(
            (
                listed[0][0] % 360.0,
                abs(listed[0][1]),
                float(entry["oracle_at_3_error_deg"]),
                base.axial_error_deg(standard, references),
            )
        )
    low = np.array(low_rows)
    lower_quality = {
        "n": int(len(low)),
        "spearman_abs_latitude": float(spearmanr(low[:, 1], low[:, 2])[0]),
        "latitude_bins": binned(low[:, 1], low[:, 2], [0, 15, 30, 45, 60, 75, 90]),
        "standard_latitude_bins": binned(low[:, 1], low[:, 3], [0, 15, 30, 45, 60, 75, 90]),
        "longitude_kruskal_p": float(
            kruskal(
                *[low[(low[:, 0] >= lo) & (low[:, 0] < lo + 60), 2] for lo in range(0, 360, 60)]
            ).pvalue
        ),
        "longitude_bins": binned(low[:, 0], low[:, 2], list(range(0, 361, 60))),
    }

    # Split the oracle error into latitude and longitude parts for the closest
    # candidate, taken at the end of its axis that lies nearer the reference.
    split_rows = []
    for k, name in enumerate(names):
        best = None
        for candidate in axes[k]:
            for pole in poles[name]:
                aligned = candidate if candidate @ pole >= 0 else -candidate
                angle = axial_angle(candidate, pole)
                if best is None or angle < best[0]:
                    best = (angle, aligned, pole)
        _, candidate, pole = best
        beta_p = np.degrees(np.arcsin(np.clip(pole[2], -1, 1)))
        beta_c = np.degrees(np.arcsin(np.clip(candidate[2], -1, 1)))
        dlon = (
            np.degrees(np.arctan2(candidate[1], candidate[0]) - np.arctan2(pole[1], pole[0]))
            + 180.0
        ) % 360.0 - 180.0
        split_rows.append(
            (abs(beta_p), abs(beta_c - beta_p), abs(dlon) * np.cos(np.radians(beta_p)))
        )
    split = np.array(split_rows)
    coordinate_split = {
        "mean_latitude_part_deg": float(split[:, 1].mean()),
        "mean_longitude_part_deg": float(split[:, 2].mean()),
        "bins": [],
    }
    for low, high in ((0, 30), (30, 60), (60, 90)):
        mask = (split[:, 0] >= low) & ((split[:, 0] < high) if high < 90 else (split[:, 0] <= high))
        coordinate_split["bins"].append(
            {
                "low": low,
                "high": high,
                "n": int(mask.sum()),
                "latitude_part_deg": float(split[mask, 1].mean()),
                "longitude_part_deg": float(split[mask, 2].mean()),
            }
        )

    summary = {
        "coordinate_split": coordinate_split,
        "n": len(names),
        "mirror_correlation_median": float(np.median(table["mirror_correlation"])),
        "quarter_turn_correlation_median": float(np.median(table["quarter_turn_correlation"])),
        "spearman_asymmetry_observer_latitude": float(
            spearmanr(1 - table["mirror_correlation"], table["observer_abs_latitude_deg"])[0]
        ),
        "spearman_asymmetry_sun_latitude": float(
            spearmanr(1 - table["mirror_correlation"], table["sun_abs_latitude_deg"])[0]
        ),
        "symmetry_by_observer_latitude": symmetry_bins,
        "mirror_pairs_among_candidates": pairs,
        "reference_preferred_over_mirror": preference,
        "main_sample_direction": main_sample,
        "lower_quality_direction": lower_quality,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({"summary": summary, "objects": per_object}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
