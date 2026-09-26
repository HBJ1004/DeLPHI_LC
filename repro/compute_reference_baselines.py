"""Oracle errors of two photometry-independent references.

The first reference is the set of six standard starting poles used by a
classical convex-inversion search.  The second is a set of three axes drawn at
random.  Neither reference looks at the photometry of the asteroid, so both
measure what can be reached without it.  The script reads frozen evidence only
and writes a summary plus LaTeX macros.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
FOLLOWUP = ROOT / "DeLPHI-followup"
CATALOG = FOLLOWUP / "source/repro/data/damit-20250610T000301Z/catalog.jsonl"
ENSEMBLE = (
    FOLLOWUP
    / "inputs/frozen-artifacts/k3-definitive-7874092/evaluations/real-oof-ensemble.npz"
)
OUT = Path(__file__).resolve().parent / "baselines_out"

STANDARD_POLES_DEG = (
    (0.0, 0.0),
    (180.0, 0.0),
    (90.0, 60.0),
    (240.0, 60.0),
    (90.0, -60.0),
    (240.0, -60.0),
)
GAIA_SOURCE_KEY = "2023arxiv230510798d"
ASAS_SOURCE_KEY = "2021a&a...654a..48h"
RANDOM_DRAWS = 10_000
RANDOM_SEED = 20260922
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260922


def unit_from_degrees(longitude: float, latitude: float) -> np.ndarray:
    lon, lat = np.radians(longitude), np.radians(latitude)
    return np.array(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
    )


def standard_axes() -> np.ndarray:
    """The distinct axes among the six standard starting poles."""
    kept: list[np.ndarray] = []
    for longitude, latitude in STANDARD_POLES_DEG:
        axis = unit_from_degrees(longitude, latitude)
        if not any(abs(abs(float(axis @ other)) - 1.0) < 1e-9 for other in kept):
            kept.append(axis)
    return np.array(kept)


def axial_error_deg(axes: np.ndarray, references: np.ndarray) -> float:
    """Smallest angle between any trial axis and any reference pole."""
    products = np.abs(np.atleast_2d(axes) @ references.T)
    return float(np.degrees(np.arccos(np.clip(products.max(), 0.0, 1.0))))


def reference_poles() -> dict[str, np.ndarray]:
    poles: dict[str, np.ndarray] = {}
    for line in CATALOG.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        vectors = [
            solution["vector"]
            for solution in record["solutions"]
            if solution.get("vector")
        ]
        if not vectors:
            continue
        array = np.array(vectors, dtype=float).reshape(-1, 3)
        poles[record["object_id"]] = array / np.linalg.norm(
            array, axis=1, keepdims=True
        )
    return poles


def paired_difference(first: np.ndarray, second: np.ndarray, seed: int) -> dict:
    """Mean of first minus second, with a bootstrap interval over asteroids."""
    generator = np.random.default_rng(seed)
    differences = first - second
    count = differences.size
    means = np.empty(BOOTSTRAP_RESAMPLES)
    for index in range(BOOTSTRAP_RESAMPLES):
        draw = generator.integers(0, count, count)
        means[index] = differences[draw].mean()
    low, high = np.percentile(means, (2.5, 97.5))
    return {
        "mean_difference_deg": float(differences.mean()),
        "ci95_low_deg": float(low),
        "ci95_high_deg": float(high),
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": seed,
    }


def summarize(errors: np.ndarray) -> dict:
    return {
        "mean_deg": float(errors.mean()),
        "median_deg": float(np.median(errors)),
        "within_twenty_percent": float((errors < 20.0).mean() * 100.0),
        "object_count": int(errors.size),
    }


def damit_table_poles() -> dict[int, np.ndarray]:
    """Pole vectors of every DAMIT model, keyed by asteroid number."""
    import csv

    path = ROOT / "damit-20250610T000301Z/tables/asteroid_models.csv"
    grouped: dict[int, list[np.ndarray]] = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                number = int(row["asteroid_id"])
                longitude = float(row["lambda"])
                latitude = float(row["beta"])
            except (TypeError, ValueError):
                continue
            grouped.setdefault(number, []).append(
                unit_from_degrees(longitude, latitude)
            )
    return {number: np.array(value) for number, value in grouped.items()}


def gaia_reference_poles() -> dict[str, np.ndarray]:
    """Pole axes of the Gaia DR3 spin table, keyed by MPC identifier."""
    path = (
        FOLLOWUP
        / "data/generalization-20260910/sources/gaia-dr3-spins-20260911/table3.dat"
    )
    poles: dict[str, np.ndarray] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        if not line.strip():
            continue
        number = int(line[0:6])
        axes = [unit_from_degrees(float(line[24:27]), float(line[28:31]))]
        second_longitude, second_latitude = line[32:35].strip(), line[36:39].strip()
        if second_longitude and second_latitude:
            axes.append(
                unit_from_degrees(float(second_longitude), float(second_latitude))
            )
        poles[f"mpc:{number}"] = np.array(axes)
    return poles


def cohort_reference(name: str, table: dict[int, np.ndarray],
                     catalog: dict[str, np.ndarray]) -> np.ndarray | None:
    """Reference poles of one cohort member, whatever identifier it uses."""
    if name in catalog:
        return catalog[name]
    for prefix in ("damit:", "mpc:", "asteroid_"):
        if name.startswith(prefix):
            try:
                number = int(name[len(prefix):])
            except ValueError:
                return None
            return table.get(number)
    return None


def random_reference_errors(references: list[np.ndarray], seed: int,
                            draws: int) -> tuple[np.ndarray, dict]:
    """Mean oracle error of three random axes, per object and in summary."""
    generator = np.random.default_rng(seed)
    per_object = np.zeros(len(references))
    means, medians, within = [], [], []
    for _ in range(draws):
        errors = np.empty(len(references))
        for position, poles in enumerate(references):
            trial = generator.normal(size=(3, 3))
            trial /= np.linalg.norm(trial, axis=1, keepdims=True)
            errors[position] = axial_error_deg(trial, poles)
        per_object += errors
        means.append(errors.mean())
        medians.append(np.median(errors))
        within.append((errors < 20.0).mean() * 100.0)
    low, high = np.percentile(means, (2.5, 97.5))
    return per_object / draws, {
        "mean_deg": float(np.mean(means)),
        "mean_scatter_deg": float(np.std(means)),
        "median_deg": float(np.mean(medians)),
        "within_twenty_percent": float(np.mean(within)),
        "randomization_interval95_low_deg": float(low),
        "randomization_interval95_high_deg": float(high),
        "method": "three_independent_uniform_sphere_axes_per_identity",
        "resamples": draws,
        "object_count": len(references),
    }


def main() -> None:
    poles = reference_poles()
    ensemble = np.load(ENSEMBLE)
    identifiers = [str(value) for value in ensemble["object_ids"]]
    delphi = np.asarray(ensemble["oracle_errors_deg"], dtype=float)

    axes = standard_axes()
    standard = np.array([axial_error_deg(axes, poles[name]) for name in identifiers])

    generator = np.random.default_rng(RANDOM_SEED)
    random_means, random_medians, random_within = [], [], []
    random_per_object = np.zeros(len(identifiers))
    for _ in range(RANDOM_DRAWS):
        errors = np.empty(len(identifiers))
        for position, name in enumerate(identifiers):
            trial = generator.normal(size=(3, 3))
            trial /= np.linalg.norm(trial, axis=1, keepdims=True)
            errors[position] = axial_error_deg(trial, poles[name])
        random_means.append(errors.mean())
        random_medians.append(np.median(errors))
        random_within.append((errors < 20.0).mean() * 100.0)
        random_per_object += errors
    random_per_object /= RANDOM_DRAWS

    # The other cohorts in which a photometry-independent reference is quoted.
    table = damit_table_poles()
    gaia = gaia_reference_poles()
    revision = FOLLOWUP / "data/publication-revision-20260917-r2"
    cohorts: dict[str, dict] = {}

    lowq = json.loads((revision / "lowq-analysis.json").read_text(encoding="utf-8"))
    names, delphi_errors = [], []
    for entry in lowq["objects"]:
        names.append(entry["object_id"])
        delphi_errors.append(float(entry["oracle_at_3_error_deg"]))
    cohorts["lowq"] = {"names": names, "delphi": np.array(delphi_errors)}

    for label, filename in (
        ("ztf", "error-channel-ztf-predictions.json"),
        ("alcdef", "error-channel-alcdef-predictions.json"),
    ):
        payload_in = json.loads((revision / filename).read_text(encoding="utf-8"))
        names, delphi_errors = [], []
        for entry in payload_in["objects"]:
            if entry.get("status") != "ok":
                continue
            lookup = gaia if label == "alcdef" else poles
            references = cohort_reference(entry["object_id"], table, lookup)
            if label == "alcdef":
                references = gaia.get(entry["object_id"])
            if references is None:
                continue
            predicted = np.array(entry["axes"], dtype=float).reshape(-1, 3)
            predicted /= np.linalg.norm(predicted, axis=1, keepdims=True)
            names.append(entry["object_id"])
            delphi_errors.append(axial_error_deg(predicted, references))
        cohorts[label] = {"names": names, "delphi": np.array(delphi_errors)}

    cohort_summary: dict[str, dict] = {}
    for label, content in cohorts.items():
        references, kept, kept_delphi = [], [], []
        for name, value in zip(content["names"], content["delphi"], strict=True):
            found = (
                gaia.get(name)
                if label == "alcdef"
                else cohort_reference(name, table, poles)
            )
            if found is None:
                continue
            references.append(found)
            kept.append(name)
            kept_delphi.append(value)
        kept_delphi = np.array(kept_delphi)
        standard_errors = np.array(
            [axial_error_deg(axes, item) for item in references]
        )
        draws = 200 if len(references) > 1000 else RANDOM_DRAWS // 10
        random_errors, random_stats = random_reference_errors(
            references, RANDOM_SEED + len(label), draws
        )
        cohort_summary[label] = {
            "object_count": len(references),
            "unmatched_count": len(content["names"]) - len(references),
            "delphi": summarize(kept_delphi),
            "standard": summarize(standard_errors),
            "random": random_stats,
            "standard_minus_delphi": paired_difference(
                standard_errors, kept_delphi, BOOTSTRAP_SEED + 2
            ),
            "random_minus_delphi": paired_difference(
                random_errors, kept_delphi, BOOTSTRAP_SEED + 3
            ),
        }

    import csv as _csv

    lineage_path = revision / "lowq-lineage-objects.csv"
    strata: dict[str, list[tuple[str, float]]] = {
        "gaia": [], "asas": [], "shared": [], "unshared": []
    }
    with lineage_path.open(encoding="utf-8", newline="") as stream:
        for row in _csv.DictReader(stream):
            keys = row.get("source_keys", "").lower()
            value = (row["object_id"], float(row["oracle_at_3_error_deg"]))
            if GAIA_SOURCE_KEY in keys:
                strata["gaia"].append(value)
            if ASAS_SOURCE_KEY in keys:
                strata["asas"].append(value)
            if row.get("overlapping_training_source_keys", "").strip():
                strata["shared"].append(value)
            else:
                strata["unshared"].append(value)
    for label, rows in strata.items():
        references, measured = [], []
        for name, value in rows:
            found = cohort_reference(name, table, poles)
            if found is None:
                continue
            references.append(found)
            measured.append(value)
        measured = np.array(measured)
        standard_errors = np.array(
            [axial_error_deg(axes, item) for item in references]
        )
        random_errors, random_stats = random_reference_errors(
            references, RANDOM_SEED + 7 + len(label), 200
        )
        cohort_summary[f"lowq_{label}"] = {
            "object_count": len(references),
            "unmatched_count": len(rows) - len(references),
            "delphi": summarize(measured),
            "standard": summarize(standard_errors),
            "random": random_stats,
            "standard_minus_delphi": paired_difference(
                standard_errors, measured, BOOTSTRAP_SEED + 6
            ),
            "random_minus_delphi": paired_difference(
                random_errors, measured, BOOTSTRAP_SEED + 7
            ),
        }

    sensitivity = json.loads(
        (revision / "error-channel-sensitivity.json").read_text(encoding="utf-8")
    )
    primary: dict[str, dict] = {}
    ztf_rows = sensitivity["ztf"]["objects"]
    # The main ALCDEF result is the locked transfer analysis (37.25 deg).  The
    # error-channel file holds only the variant without reported uncertainties
    # (26.82 deg), which an earlier version of this script used by mistake.
    alcdef_locked = json.loads(
        (FOLLOWUP / "data/generalization-20260910/locked-cohort"
         / "alcdef-gaia-transfer-analysis-30-20260911.json").read_text(encoding="utf-8")
    )["objects"]
    alcdef_rows = sensitivity["alcdef_gaia"]["without_error_channel_analysis"]["objects"]
    for label, rows, field, lookup in (
        ("ztf_primary", ztf_rows, "original_error_deg", poles),
        ("alcdef_primary", alcdef_locked, "oracle_at_3_error_deg", gaia),
        ("alcdef_without_uncertainties", alcdef_rows, "oracle_at_3_error_deg", gaia),
    ):
        names = [row["object_id"] for row in rows]
        measured = np.array([float(row[field]) for row in rows])
        references = [
            gaia.get(name) if lookup is gaia else cohort_reference(name, table, poles)
            for name in names
        ]
        standard_errors = np.array(
            [axial_error_deg(axes, item) for item in references]
        )
        random_errors, random_stats = random_reference_errors(
            references, RANDOM_SEED + len(label), RANDOM_DRAWS // 10
        )
        primary[label] = {
            "object_count": len(names),
            "delphi": summarize(measured),
            "standard": summarize(standard_errors),
            "random": random_stats,
            "standard_minus_delphi": paired_difference(
                standard_errors, measured, BOOTSTRAP_SEED + 4
            ),
            "random_minus_delphi": paired_difference(
                random_errors, measured, BOOTSTRAP_SEED + 5
            ),
        }

    # Does the oracle error depend on where the reference pole lies?
    latitude_rows = []
    for position, name in enumerate(identifiers):
        candidates = np.asarray(ensemble["refined_axes"][position], dtype=float)
        candidates /= np.linalg.norm(candidates, axis=1, keepdims=True)
        references = poles[name]
        latitude_rows.append(
            (
                abs(np.degrees(np.arcsin(np.clip(references[0][2], -1.0, 1.0)))),
                len(references),
                axial_error_deg(candidates, references),
                axial_error_deg(candidates, references[:1]),
            )
        )
    latitude_table = np.array(latitude_rows)

    def spearman(first: np.ndarray, second: np.ndarray) -> float:
        # Tied values receive their average rank.  The reference count takes
        # only a few distinct values, so arbitrary tie-breaking would make the
        # result depend on the sorting algorithm.
        from scipy.stats import spearmanr

        return float(spearmanr(first, second)[0])

    strata = {}
    for low, high in ((0, 30), (30, 60), (60, 91)):
        chosen = (latitude_table[:, 0] >= low) & (latitude_table[:, 0] < high)
        strata[f"{low}-{min(high, 90)}"] = {
            "object_count": int(chosen.sum()),
            "mean_all_references_deg": float(latitude_table[chosen, 2].mean()),
            "mean_first_reference_deg": float(latitude_table[chosen, 3].mean()),
            "mean_reference_count": float(latitude_table[chosen, 1].mean()),
        }
    latitude = {
        "strata": strata,
        "spearman_latitude_all_references": spearman(
            latitude_table[:, 0], latitude_table[:, 2]
        ),
        "spearman_latitude_first_reference": spearman(
            latitude_table[:, 0], latitude_table[:, 3]
        ),
        "spearman_latitude_reference_count": spearman(
            latitude_table[:, 0], latitude_table[:, 1]
        ),
        "note": (
            "the flat dependence is not produced by low-latitude asteroids "
            "having more catalog solutions to match"
        ),
    }

    payload = {
        "schema": "delphi.reference-baselines.v1",
        "latitude_diagnostics": latitude,
        "cohorts": cohort_summary,
        "primary_variants": primary,
        "standard_pole_count": len(STANDARD_POLES_DEG),
        "standard_axis_count": int(axes.shape[0]),
        "standard_poles_deg": [list(pole) for pole in STANDARD_POLES_DEG],
        "random_draws": RANDOM_DRAWS,
        "random_seed": RANDOM_SEED,
        "delphi": summarize(delphi),
        "standard": summarize(standard),
        "random": {
            "mean_deg": float(np.mean(random_means)),
            "mean_scatter_deg": float(np.std(random_means)),
            "median_deg": float(np.mean(random_medians)),
            "within_twenty_percent": float(np.mean(random_within)),
            "randomization_interval95_low_deg": float(
                np.percentile(random_means, 2.5)
            ),
            "randomization_interval95_high_deg": float(
                np.percentile(random_means, 97.5)
            ),
            "method": "three_independent_uniform_sphere_axes_per_identity",
            "resamples": RANDOM_DRAWS,
            "object_count": len(identifiers),
        },
        "standard_minus_delphi": paired_difference(standard, delphi, BOOTSTRAP_SEED),
        "random_minus_delphi": paired_difference(
            random_per_object, delphi, BOOTSTRAP_SEED + 1
        ),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    display_generator = np.random.default_rng(RANDOM_SEED + 99)
    single_draw = np.empty(len(identifiers))
    for position, name in enumerate(identifiers):
        trial = display_generator.normal(size=(3, 3))
        trial /= np.linalg.norm(trial, axis=1, keepdims=True)
        single_draw[position] = axial_error_deg(trial, poles[name])
    np.savez(
        OUT / "reference-baselines-per-object.npz",
        random_single_draw_errors_deg=single_draw,
        object_ids=np.array(identifiers),
        delphi_errors_deg=delphi,
        standard_errors_deg=standard,
        random_errors_deg=random_per_object,
    )
    (OUT / "reference-baselines.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
