"""Compute Paper I median replacements from frozen per-asteroid evidence.

Run with ``taskset -c 12-23 nice -n 19 python3 -m repro.own_period.median_stats``.
No inference, inversion, manuscript changes, or imports of the refine-chain code.
The input list is the scope: each line is computed, skipped, or explained as
not computable. Existing scorer modules are never executed.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import errno
from functools import cached_property
import hashlib
import json
from pathlib import Path
import random
import re
import shlex
import subprocess
import sys

import numpy as np

from repro import compute_reference_baselines as reference


REPO = Path(__file__).resolve().parents[2]
from repro.own_period.reporting_inputs import DataPath, load_npz, select
from repro import network_root as network

FOLLOWUP = DataPath("../..")
NOTEBOOKS = FOLLOWUP.parent
OWN = DataPath()
OUTPUT = OWN / "median-stats.json"
PAPER = NOTEBOOKS / "paper-rev-20261021/sample701.tex"
RESAMPLES = 10_000
CONDITION = "two_d-block_count-1-per_block_cap-None"

SKIPPED = {
    "WorkflowErrorReductionPercent": "Derived percentage reduction, not a mean pole error in degrees.",
    "KThreeInferenceMean": "Mean inference time in seconds, not a pole error.",
    "AreaThreeMean": "Search-area percentage, not a pole error.",
    "AreaStandardMean": "Search-area percentage, not a pole error.",
    "PhaethonOracleError": "One asteroid's error, not a mean over asteroids.",
    "ScErosLooMinError": "Minimum across leave-one-apparition-out variants, not a mean.",
    "ScErosSimilarMaxError": "Rounded-up maximum across context objects, not a mean.",
}

# Explicit semantic audit: several printed pole-error summaries have neither
# "Mean" nor "Error" in their names. Low/High on Removal are seed ranges,
# CheckReassign bounds are null-draw quantiles, and latitude bounds name bins.
ADDITIONAL_BASES = {
    "KThreeSwapGap", "FollowGridBaseline", "FollowBlockTenCapFifty",
    "FollowBlockTenCapTwenty", "FollowBlockTenCapFiftyChange",
    "SampFiveGain", "SampThreeGain", "LabelShuffleBefore", "LabelShuffleAfter",
    "FollowAlcdefCandidate", "CheckReassignLow", "CheckReassignHigh",
    "FollowAlcdefStandard",
    "InterpAAHighDelphi", "InterpAAHighAA", "InterpAALowDelphi", "InterpAALowAA",
    "InterpAAGain", "InterpAAOverStandard", "KThreeFoldMin", "KThreeFoldMax",
    "KThreeSeedMin", "KThreeSeedMax", "CountOneMean", "CountTwoMean", "CountThreeMean",
    "SplitLatitude", "SplitLongitude", "NoiseSmall", "NoiseLarge", "PeriodHalf", "SamplingTinyDifference",
    "PeriodDouble", "PeriodSensitivityMaxChange", "BroadGridAgreementDifference",
    "InsightLowqFoundLatHigh", "InsightLowqFoundLatLow",
    "InsightLowqFoundStandardLatHigh", "InsightLowqFoundStandardLatLow",
}


def additional_macro(name: str) -> bool:
    base = re.sub(r"(?:Low|High)$", "", name)
    return (name in ADDITIONAL_BASES or base in ADDITIONAL_BASES
            or name.startswith("Removal")
            or bool(re.fullmatch(r"KThree(?:Standard|Random|Atlas)Improvement(?:Low|High)?", name))
            or bool(re.fullmatch(r"Follow(?:Lowq\w*Found|ZtfFound\w*|Alcdef\w*)Difference(?:Positive)?(?:Low|High)?", name)))


def skip_reason(name: str, source: str) -> str:
    if name in SKIPPED:
        return SKIPPED[name]
    if name == "LabelShuffleRandom":
        return "Analytic population expectation 2/3 rad for random axes; no finite asteroid set."
    if "AbsBeta" in name or name.endswith("AbsBeta"):
        return "Absolute pole latitude, not pole error."
    if "Period" in name or "Span" in name or "Days" in name:
        return "Period, phase drift, time span, or associated count/setting; not mean pole error."
    if re.search(r"Seconds|Speed|Time|Inference", name):
        return "Runtime or speed statistic, not pole error."
    if "Area" in name:
        return "Search area, not pole error."
    if re.search(r"Within|Count|Minima|N$|Starts|Cap|Size|Observations|Detections|Trials", name):
        return "Count, hit fraction, threshold, or sampling/solver setting; not mean pole error."
    if re.search(r"Rms|Residual|Withheld", name):
        return "Photometric residual or ratio, not pole error."
    if name.startswith("Sc") or name.startswith("Spacecraft") or name == "PhaethonOracleError":
        return "Individual-object quantity, existing median, or non-error physical quantity; not a mean over asteroids."
    if re.search(r"Median|Minimum|Maximum|Spread|Oracle(?:Min|Max|First|Third)", name):
        return "Existing median, dispersion, extremum, or quantile; not a mean or bound of means."
    return "Not a mean pole error or a difference/range/bound of mean pole errors (count, correlation, physical quantity, or setting)."


def audit_manuscript(path: Path, original: list[dict]) -> tuple[list[dict], list[dict]]:
    """Read definitions and printed uses, keeping the final definition per name.

    Source comments belong only to the contiguous comment block immediately
    preceding a definition; they must not leak to unrelated macros.
    """
    text = path.read_text(encoding="utf-8")
    body = text.split(r"\begin{document}", 1)[1]
    body = re.sub(r"(?<!\\)%[^\n]*", "", body)
    definitions, comments = {}, []
    for line_no, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith("%"):
            comments.append(line.lstrip()[1:].strip())
            continue
        match = re.fullmatch(r"\\newcommand\{\\(\w+)\}\{(.*)\}", line)
        if match:
            name, value = match.groups()
            definitions[name] = (value, " | ".join(c for c in comments if c.startswith("Source")), line_no)
        comments = []
    original_names = {item["mean_macro"] for item in original}
    items, audit = list(original), []
    for name, (value, source, line_no) in definitions.items():
        if not re.search(r"\\" + re.escape(name) + r"\b", body):
            continue
        included = name in original_names and name not in SKIPPED or additional_macro(name)
        row = {"macro": name, "line": line_no, "definition": value, "source_comment": source,
               "included": included, "reason": "M1 entry retained" if name in original_names and included
               else "Printed summary of asteroid pole errors; producer audited" if included
               else skip_reason(name, source)}
        audit.append(row)
        if included and name not in original_names:
            numeric = re.match(r"[-+]?\d+(?:\.\d+)?", value.replace(r"\ensuremath{-}", "-"))
            if numeric is None:
                raise ValueError(f"No numeric definition for included macro {name}")
            number = numeric.group()
            items.append({"mean_macro": name, "mean_value_in_paper": float(number),
                          "paper_numeric_text": number, "paper_precision": len(number.split(".")[1]) if "." in number else 0,
                          "input_source_comment": source, "new_in_m1b": True})
    return items, audit


def read_json(path: Path):
    from repro.own_period.reporting_inputs import read_json as selected_json
    return selected_json(path)


def parse_inputs(path: Path) -> list[dict]:
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r"(\w+)\s*=\s*(.*?)(?:\s+%\s*(.*))?\s*", line)
        if match is None:
            raise ValueError(f"Unrecognized input line: {line!r}")
        name, printed, source = match.groups()
        numeric = printed.replace(r"\ensuremath{-}", "-")
        number = re.match(r"[-+]?\d+(?:\.\d+)?", numeric)
        if number is None:
            raise ValueError(f"No paper value: {line!r}")
        text = number.group()
        result.append({
            "mean_macro": name,
            "mean_value_in_paper": float(text),
            "paper_numeric_text": text,
            "paper_precision": len(text.split(".")[1]) if "." in text else 0,
            "input_source_comment": source or "",
        })
    names = [r["mean_macro"] for r in result]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate macro in the input list")
    return result


def check_printed_mean(recomputed: float, printed: str) -> dict:
    """Compare at the paper's actual decimal precision, including trailing zeroes."""
    precision = len(printed.split(".")[1]) if "." in printed else 0
    rounded = f"{recomputed:.{precision}f}"
    matches = rounded == printed.lstrip("+")
    return {"matches_printed_rounding": matches, "recomputed_printed": rounded,
            "flag": None if matches else "MEAN_RECOMPUTATION_MISMATCH"}


def bootstrap_indices(n: int, seed: int, resamples: int = RESAMPLES,
                      *, folds=None, order: str = "iid") -> np.ndarray:
    """Reproduce each producer's RNG and loop order, not just its seed."""
    if n < 1 or resamples < 1:
        raise ValueError("Bootstrap requires positive n and resamples")
    if order == "iid":
        return np.random.default_rng(seed).integers(0, n, size=(resamples, n))
    folds = np.asarray(folds)
    if folds.shape != (n,):
        raise ValueError("Fold labels must align with asteroids")
    groups = [np.flatnonzero(folds == f) for f in sorted(set(folds.tolist()))]
    if order == "python_fold":
        rng = random.Random(seed)
        return np.asarray([[rng.choice(group) for group in groups for _ in group]
                           for _ in range(resamples)], dtype=np.int64)
    rng = np.random.default_rng(seed)
    if order == "numpy_fold_major":
        return np.concatenate([rng.choice(g, size=(resamples, len(g))) for g in groups], axis=1)
    if order == "numpy_replicate_major":
        return np.asarray([np.concatenate([g[rng.integers(0, len(g), size=len(g))]
                                           for g in groups]) for _ in range(resamples)])
    raise ValueError(f"Unknown resampling order: {order}")


def compute_statistics(first, second=None, *, indices=None) -> dict:
    """Median(A) or median(A)-median(B), always resampling paired identities."""
    a = np.asarray(first, dtype=float)
    if a.ndim != 1 or a.size == 0 or not np.isfinite(a).all():
        raise ValueError("Errors must be a nonempty finite vector; select failures explicitly")
    result = {"mean_recomputed": float(a.mean()), "median": float(np.median(a)),
              "n": int(a.size), "interval": None, "mean_interval_recomputed": None,
              "statistic": "median"}
    if second is not None:
        b = np.asarray(second, dtype=float)
        if b.shape != a.shape or not np.isfinite(b).all():
            raise ValueError("Paired errors must have the same shape and be finite")
        result.update(mean_recomputed=float(np.mean(a - b)),
                      median=float(np.median(a) - np.median(b)),
                      median_of_paired_differences=float(np.median(a - b)),
                      statistic="difference_of_medians")
    if indices is not None:
        draw = np.asarray(indices)
        if draw.ndim != 2 or draw.shape[1] != a.size:
            raise ValueError("Each bootstrap draw must resample the original asteroid count")
        boot = np.median(a[draw], axis=1)
        mean_boot = a[draw].mean(axis=1)
        if second is not None:
            boot -= np.median(b[draw], axis=1)
            mean_boot = (a - b)[draw].mean(axis=1)
        result["interval"] = np.percentile(boot, (2.5, 97.5)).tolist()
        result["mean_interval_recomputed"] = np.percentile(mean_boot, (2.5, 97.5)).tolist()
    return result


def randomized_paired_statistics(values, own, indices) -> dict:
    """M1 randomization convention with a paired asteroid bootstrap.

    For each bootstrap identity draw, use integer multiplicities to find the
    median in every fixed original randomization draw, then average those
    medians. This exactly equals expanding values[:, indices[b]], while
    bounding memory by one randomization matrix rather than a three-axis cube.
    """
    values, own, indices = np.asarray(values, float), np.asarray(own, float), np.asarray(indices)
    if values.ndim != 2 or values.shape[1:] != own.shape or not np.isfinite(values).all() or not np.isfinite(own).all():
        raise ValueError("Randomization draws must align with finite paired asteroid errors")
    if indices.ndim != 2 or indices.shape[1] != len(own):
        raise ValueError("Bootstrap draws must preserve the paired asteroid count")
    order = np.argsort(values, axis=1)
    sorted_values = np.take_along_axis(values, order, axis=1)
    rows = np.arange(len(values))
    n = len(own)
    boot = np.empty(len(indices))
    for k, draw in enumerate(indices):
        counts = np.bincount(draw, minlength=n)
        cumulative = np.cumsum(counts[order], axis=1, dtype=np.int32)
        left = np.argmax(cumulative > (n - 1) // 2, axis=1)
        right = np.argmax(cumulative > n // 2, axis=1)
        boot[k] = np.mean((sorted_values[rows, left] + sorted_values[rows, right]) / 2) - np.median(own[draw])
    return {"median": float(np.median(values, axis=1).mean() - np.median(own)),
            "interval": np.percentile(boot, (2.5, 97.5)).tolist(),
            "median_of_paired_differences": float(np.median(values - own, axis=1).mean()),
            "statistic": "mean_of_randomization_draw_differences_of_medians"}


def source_path(comment: str) -> Path | None:
    match = re.search(r"(?:/[^:\n]*?)?DeLPHI-followup/[^:\n]+?\.json", comment)
    if match:
        path = Path(match.group())
        return path if path.is_absolute() else Path(NOTEBOOKS) / path
    return None


def own_scores_path(cohort: str) -> Path:
    # The rerun scores ZTF and ALCDEF into separate files; the published run kept both in one.
    if cohort == "alcdef" and network.switched():
        return Path(network.NetworkPath("results", "alcdef", "own-period-scores.json"))
    return OWN / "own-period-scores.json"


class Evidence:
    def __init__(self):
        self.cache = {}

    def json(self, path: Path):
        if path not in self.cache:
            self.cache[path] = read_json(path)
        return self.cache[path]

    @cached_property
    def poles(self):
        return reference.reference_poles()

    @cached_property
    def gaia(self):
        return reference.gaia_reference_poles()

    @cached_property
    def table(self):
        return reference.damit_table_poles()

    @cached_property
    def baseline(self):
        path = FOLLOWUP / "source/repro/baselines_out/reference-baselines-per-object.npz"
        with load_npz(path, allow_pickle=False) as data:
            return {k: data[k] for k in data.files}

    @cached_property
    def ensemble(self):
        path = OWN / "maps/own-period/evaluations/real-oof-ensemble.npz"
        with load_npz(path, allow_pickle=False) as data:
            return {k: data[k] for k in ("object_ids", "folds", "oracle_errors_deg",
                                         "grid_oracle_errors_deg", "refined_axes")}

    def random_errors(self, ids, references, seed: int, draws: int):
        """The original draw-major, identity-major Gaussian stream, in CPU batches."""
        rng = np.random.default_rng(seed)
        values = np.empty((draws, len(ids)))
        for start in range(0, draws, 250):
            stop = min(start + 250, draws)
            axes = rng.normal(size=(stop - start, len(ids), 3, 3))
            axes /= np.linalg.norm(axes, axis=3, keepdims=True)
            for j, oid in enumerate(ids):
                cosine = np.abs(axes[:, j] @ references[oid].T).max(axis=(1, 2))
                values[start:stop, j] = np.degrees(np.arccos(np.clip(cosine, 0, 1)))
        return values

    def randomized(self, values, ids, sources, method, seed, draws):
        medians = np.median(values, axis=1)
        return {
            "mean_recomputed": float(values.mean()), "median": float(medians.mean()),
            "interval": None, "mean_interval_recomputed": None, "n": len(ids),
            "statistic": "mean_of_randomization_draw_medians",
            "median_of_per_asteroid_expected_errors": float(np.median(values.mean(axis=0))),
            "randomization_interval95": np.percentile(medians, (2.5, 97.5)).tolist(),
            "randomization_mean_interval_recomputed": np.percentile(values.mean(axis=1), (2.5, 97.5)).tolist(),
            "randomization_seed": seed, "randomization_draws": draws,
            "object_ids": list(ids), "source_files": [str(p) for p in sources],
            "method": method + " Median is averaged over the original randomization draws. "
                      "The randomization interval describes draws, not an asteroid-bootstrap CI; "
                      "the median of per-asteroid expected errors is separately labelled.",
        }

    def stat(self, first, ids, sources, method, *, second=None,
             seed=None, folds=None, order="iid", mean_interval=None):
        if len(ids) != len(first) or len(set(ids)) != len(ids):
            raise ValueError("One unique identity is required for each error")
        indices = None if seed is None else bootstrap_indices(len(ids), seed, folds=folds, order=order)
        result = compute_statistics(first, second, indices=indices)
        result.update(object_ids=list(ids), source_files=[str(select(p)) for p in sources], method=method)
        if indices is not None:
            result["bootstrap"] = {"seed": seed, "resamples": RESAMPLES,
                                   "rng_and_order": order, "unit": "asteroid" if folds is None else "asteroid_within_fold",
                                   "interval": "95% percentile, numpy linear interpolation"}
        if mean_interval is not None:
            result["mean_interval_in_source"] = list(mean_interval)
            result["mean_interval_matches_source"] = bool(np.allclose(
                result["mean_interval_recomputed"], mean_interval, atol=1e-10, rtol=0))
        return result

    def reassignment(self):
        source = OWN / "maps/own-period/analyses/reference-reassignment.json"
        frozen = FOLLOWUP / "data/reliability-20260915/run-1/scored-rows.json"
        rows = self.json(frozen)["payload"]
        rows = [r for r in rows if r["condition"] == "full" and r["repeat"] == 0 and r["status"] == "ok"]
        e = self.ensemble
        index = {str(oid): i for i, oid in enumerate(e["object_ids"])}
        groups = defaultdict(list)
        for r in rows:
            groups[(r["fold"], len(self.poles[r["object_id"]]))].append(r["object_id"])
        old = self.json(source)
        values = np.empty((old["draws"], len(rows)))
        rng = np.random.default_rng(old["seed"])
        for k in range(old["draws"]):
            offset = 0
            for ids in groups.values():
                assigned = rng.permutation([self.poles[oid] for oid in ids])
                axes = np.asarray([e["refined_axes"][index[oid]] for oid in ids], dtype=float)
                axes /= np.linalg.norm(axes, axis=2, keepdims=True)
                assigned /= np.linalg.norm(assigned, axis=2, keepdims=True)
                cosine = np.abs(np.einsum("iaj,irj->iar", axes, assigned)).max(axis=(1, 2))
                values[k, offset:offset + len(ids)] = np.degrees(np.arccos(np.clip(cosine, 0, 1)))
                offset += len(ids)
        # Columns follow the original insertion-ordered (fold, reference-count) strata.
        ids = [oid for group in groups.values() for oid in group]
        result = self.randomized(values, ids, [source, frozen,
            OWN / "maps/own-period/evaluations/real-oof-ensemble.npz", reference.CATALOG,
            FOLLOWUP / "source-op-maps/repro/own_period/maps/analyses.py",
            REPO / "lc_pipeline/k3/internal_validation_reporting.py"],
            "Fixed own-period predictions; permute whole reference sets within (fold, reference-count) "
            "strata in original insertion order; same axial normalization and min over candidates/references.",
            old["seed"], old["draws"])
        result["null_mean_draws_match_source"] = bool(np.allclose(
            values.mean(axis=1), old["null_draws_mean_error_deg"], atol=1e-10, rtol=0))
        return result

    def resolve(self, item: dict) -> dict:
        name = item["mean_macro"]
        if item.get("new_in_m1b"):
            return self.resolve_additional(item)
        key = re.sub(r"(?:Low|High)$", "", name)
        if key not in self.cache:
            self.cache[key] = self.resolve_statistic(item)
        result = dict(self.cache[key])
        if name.endswith(("Low", "High")):
            endpoint = 0 if name.endswith("Low") else 1
            result["parent_median"] = result["median"]
            result["parent_mean_recomputed"] = result["mean_recomputed"]
            result["median"] = result["interval"][endpoint]
            result["mean_recomputed"] = result["mean_interval_recomputed"][endpoint]
            result["statistic"] += "_ci95_" + ("low" if endpoint == 0 else "high")
            result["method"] += " This macro is the corresponding interval endpoint."
        return result

    def resolve_additional(self, item: dict) -> dict:
        name = item["mean_macro"]
        intrinsic = (name.startswith(("Removal", "Insight", "CheckReassign"))
                     or name in ("InterpAAHighDelphi", "InterpAAHighAA", "InterpAALowDelphi", "InterpAALowAA"))
        base = name if intrinsic else re.sub(r"(?:Low|High)$", "", name)
        if base == "FollowAlcdefStandard":
            base = "FollowAlcdefStandardDifference"
        key = ("additional", base)
        if key not in self.cache:
            self.cache[key] = self.additional_statistic(base, item)
        result = dict(self.cache[key])
        if base != name:
            endpoint = 0 if name.endswith("Low") else 1
            result.update(parent_median=result["median"], parent_mean_recomputed=result["mean_recomputed"],
                          median=result["interval"][endpoint],
                          mean_recomputed=result["mean_interval_recomputed"][endpoint],
                          statistic=result["statistic"] + ("_ci95_low" if endpoint == 0 else "_ci95_high"))
        return result

    def pair_from_score(self, name: str) -> dict:
        main = name.startswith("KThree")
        cohort = "main_sample" if main else "ztf" if "Ztf" in name else "alcdef"
        comp = "random" if "Random" in name else "fitted_axes" if "Atlas" in name else "standard"
        path = OWN / "own-period-scores-main.json" if main else own_scores_path(cohort)
        block = self.json(path)[cohort]
        rows = block["objects"]
        # The negative ALCDEF StandardDifference is standard-own; Positive and
        # AtlasDifference are own-comparator, as confirmed in the printed text.
        reverse = not main and ("Atlas" in name or "Positive" in name)
        source_key = comp + "_minus_own_failed_" + ("at_90" if main or cohort == "alcdef" else "excluded")
        source = block[source_key]
        if cohort == "ztf":
            rows = [r for r in rows if r["own_error_deg"] is not None]
        ids = [r["object_id"] for r in rows]
        own = np.array([90.0 if r["own_error_deg"] is None else r["own_error_deg"] for r in rows])
        sources = [path, REPO / "repro/own_period/score.py"]
        draws = None
        if main:
            if comp == "fitted_axes":
                p = FOLLOWUP / "inputs/frozen-artifacts/k3-2408c563/comparators/v1-comparators.npz"
                with load_npz(p, allow_pickle=False) as d:
                    lookup = dict(zip(d["object_ids"].astype(str), d["atlas_errors_deg"]))
                sources.append(p)
            else:
                lookup = dict(zip(self.baseline["object_ids"].astype(str), self.baseline[comp + "_errors_deg"]))
                sources.append(FOLLOWUP / "source/repro/baselines_out/reference-baselines-per-object.npz")
            if comp == "random":
                draw_ids = self.baseline["object_ids"].astype(str).tolist()
                draws = self.random_errors(draw_ids, self.poles, reference.RANDOM_SEED, reference.RANDOM_DRAWS)
                draws = draws[:, [draw_ids.index(i) for i in ids]]
        else:
            p = OWN / "nettests/no-uncertainty/score-own.json"
            lookup = {r["object_id"]: r["standard_error_deg"] for r in self.json(p)["variants"][cohort]["objects"]}
            sources.append(p)
            if comp == "fitted_axes":
                p = (FOLLOWUP / "data/publication-revision-20260917-r2/error-channel-sensitivity.json" if cohort == "ztf"
                     else FOLLOWUP / "data/generalization-20260910/locked-cohort/alcdef-gaia-transfer-analysis-30-20260911.json")
                d = self.json(p)
                lookup = {r["object_id"]: r["train_only_atlas_error_deg" if cohort == "ztf" else "atlas_oracle_at_3_error_deg"]
                          for r in (d["ztf"]["objects"] if cohort == "ztf" else d["objects"])}
                sources.append(p)
            elif comp == "random":
                all_ids = [r["object_id"] for r in block["objects"]]
                draws = self.random_errors(all_ids, self.poles, 20260925, 1000)
                draws = draws[:, [all_ids.index(i) for i in ids]]
                lookup = dict(zip(ids, draws.mean(axis=0)))
        first = np.array([lookup[i] for i in ids])
        low_high = [source["ci95_low_deg"], source["ci95_high_deg"]]
        if reverse:
            low_high = [-low_high[1], -low_high[0]]
        # score.cohort_block iterates published, standard, random, fitted_axes.
        seed = 20261021 + {"standard": 2, "random": 4, "fitted_axes": 6}[comp]
        if cohort == "ztf":
            seed += 1
        result = self.stat(own if reverse else first, ids, sources,
            f"{cohort}: {'own minus ' + comp if reverse else comp + ' minus own'}; identical frozen identities, "
            "failed periods excluded for ZTF, 90 degrees for the source at-90 comparisons (no failures here). "
            "Paired asteroid bootstrap with score.cohort_block seed and original IID draw stream.",
            second=first if reverse else own, seed=seed, mean_interval=low_high)
        if draws is not None:
            if not np.allclose(first, draws.mean(axis=0), atol=1e-10, rtol=0):
                raise ValueError("Regenerated random per-asteroid expectations differ from the source comparison")
            result["source_files"].extend([str(reference.CATALOG), str(REPO / "repro/compute_reference_baselines.py")])
            result.update(expected_error_difference_of_medians=result["median"],
                          expected_error_paired_bootstrap_interval95=result["interval"],
                          expected_error_median_of_paired_differences=result["median_of_paired_differences"],
                          randomization_comparison_convention="M1: average the per-randomization-draw differences of asteroid medians; paired asteroid bootstrap keeps all original random draws fixed and resamples the same identities in both arms. Median paired differences also average the per-draw asteroid medians.",
                          randomization_difference_interval95=np.percentile(np.median(draws, axis=1) - np.median(own), (2.5, 97.5)).tolist(),
                          randomization_interval_kind="randomization draws, not a bootstrap CI",
                          randomization_seed=reference.RANDOM_SEED if main else 20260925,
                          randomization_draws=len(draws))
            print(f"Paired randomization bootstrap for {name}: {RESAMPLES} asteroid resamples, {len(draws)} fixed random draws", flush=True)
            result.update(randomized_paired_statistics(draws, own, bootstrap_indices(len(ids), seed)))
            result["method"] += " Follow M1's average-draw-median convention, using the same paired asteroid bootstrap for its interval, conditional on the fixed original randomization draws. The difference of medians of per-asteroid expected errors is separately retained with its paired bootstrap interval; the source mean CI still validates the original expected-error comparison."
        return result

    def additional_statistic(self, name: str, item: dict) -> dict:
        if name.startswith("KThree") and "Improvement" in name or name.startswith(("FollowZtf", "FollowAlcdef")) and "Difference" in name:
            return self.pair_from_score(name)
        if name == "FollowAlcdefCandidate":
            return self.resolve_statistic({"mean_macro": "FollowAlcdefMean", "input_source_comment": ""})
        if name.startswith("FollowLowq") and "Difference" in name:
            group = re.search(r":: (lowq\w*)/", item["input_source_comment"]).group(1)
            path = OWN / "lowq/scores-own/baselines-excluded.json"
            a, ids, sources = self.lowq_vectors(group)
            b = np.array([reference.axial_error_deg(reference.standard_axes(), reference.cohort_reference(i, self.table, self.poles)) for i in ids])
            source = self.json(path)[group]["standard_minus_delphi"]
            return self.stat(b, ids, [path, *sources], f"{group}: standard minus DeLPHI on the same successful-period/reference-matched asteroids; original IID paired asteroid bootstrap.",
                             second=a, seed=source["seed"], mean_interval=[source["ci95_low_deg"], source["ci95_high_deg"]])
        if name.startswith("CheckReassign"):
            result = dict(self.cache.get("CheckReassignMean") or self.reassignment())
            endpoint = 0 if name.endswith("Low") else 1
            result.update(parent_median=result["median"], parent_mean_recomputed=result["mean_recomputed"],
                          median=result["randomization_interval95"][endpoint],
                          mean_recomputed=result["randomization_mean_interval_recomputed"][endpoint],
                          statistic="randomization_draw_median_quantile_" + ("025" if endpoint == 0 else "975"),
                          pre_specified_on_mean=True)
            return result
        if name == "KThreeSwapGap" or name.startswith("Removal"):
            control = ("input-swap" if name == "KThreeSwapGap" else
                       re.search(r"summary/controls/([^/]+)/", item["input_source_comment"]).group(1))
            folder = OWN / "nettests/controls-own"
            results, sources = [], [FOLLOWUP / "source-op-nettests/repro/own_period/summarize_controls.py"]
            for seed in (17, 42, 137, 777, 2027):
                a, b, ids = [], [], []
                for fold in range(5):
                    p, q = folder / f"real-fold-{fold}-seed-{seed}-{control}.npz", folder / f"real-fold-{fold}-seed-{seed}.npz"
                    with load_npz(p, allow_pickle=False) as d, load_npz(q, allow_pickle=False) as raw:
                        if not np.array_equal(d["object_ids"], raw["object_ids"]):
                            raise ValueError("Control pairing changed")
                        ids.extend(d["object_ids"].astype(str).tolist()); a.extend(d["oracle_errors_deg"]); b.extend(raw["oracle_errors_deg"])
                    sources.extend([p, q])
                results.append(self.stat(a, ids, [], "", second=b))
            mean_gaps = [r["mean_recomputed"] for r in results]
            med_gaps = [r["median"] for r in results]
            endpoint = 0 if name.endswith("Low") else 1 if name.endswith("High") else None
            result = dict(results[0])
            result.update(mean_recomputed=float(np.mean(mean_gaps) if endpoint is None else min(mean_gaps) if endpoint == 0 else max(mean_gaps)),
                          median=float(np.mean(med_gaps) if endpoint is None else min(med_gaps) if endpoint == 0 else max(med_gaps)),
                          median_of_paired_differences=float(np.mean([r["median_of_paired_differences"] for r in results])),
                          statistic="mean_of_seed_differences_of_medians" if endpoint is None else "seed_range_endpoint_of_differences_of_medians",
                          seed_median_differences=dict(zip(map(str, (17,42,137,777,2027)), med_gaps)),
                          seed_mean_differences=dict(zip(map(str, (17,42,137,777,2027)), mean_gaps)),
                          seed_median_of_paired_differences=[r["median_of_paired_differences"] for r in results],
                          seed_range=[min(med_gaps), max(med_gaps)], source_files=list(map(str, sources)),
                          method="Same 170 asteroids per single-network seed, folds pooled in order 0..4. Compute median(control)-median(raw) per seed, then average five seeds. Low/High are extrema across seeds, not confidence limits; median paired differences are also averaged across seeds.",
                          pre_specified_on_mean=True)
            return result
        if name.startswith(("Noise", "FollowGrid", "FollowBlock", "SampFiveGain", "SampThreeGain", "SamplingTinyDifference")):
            return self.reliability_statistic(name)
        if name in ("LabelShuffleBefore", "LabelShuffleAfter"):
            folder = FOLLOWUP / "inputs/frozen-artifacts/k3-definitive-7874092/evaluations"
            p = folder / ("synthetic-test-seed-17.npz" if name.endswith("Before") else "synthetic-label-shuffle-test-seed-17.npz")
            with load_npz(p, allow_pickle=False) as d:
                return self.stat(d["oracle_errors_deg"], d["object_ids"].astype(str).tolist(), [p, REPO / "lc_pipeline/k3/release.py"],
                                 "Same 2000 simulated test asteroids; seed 17 alone, correct-label or shuffled-label network. No published bootstrap interval.")
        if name.startswith("Count"):
            return self.candidate_statistic(name)
        if name.startswith(("KThreeFold", "KThreeSeed")):
            return self.group_range(name)
        if name.startswith("InterpAA"):
            path = OWN / "maps/own-period/analyses/amplitude-aspect.json"
            rows = self.json(path)
            paired = name in ("InterpAAGain", "InterpAAOverStandard")
            if not paired:
                low, high = (.8, 1.0) if "High" in name else (-1.0, .5)
                rows = [r for r in rows if low <= r["map_correlation"] < high]
            a = ("standard_oracle_deg" if name == "InterpAAOverStandard" else
                 "amplitude_aspect_oracle_deg" if paired or name.endswith("AA") else "delphi_oracle_deg")
            b = "delphi_oracle_deg" if name == "InterpAAGain" else "amplitude_aspect_oracle_deg"
            summary = self.json(OWN / "maps/own-period/analyses/amplitude-aspect-paired.json")
            interval = summary["amplitude_aspect_minus_delphi_deg" if name == "InterpAAGain" else "standard_minus_amplitude_aspect_deg"][1:] if paired else None
            return self.stat([r[a] for r in rows], [r["object_id"] for r in rows], [path, FOLLOWUP / "source-op-maps/repro/own_period/maps/analyses.py"],
                             "Same amplitude-aspect usable-lightcurve sample; agreement bins [-1,.5) and [.8,1). Paired gain is amplitude-aspect minus DeLPHI; over-standard is standard minus amplitude-aspect.",
                             second=[r[b] for r in rows] if paired else None, seed=20260930 if paired else None, mean_interval=interval)
        if name.startswith("InsightLowq"):
            return self.latitude_statistic(name)
        if name.startswith("Split"):
            return self.coordinate_statistic(name)
        if name.startswith("Period"):
            return self.period_statistic(name)
        if name == "BroadGridAgreementDifference":
            return self.broad_grid_statistic()
        raise KeyError(f"No additional resolver for {name}")

    def lowq_vectors(self, group):
        path = OWN / "lowq/scores-own/census-analysis-excluded.json"
        lineage = OWN / "lowq/scores-own/lineage-excluded/objects.csv"
        if group == "lowq":
            rows = self.json(path)["objects"]
        else:
            with lineage.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            label = group.removeprefix("lowq_")
            if label in ("gaia", "asas"):
                code = reference.GAIA_SOURCE_KEY if label == "gaia" else reference.ASAS_SOURCE_KEY
                rows = [r for r in rows if code in r.get("source_keys", "").lower()]
            else:
                rows = [r for r in rows if bool(r.get("overlapping_training_source_keys", "").strip()) == (label == "shared")]
        rows = [r for r in rows if reference.cohort_reference(r["object_id"], self.table, self.poles) is not None]
        return np.array([float(r["oracle_at_3_error_deg"]) for r in rows]), [r["object_id"] for r in rows], [path, lineage, REPO / "repro/compute_reference_baselines.py", FOLLOWUP / "source-op-lowq/repro/own_period/lowq_score.py"]

    def reliability_statistic(self, name):
        path = OWN / "failed-excluded-derived/reliability-paired.json"
        conditions = self.json(path)["conditions"]
        condition = {
            "NoiseSmall": "full-noise_sigma-0.03", "NoiseLarge": "full-noise_sigma-0.1",
            "FollowGridBaseline": "grid-full",
            "FollowBlockTenCapFifty": "two_d-block_count-10-per_block_cap-50",
            "FollowBlockTenCapFiftyChange": "two_d-block_count-10-per_block_cap-50",
            "FollowBlockTenCapTwenty": "two_d-block_count-10-per_block_cap-20",
            "SampFiveGain": "two_d-block_count-5-per_block_cap-None",
            "SampThreeGain": "two_d-block_count-3-per_block_cap-None",
            "SamplingTinyDifference": "observation_cap-cap-20",
        }[name]
        rows = conditions[condition]["objects"]
        ids = [r["object_id"] for r in rows]
        scored_path = OWN / "reliability/score/scored-rows-own.json"
        scored = self.json(scored_path)
        if name.startswith("Noise") or name == "FollowGridBaseline":
            # excluded_summary preserves first-success insertion order, whereas
            # grid_table explicitly sorts identities for the block conditions.
            order_ids = list(dict.fromkeys(r["object_id"] for r in scored if r["condition"] == condition and r["status"] == "ok"))
            by_id = {r["object_id"]: r for r in rows}
            rows = [by_id[i] for i in order_ids]
            ids = order_ids
        folds = {r["object_id"]: r["fold"] for r in scored if r["condition"] == condition}
        a = [r["delphi_mean_deg"] for r in rows]
        b, source_interval, order, seed = None, None, "numpy_replicate_major", 20260915
        summary_path = OWN / "reliability/score/reliability-own-period.json"
        summary = self.json(summary_path)["failed_excluded"]
        if name.startswith("Samp"):
            # Both Samp*Gain and SamplingTinyDifference use the fixed standard
            # vector, not an independently resampled comparator.
            a, b = [r["standard_error_deg"] for r in rows], a
            pair = conditions[condition]["standard_minus_delphi"]
            source_interval = [pair["ci95_low_deg"], pair["ci95_high_deg"]]
            seed, order, folds = 20260930, "iid", None
        elif name.endswith("Change"):
            baseline = {r["object_id"]: r["delphi_mean_deg"] for r in conditions["grid-full"]["objects"]}
            b = [baseline[i] for i in ids]
            row = next(r for r in summary["data_degradation"] if r["condition"] == condition)
            source_interval = [row["paired_ci95_low_deg"], row["paired_ci95_high_deg"]]
        else:
            row = (next(r for r in summary["data_degradation"] if r["condition"] == condition)
                   if name.startswith("FollowBlock") else summary["conditions"][condition])
            source_interval = [row["ci95_low_deg"], row["ci95_high_deg"]]
        return self.stat(a, ids, [path, scored_path, summary_path,
            FOLLOWUP / "source-op-reliability/repro/own_period/reliability_score.py",
            REPO / "lc_pipeline/k3/reliability_scoring.py"],
            f"Condition {condition}: average successful repeats within each asteroid, exclude failed selections and asteroids with no success. "
            "Use the source's IID paired-standard or fold-stratified asteroid bootstrap. Full-data change pairs exactly the same identities with grid-full.",
            second=b, seed=seed, folds=None if folds is None else [folds[i] for i in ids], order=order, mean_interval=source_interval)

    def candidate_statistic(self, name):
        from repro.candidate_count import peaks
        from lc_pipeline.k3.grid import axial_healpix_grid
        key = ("candidate_vectors",)
        path = OWN / "maps/own-period/evaluations/real-oof-ensemble.npz"
        if key not in self.cache:
            grid = axial_healpix_grid(32)
            ids = self.ensemble["object_ids"].astype(str).tolist()
            with load_npz(path, allow_pickle=False) as d:
                maps = d["score_grids"]
            values = {k: [] for k in (1, 2, 3)}
            for i, s in zip(ids, maps):
                # The deterministic ordered peak sequence has the same prefixes
                # for K=1,2,3; extract it once per frozen map.
                axes = peaks(s.astype(np.float64), grid, 3)
                for k in values:
                    values[k].append(reference.axial_error_deg(axes[:k], self.poles[i]))
            self.cache[key] = values
        k = {"CountOneMean": 1, "CountTwoMean": 2, "CountThreeMean": 3}[name]
        return self.stat(self.cache[key][k], self.ensemble["object_ids"].astype(str).tolist(),
            [path, REPO / "repro/candidate_count.py", OWN / "maps/own-period/analyses/candidate-count.json"],
            f"Original saved float32 score maps, cast to float64; K={k}, local maxima with the original 15-degree separation and fill rule, no final axis adjustment. All 170 asteroids. No published mean CI.")

    def group_range(self, name):
        ids = self.ensemble["object_ids"].astype(str).tolist()
        sources = [OWN / "maps/own-period/evaluations/real-oof-ensemble.npz", FOLLOWUP / "source-op-maps/repro/own_period/maps/analyses.py"]
        groups = {}
        if "Fold" in name:
            for f in range(5):
                mask = self.ensemble["folds"] == f
                groups[str(f)] = self.stat(self.ensemble["oracle_errors_deg"][mask], np.array(ids)[mask].tolist(), [], "")
        else:
            for seed in (17, 42, 137, 777, 2027):
                values, current_ids = [], []
                for f in range(5):
                    p = OWN / f"maps/own-period/evaluations/real-fold-{f}-seed-{seed}.npz"
                    with load_npz(p, allow_pickle=False) as d:
                        values.extend(d["oracle_errors_deg"]); current_ids.extend(d["object_ids"].astype(str).tolist())
                    sources.append(p)
                groups[str(seed)] = self.stat(values, current_ids, [], "")
        extreme = min if name.endswith("Min") else max
        selected_mean = extreme(groups, key=lambda k: groups[k]["mean_recomputed"])
        selected_median = extreme(groups, key=lambda k: groups[k]["median"])
        result = dict(groups[selected_median])
        result.update(mean_recomputed=groups[selected_mean]["mean_recomputed"],
                      selected_mean_group=selected_mean, selected_median_group=selected_median,
                      group_summaries={k: {field: v[field] for field in ("n", "mean_recomputed", "median", "object_ids")} for k, v in groups.items()},
                      statistic="minimum_of_group_medians" if name.endswith("Min") else "maximum_of_group_medians",
                      source_files=list(map(str, sources)),
                      method="Same five OOF folds or five single-network seeds as the source. Replace each group's mean with its median and then take the same min/max operation; the group attaining the endpoint may change. n and object_ids identify the median endpoint group; every group is retained separately. Range, not a confidence interval.")
        return result

    def latitude_statistic(self, name):
        analysis = OWN / "lowq/scores-own/census-analysis-excluded.json"
        catalog = NOTEBOOKS / "damit-20250610T000301Z/tables/asteroid_models.csv"
        solutions = defaultdict(list)
        with catalog.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    solutions[int(row["asteroid_id"])].append((float(row["lambda"]), float(row["beta"])))
                except ValueError:
                    continue
        low, high = (75, 90) if name.endswith("High") else (0, 15)
        ids, values = [], []
        for row in self.json(analysis)["objects"]:
            refs = solutions[int(row["object_id"].split(":")[1])]
            if low <= abs(refs[0][1]) and (abs(refs[0][1]) < high if high < 90 else abs(refs[0][1]) <= high):
                ids.append(row["object_id"])
                values.append(reference.axial_error_deg(reference.standard_axes(), np.array([reference.unit_from_degrees(*r) for r in refs]))
                              if "Standard" in name else float(row["oracle_at_3_error_deg"]))
        return self.stat(values, ids, [analysis, catalog, FOLLOWUP / "source-op-lowq/repro/own_period/lowq_score.py"],
                         f"Original successful low-quality sample; first listed reference latitude selects [{low},{high}{']' if high == 90 else ')'} degrees in absolute latitude, inclusive final bin as mirror_symmetry.binned. Standard and DeLPHI use the same bin identities; these are errors, not absolute-latitude summaries. No source mean CI.")

    def coordinate_statistic(self, name):
        values = []
        axes = self.ensemble["refined_axes"]
        axes = axes / np.linalg.norm(axes, axis=-1, keepdims=True)
        ids = self.ensemble["object_ids"].astype(str).tolist()
        for candidates, oid in zip(axes, ids):
            best = None
            for candidate in candidates:
                for pole in self.poles[oid]:
                    aligned = candidate if candidate @ pole >= 0 else -candidate
                    angle = np.degrees(np.arccos(np.clip(abs(candidate @ pole), -1, 1)))
                    if best is None or angle < best[0]:
                        best = (angle, aligned, pole)
            _, candidate, pole = best
            beta_p, beta_c = np.degrees(np.arcsin(np.clip([pole[2], candidate[2]], -1, 1)))
            dlon = (np.degrees(np.arctan2(candidate[1], candidate[0]) - np.arctan2(pole[1], pole[0])) + 180) % 360 - 180
            values.append(abs(beta_c - beta_p) if name == "SplitLatitude" else abs(dlon) * np.cos(np.radians(beta_p)))
        return self.stat(values, ids, [OWN / "maps/own-period/evaluations/real-oof-ensemble.npz", REPO / "repro/mirror_symmetry.py"],
                         "Original closest candidate/reference pair with axial sign alignment. Absolute latitude error or longitude error times cos(reference latitude), per mirror_symmetry.py; all 170. No source CI.")

    def period_statistic(self, name):
        path = FOLLOWUP / "data/reliability-20260915/run-1/scored-rows.json"
        rows = self.json(path)["payload"]
        table_path = path.with_name("reliability-report.csv")
        with table_path.open(newline="", encoding="utf-8") as handle:
            table = {r["condition"]: r for r in csv.DictReader(handle) if r["model_id"] == "k3"}
        def per_condition(condition):
            grouped = defaultdict(list)
            for r in rows:
                if r["condition"] == condition and r["status"] != "ineligible":
                    grouped[r["object_id"]].append(r)
            ids = sorted(grouped)
            return ids, [np.mean([90.0 if r["status"] == "failed" else r["error_deg"] for r in grouped[i]]) for i in ids], [grouped[i][0]["fold"] for i in ids]
        def one(condition, paired=False):
            ids, values, folds = per_condition(condition)
            source = table[condition]
            second = None
            if paired:
                base_ids, base, _ = per_condition("full")
                base = dict(zip(base_ids, base)); second = [base[i] for i in ids]
            interval = [float(source[k]) for k in (("degradation_bootstrap_lower_95", "degradation_bootstrap_upper_95") if paired else ("mean_error_bootstrap_lower_95", "mean_error_bootstrap_upper_95"))]
            return self.stat(values, ids, [path, table_path, REPO / "lc_pipeline/k3/reliability_scoring.py"],
                f"Published-period perturbation {condition}; preserve original set, average repeats within asteroid, failed errors at 90 degrees, original fold-stratified bootstrap.",
                second=second, folds=folds, seed=20260915, order="numpy_replicate_major", mean_interval=interval)
        if name != "PeriodSensitivityMaxChange":
            return one("period-0.5" if name == "PeriodHalf" else "period-2.0")
        candidates = {c: one(c, True) for c in table if c.startswith("period-") and abs(float(c.removeprefix("period-")) - 1) <= .010000000001}
        mean_group = max(candidates, key=lambda c: abs(candidates[c]["mean_recomputed"]))
        median_group = max(candidates, key=lambda c: abs(candidates[c]["median"]))
        result = dict(candidates[median_group])
        result.update(mean_recomputed=abs(candidates[mean_group]["mean_recomputed"]), median=abs(result["median"]),
                      interval=None, mean_interval_recomputed=None, statistic="maximum_absolute_difference_of_medians",
                      selected_mean_condition=mean_group, selected_median_condition=median_group, condition_comparisons=candidates,
                      method="Maximum absolute change over exactly the published period factors within one percent of unity; medians recompute the maximum over differences of medians. Individual condition comparisons with paired fold bootstrap are retained separately; no interval is claimed for the selected maximum.")
        result.pop("mean_interval_matches_source", None)
        result.pop("mean_interval_in_source", None)
        result.pop("bootstrap", None)
        result.pop("median_of_paired_differences", None)
        return result

    def broad_grid_statistic(self):
        folder = OWN / "v67/v7-own-period"
        execution = self.json(folder / "execution.json")
        records = [self.json(folder / r["path"]) for r in execution["rows"]]
        by_arm = {arm: defaultdict(list) for arm in ("guided20", "classical20")}
        folds = {}
        for r in records:
            if r["arm"] not in by_arm or r.get("cache_state", "cold") != "cold":
                continue
            oid = r["object_id"]; folds[oid] = r["fold"]
            fit = r["selected_fit"]
            if fit is not None:
                axis = np.array(fit["axis"], dtype=float); axis /= np.linalg.norm(axis)
                by_arm[r["arm"]][oid].append(reference.axial_error_deg(axis[None, :], self.poles[oid]))
        ids = sorted(folds)
        values = {arm: [np.mean(by_arm[arm][i]) if by_arm[arm][i] else 90.0 for i in ids] for arm in by_arm}
        source = self.json(OWN / "v67/v7-own-period-score.json")["angular_comparison_step20"]
        return self.stat(values["guided20"], ids,
            [folder / "execution.json", *[folder / r["path"] for r in execution["rows"]],
             FOLLOWUP / "source-op-v67/repro/own_period/v7_grid.py"],
            "Same 170 cold broad-grid asteroids, three repeats; average selected-fit axial errors over repeats with a fit within each asteroid/arm, 90 degrees if none. Guided minus classical, paired original fold-stratified bootstrap, seed 20260915.",
            second=values["classical20"], folds=[folds[i] for i in ids], seed=20260915, order="numpy_replicate_major", mean_interval=source["interval_95_deg"])

    def resolve_statistic(self, item: dict) -> dict:
        name = item["mean_macro"]
        source = source_path(item["input_source_comment"])
        basepath = FOLLOWUP / "source/repro/baselines_out/reference-baselines-per-object.npz"
        if name == "KThreeSyntheticMean":
            paths = [FOLLOWUP / f"inputs/frozen-artifacts/k3-definitive-7874092/evaluations/synthetic-test-seed-{s}.npz"
                     for s in (17, 42, 137, 777, 2027)]
            errors, ids = [], None
            for path in paths:
                with load_npz(path, allow_pickle=False) as d:
                    current = d["object_ids"].astype(str).tolist()
                    if ids is not None and current != ids:
                        raise ValueError("Synthetic seed identities do not align")
                    ids = current
                    errors.append(d["oracle_errors_deg"])
            return self.stat(np.mean(errors, axis=0), ids, paths + [REPO / "lc_pipeline/k3/release.py"],
                "Exactly 2000 simulated test asteroids; average five single-network seed errors per asteroid, "
                "then summarize, as release.py does; no failures or bootstrap interval in the original.")
        if name == "KThreeStandardMean":
            return self.stat(self.baseline["standard_errors_deg"], self.baseline["object_ids"].astype(str).tolist(),
                [basepath, REPO / "repro/compute_reference_baselines.py"],
                "Published six standard poles (three distinct axial directions); frozen per-asteroid min axial errors; all 170.")
        if name == "KThreeRandomMean":
            ids = self.baseline["object_ids"].astype(str).tolist()
            values = self.random_errors(ids, self.poles, reference.RANDOM_SEED, reference.RANDOM_DRAWS)
            return self.randomized(values, ids, [basepath, reference.CATALOG,
                REPO / "repro/compute_reference_baselines.py"],
                "Three independent uniform-sphere axes per asteroid per draw; regenerate the original "
                "Gaussian stream (draw-major, identity-major); all 170, no failures.",
                reference.RANDOM_SEED, reference.RANDOM_DRAWS)
        if name.startswith("KThreeOofMean") or name in ("KThreeRefinedOracleMean", "KThreeGridOracleMean"):
            summary = self.json(OWN / "maps/own-period/analyses/summary.json")
            grid = name == "KThreeGridOracleMean"
            return self.stat(self.ensemble["grid_oracle_errors_deg" if grid else "oracle_errors_deg"],
                self.ensemble["object_ids"].astype(str).tolist(),
                [OWN / "maps/own-period/evaluations/real-oof-ensemble.npz",
                 OWN / "maps/own-period/analyses/summary.json",
                 FOLLOWUP / "source-op-maps/repro/own_period/maps/analyses.py"],
                "Same 170 full-data own-period OOF asteroids, no failures; " +
                ("grid oracle axial errors before candidate adjustment." if grid else
                 "adjusted oracle axial errors; asteroid bootstrap with the original 20260930 stream."),
                seed=None if grid else 20260930,
                mean_interval=None if grid else summary["primary_intervals"]["mean_ci95_deg"])
        if name == "CheckReassignMean":
            return self.reassignment()
        if name in ("CheckTrainMean", "CheckTestMean"):
            path = OWN / "nettests/train-vs-test-own-report.json"
            d = self.json(path)
            rows = d["pairs"]
            ids = [r["object_id"] for r in rows]
            cells_path = OWN / "nettests/train-vs-test-own"
            cells_files = sorted(cells_path.glob("cells-fold-*.jsonl"))
            folds = {r["object_id"]: r["fold"] for p in cells_files
                     for r in map(json.loads, p.read_text().splitlines()) if r["role"] == "test"}
            which = "train" if name == "CheckTrainMean" else "oof"
            values = [r["mean_train_error_deg" if which == "train" else "oof_error_deg"] for r in rows]
            boot = d["train_oof_pairs"]["bootstrap"]
            return self.stat(values, ids, [path, *cells_files,
                FOLLOWUP / "source-op-nettests/repro/own_period/train_vs_test.py",
                REPO / "lc_pipeline/k3/internal_validation_reporting.py"],
                "Same 169 paired asteroids; training repetitions averaged within asteroid before median; "
                "test has one OOF error; fold groups sorted, then identities sorted, replicate-major draws; "
                "conditional on fixed fitted models.", seed=boot["seed"], folds=[folds[oid] for oid in ids],
                order="numpy_replicate_major", mean_interval=boot[which + "_ci95_deg"])
        if name == "KThreeAtlasMean":
            path = FOLLOWUP / "inputs/frozen-artifacts/k3-2408c563/comparators/v1-comparators.npz"
            with load_npz(path, allow_pickle=False) as d:
                return self.stat(d["atlas_errors_deg"], d["object_ids"].astype(str).tolist(),
                    [path, NOTEBOOKS / "paper/k3-reproduction/publication-summary.json"],
                    "Published fitted-axis comparator on all 170 asteroids; frozen per-asteroid seed aggregation, as released.")
        if name in ("InterpAAMean", "InterpAADelphiMean", "InterpAAStandardMean"):
            path = (OWN / "maps/own-period/analyses/amplitude-aspect.json" if name == "InterpAADelphiMean"
                    else FOLLOWUP / "data/interpretation-20260930/amplitude-aspect.json")
            rows = self.json(path)
            key = {"InterpAAMean": "amplitude_aspect_oracle_deg", "InterpAADelphiMean": "delphi_oracle_deg",
                   "InterpAAStandardMean": "standard_oracle_deg"}[name]
            return self.stat([r[key] for r in rows], [r["object_id"] for r in rows],
                [path, REPO / "repro/amplitude_aspect_paired.py"],
                "Same 168 asteroids with >=3 usable lightcurves; frozen min axial errors; "
                "amplitude-aspect and standard references are period-independent; no individual-mean CI.")
        if name.startswith("Period"):
            path = OWN / "own-period-scores-main.json"
            rows = self.json(path)["main_sample"]["objects"]
            if name != "PeriodDamitMean":
                rows = [r for r in rows if (r["period_class"] == "P") == (name == "PeriodMatchMean")]
            key = "published_error_deg" if name == "PeriodDamitMean" else "own_error_deg"
            return self.stat([r[key] for r in rows], [r["object_id"] for r in rows],
                [path, REPO / "repro/own_period/score.py"],
                "Original score.py asteroid set and period-class labels; P matches versus every non-P class; "
                "published comparison uses all 170. No period failures in this main sample.")
        if source and "benchmark" in str(source):
            folder = source.parent
            if name.startswith("AccuracyCost"):
                d = self.json(source)
                quartile = int(re.search(r"quartiles/(\d)", item["input_source_comment"]).group(1))
                low, high = d["quartiles"][quartile]["oracle_range_deg"]
                rows = [r for r in d["objects"] if low <= r["oracle_deg"] and
                        (r["oracle_deg"] <= high if quartile == 3 else r["oracle_deg"] < high)]
                if len(rows) != d["quartiles"][quartile]["n"]:
                    raise ValueError("Quartile membership changed")
                key = "classical_error_deg" if "Classical" in name else "delphi_error_deg"
                return self.stat([r[key] for r in rows], [r["object_id"] for r in rows],
                    [source, folder / "candidate_accuracy_cost.py"],
                    f"Original oracle-error quartile {quartile + 1}; lower-inclusive, upper-exclusive except "
                    "last quartile upper-inclusive; all 35 saved final selected fits, no failed fits.")
            if "budget-equivalence" in str(source):
                budget = self.json(source)
                ladder_path = folder / "classical-ladder-exploratory.json"
                rows = self.json(ladder_path)["object_rows"]
                delphi = name == "BudgetDelphiError"
                if delphi:
                    score_path = folder / "score-loaded.json"
                    by_id = {r["object_id"]: r["arms"]["delphi"]["reference_error_degrees"]
                             for r in self.json(score_path)["object_rows"]}
                    values = [by_id[r["object_id"]] for r in rows]
                    extra = [score_path]
                    description = "Six DeLPHI starts, matched to ladder identities by object ID."
                else:
                    position = int(re.search(r"budgets/(\d+)", item["input_source_comment"]).group(1))
                    starts = str(budget["budgets"][position]["starts"])
                    values = [r["classical"][starts]["reference_error_degrees"] for r in rows]
                    extra = []
                    description = f"Classical {starts}-start saved-fit budget."
                return self.stat(values, [r["object_id"] for r in rows],
                    [source, ladder_path, *extra, folder / "budget_equivalence.py"],
                    description + " Same 140 evaluation asteroids, all selected fits complete; "
                    "minimum final relative RMS selection already frozen in ladder, no reselection by pole error.")
            if name.startswith("Extended"):
                ladder_path = folder / "classical-ladder-exploratory.json"
                interval_path = folder / "classical-ladder-intervals.json"
                rows = self.json(ladder_path)["object_rows"]
                starts = "18" if "Eighteen" in name else "96"
                pos = 2 if starts == "18" else 7
                return self.stat([r["classical"][starts]["reference_error_degrees"] for r in rows],
                    [r["object_id"] for r in rows], [ladder_path, interval_path, folder / "score_classical_ladder.py"],
                    f"Classical {starts} starts minus DeLPHI on the same 140 selected fits; "
                    "difference of medians with paired within-fold bootstrap; original fold-major NumPy choice stream. "
                    "Median of per-asteroid differences separately labelled. Post hoc descriptive budget scan.",
                    second=[r["delphi"]["reference_error_degrees"] for r in rows], seed=20260924,
                    folds=[r["fold"] for r in rows], order="numpy_fold_major",
                    mean_interval=self.json(interval_path)["budgets"][pos]["classical_minus_delphi_mean_error_95_interval_degrees"])
            d = self.json(source)
            rows = d["object_rows"]
            paired = name.startswith("WorkflowClassicalMinus")
            arm = "classical" if "Classical" in name else "delphi"
            return self.stat([r["arms"][arm]["reference_error_degrees"] for r in rows],
                [r["object_id"] for r in rows], [source, folder / "run_direct_benchmark.py"],
                "Same 140 evaluation asteroids; final selected-fit axial errors against all references; "
                "all fits complete. " + ("Classical minus DeLPHI; paired asteroid bootstrap within five folds, "
                "Python random.Random(20260923).choice, replicate-major; median of paired differences also given."
                if paired else "Original individual mean has no bootstrap interval."),
                second=[r["arms"]["delphi"]["reference_error_degrees"] for r in rows] if paired else None,
                seed=20260923 if paired else None, folds=[r["fold"] for r in rows] if paired else None,
                order="python_fold", mean_interval=d["bootstrap_95_intervals"]["classical_minus_delphi_mean_error_degrees"] if paired else None)
        if source and "no-uncertainty" in str(source):
            cohort = "alcdef" if "Alcdef" in name else "ztf"
            d = self.json(source)["variants"][cohort]
            rows = [r for r in d["objects"] if r["error_deg"] is not None]
            paired = "Standard" in name
            original = d["standard_minus_delphi_failed_excluded_seed20260924"]
            return self.stat([r["error_deg"] for r in rows], [r["object_id"] for r in rows],
                [source, FOLLOWUP / "source-op-nettests/repro/own_period/external_variants.py",
                 REPO / "repro/compute_reference_baselines.py"],
                "No-uncertainty variant, same successful-period asteroids, failures excluded. " +
                ("Paper direction is DeLPHI minus standard in BOTH cohorts, including ALCDEF whose Source "
                 "comment omits the sign reversal. Reverse source mean and swap/negate CI endpoints; "
                 "paired seed 20260924, 10000 asteroid resamples. Median of paired differences also given."
                 if paired else "Frozen adjusted oracle axial errors; no CI for the individual mean."),
                second=[r["standard_error_deg"] for r in rows] if paired else None,
                seed=20260924 if paired else None,
                mean_interval=[-original["ci95_high_deg"], -original["ci95_low_deg"]] if paired else None)
        if source and "baselines-excluded" in str(source):
            group, comparator = re.search(r":: (lowq\w*)/(delphi|standard)/", item["input_source_comment"]).groups()
            analysis = OWN / "lowq/scores-own/census-analysis-excluded.json"
            lineage = OWN / "lowq/scores-own/lineage-excluded/objects.csv"
            if group == "lowq":
                rows = self.json(analysis)["objects"]
            else:
                import csv
                with lineage.open(encoding="utf-8", newline="") as stream:
                    rows = list(csv.DictReader(stream))
                label = group.removeprefix("lowq_")
                if label in ("gaia", "asas"):
                    code = reference.GAIA_SOURCE_KEY if label == "gaia" else reference.ASAS_SOURCE_KEY
                    rows = [r for r in rows if code in r.get("source_keys", "").lower()]
                else:
                    rows = [r for r in rows if bool(r.get("overlapping_training_source_keys", "").strip()) == (label == "shared")]
            kept, values = [], []
            for r in rows:
                refs = reference.cohort_reference(r["object_id"], self.table, self.poles)
                if refs is not None:
                    kept.append(r["object_id"])
                    values.append(float(r["oracle_at_3_error_deg"]) if comparator == "delphi" else
                                  reference.axial_error_deg(reference.standard_axes(), refs))
            if len(kept) != self.json(source)[group]["object_count"]:
                raise ValueError("Lower-quality comparator membership changed")
            return self.stat(values, kept, [source, analysis, lineage, reference.CATALOG,
                NOTEBOOKS / "damit-20250610T000301Z/tables/asteroid_models.csv",
                FOLLOWUP / "source-op-lowq/repro/own_period/lowq_score.py"],
                f"Original {group} successful-period subset, identical comparator identities; "
                "same source-key/shared-lineage membership and unmatched-reference exclusion; "
                "reuse standard_axes/cohort_reference/axial_error_deg helpers; no individual-mean CI.")
        if name in ("SamplingOneFoundMean", "SamplingOneFoundStandardMean"):
            path = OWN / "failed-excluded-derived/reliability-paired.json"
            rows = self.json(path)["conditions"][CONDITION]["objects"]
            delphi = name == "SamplingOneFoundMean"
            fold_rows_path = OWN / "reliability/score/scored-rows-own.json"
            fold_rows = self.json(fold_rows_path)
            folds = {r["object_id"]: r["fold"] for r in fold_rows if r["condition"] == CONDITION}
            mean_interval = None
            if delphi:
                summary = self.json(OWN / "reliability/score/reliability-own-period.json")
                row = next(r for r in summary["failed_excluded"]["data_degradation"] if r["condition"] == CONDITION)
                mean_interval = [row["ci95_low_deg"], row["ci95_high_deg"]]
            return self.stat([r["delphi_mean_deg" if delphi else "standard_error_deg"] for r in rows],
                [r["object_id"] for r in rows], [path, fold_rows_path,
                 OWN / "reliability/score/reliability-own-period.json",
                 FOLLOWUP / "source-op-reliability/repro/own_period/reliability_score.py",
                 REPO / "lc_pipeline/k3/reliability_scoring.py"],
                "Same 80 asteroids with successful repeat(s); exclude failed selections, average each "
                "asteroid's successful-repeat errors first. Standard uses identical identities. "
                "Asteroids sorted, then fold-stratified replicate-major bootstrap, seed 20260915 for DeLPHI.",
                seed=20260915 if delphi else None, folds=[folds[r["object_id"]] for r in rows] if delphi else None,
                order="numpy_replicate_major", mean_interval=mean_interval)
        if name.startswith(("FollowZtf", "FollowAlcdef")):
            cohort = "ztf" if name.startswith("FollowZtf") else "alcdef"
            path = own_scores_path(cohort)
            rows = self.json(path)[cohort]["objects"]
            comparator = next((v for label, v in (("Standard", "standard"), ("Random", "random"),
                                                  ("Atlas", "fitted_axes")) if label in name), "delphi")
            if cohort == "ztf":
                rows = [r for r in rows if r["own_error_deg"] is not None]
            ids = [r["object_id"] for r in rows]
            if comparator == "delphi":
                return self.stat([r["own_error_deg"] for r in rows], ids, [path, REPO / "repro/own_period/score.py"],
                    "Same own-period successful external asteroids; failed periods excluded; frozen oracle axial errors.")
            noerr_path = OWN / "nettests/no-uncertainty/score-own.json"
            noerr = {r["object_id"]: r for r in self.json(noerr_path)["variants"][cohort]["objects"]}
            if comparator == "standard":
                return self.stat([noerr[i]["standard_error_deg"] for i in ids], ids,
                    [path, noerr_path, REPO / "repro/own_period/score.py"],
                    "Six standard poles on exactly the own-period successful external identities; "
                    "reference-only per-object errors reused from the no-uncertainty scorer.")
            if comparator == "fitted_axes":
                p = (FOLLOWUP / "data/publication-revision-20260917-r2/error-channel-sensitivity.json" if cohort == "ztf"
                     else FOLLOWUP / "data/generalization-20260910/locked-cohort/alcdef-gaia-transfer-analysis-30-20260911.json")
                raw = self.json(p)
                comp = {r["object_id"]: r["train_only_atlas_error_deg" if cohort == "ztf" else "atlas_oracle_at_3_error_deg"]
                        for r in (raw["ztf"]["objects"] if cohort == "ztf" else raw["objects"])}
                return self.stat([comp[i] for i in ids], ids, [path, p, REPO / "repro/own_period/score.py"],
                    "Frozen published fitted-axis comparator, aligned by object ID to the same successful external subset.")
            if cohort == "ztf":
                all_ids = [r["object_id"] for r in self.json(path)["ztf"]["objects"]]
                values = self.random_errors(all_ids, self.poles, 20260925, 1000)
                values = values[:, [all_ids.index(i) for i in ids]]
                seed = 20260925
                extra = [reference.CATALOG]
            else:
                p = FOLLOWUP / "data/generalization-20260910/locked-cohort/alcdef-gaia-transfer-analysis-30-20260911.json"
                ids = [r["object_id"] for r in self.json(p)["objects"]]
                if set(ids) != {r["object_id"] for r in rows}:
                    raise ValueError("Published ALCDEF random subset differs")
                values = self.random_errors(ids, self.gaia, 20260936, 1000)
                seed = 20260936
                extra = [p, FOLLOWUP / "data/generalization-20260910/sources/gaia-dr3-spins-20260911/table3.dat"]
            return self.randomized(values, ids, [path, *extra, REPO / "repro/compute_reference_baselines.py",
                OWN / "failed-excluded-derived/derive.py"],
                "Three random axes; same original 1000 draws and identity order; "
                "ZTF generates all 161 first then selects the same 108 successes; "
                "ALCDEF keeps the published 30-object reference-only run.", seed, 1000)
        raise KeyError(f"No per-object resolver for {name}: {item['input_source_comment']}")


def build(input_list: Path, manuscript: Path = PAPER) -> dict:
    items = parse_inputs(input_list)
    from repro.own_period import DataPath as FixedDataPath
    old_scope = FixedDataPath("median-stats.json")
    if old_scope.is_file():
        additions = {r["mean_macro"] for r in json.loads(old_scope.read_text())["entries"] if r.get("new_in_m1b")}
        items = [{**r, **({"new_in_m1b": True} if r["mean_macro"] in additions else {})} for r in items]
    items, audit = audit_manuscript(manuscript, items)
    evidence = Evidence()
    fixed_names = {"KThreeStandardMean", "KThreeRandomMean", "KThreeAtlasMean", "InterpAAMean", "InterpAAStandardMean",
                   "WorkflowClassicalError", "WorkflowDamitClassicalError", "ExtendedNinetysixMeanError",
                   "InterpAAOverStandard", "InterpAAOverStandardLow", "InterpAAOverStandardHigh", "FollowAlcdefRandomMean"}
    # Fixed classical budget fits are protocol inputs. Comparators on selected
    # successful cohorts or network-defined quartiles are still recomputed.
    fixed_names.update(r["mean_macro"] for r in items
                       if r["mean_macro"].startswith("Budget") and r["mean_macro"] != "BudgetDelphiError")
    carried = {}
    if network.switched():
        from repro.own_period import DataPath as FixedDataPath
        published = FixedDataPath("median-stats.json")
        carried = {e["mean_macro"]: e for e in json.loads(published.read_text())["entries"]
                   if e["mean_macro"] in fixed_names}
    entries, skipped, unavailable = [], [], []
    for item in items:
        name = item["mean_macro"]
        if name in SKIPPED:
            skipped.append({**item, "reason": SKIPPED[name]})
            continue
        print(f"Computing {name}", flush=True)
        try:
            if name in carried:
                result = dict(carried[name])
                result["carried_reference_only"] = True
                result["carried_from"] = {"path": str(published), "sha256": network.sha256(published)}
            else:
                result = evidence.resolve(item)
        except (FileNotFoundError, KeyError) as exc:
            unavailable.append({**item, "reason": str(exc)})
            continue
        result.update(item)
        if network.switched():
            result["network_dependent"] = not result.get("carried_reference_only", False) and any(select(p) != Path(p).absolute() or str(network.NetworkPath("results")) in str(p) or str(network.NetworkPath("evaluations")) in str(p) for p in result["source_files"])
            source_files = result["source_files"]
            result["provenance"] = {str(p if result.get("carried_reference_only") else select(p)):
                network.sha256(p if result.get("carried_reference_only") else select(p))
                for p in source_files if Path(p if result.get("carried_reference_only") else select(p)).is_file()}
        result.update(check_printed_mean(result["mean_recomputed"], item["paper_numeric_text"]))
        entries.append(result)
    return {**network.json_metadata(), "schema": "delphi.paper1-median-stats.v1", "input_list": str(input_list),
            "manuscript": str(manuscript), "manuscript_sha256": hashlib.sha256(manuscript.read_bytes()).hexdigest(),
            "audit": audit,
            "input_sha256": hashlib.sha256(input_list.read_bytes()).hexdigest(),
            "units": "degrees", "bootstrap_replicates": RESAMPLES,
            "interval_convention": "95% percentile asteroid bootstrap wherever the original mean has a CI; "
                "original RNG, replicate count, fold strata and loop order. Null/random-axis draw intervals "
                "are separately labelled randomization_interval95, never called bootstrap intervals.",
            "paired_convention": "median(A)-median(B), resampling the same identities in both arms; "
                "median(A-B) separately labelled median_of_paired_differences.",
            "entries": entries, "skipped": skipped, "not_computable": unavailable,
            "mean_mismatches": [e["mean_macro"] for e in entries if not e["matches_printed_rounding"]],
            "source_validation_failures": [e["mean_macro"] for e in entries if
                e.get("mean_interval_matches_source") is False or e.get("null_mean_draws_match_source") is False],
            "counts": {"input": len(items), "computed": len(entries), "skipped": len(skipped),
                       "not_computable": len(unavailable)}}


def write_report(result: dict, path: Path, pytest_output: str, *, output_path: Path,
                 output_error: str | None = None) -> None:
    lines = ["# Paper I M1b: complete median computations", "", f"JSON: `{output_path}`",
             "", f"Coverage: {result['counts']}",
             f"Mean rounding mismatches: {result['mean_mismatches'] or 'none'}.",
             f"Source interval / permutation validation failures: {result['source_validation_failures'] or 'none'}.",
             "", "Units: degrees. CI rows show the median CI endpoint, with the full CI repeated. "
             "For paired statistics, median = difference of medians; median of paired differences is listed separately.",
             "Random-axis and reassignment medians average the per-draw asteroid medians, preserving "
             "the published randomization convention. Their draw intervals are not bootstrap CIs. "
             "Paired random comparisons average per-draw differences of asteroid medians and use a paired asteroid bootstrap conditional on the original fixed random draws. "
             "Medians of per-asteroid expected errors and their paired CIs are separately labelled in JSON.",
             "Removal Low/High values are the range over five seed-specific median differences, not confidence bounds. "
             "Fold and seed range endpoints can be attained by different groups after replacing means with medians. "
             "Pre-specified mean-based control gaps are marked pre_specified_on_mean in JSON; their original tests are not reinterpreted as median tests.",
             "", "## New M1b values",
             "", "| macro | mean (paper) | mean (recomputed) | median | interval | n |",
             "|---|---:|---:|---:|---|---:|"]
    for e in result["entries"]:
        if not e.get("new_in_m1b"):
            continue
        interval = e["interval"]
        ci = "—" if interval is None else f"[{interval[0]:.8f}, {interval[1]:.8f}]"
        if e.get("seed_range") is not None:
            ci = f"seed range [{e['seed_range'][0]:.8f}, {e['seed_range'][1]:.8f}]"
        if e["statistic"].startswith("randomization_draw_median_quantile"):
            ci = "null draw range [" + ", ".join(f"{v:.8f}" for v in e["randomization_interval95"]) + "]"
        flag = " **MISMATCH**" if not e["matches_printed_rounding"] else ""
        lines.append(f"| `{e['mean_macro']}` | {e['paper_numeric_text']} | {e['mean_recomputed']:.10f}{flag} "
                     f"| {e['median']:.10f} | {ci} | {e['n']} |")
    lines += ["", "## Paired differences", "", "| macro | difference of medians | median of paired differences |",
              "|---|---:|---:|"]
    for e in result["entries"]:
        if e["statistic"] in ("difference_of_medians", "mean_of_seed_differences_of_medians", "mean_of_randomization_draw_differences_of_medians"):
            lines.append(f"| `{e['mean_macro']}` | {e['median']:.10f} | {e['median_of_paired_differences']:.10f} |")
    lines += ["", "## Skipped", ""]
    lines += [f"- `{e['mean_macro']}`: {e['reason']}" for e in result["skipped"]] or ["None."]
    lines += ["", "## Manuscript macro audit", "",
              f"Read-only manuscript: `{result.get('manuscript', PAPER)}`. Only macros used outside comments after begin-document are audited; final definitions take precedence. M1 entries are all retained, including entries not currently printed.",
              "", "### Included", ""]
    lines += [f"- `{r['macro']}` (line {r['line']}): {r['reason']}. {r['source_comment']}" for r in result.get("audit", []) if r["included"]]
    lines += ["", "### Skipped with reason", ""]
    lines += [f"- `{r['macro']}` (line {r['line']}): {r['reason']}" for r in result.get("audit", []) if not r["included"]]
    lines += ["", "## Not computable", ""]
    lines += [f"- `{e['mean_macro']}`: {e['reason']}" for e in result["not_computable"]] or ["None. "
        "Per-object published DAMIT-period benchmark and budget-ladder records were found under "
        "`DeLPHI-followup/data/direct-workflow-benchmark/` and used directly."]
    lines += ["", "## Method details", ""]
    for e in result["entries"]:
        lines += [f"- `{e['mean_macro']}`: {e['method']}",
                  "  Sources: " + "; ".join(f"`{p}`" for p in e["source_files"]) + "."]
    lines += ["", "## Pytest output", "", "```text", pytest_output.rstrip(), "```"]
    command = ["taskset", "-c", "12-23", "nice", "-n", "19", "env", "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=1", "PYTHONDONTWRITEBYTECODE=1",
               sys.executable, "-m", "repro.own_period.median_stats", "--input-list", result["input_list"],
               "--manuscript", result.get("manuscript", str(PAPER)),
               "--out", str(output_path), "--report", str(path)]
    lines += ["", "## Regenerate", "", f"From `{REPO}`:", "", "```bash", shlex.join(command), "```",
              "", "No manuscript edits, refine-chain imports/edits, GPU use, commits, or pushes. One process with one numerical-library thread."]
    for title, args in [("git diff --stat", ["git", "diff", "--stat"]),
                        ("git status --short", ["git", "status", "--short"])]:
        out = subprocess.run(args, cwd=REPO, text=True, capture_output=True, check=True).stdout
        lines += ["", f"## {title}", "", "```text", out.rstrip() or "(empty)", "```"]
    if output_error:
        lines += ["", "## Output limitation", "", output_error,
                  "The complete computed JSON is retained below in the requested report so the "
                  "calculation can be reviewed and the intended JSON file can be restored when its "
                  "directory is writable.", "", "## Computed JSON (output write blocked)", "",
                  "```json", json.dumps(result, indent=2, allow_nan=False), "```"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-list", type=Path, help="macro inventory (required unless --report-only)")
    parser.add_argument("--out", "--output", dest="output", type=Path, default=OUTPUT)
    parser.add_argument("--manuscript", type=Path, default=PAPER)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--pytest-output", type=Path)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if not args.report_only and args.input_list is None:
        parser.error("--input-list is required unless --report-only")
    result = read_json(args.output) if args.report_only else build(args.input_list, args.manuscript)
    output_error = None
    if not args.report_only:
        network.guard_output(args.output)
        network.guard_output(args.report)
        try:
            args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        except OSError as exc:
            if exc.errno not in (errno.EROFS, errno.EACCES):
                raise
            output_error = f"Required JSON could not be written: `{args.output}`. {exc}"
    pytest_output = args.pytest_output.read_text() if args.pytest_output else "Not yet run."
    write_report(result, args.report, pytest_output, output_path=args.output, output_error=output_error)
    print(json.dumps({"counts": result["counts"], "mean_mismatches": result["mean_mismatches"],
                      "source_validation_failures": result["source_validation_failures"],
                      "output": str(args.output), "report": str(args.report), "output_error": output_error}), flush=True)
    if output_error:
        return 2
    return int(bool((result["mean_mismatches"] and not network.switched()) or result["source_validation_failures"] or result["not_computable"]))


if __name__ == "__main__":
    raise SystemExit(main())
