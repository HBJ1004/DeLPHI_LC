"""Read-only manuscript audit: old and selected values for every inventory macro.

Expressions in the inventory are descriptions, never executed as Python. A
small explicit evaluator handles the published formulas and otherwise requires
an unambiguous JSON key. Missing values fail the table instead of copying old
network results into the new column.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

from repro import network_root as network
from repro.own_period import DataPath as FixedDataPath
from repro.own_period.reporting_inputs import check, load_npz, read_json, select


def lookup(value, key):
    parts = key.strip("/").split("/")
    while parts:
        if isinstance(value, dict) and "/".join(parts) in value:
            return value["/".join(parts)]
        if isinstance(value, dict):
            joined = next(
                (i for i in range(len(parts), 0, -1) if "/".join(parts[:i]) in value), None
            )
            if joined:
                value = value["/".join(parts[:joined])]
                parts = parts[joined:]
                continue
        part = parts.pop(0)
        if isinstance(value, list):
            if part.isdigit():
                value = value[int(part)]
            else:
                value = next(row for row in value if row.get("mean_macro") == part)
        else:
            value = value[part]
    return value


def simple_key(text):
    text = text.split(" (", 1)[0].split(" = ", 1)[0].split(";", 1)[0].strip()
    return text


class Values:
    def __init__(self, *, selected=True):
        self.selected = selected
        self.cache = {}
        self.extra_sources = set()

    def path(self, path):
        return select(path) if self.selected else Path(path)

    def json(self, path):
        path = self.path(path)
        if path not in self.cache and path.name == "paired-gains.json" and not network.switched():
            from repro.own_period.reporting_aggregates import paired_gains

            self.cache[path] = paired_gains()
        if path not in self.cache:
            self.cache[path] = read_json(path) if self.selected else json.loads(path.read_text())
        return self.cache[path]

    def resolve(self, macro):
        value = self.raw(macro)
        name = macro["name"]
        if name == "WorkflowColdTimeFactor":
            return 1 / value
        if name == "WorkflowLoadedTimeReductionPercent":
            return 100 * (1 - 1 / value)
        if name == "WorkflowWithinTwentyDeficitPercentagePoints":
            return -100 * value
        if name == "PropertyReferenceCountP":
            from scipy.stats import t

            return float(2 * t.sf(abs(value) * np.sqrt(168 / (1 - value**2)), 168))
        if name == "ScErosSimilarMaxError":
            return int(np.ceil(value))
        if (
            name.startswith(("SampWithin", "WorkflowClassicalMinusDelphiWithinTwenty"))
            or name
            in (
                "SunRaisesAboveOnePercent",
                "SunLowersAboveOnePercent",
                "BrightnessLowersAboveOnePercent",
            )
            or (
                r"\%" in macro["value"]
                and "percent" not in macro["source_field"]
                and isinstance(value, (int, float))
            )
        ):
            return 100 * value
        if name.startswith("Shap") and name.endswith("Share"):
            return 100 * value
        return value

    def raw(self, macro):
        name, source, field = macro["name"], Path(macro["source_file"]), macro["source_field"]
        if source.name == "own-period-scores.json" and field.startswith("alcdef/") and self.selected and network.switched():
            # The rerun scores ALCDEF in its own file; the published run kept it beside ZTF.
            source = Path(network.NetworkPath("results", "alcdef", "own-period-scores.json"))
        if name.startswith("TOne"):
            groups = self.json(FixedDataPath("paired-gains.json"))["groups"]
            match = re.fullmatch(r"(TOne.+?)(Mean|Median)(Low|High)?", name)
            base, statistic, bound = match.groups()
            row = groups[base]
            if bound:
                return row["mean_interval_recomputed" if statistic == "Mean" else "interval"][
                    0 if bound == "Low" else 1
                ]
            return row["mean_recomputed" if statistic == "Mean" else "median"]
        if source.name == "median-stats.json":
            data = self.json(source)
            if name == "WorkflowErrorReductionPercentMedian":
                by = {e["mean_macro"]: e for e in data["entries"]}
                return 100 * (
                    1 - by["WorkflowDelphiError"]["median"] / by["WorkflowClassicalError"]["median"]
                )
            return lookup({e["mean_macro"]: e for e in data["entries"]}, simple_key(field))
        if name == "KThreeSwapGap" and self.selected and network.switched():
            # The rerun summarizes the input-swap control as its own family, not with the removals.
            return lookup(self.json(Path(network.NetworkPath("results", "controls-input-swap", "own-summary.json"))), field)
        if source.suffix == ".npz":
            self.extra_sources.add(self.path(source))
            with load_npz(self.path(source), allow_pickle=False) as data:
                return self.npz(name, field, data)
        if source.suffix == ".csv":
            self.extra_sources.add(self.path(source))
            import csv

            if self.selected:
                check(self.path(source))
            with self.path(source).open() as stream:
                rows = {r["condition"]: r for r in csv.DictReader(stream) if r["model_id"] == "k3"}
            if name == "PeriodSensitivityMaxChange":
                return max(
                    abs(float(r["mean_degradation"]))
                    for c, r in rows.items()
                    if c.startswith("period-") and abs(float(c[7:]) - 1) <= 0.010000000001
                )
            return float(
                rows["period-0.5" if name == "PeriodHalf" else "period-2.0"]["mean_error_deg"]
            )
        if source.suffix == ".md":
            return self.markdown(name)
        if source.name == "publication-summary.json":
            stats = self.json(FixedDataPath("median-stats.json"))
            by_name = {e["mean_macro"]: e for e in stats["entries"]}
            if name == "KThreeSyntheticMean":
                return by_name["KThreeSyntheticMean"]["mean_recomputed"]
            return (
                by_name["LabelShuffleAfter"]["mean_recomputed"]
                - by_name["LabelShuffleBefore"]["mean_recomputed"]
            )
        value = self.json(source)
        key = simple_key(field)
        if name.startswith("PeriodMatch") or name.startswith("PeriodMismatch"):
            return float(
                np.mean(
                    [
                        r["own_error_deg"]
                        for r in value["main_sample"]["objects"]
                        if (r["period_class"] == "P") == name.startswith("PeriodMatch")
                    ]
                )
            )
        if name.startswith("KThreeFold") and "Median" not in name:
            return (min if name.endswith("Min") else max)(r["mean_deg"] for r in value["folds"])
        if name.startswith("KThreeSeed") and "Median" not in name:
            return (min if name.endswith("Min") else max)(value["per_seed_mean_deg"].values())
        if name == "ClassicalResidualMatchingBudget":
            return next(
                r["starts"] for r in value["budgets"] if r["rms_ratio_classical_over_delphi"] < 1
            )
        if name.startswith(("WorkflowSameSolution", "WorkflowDifferent", "WorkflowLowerRmsCloser")):
            return self.fit_comparison(name, value)
        if name.startswith("CensusBlind"):
            group = "new3" if name == "CensusBlindNewMedian" else "all7"
            return value["summaries"][group]["delphi_own_error_deg"][
                "number_within_20_deg" if "Within" in name else "median_deg"
            ]
        if name.startswith(("Sc", "Spacecraft")) and source.name in (
            "scores.json",
            "results.json",
            "context.json",
        ):
            return self.spacecraft(name, source, value)
        if name.startswith("Split") and name.endswith(("Low", "High")):
            column = "latitude_part_deg" if name.startswith("SplitLat") else "longitude_part_deg"
            bins = [r[column] for r in value["summary"]["coordinate_split"]["bins"]]
            return float((min if name.endswith("Low") else max)(bins))
        if name == "WorkflowErrorReductionPercent":
            point = value["point_estimates"]
            return 100 * (
                1 - point["delphi_mean_error_degrees"] / point["classical_mean_error_degrees"]
            )
        if name == "WorkflowDelphiWithinTwentyPercent":
            return 100 * value["point_estimates"]["delphi_within20"] / 140
        if name == "WorkflowFullSearchSpeedFactor":
            from repro.own_period.median_figures import full_scan_costs

            seconds = full_scan_costs(value)["seconds"]
            loaded = self.json(FixedDataPath("benchmark/score-loaded.json"))["point_estimates"][
                "delphi_mean_seconds"
            ]
            # The fixed Fourier cost is reported in the period-cost summary.
            return seconds / (self.period_search_seconds() + round(loaded, 2))
        if name == "AreaThreeMedian":
            return value["k"][2]["search_area_median_radius"]
        if name.startswith("InsightRho") or name == "PropertyReferenceCountP":
            key = simple_key(field)
            key = key.split(" (", 1)[0]
            result = lookup(value, key)
            if isinstance(result, dict):
                return result["p_value" if name == "PropertyReferenceCountP" else "rho"]
            return result
        if name == "SamplingOneFoundMean":
            return next(
                r["mean_error_deg"]
                for r in value["failed_excluded"]["data_degradation"]
                if r["condition"] == "two_d-block_count-1-per_block_cap-None"
            )
        if name.startswith("SampWithin"):
            key = key.split(" (", 1)[0]
            return lookup(value, key)
        if "objects[run_id=" in key:
            raise KeyError(key)
        # Most inventory fields already are exact slash-separated JSON paths.
        try:
            answer = lookup(value, key)
        except (KeyError, TypeError, StopIteration, IndexError):
            # A few older source notes use dots for the same schema.
            answer = lookup(value, key.replace(".", "/"))
        if (
            "sign reversed" in field
            or "DifferencePositive" in name
            or name.startswith(
                (
                    "FollowAlcdefAtlasDifference",
                    "FollowZtfFoundAtlasDifference",
                    "FollowAlcdefNoErrorStandard",
                    "FollowZtfNoErrorFoundStandard",
                )
            )
        ) and isinstance(answer, (int, float)):
            answer = -answer
        return answer

    def npz(self, name, field, data):
        if name == "LabelShuffleBefore" or name == "LabelShuffleAfter":
            return float(data["oracle_errors_deg"].mean())
        if name.startswith("KThreeRefinement"):
            return int(
                np.sum(
                    data["oracle_errors_deg"] > data["grid_oracle_errors_deg"]
                    if "Worse" in name
                    else data["oracle_errors_deg"] < data["grid_oracle_errors_deg"]
                )
            )
        from repro import compute_reference_baselines as ref

        poles = ref.reference_poles()
        ids = data["object_ids"].astype(str)
        latitudes = np.array([abs(np.degrees(np.arcsin(poles[i][0, 2]))) for i in ids])
        errors = np.array(
            [ref.axial_error_deg(axes, poles[i][:1]) for i, axes in zip(ids, data["refined_axes"])]
        )
        if name == "InsightRhoLatitudeFirst":
            from scipy.stats import spearmanr

            return float(spearmanr(latitudes, errors).statistic)
        low, high = (0, 30) if "Low" in name else (30, 60) if "Mid" in name else (60, 91)
        return float(errors[(latitudes >= low) & (latitudes < high)].mean())

    def spacecraft(self, name, source, value):
        if name == "ScErosLooMinError":
            return min(
                r["oracle_error_deg"]
                for r in value["variants"]
                if r["kind"] == "leave_one_apparition_out"
            )
        if name == "ScErosSimilarMaxError":
            return max(r["oof_error_deg"] for r in value["similar_objects"].values())
        if name == "SpacecraftMedian":
            posthoc = self.json(source.parent / "posthoc/scores.json")
            chosen = {
                "itokawa-combined",
                "lutetia-damit",
                "ida-damit",
                "gaspra-damit",
                "eros-oof",
                "ceres-oof",
                "vesta-oof",
                "ryugu-combined-grond",
                "bennu-all",
            }
            return float(
                np.median(
                    [
                        r["own"]["oracle_deg"]
                        for r in value["objects"] + posthoc["objects"]
                        if r["run_id"] in chosen
                    ]
                )
            )
        key = dict(
            ScItokawaCombinedDelphi="itokawa-combined",
            ScItokawaLowryDelphi="itokawa-lowry2014",
            ScLutetiaDelphi="lutetia-damit",
            ScIdaDelphi="ida-damit",
            ScGaspraDelphi="gaspra-damit",
            ScErosDelphi="eros-oof",
            ScCeresDelphi="ceres-oof",
            ScVestaDelphi="vesta-oof",
            ScRyuguDelphi="ryugu-combined-grond",
            ScBennuDelphi="bennu-all",
        )[name]
        return next(r["own"]["oracle_deg"] for r in value["objects"] if r["run_id"] == key)

    def fit_comparison(self, name, value):
        from repro.own_period.maps.validate_damit_run import axis_angle_deg

        rows = value["object_rows"]
        different = [
            r
            for r in rows
            if axis_angle_deg(
                np.array(r["arms"]["classical"]["axis"]), np.array(r["arms"]["delphi"]["axis"])
            )
            > 10
        ]
        if name == "WorkflowSameSolutionN":
            return len(rows) - len(different)
        if name == "WorkflowDifferentSolutionN":
            return len(different)
        pairs = [(r["arms"]["classical"], r["arms"]["delphi"]) for r in different]
        if name == "WorkflowDifferentRmsMedian":
            return float(
                np.median(
                    [
                        100 * abs(a["final_relative_rms"] / b["final_relative_rms"] - 1)
                        for a, b in pairs
                    ]
                )
            )
        if name == "WorkflowDifferentClassicalLower":
            return sum(a["final_relative_rms"] < b["final_relative_rms"] for a, b in pairs)
        if name == "WorkflowDifferentDelphiLower":
            return sum(b["final_relative_rms"] < a["final_relative_rms"] for a, b in pairs)
        return sum(
            (
                a["final_relative_rms"] < b["final_relative_rms"]
                and a["reference_error_degrees"] < b["reference_error_degrees"]
            )
            or (
                b["final_relative_rms"] < a["final_relative_rms"]
                and b["reference_error_degrees"] < a["reference_error_degrees"]
            )
            for a, b in pairs
        )

    def period_search_seconds(self):
        # Use the fixed period-cost macro's source, never a rounded paper number.
        value = self.json(FixedDataPath("period-cost/period-cost-summary.json"))
        return value["fourier_seconds"]

    def markdown(self, name):
        if name.startswith(("FollowGrid", "FollowBlock", "Noise")):
            from repro.own_period.median_stats import Evidence

            result = Evidence().resolve_additional({"mean_macro": name})
            self.extra_sources.update(map(select, result["source_files"]))
            return result["mean_recomputed"]
        if name.startswith("DownSix"):
            value = self.json(FixedDataPath("v67/v6-own-period-score.json"))
            return lookup(
                value,
                "verdict_published_convention/"
                + ("speed_ratio_ci95_low" if name.endswith("Low") else "speed_ratio"),
            )
        if name == "MaxPropertyAbsoluteSpearman":
            value = self.json(
                FixedDataPath("maps/own-period/analyses/reliability-strata-spearman.json")
            )
            return max(
                abs(v)
                for k, v in value["spearman_matrix"]["oracle_error_deg"].items()
                if k != "oracle_error_deg"
            )
        raise KeyError(name)


def display_value(value, definition, name):
    text = definition.replace(r"\ensuremath{-}", "-").replace("{,}", "")
    if isinstance(value, list):
        if len(value) != 2:
            raise ValueError("unsupported macro range")
        return "--".join(str(round(v)) for v in value)
    if name == "CensusBlindWithinTwenty":
        return str(int(value))
    if name == "WorkflowFullSearchSpeedFactor":
        return str(int(float(f"{value:.2g}")))
    match = re.match(r"[-+]?\d+(?:\.\d+)?", text)
    if not match or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"unsupported or nonfinite macro: {name}")
    precision = len(match.group().partition(".")[2])
    if round(value, precision) == 0:
        value = 0.0
    sign = "+" if text.startswith("+") else ""
    return format(value, f"{sign}.{precision}f")


def check_published(table):
    mismatches = []
    for row in table["macros"]:
        name, old = row["macro"], row["old_value"]
        text = old.replace(r"\ensuremath{-}", "-").replace("{,}", "")
        if name == "CensusBlindWithinTwenty":
            expected = {
                "zero": "0",
                "one": "1",
                "two": "2",
                "three": "3",
                "four": "4",
                "five": "5",
                "six": "6",
                "seven": "7",
            }[old]
        elif isinstance(row["new_value"], list):
            expected = "--".join(re.findall(r"\d+(?:\.\d+)?", text))
        else:
            expected = re.match(r"[-+]?\d+(?:\.\d+)?", text).group()
        if row["new_display_value"] != expected:
            mismatches.append(
                dict(macro=name, expected=expected, computed=row["new_display_value"])
            )
    return mismatches


def build(inventory, manuscript, *, selected=True):
    catalog = json.loads(Path(inventory).read_text())
    definitions = dict(
        re.findall(r"^\\newcommand\{\\(\w+)\}\{(.*)\}$", Path(manuscript).read_text(), re.M)
    )
    values, rows = Values(selected=selected), []
    for macro in catalog["macros"]:
        if not macro["network_dependent"]:
            continue
        name = macro["name"]
        new = values.resolve(macro)
        if isinstance(new, np.generic):
            new = new.item()
        source_file, source_key = macro["source_file"], macro["source_field"]
        if name.startswith("TOne"):
            base, kind, bound = re.fullmatch(r"(TOne.+?)(Mean|Median)(Low|High)?", name).groups()
            source_file = FixedDataPath("paired-gains.json")
            source_key = (
                "groups/"
                + base
                + "/"
                + (
                    ("mean_interval_recomputed" if kind == "Mean" else "interval")
                    + ("/0" if bound == "Low" else "/1")
                    if bound
                    else ("mean_recomputed" if kind == "Mean" else "median")
                )
            )
        if Path(macro["source_file"]).suffix == ".md":
            if name.startswith(("FollowGrid", "FollowBlock", "Noise")):
                source_file = FixedDataPath("reliability/score/reliability-own-period.json")
                source_key = (
                    "failed_excluded; condition and original bootstrap bound selected by macro name"
                )
            elif name.startswith("DownSix"):
                source_file = FixedDataPath("v67/v6-own-period-score.json")
                source_key = "verdict_published_convention/" + (
                    "speed_ratio_ci95_low" if name.endswith("Low") else "speed_ratio"
                )
            elif name == "MaxPropertyAbsoluteSpearman":
                source_file = FixedDataPath(
                    "maps/own-period/analyses/reliability-strata-spearman.json"
                )
                source_key = (
                    "max(abs(spearman_matrix/oracle_error_deg/*)) excluding self correlation"
                )
        if Path(macro["source_file"]).name == "publication-summary.json":
            source_file = FixedDataPath("median-stats.json")
            source_key = (
                "KThreeSyntheticMean/mean_recomputed"
                if name == "KThreeSyntheticMean"
                else "LabelShuffleAfter/mean_recomputed - LabelShuffleBefore/mean_recomputed"
            )
        rows.append(
            dict(
                macro=name,
                old_value=definitions[name],
                new_value=new,
                new_display_value=display_value(new, definitions[name], name),
                source_file=str(values.path(source_file)),
                source_key=source_key,
                old_source_file=macro["source_file"],
            )
        )
    return dict(
        source_sha256={
            str(p): network.sha256(p)
            for p in set(values.cache) | values.extra_sources
            if p.is_file()
        },
        schema="delphi.paper1-macros-old-new.v1",
        **network.json_metadata(),
        manuscript=dict(path=str(manuscript), sha256=network.sha256(manuscript)),
        inventory_sha256=network.sha256(inventory),
        count=len(rows),
        macros=rows,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-published", action="store_true")
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument(
        "--manuscript",
        type=Path,
        default=FixedDataPath("../../../paper-rev-20261021/sample701.tex"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    network.guard_output(args.output)
    result = build(args.inventory, args.manuscript)
    if args.check_published:
        mismatches = check_published(result)
        if mismatches:
            raise ValueError(f"published macro mismatch: {mismatches}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
