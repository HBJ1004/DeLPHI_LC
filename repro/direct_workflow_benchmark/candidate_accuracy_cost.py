"""Time and final pole error of the two loaded searches, grouped by the oracle error of the DeLPHI candidates.

Post hoc descriptive analysis of the frozen loaded benchmark (score-loaded.json). The candidates are
those of repeat 0 (all repeats use the same frozen networks and give the same axes). Oracle error is the
smallest axial angle between any candidate and any DAMIT solution.
"""
import glob
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
# Benchmark outputs (release paper-v1, part-b-analyses/workflow-benchmark, plus the raw fits).
DATA = HERE.parents[2] / "data/direct-workflow-benchmark"
CATALOG = HERE.parents[2] / "inputs/training-run/repro/data/damit-20250610T000301Z/catalog.jsonl"
OUTPUT = DATA / "candidate-accuracy-cost.json"


def angle(a, b):
    a = np.asarray(a) / np.linalg.norm(a)
    b = np.asarray(b) / np.linalg.norm(b)
    return float(np.degrees(np.arccos(min(1.0, abs(float(a @ b))))))


def mean_iterations(object_id, arm):
    values = [json.load(open(f)).get("iterations")
              for f in glob.glob(str(DATA / f"evaluation-loaded/cases/{object_id}-r*-{arm}/starts/start-*/record.json"))]
    return float(np.mean([v for v in values if v is not None]))


def main():
    catalog = {r["object_id"]: [s["vector"] for s in r["solutions"]] for r in map(json.loads, CATALOG.read_text().splitlines())}
    objects = []
    for row in json.load(open(DATA / "score-loaded.json"))["object_rows"]:
        oid = row["object_id"]
        axes = [a["axis_xyz"] for a in json.load(open(DATA / f"evaluation-loaded/cases/{oid}-r0-delphi/prediction.json"))["axes"]]
        c, d = row["arms"]["classical"], row["arms"]["delphi"]
        objects.append(dict(object_id=oid, oracle_deg=min(angle(a, s) for a in axes for s in catalog[oid]),
                            classical_seconds=c["mean_wall_seconds"], delphi_seconds=d["mean_wall_seconds"],
                            classical_error_deg=c["reference_error_degrees"], delphi_error_deg=d["reference_error_degrees"],
                            classical_rms=c["final_relative_rms"], delphi_rms=d["final_relative_rms"],
                            classical_iterations=mean_iterations(oid, "classical"), delphi_iterations=mean_iterations(oid, "delphi")))
    oracle = np.array([o["oracle_deg"] for o in objects])
    edges = np.quantile(oracle, [0, .25, .5, .75, 1])
    groups = []
    for i in range(4):
        m = (oracle >= edges[i]) & ((oracle <= edges[i + 1]) if i == 3 else (oracle < edges[i + 1]))
        s = [o for o, keep in zip(objects, m) if keep]
        def mean(key, chosen=s):
            return float(np.mean([o[key] for o in chosen]))

        groups.append(dict(oracle_range_deg=[float(edges[i]), float(edges[i + 1])], n=len(s),
                           classical_seconds=mean("classical_seconds"), delphi_seconds=mean("delphi_seconds"),
                           classical_iterations=mean("classical_iterations"), delphi_iterations=mean("delphi_iterations"),
                           classical_error_deg=mean("classical_error_deg"), delphi_error_deg=mean("delphi_error_deg"),
                           classical_within20=sum(o["classical_error_deg"] <= 20 for o in s),
                           delphi_within20=sum(o["delphi_error_deg"] <= 20 for o in s)))
    bad = [o for o in objects if o["oracle_deg"] > 30]
    summary = dict(
        scope="post_hoc_descriptive_loaded_benchmark_140_evaluation_asteroids",
        spearman_oracle_vs_delphi_seconds=float(spearmanr(oracle, [o["delphi_seconds"] for o in objects])[0]),
        spearman_oracle_vs_speed_ratio=float(spearmanr(oracle, [o["classical_seconds"] / o["delphi_seconds"] for o in objects])[0]),
        spearman_oracle_vs_delphi_iterations=float(spearmanr(oracle, [o["delphi_iterations"] for o in objects])[0]),
        quartiles=groups,
        oracle_above_30=dict(n=len(bad), classical_closer=sum(o["classical_error_deg"] < o["delphi_error_deg"] for o in bad),
                             median_rms_ratio_classical_over_delphi=float(np.median([o["classical_rms"] / o["delphi_rms"] for o in bad]))),
        objects=objects)
    OUTPUT.write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "objects"}, indent=1))


if __name__ == "__main__":
    main()
