"""Read-only independent numeric check; does not import lc_pipeline.

Uses atan2/cross products for axial angles and SciPy's linear assignment
solver, rather than the analysis's acos and enumeration of six assignments.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def unseal(path):
    doc = json.loads(Path(path).read_text())
    text = json.dumps(doc["payload"], sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    if hashlib.sha256(text.encode()).hexdigest() != doc["payload_sha256"]:
        raise ValueError(f"invalid seal: {path}")
    return doc["payload"]


def normalize(x):
    x = np.asarray(x, dtype=float)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def angles(x, y):
    x, y = normalize(x), normalize(y)
    cross = np.linalg.norm(np.cross(x[:, None, :], y[None, :, :]), axis=-1)
    return np.degrees(np.arctan2(cross, np.abs(x @ y.T)))


def verify(root):
    root = Path(root)
    receipt = unseal(root / "manifest.json")
    for filename, checksum in receipt["files"].items():
        if sha(root / filename) != checksum:
            raise ValueError(f"changed report: {filename}")
    lock = unseal(root / "analysis-lock.json")
    for item in lock["inputs"].values():
        if sha(item["path"]) != item["sha256"]:
            raise ValueError("changed input")
    pred = unseal(root / "neighbour-predictions.json")
    features = {r["object_id"]: np.asarray(r["values"]) for r in pred["features"]}
    roles = {
        m["fold"]: m["object_roles"]
        for key, item in lock["inputs"].items()
        if key.startswith("bundle:")
        for m in [json.loads(Path(item["path"]).read_text())]
    }
    refs = {
        r["object_id"]: [s["vector"] for s in r["solutions"]]
        for line in Path(lock["inputs"]["catalog"]["path"]).read_text().splitlines()
        if line.strip()
        for r in [json.loads(line)]
        if r.get("eligible") is True
    }
    study = unseal(lock["inputs"]["study.json"]["path"])
    source = unseal(lock["inputs"]["scored-rows.json"]["path"])
    full = {r["object_id"]: r for r in source if r["condition"] == "full"}
    scored = {(r["object_id"], r["condition"], r["repeat"]): r for r in source}
    baseline_axes = {}
    for p in pred["objects"]:
        fold, oid = p["fold"], p["object_id"]
        train_ids = sorted(roles[fold]["train_ids"])
        assert oid in roles[fold]["test_ids"] and oid not in train_ids
        train = np.stack([features[i] for i in train_ids])
        mean, std = np.mean(train, axis=0), np.std(train, axis=0)
        scaler = next(s for s in pred["scalers"] if s["fold"] == fold)
        np.testing.assert_allclose(scaler["mean"], mean, atol=1e-12, rtol=0)
        np.testing.assert_allclose(scaler["std"], std, atol=1e-12, rtol=0)
        distance = np.zeros(len(train))
        for lo, hi in [(0, 6), (6, 26), (26, 32)]:
            indices = [i for i in range(lo, hi) if std[i] > 1e-12]
            if indices:
                distance += (
                    np.mean(
                        ((train[:, indices] - features[oid][indices]) / std[indices]) ** 2, axis=1
                    )
                    / 3
                )
        order = sorted(range(len(train)), key=lambda i: (distance[i], train_ids[i]))[:3]
        neighbours = [train_ids[i] for i in order]
        assert p["neighbour_ids"] == neighbours
        np.testing.assert_allclose(p["squared_distances"], distance[order], atol=1e-12, rtol=0)
        primary = normalize([refs[i][0] for i in neighbours])
        single = normalize(refs[neighbours[0]])
        single = np.vstack([single, np.tile(single[0], (3 - len(single), 1))])
        np.testing.assert_allclose(p["three_neighbour_primary_axes"], primary, atol=1e-12, rtol=0)
        np.testing.assert_allclose(p["one_neighbour_all_axes"], single, atol=1e-12, rtol=0)
        baseline_axes[oid] = {
            "K3 ensemble": full[oid]["axes"],
            "Three-neighbour first-axis": primary,
            "One-neighbour all-axes": single,
            "Train-only atlas": study["atlases"][fold]["axes"],
        }
    with (root / "report/similarity-objects.csv").open() as f:
        baseline_rows = list(csv.DictReader(f))
    baseline_delta = max(
        abs(
            float(angles(baseline_axes[r["object_id"]][r["method"]], refs[r["object_id"]]).min())
            - float(r["error_deg"])
        )
        for r in baseline_rows
    )
    with (root / "report/stability-repeats.csv").open() as f:
        stability_rows = list(csv.DictReader(f))
    displacement_delta = 0.0
    checked = 0
    values = {}
    for r in stability_rows:
        if r["status"] != "ok":
            assert r["assignment_mean_deg"] == ""
            continue
        oid, condition, repeat = r["object_id"], r["condition"], int(r["repeat"])
        cost = angles(full[oid]["axes"], scored[(oid, condition, repeat)]["axes"])
        i, j = linear_sum_assignment(cost)
        value = float(cost[i, j].mean())
        displacement_delta = max(displacement_delta, abs(value - float(r["assignment_mean_deg"])))
        values.setdefault((condition, oid), []).append(value)
        checked += 1
    summary = json.loads((root / "report/summary.json").read_text())
    summary_delta = 0.0
    for s in summary["stability"]:
        object_means = [np.mean(v) for (c, _), v in values.items() if c == s["condition"]]
        summary_delta = max(
            summary_delta, abs(float(np.mean(object_means)) - s["assignment_mean_deg"]["mean"])
        )
    if max(baseline_delta, displacement_delta, summary_delta) > 1e-5:
        raise ValueError("independent angle or aggregation check differs by more than 1e-5 degrees")
    return {
        "passed": True,
        "neighbour_queries": len(pred["objects"]),
        "baseline_object_rows": len(baseline_rows),
        "successful_stability_repeats": checked,
        "stability_conditions": len(summary["stability"]),
        "max_baseline_difference_deg": baseline_delta,
        "max_assignment_difference_deg": displacement_delta,
        "max_stability_mean_difference_deg": summary_delta,
        "manifest_sha256": sha(root / "manifest.json"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.output), sort_keys=True))


if __name__ == "__main__":
    main()
