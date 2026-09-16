"""CPU-only final reporting for the sealed internal-role diagnostic.

Reference reassignment below keeps predictions fixed; it is a descriptive null
for the reference-set association, not a retraining permutation p-value.
"""

from __future__ import annotations

import csv
import io
import json
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from .reliability_study import digest, unseal, verify, write_json

SEEDS = (17, 42, 137, 777, 2027)
SCHEMA = "delphi.k3-internal-validation-report.v1"


def _mean(values):
    return float(np.mean(values)) if values else None


def require_complete(root: Path) -> tuple[dict, list[dict]]:
    lock = unseal(root / "study.json")
    cells = [unseal(p) for p in root.glob("cells/*/*/*/result.json")]
    expected = {(x["fold"], x["role"], x["object_id"]): x for x in lock["rows"]}
    actual = {(x.get("fold"), x.get("role"), x.get("object_id")): x for x in cells}
    if len(actual) != len(cells) or set(actual) != set(expected):
        raise ValueError("final report cell inventory differs from locked rows")
    for key, row in actual.items():
        locked = expected[key]
        if (
            row.get("status") != "ok"
            or row.get("input", {}).get("sha256") != locked["lightcurve"]["sha256"]
        ):
            raise ValueError(f"invalid completed cell: {key}")
        if not np.isfinite(row.get("error_deg", np.nan)):
            raise ValueError(f"nonfinite cell error: {key}")
        if row["role"] == "test" and row.get("oof_parity_max_abs_axis", np.inf) > 1e-6:
            raise ValueError(f"OOF parity failed: {key}")
    return lock, cells


def train_oof_pairs(cells: list[dict], oof: list[dict]) -> list[dict]:
    train = defaultdict(list)
    for row in cells:
        if row["role"] == "train":
            train[row["object_id"]].append(row["error_deg"])
    old = {x["object_id"]: x["error_deg"] for x in oof if x.get("status") == "ok"}
    return [
        {
            "object_id": oid,
            "mean_train_error_deg": _mean(values),
            "oof_error_deg": old[oid],
            "difference_deg": _mean(values) - old[oid],
            "train_repetitions": len(values),
        }
        for oid, values in sorted(train.items())
        if oid in old
    ]


def fold_bootstrap(
    pairs: list[dict], folds: dict[str, int], *, n: int = 10_000, seed: int = 20260916
) -> dict:
    """Paired asteroid bootstrap sampled independently inside each OOF fold."""
    groups = defaultdict(list)
    for row in pairs:
        groups[folds[row["object_id"]]].append(row)
    ordered = [
        (fold, sorted(values, key=lambda row: row["object_id"]))
        for fold, values in sorted(groups.items())
    ]
    rng = np.random.default_rng(seed)
    draws = []
    train_draws = []
    oof_draws = []
    for _ in range(n):
        selected_pairs = [
            np.asarray(values, dtype=object)[rng.integers(len(values), size=len(values))]
            for _, values in ordered
        ]
        draws.append(
            float(np.mean([x["difference_deg"] for group in selected_pairs for x in group]))
        )
        train_draws.append(
            float(np.mean([x["mean_train_error_deg"] for group in selected_pairs for x in group]))
        )
        oof_draws.append(
            float(np.mean([x["oof_error_deg"] for group in selected_pairs for x in group]))
        )
    return {
        "seed": seed,
        "draws": n,
        "conditional_on_fixed_models": True,
        "mean_difference_deg": _mean([x["difference_deg"] for x in pairs]),
        "ci95_deg": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "train_mean_error_deg": _mean([x["mean_train_error_deg"] for x in pairs]),
        "oof_mean_error_deg": _mean([x["oof_error_deg"] for x in pairs]),
        "train_ci95_deg": [
            float(np.quantile(train_draws, 0.025)),
            float(np.quantile(train_draws, 0.975)),
        ],
        "oof_ci95_deg": [
            float(np.quantile(oof_draws, 0.025)),
            float(np.quantile(oof_draws, 0.975)),
        ],
        "joint_draw_consistency_max_abs": float(
            np.max(np.abs(np.asarray(draws) - (np.asarray(train_draws) - np.asarray(oof_draws))))
        ),
        "failure_count": 0,
    }


def reassignment_null(
    oof: list[dict], refs: dict[str, list], *, n: int = 10_000, seed: int = 20260916
) -> dict:
    """Shuffle complete reference sets only within (fold, reference-count) strata."""
    groups = defaultdict(list)
    for row in oof:
        oid = row["object_id"]
        groups[(row["fold"], len(refs[oid]))].append(row)
    movable = sum(len(v) for v in groups.values() if len(v) > 1)
    observed = _mean([row["error_deg"] for row in oof])
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n):
        errors = []
        for values in groups.values():
            assigned = rng.permutation([refs[x["object_id"]] for x in values])
            axes = np.asarray([row["axes"] for row in values], float)
            reference = np.asarray(assigned, float)
            axes /= np.linalg.norm(axes, axis=2, keepdims=True)
            reference /= np.linalg.norm(reference, axis=2, keepdims=True)
            best = np.abs(np.einsum("iaj,irj->iar", axes, reference)).max(axis=(1, 2))
            errors.extend(np.degrees(np.arccos(np.clip(best, 0, 1))).tolist())
        draws.append(float(np.mean(errors)))
    return {
        "seed": seed,
        "draws": n,
        "observed_mean_error_deg": observed,
        "null_mean_error_deg": _mean(draws),
        "null_q025_q975_deg": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "null_draws_mean_error_deg": draws,
        "movable_objects": movable,
        "fraction_movable": movable / len(oof),
        "interpretation": "fixed-prediction reference-set reassignment diagnostic; not retraining permutation significance",
    }


def histories(archive: Path, expected_sha256: str) -> list[dict]:
    if digest(archive) != expected_sha256:
        raise ValueError("training archive checksum changed")
    result = []
    with tarfile.open(archive, "r:gz") as tar:
        members = {item.name.rsplit("/", 1)[-1]: item for item in tar.getmembers()}
        for fold in range(5):
            for seed in SEEDS:
                filename = f"real-fold-{fold}-seed-{seed}.jsonl"
                member = members.get(filename)
                if member is None:
                    raise ValueError(f"training archive lacks models/{filename}")
                lines = [
                    json.loads(x) for x in tar.extractfile(member).read().decode().splitlines() if x
                ]
                losses = [float(x["validation_loss"]) for x in lines]
                best = int(np.argmin(losses))
                result.append(
                    {
                        "fold": fold,
                        "seed": seed,
                        "epochs": len(lines),
                        "minimum_recorded_validation_epoch": lines[best]["epoch"],
                        "minimum_recorded_validation_loss": losses[best],
                        "checkpoint_confirmation": "not confirmed from history alone",
                        "train_mix": "real/synthetic",
                        "validation_mix": "real-only",
                        "curves": lines,
                    }
                )
    return result


def archived_seed_oof(archive: Path, expected_sha256: str) -> list[dict]:
    """Read the 25 archived real-fold/seed evaluations, never training losses."""
    if digest(archive) != expected_sha256:
        raise ValueError("training archive checksum changed")
    output = []
    with tarfile.open(archive, "r:gz") as tar:
        names = {item.name.rsplit("/", 1)[-1]: item for item in tar.getmembers()}
        for seed in SEEDS:
            ids, errors, folds = [], [], []
            for fold in range(5):
                name = f"real-fold-{fold}-seed-{seed}.npz"
                member = names.get(name)
                if member is None:
                    raise ValueError(f"archive lacks evaluations/{name}")
                arrays = np.load(io.BytesIO(tar.extractfile(member).read()), allow_pickle=False)
                object_ids = arrays["object_ids"].astype(str).tolist()
                values = arrays["oracle_errors_deg"].astype(float).tolist()
                if len(object_ids) != 34 or len(values) != 34:
                    raise ValueError(f"invalid OOF member {name}")
                ids.extend(object_ids)
                errors.extend(values)
                folds.extend([fold] * 34)
            if len(ids) != 170 or len(set(ids)) != 170:
                raise ValueError(f"seed {seed} does not contain disjoint 170-object OOF cohort")
            output.append(
                {
                    "seed": seed,
                    "n": 170,
                    "mean_oracle_error_deg": _mean(errors),
                    "per_fold": [
                        {
                            "fold": f,
                            "n": 34,
                            "mean_oracle_error_deg": _mean(
                                [e for e, g in zip(errors, folds, strict=True) if g == f]
                            ),
                        }
                        for f in range(5)
                    ],
                    "source": "archived per-model real-fold evaluation NPZ; not training history",
                }
            )
    return output


def oof_tables(oof: list[dict]) -> dict:
    groups = {
        "per_fold": "fold",
        "reference_count_strata": "reference_solution_count",
        "per_prediction_sampling_seed": "seed",
    }
    output = {}
    for name, key in groups.items():
        values = defaultdict(list)
        for row in oof:
            values[row.get(key, 20260915)].append(row["error_deg"])
        output[name] = [
            {key: value, "n": len(errors), "mean_error_deg": _mean(errors)}
            for value, errors in sorted(values.items())
        ]
    folds = {row["fold"] for row in oof}
    output["leave_one_fold_out"] = [
        {
            "omitted_fold": fold,
            "n": sum(row["fold"] != fold for row in oof),
            "mean_error_deg": _mean([row["error_deg"] for row in oof if row["fold"] != fold]),
        }
        for fold in sorted(folds)
    ]
    return output


def tex_macros(report: dict) -> str:
    boot = report["train_oof_pairs"]["bootstrap"]
    return (
        "% Auto-generated internal diagnostic only.\n"
        f"\\newcommand{{\\InternalPairN}}{{{report['train_oof_pairs']['n']}}}\n"
        f"\\newcommand{{\\InternalTrainOOFDiff}}{{{boot['mean_difference_deg']:.2f}}}\n"
        f"\\newcommand{{\\InternalTrainMean}}{{{boot['train_mean_error_deg']:.2f}}}\n"
        f"\\newcommand{{\\InternalOOFMean}}{{{boot['oof_mean_error_deg']:.2f}}}\n"
        f"\\newcommand{{\\InternalDiffLo}}{{{boot['ci95_deg'][0]:.2f}}}\n"
        f"\\newcommand{{\\InternalDiffHi}}{{{boot['ci95_deg'][1]:.2f}}}\n"
        f"\\newcommand{{\\InternalReassignMean}}{{{report['reference_reassignment']['null_mean_error_deg']:.2f}}}\n"
        f"\\newcommand{{\\InternalMovableFraction}}{{{report['reference_reassignment']['fraction_movable']:.3f}}}\n"
    )


def write_plots(
    output: Path, pairs: list[dict], report: dict, histories_rows: list[dict] | None
) -> list[Path]:
    import matplotlib.pyplot as plt

    output.mkdir(exist_ok=True)
    paths = []
    fig, ax = plt.subplots(figsize=(5, 4))
    x = [r["oof_error_deg"] for r in pairs]
    y = [r["mean_train_error_deg"] for r in pairs]
    ax.scatter(x, y, s=12)
    ax.plot([0, 90], [0, 90], color="0.5")
    ax.set(
        xlabel="OOF error (deg)",
        ylabel="mean training-role error (deg)",
        xlim=(0, 90),
        ylim=(0, 90),
    )
    fig.tight_layout()
    path = output / "train-oof-scatter.pdf"
    fig.savefig(path)
    plt.close(fig)
    paths.append(path)
    null = report["reference_reassignment"]
    fig, ax = plt.subplots(figsize=(5, 2.5))
    ax.hist(null["null_draws_mean_error_deg"], bins=40, color="0.7")
    ax.axvline(null["observed_mean_error_deg"], color="#cc6677")
    ax.set(xlabel="mean fixed-prediction error (deg)", ylabel="reassignments")
    fig.tight_layout()
    path = output / "reference-reassignment-null.pdf"
    fig.savefig(path)
    plt.close(fig)
    paths.append(path)
    if histories_rows:
        fig, axes = plt.subplots(5, 2, figsize=(7.1, 8.1))
        for fold in range(5):
            for column, key in enumerate(("train_loss", "validation_loss")):
                ax = axes[fold, column]
                for row in [item for item in histories_rows if item["fold"] == fold]:
                    epochs = [float(x["epoch"]) for x in row["curves"]]
                    ax.plot(
                        epochs,
                        [float(x[key]) for x in row["curves"]],
                        linewidth=0.9,
                        label=str(row["seed"]),
                    )
                    if key == "validation_loss":
                        ax.axvline(
                            row["minimum_recorded_validation_epoch"], color="0.5", linewidth=0.5
                        )
                ax.set_title(f"fold {fold}: {key.replace('_', ' ')}", fontsize=10)
                ax.tick_params(labelsize=10)
                if fold == 0:
                    ax.legend(title="seed", fontsize=9.5, title_fontsize=10, ncol=2)
        fig.text(
            0.5,
            0.01,
            "epoch; train real/synthetic, validation real-only (not like-for-like)",
            ha="center",
            fontsize=10,
        )
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        path = output / "real-training-histories.pdf"
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)
    return paths


def export(
    *,
    root: Path,
    archive: Path | None = None,
    archive_sha256: str | None = None,
    output: Path | None = None,
) -> dict:
    root = Path(root)
    out = Path(output) if output is not None else root / "report-export"
    if out.exists():
        raise FileExistsError(f"refusing to overwrite report export: {out}")
    lock, cells = require_complete(root)
    old = unseal(verify(lock["original_scored_rows"]))
    oof = [
        x
        for x in old
        if x.get("condition") == "full" and x.get("repeat") == 0 and x.get("status") == "ok"
    ]
    catalog_path = verify(unseal(verify(lock["original_study"]))["catalog"])
    from ..v2.data import load_catalog

    catalog = load_catalog(catalog_path)
    refs = {
        x["object_id"]: [list(v) for v in catalog[x["object_id"]].solution_vectors] for x in oof
    }
    folds = {x["object_id"]: x["fold"] for x in oof}
    pairs = train_oof_pairs(cells, oof)
    if len(pairs) != 169:
        raise ValueError(f"expected 169 matched training-object/OOF pairs, found {len(pairs)}")
    report = {
        "schema": SCHEMA,
        "complete_counts": {"train": 540, "validation": 70, "test": 170},
        "validation_role_cells": {
            "n": sum(x["role"] == "validation" for x in cells),
            "mean_error_deg": _mean([x["error_deg"] for x in cells if x["role"] == "validation"]),
            "unit": "model-role cell; not object-weighted",
        },
        "train_oof_pairs": {
            "n": len(pairs),
            "unit": "asteroid, training errors averaged within object",
            "bootstrap": fold_bootstrap(pairs, folds),
        },
        "reference_reassignment": reassignment_null(oof, refs),
        "oof_tables": oof_tables(oof),
    }
    history_rows = histories(Path(archive), archive_sha256) if archive and archive_sha256 else None
    if history_rows:
        report["training_histories"] = [
            {k: v for k, v in row.items() if k != "curves"} for row in history_rows
        ]
        report["archived_per_model_seed_oof"] = archived_seed_oof(Path(archive), archive_sha256)
    out.mkdir()
    write_json(out / "summary.json", report)
    with (out / "train-oof-pairs.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    (out / "macros.tex").write_text(tex_macros(report), encoding="utf-8")
    figures = write_plots(out, pairs, report, history_rows)
    files = {
        p.name: digest(p)
        for p in [out / "summary.json", out / "train-oof-pairs.csv", out / "macros.tex", *figures]
    }
    write_json(
        out / "manifest.json",
        {
            "schema": SCHEMA,
            "source_sha256": {
                "diagnostic_study": digest(root / "study.json"),
                "original_study": lock["original_study"]["sha256"],
                "original_scored_rows": lock["original_scored_rows"]["sha256"],
                "archive": archive_sha256,
                "exporter": digest(Path(__file__)),
            },
            "exporter_sha256": digest(Path(__file__)),
            "cell_hashes": {
                str(path.relative_to(root)): digest(path)
                for path in sorted(root.glob("cells/*/*/*/result.json"))
            },
            "files": files,
        },
    )
    return report
