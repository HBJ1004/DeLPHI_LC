"""Sealed, diagnostic-only inference on K3 train/validation roles.

This deliberately does *not* change :class:`K3EnsemblePredictor`: the public
predictor continues to reject development IDs.  The private entry point here is
only for a locked retrospective diagnostic and never attaches calibration.
"""

from __future__ import annotations

import contextlib
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path

import numpy as np

from .reliability_study import binding, canonical, digest, seal, unseal, verify, write_json

SCHEMA = "delphi.k3-internal-validation.v1"
ACTIVE_SECONDS = 6 * 3600
ROLES = ("train_ids", "validation_ids", "test_ids")


@contextlib.contextmanager
def exclusive(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    with (root / "run.lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("another internal-validation process holds the run lock") from exc
        yield


class ActiveBudget:
    """Checkpointed active time; an interrupted in-flight cell is charged fully."""

    def __init__(self, root: Path, seconds: int = ACTIVE_SECONDS):
        self.path = root / "budget.json"
        if self.path.exists():
            self.state = unseal(self.path)
            if self.state["limit_seconds"] != seconds:
                raise ValueError("cannot change registered active-hour cap")
        else:
            self.state = {
                "limit_seconds": seconds,
                "charged_seconds": 0.0,
                "active_checkpoint_epoch": time.time(),
            }
            seal(self.path, self.state)

    def remaining(self) -> float:
        return max(0.0, self.state["limit_seconds"] - self.state["charged_seconds"])

    def check(self) -> None:
        if self.remaining() <= 0:
            raise TimeoutError("six active-hour computation cap exhausted")

    def charge(self, seconds: float) -> None:
        self.state["charged_seconds"] += max(0.0, float(seconds))
        self.state["active_checkpoint_epoch"] = time.time()
        write_json(
            self.path,
            {
                "seal_schema": "sha256-canonical-json-v1",
                "payload": self.state,
                "payload_sha256": hashlib.sha256(canonical(self.state).encode()).hexdigest(),
            },
            replace=True,
        )

    def recover_interrupted(self, root: Path) -> None:
        """Do not count paused wall time, but conservatively debit an unfinished cell."""
        inflight = root / "inflight.json"
        if inflight.exists():
            value = json.loads(inflight.read_text(encoding="utf-8"))
            self.charge(float(value["reserved_active_seconds"]))
            inflight.unlink()


def _source_binding() -> dict:
    root = Path(__file__).resolve().parents[2]
    paths = [
        root / "lc_pipeline" / "k3" / name
        for name in (
            "internal_validation.py",
            "bundle.py",
            "reliability_study.py",
            "reliability_sampling.py",
            "evaluation.py",
            "inference.py",
            "model.py",
            "tokenizer.py",
            "config.py",
            "protocol.py",
        )
    ]
    paths += [root / "lc_pipeline" / "v2" / name for name in ("preprocessing.py", "data.py")]
    paths += [
        root / "lc_pipeline" / "physics" / "axial.py",
        root / "lc_pipeline" / "grid_benchmark.py",
    ]
    return {str(path.relative_to(root)): digest(path) for path in paths}


def _seal_replace(path: Path, value: dict) -> None:
    """Atomic replacement is allowed for derived, reproducible progress exports."""
    checksum = hashlib.sha256(canonical(value).encode()).hexdigest()
    write_json(
        path,
        {"seal_schema": "sha256-canonical-json-v1", "payload": value, "payload_sha256": checksum},
        replace=True,
    )


def role_for(manifest: dict, object_id: str) -> str:
    hits = [
        name
        for name in ("train_ids", "validation_ids", "calibration_ids", "test_ids")
        if object_id in manifest["object_roles"][name]
    ]
    if len(hits) != 1:
        raise ValueError(f"object {object_id!r} does not have exactly one bundle role")
    return hits[0]


def diagnostic_predict(
    predictor, epochs, *, known_period, object_id: str, expected_role: str
) -> dict:
    """Same numerical path as K3EnsemblePredictor.predict, with an explicit role gate."""
    from .bundle import _model_inputs
    from .evaluation import ensemble_score_grids, modes_from_score_grid
    from .inference import refine_ensemble_axes, score_axial_grid

    actual = role_for(predictor.manifest, object_id)
    if actual != expected_role:
        raise ValueError(
            f"diagnostic role mismatch: expected {expected_role}, bundle says {actual}"
        )
    inputs = _model_inputs(epochs, known_period, predictor.device)
    maps = [score_axial_grid(model, inputs, chunk_size=1024) for model in predictor.models]
    mean = ensemble_score_grids(np.stack(maps)[:, None, :])[0]
    modes = modes_from_score_grid(mean)
    axes, scores = refine_ensemble_axes(
        predictor.models, inputs, np.asarray([m.axis_xyz for m in modes])
    )
    return {
        "schema": "delphi.k3-internal-diagnostic-prediction.v1",
        "status": "ok",
        "object_id": object_id,
        "fold": predictor.manifest["fold"],
        "bundle_id": predictor.manifest["bundle_id"],
        "diagnostic_role": actual,
        "calibration": None,
        "calibration_claim": "none; diagnostic-only internal role inference",
        "axes": [
            {"axis_xyz": a.tolist(), "score": float(s), "grid_index": int(m.grid_index)}
            for a, s, m in zip(axes, scores, modes, strict=True)
        ],
    }


def _catalog_rows(lock: dict) -> list[dict]:
    from ..v2.data import load_catalog

    catalog = load_catalog(verify(lock["catalog"]))
    # The original held-out objects provide the absolute root of the frozen DAMIT tree.
    sample = Path(lock["objects"][0]["lightcurve"]["path"]).resolve()
    data_root = sample.parents[2]
    frozen_inputs = {row["object_id"]: row for row in lock["objects"]}
    if len(frozen_inputs) != 170:
        raise ValueError("original study lacks the complete 170-object period lock")
    rows = []
    for bundle in lock["bundles"]:
        manifest = json.loads((Path(bundle["directory"]) / "bundle.json").read_text())
        fold = manifest["fold"]
        for role in ROLES:
            for oid in manifest["object_roles"][role]:
                item = catalog[oid]
                if oid not in frozen_inputs:
                    raise ValueError(f"role object absent from original period lock: {oid}")
                frozen = frozen_inputs[oid]
                path = Path(frozen["lightcurve"]["path"])
                if (
                    path != data_root / item.lightcurve_path
                    or digest(path) != item.lightcurve_sha256
                    or frozen["lightcurve"]["sha256"] != item.lightcurve_sha256
                ):
                    raise ValueError(f"catalog lightcurve changed: {oid}")
                rows.append(
                    {
                        "object_id": oid,
                        "fold": fold,
                        "role": role.removesuffix("_ids"),
                        "period_hours": frozen["period_hours"],
                        "lightcurve": frozen["lightcurve"],
                        "references": [list(v) for v in item.solution_vectors],
                    }
                )
    return rows


def prepare(*, root: Path, original_study: Path) -> dict:
    root, original_study = Path(root).resolve(), Path(original_study).resolve()
    if root.exists():
        raise FileExistsError("internal-validation output must be a new directory")
    lock = unseal(original_study / "study.json")
    for item in lock["bindings"]:
        verify(item)
    for bundle in lock["bundles"]:
        for item in bundle["files"]:
            verify(item)
    rows = _catalog_rows(lock)
    counts = {r: sum(x["role"] == r for x in rows) for r in ("train", "validation", "test")}
    if counts != {"train": 540, "validation": 70, "test": 170}:
        raise ValueError(f"unexpected locked role counts: {counts}")
    root.mkdir(parents=True)
    budget = ActiveBudget(root)
    frozen = {
        "schema": SCHEMA,
        "original_study": binding(original_study / "study.json"),
        "original_scored_rows": binding(original_study / "scored-rows.json"),
        "bundles": lock["bundles"],
        "rows": rows,
        "python": binding(sys.executable),
        "device": lock["device"],
        "source": _source_binding(),
        "environment": {k: importlib.metadata.version(k) for k in lock["environment"]},
        "registered": {
            "active_hours": 6,
            "new_inference_cells": 610,
            "rechecked_oof_cells": 170,
            "diagnostic_only": True,
            "calibration_claim": "none",
        },
    }
    seal(root / "study.json", frozen)
    return {"root": str(root), "counts": counts, "remaining_active_seconds": budget.remaining()}


def _error(axes, references) -> float:
    a, r = np.asarray(axes, float), np.asarray(references, float)
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    r /= np.linalg.norm(r, axis=1, keepdims=True)
    return float(np.degrees(np.arccos(np.clip(np.abs(a @ r.T).max(), 0, 1))))


def _read_old_rows(path: Path) -> list[dict]:
    value = unseal(path)
    if not isinstance(value, list):
        raise ValueError("original scored rows must be a sealed list")
    return value


def _cell_path(root: Path, row: dict) -> Path:
    return root / "cells" / f"fold-{row['fold']}" / row["role"] / row["object_id"] / "result.json"


def run(*, root: Path, parity_only: bool = False) -> dict:
    import torch

    from ..v2.preprocessing import KnownPeriod
    from .bundle import K3EnsemblePredictor
    from .reliability_sampling import read_lc, to_epochs

    root = Path(root).resolve()
    with exclusive(root):
        torch.set_num_threads(6)
        lock = unseal(root / "study.json")
        if binding(sys.executable) != lock["python"]:
            raise ValueError("run with frozen Python interpreter")
        if _source_binding() != lock["source"]:
            raise ValueError("diagnostic source changed after prepare")
        if {k: importlib.metadata.version(k) for k in lock["environment"]} != lock["environment"]:
            raise ValueError("diagnostic environment changed after prepare")
        for bundle in lock["bundles"]:
            for item in bundle["files"]:
                verify(item)
        budget = ActiveBudget(root)
        budget.recover_interrupted(root)
        old = _read_old_rows(verify(lock["original_scored_rows"]))
        old_full = {
            (x["object_id"], x["fold"]): x
            for x in old
            if x.get("condition") == "full" and x.get("repeat") == 0
        }
        by_fold = {
            json.loads((Path(b["directory"]) / "bundle.json").read_text())["fold"]: b["directory"]
            for b in lock["bundles"]
        }
        # Exact OOF parity is a gate, never an after-the-fact descriptive field.
        tests = [r for r in lock["rows"] if r["role"] == "test"]
        rows = tests if parity_only else tests + [r for r in lock["rows"] if r["role"] != "test"]
        completed = 0
        for fold in range(5):
            budget.check()
            model_started = time.monotonic()
            write_json(
                root / "inflight.json",
                {
                    "operation": "model_load",
                    "fold": fold,
                    "reserved_active_seconds": min(300.0, budget.remaining()),
                },
                replace=True,
            )
            predictor = K3EnsemblePredictor(by_fold[fold], device=lock["device"])
            budget.charge(time.monotonic() - model_started)
            (root / "inflight.json").unlink()
            for row in [x for x in rows if x["fold"] == fold]:
                target = _cell_path(root, row)
                if target.exists():
                    cached = unseal(target)
                    expected = (
                        row["object_id"],
                        row["fold"],
                        row["role"],
                        row["lightcurve"]["sha256"],
                    )
                    if (
                        cached.get("object_id"),
                        cached.get("fold"),
                        cached.get("role"),
                        cached.get("input", {}).get("sha256"),
                    ) != expected:
                        raise ValueError("cell identity mismatch")
                    if (
                        row["role"] == "test"
                        and cached.get("oof_parity_max_abs_axis", float("inf")) > 1e-6
                    ):
                        raise ValueError("cached OOF cell lacks exact private/public parity")
                    completed += 1
                    continue
                budget.check()
                started = time.monotonic()
                # A crash cannot create a fresh six-hour allowance on resume.
                write_json(
                    root / "inflight.json",
                    {
                        "object_id": row["object_id"],
                        "fold": fold,
                        "role": row["role"],
                        "reserved_active_seconds": min(1800.0, budget.remaining()),
                    },
                    replace=True,
                )
                curve = read_lc(
                    verify(row["lightcurve"]),
                    object_id=row["object_id"],
                    period_hours=row["period_hours"],
                )
                result = diagnostic_predict(
                    predictor,
                    to_epochs(curve),
                    known_period=KnownPeriod(
                        hours=row["period_hours"], provenance="locked-internal-diagnostic"
                    ),
                    object_id=row["object_id"],
                    expected_role=row["role"] + "_ids",
                )
                axes = [x["axis_xyz"] for x in result["axes"]]
                value = {
                    "object_id": row["object_id"],
                    "fold": fold,
                    "role": row["role"],
                    "status": "ok",
                    "axes": axes,
                    "error_deg": _error(axes, row["references"]),
                    "input": row["lightcurve"],
                    "runtime_s": time.monotonic() - started,
                    "diagnostic_only": True,
                    "calibration": None,
                    "calibration_claim": result["calibration_claim"],
                }
                if row["role"] == "test" and (row["object_id"], fold) in old_full:
                    value["oof_parity_max_abs_axis"] = float(
                        np.max(
                            np.abs(
                                np.asarray(axes)
                                - np.asarray(old_full[(row["object_id"], fold)]["axes"])
                            )
                        )
                    )
                    if value["oof_parity_max_abs_axis"] > 1e-6:
                        raise ValueError("private diagnostic/public OOF parity failed")
                seal(target, value)
                budget.charge(value["runtime_s"])
                (root / "inflight.json").unlink()
                completed += 1
                write_json(
                    root / "progress.json",
                    {
                        "completed": completed,
                        "remaining_active_seconds": budget.remaining(),
                        "updated_epoch": time.time(),
                    },
                    replace=True,
                )
            del predictor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return export(root=root)


def export(*, root: Path) -> dict:
    root = Path(root).resolve()
    lock = unseal(root / "study.json")
    rows = [unseal(p) for p in sorted((root / "cells").glob("*/*/*/result.json"))]
    old = _read_old_rows(verify(lock["original_scored_rows"]))
    oof = [x for x in old if x.get("condition") == "full" and x.get("repeat") == 0]
    train_groups = {}
    for row in rows:
        if row["role"] == "train":
            train_groups.setdefault(row["object_id"], []).append(row["error_deg"])
    train_means = {oid: float(np.mean(values)) for oid, values in train_groups.items()}
    oof_errors = {x["object_id"]: x.get("error_deg") for x in oof if x.get("status") == "ok"}
    paired = [
        {
            "object_id": oid,
            "mean_train_error_deg": value,
            "oof_error_deg": oof_errors[oid],
            "train_minus_oof_deg": value - oof_errors[oid],
        }
        for oid, value in sorted(train_means.items())
        if oid in oof_errors
    ]
    summary = {
        "schema": SCHEMA,
        "diagnostic_only": True,
        "calibration_claim": "none",
        "completed_cells": len(rows),
        "by_role": {
            role: {
                "n": sum(x["role"] == role for x in rows),
                "mean_error_deg": float(
                    np.mean([x["error_deg"] for x in rows if x["role"] == role])
                )
                if any(x["role"] == role for x in rows)
                else None,
            }
            for role in ("train", "validation", "test")
        },
        "oof_rows_kept_separate": len(oof),
        "training_object_averages": len(train_means),
        "paired_train_object_vs_oof": {
            "n": len(paired),
            "mean_difference_deg": float(np.mean([x["train_minus_oof_deg"] for x in paired]))
            if paired
            else None,
        },
        "note": "training-role errors may repeat by object across folds and must be object-averaged before comparison.",
    }
    exports = root / "exports"
    exports.mkdir(exist_ok=True)
    files = {}
    if rows:
        fields = sorted({k for row in rows for k in row if k not in ("axes", "input")})
        output = exports / "internal-role-cells.csv"
        with output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fields)
            writer.writeheader()
            writer.writerows([{k: x.get(k) for k in fields} for x in rows])
        files[output.name] = digest(output)
    write_json(exports / "summary.json", summary, replace=True)
    files["summary.json"] = digest(exports / "summary.json")
    if paired:
        output = exports / "train-object-vs-oof.csv"
        with output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(paired[0]))
            writer.writeheader()
            writer.writerows(paired)
        files[output.name] = digest(output)
    write_json(
        exports / "manifest.json",
        {
            "schema": "delphi.k3-internal-validation-export.v1",
            "sources": {
                "original_study": lock["original_study"]["sha256"],
                "original_scored_rows": lock["original_scored_rows"]["sha256"],
            },
            "files": files,
        },
        replace=True,
    )
    _seal_replace(root / "report.json", summary)
    return summary


def status(root: Path) -> dict:
    root = Path(root)
    result = {
        "root": str(root),
        "completed_cells": sum(1 for _ in (root / "cells").glob("*/*/*/result.json")),
    }
    if (root / "budget.json").exists():
        result["remaining_active_seconds"] = ActiveBudget(root).remaining()
    if (root / "progress.json").exists():
        result.update(json.loads((root / "progress.json").read_text()))
    return result
