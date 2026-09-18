"""Adversarial contracts for the grid-benchmark freeze boundary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from lc_pipeline import grid_benchmark as grid
from lc_pipeline.k3.solver_capacity import internal_capacity_padding


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _fixture(tmp_path: Path) -> dict[str, Path]:
    object_ids = [f"asteroid_{number}" for number in range(170)]
    folds = []
    for fold in range(5):
        tests = object_ids[fold * 34 : (fold + 1) * 34]
        folds.append(
            {
                "fold": fold,
                "train_ids": [f"train-{fold}-{index}" for index in range(108)],
                "validation_ids": [f"validation-{fold}-{index}" for index in range(14)],
                "calibration_ids": [f"calibration-{fold}-{index}" for index in range(14)],
                "test_ids": tests,
            }
        )
    splits = tmp_path / "splits.json"
    _write(splits, {"schema": "delphi.grouped-splits.v2", "folds": folds})
    salt = "delphi-k3-convergence-followup-20260910"
    development = []
    for fold in folds:
        development.extend(
            sorted(
                fold["test_ids"],
                key=lambda value: hashlib.sha256(f"{salt}:{value}".encode("ascii")).hexdigest(),
            )[:6]
        )
    development_path = tmp_path / "development.json"
    _write(
        development_path,
        {
            "schema": "delphi.k3-convergence-object-subset.v1",
            "role": "development",
            "salt": salt,
            "selection": "six_development_objects_per_original_outer_fold_by_sha256_rank",
            "source_full_split_sha256": _digest(splits),
            "object_ids": development,
        },
    )
    dump = tmp_path / "dump"
    blind_rows, capacity_rows = [], []
    for fold in folds:
        for object_id in fold["test_ids"]:
            lightcurve = dump / "files" / object_id / "lc.txt"
            lightcurve.parent.mkdir(parents=True, exist_ok=True)
            lightcurve.write_text(
                "1\n3 0\n0 1 1 0 0 0 1 0\n1 1 1 0 0 0 1 0\n2 1 1 0 0 0 1 0\n",
                encoding="ascii",
            )
            structure = {"lightcurve_count": 1, "total_observations": 3, "max_points_per_lightcurve": 3}
            blind_rows.append(
                {
                    "object_id": object_id,
                    "fold": fold["fold"],
                    "period_hours": 5.0,
                    "lightcurve": {"source_path": f"files/{object_id}/lc.txt", "source_sha256": _digest(lightcurve)},
                }
            )
            capacity_rows.append(
                {"object_id": object_id, "structure": structure,
                 "internal_required_capacities": {"POINTS_MAX": 3, "MAX_N_OBS": 6, "MAX_LC": 2},
                 "violations": []}
            )
    catalog = tmp_path / "catalog.jsonl"
    catalog.write_text("{}\n", encoding="utf-8")
    blind = tmp_path / "blind.json"
    _write(
        blind,
        {"source_full_split_sha256": _digest(splits), "source_catalog_sha256": _digest(catalog), "objects": blind_rows},
    )
    solver = tmp_path / "solver"
    solver.mkdir()
    (solver / "constants.h").write_text(
        "#define POINTS_MAX 3\n#define MAX_N_OBS 6\n#define MAX_LC 2\n", encoding="ascii"
    )
    for name in ("convexinv", "period_scan"):
        (solver / name).write_bytes(name.encode("ascii"))
    report = tmp_path / "capacity.json"
    _write(
        report,
        {
            "schema": "delphi.k3-solver-internal-capacity-preflight.v2",
            "internal_capacity_padding": internal_capacity_padding(),
            "development_supported_input_parity_smoke": {
                "status": "passed", "output_hashes_match": True, "object_id": "asteroid_227",
                "official": {"return_code": 0}, "expanded": {"return_code": 0}},
            "expanded_solver": {
                "source_root": str(solver), "source_tree_sha256": grid.digest(solver / "constants.h"),
                "binary": str(solver / "convexinv"), "binary_sha256": _digest(solver / "convexinv"),
                "period_scan_binary": str(solver / "period_scan"),
                "period_scan_binary_sha256": _digest(solver / "period_scan"),
            },
            "expanded_solver_preflight": {
                "capacities": {"POINTS_MAX": 3, "MAX_N_OBS": 6, "MAX_LC": 2},
                "development": {"objects": [row for row in capacity_rows if row["object_id"] in development]},
                "locked_evaluation": {"objects": [row for row in capacity_rows if row["object_id"] not in development]},
            },
            "sanitizer_boundary_checks": {
                "status": "passed",
                "runs": {object_id: {name: {"return_code": 0} for name in ("convexinv", "period_scan")}
                         for object_id in ("asteroid_109", "asteroid_2512")},
            },
        },
    )
    # tree_sha256 includes every file, unlike the short digest deliberately used above.
    from lc_pipeline.k3.solver_capacity import tree_sha256
    receipt = json.loads(report.read_text(encoding="utf-8"))
    receipt["expanded_solver"]["source_tree_sha256"] = tree_sha256(solver)
    _write(report, receipt)
    bundles = tmp_path / "bundles"
    for fold in folds:
        directory = bundles / f"k3-oof-fold-{fold['fold']}"
        directory.mkdir(parents=True)
        members = []
        for seed in (17, 42, 137, 777, 2027):
            weight = directory / f"seed-{seed}.safetensors"
            weight.write_bytes(f"{fold['fold']}:{seed}".encode("ascii"))
            members.append({"seed": seed, "file": weight.name, "sha256": _digest(weight),
                            "source_checkpoint_sha256": "a" * 64})
        _write(directory / "bundle.json", {"schema": "delphi.k3-ensemble-bundle.v1",
               "bundle_id": directory.name, "fold": fold["fold"], "splits_sha256": _digest(splits),
               "object_roles": {key: fold[key] for key in ("train_ids", "validation_ids", "calibration_ids", "test_ids")},
               "members": members})
    return {"blind": blind, "splits": splits, "development": development_path, "dump": dump,
            "bundles": bundles, "capacity": report, "catalog": catalog}


def _freeze(values: dict[str, Path], output: Path) -> dict:
    return grid.freeze(blind_inputs=values["blind"], splits=values["splits"], development=values["development"],
                       dump_root=values["dump"], bundle_root=values["bundles"], capacity_report=values["capacity"],
                       reference_catalog=values["catalog"], output=output, device="cpu")


def test_environment_rejects_insufficient_benchmark_affinity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        grid.os,
        "sched_getaffinity",
        lambda _pid: set(range(grid.SETTINGS["workers"] - 1)),
        raising=False,
    )
    with pytest.raises(grid.GridBenchmarkError, match="at least six"):
        grid._environment("cpu")


def test_freeze_requires_exact_salted_development_manifest_and_bundle_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        grid,
        "_environment",
        lambda device: {"device": device, "cpu_affinity": list(range(6))},
    )
    values = _fixture(tmp_path)
    lock = _freeze(values, tmp_path / "lock.json")
    assert len(lock["pilot_ids"]) == 5
    development = json.loads(values["development"].read_text(encoding="utf-8"))
    development["object_ids"] = development["object_ids"][:-1]
    _write(values["development"], development)
    with pytest.raises(grid.GridBenchmarkError, match="exact frozen 30-object"):
        _freeze(values, tmp_path / "bad-development.json")


def test_freeze_rejects_missing_bundle_roles_and_seed_membership(tmp_path: Path):
    values = _fixture(tmp_path)
    manifest_path = values["bundles"] / "k3-oof-fold-0" / "bundle.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["object_roles"].pop("calibration_ids")
    _write(manifest_path, manifest)
    with pytest.raises(grid.GridBenchmarkError, match="bundle roles"):
        _freeze(values, tmp_path / "bad-roles.json")
    values = _fixture(tmp_path / "seed-case")
    manifest_path = values["bundles"] / "k3-oof-fold-0" / "bundle.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["members"][0]["seed"] = 99
    _write(manifest_path, manifest)
    with pytest.raises(grid.GridBenchmarkError, match="ordered evaluated seeds"):
        _freeze(values, tmp_path / "bad-seeds.json")


def test_validated_capacity_requires_both_boundary_sanitizer_runs(tmp_path: Path):
    values = _fixture(tmp_path)
    blind = json.loads(values["blind"].read_text(encoding="utf-8"))
    rows = [{"object_id": row["object_id"], "structure": {"lightcurve_count": 1,
            "total_observations": 3, "max_points_per_lightcurve": 3}} for row in blind["objects"]]
    receipt = json.loads(values["capacity"].read_text(encoding="utf-8"))
    receipt["sanitizer_boundary_checks"]["runs"].pop("asteroid_109")
    _write(values["capacity"], receipt)
    with pytest.raises(grid.GridBenchmarkError, match="sanitizer evidence"):
        grid._validated_capacity(values["capacity"], rows)
