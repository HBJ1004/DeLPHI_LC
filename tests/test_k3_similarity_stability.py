import itertools
import json

import numpy as np
import pytest

from lc_pipeline.k3 import similarity_stability as a
from lc_pipeline.k3.reliability_study import binding, seal


def test_axes_sign_permutation_and_known_angles():
    axes = np.eye(3)
    assert a.set_displacement(axes, -axes[[2, 0, 1]]) == {
        "assignment_mean_deg": 0,
        "bottleneck_deg": 0,
    }
    b = np.repeat([[np.cos(np.pi / 6), np.sin(np.pi / 6), 0]], 3, axis=0)
    x = np.repeat([[1, 0, 0]], 3, axis=0)
    assert a.set_displacement(x, b)["assignment_mean_deg"] == pytest.approx(30)
    assert a.set_displacement(x, b)["bottleneck_deg"] == pytest.approx(30)


def test_assignment_matches_independent_exhaustive_distances():
    rng = np.random.default_rng(7)
    for _ in range(20):
        x, y = rng.normal(size=(2, 3, 3))
        costs = []
        for perm in itertools.permutations(range(3)):
            angles = [
                np.degrees(
                    np.arccos(
                        np.clip(abs(np.dot(u, v)) / (np.linalg.norm(u) * np.linalg.norm(v)), 0, 1)
                    )
                )
                for u, v in zip(x, y[list(perm)], strict=True)
            ]
            costs.append(angles)
        result = a.set_displacement(x, y)
        assert result["assignment_mean_deg"] == pytest.approx(min(map(np.mean, costs)))
        assert result["bottleneck_deg"] == pytest.approx(min(map(max, costs)))
        assert a.set_displacement(y, x) == pytest.approx(result)


@pytest.mark.parametrize(
    "bad", [np.zeros((3, 3)), np.ones((2, 3)), np.full((3, 3), np.nan), np.ones((3, 2))]
)
def test_invalid_candidate_axes_rejected(bad):
    with pytest.raises(ValueError):
        a.set_displacement(np.eye(3), bad)


def session():
    t = np.arange(32) / 24 + 2450000
    flux = 2 + 0.2 * np.sin(2 * np.pi * np.arange(32) / 8)
    return np.column_stack([t, flux, np.tile([1, 0, 0], (32, 1)), np.tile([0, 1, 0], (32, 1))])


def test_features_flux_scale_time_offset_and_session_order_invariance():
    x = session()
    y = x.copy()
    y[:, 0] += 10
    first = a.features_from_sessions([x, y], 8)
    x[:, 1] *= 17
    y[:, 1] *= 3
    x[:, 0] += 100
    y[:, 0] += 100
    other = a.features_from_sessions([y, x], 8)
    np.testing.assert_allclose(first, other, atol=1e-12)
    assert first.shape == (32,)
    assert first[24] == pytest.approx(np.pi / 2)
    assert first[25] == 0
    assert first[26] == pytest.approx(np.log(8))
    assert first[27] == pytest.approx(np.log(64))
    assert first[28] == pytest.approx(np.log(2))


@pytest.mark.parametrize("period", [0, -1, np.nan, np.inf])
def test_features_reject_invalid_period(period):
    with pytest.raises(ValueError):
        a.features_from_sessions([session()], period)


def test_scaler_training_only_query_independence_and_ties():
    train = {"c": np.zeros(32), "a": np.zeros(32), "b": np.ones(32)}
    result, scaler = a.nearest_training(train, {"q": np.zeros(32)})
    extended, other = a.nearest_training(train, {"q": np.zeros(32), "z": np.full(32, 1e6)})
    assert scaler == other
    assert result[0] == next(r for r in extended if r["object_id"] == "q")
    assert result[0]["neighbour_ids"] == ["a", "c", "b"]
    assert scaler["mean"] == pytest.approx(np.full(32, 1 / 3))
    assert result[0]["squared_distances"][2] == pytest.approx(4.5)
    with pytest.raises(ValueError, match="disjoint"):
        a.nearest_training(train, {"a": np.zeros(32)})


def test_training_constant_features_do_not_use_query_variance():
    train = {str(i): np.zeros(32) for i in range(3)}
    result, scaler = a.nearest_training(train, {"q": np.ones(32)})
    assert not any(scaler["active"])
    assert result[0]["squared_distances"] == [0, 0, 0]


def test_neighbour_candidates_use_fixed_catalog_order_and_keep_alternatives():
    refs = {"a": [[1, 0, 0], [0, 1, 0]], "b": [[0, 0, 1]], "c": [[1, 1, 0]]}
    result = a.neighbour_axes(["a", "b", "c"], refs)
    np.testing.assert_allclose(
        result["three_neighbour_primary_axes"], [[1, 0, 0], [0, 0, 1], [2**-0.5, 2**-0.5, 0]]
    )
    assert result["one_neighbour_all_axes"] == [[1, 0, 0], [0, 1, 0], [1, 0, 0]]


def test_predictions_are_unchanged_by_heldout_or_other_excluded_labels():
    roles = {
        "train_ids": ["a", "b", "c"],
        "test_ids": ["q"],
        "validation_ids": ["v"],
        "calibration_ids": ["cal"],
    }
    features = {oid: np.full(32, i) for i, oid in enumerate(["a", "b", "c", "q"])}
    refs = {oid: [[1, 0, 0]] for oid in ("a", "b", "c", "q", "v", "cal")}
    manifest = {"fold": 0, "object_roles": roles}
    first = a.predict_baselines({}, [manifest], features, refs)
    for oid in ("q", "v", "cal"):
        refs[oid] = [[0, 0, 1]]
    assert a.predict_baselines({}, [manifest], features, refs) == first


def test_roles_reject_overlap_and_wrong_folds():
    obj = [{"object_id": "a", "fold": 0}, {"object_id": "b", "fold": 1}]
    manifests = [
        {
            "fold": i,
            "object_roles": {
                "test_ids": [t],
                "train_ids": [r],
                "calibration_ids": [],
                "validation_ids": [],
            },
        }
        for i, t, r in [(0, "a", "b"), (1, "b", "a")]
    ]
    a.validate_roles(manifests, obj)
    manifests[0]["object_roles"]["train_ids"] = ["a"]
    with pytest.raises(ValueError, match="overlap"):
        a.validate_roles(manifests, obj)


def row(oid="a", condition="full", repeat=0, error=10, axes=None, status="ok"):
    return {
        "object_id": oid,
        "fold": 0,
        "condition": condition,
        "repeat": repeat,
        "status": status,
        "error_deg": error,
        "axes": np.eye(3).tolist() if axes is None and status == "ok" else axes,
        "source_sha256": "raw",
    }


def test_stability_averages_repeats_inside_object_and_reports_failures():
    x = np.repeat([[1, 0, 0]], 3, axis=0).tolist()
    y = np.repeat([[0, 1, 0]], 3, axis=0).tolist()
    rows = [
        row(axes=x),
        row(condition="c", axes=x, error=12),
        row(condition="c", repeat=1, axes=y, error=20),
        row(condition="c", repeat=2, status="failed", error=90),
        row(oid="b", axes=x),
        row(oid="b", condition="c", axes=x, error=11),
    ]
    objects, repeats, summaries = a.stability_tables(rows, draws=100)
    assert len(repeats) == 4
    assert objects[0]["assignment_mean_deg"] == 45
    assert objects[0]["repeat_pair_mean_deg"] == 90
    assert objects[0]["oracle_change_deg"] == pytest.approx((12 + 20 + 90) / 3 - 10)
    summary = summaries[0]
    assert summary["failed_rows"] == 1
    assert summary["assignment_mean_deg"]["mean"] == 22.5
    assert summary["assignment_mean_deg"]["n"] == 2


def test_stability_all_failed_does_not_invent_distances():
    rows = [row(), row(condition="c", status="failed", error=90)]
    objects, _, summaries = a.stability_tables(rows, draws=10)
    assert objects[0]["assignment_mean_deg"] is None
    assert summaries[0]["assignment_mean_deg"]["n"] == 0
    assert summaries[0]["oracle_change_deg"]["mean"] == 80


def test_inventory_checks_completeness_duplicate_source_and_fold():
    study = {
        "objects": [{"object_id": "a", "fold": 0, "lightcurve": {"sha256": "raw"}}],
        "conditions": [{"condition": "full", "family": "full", "repeatable": False}],
        "grid_ids": ["a"],
    }
    rows = [row(), row(condition="grid-full")]
    a.validate_saved_rows(study, {"repeats": 3}, rows)
    for bad in (
        rows[:1],
        rows + rows[:1],
        [rows[0], {**rows[1], "fold": 2}],
        [rows[0], {**rows[1], "source_sha256": "changed"}],
    ):
        with pytest.raises(ValueError):
            a.validate_saved_rows(study, {"repeats": 3}, bad)


def test_bootstrap_deterministic_and_asteroid_unit():
    rows = [{"object_id": str(i), "fold": i % 2, "value": i} for i in range(6)]
    assert a.distribution(rows, "value", draws=100) == a.distribution(
        rows[::-1], "value", draws=100
    )
    assert a.distribution(rows, "value", draws=100)["mean"] == 2.5
    with pytest.raises(ValueError, match="unique asteroid"):
        a.distribution(rows + rows[:1], "value", draws=100)


def test_prepare_rejects_existing_directory(tmp_path):
    with pytest.raises(FileExistsError):
        a.prepare(original_study=tmp_path / "absent", output=tmp_path)


def test_runner_rejects_changed_bound_input(tmp_path):
    source = tmp_path / "input.json"
    source.write_text("original")
    seal(tmp_path / "analysis-lock.json", {"inputs": {"input": binding(source)}})
    source.write_text("changed")
    with pytest.raises(ValueError, match="bound input changed"):
        a.run(output=tmp_path)


def test_saved_text_is_not_overwritten(tmp_path):
    p = tmp_path / "report.md"
    a._text_once(p, "first")
    with pytest.raises(FileExistsError):
        a._text_once(p, "second")
    assert p.read_text() == "first"


def test_reference_loader_retains_all_catalog_solutions(tmp_path):
    p = tmp_path / "catalog.jsonl"
    p.write_text(
        json.dumps(
            {
                "eligible": True,
                "object_id": "a",
                "solutions": [{"vector": [2, 0, 0]}, {"vector": [0, 3, 0]}],
            }
        )
        + "\n"
    )
    assert a._references(p) == {"a": [[1, 0, 0], [0, 1, 0]]}


def test_complete_prepare_predict_score_and_idempotent_verification(tmp_path):
    """Exercise the actual CLI functions on a complete synthetic 170-object fixture."""
    original = tmp_path / "original"
    original.mkdir()
    raw = original / "lc.txt"
    raw.write_text("1\n2 0\n2450000 1 1 0 0 0 1 0\n2450000.1 1.1 1 0 0 0 1 0\n")
    raw_binding = binding(raw)
    ids = [f"object_{i:03}" for i in range(170)]
    folds = []
    for fold in range(5):
        test = ids[fold * 34 : (fold + 1) * 34]
        rest = [i for i in ids if i not in test]
        folds.append(
            {
                "fold": fold,
                "train_ids": rest[:108],
                "validation_ids": rest[108:122],
                "calibration_ids": rest[122:],
                "test_ids": test,
            }
        )
    split_path = original / "publication-splits-v2.3.json"
    split_path.write_text(json.dumps({"folds": folds}))
    split_binding = binding(split_path)
    bundles = []
    for fold in folds:
        directory = original / f"bundle-{fold['fold']}"
        directory.mkdir()
        p = directory / "bundle.json"
        p.write_text(
            json.dumps(
                {
                    "fold": fold["fold"],
                    "splits_sha256": split_binding["sha256"],
                    "object_roles": {k: v for k, v in fold.items() if k != "fold"},
                }
            )
        )
        bundles.append({"directory": str(directory), "files": [binding(p)]})
    catalog = original / "catalog.jsonl"
    catalog.write_text(
        "\n".join(
            json.dumps({"object_id": i, "eligible": True, "solutions": [{"vector": [1, 0, 0]}]})
            for i in ids
        )
        + "\n"
    )
    objects = [
        {
            "object_id": oid,
            "fold": i // 34,
            "lightcurve": raw_binding,
            "period_hours": 8,
            "native_observations": 2,
            "native_sessions": 1,
        }
        for i, oid in enumerate(ids)
    ]
    conditions = [{"condition": "full", "family": "full", "repeatable": False}]
    conditions += [{"condition": c, "family": "point", "repeatable": True} for c in a.HIGHLIGHTS]
    study = {
        "objects": objects,
        "conditions": conditions,
        "grid_ids": ids,
        "bundles": bundles,
        "catalog": binding(catalog),
        "bindings": [split_binding],
        "atlases": [{"axes": np.eye(3).tolist()} for _ in folds],
    }
    rows = []
    for o in objects:
        for c in conditions + [{"condition": "grid-full", "repeatable": False}]:
            for repeat in range(3 if c["repeatable"] else 1):
                rows.append(
                    {
                        **row(o["object_id"], c["condition"], repeat, error=0),
                        "fold": o["fold"],
                        "source_sha256": raw_binding["sha256"],
                    }
                )
    for name, payload in (
        ("study.json", study),
        ("schedule.json", {"repeats": 3}),
        ("scored-rows.json", rows),
        ("inference-complete.json", {}),
    ):
        seal(original / name, payload)
    output = tmp_path / "analysis"
    assert a.prepare(original_study=original, output=output)["status"] == "prepared"
    result = a.run(output=output)
    assert result["status"] == "complete"
    summary = json.loads((output / "report/summary.json").read_text())
    assert summary["similarity"][0]["error_deg"]["mean"] == 0
    assert summary["max_saved_oracle_recomputation_difference_deg"] == 0
    assert summary["gpu_used"] is False
    assert a.run(output=output)["status"] == "already_complete"
    (output / "report/report.md").write_text("changed")
    with pytest.raises(ValueError, match="completed analysis output changed"):
        a.run(output=output)
