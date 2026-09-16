import numpy as np
import pytest

from lc_pipeline.k3.internal_validation_reporting import (
    export,
    fold_bootstrap,
    oof_tables,
    reassignment_null,
    train_oof_pairs,
)


def test_train_pairs_average_repeated_asteroid_errors():
    cells = [
        {"role": "train", "object_id": "a", "error_deg": 10},
        {"role": "train", "object_id": "a", "error_deg": 30},
    ]
    pairs = train_oof_pairs(cells, [{"object_id": "a", "status": "ok", "error_deg": 25}])
    assert pairs[0]["mean_train_error_deg"] == 20
    assert pairs[0]["train_repetitions"] == 2


def test_reassignment_preserves_singleton_reference_strata():
    oof = [
        {"object_id": "a", "fold": 0, "axes": [[1, 0, 0]] * 3, "error_deg": 1},
        {"object_id": "b", "fold": 1, "axes": [[1, 0, 0]] * 3, "error_deg": 2},
    ]
    refs = {"a": [[1, 0, 0]], "b": [[0, 1, 0], [0, 0, 1]]}
    out = reassignment_null(oof, refs, n=3)
    assert out["movable_objects"] == 0 and out["fraction_movable"] == 0


def test_oof_tables_include_leave_one_fold_out_and_reference_count():
    rows = [
        {"fold": 0, "seed": 1, "reference_solution_count": 1, "error_deg": 10},
        {"fold": 1, "seed": 1, "reference_solution_count": 2, "error_deg": 20},
    ]
    table = oof_tables(rows)
    assert table["leave_one_fold_out"][0]["mean_error_deg"] == 20
    assert table["reference_count_strata"][1]["n"] == 1


def test_existing_report_is_not_overwritten(tmp_path):
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        export(root=tmp_path, output=tmp_path)


def test_paired_bootstrap_uses_same_resampled_objects_for_all_means():
    pairs = [
        {"object_id": "a", "mean_train_error_deg": 10, "oof_error_deg": 20, "difference_deg": -10},
        {"object_id": "b", "mean_train_error_deg": 15, "oof_error_deg": 12, "difference_deg": 3},
        {"object_id": "c", "mean_train_error_deg": 5, "oof_error_deg": 9, "difference_deg": -4},
    ]
    folds = {"a": 0, "b": 0, "c": 1}
    result = fold_bootstrap(pairs, folds, n=100, seed=7)
    assert result["joint_draw_consistency_max_abs"] < 1e-12
    assert result == fold_bootstrap(list(reversed(pairs)), folds, n=100, seed=7)
    assert result["mean_difference_deg"] == pytest.approx(-11 / 3)


def test_reassignment_matches_literal_axial_formula():
    rows = [
        {"object_id": "a", "fold": 0, "axes": [[1, 0, 0]] * 3, "error_deg": 0},
        {"object_id": "b", "fold": 0, "axes": [[0, 1, 0]] * 3, "error_deg": 0},
        {"object_id": "c", "fold": 1, "axes": [[0, 0, 1]] * 3, "error_deg": 0},
    ]
    refs = {"a": [[1, 0, 0]], "b": [[0, 1, 0]], "c": [[0, 0, 1]]}
    rng = np.random.default_rng(19)
    expected = []
    for _ in range(20):
        errors = []
        for group in (rows[:2], rows[2:]):
            assigned = rng.permutation([refs[row["object_id"]] for row in group])
            for row, references in zip(group, assigned, strict=True):
                dot = np.abs(np.asarray(row["axes"]) @ np.asarray(references).T).max()
                errors.append(float(np.degrees(np.arccos(np.clip(dot, 0, 1)))))
        expected.append(float(np.mean(errors)))
    actual = reassignment_null(rows, refs, n=20, seed=19)
    assert actual["null_draws_mean_error_deg"] == expected
