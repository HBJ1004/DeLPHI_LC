import json

import pytest

from lc_pipeline.k3.internal_validation import ActiveBudget, _cell_path, role_for


def test_role_for_requires_exactly_one_role():
    m = {
        "object_roles": {
            "train_ids": ["a"],
            "validation_ids": ["b"],
            "calibration_ids": ["c"],
            "test_ids": ["d"],
        }
    }
    assert role_for(m, "a") == "train_ids"
    with pytest.raises(ValueError):
        role_for(m, "missing")
    m["object_roles"]["test_ids"].append("a")
    with pytest.raises(ValueError):
        role_for(m, "a")


def test_active_budget_is_new_and_checkpointed(tmp_path):
    budget = ActiveBudget(tmp_path, seconds=10)
    budget.charge(3.5)
    assert ActiveBudget(tmp_path, seconds=10).remaining() == pytest.approx(6.5)
    with pytest.raises(ValueError):
        ActiveBudget(tmp_path, seconds=11)


def test_interrupted_cell_is_conservatively_charged_without_wall_downtime(tmp_path):
    budget = ActiveBudget(tmp_path, seconds=100)
    (tmp_path / "inflight.json").write_text(json.dumps({"reserved_active_seconds": 17}))
    budget.recover_interrupted(tmp_path)
    assert budget.remaining() == pytest.approx(83)
    assert not (tmp_path / "inflight.json").exists()


def test_cell_paths_keep_repeated_training_object_fold_distinct(tmp_path):
    one = {"fold": 0, "role": "train", "object_id": "asteroid_1"}
    two = {"fold": 3, "role": "train", "object_id": "asteroid_1"}
    assert _cell_path(tmp_path, one) != _cell_path(tmp_path, two)
    assert "fold-0" in str(_cell_path(tmp_path, one))
