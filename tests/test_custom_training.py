"""Custom JSONL reader keeps user data validation close to the guide."""

from __future__ import annotations

from pathlib import Path

import pytest

from repro.train_k3_custom import read_examples


def test_documented_custom_example_loads() -> None:
    root = Path(__file__).resolve().parents[1]
    values = read_examples(root / "examples/training.example.jsonl")
    assert len(values) == 1
    assert values[0].object_id == "training-example-001"
    assert values[0].target_axes.shape == (1, 3)


def test_custom_reader_rejects_duplicate_object_ids(tmp_path) -> None:
    path = tmp_path / "duplicate.jsonl"
    source = (Path(__file__).resolve().parents[1] / "examples/training.example.jsonl").read_text()
    path.write_text(source + source)
    with pytest.raises(ValueError, match="unique"):
        read_examples(path)
