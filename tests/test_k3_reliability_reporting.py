from pathlib import Path

import pytest

from lc_pipeline.k3.reliability_reporting import ReliabilityReportingError, write_reports
from lc_pipeline.k3.reliability_scoring import score_reliability


def _rows():
    axes = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    return [
        {
            "object_id": oid,
            "fold": index,
            "condition": "full",
            "repeat": 0,
            "seed": 20260915,
            "status": "ok",
            "axes": axes,
            "error_deg": 10.0 + index,
            "metadata": {"n_points": 25 + index, "period_hours": 8.0 + index},
            "reference_solution_count": 1 + index,
        }
        for index, oid in enumerate(("a", "b"))
    ]


def test_write_reports_materialises_tables_figures_and_conditional_guidance(tmp_path: Path):
    rows = _rows()
    score = score_reliability(
        rows,
        expected_objects={"a": 0, "b": 1},
        expected_conditions=("full",),
        expected_repeats=(0,),
        sampling_seed_by_repeat={0: 20260915},
    )
    lock = {
        "objects": [{"object_id": "a"}, {"object_id": "b"}],
        "conditions": [{"condition": "full", "family": "full"}],
    }
    paths = write_reports(tmp_path, rows, {"sampling": score}, lock)
    names = {path.name for path in paths}
    assert {
        "reliability-report.csv",
        "reliability-report.tex",
        "software-requirements.csv",
        "README-guidance.md",
    } <= names
    assert (tmp_path / "sampling-caps.pdf").is_file()
    assert "universal observation-count" in (tmp_path / "README-guidance.md").read_text()
    # Resume verifies the numeric manifest and leaves existing PDFs intact.
    write_reports(tmp_path, rows, {"sampling": score}, lock)
    changed = [dict(row) for row in rows]
    changed[0]["error_deg"] = 11.0
    with pytest.raises(ReliabilityReportingError, match="numeric report inputs"):
        write_reports(tmp_path, changed, {"sampling": score}, lock)
