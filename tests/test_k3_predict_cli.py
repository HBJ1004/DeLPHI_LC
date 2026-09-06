"""User-facing validation errors for the JSON prediction command."""

from __future__ import annotations

import json

import pytest

from lc_pipeline.k3.predict import main, read_observations


def test_read_observations_names_missing_required_field(tmp_path) -> None:
    path = tmp_path / "observations.json"
    path.write_text(
        json.dumps(
            {
                "schema": "delphi.k3-observations.v1",
                "object_id": "example",
                "epochs": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing required observation field: known_period"):
        read_observations(path)


def test_cli_reports_validation_error_without_traceback(tmp_path, capsys) -> None:
    path = tmp_path / "observations.json"
    path.write_text(
        json.dumps({"schema": "delphi.k3-observations.v1", "object_id": "example"}),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "--bundle",
                str(tmp_path / "bundle"),
                "--input",
                str(path),
                "--output",
                str(tmp_path / "prediction.json"),
            ]
        )
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "missing required observation field: known_period" in captured.err
    assert "Traceback" not in captured.err
