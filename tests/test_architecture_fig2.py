"""Regression checks for Fig. 2 of Paper I, without requiring Inkscape."""

from __future__ import annotations

import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from repro.figures import architecture_fig2

REFERENCE = Path(__file__).parent / "data" / "architecture_fig2_reference.svg"


def test_svg_matches_reference(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PATH", "")
    output = tmp_path / "figures"
    subprocess.run(
        [sys.executable, "-m", "repro.figures.architecture_fig2", "--out-dir", str(output)],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )

    assert (output / "architecture.svg").read_bytes() == REFERENCE.read_bytes()
    assert not (output / "architecture.pdf").exists()


@pytest.mark.parametrize("title", ["A Encoder", "B Scorer", "C Starting poles"])
def test_panel_titles(title) -> None:
    root = ET.fromstring(architecture_fig2.build_svg())
    assert root.attrib["viewBox"] == "0 0 1000 950"
    assert title in " ".join(root.itertext())


def test_pdf_requires_inkscape(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(architecture_fig2.shutil, "which", lambda name: None)
    with pytest.raises(SystemExit) as error:
        architecture_fig2.main(["--out-dir", str(tmp_path / "figures"), "--pdf"])

    assert error.value.code == 2
    assert "--pdf requires Inkscape" in capsys.readouterr().err
