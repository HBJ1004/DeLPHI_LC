import subprocess

import numpy as np

from lc_pipeline.k3.reliability_supplement import _canon, _tex, axial_cap_fraction


def test_canonical_is_stable():
    assert _canon({"b": 1, "a": 2}) == '{"a":2,"b":1}\n'


def test_axial_cap_is_antipode_and_duplicate_invariant():
    n = 32768
    i = np.arange(n)
    z = 1 - 2 * (i + 0.5) / n
    phi = np.pi * (3 - np.sqrt(5)) * i
    sphere = np.c_[np.sqrt(1 - z * z) * np.cos(phi), np.sqrt(1 - z * z) * np.sin(phi), z]
    axis = np.array([[0.0, 0.0, 1.0]])
    radius = 20
    fraction = axial_cap_fraction(axis, sphere, radius)
    duplicated = axial_cap_fraction(np.r_[axis, axis], sphere, radius)
    antipode = axial_cap_fraction(-axis, sphere, radius)
    assert abs(fraction - (1 - np.cos(np.deg2rad(radius)))) < 0.002
    assert fraction == duplicated == antipode


def test_tex_tables_compile_and_escape(tmp_path):
    for name, rows in [
        (
            "data-summary.tex",
            [{"property": "merged_blocks", "n": 170, "min": 1.0, "median": 3.125, "max": 11.0}],
        ),
        (
            "fold-metrics.tex",
            [{"fold": 0, "n": 34, "mean_error_deg": 15.123456, "median_error_deg": 12.0}],
        ),
    ]:
        table = tmp_path / name
        _tex(table, rows)
        text = table.read_text()
        assert r"\begin{tabular}{" in text and r"\end{tabular}" in text
        if name == "data-summary.tex":
            assert "Merged blocks" in text
        wrapper = tmp_path / "wrapper.tex"
        wrapper.write_text(
            "\\documentclass{article}\n\\begin{document}\n" + text + "\\end{document}\n"
        )
        if __import__("shutil").which("pdflatex"):
            assert (
                subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode", "wrapper.tex"],
                    cwd=tmp_path,
                    capture_output=True,
                ).returncode
                == 0
            )
