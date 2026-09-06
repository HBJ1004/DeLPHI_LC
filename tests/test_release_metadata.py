from __future__ import annotations

from pathlib import Path

import yaml

from lc_pipeline.version import __version__, get_version_info

ROOT = Path(__file__).resolve().parents[1]


def test_release_identity_is_consistent():
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    assert __version__ == citation["version"] == "1.0"
    assert citation["url"].endswith("/releases/tag/v1.0")
    assert get_version_info()["scientific_release_approved"] is True
    assert "Phase 37" not in (ROOT / "LICENSE").read_text(encoding="utf-8")
