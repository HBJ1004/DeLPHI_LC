from __future__ import annotations

from pathlib import Path

import yaml

from lc_pipeline.version import __version__, get_version_info

ROOT = Path(__file__).resolve().parents[1]


def test_release_identity_is_consistent():
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    assert __version__ == citation["version"] == "1.1.0"
    assert citation["url"] == "https://github.com/HBJ1004/DeLPHI_LC/releases/tag/v1.1.0"
    assert str(citation["date-released"]) == "2026-11-01"
    assert any(identifier["type"] == "doi" and identifier["value"] == "10.5281/zenodo.23757942"
               for identifier in citation["identifiers"])
    for doi in ("10.5281/zenodo.23787947", "10.5281/zenodo.23857881"):
        assert any(identifier["type"] == "doi" and identifier["value"] == doi
                   for identifier in citation["identifiers"])
    code = next(identifier for identifier in citation["identifiers"]
                if identifier["value"] == "10.5281/zenodo.23787947")
    assert "all versions (concept DOI)" in code["description"]
    for authors in (citation["authors"], citation["preferred-citation"]["authors"]):
        lee = next(author for author in authors if author["family-names"] == "Lee")
        assert lee["orcid"] == "https://orcid.org/0000-0002-6839-075X"
    assert get_version_info()["scientific_release_approved"] is True
    assert "Phase 37" not in (ROOT / "LICENSE").read_text(encoding="utf-8")
