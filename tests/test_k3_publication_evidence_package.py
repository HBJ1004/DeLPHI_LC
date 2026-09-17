from __future__ import annotations

import json
import tarfile
from pathlib import Path

from repro.package_k3_publication_evidence import (
    ARCHIVE_NAME,
    DERIVED_FILES,
    REVISION_CODE,
    package,
)


def test_package_contains_revision_and_is_deterministic(tmp_path: Path) -> None:
    base = tmp_path / "base"
    (base / "evidence/exports").mkdir(parents=True)
    (base / "evidence/exports/old.txt").write_text("old\n")
    imported = tmp_path / "import"
    revision = tmp_path / "revision"
    paper = tmp_path / "paper"
    source = tmp_path / "source"
    imported.mkdir()
    revision.mkdir()
    paper.mkdir()
    source.mkdir()
    (imported / "manifest.json").write_text(
        json.dumps({"schema": "delphi.paper-followup-import.v1"})
    )
    (revision / "manifest.json").write_text(
        json.dumps({"schema": "delphi.k3-publication-revision-export.v1"})
    )
    (revision / "timing-cases.csv").write_text("object_id\n1\n")
    for name in DERIVED_FILES:
        (paper / name).write_text(name)
    for name in ("tools/import_followup.py", "tools/derive_followup.py"):
        path = paper / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    for name in REVISION_CODE:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)

    first = package(
        base_staging=base,
        followup_import=imported,
        revision_export=revision,
        paper_root=paper,
        source_root=source,
        output=tmp_path / "one",
    )
    second = package(
        base_staging=base,
        followup_import=imported,
        revision_export=revision,
        paper_root=paper,
        source_root=source,
        output=tmp_path / "two",
    )
    assert first["sha256"] == second["sha256"]
    with tarfile.open(tmp_path / "one" / ARCHIVE_NAME, "r:gz") as archive:
        names = archive.getnames()
    assert "./k3-followup/revision/timing-cases.csv" in names
    assert "./k3-followup/manifest.json" in names
    assert "./manuscript-derived/k3-followup-lowq-table.tex" in names
