from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

import pytest

from repro.package_k3_publication_evidence import (
    ARCHIVE_NAME,
    DERIVED_FILES,
    REVISION_CODE,
    package,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, *, sensitive_value: str | None = None) -> dict[str, Path]:
    base = tmp_path / "base"
    imported = tmp_path / "import"
    revision = tmp_path / "revision"
    paper = tmp_path / "paper"
    source = tmp_path / "source"
    (base / "evidence/exports").mkdir(parents=True)
    (base / "evidence/exports/old.txt").write_text("old\n")
    (imported / "internal").mkdir(parents=True)
    (imported / "revision").mkdir(parents=True)
    revision.mkdir()
    paper.mkdir()
    source.mkdir()

    internal = imported / "internal/result.json"
    internal.write_text(json.dumps({"value": sensitive_value or "public"}))
    revision_row = revision / "timing-cases.csv"
    revision_row.write_text("object_id\n1\n")
    revision_manifest = {
        "schema": "delphi.k3-publication-revision-export.v1",
        "files": {"timing-cases.csv": _sha256(revision_row)},
    }
    encoded_revision = json.dumps(revision_manifest, sort_keys=True)
    (revision / "manifest.json").write_text(encoded_revision)
    (imported / "revision/timing-cases.csv").write_bytes(revision_row.read_bytes())
    (imported / "revision/export-manifest.json").write_text(encoded_revision)
    imported_manifest = {
        "schema": "delphi.paper-followup-import.v1",
        "reports": {},
        "files": {
            "internal/result.json": _sha256(internal),
            "revision/timing-cases.csv": _sha256(imported / "revision/timing-cases.csv"),
            "revision/export-manifest.json": _sha256(
                imported / "revision/export-manifest.json"
            ),
        },
    }
    (imported / "manifest.json").write_text(json.dumps(imported_manifest))

    for name in DERIVED_FILES:
        (paper / name).write_text(name)
    scripts = paper / "tools"
    scripts.mkdir()
    (scripts / "import_followup.py").write_text(
        "import argparse\n"
        "from pathlib import Path\n"
        "p=argparse.ArgumentParser(); p.add_argument('--verify',action='store_true'); "
        "p.add_argument('--output'); a=p.parse_args(); "
        "assert (Path(a.output)/'manifest.json').is_file()\n"
    )
    required = repr(list(DERIVED_FILES))
    (scripts / "derive_followup.py").write_text(
        "import argparse\n"
        "from pathlib import Path\n"
        "p=argparse.ArgumentParser(); p.add_argument('--verify',action='store_true'); "
        "p.parse_args(); root=Path(__file__).resolve().parents[1]; "
        f"assert all((root/name).is_file() for name in {required})\n"
    )
    for name in REVISION_CODE:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    return {
        "base": base,
        "imported": imported,
        "revision": revision,
        "paper": paper,
        "source": source,
    }


def _package(paths: dict[str, Path], output: Path) -> dict[str, object]:
    return package(
        base_staging=paths["base"],
        followup_import=paths["imported"],
        revision_export=paths["revision"],
        paper_root=paths["paper"],
        source_root=paths["source"],
        output=output,
    )


def test_package_contains_revision_and_is_deterministic(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    first = _package(paths, tmp_path / "one")
    second = _package(paths, tmp_path / "two")
    assert first["sha256"] == second["sha256"]
    with tarfile.open(tmp_path / "one" / ARCHIVE_NAME, "r:gz") as archive:
        names = archive.getnames()
    assert "./k3-followup/revision/timing-cases.csv" in names
    assert "./k3-followup/manifest.json" in names
    assert "./k3-followup/revision/manifest.json" not in names
    assert "./evidence/exports/revision/timing-cases.csv" in names
    assert "./evidence/exports/revision/manifest.json" in names
    assert "./evidence/exports/manifest.json" in names
    assert "./manuscript-derived/k3-followup-lowq-table.tex" in names
    assert "./k3-followup-lowq-table.tex" in names


@pytest.mark.parametrize(
    "sensitive",
    ["/mnt/d/private/run", "NVIDIA RTX, GPU-01234567-abcd, 12 GiB"],
)
def test_package_rejects_private_machine_identifiers(
    tmp_path: Path, sensitive: str
) -> None:
    paths = _fixture(tmp_path, sensitive_value=sensitive)
    with pytest.raises(ValueError, match="private machine identifier"):
        _package(paths, tmp_path / "release")


def test_package_rejects_failed_documented_verification(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    (paths["paper"] / "tools/derive_followup.py").write_text("raise SystemExit(2)\n")
    with pytest.raises(ValueError, match="documented verification command failed"):
        _package(paths, tmp_path / "release")
