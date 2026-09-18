#!/usr/bin/env python3
"""Build the date-named public evidence archive for the final manuscript revision."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

DATE = "2026-09-17"
RELEASE_ID = f"{DATE}-r4"
ARCHIVE_NAME = f"delphi-k3-publication-evidence-{RELEASE_ID}.tar.gz"
DERIVED_FILES = (
    "k3-followup-derived.json",
    "k3-followup-results.tex",
    "k3-followup-withheld-table.tex",
    "k3-followup-grid-table.tex",
    "k3-followup-transfer-table.tex",
    "k3-followup-lowq-table.tex",
    "k3-followup-ensemble-table.tex",
    "k3-followup-seed-table.tex",
)
REVISION_CODE = (
    "lc_pipeline/k3/followup_transfer_reporting.py",
    "repro/analyze_k3_lowq_lineage.py",
    "repro/audit_k3_lowq_exposure.py",
    "repro/build_k3_external_atlas.py",
    "repro/export_k3_grid_timing_cases.py",
    "repro/export_k3_revision_evidence.py",
    "repro/package_k3_publication_evidence.py",
    "repro/run_k3_error_channel_sensitivity.py",
    "repro/run_k3_lowq_ablation.py",
    "repro/run_k3_lowq_census.py",
    "repro/run_k3_lowq_damit.py",
)
SENSITIVE_PROVENANCE = re.compile(
    r"(?:/mnt/[a-z]/|/home/[^/\s]+/|/users/|[a-z]:\\+users\\+|GPU-[0-9a-f-]{8,})",
    re.I,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_file(source: Path, target: Path) -> None:
    if not source.is_file():
        raise ValueError(f"missing evidence input: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def _write_checksums(root: Path) -> None:
    files = sorted(
        path for path in root.rglob("*") if path.is_file() and path.name != "SHA256SUMS"
    )
    lines = [f"{sha256(path)}  {path.relative_to(root).as_posix()}" for path in files]
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _verify_file_manifest(root: Path, manifest: dict[str, object], label: str) -> None:
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError(f"{label} contains no files")
    for name, expected in files.items():
        relative = Path(str(name))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe path in {label}: {name}")
        path = root / relative
        if not path.is_file() or not isinstance(expected, str) or sha256(path) != expected:
            raise ValueError(f"hash mismatch in {label}: {name}")


def _reject_sensitive_provenance(root: Path) -> None:
    inspected = {".csv", ".json", ".md", ".tex", ".txt", ".yaml", ".yml"}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in inspected:
            text = path.read_text(encoding="utf-8", errors="replace")
            if SENSITIVE_PROVENANCE.search(text):
                raise ValueError(
                    f"private machine identifier in public evidence: {path.relative_to(root)}"
                )


def _run_documented_verification(root: Path) -> None:
    commands = (
        [
            sys.executable,
            "manuscript-derived/import_followup.py",
            "--verify",
            "--output",
            "k3-followup",
        ],
        [sys.executable, "manuscript-derived/derive_followup.py", "--verify"],
    )
    for command in commands:
        result = subprocess.run(
            command,
            cwd=root,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if result.returncode:
            raise ValueError(
                "documented verification command failed: "
                + " ".join(command[1:])
                + "\n"
                + result.stdout
            )


def _tar_deterministic(root: Path, output: Path) -> None:
    with output.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for path in [root, *sorted(root.rglob("*"))]:
                    relative = "." if path == root else f"./{path.relative_to(root).as_posix()}"
                    info = archive.gettarinfo(str(path), arcname=relative)
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    info.mode = 0o755 if path.is_dir() else 0o644
                    if path.is_file():
                        with path.open("rb") as handle:
                            archive.addfile(info, handle)
                    else:
                        archive.addfile(info)


def package(
    *,
    base_staging: Path,
    followup_import: Path,
    revision_export: Path,
    paper_root: Path,
    source_root: Path,
    output: Path,
) -> dict[str, object]:
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    manifest = json.loads((followup_import / "manifest.json").read_text(encoding="utf-8"))
    revision_manifest = json.loads((revision_export / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "delphi.paper-followup-import.v1":
        raise ValueError("follow-up import has the wrong schema")
    if revision_manifest.get("schema") != "delphi.k3-publication-revision-export.v1":
        raise ValueError("revision export has the wrong schema")
    _verify_file_manifest(followup_import, manifest, "follow-up import manifest")
    _verify_file_manifest(revision_export, revision_manifest, "revision export manifest")
    imported_revision_manifest = followup_import / "revision" / "export-manifest.json"
    if (
        not imported_revision_manifest.is_file()
        or imported_revision_manifest.read_bytes()
        != (revision_export / "manifest.json").read_bytes()
    ):
        raise ValueError("follow-up import does not contain the requested revision export")

    staging = output / "staging"
    shutil.copytree(base_staging, staging)
    followup = staging / "k3-followup"
    shutil.rmtree(followup, ignore_errors=True)
    shutil.copytree(followup_import, followup)
    public_exports = staging / "evidence" / "exports"
    shutil.rmtree(public_exports, ignore_errors=True)
    shutil.copytree(followup_import, public_exports)
    shutil.copyfile(
        revision_export / "manifest.json", public_exports / "revision" / "manifest.json"
    )

    derived = staging / "manuscript-derived"
    for name in DERIVED_FILES:
        _copy_file(paper_root / name, derived / name)
        _copy_file(paper_root / name, staging / name)
    for name in ("tools/import_followup.py", "tools/derive_followup.py"):
        _copy_file(paper_root / name, derived / Path(name).name)
    code = staging / "revision-code"
    for name in REVISION_CODE:
        _copy_file(source_root / name, code / Path(name).name)

    (staging / "README.md").write_text(
        "# DeLPHI K3 publication evidence, 2026-09-17 revision 4\n\n"
        "This archive adds the lower-quality DAMIT census, source-lineage split, "
        "ensemble-cost sensitivity, error-channel sensitivity, and public per-case "
        "broad-grid timing table used by the revised manuscript. The analyses are "
        "post hoc and do not alter the frozen v1.0.0 weights or primary out-of-fold "
        "predictions. The broad-grid result is a host-specific descriptive speed--fit "
        "trade-off, not a general acceleration claim. The ZTF and ALCDEF--Gaia checks "
        "do not establish cross-survey reliability.\n",
        encoding="utf-8",
    )
    (staging / "VERIFY.md").write_text(
        "# Verification\n\n"
        "From this directory, run `sha256sum -c SHA256SUMS`, then "
        "`python manuscript-derived/import_followup.py --verify --output k3-followup`, "
        "and then `python manuscript-derived/derive_followup.py --verify`. The file "
        "`k3-followup/manifest.json` binds the imported report groups, and "
        "`k3-followup/revision/export-manifest.json` binds the added evidence. The revision "
        "export was generated with "
        "`python -m repro.export_k3_revision_evidence`; its manifest records every "
        "input hash.\n",
        encoding="utf-8",
    )
    _reject_sensitive_provenance(staging)
    _write_checksums(staging)
    _run_documented_verification(staging)
    output.mkdir(parents=True, exist_ok=True)
    archive = output / ARCHIVE_NAME
    _tar_deterministic(staging, archive)
    digest = sha256(archive)
    (output / "SHA256SUMS").write_text(f"{digest}  {ARCHIVE_NAME}\n", encoding="utf-8")
    (output / "VERIFY.md").write_text(
        f"Run `sha256sum -c SHA256SUMS`, then extract `{ARCHIVE_NAME}` and run "
        "`sha256sum -c SHA256SUMS` inside it.\n",
        encoding="utf-8",
    )
    return {
        "schema": "delphi.k3-publication-evidence-package.v1",
        "date": DATE,
        "release_id": RELEASE_ID,
        "archive": ARCHIVE_NAME,
        "sha256": digest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "base-staging",
        "followup-import",
        "revision-export",
        "paper-root",
        "source-root",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = package(
            base_staging=args.base_staging,
            followup_import=args.followup_import,
            revision_export=args.revision_export,
            paper_root=args.paper_root,
            source_root=args.source_root,
            output=args.output,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
