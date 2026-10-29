#!/usr/bin/env python3
"""Acquire Job 1 inputs, preserving bytes and SHA256 receipts; never run models.

Run with the prescribed interpreter's -I option. A failed source is recorded and
the command exits nonzero. In particular, an inaccessible CDS catalogue is not
replaced with observations from another source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

from repro.own_period import DataPath

FOLLOWUP = DataPath("../..")
OUTPUT = DataPath('spacecraft')
ALCDEF_ZIP = FOLLOWUP / "data/training-expansion-cache/ALCDEF_ALL-20250428.zip"
PDS = "https://sbnarchive.psi.edu/pds4/non_mission/orex.gbo.ast-bennu.lightcurves-images_V1_0/"
PCK_URLS = {
    "pck00011.tpc": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00011.tpc",
    "itokawa_aizu504.tpc": "https://naif.jpl.nasa.gov/pub/naif/pds/data/hay-a-spice-6-v1.0/haysp_1000/data/pck/itokawa_aizu504.tpc",
    "ryugu_v10.tpc": "https://naif.jpl.nasa.gov/pub/naif/pds/pds4/hyb2/hyb2_spice/spice_kernels/pck/ryugu_v10.tpc",
    "bennu_v17.tpc": "https://naif.jpl.nasa.gov/pub/naif/ORX/kernels/pck/bennu_v17.tpc",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)


def write_json(path: Path, value: object) -> None:
    write_new(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


class Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.names: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            href = dict(attrs).get("href")
            if href and re.fullmatch(r"[A-Za-z0-9_.-]+", href) and href not in (".", ".."):
                self.names.add(href)


class Acquisition:
    def __init__(self, output: Path, timeout: float = 30) -> None:
        self.output, self.timeout = output, timeout
        self.files: list[dict] = []
        self.failures: list[dict] = []

    def read_url(self, url: str) -> tuple[bytes, dict]:
        request = urllib.request.Request(url, headers={"User-Agent": "DeLPHI-spacecraft-Job1/1.0"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = response.read(20_000_001)
            if len(payload) > 20_000_000:
                raise ValueError(f"input exceeds text/document size limit: {url}")
            info = {"url": url, "final_url": response.geturl(),
                    "http_date": response.headers.get("Date"),
                    "content_type": response.headers.get("Content-Type")}
        return payload, info

    def save(self, relative: str, payload: bytes, info: dict) -> bytes:
        path = self.output / "raw" / relative
        write_new(path, payload)
        self.files.append({"path": relative, "bytes": len(payload),
                           "sha256": hashlib.sha256(payload).hexdigest(), **info})
        return payload

    def get(self, relative: str, urls: list[str], *, text: bool = True) -> bytes:
        attempts = []
        for url in urls:
            try:
                payload, info = self.read_url(url)
                if text:
                    decoded = payload.decode("utf-8-sig")
                    if "\x00" in decoded or re.search(r"<(?:!doctype|html)\b", decoded[:1000], re.I):
                        raise ValueError("received binary data or an HTML error/challenge instead of text")
                return self.save(relative, payload, {**info, "prior_attempts": attempts})
            except (OSError, ValueError, urllib.error.URLError) as exc:
                attempts.append({"url": url, "error": str(exc)})
        raise ValueError(json.dumps({"path": relative, "attempts": attempts}))

    def directory(self, relative: str, urls: list[str], extensions: tuple[str, ...]) -> None:
        attempts = []
        for url in urls:
            try:
                payload, info = self.read_url(url)
                links = Links()
                links.feed(payload.decode("utf-8-sig"))
                names = sorted(n for n in links.names if n.lower().endswith(extensions))
                if not names:
                    raise ValueError("directory listing contains no requested files")
                self.save(relative + "/directory.html", payload, {**info, "prior_attempts": attempts})
                for name in names:
                    self.get(relative + "/" + name, [urllib.parse.urljoin(url, name)], text=not name.endswith(".pdf"))
                return
            except (OSError, ValueError, urllib.error.URLError) as exc:
                attempts.append({"url": url, "error": str(exc)})
        raise ValueError(json.dumps({"directory": relative, "attempts": attempts}))

    def alcdef(self, archive: Path) -> None:
        archive_hash = sha256(archive)
        with zipfile.ZipFile(archive) as source:
            for name in ("ALCDEF_25143_Itokawa.txt", "ALCDEF_162173_Ryugu.txt"):
                payload = source.read(name)
                payload.decode("utf-8-sig")
                self.save("alcdef/" + name, payload, {"archive": str(archive),
                          "archive_sha256": archive_hash, "member": name})

    def vizier(self, catalog: str, source: str, names: tuple[str, ...]) -> None:
        bases = [f"{scheme}://{host}/ftp/{catalog}/" for host in
                 ("cdsarc.cds.unistra.fr", "cdsarc.u-strasbg.fr") for scheme in ("https", "http")]
        bases += [f"ftp://{host}/pub/cats/{catalog}/" for host in
                  ("cdsarc.cds.unistra.fr", "cdsarc.u-strasbg.fr")]
        # ReadMe must be available before parsing or interpreting any tables.
        try:
            for name in names:
                self.get(source + "/" + name, [base + name for base in bases])
            if source == "kim2013":
                self.directory(source + "/lc", [base + "lc/" for base in bases[:4]], (".dat", ".txt"))
        except ValueError as exc:
            # A VOTable query cannot substitute for the required lc/* files.
            try:
                from astroquery.vizier import Vizier
            except ImportError:
                fallback = "astroquery.vizier is not installed"
            else:
                try:
                    tables = Vizier(row_limit=-1).get_catalogs(catalog)
                    fallback = f"astroquery returned {len(tables)} tables; raw ReadMe/lc files still required"
                except Exception as query_error:
                    fallback = f"astroquery failed: {query_error}"
            raise ValueError(f"STOP {catalog}: {exc}; {fallback}; obtain original archive manually") from exc

    def bennu(self) -> None:
        self.get("hergenrother2020/bundle.xml", [PDS + "bundle_orex.gbo.ast-bennu.lightcurves-images.xml"])
        self.directory("hergenrother2020/data/lightcurve", [PDS + "data/lightcurve/"], (".tab", ".xml"))
        self.directory("hergenrother2020/document", [PDS + "document/"], (".txt", ".xml", ".csv", ".pdf"))


def verify(output: Path) -> int:
    receipt = json.loads((output / "raw/receipts.json").read_text())
    for row in receipt["files"]:
        path = (output / "raw" / row["path"]).resolve()
        if not path.is_relative_to((output / "raw").resolve()):
            raise ValueError("receipt path leaves raw directory")
        if path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            raise ValueError(f"receipt mismatch: {path}")
        if "archive" in row and sha256(Path(row["archive"])) != row["archive_sha256"]:
            raise ValueError(f"source archive changed: {row['archive']}")
    print(json.dumps({"receipt_hashes": "PASS", "files": len(receipt["files"]),
                      "acquisition_failures": receipt["failures"]}))
    return int(bool(receipt["failures"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--source-zip", type=Path, default=ALCDEF_ZIP)
    parser.add_argument("--only", choices=("all", "alcdef", "vizier", "pck", "bennu"), default="all")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        return verify(args.output)
    if (args.output / "raw").exists():
        parser.error("raw directory already exists; refusing to alter existing acquisition")
    fetch = Acquisition(args.output)
    jobs = {"alcdef": lambda: fetch.alcdef(args.source_zip),
            "lowry2014": lambda: fetch.vizier("J/A+A/562/A48", "lowry2014", ("ReadMe", "table1.dat", "table2.dat")),
            "kim2013": lambda: fetch.vizier("J/A+A/550/L11", "kim2013", ("ReadMe", "table.dat")),
            "bennu": fetch.bennu}
    for name, url in PCK_URLS.items():
        jobs[name] = lambda name=name, url=url: fetch.get("pck/" + name, [url])
    for name, job in jobs.items():
        category = "vizier" if name in ("lowry2014", "kim2013") else "pck" if name.endswith(".tpc") else name
        if args.only not in ("all", category):
            continue
        try:
            job()
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            fetch.failures.append({"source": name, "error": str(exc)})
    write_json(args.output / "raw/receipts.json", {
        "schema": "delphi.spacecraft-acquisition.v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "files": fetch.files, "failures": fetch.failures,
        "kernel_policy": "pinned pck00011, Hayabusa Aizu504, Hayabusa2 SPC/v20200323 as ryugu_v10, Bennu v17",
    })
    print(json.dumps({"files": len(fetch.files), "failures": fetch.failures}))
    return int(bool(fetch.failures))


if __name__ == "__main__":
    raise SystemExit(main())
