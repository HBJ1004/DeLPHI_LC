#!/usr/bin/env python3
"""Addendum 1 spacecraft runs; all writes stay under --output-dir.

Order: inspect, convert, lock, geometry (network), sbdb (network), prepare,
validate (reuse the primary validation), search (GPU), predict (GPU), score.
Default output: spacecraft/posthoc/. Every stage refuses existing outputs.
Use --runs on convert to select a subset; later stages use that frozen subset.

C.1 must contain exactly 120 rows under the protocol's weather-gap acceptance;
C.2 must match its caption (96 rows). HST times use MAST mid-exposure times for
all 73 image ids, retaining original CSV JDs for audit. There are no overrides.

Run with PYTHONPATH=<worktree> <python> -I <this-file> <command>. Since -I
ignores PYTHONPATH, only this script's trusted checkout is added explicitly.
There is deliberately no synthetic geometry or inference mode in this CLI.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from urllib.parse import urlencode

sys.dont_write_bytecode = True
WORKTREE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORKTREE))

from repro import prepare_k3_alcdef_transfer as adapter  # noqa: E402 (trusted path for -I)
from repro.own_period.spacecraft import convert, run  # noqa: E402

RUNS = {
    "ryugu-combined-grond": ("ryugu", 162173, "combined-grond"),
    "bennu-mapcam": ("bennu", 101955, "mapcam"),
    "bennu-mapcam-2005": ("bennu", 101955, "mapcam-2005"),
    "bennu-ground-all": ("bennu", 101955, "ground-all"),
    "bennu-all": ("bennu", 101955, "all"),
}
PDF = "mueller2017/arXiv-1611.05625.pdf"
TAB = "orex-mapcam/G_000PS_AST_ALTCRV_v002.tab"
XML = "orex-mapcam/G_000PS_AST_ALTCRV_v002.xml"
HST = "nolan2019/grl58586-sup-0002-2018gl080658-ds01.csv"
MAST = "nolan2019/mast-times.csv"
POSTHOC = "POST HOC: Addendum 1; does not replace the primary spacecraft results"
SCHEMA = "delphi.spacecraft-posthoc-conversion.v1"


def receipt_inputs(root: Path, *, include_hst: bool = False) -> tuple[dict[str, Path], list[dict]]:
    receipts = [(root / "receipts.json", (PDF, TAB, XML))]
    if include_hst:
        # This receipt lives beside the CSV, but its paths are relative to
        # raw-posthoc/, just like the original acquisition receipt.
        receipts.append((root / "nolan2019/receipt.json", (HST, MAST)))
    paths, bindings = {}, []
    for receipt_path, required in receipts:
        receipt = run.read(receipt_path)
        if receipt["schema"] != "delphi.spacecraft-acquisition.v1":
            raise ValueError(f"unexpected acquisition receipt schema: {receipt_path}")
        records = receipt["files"]
        if len({r["path"] for r in records}) != len(records):
            raise ValueError("duplicate acquisition receipt paths")
        by_path = {r["path"]: r for r in records}
        bindings.append(run.binding(receipt_path))
        for relative in required:
            row = by_path[relative]
            path = run.checked(row, root)
            if path.stat().st_size != row["bytes"]:
                raise ValueError(f"acquisition byte count differs: {relative}")
            paths[relative] = path
            bindings.append(run.binding(path))
    return paths, bindings


def extract_grond(pdf: Path) -> tuple[list, dict]:
    """Parse column groups, including dash placeholders, before selecting r'."""
    text = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], check=True,
                          capture_output=True, text=True, encoding="utf-8").stdout
    tables, counts = [], {}
    for number, width, zero in ((1, 3, "56087.04942"), (2, 7, "56088.18661")):
        table_id = f"C.{number}"
        matches = list(re.finditer(rf"Table C\.{number}\. GROND", text))
        if len(matches) != 1:
            raise ValueError(f"expected one Table {table_id} heading")
        # Both tables occupy one page in this receipt-bound PDF.
        page = text[matches[0].start():].split("\f", 1)[0]
        caption, body = page.split("TDT", 1)
        found_zero = re.search(r"MJD\s+(\d+\.\d+)", caption)
        if found_zero is None or found_zero.group(1) != zero:
            raise ValueError(f"Table {table_id}: unexpected zero MJD")
        identifiers = caption.split("are ", 1)[1].split(", with TDP", 1)[0]
        observations = []
        pattern = r"OB(\d+)_(\d+)(?:\s*\.\.\.\s*OB\1_(\d+))?"
        for match in re.finditer(pattern, identifiers):
            ob, start, end = int(match[1]), int(match[2]), int(match[3] or match[2])
            observations.extend((ob, i) for i in range(start, end + 1))
        remainder = re.sub(pattern, "", identifiers).replace(",", "").strip()
        tdps = set(re.findall(r"TDP(\d+)", caption.split(", with ", 1)[1]))
        if remainder or not observations or len(set(observations)) != len(observations) or tdps != {"1", "2", "3", "4"}:
            raise ValueError(f"Table {table_id}: cannot verify observation/TDP caption")
        rows = []
        for line in body.splitlines():
            tokens = line.split()
            if not tokens or not re.fullmatch(r"\d+\.\d+|—", tokens[0]):
                continue
            if len(tokens) % width or len(tokens) > width * (3 if number == 1 else 2):
                raise ValueError(f"Table {table_id}: malformed numeric row: {line}")
            for offset in range(0, len(tokens), width):
                group = tokens[offset:offset + width]
                if group == ["—"] * width:
                    continue  # C.1's weather-gap marker, not an observation.
                if any(t != "—" and not re.fullmatch(r"\d+\.\d+", t) for t in group):
                    raise ValueError(f"Table {table_id}: unrecognized photometry")
                seconds = Decimal(group[0])
                mag, error = group[1:3] if number == 1 else group[3:5]
                if seconds < 0 or "—" in (mag, error):
                    raise ValueError(f"Table {table_id}: missing r' measurement")
                jd = Decimal("2400000.5") + Decimal(zero) + seconds / Decimal(86400)
                rows.append((seconds, convert.photometry(format(jd, ".12f"), mag, error)))
        rows.sort(key=lambda row: row[0])
        if not rows or rows[0][0] != 0 or len({r[0] for r in rows}) != len(rows):
            raise ValueError(f"Table {table_id}: empty, repeated, or missing zero time")
        points = [r[1] for r in rows]
        if len({convert.ut_night(float(p[0])) for p in points}) != 1:
            raise ValueError(f"Table {table_id}: expected one UT night")
        caption_expected = len(observations) * len(tdps)
        expected = 120 if number == 1 else caption_expected
        counts[table_id] = {"rows": len(rows), "caption_observations": len(observations),
                            "tdps_per_observation": len(tdps), "caption_expected": caption_expected,
                            "expected": expected, "matches_expected": len(rows) == expected,
                            "count_basis": "protocol acceptance: 120 rows; 16 missing in weather gap" if number == 1 else "caption observation identifiers x four TDPs",
                            "matches_caption": len(rows) == caption_expected, "zero_mjd": zero,
                            "caption": " ".join(caption.split())}
        tables.append((points, {"filename": PDF, "table": table_id, "zero_mjd": zero,
                                "filter": "r'", "observer_center": "500@399"}))
    counts["pdftotext_layout_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return tables, counts


def require_grond_counts(counts: dict) -> None:
    failures = [f"{key}: extracted {counts[key]['rows']}, expected {counts[key]['expected']} "
                f"({counts[key]['count_basis']})"
                for key in ("C.1", "C.2") if counts[key]["rows"] != counts[key]["expected"]]
    if failures:
        raise ValueError("GROND row-count mismatch; refusing Ryugu conversion: " + "; ".join(failures))


def extract_mapcam(tab: Path, label: Path) -> tuple[list, dict]:
    ns = {"p": "http://pds.nasa.gov/pds4/pds/v1"}
    xml = ET.fromstring(label.read_text(encoding="utf-8-sig"))
    area = xml.find("p:File_Area_Observational", ns)
    if area is None or area.findtext("p:File/p:file_name", namespaces=ns) != Path(TAB).name:
        raise ValueError("unexpected MapCam label filename")
    table = area.find("p:Table_Character", ns)
    if table is None:
        raise ValueError("MapCam label lacks a character table")
    fields, descriptions = {}, {}
    for field in table.findall("p:Record_Character/p:Field_Character", ns):
        name = field.findtext("p:name", namespaces=ns)
        start = int(field.findtext("p:field_location", namespaces=ns)) - 1
        length = int(field.findtext("p:field_length", namespaces=ns))
        if name in fields or start < 0 or length <= 0:
            raise ValueError("invalid MapCam label column")
        fields[name] = slice(start, start + length)
        descriptions[name] = field.findtext("p:description", namespaces=ns)
    if not {"Year", "Month", "Day", "Filter", "App_Mag"} <= fields.keys() or descriptions["App_Mag"].strip().lower() != "apparent magnitude":
        raise ValueError("MapCam label does not identify apparent magnitude/time/filter")
    lines = tab.read_bytes().splitlines()
    expected = int(table.findtext("p:records", namespaces=ns))
    if len(lines) != expected:
        raise ValueError(f"MapCam table has {len(lines)} rows, label requires {expected}")
    groups, originals = {}, {}
    for line in lines:
        if len(line) < max(f.stop for f in fields.values()):
            raise ValueError("truncated MapCam record")
        row = {name: line[part].decode("ascii").strip() for name, part in fields.items()}
        if row["Filter"] != "p":
            continue
        year, month, day = int(row["Year"]), int(row["Month"]), Decimal(row["Day"])
        civil = datetime(year, month, int(day), tzinfo=timezone.utc)
        midnight_jd = Decimal(civil.toordinal()) + Decimal("1721424.5")
        jd = midnight_jd + day - int(day)
        point = convert.photometry(format(jd, ".12f"), row["App_Mag"])
        night = civil.date().isoformat()
        groups.setdefault(night, []).append(point)
        originals.setdefault(night, []).append(row)
    count = sum(map(len, groups.values()))
    if count != 1120:
        raise ValueError(f"MapCam pan row-count mismatch: extracted {count}, expected 1120")
    if (min(groups), max(groups)) != ("2018-10-11", "2018-11-16"):
        raise ValueError("MapCam pan date range differs from Addendum 1")
    result = [(sorted(rows, key=lambda r: Decimal(r[0])),
               {"filename": TAB, "ut_night": night, "filter": "p",
                "observer_center": "500@-64", "original_rows": originals[night]})
              for night, rows in sorted(groups.items())]
    return result, {"all_filters_rows": len(lines), "pan_rows": count, "nights": len(groups),
                    "date_min": min(groups), "date_max": max(groups),
                    "magnitude_column": "App_Mag", "magnitude_error": "absent; empty string"}


def extract_hst(path: Path, mast_path: Path) -> tuple[list, dict]:
    """Use matched MAST mid-exposure times; retain original CSV JDs per image."""
    times = {}
    with mast_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, skipinitialspace=True)
        if reader.fieldnames != ["obs_id", "t_min", "t_max"]:
            raise ValueError("MAST times must have columns obs_id,t_min,t_max")
        for index, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"MAST row {index}: expected three columns")
            image, start, stop = (row[key].strip() for key in reader.fieldnames)
            if not image or image in times:
                raise ValueError(f"MAST row {index}: missing or duplicate obs_id")
            try:
                t_min, t_max = Decimal(start), Decimal(stop)
            except InvalidOperation as error:
                raise ValueError(f"MAST row {index}: nonnumeric exposure bounds") from error
            if not t_min.is_finite() or not t_max.is_finite() or t_min <= 0 or t_max < t_min:
                raise ValueError(f"MAST row {index}: invalid exposure bounds")
            jd = (t_min + t_max) / 2 + Decimal("2400000.5")
            times[image] = {"jd": format(jd, "f"), "t_min": start, "t_max": stop}
    rows = list(csv.reader(path.read_text(encoding="utf-8-sig").splitlines(), skipinitialspace=True))
    if len(rows) != 73:
        raise ValueError(f"Nolan HST row-count mismatch: extracted {len(rows)}, expected 73")
    originals, disagreements, images = {}, [], set()
    adopted_date_disagreements = 0
    for index, fields in enumerate(rows, 1):
        if len(fields) != 5:
            raise ValueError(f"Nolan HST row {index}: expected five columns without a header")
        jd, night, magnitude, error, image = (f.strip() for f in fields)
        if datetime.strptime(night, "%Y-%m-%d").date().isoformat() != night:
            raise ValueError(f"Nolan HST row {index}: invalid UT date")
        convert.photometry(jd, magnitude, error)  # Original JD is audit data, not the adopted time.
        if not error or not image or image in images:
            raise ValueError(f"Nolan HST row {index}: missing error/image id or duplicate image")
        if image not in times:
            raise ValueError(f"Nolan HST row {index}: image id {image!r} missing from MAST times")
        images.add(image)
        timing = times[image]
        adopted = convert.photometry(timing["jd"], magnitude, error)
        originals.setdefault(night, []).append({"JD": adopted[0], "original_csv_jd": jd,
            "UT_date": night, "magnitude": magnitude, "error": error, "image_id": image,
            "mast_t_min": timing["t_min"], "mast_t_max": timing["t_max"]})
        adopted_date_disagreements += convert.ut_night(float(adopted[0])) != night
        jd_date = convert.ut_night(float(jd))
        if jd_date != night:
            disagreements.append({"row": index, "image_id": image, "jd": jd,
                                  "supplied_ut_date": night, "jd_calendar_date": jd_date})
    if set(originals) != {"2012-09-17", "2012-09-18", "2012-12-10", "2012-12-11"}:
        raise ValueError("Nolan HST UT dates differ from Addendum 1")
    result = []
    for night, records in sorted(originals.items()):
        records.sort(key=lambda r: Decimal(r["JD"]))
        points = [(r["JD"], r["magnitude"], r["error"]) for r in records]
        result.append((points, {"filename": HST, "timing_filename": MAST, "ut_night": night,
            "filter": "F350LP", "observer_center": "500@399", "original_rows": records}))
    return result, {"rows": len(rows), "expected": 73, "nights": len(originals),
        "mast_rows": len(times), "matched_image_ids": len(images),
        "points_per_supplied_ut_date": {night: len(records) for night, records in sorted(originals.items())},
        "time_rule": "JD = (MAST t_min + t_max) / 2 + 2400000.5, for every image id",
        "session_definition": "one session per supplied CSV UT date; MAST mid-exposure JD",
        "original_csv_jd_ut_date_disagreement_count": len(disagreements),
        "original_csv_jd_ut_date_disagreements": disagreements,
        "adopted_jd_ut_date_disagreement_count": adopted_date_disagreements}


def cmd_inspect(args) -> dict:
    paths, _ = receipt_inputs(args.raw_posthoc_dir, include_hst=True)
    _, grond = extract_grond(paths[PDF])
    _, mapcam = extract_mapcam(paths[TAB], paths[XML])
    _, hst = extract_hst(paths[HST], paths[MAST])
    return {"stage": "inspect", "scope": POSTHOC, "grond": grond, "mapcam": mapcam,
            "hst": hst,
            "status": "PASS" if all(grond[t]["matches_expected"] for t in ("C.1", "C.2")) else "FAIL"}


def primary_source(spacecraft: Path, run_id: str, bindings: list[dict]) -> dict:
    """Verify a reused raw extraction against its original frozen binding.

    The primary protocol gained Addendum 1 after its lock: do not pretend that
    run.locked(primary) would still validate that protocol. Bind the original
    lock itself and check only the raw source entry being reused here.
    """
    lock_path = spacecraft / "lock/cohort-lock.json"
    matches = [r for r in run.read(lock_path)["inputs"] if r["run_id"] == run_id]
    if len(matches) != 1:
        raise ValueError(f"primary raw input missing or repeated: {run_id}")
    path = run.checked(matches[0]["source"])
    if path.resolve() != (spacecraft / f"raw-extraction-{run_id}.json").resolve():
        raise ValueError("primary input is outside the requested spacecraft directory")
    bindings.extend((run.binding(lock_path), run.binding(path)))
    return run.read(path)


def data_markdown(products: list[dict], counts: dict) -> str:
    lines = ["# Spacecraft photometry: POST HOC (Addendum 1)", "", POSTHOC + ".", "",
             "| Run | Nights | Sessions | Points | Numeric JD calendar range | Time system | Magnitude type | Observer |",
             "| --- | ---: | ---: | ---: | --- | --- | --- | --- |"]
    for product in products:
        s = product["source_summary"]
        lines.append(f"| {product['run_id']} | {s['nights']} | {s['sessions']} | {s['points']} | "
                     f"{s['calendar_min']} to {s['calendar_max']} | {s['time_system']} | "
                     f"{s['magnitude_type']} | {s['observer']} |")
    hst_notes = []
    if "hst" in counts:
        hst = counts["hst"]
        date_counts = ", ".join(f"{night} ({count} points)"
                                for night, count in hst["points_per_supplied_ut_date"].items())
        hst_notes = [
            f"HST: Nolan et al. 2019 Data Set S1, {hst['rows']} WFC3 F350LP magnitudes and errors. "
            "For every image, adopted JD = (MAST t_min + t_max) / 2 + 2400000.5 from mast-times.csv, matched on stripped image id. "
            "Both the photometry CSV and MAST timing CSV are hash-checked against nolan2019/receipt.json. "
            "Original CSV JD strings, image ids and MAST bounds are retained alongside adopted JDs; magnitude/error strings are unchanged.",
            f"One HST session per CSV UT date: {date_counts}. "
            "The protocol records that the original December JDs are 0.5 d early and two September times also differ from MAST. "
            f"All {hst['matched_image_ids']} image ids must match; geometry, preparation and search use MAST times for every row. "
            f"Adopted JD/date disagreements: {hst['adopted_jd_ut_date_disagreement_count']}. "
            "No additional time-scale conversion or light-time correction is applied. The calendar range renders adopted numeric JD.",
            "bennu-ground-all = 2005 ground + HST 2012; bennu-all = MapCam + 2005 ground + HST 2012."]
    lines += ["", "GROND: the caption calls the offset column TDT mid time [s], but explicitly anchors it in UT in the observer's frame. "
              "Use that UT anchor literally; no TT/UTC conversion. Each table is one session/night.",
              "GROND C.1 requires exactly 120 rows under the protocol's data-check acceptance: the 16 points missing "
              "from the caption's implied 136 points lie in the approximately 2 h weather gap (7,033 s). "
              "C.2 requires 96 rows. Any other count fails; no missing points are filled.",
              "MapCam: Year/Month/decimal Day are UT mid-exposure. Use App_Mag (field 7); H_Mag is distance normalized and is excluded. "
              "No magnitude uncertainties are supplied in the MapCam table.",
              "One MapCam session per UT date. Ground sessions retain their original definitions and time-scale caveats.",
              *hst_notes,
              "Ryugu combines the original ALCDEF and Kim components with GROND, reapplying convert.combine's duplicate-night rule. "
              "Only ALCDEF sessions overlapping catalogue dates are dropped; catalogue sessions sharing a date remain.",
              "Preparation uses session-median normalization and session-midpoint Horizons geometry; no added light-time, distance or phase correction.",
              "Geometry's legacy asteroid_to_earth field carries the actual observer vector: MapCam = OSIRIS-REx (500@-64); "
              "Ryugu/2005 Bennu/HST = Earth (500@399), including geocentric approximation for ESO La Silla and HST.",
              "", "Extraction checks:", "", "```json", json.dumps(counts, indent=2, sort_keys=True), "```", ""]
    for product in products:
        lines += [f"## {product['run_id']}", "",
                  "Dropped ALCDEF sessions: " + json.dumps(product["source_summary"]["sessions_dropped_in_combined"], sort_keys=True), ""]
    return "\n".join(lines)


def cmd_convert(args) -> dict:
    out = args.output_dir
    run.available(out, "converted")
    run.available(out, "DATA-posthoc.md")
    selected = [key for key in RUNS if key in args.runs]
    include_hst = any(key in selected for key in ("bennu-ground-all", "bennu-all"))
    paths, bindings = receipt_inputs(args.raw_posthoc_dir, include_hst=include_hst)
    counts, sources, components = {}, {}, {}
    inputs = {key: run.binding(path) for key, path in paths.items()}
    if "ryugu-combined-grond" in selected:
        groups, counts["grond"] = extract_grond(paths[PDF])
        require_grond_counts(counts["grond"])  # Before any output or relaxed product.
        grond = convert.catalogue(162173, "ryugu", "grond2017", groups, [inputs[PDF]],
            time_system="UT anchor MJD + TDT mid-time seconds / 86400; JD = MJD + 2400000.5",
            light_time_correction="no correction added; observer-frame UT as captioned",
            magnitude_type="calibrated GROND r' (PDF extraction renders r0)",
            corrections_already_applied="reduced/calibrated as published; no extra correction",
            session_definition="one session per table/night (C.1, C.2)")
        grond["acquisition_receipt"] = run.binding(args.raw_posthoc_dir / "receipts.json")
        alcdef = primary_source(args.spacecraft_dir, "ryugu-alcdef", bindings)
        kim = primary_source(args.spacecraft_dir, "ryugu-kim2013", bindings)
        combined = primary_source(args.spacecraft_dir, "ryugu-combined", bindings)
        if convert.combine(copy.deepcopy([alcdef, kim]))["objects"] != combined["objects"]:
            raise ValueError("original Ryugu combined object cannot be reproduced")
        product = convert.combine([alcdef, kim, grond])
        product["source_summary"]["observer"] = "Earth geocentric (500@399); GROND at ESO La Silla"
        sources["ryugu-combined-grond"] = product
    if any(key in selected for key in ("bennu-mapcam", "bennu-mapcam-2005", "bennu-all")):
        groups, counts["mapcam"] = extract_mapcam(paths[TAB], paths[XML])
        mapcam = convert.catalogue(101955, "bennu", "mapcam", groups, [inputs[TAB], inputs[XML]],
            time_system="UT Year/Month/decimal Day at mid-exposure; converted to JD without scale conversion",
            light_time_correction="unspecified in label; no correction added",
            magnitude_type="apparent MapCam pan; App_Mag (field 7)",
            corrections_already_applied="published IRAF aperture photometry and 18 Scorpii absolute calibration",
            session_definition="one session per UT day")
        mapcam["acquisition_receipt"] = run.binding(args.raw_posthoc_dir / "receipts.json")
        mapcam["source_summary"]["observer"] = "OSIRIS-REx (NAIF -64; 500@-64)"
        sources["bennu-mapcam"] = mapcam
        components["mapcam"] = mapcam
    if any(key in selected for key in ("bennu-mapcam-2005", "bennu-ground-all", "bennu-all")):
        components["ground"] = primary_source(args.spacecraft_dir, "bennu-hergenrother2020", bindings)
    if include_hst:
        groups, counts["hst"] = extract_hst(paths[HST], paths[MAST])
        nights = [fields["ut_night"] for _, fields in groups]
        hst = convert.catalogue(101955, "bennu", "nolan2019", groups, [inputs[HST], inputs[MAST]],
            time_system="MAST mid-exposure JD = (t_min+t_max)/2 + 2400000.5 for every image; sessions use CSV UT dates",
            light_time_correction="unspecified in supplied CSV; no correction added",
            magnitude_type="published HST WFC3 F350LP magnitude",
            corrections_already_applied="unspecified in supplied CSV; no extra correction",
            corrections_applied_by_adapter="CSV JD replaced by MAST mid-exposure JD; photometry unchanged",
            session_definition="one session per supplied CSV UT date; MAST mid-exposure JD",
            night_definition="supplied CSV UT date",
            ut_nights=nights, nights=len(nights),
            adopted_jd_ut_date_disagreement_count=counts["hst"]["adopted_jd_ut_date_disagreement_count"])
        for session, (_, fields) in zip(hst["objects"][0]["sessions"], groups, strict=True):
            session["SuppliedUTDate"] = fields["ut_night"]
        records = [row for _, fields in groups for row in fields["original_rows"]]
        for point, original in zip(hst["objects"][0]["points"], records, strict=True):
            point.update(OriginalCSVJD=original["original_csv_jd"], ImageID=original["image_id"])
        hst["acquisition_receipt"] = run.binding(args.raw_posthoc_dir / "nolan2019/receipt.json")
        hst["source_summary"]["observer"] = "HST approximated by Earth geocentric (500@399)"
        components["hst"] = hst
    for key, constituents, observer in (
            ("bennu-mapcam-2005", ("mapcam", "ground"), "MapCam: OSIRIS-REx (500@-64); 2005: Earth geocentric (500@399)"),
            ("bennu-ground-all", ("ground", "hst"), "2005/HST: Earth geocentric (500@399)"),
            ("bennu-all", ("mapcam", "ground", "hst"), "MapCam: OSIRIS-REx (500@-64); 2005/HST: Earth geocentric (500@399)")):
        if key in selected:
            product = convert.combine(copy.deepcopy([components[name] for name in constituents]))
            product["source_summary"]["observer"] = observer
            sources[key] = product
    products = []
    for key in selected:
        product = sources[key]
        product.update(run_id=key, posthoc=True, purpose=POSTHOC)
        products.append(product)
    markdown = data_markdown(products, counts)
    root = run.stage(out, "converted")
    entries = []
    for product in products:
        path = root / f"{product['run_id']}.json"
        run.write(path, product)
        entries.append({"run_id": product["run_id"], **run.relative_binding(path, root)})
    with (out / "DATA-posthoc.md").open("x", encoding="utf-8") as stream:
        stream.write(markdown)
    dependencies = {r["path"]: r for r in bindings}
    run.write(root / "manifest.json", {"schema": SCHEMA, "scope": POSTHOC,
        "script": run.binding(Path(__file__)), "sources": list(dependencies.values()),
        "data_markdown": run.binding(out / "DATA-posthoc.md"), "counts": counts, "objects": entries})
    return {"stage": "convert", "runs": selected, "counts": counts,
            "summaries": [p["source_summary"] for p in products]}


def cmd_lock(args) -> dict:
    out = args.output_dir
    run.available(out, "lock")
    manifest_path = out / "converted/manifest.json"
    manifest = run.read(manifest_path)
    if manifest["schema"] != SCHEMA or run.checked(manifest["script"]) != Path(__file__).resolve():
        raise ValueError("posthoc conversion schema/script differs")
    rows = []
    for entry in manifest["objects"]:
        path = run.checked(entry, out / "converted")
        name, number, source = RUNS[entry["run_id"]]
        raw = adapter._read(path, convert.SCHEMA)
        objects = adapter._numbers(raw)
        if len(objects) != 1 or objects[0][:2] != (f"mpc:{number}", number) or raw["run_id"] != entry["run_id"]:
            raise ValueError("posthoc raw identity differs")
        adapter._midpoints(objects[0][2], objects[0][3])
        rows.append({"run_id": entry["run_id"], "kind": "external", "name": name,
                     "source_name": source, "primary": False, "object_id": f"mpc:{number}",
                     "source": run.binding(path)})
    if not rows or len({r["run_id"] for r in rows}) != len(rows):
        raise ValueError("empty or repeated posthoc inputs")
    files = {"conversion_manifest": run.binding(manifest_path),
             "data_markdown": manifest["data_markdown"],
             "protocol": run.binding(args.spacecraft_dir / "PROTOCOL.md"),
             "reference_poles": run.binding(args.spacecraft_dir / "reference-poles.json"),
             "script": run.binding(Path(__file__)), "model_manifest": run.binding(run.MODEL_MANIFEST),
             "model_table": run.binding(args.snapshot_dir / "tables/asteroid_models.csv"),
             "splits": run.binding(convert.SPLIT), "lowq_index": run.binding(convert.LOWQ),
             "asteroids_table": run.binding(convert.SNAPSHOT / "tables/asteroids.csv")}
    for i, item in enumerate(manifest["sources"]):
        run.checked(item)
        files[f"source_{i}"] = item
    for i, relative in enumerate(("repro/own_period/spacecraft/run.py", "repro/own_period/spacecraft/convert.py",
            "repro/prepare_k3_alcdef_transfer.py", "repro/fetch_k3_alcdef_periods.py",
            "repro/own_period/search_external.py", "repro/own_period/predict_external.py",
            "repro/own_period/score.py", "repro/period_search/search.py",
            "lc_pipeline/k3/ztf_external.py", "lc_pipeline/k3/ztf_prediction.py", "lc_pipeline/v2/data.py")):
        files[f"implementation_{i}"] = run.binding(WORKTREE / relative)
    for item in files.values():
        run.checked(item)
    check = convert.training_check()
    root = run.stage(out, "lock")
    path = root / "cohort-lock.json"
    run.write(path, {"schema": "delphi.k3-alcdef-gaia-cohort-lock.v1", "purpose": POSTHOC,
        "objects": [{"object_id": f"mpc:{n}", "mpc_number": n, "name": name}
                    for name, n in dict((r["name"], RUNS[r["run_id"]][1]) for r in rows).items()],
        "inputs": rows, "files": files, "training_check": check,
        "catalogue_period_rule": "SBDB rot_per; same receipt for all Bennu runs", "rescore_only": {}})
    bound = []
    for row in rows:
        value = run.read(Path(row["source"]["path"]))
        value["cohort_lock"] = run.binding(path)
        target = root / "raw" / f"{row['run_id']}.json"
        run.write(target, value)
        bound.append({k: v for k, v in row.items() if k != "source"} | run.relative_binding(target, root))
    run.write(root / "manifest.json", {"schema": "delphi.spacecraft-lock.v1",
        "cohort_lock": run.relative_binding(path, root), "inputs": bound})
    return {"stage": "lock", "runs": [r["run_id"] for r in rows], "sha256": run.sha(path)}


def observer_center(session: dict) -> str:
    return "500@-64" if session.get("Source") == "mapcam" else "500@399"


def fetch_geometry(raw_path: Path, output: Path, *, batch_size: int,
                   timeout_seconds: float, interval_seconds: float) -> dict:
    """Same request/parser/sign convention as adapter.fetch_geometry.

    Only observer-center selection is extended. The adapter's query builder
    allows Sun/Earth only; build its exact Earth query and replace CENTER for
    -64 here, without changing the shared implementation or any other setting.
    """
    raw = adapter._read(raw_path, convert.SCHEMA)
    if not 0 < batch_size <= 20 or timeout_seconds <= 0 or interval_seconds < 0:
        raise ValueError("invalid geometry-fetch settings")
    records = []
    for identity, number, sessions, points in adapter._numbers(raw):
        midpoints = adapter._midpoints(sessions, points)
        centers = {s["ID"]: observer_center(s) for s in sessions}
        vectors, responses, identities = {}, [], []
        for start in range(0, len(midpoints), batch_size):
            indices = list(range(start, min(start + batch_size, len(midpoints))))
            requests = [("sun", "500@10", indices)]
            for center in ("500@399", "500@-64"):
                chosen = [i for i in indices if centers[midpoints[i][0]] == center]
                if chosen:
                    requests.append(("observer", center, chosen))
            for role, center, chosen in requests:
                if responses and interval_seconds:
                    time.sleep(interval_seconds)
                batch = [midpoints[i][1] for i in chosen]
                parameters = adapter.horizons_query_parameters(number, "500@399" if center == "500@-64" else center, batch)
                parameters["CENTER"] = f"'{center}'"
                url = f"{adapter.HORIZONS_API_URL}?{urlencode(parameters)}"
                body, final_url, http_date, content_type = adapter._default_http_fetcher(url, timeout_seconds)
                parsed = adapter._parse_horizons_vectors(body, batch)
                identities.append(adapter._parse_horizons_target_identity(body))
                center_match = re.search(r"^Center body name:\s*[^\n]*?\((-?\d+)\)", json.loads(body)["result"], re.MULTILINE)
                if center_match is None or int(center_match[1]) != int(center.split("@")[1]):
                    raise ValueError(f"Horizons observer/center differs: expected {center}")
                for i, vector in zip(chosen, parsed, strict=True):
                    vectors[role, i] = [-float(v) for v in vector]
                responses.append({"batch": start // batch_size, "role": role, "center": center,
                    "metadata_ids": [midpoints[i][0] for i in chosen], "request_url": url,
                    "final_url": final_url, "http_date": http_date, "content_type": content_type,
                    "response_sha256": hashlib.sha256(body).hexdigest(), "response_bytes": len(body),
                    "raw_response_utf8": body.decode("utf-8")})
        first = identities[0]
        if any(item != first for item in identities) or first.get("asteroid_number") != number:
            raise ValueError(f"Horizons identity mismatch for {identity}")
        records.append({"object_id": identity, "mpc_number": number, "resolved_target_identity": first,
            "session_midpoints": [{"metadata_id": identifier, "jd": jd, "observer_center": centers[identifier],
                "asteroid_to_sun_ecliptic_j2000_au": vectors["sun", i],
                "asteroid_to_earth_ecliptic_j2000_au": vectors["observer", i]}
                for i, (identifier, jd) in enumerate(midpoints)], "raw_responses": responses})
    payload = {"schema": "delphi.k3-alcdef-session-geometry.v1", "purpose": POSTHOC,
        "observer_field_note": "legacy asteroid_to_earth field stores actual observer; see observer_center per session",
        "raw_extraction_path": str(raw_path), "raw_extraction_sha256": run.sha(raw_path),
        "endpoint": adapter.HORIZONS_API_URL, "objects": records}
    adapter._write_new(output, payload)
    return payload


def cmd_geometry(args) -> dict:
    out = args.output_dir
    run.available(out, "geometry")
    _, rows = run.locked(out)
    root = run.stage(out, "geometry")
    entries = []
    for row in rows:
        path = root / f"{row['run_id']}.json"
        fetch_geometry(run.checked(row, out / "lock"), path, batch_size=args.batch_size,
                       timeout_seconds=args.timeout_seconds, interval_seconds=args.request_interval_seconds)
        entries.append({"run_id": row["run_id"], **run.relative_binding(path, root)})
        print("geometry", row["run_id"], flush=True)
    run.write(root / "manifest.json", {"cohort_lock_sha256": run.sha(out / "lock/cohort-lock.json"), "objects": entries})
    return {"stage": "geometry", "count": len(entries)}


def cmd_validate(args) -> dict:
    """Reuse the already successful, unchanged run.py prediction-path check."""
    out = args.output_dir
    run.available(out, "validation")
    lock, _ = run.locked(out)
    report = run.validation(args.spacecraft_dir)
    if report["model_manifest_sha256"] != lock["files"]["model_manifest"]["sha256"]:
        raise ValueError("primary validation uses a different ensemble")
    root = run.stage(out, "validation")
    run.write(root / "result.json", report)
    run.write(root / "reuse.json", {"scope": POSTHOC,
        "source": run.binding(args.spacecraft_dir / "validation/result.json"),
        "cohort_lock_sha256": run.sha(out / "lock/cohort-lock.json")})
    return {"stage": "validate", "status": report["status"], "reused_primary_validation": True}


@contextmanager
def registered_runs(out: Path):
    """Adapt run.py's expected cohort in memory; all numerical paths stay shared."""
    _, rows = run.locked(out)
    previous = run.NEW
    cohort = {}
    for row in rows:
        name, number, source = RUNS[row["run_id"]]
        cohort.setdefault(name, (number, []))[1].append(source)
    run.NEW = cohort
    try:
        yield
    finally:
        run.NEW = previous


def cmd_score(args) -> dict:
    """Use run.errors for the same axial metrics, with post hoc-only reporting."""
    import numpy as np

    from repro.compute_reference_baselines import unit_from_degrees
    from repro.own_period.score import period_class

    out = args.output_dir
    run.available(out, "scores.json")
    run.available(out, "RESULTS.md")
    lock, rows = run.locked(out)
    inputs = run.prepared(out)
    own = run.read(out / "search/periods.json")
    manifest = run.read(out / "predictions/manifest.json")
    for key, path in (("cohort_lock_sha256", out / "lock/cohort-lock.json"),
                      ("search_sha256", out / "search/periods.json"),
                      ("prepared_manifest_sha256", out / "prepared/manifest.json"),
                      ("validation_sha256", out / "validation/result.json")):
        if manifest[key] != run.sha(path):
            raise ValueError(f"prediction dependency differs: {path}")
    if (own["cohort_lock_sha256"] != manifest["cohort_lock_sha256"]
            or own["prepared_manifest_sha256"] != manifest["prepared_manifest_sha256"]
            or set(own["periods_hours"]) != {r["run_id"] for r in rows}):
        raise ValueError("search inputs differ")
    run.validation(out)
    table = run.model_rows(lock)
    predictions = {}
    for mode in run.OOF_FILES:
        path = run.checked(manifest["outputs"][mode], out / "predictions")
        values = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        predictions[mode] = {r["run_id"]: r for r in values}
        if len(values) != len(rows) or set(predictions[mode]) != {r["run_id"] for r in rows}:
            raise ValueError("missing or duplicate posthoc predictions")
        for row in rows:
            key = row["run_id"]
            value = predictions[mode][key]
            expected = own["periods_hours"][key] if mode == "own" else run.read(inputs[key])["known_period_hours"]
            if (value["object_id"] != row["object_id"] or value["input_sha256"] != run.sha(inputs[key])
                    or own["details"][key]["input_sha256"] != run.sha(inputs[key])
                    or value["period_hours"] != expected or (value["status"] == "no_period") != (expected is None)):
                raise ValueError("prediction observations or period differs")
    reference_path = run.checked(lock["files"]["reference_poles"])
    reference = run.read(reference_path)
    if reference["schema"] != "delphi.spacecraft-reference-poles.v1":
        raise ValueError("reference pole schema differs")
    refs = {r["object"].lower(): r for r in reference["objects"]}
    objects = []
    for row in rows:
        ref = refs[row["name"]]
        axis = np.asarray(ref["unit_vector"], float).reshape(1, 3)
        if not np.isfinite(axis).all() or np.linalg.norm(axis) == 0:
            raise ValueError("invalid spacecraft reference axis")
        axis /= np.linalg.norm(axis, axis=1, keepdims=True)
        poles = (np.asarray([unit_from_degrees(float(r["lambda"]), float(r["beta"]))
                            for r in table[int(ref["damit_id"])]]) if ref["damit_id"] is not None else None)
        values = {mode: run.errors(predictions[mode][row["run_id"]], axis, poles) for mode in run.OOF_FILES}
        objects.append({k: row[k] for k in ("run_id", "object_id", "name", "source_name", "primary", "kind")} |
            {"posthoc": True, "spacecraft_reference": ref,
             "period_class": period_class(values["own"]["period_hours"], values["catalogue"]["period_hours"]), **values})
    run.write(out / "scores.json", {"schema": "delphi.spacecraft-posthoc-scores.v1", "scope": POSTHOC,
        "cohort_lock_sha256": run.sha(out / "lock/cohort-lock.json"),
        "reference_poles_sha256": run.sha(reference_path), "predictions_manifest_sha256": run.sha(out / "predictions/manifest.json"),
        "objects": objects, "aggregation": "none; post hoc variants reported per run"})
    lines = ["# Spacecraft poles: POST HOC (Addendum 1)", "", POSTHOC + ".", "",
             "All 25 frozen networks, own period and SBDB period; axial errors in degrees.",
             "Oracle = minimum over three candidates; best = highest network score. No period = 90 degrees.", "",
             "| Run | Period class | Own P (h) | SBDB P (h) | Own oracle | Own best | SBDB oracle | SBDB best |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    def fmt(value):
        return "n/a" if value is None else f"{value:.6f}"
    for row in objects:
        a, b = row["own"], row["catalogue"]
        cells = [row["run_id"], row["period_class"], fmt(a["period_hours"]), fmt(b["period_hours"]),
                 fmt(a["oracle_deg"]), fmt(a["best_scored_deg"]), fmt(b["oracle_deg"]), fmt(b["best_scored_deg"])]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "No across-variant aggregate. Reference provenance and secondary DAMIT errors, where available, are in scores.json.",
              f"Cohort lock SHA-256: `{run.sha(out / 'lock/cohort-lock.json')}`.", ""]
    with (out / "RESULTS.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    return {"stage": "score", "objects": len(objects), "scope": POSTHOC}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    handlers = {"inspect": cmd_inspect, "convert": cmd_convert, "lock": cmd_lock,
                "geometry": cmd_geometry, "sbdb": run.cmd_sbdb, "prepare": run.cmd_prepare,
                "validate": cmd_validate, "search": run.cmd_search, "predict": run.cmd_predict, "score": cmd_score}
    for name, handler in handlers.items():
        p = commands.add_parser(name)
        p.add_argument("--output-dir", type=Path, default=run.OUTPUT / "posthoc")
        p.add_argument("--spacecraft-dir", type=Path, default=run.OUTPUT, help="read-only primary inputs")
        p.set_defaults(handler=handler)
        if name in ("inspect", "convert"):
            p.add_argument("--raw-posthoc-dir", type=Path, help="default: spacecraft-dir/raw-posthoc")
        if name == "convert":
            p.add_argument("--runs", nargs="+", choices=tuple(RUNS), default=list(RUNS))
        if name == "lock":
            p.add_argument("--snapshot-dir", type=Path, default=run.SNAPSHOT)
        if name in ("geometry", "sbdb"):
            p.add_argument("--timeout-seconds", type=float, default=120 if name == "geometry" else 60)
        if name == "geometry":
            p.add_argument("--batch-size", type=int, default=20)
            p.add_argument("--request-interval-seconds", type=float, default=0.2)
        if name == "predict":
            p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
            p.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    args.output_dir = args.output_dir.resolve()
    args.spacecraft_dir = args.spacecraft_dir.resolve()
    if args.output_dir == args.spacecraft_dir or args.spacecraft_dir.is_relative_to(args.output_dir):
        parser.error("output-dir must be separate from the primary spacecraft directory")
    if hasattr(args, "raw_posthoc_dir"):
        args.raw_posthoc_dir = (args.raw_posthoc_dir or args.spacecraft_dir / "raw-posthoc").resolve()
        if (args.output_dir.is_relative_to(args.raw_posthoc_dir)
                or args.raw_posthoc_dir.is_relative_to(args.output_dir)):
            parser.error("output-dir must be separate from raw-posthoc inputs")
    try:
        if args.command in ("prepare", "search", "predict", "score"):
            with registered_runs(args.output_dir):
                result = args.handler(args)
        else:
            result = args.handler(args)
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"posthoc {args.command}: {error}\n")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 1 if result.get("status") == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
