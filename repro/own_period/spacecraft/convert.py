#!/usr/bin/env python3
"""Convert receipt-bound spacecraft photometry, without network access (-I).

``convert`` writes all sources, three combined products and DATA.md exclusively.
Raw inputs always default to the acquired spacecraft/raw directory, independently
of --output-dir. Observation JD, magnitude and error strings are preserved.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from repro.own_period import DataPath

FOLLOWUP = DataPath("../..")
OUTPUT = DataPath('spacecraft')
RAW = OUTPUT / "raw"
ARCHIVE = FOLLOWUP / "data/training-expansion-cache/ALCDEF_ALL-20250428.zip"
PUBLISHED = FOLLOWUP / "data/generalization-20260910/locked-cohort/alcdef-raw-30-20260911.json"
LOCK = FOLLOWUP / "data/generalization-20260910/locks/alcdef-gaia-model-specific-cohort-30-20260911.json"
SPLIT = FOLLOWUP / "inputs/training-run/repro/data/damit-20250610T000301Z/publication-splits-v2.3.json"
SNAPSHOT = FOLLOWUP.parent / "damit-20250610T000301Z"
LOWQ = FOLLOWUP / "data/damit-postsnapshot-check/lowq-input-index.json"
SCHEMA = "delphi.k3-locked-alcdef-raw-extract.v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def parse_alcdef(text: str, number: int) -> tuple[dict, list[dict]]:
    """Preserve observations as strings; map metadata to the published PDS schema.

    PDS missing-value spellings belong to the adapter only. The original
    metadata, including repeated fields and comments, is retained separately.
    """
    sessions, points, originals = [], [], []
    metadata, records, state = {}, [], "start"

    def finish() -> None:
        if not metadata:
            return
        if state != "points" or not records:
            raise ValueError("incomplete or empty ALCDEF session")
        def value(key, default=""):
            return metadata.get(key, [default])[-1]
        if int(value("OBJECTNUMBER")) != number:
            raise ValueError("ALCDEF identity mismatch")
        identifier = value("LCBLOCKID")
        if not identifier.isdigit() or int(identifier) <= 0:
            raise ValueError("missing positive LCBLOCKID")
        timestamp = value("SESSIONDATE") + "T" + value("SESSIONTIME")
        datetime.fromisoformat(timestamp)  # Validate syntax without changing time.
        session = {
            "ID": identifier, "ObjNumber": str(number), "ObjName": value("OBJECTNAME"),
            "Desig": value("MPCDESIG"), "SessionDateTime": timestamp + "Z",
            "Submitter": value("CONTACTNAME"), "Observers": value("OBSERVERS"),
            "Facility": value("FACILITY"), "MPCCode": value("MPCCODE") or "-",
            "Instrument": value("TELESCOPE"), "Detector": value("DETECTOR"),
            "ObsLong": value("OBSLONGITUDE") or "-99.9", "ObsLat": value("OBSLATITUDE") or "-9.99",
            "Exposure": value("EXPOSURE"), "ExpJD": value("EXPJD"),
            "ObjRA": value("OBJECTRA"), "ObjDec": value("OBJECTDEC"), "Epoch": value("EQUINOX"),
            "Phase": value("PHASE"), "PABL": value("PABL"), "PABB": value("PABB"),
            "Filter": value("FILTER"), "MagBand": value("MAGBAND"),
            "CICorr": {"TRUE": "Y", "FALSE": "N"}.get(value("CICORRECTION"), value("CICORRECTION")),
            "CIBand": value("CIBAND"), "CIObject": value("CITARGET", "-99.9"),
            "DiffMags": {"TRUE": "Y", "FALSE": "N"}.get(value("DIFFERMAGS"), value("DIFFERMAGS")),
            "DiffZeroMag": value("MAGADJUST", "-99.9"),
            "MagStd": {"INTERNAL": "INTSD"}.get(value("STANDARD"), value("STANDARD")),
            "LTCApplied": value("LTCAPP"), "LTCType": value("LTCTYPE", "NONE"),
            "LTCDays": value("LTCDAYS", "-99.9"), "ReducedMags": value("REDUCEDMAGS"),
            "UnityCor": value("UCORMAG", "-99.9"), "BibCode": value("BIBCODE") or "-",
            "Publication": value("PUBLICATION") or "-", "Comments": "; ".join(metadata.get("COMMENT", [])),
        }
        sessions.append(session)
        originals.append({"metadata_id": identifier, "fields": dict(metadata)})
        delimiter = {"PIPE": "|", "COMMA": ",", "TAB": "\t"}.get(value("DELIMITER"))
        if delimiter is None:
            raise ValueError("unsupported or missing ALCDEF DELIMITER")
        for line in records:
            fields = [v.strip() for v in line.split(delimiter)]
            if len(fields) not in (2, 3, 4):
                raise ValueError("invalid ALCDEF DATA field count")
            jd, magnitude = map(float, fields[:2])
            if not math.isfinite(jd) or jd <= 0 or not math.isfinite(magnitude):
                raise ValueError("invalid ALCDEF JD/magnitude")
            if len(fields) > 2 and fields[2] and not math.isfinite(float(fields[2])):
                raise ValueError("invalid ALCDEF magnitude error")
            points.append({"ObjectNumber": f"{number:06d}", "ObjectName": value("OBJECTNAME"),
                           "MDID": identifier, "JD": fields[0], "Mag": fields[1],
                           "MagErr": fields[2] if len(fields) > 2 else ""})

    for line_number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line == "STARTMETADATA":
            finish()
            metadata, records, state = {}, [], "metadata"
        elif line == "ENDMETADATA" and state == "metadata":
            state = "points"
        elif line in ("ENDDATA", "ENDLIGHTCURVE") and state == "points":
            continue
        elif "=" in line:
            key, value = line.split("=", 1)
            if state == "metadata":
                metadata.setdefault(key, []).append(value.strip())
            elif state == "points" and key == "DATA":
                records.append(value)
            else:
                raise ValueError(f"unexpected ALCDEF field on line {line_number}")
        else:
            raise ValueError(f"unexpected ALCDEF syntax on line {line_number}")
    finish()
    if not sessions or len({row["ID"] for row in sessions}) != len(sessions):
        raise ValueError("missing sessions or duplicate LCBLOCKID")
    return {"object_id": f"mpc:{number}", "source_mpc_number": number,
            "session_count": len(sessions), "raw_point_count": len(points),
            "sessions": sorted(sessions, key=lambda row: int(row["ID"])), "points": points}, originals


def from_zip(archive: Path, number: int) -> tuple[dict, list[dict], dict]:
    with zipfile.ZipFile(archive) as source:
        names = [name for name in source.namelist() if Path(name).name.startswith(f"ALCDEF_{number}_") and name.endswith(".txt")]
        if len(names) != 1:
            raise ValueError(f"expected one ALCDEF member for {number}, found {len(names)}")
        info = source.getinfo(names[0])
        if info.file_size > 20_000_000:
            raise ValueError("ALCDEF member too large")
        payload = source.read(names[0])
    obj, metadata = parse_alcdef(payload.decode("utf-8-sig"), number)
    return obj, metadata, {"path": names[0], "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}


def summary(obj: dict, metadata: list[dict]) -> dict:
    jds = [float(row["JD"]) for row in obj["points"]]
    def date(jd):
        return (datetime(2000, 1, 1, 12, tzinfo=timezone.utc) + timedelta(days=jd - 2451545)).isoformat()
    fields = {key: sorted({v for row in metadata for v in row["fields"].get(key, [])})
              for key in ("SESSIONDATE", "MAGBAND", "FILTER", "DIFFERMAGS", "STANDARD", "LTCAPP", "REDUCEDMAGS", "TIMESYS", "TIMESYSTEM")}
    return {"object_id": obj["object_id"], "source": "alcdef", "nights": len(fields["SESSIONDATE"]),
            "night_definition": "distinct native SESSIONDATE values", "sessions": obj["session_count"],
            "points": len(jds), "jd_min": min(jds), "jd_max": max(jds),
            "calendar_min": date(min(jds)), "calendar_max": date(max(jds)),
            "time_system": "JD as supplied; scale not explicitly declared in these files; no conversion applied",
            "calendar_note": "calendar renderings treat numeric JD as a civil day count; they do not establish its time scale",
            "source_fields": fields, "corrections_applied_by_adapter": "none"}


def adapter_check(archive: Path, published: Path, lock: Path, number: int) -> dict:
    raw = json.loads(published.read_text())
    cohort = json.loads(lock.read_text())
    if raw["schema"] != SCHEMA or sha256(lock) != raw["cohort_lock"]["sha256"]:
        raise ValueError("published raw extraction/lock binding mismatch")
    if f"mpc:{number}" not in {row["object_id"] for row in cohort["objects"]}:
        raise ValueError("adapter-check object is not in the locked cohort")
    expected = next(row for row in raw["objects"] if row["object_id"] == f"mpc:{number}")
    actual, _, member = from_zip(archive, number)
    if actual != expected:
        raise ValueError("adapter check FAILED: converted object differs from published raw extraction")
    return {"result": "PASS", "comparison": "full object dictionary equality, including all session metadata and ordered point strings",
            "object_id": actual["object_id"], "sessions": actual["session_count"], "points": actual["raw_point_count"],
            "published_raw_sha256": sha256(published), "cohort_lock_sha256": sha256(lock), "member": member,
            "source_zip_sha256": sha256(archive)}


def training_check() -> dict:
    split = json.loads(SPLIT.read_text())
    leaves = []
    def walk(value: object, path: str = "$") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                walk(child, path + "." + key)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, path + f"[{index}]")
        else:
            leaves.append((path, value))
    walk(split)
    with (SNAPSHOT / "tables/asteroids.csv").open(encoding="utf-8-sig", newline="") as stream:
        asteroids = list(csv.DictReader(stream))
    with (SNAPSHOT / "tables/asteroid_models.csv").open(encoding="utf-8-sig", newline="") as stream:
        models = list(csv.DictReader(stream))
    lowq = json.loads(LOWQ.read_text())
    rows = []
    for number in (113, 477, 478, 2836, 25143, 162173, 101955):
        forms = {number, str(number), f"asteroid_{number}", f"damit:{number}", f"mpc:{number}"}
        matches = [(path, value) for path, value in leaves if isinstance(value, (str, int)) and value in forms]
        if matches:
            raise ValueError(f"training-split contamination: {number}: {matches}")
        identity = [row for row in asteroids if row["id"] == str(number)] if number in (113, 477, 478, 2836) else [row for row in asteroids if row["number"] == str(number)]
        if number in (25143, 162173, 101955) and identity:
            # Check mapped DAMIT identity too; MPC and DAMIT numbers are different namespaces.
            for row in identity:
                mapped = {f"asteroid_{row['id']}", f"damit:{row['id']}", row["id"]}
                if any(isinstance(v, str) and v in mapped for _, v in leaves):
                    raise ValueError(f"mapped DAMIT identity occurs in training split: {number}")
        ids = {row["id"] for row in identity}
        flags = [row["quality_flag"] for row in models if row["asteroid_id"] in ids]
        in_lowq = any(row["object_id"] in {f"damit:{i}" for i in ids} for row in lowq["objects"])
        rows.append({"requested_id": number, "namespace": "damit" if number < 3000 else "mpc",
                     "split_matches": [], "snapshot_identities": [{k: row[k] for k in ("id", "number", "name")} for row in identity],
                     "snapshot_quality_flags": flags, "in_lowq_census": in_lowq})
    return {"result": "PASS", "scope": "every scalar leaf in every list/dictionary of publication-splits-v2.3.json",
            "split_sha256": sha256(SPLIT), "lowq_index_sha256": sha256(LOWQ), "objects": rows,
            "lowq_explanation": "113/477/478 have only blank quality_flag values; census build_index skips unparseable flags and requires a highest quality of 1 or 2, with no quality >=3. Mathilde 2836 has Q1 and is in the census."}


def calendar(jd: float) -> str:
    """Civil rendering only: this does not assign a physical JD time scale."""
    return (datetime(2000, 1, 1, 12, tzinfo=timezone.utc) + timedelta(days=jd - 2451545)).isoformat()


def ut_night(jd: float) -> str:
    return calendar(jd)[:10]


def numeric_rows(text: str, columns: int) -> list[list[str]]:
    rows = [line.split() for line in text.splitlines() if line.strip()]
    if not rows or any(len(row) != columns for row in rows):
        raise ValueError(f"expected nonempty {columns}-column source table")
    return rows


def photometry(jd: str, magnitude: str, error: str = "") -> tuple[str, str, str]:
    if not math.isfinite(float(jd)) or float(jd) <= 0 or not math.isfinite(float(magnitude)):
        raise ValueError("invalid catalogue JD/magnitude")
    if error and (not math.isfinite(float(error)) or float(error) < 0):
        raise ValueError("invalid catalogue magnitude error")
    return jd, magnitude, error


def describe(obj: dict, name: str, source: str, **policy: object) -> dict:
    jds = [float(point["JD"]) for point in obj["points"]]
    nights = sorted({ut_night(jd) for jd in jds})
    counts = {row["ID"]: 0 for row in obj["sessions"]}
    for point in obj["points"]:
        counts[point["MDID"]] += 1
    return {"object_id": obj["object_id"], "object_name": name, "source": source,
            "nights": len(nights), "ut_nights": nights,
            "night_definition": "distinct civil UT dates of numeric observation JD; scale caveats below",
            "sessions": obj["session_count"], "points": len(jds),
            "jd_min": min(jds), "jd_max": max(jds),
            "calendar_min": calendar(min(jds)), "calendar_max": calendar(max(jds)),
            "singleton_sessions": [identifier for identifier, count in counts.items() if count == 1],
            "corrections_applied_by_adapter": "none", "sessions_dropped_in_combined": [], **policy}


def catalogue(number: int, name: str, source: str, groups: list[tuple[list, dict]],
              inputs: list[dict], **policy: object) -> dict:
    sessions, points, metadata = [], [], []
    for index, (rows, fields) in enumerate(groups, 1):
        identifier = f"{source}:{index}"
        median = statistics.median(float(row[0]) for row in rows)
        sessions.append({"ID": identifier, "ObjNumber": str(number), "ObjName": name.title(),
                         "SessionDateTime": calendar(median).replace("+00:00", "Z"),
                         "Filter": fields.get("filter", "R"), "MagBand": fields.get("filter", "R"),
                         "SourceFile": fields["filename"], "Source": source,
                         "TimeSystem": policy["time_system"],
                         "LTCApplied": policy["light_time_correction"],
                         "MagnitudeType": fields.get("magnitude_type", policy["magnitude_type"]),
                         "UTNightOfMedianJD": ut_night(median)})
        metadata.append({"metadata_id": identifier, "fields": fields})
        for jd, magnitude, error in rows:
            points.append({"ObjectNumber": f"{number:06d}", "ObjectName": name.title(),
                           "MDID": identifier, "JD": jd, "Mag": magnitude, "MagErr": error})
    obj = {"object_id": f"mpc:{number}", "source_mpc_number": number,
           "session_count": len(sessions), "raw_point_count": len(points),
           "sessions": sessions, "points": points}
    return source_payload(obj, metadata, inputs, describe(obj, name, source, **policy))


def source_payload(obj: dict, metadata: list[dict], inputs: list[dict], description: dict) -> dict:
    return {"schema": SCHEMA, "purpose": "spacecraft raw photometry before geometry, period search, or prediction",
            "input_tables": inputs, "objects": [obj], "source_metadata": metadata,
            "source_summary": description,
            "cohort_lock": {"path": None, "sha256": None, "status": "spacecraft cohort lock not yet created"},
            "status": "converted; geometry and cohort lock pending"}


def parse_lowry(text: str, log: str, inputs: list[dict]) -> dict:
    segments = {}
    for lc, jd, magnitude, error in numeric_rows(text, 4):
        if not 1 <= int(lc) <= 10:
            raise ValueError("invalid Lowry LC number")
        segments.setdefault(int(lc), []).append(photometry(jd, magnitude, error))
    # table1 is retained as supplied, but its dates/counts disagree with table2.
    # Do not transfer its telescope or geometry onto mismatched LC identifiers.
    logs = {int(line[7:9]): line for line in log.splitlines() if line.strip()}
    if set(logs) != set(segments):
        raise ValueError("Lowry segment identifiers differ between tables")
    groups = []
    for lc, rows in sorted(segments.items()):
        nights = []
        for row in sorted(rows, key=lambda row: float(row[0])):
            if not nights or float(row[0]) - float(nights[-1][-1][0]) > 0.3:
                nights.append([])
            nights[-1].append(row)
        for index, night in enumerate(nights, 1):
            groups.append((night, {"filename": "lowry2014/table2.dat", "LC": str(lc),
                                   "night_within_segment": index, "table1_row": logs[lc],
                                   "table1_warning": "log dates/counts are not reliable for table2 LC mapping"}))
    return catalogue(25143, "itokawa", "lowry2014", groups, inputs,
                     time_system="unspecified", light_time_correction="unspecified",
                     magnitude_type="relative R", corrections_already_applied="unspecified in ReadMe",
                     session_definition="within each LC, sorted JD split at consecutive gaps > 0.3 d; may cross UT midnight",
                     format_notes=["table1 dates/counts disagree with table2 for several LC identifiers; table2 alone defines photometry and sessions.",
                                   "Separate gap-defined sessions can share a UT date; all observed UT dates enter combined-source overlap checks."])


def parse_kim(table: str, lightcurves: dict[str, str], inputs: list[dict]) -> dict:
    labels = ("Ast", "Obs.date", "JD", "r", "Delta", "alpha", "PABLon", "PABLat", "F", "Obs", "LCN", "FileName")
    metadata = {}
    for fields in numeric_rows(table, len(labels)):
        row = dict(zip(labels, fields))
        key = row["FileName"], int(row["LCN"])
        if row["Ast"] != "162173" or Path(key[0]).name != key[0] or key in metadata:
            raise ValueError("invalid or duplicate Kim per-lightcurve metadata")
        metadata[key] = row
    if {key[0] for key in metadata} != set(lightcurves):
        raise ValueError("Kim lightcurve filenames differ from table.dat")
    grouped = {}
    seen = set()
    for filename, text in sorted(lightcurves.items()):
        for jd, magnitude, lc in numeric_rows(text, 3):
            row = photometry(jd, magnitude)
            key = filename, int(lc)
            if key not in metadata:
                raise ValueError("Kim point has no matching (filename, LCN) metadata")
            seen.add(key)
            grouped.setdefault((*key, ut_night(float(jd))), []).append(row)
    if seen != set(metadata):
        raise ValueError("Kim table.dat contains an unrepresented lightcurve")
    groups = []
    for (filename, lc, night), rows in sorted(grouped.items()):
        groups.append((rows, {"filename": "kim2013/lc/" + filename, "LCN": str(lc), "ut_night": night,
                              "table.dat": metadata[filename, lc], "filter": metadata[filename, lc]["F"],
                              "magnitude_type": "absolute mag. (source label)" if filename == "UH2011lcs.dat" else "relative R"}))
    return catalogue(162173, "ryugu", "kim2013", groups, inputs,
                     time_system="unspecified JD scale", light_time_correction="not applied (ReadMe)",
                     magnitude_type="relative R except UH2011lcs.dat: absolute mag. (source label)",
                     corrections_already_applied="no light-time correction; UH2011 absolute-magnitude reduction not defined in ReadMe; other reductions unspecified",
                     session_definition="one session per (filename, LCN, UT date of numeric JD)",
                     format_notes=["Magnitude uncertainties are absent; MagErr is an empty string.",
                                   "table.dat observation date describes the lightcurve; points can cross UT midnight. Original per-lightcurve metadata and filename links are retained.",
                                   "UH2011 absolute mag. is preserved literally; no inference about its distance/phase reduction is made."])


def parse_bennu(text: str, label: str, inputs: list[dict]) -> dict:
    root = ET.fromstring(label)
    ns = {"p": "http://pds.nasa.gov/pds4/pds/v1"}
    table = root.find("p:File_Area_Observational/p:Table_Character", ns)
    if table is None or root.findtext("p:File_Area_Observational/p:File/p:file_name", namespaces=ns) != "bennu_ecas_w.tab":
        raise ValueError("unexpected Bennu w-band label")
    fields = {}
    for field in table.findall("p:Record_Character/p:Field_Character", ns):
        name = field.findtext("p:name", namespaces=ns)
        start = int(field.findtext("p:field_location", namespaces=ns)) - 1
        fields[name] = slice(start, start + int(field.findtext("p:field_length", namespaces=ns)))
    required = {"OBS_DATE_TIME", "EXPOSURE", "MAGNITUDE", "MAGNITUDE_ERR", "JULIAN_DATE"}
    if not required <= fields.keys():
        raise ValueError("Bennu label lacks required columns")
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) != int(table.findtext("p:records", namespaces=ns)):
        raise ValueError("Bennu data/label record count mismatch")
    grouped, originals = {}, {}
    for line in lines:
        if len(line) < max(field.stop for field in fields.values()):
            raise ValueError("truncated Bennu record")
        row = {name: line[field].strip() for name, field in fields.items()}
        point = photometry(row["JULIAN_DATE"], row["MAGNITUDE"], row["MAGNITUDE_ERR"])
        start = datetime.fromisoformat(row["OBS_DATE_TIME"]).replace(tzinfo=timezone.utc)
        midpoint = start + timedelta(seconds=float(row["EXPOSURE"]) / 2)
        # JD is printed to 1e-5 d. This also verifies its relation to UT.
        if abs((datetime.fromisoformat(calendar(float(point[0]))) - midpoint).total_seconds()) > 0.6:
            raise ValueError("Bennu JD is inconsistent with labelled UT mid-exposure time")
        night = ut_night(float(point[0]))
        grouped.setdefault(night, []).append(point)
        originals.setdefault(night, []).append(row)
    if set(grouped) != {f"2005-09-{day:02d}" for day in range(14, 18)}:
        raise ValueError("unexpected Bennu UT nights")
    groups = [(rows, {"filename": "hergenrother2020/data/lightcurve/bennu_ecas_w.tab",
                      "ut_night": night, "filter": "w", "original_rows": originals[night]})
              for night, rows in sorted(grouped.items())]
    return catalogue(101955, "bennu", "hergenrother2020", groups, inputs,
                     time_system="UT observation start; numeric JD at mid-exposure; UTC/TT realization unspecified",
                     light_time_correction="unspecified in label/bundle description",
                     magnitude_type="apparent ECAS w",
                     corrections_already_applied="bias subtraction, flat fielding, variable-aperture photometry; ECAS standard-star/solar-colour calibration; contaminated measurements removed; light-time/phase/distance corrections unspecified",
                     session_definition="one session per UT date of mid-exposure JD",
                     format_notes=["Only bennu_ecas_w.tab is used; other ECAS filters are excluded colour sequences.",
                                   "Label declares 366 w-band rows; bundle prose says 339 images across filters. Use the actual labelled table without discarding rows."])


def load_sources(raw: Path, archive: Path, *, alcdef_only: bool = False) -> list[dict]:
    receipt_path = raw / "receipts.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    recorded = {row["path"]: row for row in receipt["files"]}

    def read(relative: str) -> tuple[str, dict]:
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("unsafe source path")
        path = raw / relative
        row = recorded[relative]
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != row["sha256"] or len(payload) != row["bytes"]:
            raise ValueError(f"acquisition receipt mismatch: {relative}")
        return payload.decode("utf-8-sig"), dict(row)

    sources = []
    archive_hash = sha256(archive)
    for name, number in (("itokawa", 25143), ("ryugu", 162173)):
        obj, metadata, member = from_zip(archive, number)
        _, row = read("alcdef/" + member["path"])
        if row["sha256"] != member["sha256"] or row["archive_sha256"] != archive_hash:
            raise ValueError(f"ALCDEF acquisition receipt mismatch: {name}")
        native = summary(obj, metadata)
        description = describe(obj, name, "alcdef", time_system=native["time_system"],
                               magnitude_type="relative R" if name == "itokawa" else "source DIFFERMAGS=FALSE; MagBand=SR, internal standard",
                               light_time_correction="none (LTCAPP=NONE)",
                               corrections_already_applied="LTCAPP=NONE; REDUCEDMAGS=NONE; CICORRECTION=FALSE",
                               session_definition="native ALCDEF LCBLOCKID; original adapter unchanged",
                               source_fields=native["source_fields"])
        payload = source_payload(obj, metadata, [row], description)
        payload["source_zip"] = {"path": str(archive), "sha256": archive_hash}
        sources.append(payload)
    if not alcdef_only:
        _, lowry_readme = read("lowry2014/ReadMe")
        lowry, lowry_input = read("lowry2014/table2.dat")
        log, log_input = read("lowry2014/table1.dat")
        sources.append(parse_lowry(lowry, log, [lowry_readme, log_input, lowry_input]))
        _, kim_readme = read("kim2013/ReadMe")
        table, table_input = read("kim2013/table.dat")
        filenames = {row[-1] for row in numeric_rows(table, 12)}
        curves, inputs = {}, [kim_readme, table_input]
        for filename in sorted(filenames):
            if Path(filename).name != filename:
                raise ValueError("unsafe Kim filename")
            text, row = read("kim2013/lc/" + filename)
            curves[filename] = text
            inputs.append(row)
        sources.append(parse_kim(table, curves, inputs))
        bennu, bennu_input = read("hergenrother2020/data/lightcurve/bennu_ecas_w.tab")
        label, label_input = read("hergenrother2020/data/lightcurve/bennu_ecas_w.xml")
        _, bundle_input = read("hergenrother2020/document/bundle_description.txt")
        sources.append(parse_bennu(bennu, label, [bennu_input, label_input, bundle_input]))
    for payload in sources:
        payload["acquisition_receipt"] = {"path": str(receipt_path), "sha256": sha256(receipt_path)}
    return sources


def combine(sources: list[dict]) -> dict:
    descriptions = [source["source_summary"] for source in sources]
    if len({row["object_id"] for row in descriptions}) != 1:
        raise ValueError("combined sources must refer to the same object")
    catalogue_nights = {night for row in descriptions if row["source"] != "alcdef" for night in row["ut_nights"]}
    sessions, points, metadata, inputs, dropped = [], [], [], [], []
    for source in sources:
        obj, description = source["objects"][0], source["source_summary"]
        by_session = {}
        for point in obj["points"]:
            by_session.setdefault(point["MDID"], []).append(float(point["JD"]))
        removed = []
        if description["source"] == "alcdef":
            for session in obj["sessions"]:
                night = ut_night(statistics.median(by_session[session["ID"]]))
                if night in catalogue_nights:
                    removed.append({"metadata_id": session["ID"], "ut_night": night,
                                    "points": len(by_session[session["ID"]]), "source": "alcdef"})
        # Only ALCDEF/catalogue overlaps are removed, never catalogue sessions
        # sharing a date with another catalogue session.
        description["sessions_dropped_in_combined"] = removed
        dropped.extend(removed)
        keep = {row["ID"] for row in obj["sessions"]} - {row["metadata_id"] for row in removed}
        sessions.extend(dict(row) for row in obj["sessions"] if row["ID"] in keep)
        points.extend(dict(row) for row in obj["points"] if row["MDID"] in keep)
        metadata.extend({**row, "source": description["source"]} for row in source["source_metadata"] if row["metadata_id"] in keep)
        for row in source["input_tables"]:
            if row not in inputs:
                inputs.append(row)
    if len({row["ID"] for row in sessions}) != len(sessions):
        raise ValueError("session ID collision between combined sources")
    first = sources[0]["objects"][0]
    obj = {"object_id": first["object_id"], "source_mpc_number": first["source_mpc_number"],
           "session_count": len(sessions), "raw_point_count": len(points), "sessions": sessions, "points": points}
    description = describe(obj, descriptions[0]["object_name"], "combined",
                           time_system="; ".join(f"{row['source']}: {row['time_system']}" for row in descriptions),
                           magnitude_type="; ".join(f"{row['source']}: {row['magnitude_type']}" for row in descriptions),
                           corrections_already_applied="see constituent source summaries",
                           sessions_dropped_in_combined=dropped)
    result = source_payload(obj, metadata, inputs, description)
    result["combined_policy"] = "drop an ALCDEF session when the UT date of its median JD occurs among any catalogue observation JDs of the same object"
    result["constituent_sources"] = descriptions
    result["acquisition_receipt"] = sources[0]["acquisition_receipt"]
    return result


def data_markdown(summaries: list[dict]) -> str:
    """Return the complete DATA.md content without writing any files."""
    lines = ["# Spacecraft raw photometry", "", "Conversion only; no geometry, period search, models or predictions.", "",
             "| Object | Source | UT dates | Sessions | Points | Date range | Sessions dropped in combined |",
             "| --- | --- | ---: | ---: | ---: | --- | ---: |"]
    for row in summaries:
        lines.append(f"| {row['object_name'].title()} | {row['source']} | {row['nights']} | {row['sessions']} | {row['points']} | {row['calendar_min'][:10]} to {row['calendar_max'][:10]} | {len(row['sessions_dropped_in_combined'])} |")
    lines.extend(["", "Numeric JD, magnitude and error strings are unchanged. Missing Kim errors remain empty strings.",
                  "UT dates are civil renderings of numeric JD, not an assertion of an unspecified source time scale.",
                  "Lowry uses the requested 0.3-day gap rule within each LC; sessions can cross midnight or share a UT date.",
                  "Kim uses (filename, LCN, UT date); Bennu uses UT date. Original metadata and filename links are retained.",
                  "Combined removes only ALCDEF sessions whose median-JD UT date occurs in catalogue observation JDs of that object.",
                  "No light-time, distance, phase, colour or brightness correction is added by this adapter.",
                  "SessionDateTime Z follows the accepted schema; it does not establish the source JD time scale.",
                  "The cohort lock is pending; Job 2 must establish a hash-bound lock before freezing SBDB receipts.", ""])
    for row in summaries:
        lines.extend([f"## {row['object_name'].title()}: {row['source']}", "",
                      f"Time system: {row['time_system']}.", f"Magnitude type: {row['magnitude_type']}.",
                      f"Light-time correction: {row.get('light_time_correction', 'see constituent source summaries')}.",
                      f"Corrections already applied: {row['corrections_already_applied']}.",
                      f"Sessions dropped in combined: {json.dumps(row['sessions_dropped_in_combined'], sort_keys=True)}."])
        lines.extend(row.get("format_notes", []))
        if row["singleton_sessions"]:
            lines.append(f"Single-point sessions retained in raw data (downstream prepare skips these): {', '.join(row['singleton_sessions'])}.")
        if "source_fields" in row:
            lines.extend(["", "Native ALCDEF fields:", "```json", json.dumps(row["source_fields"], indent=2, sort_keys=True), "```"])
        lines.append("")
    return "\n".join(lines)


def convert_sources(output: Path, archive: Path, raw: Path = RAW, *, alcdef_only: bool = False) -> dict:
    # Preflight every destination before parsing or writing; open('x') also
    # protects against a file appearing after this check.
    pairs = [("itokawa", "alcdef"), ("ryugu", "alcdef")]
    if not alcdef_only:
        pairs += [("itokawa", "lowry2014"), ("ryugu", "kim2013"), ("bennu", "hergenrother2020")]
        pairs += [(name, "combined") for name in ("itokawa", "ryugu", "bennu")]
    paths = [output / f"raw-extraction-{name}-{source}.json" for name, source in pairs] + [output / "DATA.md"]
    for path in paths:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"conversion output already exists: {path}")
    sources = load_sources(raw, archive, alcdef_only=alcdef_only)
    products = list(sources)
    if not alcdef_only:
        for name in ("itokawa", "ryugu", "bennu"):
            products.append(combine([source for source in sources if source["source_summary"]["object_name"] == name]))
    summaries = [product["source_summary"] for product in products]
    markdown = data_markdown(summaries)
    for path, payload in zip(paths[:-1], products):
        write_json(path, payload)
    with paths[-1].open("x", encoding="utf-8") as stream:
        stream.write(markdown)
    return {"status": "converted", "outputs": [str(path) for path in paths],
            "sources": summaries, "data_md": markdown}


def convert_alcdef(output: Path, archive: Path) -> dict:
    return convert_sources(output, archive, alcdef_only=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("convert", "alcdef", "adapter-check", "training-check", "summary"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--raw-dir", type=Path, default=RAW)
    parser.add_argument("--source-zip", type=Path, default=ARCHIVE)
    parser.add_argument("--published-raw", type=Path, default=PUBLISHED)
    parser.add_argument("--cohort-lock", type=Path, default=LOCK)
    parser.add_argument("--mpc-number", type=int, default=26628)
    args = parser.parse_args()
    if args.command == "adapter-check":
        result = adapter_check(args.source_zip, args.published_raw, args.cohort_lock, args.mpc_number)
    elif args.command == "training-check":
        result = training_check()
    elif args.command == "summary":
        sources = load_sources(args.raw_dir, args.source_zip)
        products = sources + [combine([source for source in sources if source["source_summary"]["object_name"] == name])
                              for name in ("itokawa", "ryugu", "bennu")]
        summaries = [product["source_summary"] for product in products]
        result = {"sources": summaries, "data_md": data_markdown(summaries)}
    else:
        result = convert_sources(args.output_dir, args.source_zip, args.raw_dir, alcdef_only=args.command == "alcdef")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
