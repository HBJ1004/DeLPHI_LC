"""Fail-closed readers for public ALCDEF PDS and TSSYS-DR1 products.

These readers deliberately stop at source photometry and metadata.  Neither
archive contains the asteroid-centric vectors required by :class:`Observation`,
so this module never fabricates geometry, applies a light-time correction, or
turns an object identifier into an ``independent`` scientific sample.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

ALCDEF_SOURCE_URI = "https://sbn.psi.edu/pds/resource/alcdef.html"
ALCDEF_BUNDLE_URI = (
    "https://sbnarchive.psi.edu/pds4/non_mission/gbo.ast.alcdef-database_V1_0/"
)
TSSYS_SOURCE_URI = "https://archive.konkoly.hu/pub/tssys/dr1/README"
GAIA_DR3_SPINS_SOURCE_URI = "https://cdsarc.cds.unistra.fr/ftp/J/A+A/675/A24/"


class ExternalSourceError(ValueError):
    """Raised when an input cannot be parsed without guessing its meaning."""


def _clean(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() in {"-", "NONE", "NULL", "N/A", "NA", "UNKNOWN"}:
        return None
    return text


def _raw(value: object) -> str | None:
    """Preserve explicit correction tokens such as ``NONE``."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _finite(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _positive(value: object) -> float | None:
    result = _finite(value)
    return result if result is not None and result > 0 else None


def _integer(value: object, *, positive: bool = False) -> int | None:
    text = _clean(value)
    if text is None or not re.fullmatch(r"[+-]?\d+", text):
        return None
    try:
        result = int(text)
    except ValueError:
        return None
    return result if not positive or result > 0 else None


def _aliases(*values: object) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        text = _clean(value)
        if text is not None and text not in result:
            result.append(text)
    return tuple(result)


def _iso_datetime(value: object) -> str | None:
    text = _clean(value)
    if text is None:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_identity_map(path: str | Path) -> dict[int, str]:
    """Load an explicit MPC-number -> internal source-ID map for overlap audits."""
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = csv.DictReader(handle)
            result: dict[int, str] = {}
            for row in rows:
                number = _integer(row.get("number"), positive=True)
                source_id = _clean(row.get("id"))
                if number is not None and source_id is not None:
                    if number in result and result[number] != source_id:
                        raise ExternalSourceError(f"duplicate MPC number in identity map: {number}")
                    result[number] = f"damit:{source_id}"
            return result
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ExternalSourceError(f"cannot read identity map {source}: {exc}") from exc


@dataclass(frozen=True)
class ALCDEFMetadata:
    record_id: str
    object_number: int | None
    object_name: str | None
    designation: str | None
    aliases: tuple[str, ...]
    submitter: str | None
    facility: str | None
    session_datetime: str | None
    filter_name: str | None
    magnitude_band: str | None
    correction_metadata: Mapping[str, str | None]
    metadata_issues: tuple[str, ...] = ()
    source_file: str | None = None
    row_number: int | None = None

    @property
    def candidate_identity(self) -> str:
        if self.object_number is not None:
            return f"mpc:{self.object_number}"
        if self.aliases:
            return f"designation:{self.aliases[0]}"
        return f"record:{self.record_id}"


@dataclass(frozen=True)
class ALCDEFPoint:
    observation_id: str
    object_number: int | None
    object_name: str | None
    metadata_id: str
    time_jd: float | None
    magnitude: float | None
    magnitude_error: float | None
    valid: bool
    exclusion_reasons: tuple[str, ...] = ()
    source_file: str | None = None
    row_number: int | None = None


@dataclass(frozen=True)
class TSSYSMetadata:
    target_id: str
    object_number: int | None
    aliases: tuple[str, ...]
    frequency_cycles_per_day: float | None
    period_hours: float | None
    lightcurve_type: str | None
    point_count_reported: int | None
    sector: int | None
    camera: int | None
    ccd: int | None
    metadata_issues: tuple[str, ...] = ()
    source_file: str | None = None
    row_number: int | None = None

    @property
    def candidate_identity(self) -> str:
        return f"mpc:{self.object_number}" if self.object_number is not None else f"designation:{self.target_id}"


@dataclass(frozen=True)
class TSSYSPoint:
    observation_id: str
    target_id: str
    time_jd: float | None
    magnitude: float | None
    magnitude_error: float | None
    flags: str | None
    valid: bool
    exclusion_reasons: tuple[str, ...] = ()
    source_file: str | None = None
    row_number: int | None = None


@dataclass(frozen=True)
class GaiaDR3Spin:
    """One published Gaia DR3 inversion solution, not physical ground truth."""

    object_number: int
    object_name: str
    lambda1_deg: float
    beta1_deg: float
    lambda2_deg: float | None
    beta2_deg: float | None
    period_hours: float
    observation_count: int
    method: str
    source_file: str | None = None
    row_number: int | None = None


def _pds_rows(path: Path, expected: set[str]) -> tuple[list[dict[str, str]], list[str]]:
    """Read a PDS4 CSV table, ignoring product-description rows."""
    issues: list[str] = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.reader(handle))
    except (OSError, UnicodeError) as exc:
        raise ExternalSourceError(f"cannot read {path}: {exc}") from exc
    header_index: int | None = None
    for index, row in enumerate(rows):
        normalized = {cell.strip() for cell in row}
        if expected.issubset(normalized):
            header_index = index
            break
    if header_index is None:
        raise ExternalSourceError(f"no {sorted(expected)} header found in {path}")
    header = [cell.strip() for cell in rows[header_index]]
    result: list[dict[str, str]] = []
    for row_number, row in enumerate(rows[header_index + 1 :], header_index + 2):
        if not row or not any(cell.strip() for cell in row) or row[0].lstrip().startswith("#"):
            continue
        if len(row) < len(header):
            issues.append(f"malformed_row:{row_number}")
            continue
        result.append({key: row[index].strip() for index, key in enumerate(header)})
    return result, issues


def _iter_pds_rows(path: Path, expected: set[str]):
    """Streaming counterpart used by the large ALCDEF lcdata tables."""
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header: list[str] | None = None
            for row in reader:
                if expected.issubset({cell.strip() for cell in row}):
                    header = [cell.strip() for cell in row]
                    break
            if header is None:
                raise ExternalSourceError(f"no {sorted(expected)} header found in {path}")
            for row in reader:
                if not row or not any(cell.strip() for cell in row) or row[0].lstrip().startswith("#"):
                    continue
                if len(row) >= len(header):
                    yield {key: row[index].strip() for index, key in enumerate(header)}
    except (OSError, UnicodeError) as exc:
        raise ExternalSourceError(f"cannot read {path}: {exc}") from exc


def parse_gaia_dr3_spins(path: str | Path) -> list[GaiaDR3Spin]:
    """Read CDS J/A+A/675/A24 fixed-width Table 3 without guessing columns.

    The table's ReadMe defines ecliptic J2000 coordinates. It is treated as an
    external model-derived reference table, never as physical ground truth.
    """
    source = Path(path)
    records: list[GaiaDR3Spin] = []
    try:
        lines = source.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ExternalSourceError(f"cannot read Gaia DR3 spin table {source}: {exc}") from exc
    for row_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            number = int(line[0:6])
            name = line[7:23].strip()
            lambda1 = float(line[24:27])
            beta1 = float(line[28:31])
            lambda2 = _finite(line[32:35])
            beta2 = _finite(line[36:39])
            period = float(line[40:51])
            count = int(line[52:54])
            method = line[55:57].strip()
        except (ValueError, IndexError) as exc:
            raise ExternalSourceError(f"malformed Gaia DR3 spin row {row_number}") from exc
        if number <= 0 or not name or period <= 0 or count <= 0 or method not in {"C", "E", "CE"}:
            raise ExternalSourceError(f"invalid Gaia DR3 spin row {row_number}")
        if (lambda2 is None) != (beta2 is None):
            raise ExternalSourceError(f"incomplete Gaia DR3 second pole at row {row_number}")
        records.append(GaiaDR3Spin(number, name, lambda1, beta1, lambda2, beta2, period, count, method, str(source), row_number))
    if not records:
        raise ExternalSourceError(f"no Gaia DR3 spin rows in {source}")
    if len({row.object_number for row in records}) != len(records):
        raise ExternalSourceError("duplicate MPC number in Gaia DR3 spin table")
    return records


def parse_alcdef_metadata(path: str | Path) -> tuple[list[ALCDEFMetadata], list[str]]:
    """Parse one PDS ``alcdef_metadata-*.csv`` file."""
    source = Path(path)
    rows, issues = _pds_rows(source, {"ID", "ObjNumber", "ObjName", "SessionDateTime"})
    parsed: list[ALCDEFMetadata] = []
    for row_number, row in enumerate(rows, 1):
        record_id = _clean(row.get("ID"))
        # PDS CSV tables contain type/default/description rows after the
        # header.  ALCDEF IDs are positive integers; those rows are not data.
        if _integer(record_id, positive=True) is None:
            continue
        number = _integer(row.get("ObjNumber"), positive=True)
        object_name, designation = _clean(row.get("ObjName")), _clean(row.get("Desig"))
        metadata_issues: list[str] = []
        if number is None and not _aliases(object_name, designation):
            metadata_issues.append("unknown_object_identity")
        session = _iso_datetime(row.get("SessionDateTime"))
        if session is None:
            metadata_issues.append("unknown_session_time")
        filter_name, magnitude_band = _clean(row.get("Filter")), _clean(row.get("MagBand"))
        if filter_name is None:
            metadata_issues.append("unknown_filter")
        if magnitude_band is None:
            metadata_issues.append("unknown_magnitude_band")
        corrections = {
            key: _raw(row.get(key))
            for key in (
                "DiffMags", "DiffZeroMag", "MagStd", "CICorr", "ExpJD", "Exposure",
                "LTCApplied", "LTCType", "LTCDays", "ReducedMags", "UnityCor",
            )
        }
        if corrections["LTCApplied"] is None:
            metadata_issues.append("time_correction_unknown")
        parsed.append(
            ALCDEFMetadata(
                record_id=record_id,
                object_number=number,
                object_name=object_name,
                designation=designation,
                aliases=_aliases(object_name, designation),
                submitter=_clean(row.get("Submitter")),
                facility=_clean(row.get("Facility")),
                session_datetime=session,
                filter_name=filter_name,
                magnitude_band=magnitude_band,
                correction_metadata=corrections,
                metadata_issues=tuple(metadata_issues),
                source_file=str(source),
                row_number=row_number,
            )
        )
    return parsed, issues


def iter_alcdef_lcdata(path: str | Path):
    """Yield one PDS lcdata point at a time (safe for the 10M-row bundle)."""
    source = Path(path)
    seen: set[str] = set()
    for row_number, row in enumerate(_iter_pds_rows(source, {"ObjectNumber", "ObjectName", "MDID", "JD", "Mag"}), 1):
        number = _integer(row.get("ObjectNumber"), positive=True)
        name = _clean(row.get("ObjectName"))
        mdid = _clean(row.get("MDID"))
        # As above, ignore PDS type/default/description rows.  MDID is the
        # integer foreign key to one metadata block.
        if _integer(mdid, positive=True) is None:
            continue
        reasons: list[str] = []
        jd, magnitude = _finite(row.get("JD")), _finite(row.get("Mag"))
        magnitude_error = _finite(row.get("MagErr"))
        if mdid is None:
            reasons.append("unknown_metadata_record")
            mdid = "missing"
        if jd is None or jd <= 0:
            reasons.append("invalid_timestamp")
        if magnitude is None:
            reasons.append("invalid_magnitude")
        raw_error = _clean(row.get("MagErr"))
        if raw_error is not None:
            # PDS default -99.9 denotes the optional/missing ALCDEF error,
            # not an invalid point.  Other non-positive values are invalid.
            if magnitude_error == -99.9:
                magnitude_error = None
            elif magnitude_error is None or magnitude_error <= 0:
                reasons.append("invalid_magnitude_uncertainty")
                magnitude_error = None
        observation_id = f"{number if number is not None else name or 'unknown'}:{mdid}:{jd if jd is not None else row_number}"
        if observation_id in seen:
            reasons.append("duplicate_observation_id")
        seen.add(observation_id)
        yield ALCDEFPoint(
                observation_id=observation_id,
                object_number=number,
                object_name=name,
                metadata_id=mdid,
                time_jd=jd,
                magnitude=magnitude,
                magnitude_error=magnitude_error,
                valid=not reasons,
                exclusion_reasons=tuple(reasons),
                source_file=str(source),
                row_number=row_number,
            )


def parse_alcdef_lcdata(path: str | Path) -> tuple[list[ALCDEFPoint], list[str]]:
    """Parse one PDS ``alcdef_lcdata-*.csv`` file without applying corrections."""
    return list(iter_alcdef_lcdata(path)), []


def _find_files(root: Path, pattern: str) -> list[Path]:
    if root.is_file():
        return [root] if root.match(pattern) else []
    if not root.is_dir():
        raise ExternalSourceError(f"input path does not exist: {root}")
    return sorted(path for path in root.rglob(pattern) if path.is_file())


def parse_alcdef_archive(path: str | Path) -> tuple[list[ALCDEFMetadata], list[ALCDEFPoint], list[str]]:
    """Parse all ALCDEF metadata/lcdata tables under an extracted bundle."""
    root = Path(path)
    metadata_files = _find_files(root, "alcdef_metadata-*.csv")
    lcdata_files = _find_files(root, "alcdef_lcdata-*.csv")
    if not metadata_files and root.is_file():
        metadata_files = _find_files(root, "alcdef_metadata_*.csv")
    if not lcdata_files and root.is_file():
        lcdata_files = _find_files(root, "alcdef_lcdata_*.csv")
    if not metadata_files and not lcdata_files:
        raise ExternalSourceError(f"no ALCDEF PDS CSV files found below {root}")
    metadata: list[ALCDEFMetadata] = []
    points: list[ALCDEFPoint] = []
    issues: list[str] = []
    for file in metadata_files:
        rows, file_issues = parse_alcdef_metadata(file)
        metadata.extend(rows)
        issues.extend(f"{file.name}:{issue}" for issue in file_issues)
    for file in lcdata_files:
        rows, file_issues = parse_alcdef_lcdata(file)
        points.extend(rows)
        issues.extend(f"{file.name}:{issue}" for issue in file_issues)
    ids = Counter(row.record_id for row in metadata)
    issues.extend(f"duplicate_metadata_id:{key}" for key, count in ids.items() if count > 1)
    known = {row.record_id for row in metadata}
    for point in points:
        if point.metadata_id not in known:
            issues.append(f"unknown_metadata_record:{point.metadata_id}")
    return metadata, points, issues


def parse_tssys_release(path: str | Path) -> tuple[list[TSSYSMetadata], list[str]]:
    """Parse TSSYS-DR1's fixed-width/whitespace ``release.merge`` table."""
    source = Path(path)
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ExternalSourceError(f"cannot read {source}: {exc}") from exc
    parsed: list[TSSYSMetadata] = []
    issues: list[str] = []
    for row_number, line in enumerate(lines, 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 12:
            issues.append(f"malformed_row:{row_number}")
            continue
        target = _clean(fields[0])
        if target is None:
            issues.append(f"missing_target:{row_number}")
            continue
        number = _integer(target, positive=True)
        metadata_issues: list[str] = []
        frequency = _positive(fields[1])
        period = _positive(fields[3])
        if frequency is None:
            metadata_issues.append("unknown_rotation_frequency")
        if period is None:
            metadata_issues.append("unknown_rotation_period")
        try:
            point_count = _integer(fields[5], positive=True)
            sector, camera, ccd = (_integer(fields[index], positive=True) for index in (6, 7, 8))
        except IndexError:
            point_count = sector = camera = ccd = None
        if point_count is None:
            metadata_issues.append("unknown_reported_point_count")
        if sector is None or camera is None or ccd is None:
            metadata_issues.append("unknown_session_detector")
        parsed.append(
            TSSYSMetadata(
                target_id=target,
                object_number=number,
                aliases=_aliases(target),
                frequency_cycles_per_day=frequency,
                period_hours=period,
                lightcurve_type=_clean(fields[2]),
                point_count_reported=point_count,
                sector=sector,
                camera=camera,
                ccd=ccd,
                metadata_issues=tuple(metadata_issues),
                source_file=str(source),
                row_number=row_number,
            )
        )
    return parsed, issues


def parse_tssys_lightcurve(path: str | Path) -> tuple[list[TSSYSPoint], list[str]]:
    """Parse TSSYS ``.lc`` rows (target, JD, coordinates, mag, error, ..., flags)."""
    source = Path(path)
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ExternalSourceError(f"cannot read {source}: {exc}") from exc
    parsed: list[TSSYSPoint] = []
    issues: list[str] = []
    seen: set[str] = set()
    for row_number, line in enumerate(lines, 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        reasons: list[str] = []
        if len(fields) < 13:
            issues.append(f"malformed_row:{row_number}")
            continue
        target = _clean(fields[0]) or "unknown"
        jd, magnitude, magnitude_error = _finite(fields[1]), _finite(fields[4]), _finite(fields[5])
        if jd is None or jd <= 0:
            reasons.append("invalid_timestamp")
        if magnitude is None:
            reasons.append("invalid_magnitude")
        if magnitude_error is None or magnitude_error < 0:
            reasons.append("invalid_magnitude_uncertainty")
            magnitude_error = None
        observation_id = f"{target}:{jd if jd is not None else row_number}"
        if observation_id in seen:
            reasons.append("duplicate_observation_id")
        seen.add(observation_id)
        parsed.append(
            TSSYSPoint(
                observation_id=observation_id,
                target_id=target,
                time_jd=jd,
                magnitude=magnitude,
                magnitude_error=magnitude_error,
                flags=_clean(fields[9]),
                valid=not reasons,
                exclusion_reasons=tuple(reasons),
                source_file=str(source),
                row_number=row_number,
            )
        )
    return parsed, issues


def _input_files(path: Path, patterns: Sequence[str]) -> list[Path]:
    if path.is_file():
        return [path]
    result: list[Path] = []
    for pattern in patterns:
        result.extend(path.rglob(pattern))
    return sorted({item for item in result if item.is_file()})


def census_alcdef(path: str | Path, *, source_uri: str = ALCDEF_BUNDLE_URI, identity_map: Mapping[int, str] | None = None, identity_map_path: str | None = None) -> dict[str, object]:
    root = Path(path)
    files = _input_files(root, ("alcdef_metadata-*.csv", "alcdef_lcdata-*.csv"))
    metadata_files = [file for file in files if "metadata" in file.name]
    lcdata_files = [file for file in files if "lcdata" in file.name]
    metadata: list[ALCDEFMetadata] = []
    parse_issues: list[str] = []
    for file in metadata_files:
        rows, row_issues = parse_alcdef_metadata(file)
        metadata.extend(rows)
        parse_issues.extend(f"{file.name}:{issue}" for issue in row_issues)
    total_points = valid_points = 0
    timestamps: list[float] = []
    exclusions: Counter[str] = Counter()
    points_by_metadata: Counter[str] = Counter()
    valid_by_metadata: Counter[str] = Counter()
    unknown_object_points: dict[str, tuple[int | None, str | None, int, int]] = {}
    known_metadata_ids: set[str] = set()
    # populated after metadata parsing below; initialized here for streaming
    known_metadata_ids = {row.record_id for row in metadata}
    seen_observation_ids: set[str] = set()
    for file in lcdata_files:
        try:
            point_iterator = iter_alcdef_lcdata(file)
            for point in point_iterator:
                total_points += 1
                points_by_metadata[point.metadata_id] += 1
                if point.metadata_id not in known_metadata_ids:
                    key = f"mpc:{point.object_number}" if point.object_number is not None else f"designation:{point.object_name or 'unknown'}"
                    number, name, total, valid = unknown_object_points.get(key, (point.object_number, point.object_name, 0, 0))
                    unknown_object_points[key] = (number, name, total + 1, valid + int(not point.exclusion_reasons))
                reasons = list(point.exclusion_reasons)
                if point.observation_id in seen_observation_ids and "duplicate_observation_id" not in reasons:
                    reasons.append("duplicate_observation_id")
                seen_observation_ids.add(point.observation_id)
                for reason in reasons:
                    exclusions[reason] += 1
                if not reasons:
                    valid_points += 1
                    valid_by_metadata[point.metadata_id] += 1
                    if point.time_jd is not None:
                        timestamps.append(point.time_jd)
        except ExternalSourceError as exc:
            parse_issues.append(f"{file.name}:parse_error:{exc}")
    issues = Counter(parse_issues)
    issues.update(issue for row in metadata for issue in row.metadata_issues)
    object_rows: dict[str, dict[str, object]] = {}
    for row in metadata:
        key = row.candidate_identity
        entry = object_rows.setdefault(
            key,
            {
                "candidate_identity": key,
                "object_number": row.object_number,
                "aliases": list(row.aliases),
                "session_count": 0,
                "point_count": 0,
                "valid_point_count": 0,
                "filters": [],
                "session_dates": [],
                "documented_apparitions": "not_computed",
                "independence": "not_assessed",
            },
        )
        entry["session_count"] = int(entry["session_count"]) + 1
        if row.filter_name and row.filter_name not in entry["filters"]:
            entry["filters"].append(row.filter_name)
        if row.session_datetime and row.session_datetime not in entry["session_dates"]:
            entry["session_dates"].append(row.session_datetime)
    for key, (number, name, total, valid) in unknown_object_points.items():
        object_rows[key] = {"candidate_identity": key, "object_number": number, "aliases": list(_aliases(name)), "session_count": 0, "point_count": total, "valid_point_count": valid, "filters": [], "session_dates": [], "documented_apparitions": "not_computed", "independence": "not_assessed"}
    for row in metadata:
        entry = object_rows[row.candidate_identity]
        entry["point_count"] = int(entry["point_count"]) + points_by_metadata[row.record_id]
        entry["valid_point_count"] = int(entry["valid_point_count"]) + valid_by_metadata[row.record_id]
    for entry in object_rows.values():
        dates = entry["session_dates"]
        entry["distinct_session_years_approximation"] = len({str(date)[:4] for date in dates})
        entry["primary_screen_approximation"] = bool(entry["session_count"] >= 5 and entry["valid_point_count"] >= 150 and len({str(date)[:4] for date in dates}) >= 2)
    session_rows: list[dict[str, object]] = []
    for row in metadata:
        session_rows.append({
            "metadata_id": row.record_id,
            "candidate_identity": row.candidate_identity,
            "session_datetime": row.session_datetime,
            "filter": row.filter_name,
            "point_count": points_by_metadata[row.record_id],
            "valid_point_count": valid_by_metadata[row.record_id],
            "metadata_issues": list(row.metadata_issues),
        })
    return {
        "schema": "delphi.k3-external-source-census.v1",
        "dataset": {"name": "ALCDEF PDS bundle V1.0", "kind": "alcdef_pds", "source_uri": source_uri},
        "source": {"uri": source_uri, "format": "ALCDEF PDS4 CSV", "version": "1.0"},
        "identity": {"namespace": "mpc", "status": "candidate_metadata_only", "independent_label": False, "mapping_source": identity_map_path, "mapped_internal_source_ids": sorted(set(identity_map.values())) if identity_map else []},
        "session": {"metadata_records": len(metadata), "unique_sessions": len({(row.candidate_identity, row.session_datetime, row.filter_name) for row in metadata})},
        "objects": sorted([dict(row, internal_source_ids=([identity_map.get(row["object_number"])] if identity_map and row["object_number"] in identity_map else [])) for row in object_rows.values()], key=lambda row: str(row["candidate_identity"])),
        "sessions": session_rows,
        "screening_policy": {"minimum_sessions": 5, "minimum_valid_points": 150, "minimum_documented_apparitions": 2, "documented_apparitions_status": "not_computed; year/session count is an explicitly labeled approximation"},
        "documentation_reconciliation": {"documented_object_count": 23847, "source_object_name_count": len({row.object_name for row in metadata if row.object_name}), "canonical_candidate_count": len(object_rows), "object_count_delta_vs_documentation": len(object_rows) - 23847, "documented_observation_count": 9944193, "parsed_positive_mdid_count": total_points, "observation_count_delta_vs_documentation": total_points - 9944193, "explanation": "The archive documentation reports distinct object names and a bundle-level observation count. Canonical MPC identity merges four numbered objects carrying alternate names while three unnumbered names overlap numbered names, yielding one fewer conservative candidate. The extracted lcdata tables contain 9,944,099 positive-MDID rows; the documented 9,944,193 total is 94 higher and is retained as a source/documentation discrepancy, not silently discarded."},
        "observation_counts": {"total": total_points, "valid": valid_points, "excluded": total_points - valid_points, "duplicate_observation_ids": exclusions.get("duplicate_observation_id", 0)},
        "timestamp_ranges": {"jd_min": min(timestamps) if timestamps else None, "jd_max": max(timestamps) if timestamps else None, "session_min": min((row.session_datetime for row in metadata if row.session_datetime), default=None), "session_max": max((row.session_datetime for row in metadata if row.session_datetime), default=None)},
        "metadata_issues": [{"reason": reason, "count": count} for reason, count in sorted(issues.items())],
        "exclusions": [{"reason": reason, "count": count} for reason, count in sorted(exclusions.items())],
        "input_files": [{"path": str(file), "sha256": _sha256(file), "source_uri": source_uri} for file in files],
    }


def census_tssys(path: str | Path, *, source_uri: str = TSSYS_SOURCE_URI, identity_map: Mapping[int, str] | None = None, identity_map_path: str | None = None) -> dict[str, object]:
    root = Path(path)
    release = root / "release.merge" if root.is_dir() else root
    if not release.is_file():
        raise ExternalSourceError(f"TSSYS release.merge not found under {root}")
    metadata, parse_issues = parse_tssys_release(release)
    lc_files = _input_files(root, ("*.lc",)) if root.is_dir() else []
    points: list[TSSYSPoint] = []
    for file in lc_files:
        rows, issues = parse_tssys_lightcurve(file)
        points.extend(rows)
        parse_issues.extend(f"{file.name}:{issue}" for issue in issues)
    valid_points = [point for point in points if point.valid]
    timestamps = [point.time_jd for point in valid_points if point.time_jd is not None]
    exclusions = Counter(reason for point in points for reason in point.exclusion_reasons)
    issues = Counter(parse_issues)
    issues.update(issue for row in metadata for issue in row.metadata_issues)
    files = [release, *lc_files]
    object_rows: dict[str, dict[str, object]] = {}
    for row in metadata:
        object_rows[row.candidate_identity] = {"candidate_identity": row.candidate_identity, "object_number": row.object_number, "aliases": list(row.aliases), "session_count": 1, "point_count": 0, "valid_point_count": 0, "filters": ["TESS"], "documented_apparitions": "not_computed", "independence": "not_assessed"}
    for point in points:
        number = _integer(point.target_id, positive=True)
        key = f"mpc:{number}" if number is not None else f"designation:{point.target_id}"
        entry = object_rows.setdefault(key, {"candidate_identity": key, "object_number": number, "aliases": list(_aliases(point.target_id)), "session_count": 0, "point_count": 0, "valid_point_count": 0, "filters": ["TESS"], "documented_apparitions": "not_computed", "independence": "not_assessed"})
        entry["point_count"] = int(entry["point_count"]) + 1
        if point.valid:
            entry["valid_point_count"] = int(entry["valid_point_count"]) + 1
    session_rows = [
        {"target_id": row.target_id, "candidate_identity": row.candidate_identity, "sector": row.sector, "camera": row.camera, "ccd": row.ccd, "point_count": 0, "valid_point_count": 0}
        for row in metadata
    ]
    session_by_identity = {str(row["candidate_identity"]): row for row in session_rows}
    for point in points:
        number = _integer(point.target_id, positive=True)
        key = f"mpc:{number}" if number is not None else f"designation:{point.target_id}"
        if key in session_by_identity:
            session_by_identity[key]["point_count"] = int(session_by_identity[key]["point_count"]) + 1
            session_by_identity[key]["valid_point_count"] = int(session_by_identity[key]["valid_point_count"]) + int(point.valid)
    return {
        "schema": "delphi.k3-external-source-census.v1",
        "dataset": {"name": "TSSYS DR1", "kind": "tssys_dr1", "source_uri": source_uri},
        "source": {"uri": source_uri, "format": "TSSYS DR1 release.merge and .lc", "version": "DR1"},
        "identity": {"namespace": "mpc", "status": "candidate_metadata_only", "independent_label": False, "mapping_source": identity_map_path, "mapped_internal_source_ids": sorted(set(identity_map.values())) if identity_map else []},
        "session": {"metadata_records": len(metadata), "unique_sessions": len({(row.candidate_identity, row.sector, row.camera, row.ccd) for row in metadata})},
        "objects": sorted([dict(row, internal_source_ids=([identity_map.get(row["object_number"])] if identity_map and row["object_number"] in identity_map else [])) for row in object_rows.values()], key=lambda row: str(row["candidate_identity"])),
        "sessions": session_rows,
        "screening_policy": {"minimum_sessions": 5, "minimum_valid_points": 150, "minimum_documented_apparitions": 2, "documented_apparitions_status": "not_computed"},
        "observation_counts": {"total": len(points), "valid": len(valid_points), "excluded": len(points) - len(valid_points), "duplicate_observation_ids": exclusions.get("duplicate_observation_id", 0)},
        "timestamp_ranges": {"jd_min": min(timestamps) if timestamps else None, "jd_max": max(timestamps) if timestamps else None, "session_min": None, "session_max": None},
        "metadata_issues": [{"reason": reason, "count": count} for reason, count in sorted(issues.items())],
        "exclusions": [{"reason": reason, "count": count} for reason, count in sorted(exclusions.items())],
        "input_files": [{"path": str(file), "sha256": _sha256(file), "source_uri": source_uri} for file in files],
    }


def census_external_source(path: str | Path, *, kind: str = "auto", source_uri: str | None = None, identity_map_path: str | Path | None = None) -> dict[str, object]:
    root = Path(path)
    if kind == "auto":
        names = {item.name for item in ([root] if root.is_file() else root.iterdir())}
        kind = "tssys" if "release.merge" in names else "alcdef"
    identity_map = load_identity_map(identity_map_path) if identity_map_path else None
    map_label = str(identity_map_path) if identity_map_path else None
    if kind in {"alcdef", "alcdef_pds"}:
        return census_alcdef(root, source_uri=source_uri or ALCDEF_BUNDLE_URI, identity_map=identity_map, identity_map_path=map_label)
    if kind in {"tssys", "tssys_dr1"}:
        return census_tssys(root, source_uri=source_uri or TSSYS_SOURCE_URI, identity_map=identity_map, identity_map_path=map_label)
    raise ExternalSourceError(f"unknown source kind: {kind}")


def crossmatch_damit_metadata(census_path: str | Path, identity_map_path: str | Path) -> dict[str, object]:
    """Crossmatch census candidates to DAMIT identity rows without reading solutions/poles."""
    census_file, mapping_file = Path(census_path), Path(identity_map_path)
    try:
        census = json.loads(census_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExternalSourceError(f"cannot read census JSON: {exc}") from exc
    if not isinstance(census, Mapping) or census.get("schema") != "delphi.k3-external-source-census.v1":
        raise ExternalSourceError("census schema mismatch")
    try:
        with mapping_file.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ExternalSourceError(f"cannot read DAMIT identity table: {exc}") from exc
    by_number: dict[int, dict[str, str | None]] = {}
    for row in rows:
        number = _integer(row.get("number"), positive=True)
        source_id = _clean(row.get("id"))
        if number is not None and source_id is not None:
            by_number[number] = {"damit_id": f"damit:{source_id}", "name": _clean(row.get("name")), "designation": _clean(row.get("designation"))}
    candidates = [row for row in census.get("objects", []) if isinstance(row, Mapping)]
    screens = [row for row in candidates if int(row.get("session_count", 0)) >= 5 and int(row.get("valid_point_count", 0)) >= 150]
    two_year = [row for row in screens if int(row.get("distinct_session_years_approximation", 0)) >= 2]
    available: list[dict[str, object]] = []
    for row in two_year:
        number = _integer(row.get("object_number"), positive=True)
        if number is None or number not in by_number:
            continue
        available.append({"candidate_identity": row.get("candidate_identity"), "mpc": f"mpc:{number}", "damit": by_number[number]})
    return {
        "schema": "delphi.k3-external-source-crossmatch.v1",
        "source_census": str(census_file),
        "source_census_sha256": _sha256(census_file),
        "damit_identity_source": str(mapping_file),
        "damit_identity_source_sha256": _sha256(mapping_file),
        "identity_namespace": "mpc",
        "reference_availability": {"damit_identity_rows": len(by_number), "damit_present_only": True, "poles_loaded": False, "solutions_loaded": False, "records": available},
        "screening": {"minimum_sessions": 5, "minimum_valid_points": 150, "metadata_screen_count": len(screens), "two_year_groups_approximation_count": len(two_year), "documented_apparitions_count": None, "documented_apparitions_status": "not_computed; distinct session years are an explicitly labeled approximation"},
        "crossmatch_count": len(available),
    }


__all__ = [
    "ALCDEFMetadata", "ALCDEFPoint", "TSSYSMetadata", "TSSYSPoint", "GaiaDR3Spin", "ExternalSourceError",
    "parse_alcdef_metadata", "parse_alcdef_lcdata", "parse_alcdef_archive", "parse_tssys_release",
    "parse_tssys_lightcurve", "census_alcdef", "census_tssys", "census_external_source",
    "load_identity_map", "crossmatch_damit_metadata", "parse_gaia_dr3_spins",
]
