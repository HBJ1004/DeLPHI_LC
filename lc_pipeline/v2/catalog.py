"""Deterministic, same-object DAMIT catalog construction for DeLPHI V2.

The builder deliberately has no machine-learning dependencies.  It turns a
pinned DAMIT dump and an explicit membership list into a compact, hash-bound
catalog.  Labels are joined by normalized asteroid ID and model ID; positional
joins and directory-order joins are impossible by construction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..physics.directional import directed_angular_error_deg_scalar

CATALOG_SCHEMA = "delphi.data-catalog.v2"
SPLIT_SCHEMA = "delphi.grouped-splits.v2"
DEVELOPMENT_SPLIT_SCHEMA = "delphi.development-split.v2"
PINNED_DUMP_ID = "damit-20250610T000301Z"
COORDINATE_FRAME = "ecliptic_j2000"
DIRECTIONALITY = "directed_spin_vector"
DEFAULT_MIN_QUALITY_FLAG = 3.0
DEFAULT_SPLIT_SEED = 20260901


class CatalogError(ValueError):
    """Raised when source data violate a fail-closed catalog contract."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 for *path*."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_object_id(value: str | int) -> str:
    """Normalize supported DAMIT identifiers to ``asteroid_<integer>``."""
    text = str(value).strip().lower()
    if text.startswith("asteroid_"):
        text = text[len("asteroid_") :]
    if not text or not text.isdigit():
        raise CatalogError(f"Invalid DAMIT object identifier: {value!r}")
    number = int(text)
    if number <= 0:
        raise CatalogError(f"DAMIT object identifier must be positive: {value!r}")
    return f"asteroid_{number}"


def object_number(value: str | int) -> int:
    return int(normalize_object_id(value).split("_", 1)[1])


def ecliptic_vector(lambda_deg: float, beta_deg: float) -> tuple[float, float, float]:
    """Convert ecliptic longitude/latitude to a directed Cartesian unit vector."""
    if not (math.isfinite(lambda_deg) and math.isfinite(beta_deg)):
        raise CatalogError("Pole coordinates must be finite")
    if not -90.0 <= beta_deg <= 90.0:
        raise CatalogError(f"Ecliptic latitude outside [-90, 90]: {beta_deg}")
    lam = math.radians(lambda_deg % 360.0)
    beta = math.radians(beta_deg)
    cos_beta = math.cos(beta)
    vector = (cos_beta * math.cos(lam), cos_beta * math.sin(lam), math.sin(beta))
    norm = math.sqrt(sum(component * component for component in vector))
    if not math.isfinite(norm) or norm <= 0.0:
        raise CatalogError("Pole conversion produced an invalid vector")
    return tuple(component / norm for component in vector)


@dataclass(frozen=True)
class SpinRecord:
    lambda_deg: float
    beta_deg: float
    period_hours: float


def parse_spin(path: Path) -> SpinRecord:
    """Parse and validate the first DAMIT spin-file record."""
    tokens = Path(path).read_text(encoding="utf-8").split()
    if len(tokens) < 3:
        raise CatalogError(f"Malformed spin file: {path}")
    try:
        lambda_deg, beta_deg, period_hours = (float(token) for token in tokens[:3])
    except ValueError as exc:
        raise CatalogError(f"Non-numeric spin record: {path}") from exc
    if not all(math.isfinite(value) for value in (lambda_deg, beta_deg, period_hours)):
        raise CatalogError(f"Non-finite spin record: {path}")
    if period_hours <= 0:
        raise CatalogError(f"Non-positive spin period: {path}")
    ecliptic_vector(lambda_deg, beta_deg)
    return SpinRecord(lambda_deg=lambda_deg, beta_deg=beta_deg, period_hours=period_hours)


def audit_lightcurve_file(path: Path) -> dict[str, Any]:
    """Parse a DAMIT ``lc.txt`` and return validation statistics.

    Epoch boundaries in the source file are retained.  Geometry is required to
    be finite and nonzero for every observation; V2 never silently substitutes
    zero geometry.
    """
    tokens = Path(path).read_text(encoding="utf-8").split()
    if not tokens:
        raise CatalogError(f"Empty lightcurve file: {path}")
    position = 0
    try:
        n_epochs = int(tokens[position])
    except (ValueError, IndexError) as exc:
        raise CatalogError(f"Malformed lightcurve header: {path}") from exc
    position += 1
    if n_epochs <= 0:
        raise CatalogError(f"Lightcurve has no epochs: {path}")

    epoch_sizes: list[int] = []
    calibrated_flags: list[bool] = []
    n_observations = 0
    min_sun_norm = math.inf
    min_observer_norm = math.inf
    first_jd = math.inf
    last_jd = -math.inf
    epochs_requiring_stable_sort = 0
    out_of_order_steps = 0

    for epoch_index in range(n_epochs):
        try:
            n_points = int(tokens[position])
            calibrated = int(tokens[position + 1])
        except (ValueError, IndexError) as exc:
            raise CatalogError(f"Malformed epoch {epoch_index} in {path}") from exc
        position += 2
        if n_points <= 0 or calibrated not in (0, 1):
            raise CatalogError(f"Invalid epoch header at epoch {epoch_index} in {path}")
        required = n_points * 8
        if position + required > len(tokens):
            raise CatalogError(f"Truncated epoch {epoch_index} in {path}")
        previous_jd = -math.inf
        epoch_out_of_order = False
        for row_index in range(n_points):
            start = position + row_index * 8
            try:
                row = [float(token) for token in tokens[start : start + 8]]
            except ValueError as exc:
                raise CatalogError(
                    f"Non-numeric observation {row_index} in epoch {epoch_index}: {path}"
                ) from exc
            if len(row) != 8 or not all(math.isfinite(value) for value in row):
                raise CatalogError(
                    f"Invalid observation {row_index} in epoch {epoch_index}: {path}"
                )
            jd = row[0]
            if jd < previous_jd:
                # DAMIT preserves source epoch membership but does not promise
                # row order.  V2's canonical preprocessor applies a stable sort
                # within the supplied epoch.  Record this transformation rather
                # than silently rejecting otherwise valid observations.
                epoch_out_of_order = True
                out_of_order_steps += 1
            previous_jd = jd
            sun_norm = math.sqrt(sum(value * value for value in row[2:5]))
            observer_norm = math.sqrt(sum(value * value for value in row[5:8]))
            if sun_norm <= 1e-12 or observer_norm <= 1e-12:
                raise CatalogError(f"Zero geometry at epoch {epoch_index}, row {row_index}: {path}")
            min_sun_norm = min(min_sun_norm, sun_norm)
            min_observer_norm = min(min_observer_norm, observer_norm)
            first_jd = min(first_jd, jd)
            last_jd = max(last_jd, jd)
        position += required
        if epoch_out_of_order:
            epochs_requiring_stable_sort += 1
        epoch_sizes.append(n_points)
        calibrated_flags.append(bool(calibrated))
        n_observations += n_points

    if position != len(tokens):
        raise CatalogError(f"Unexpected trailing tokens in {path}: {len(tokens) - position}")
    return {
        "n_epochs": n_epochs,
        "n_observations": n_observations,
        "epoch_sizes": epoch_sizes,
        "calibrated_flags": calibrated_flags,
        "first_jd": first_jd,
        "last_jd": last_jd,
        "epochs_requiring_stable_sort": epochs_requiring_stable_sort,
        "out_of_order_steps": out_of_order_steps,
        "min_sun_distance_au": min_sun_norm,
        "min_observer_distance_au": min_observer_norm,
    }


def load_membership(path: Path) -> list[str]:
    """Load membership from JSON, JSONL, text, or a legacy fold directory.

    A fold directory contributes only the union of ``fold*_test.json``.  Its old
    assignments are not reused; V2 creates new grouped splits.
    """
    source = Path(path)
    values: list[Any] = []
    if source.is_dir():
        files = sorted(source.glob("fold*_test.json"))
        if not files:
            raise CatalogError(f"No fold*_test.json membership files in {source}")
        for item in files:
            payload = json.loads(item.read_text(encoding="utf-8"))
            if not isinstance(payload, list):
                raise CatalogError(f"Membership file is not a list: {item}")
            values.extend(payload)
    elif source.suffix == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise CatalogError(f"Membership JSON is not a list: {source}")
        values.extend(payload)
    elif source.suffix == ".jsonl":
        for line in source.read_text(encoding="utf-8").splitlines():
            if line.strip():
                payload = json.loads(line)
                values.append(payload.get("object_id") if isinstance(payload, dict) else payload)
    else:
        values.extend(line.strip() for line in source.read_text().splitlines() if line.strip())

    normalized = [normalize_object_id(value) for value in values]
    if len(normalized) != len(set(normalized)):
        # A legacy CV directory repeats members in train files but not in the
        # union of test files.  Duplicate test membership is always an error.
        duplicates = sorted({item for item in normalized if normalized.count(item) > 1})
        raise CatalogError(f"Duplicate object membership: {duplicates[:10]}")
    return sorted(normalized, key=object_number)


def _load_model_table(path: Path) -> tuple[dict[int, list[dict[str, str]]], list[str]]:
    by_object: dict[int, list[dict[str, str]]] = defaultdict(list)
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        required = {"id", "asteroid_id", "lambda", "beta", "period", "quality_flag"}
        missing = sorted(required - set(fieldnames))
        if missing:
            raise CatalogError(f"Model table missing columns: {missing}")
        for row in reader:
            try:
                asteroid_id = int(row["asteroid_id"])
                int(row["id"])
            except (TypeError, ValueError) as exc:
                raise CatalogError("Model table contains a non-integer ID") from exc
            by_object[asteroid_id].append(row)
    for rows in by_object.values():
        rows.sort(key=lambda row: int(row["id"]))
    return dict(by_object), fieldnames


def _solution_set_hash(solutions: Sequence[Mapping[str, Any]]) -> str:
    directed_vectors = sorted(
        [[float(f"{component:.17g}") for component in solution["vector"]] for solution in solutions]
    )
    return _sha256_bytes(_canonical_json(directed_vectors).encode("ascii"))


def _relative_source_path(dump_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(dump_root.resolve()).as_posix()
    except ValueError as exc:
        raise CatalogError(f"Source escaped the pinned dump: {path}") from exc


def build_catalog(
    dump_root: Path,
    object_ids: Sequence[str | int],
    *,
    min_quality_flag: float = DEFAULT_MIN_QUALITY_FLAG,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Build a deterministic catalog and its source/collision audit records."""
    root = Path(dump_root)
    if root.name != PINNED_DUMP_ID:
        raise CatalogError(f"Expected pinned dump directory {PINNED_DUMP_ID!r}, got {root.name!r}")
    if not math.isfinite(min_quality_flag):
        raise CatalogError("Minimum quality flag must be finite")
    normalized_ids = [normalize_object_id(value) for value in object_ids]
    if len(normalized_ids) != len(set(normalized_ids)):
        raise CatalogError("Object membership contains duplicates")
    normalized_ids.sort(key=object_number)

    table_path = root / "tables" / "asteroid_models.csv"
    if not table_path.is_file():
        raise CatalogError(f"Missing DAMIT model table: {table_path}")
    rows_by_object, _ = _load_model_table(table_path)
    source_files: list[dict[str, Any]] = [
        {
            "role": "model_table",
            "path": _relative_source_path(root, table_path),
            "bytes": table_path.stat().st_size,
            "sha256": sha256_file(table_path),
        }
    ]
    records: list[dict[str, Any]] = []

    for object_id in normalized_ids:
        asteroid_id = object_number(object_id)
        asteroid_dir = root / "files" / object_id
        lc_path = asteroid_dir / "lc.txt"
        quarantine_reasons: list[str] = []
        lc_audit: dict[str, Any] | None = None
        lc_sha: str | None = None
        if not lc_path.is_file():
            quarantine_reasons.append("missing_lightcurve")
        else:
            try:
                lc_audit = audit_lightcurve_file(lc_path)
                lc_sha = sha256_file(lc_path)
                source_files.append(
                    {
                        "object_id": object_id,
                        "role": "observed_lightcurves",
                        "path": _relative_source_path(root, lc_path),
                        "bytes": lc_path.stat().st_size,
                        "sha256": lc_sha,
                    }
                )
            except CatalogError as exc:
                quarantine_reasons.append(f"invalid_lightcurve:{exc}")

        all_rows = rows_by_object.get(asteroid_id, [])
        qualifying: list[dict[str, str]] = []
        for row in all_rows:
            raw_quality = (row.get("quality_flag") or "").strip()
            try:
                quality = float(raw_quality)
            except ValueError:
                quality = math.nan
            if math.isfinite(quality) and quality >= min_quality_flag:
                qualifying.append(row)
        if not qualifying:
            quarantine_reasons.append("no_model_at_minimum_quality")

        solutions: list[dict[str, Any]] = []
        for row in qualifying:
            model_id = int(row["id"])
            spin_path = asteroid_dir / f"model_{model_id}" / "spin.txt"
            if not spin_path.is_file():
                quarantine_reasons.append(f"missing_spin:model_{model_id}")
                continue
            try:
                spin = parse_spin(spin_path)
                table_values = (
                    float(row["lambda"]),
                    float(row["beta"]),
                    float(row["period"]),
                )
            except (CatalogError, ValueError) as exc:
                quarantine_reasons.append(f"invalid_spin:model_{model_id}:{exc}")
                continue
            spin_values = (spin.lambda_deg, spin.beta_deg, spin.period_hours)
            if any(abs(left - right) > 1e-9 for left, right in zip(spin_values, table_values)):
                quarantine_reasons.append(f"spin_table_mismatch:model_{model_id}")
                continue
            vector = ecliptic_vector(spin.lambda_deg, spin.beta_deg)
            spin_sha = sha256_file(spin_path)
            source_files.append(
                {
                    "object_id": object_id,
                    "model_id": model_id,
                    "role": "spin_solution",
                    "path": _relative_source_path(root, spin_path),
                    "bytes": spin_path.stat().st_size,
                    "sha256": spin_sha,
                }
            )
            solutions.append(
                {
                    "model_id": model_id,
                    "quality_flag": float(row["quality_flag"]),
                    "lambda_deg": spin.lambda_deg % 360.0,
                    "beta_deg": spin.beta_deg,
                    "period_hours": spin.period_hours,
                    "vector": list(vector),
                    "coordinate_frame": COORDINATE_FRAME,
                    "directionality": DIRECTIONALITY,
                    "source_path": _relative_source_path(root, spin_path),
                    "source_sha256": spin_sha,
                }
            )
        if qualifying and len(solutions) != len(qualifying):
            quarantine_reasons.append("incomplete_qualifying_solution_set")

        eligible = not quarantine_reasons and bool(solutions) and lc_audit is not None
        record: dict[str, Any] = {
            "object_id": object_id,
            "source_asteroid_id": asteroid_id,
            "eligible": eligible,
            "quarantine_reasons": sorted(set(quarantine_reasons)),
            "coordinate_frame": COORDINATE_FRAME,
            "directionality": DIRECTIONALITY,
            "lightcurve": None
            if lc_audit is None
            else {
                **lc_audit,
                "source_path": _relative_source_path(root, lc_path),
                "source_sha256": lc_sha,
            },
            "solutions": solutions,
            "directed_solution_set_sha256": _solution_set_hash(solutions) if solutions else None,
        }
        records.append(record)

    source_files.sort(key=lambda item: (item.get("object_id", ""), item["role"], item["path"]))
    collisions = audit_label_collisions(records)
    eligible_records = [record for record in records if record["eligible"]]
    quarantined_records = [record for record in records if not record["eligible"]]
    manifest = {
        "schema": CATALOG_SCHEMA,
        "dump_id": PINNED_DUMP_ID,
        "source_artifact": {
            "repository": "DAMIT",
            "immutable_uri": None,
            "release_blocked_until_immutable_uri": True,
        },
        "coordinate_frame": COORDINATE_FRAME,
        "directionality": DIRECTIONALITY,
        "selection": {
            "membership": "explicit_object_id_list",
            "label_join": "same_normalized_asteroid_id_and_model_id",
            "minimum_quality_flag": min_quality_flag,
            "retain_all_explicit_qualifying_solutions": True,
        },
        "counts": {
            "requested": len(records),
            "eligible": len(eligible_records),
            "quarantined": len(quarantined_records),
            "explicit_solutions": sum(len(record["solutions"]) for record in eligible_records),
        },
        "requested_object_ids_sha256": _sha256_bytes(
            ("\n".join(normalized_ids) + "\n").encode("ascii")
        ),
        "eligible_object_ids_sha256": _sha256_bytes(
            ("\n".join(record["object_id"] for record in eligible_records) + "\n").encode("ascii")
        ),
        "model_table_sha256": sha256_file(table_path),
        "source_file_manifest_sha256": _sha256_bytes(
            ("\n".join(_canonical_json(item) for item in source_files) + "\n").encode("ascii")
        ),
        "catalog_records_sha256": _sha256_bytes(
            ("\n".join(_canonical_json(item) for item in records) + "\n").encode("ascii")
        ),
        "collision_audit_sha256": _sha256_bytes(_canonical_json(collisions).encode("ascii")),
        "release_eligible": False,
        "release_blockers": [
            "source_artifact_immutable_uri_missing",
            "independent_custodian_review_pending",
            "prospective_test_pending",
        ],
    }
    return manifest, records, source_files, collisions


def audit_label_collisions(
    records: Sequence[Mapping[str, Any]], *, near_tolerance_deg: float = 1e-4
) -> dict[str, Any]:
    """Audit exact solution sets and cross-object near-equal individual poles."""
    eligible = [record for record in records if record.get("eligible")]
    by_set_hash: dict[str, list[str]] = defaultdict(list)
    poles: list[tuple[str, Mapping[str, Any], str | None]] = []
    for record in eligible:
        by_set_hash[str(record["directed_solution_set_sha256"])].append(str(record["object_id"]))
        lc_hash = record.get("lightcurve", {}).get("source_sha256")
        for solution in record.get("solutions", []):
            poles.append((str(record["object_id"]), solution, lc_hash))

    duplicate_sets = [
        {"solution_set_sha256": digest, "object_ids": sorted(ids)}
        for digest, ids in sorted(by_set_hash.items())
        if len(ids) > 1
    ]
    near_collisions: list[dict[str, Any]] = []
    for index, (left_id, left_solution, left_lc_hash) in enumerate(poles):
        for right_id, right_solution, right_lc_hash in poles[index + 1 :]:
            if left_id == right_id:
                continue
            try:
                angle = directed_angular_error_deg_scalar(
                    left_solution["vector"], right_solution["vector"]
                )
            except (TypeError, ValueError) as exc:
                raise CatalogError("Collision audit received an invalid pole vector") from exc
            if angle <= near_tolerance_deg:
                independent = (
                    left_solution["model_id"] != right_solution["model_id"]
                    and left_lc_hash != right_lc_hash
                )
                near_collisions.append(
                    {
                        "left_object_id": left_id,
                        "left_model_id": left_solution["model_id"],
                        "right_object_id": right_id,
                        "right_model_id": right_solution["model_id"],
                        "directed_angle_deg": angle,
                        "resolution": (
                            "independent_source_lineage"
                            if independent
                            else "unresolved_lineage_collision"
                        ),
                    }
                )
    unresolved = [
        item for item in near_collisions if item["resolution"] == "unresolved_lineage_collision"
    ]
    return {
        "schema": "delphi.label-collision-audit.v2",
        "near_tolerance_deg": near_tolerance_deg,
        "exact_cross_object_solution_sets": duplicate_sets,
        "near_cross_object_solutions": near_collisions,
        "unresolved_count": len(duplicate_sets) + len(unresolved),
        "passed": not duplicate_sets and not unresolved,
    }


def _stable_order(object_ids: Iterable[str], seed: int, namespace: str) -> list[str]:
    return sorted(
        object_ids,
        key=lambda object_id: hashlib.sha256(
            f"{seed}:{namespace}:{object_id}".encode("ascii")
        ).hexdigest(),
    )


def build_grouped_splits(
    records: Sequence[Mapping[str, Any]],
    *,
    n_outer_folds: int = 5,
    seed: int = DEFAULT_SPLIT_SEED,
    validation_fraction: float = 0.10,
    calibration_fraction: float = 0.10,
) -> dict[str, Any]:
    """Create frozen balanced outer folds plus disjoint inner partitions.

    Each object is an indivisible group.  Ordering is determined by SHA-256,
    not Python's process-randomized hash.  The historical fold assignment is not
    reused.
    """
    if n_outer_folds < 2:
        raise CatalogError("At least two outer folds are required")
    if validation_fraction <= 0 or calibration_fraction <= 0:
        raise CatalogError("Validation and calibration fractions must be positive")
    if validation_fraction + calibration_fraction >= 0.5:
        raise CatalogError("Inner holdouts must leave at least half the remainder for training")
    eligible = sorted(str(record["object_id"]) for record in records if record.get("eligible"))
    if len(eligible) < n_outer_folds * 3:
        raise CatalogError("Too few eligible objects for grouped outer/inner splits")

    outer_order = _stable_order(eligible, seed, "outer")
    outer_tests: list[list[str]] = [[] for _ in range(n_outer_folds)]
    for index, object_id in enumerate(outer_order):
        outer_tests[index % n_outer_folds].append(object_id)

    folds: list[dict[str, Any]] = []
    all_ids = set(eligible)
    for fold_index, test_ids in enumerate(outer_tests):
        test_set = set(test_ids)
        remainder = _stable_order(all_ids - test_set, seed, f"inner:{fold_index}")
        n_remainder = len(remainder)
        n_validation = max(1, round(n_remainder * validation_fraction))
        n_calibration = max(1, round(n_remainder * calibration_fraction))
        validation_ids = sorted(remainder[:n_validation], key=object_number)
        calibration_ids = sorted(
            remainder[n_validation : n_validation + n_calibration], key=object_number
        )
        train_ids = sorted(remainder[n_validation + n_calibration :], key=object_number)
        fold = {
            "fold": fold_index,
            "train_ids": train_ids,
            "validation_ids": validation_ids,
            "calibration_ids": calibration_ids,
            "test_ids": sorted(test_ids, key=object_number),
        }
        fold["sha256"] = _sha256_bytes(_canonical_json(fold).encode("ascii"))
        folds.append(fold)

    audit = audit_grouped_splits(records, folds)
    payload = {
        "schema": SPLIT_SCHEMA,
        "seed": seed,
        "n_outer_folds": n_outer_folds,
        "group_key": "normalized_object_id",
        "inner_partitions": ["validation", "calibration"],
        "folds": folds,
        "audit": audit,
    }
    payload["sha256"] = _sha256_bytes(_canonical_json(payload).encode("ascii"))
    return payload


def build_development_split(
    records: Sequence[Mapping[str, Any]], *, seed: int = 20260902,
    validation_count: int = 25, calibration_count: int = 25,
) -> dict[str, Any]:
    """Freeze one disjoint train/validation/calibration development partition.

    This partition intentionally has no outer-test role.  It is for candidate
    selection only and cannot be promoted into a publication estimate.
    """
    if validation_count <= 0 or calibration_count <= 0:
        raise CatalogError("development holdout counts must be positive")
    eligible = sorted(str(record["object_id"]) for record in records if record.get("eligible"))
    if len(eligible) <= validation_count + calibration_count:
        raise CatalogError("development split leaves no training objects")
    order = _stable_order(eligible, seed, "development")
    validation_ids = sorted(order[:validation_count], key=object_number)
    calibration_ids = sorted(order[validation_count : validation_count + calibration_count], key=object_number)
    train_ids = sorted(order[validation_count + calibration_count :], key=object_number)
    payload = {
        "schema": DEVELOPMENT_SPLIT_SCHEMA,
        "seed": seed,
        "roles": {
            "train_ids": train_ids,
            "validation_ids": validation_ids,
            "calibration_ids": calibration_ids,
        },
        "purpose": "candidate_selection_only_no_outer_test",
    }
    payload["sha256"] = _sha256_bytes(_canonical_json(payload).encode("ascii"))
    return payload


def audit_grouped_splits(
    records: Sequence[Mapping[str, Any]], folds: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    eligible_records = {
        str(record["object_id"]): record for record in records if record.get("eligible")
    }
    eligible = set(eligible_records)
    errors: list[str] = []
    test_counts = {object_id: 0 for object_id in eligible}
    roles = ("train_ids", "validation_ids", "calibration_ids", "test_ids")

    for fold in folds:
        role_sets = {role: set(map(str, fold.get(role, []))) for role in roles}
        for role, values in role_sets.items():
            unknown = values - eligible
            if unknown:
                errors.append(f"fold {fold.get('fold')} {role} has unknown IDs: {sorted(unknown)}")
        for left_index, left_role in enumerate(roles):
            for right_role in roles[left_index + 1 :]:
                overlap = role_sets[left_role] & role_sets[right_role]
                if overlap:
                    errors.append(
                        f"fold {fold.get('fold')} overlap {left_role}/{right_role}: {sorted(overlap)}"
                    )
        union = set().union(*role_sets.values())
        if union != eligible:
            errors.append(f"fold {fold.get('fold')} does not cover the eligible catalog")
        for object_id in role_sets["test_ids"]:
            test_counts[object_id] += 1

        # Hash-level isolation guards accidental duplicate inputs/labels across
        # different IDs even when the string groups are disjoint.
        for hash_field, nested in (
            ("directed_solution_set_sha256", False),
            ("source_sha256", True),
        ):
            seen: dict[str, tuple[str, str]] = {}
            for role, values in role_sets.items():
                for object_id in values:
                    record = eligible_records[object_id]
                    digest = (
                        record.get("lightcurve", {}).get(hash_field)
                        if nested
                        else record.get(hash_field)
                    )
                    if not digest:
                        errors.append(f"{object_id} missing {hash_field}")
                        continue
                    previous = seen.get(str(digest))
                    if previous and previous[0] != object_id:
                        errors.append(
                            f"fold {fold.get('fold')} cross-object {hash_field} collision: "
                            f"{previous[0]}({previous[1]})/{object_id}({role})"
                        )
                    seen[str(digest)] = (object_id, role)

    wrong_test_counts = {object_id: count for object_id, count in test_counts.items() if count != 1}
    if wrong_test_counts:
        errors.append(f"outer-test coverage is not exactly once: {wrong_test_counts}")
    return {
        "passed": not errors,
        "errors": errors,
        "n_eligible": len(eligible),
        "outer_test_exactly_once": not wrong_test_counts,
        "object_overlap_count": sum(" overlap " in error for error in errors),
        "hash_collision_count": sum("collision" in error for error in errors),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(_canonical_json(row) + "\n" for row in rows), encoding="utf-8")


def write_catalog_bundle(
    output_dir: Path,
    manifest: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    source_files: Sequence[Mapping[str, Any]],
    collisions: Mapping[str, Any],
    splits: Mapping[str, Any],
) -> dict[str, str]:
    """Write deterministic compact evidence and return output SHA-256 values."""
    target = Path(output_dir)
    # This is a deterministic source-catalog audit, not a release data
    # manifest: its immutable URI, clean builder commit, and prospective
    # evidence do not exist yet. Naming it honestly prevents accidental use
    # with the stricter release-manifest validator.
    _write_json(target / "catalog_manifest.json", manifest)
    _write_jsonl(target / "catalog.jsonl", records)
    _write_jsonl(target / "source_files.jsonl", source_files)
    _write_json(target / "label_collision_audit.json", collisions)
    _write_json(target / "splits.json", splits)
    files = sorted(path for path in target.iterdir() if path.is_file())
    hashes = {path.name: sha256_file(path) for path in files}
    (target / "SHA256SUMS").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(hashes.items())),
        encoding="ascii",
    )
    return hashes


def build_cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-root", type=Path, required=True)
    parser.add_argument("--membership", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-quality-flag", type=float, default=DEFAULT_MIN_QUALITY_FLAG)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    args = parser.parse_args(argv)

    object_ids = load_membership(args.membership)
    manifest, records, source_files, collisions = build_catalog(
        args.dump_root,
        object_ids,
        min_quality_flag=args.minimum_quality_flag,
    )
    splits = build_grouped_splits(records, seed=args.split_seed)
    write_catalog_bundle(args.output_dir, manifest, records, source_files, collisions, splits)
    if not collisions["passed"] or not splits["audit"]["passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(build_cli())
