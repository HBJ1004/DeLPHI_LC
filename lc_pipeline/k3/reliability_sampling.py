"""Deterministic, label-blind sampling for the K3 reliability study.

The functions in this module operate on small immutable value objects.  They do
not infer any physical quantity and never use a label, pole, or measurement
error to decide which observations are retained.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..v2.preprocessing import Observation, ObservationEpoch

SEEDS = (20260915, 20260916, 20260917)
_POINT_FRACTIONS = (0.5, 0.25, 0.1, 0.05)
_OBSERVATION_CAPS = (10, 20, 50, 100, 200, 500, None)
_BLOCK_COUNTS = (1, 3, 5, 10, None)
_PER_BLOCK_CAPS = (5, 10, 20, 50, None)


class ReliabilitySamplingError(ValueError):
    """Raised for malformed native data or an invalid sampling condition."""


@dataclass(frozen=True)
class CurveObservation:
    """An observation retaining DAMIT's time, flux, and two geometry vectors."""

    time_jd: float
    flux: float
    sun: tuple[float, float, float]
    observer: tuple[float, float, float]

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.time_jd)) or float(self.time_jd) <= 0:
            raise ReliabilitySamplingError("observation time must be finite and positive")
        if not math.isfinite(float(self.flux)) or float(self.flux) <= 0:
            raise ReliabilitySamplingError("observation flux must be finite and positive")
        for name, vector in (("sun", self.sun), ("observer", self.observer)):
            values = tuple(float(v) for v in vector)
            if len(values) != 3 or not all(math.isfinite(v) for v in values):
                raise ReliabilitySamplingError(f"{name} geometry must contain three finite values")
            if math.sqrt(sum(v * v for v in values)) <= 1e-12:
                raise ReliabilitySamplingError(f"{name} geometry must be nonzero")
            object.__setattr__(self, name, values)
        object.__setattr__(self, "time_jd", float(self.time_jd))
        object.__setattr__(self, "flux", float(self.flux))

    @property
    def relative_brightness(self) -> float:
        """Compatibility spelling used by :class:`v2.preprocessing.Observation`."""
        return self.flux

    def as_observation(self) -> Observation:
        return Observation(self.time_jd, self.flux, self.sun, self.observer)


@dataclass(frozen=True)
class NativeSession:
    """One native DAMIT session, including both values from its header."""

    session_id: str
    observations: tuple[CurveObservation, ...]
    calibrated: int = 0
    header_count: int | None = None
    source_index: int = 0

    def __post_init__(self) -> None:
        values = tuple(self.observations)
        if not isinstance(self.session_id, str) or not self.session_id.strip():
            raise ReliabilitySamplingError("session_id must be nonempty")
        if self.calibrated not in (0, 1):
            raise ReliabilitySamplingError("session header absolute/relative flag must be 0 or 1")
        if any(not isinstance(item, CurveObservation) for item in values):
            raise ReliabilitySamplingError("session observations must be CurveObservation values")
        object.__setattr__(self, "observations", values)
        object.__setattr__(
            self,
            "header_count",
            len(values) if self.header_count is None else int(self.header_count),
        )

    @property
    def absolute_relative(self) -> int:
        return self.calibrated

    @property
    def epoch_id(self) -> str:
        return self.session_id

    @property
    def rows(self) -> np.ndarray:
        """A read-only native N×8 view (time, flux, Sun xyz, observer xyz)."""
        rows = np.asarray(
            [(item.time_jd, item.flux, *item.sun, *item.observer) for item in self.observations],
            dtype=np.float64,
        )
        rows.setflags(write=False)
        return rows

    @property
    def valid(self) -> bool:
        return len(self.observations) >= 2

    def as_epoch(self) -> ObservationEpoch:
        return ObservationEpoch(
            self.session_id, tuple(item.as_observation() for item in self.observations)
        )


@dataclass(frozen=True)
class ObservingBlock:
    """A connected interval of native sessions (a block is not an apparition)."""

    block_id: str
    sessions: tuple[NativeSession, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "sessions", tuple(self.sessions))

    @property
    def observations(self) -> tuple[CurveObservation, ...]:
        return tuple(item for session in self.sessions for item in session.observations)

    @property
    def start_jd(self) -> float:
        return min(item.time_jd for item in self.observations)

    @property
    def end_jd(self) -> float:
        return max(item.time_jd for item in self.observations)


@dataclass(frozen=True)
class LightCurve:
    """Immutable native curve and (optionally computed) merged observing blocks."""

    sessions: tuple[NativeSession, ...]
    object_id: str = ""
    period_hours: float | None = None
    blocks: tuple[ObservingBlock, ...] = ()

    def __post_init__(self) -> None:
        sessions = tuple(self.sessions)
        if any(not isinstance(item, NativeSession) for item in sessions):
            raise ReliabilitySamplingError("sessions must be NativeSession values")
        if self.period_hours is not None and (
            not math.isfinite(float(self.period_hours)) or self.period_hours <= 0
        ):
            raise ReliabilitySamplingError("period_hours must be finite and positive")
        object.__setattr__(self, "sessions", sessions)
        object.__setattr__(self, "blocks", tuple(self.blocks))

    @property
    def valid_sessions(self) -> tuple[NativeSession, ...]:
        return tuple(session for session in self.sessions if session.valid)

    @property
    def observations(self) -> tuple[CurveObservation, ...]:
        return tuple(item for session in self.valid_sessions for item in session.observations)

    @property
    def native_sessions(self) -> int:
        return len(self.valid_sessions)

    @property
    def total_observations(self) -> int:
        return len(self.observations)


@dataclass(frozen=True)
class Holdout:
    train: LightCurve
    withheld: LightCurve
    block: ObservingBlock

    @property
    def eligible(self) -> bool:
        return bool(self.withheld.observations)


def _read_rows(path: Path) -> tuple[NativeSession, ...]:
    try:
        tokens = path.read_text(encoding="ascii").split()
        count = int(tokens[0])
        if count < 1:
            raise ReliabilitySamplingError("lightcurve has no sessions")
        cursor = 1
        sessions: list[NativeSession] = []
        for index in range(count):
            n_points, flag = int(tokens[cursor]), int(tokens[cursor + 1])
            cursor += 2
            if n_points < 0 or flag not in (0, 1):
                raise ReliabilitySamplingError(f"invalid session header {index}")
            rows = tokens[cursor : cursor + 8 * n_points]
            if len(rows) != 8 * n_points:
                raise ReliabilitySamplingError(f"truncated session {index}")
            cursor += 8 * n_points
            observations: list[CurveObservation] = []
            for row_index in range(n_points):
                values = tuple(float(value) for value in rows[8 * row_index : 8 * row_index + 8])
                try:
                    observations.append(
                        CurveObservation(values[0], values[1], values[2:5], values[5:8])
                    )
                except (ValueError, TypeError, ReliabilitySamplingError):
                    # Invalid points are not silently made valid; they are excluded
                    # and native session validity is subsequently enforced.
                    continue
            sessions.append(
                NativeSession(f"session-{index:04d}", tuple(observations), flag, n_points, index)
            )
        if cursor != len(tokens):
            raise ReliabilitySamplingError("trailing tokens in lightcurve")
        return tuple(sessions)
    except (OSError, UnicodeDecodeError, IndexError, ValueError) as exc:
        if isinstance(exc, ReliabilitySamplingError):
            raise
        raise ReliabilitySamplingError(f"cannot parse lightcurve {path}: {exc}") from exc


def read_lc(
    path: str | Path, *, object_id: str | None = None, period_hours: float | None = None
) -> LightCurve:
    """Read native DAMIT ``lc.txt`` while preserving each session header."""
    return LightCurve(_read_rows(Path(path)), object_id or "", period_hours)


def read_native(path: str | Path) -> tuple[NativeSession, ...]:
    """Read only the native sessions (a convenient runner-facing adapter)."""
    return _read_rows(Path(path))


def write_lc(curve: LightCurve, path: str | Path) -> Path:
    """Write a curve in native DAMIT format without changing ``curve``."""
    target = Path(path)
    lines = [str(len(curve.sessions))]
    for session in curve.sessions:
        lines.append(f"{len(session.observations)} {session.calibrated}")
        for item in session.observations:
            values = (item.time_jd, item.flux, *item.sun, *item.observer)
            lines.append(" ".join(f"{value:.17g}" for value in values))
    target.write_text("\n".join(lines) + "\n", encoding="ascii")
    return target


def write_native(path: str | Path, sessions: Sequence[NativeSession] | LightCurve) -> Path:
    """Write native sessions; accepts either a session sequence or a curve."""
    curve = sessions if isinstance(sessions, LightCurve) else LightCurve(tuple(sessions))
    return write_lc(curve, path)


def to_epochs(sessions: Sequence[NativeSession] | LightCurve) -> tuple[ObservationEpoch, ...]:
    """Convert native sessions into the package's lossless epoch objects."""
    values = sessions.sessions if isinstance(sessions, LightCurve) else tuple(sessions)
    return tuple(session.as_epoch() for session in values if session.valid)


def merge_observing_blocks(curve: LightCurve, *, gap_days: float = 30.0) -> LightCurve:
    """Merge overlapping/nearby session intervals into connected blocks."""
    if gap_days < 0 or not math.isfinite(float(gap_days)):
        raise ReliabilitySamplingError("gap_days must be finite and nonnegative")
    sessions = tuple(
        sorted(
            curve.valid_sessions,
            key=lambda s: (min(o.time_jd for o in s.observations), s.source_index),
        )
    )
    blocks: list[ObservingBlock] = []
    current: list[NativeSession] = []
    current_end = -math.inf
    for session in sessions:
        start = min(item.time_jd for item in session.observations)
        end = max(item.time_jd for item in session.observations)
        if current and start - current_end >= gap_days:
            blocks.append(ObservingBlock(f"block-{len(blocks):04d}", tuple(current)))
            current = []
        current.append(session)
        current_end = max(current_end, end)
    if current:
        blocks.append(ObservingBlock(f"block-{len(blocks):04d}", tuple(current)))
    return replace(curve, blocks=tuple(blocks))


def make_condition_config(
    *,
    kind: str = "full",
    fraction: float | None = None,
    cap: int | None = None,
    block_count: int | None = None,
    per_block_cap: int | None = None,
    noise_sigma: float | None = None,
    seed: int = SEEDS[0],
    repeat: int = 0,
) -> dict[str, Any]:
    """Construct and validate a serialisable study condition dictionary."""
    value: dict[str, Any] = {"kind": kind, "seed": int(seed), "repeat": int(repeat)}
    if kind == "point":
        value["fraction"] = fraction
    elif kind in ("observation_cap", "session_cap"):
        value["cap"] = cap
    elif kind == "whole_session":
        value["fraction"] = fraction
    elif kind == "two_d":
        value.update(block_count=block_count, per_block_cap=per_block_cap)
    elif kind != "full":
        raise ReliabilitySamplingError(f"unknown condition kind: {kind}")
    if noise_sigma is not None:
        value["noise_sigma"] = float(noise_sigma)
    return _validate_condition(value)


def condition_configs() -> tuple[dict[str, Any], ...]:
    """Return the complete planned condition grid, in stable nested order."""

    def add(config: dict[str, Any], identifier: str) -> None:
        config["id"] = identifier
        config["family"] = config["kind"]
        result.append(config)

    result: list[dict[str, Any]] = []
    add(make_condition_config(kind="full"), "full")
    add(make_condition_config(kind="full", noise_sigma=0.03), "full_noise_0.03")
    add(make_condition_config(kind="full", noise_sigma=0.10), "full_noise_0.10")
    for fraction in _POINT_FRACTIONS:
        add(make_condition_config(kind="point", fraction=fraction), f"point_{fraction:g}")
    for fraction in (0.5, 0.25):
        add(
            make_condition_config(kind="whole_session", fraction=fraction),
            f"whole_session_{fraction:g}",
        )
    add(make_condition_config(kind="point", fraction=0.1, noise_sigma=0.03), "point_0.1_noise_0.03")
    add(make_condition_config(kind="point", fraction=0.1, noise_sigma=0.10), "point_0.1_noise_0.10")
    for cap in _OBSERVATION_CAPS:
        add(
            make_condition_config(kind="observation_cap", cap=cap),
            f"observation_cap_{cap or 'all'}",
        )
    for block_count in _BLOCK_COUNTS:
        for cap in _PER_BLOCK_CAPS:
            add(
                make_condition_config(kind="two_d", block_count=block_count, per_block_cap=cap),
                f"two_d_{block_count or 'all'}_{cap or 'all'}",
            )
    return tuple(result)


def _validate_condition(condition: Mapping[str, Any] | str) -> dict[str, Any]:
    if isinstance(condition, str):
        if condition == "full":
            return {"kind": "full", "seed": SEEDS[0], "repeat": 0}
        raise ReliabilitySamplingError(
            "string condition must be 'full'; use config for other levels"
        )
    value = dict(condition)
    kind = value.get("kind", "full")
    if kind not in {"full", "point", "whole_session", "observation_cap", "session_cap", "two_d"}:
        raise ReliabilitySamplingError(f"unknown condition kind: {kind}")
    value["kind"] = kind
    value["seed"] = int(value.get("seed", SEEDS[0]))
    value["repeat"] = int(value.get("repeat", 0))
    if kind in ("point", "whole_session"):
        fraction = float(value.get("fraction"))
        if not 0 < fraction <= 1:
            raise ReliabilitySamplingError("fraction must be in (0, 1]")
        value["fraction"] = fraction
    if kind in ("observation_cap", "session_cap"):
        cap = value.get("cap")
        if cap is not None and (isinstance(cap, bool) or int(cap) != cap or int(cap) < 1):
            raise ReliabilitySamplingError("cap must be a positive integer or None")
        value["cap"] = None if cap is None else int(cap)
    if kind == "two_d":
        for key in ("block_count", "per_block_cap"):
            x = value.get(key)
            if x is not None and (isinstance(x, bool) or int(x) != x or int(x) < 1):
                raise ReliabilitySamplingError(f"{key} must be positive")
            value[key] = None if x is None else int(x)
    if "noise_sigma" in value:
        sigma = float(value["noise_sigma"])
        if sigma < 0 or not math.isfinite(sigma):
            raise ReliabilitySamplingError("noise_sigma must be nonnegative")
        value["noise_sigma"] = sigma
    return value


def _rng(curve: LightCurve, condition: Mapping[str, Any], repeat: int = 0) -> np.random.Generator:
    # Retention levels in one family share the same random ranking. Thus every
    # lower-retention result is a prefix/subset of its higher-retention result.
    # The retention value itself deliberately does not enter this hash.
    family = {
        "kind": condition.get("kind", "full"),
        "seed": condition.get("seed", SEEDS[0]),
        "scope": condition.get("scope", ""),
    }
    payload = json.dumps(
        [curve.object_id, family, int(repeat)], sort_keys=True, separators=(",", ":")
    )
    digest = hashlib.sha256(payload.encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little", signed=False))


def _subset_session(session: NativeSession, indexes: Iterable[int]) -> NativeSession:
    selected = tuple(session.observations[index] for index in indexes)
    return replace(session, observations=selected, header_count=len(selected))


def _point(curve: LightCurve, fraction: float, rng: np.random.Generator) -> LightCurve:
    sessions = []
    for session in curve.valid_sessions:
        n = min(
            len(session.observations), max(2, int(math.floor(len(session.observations) * fraction)))
        )
        order = rng.permutation(len(session.observations))
        sessions.append(_subset_session(session, sorted(order[:n])))
    return replace(curve, sessions=tuple(sessions))


def _whole_session(curve: LightCurve, fraction: float, rng: np.random.Generator) -> LightCurve:
    sessions = curve.valid_sessions
    n = min(len(sessions), max(1, int(math.floor(len(sessions) * fraction))))
    order = rng.permutation(len(sessions))
    selected = tuple(
        sessions[index] for index in sorted(order[:n], key=lambda i: sessions[i].source_index)
    )
    return replace(curve, sessions=selected)


def _cap(curve: LightCurve, cap: int | None, rng: np.random.Generator) -> LightCurve:
    if cap is None:
        return replace(curve, sessions=curve.valid_sessions)
    valid_sessions = curve.valid_sessions
    order = list(rng.permutation(len(valid_sessions)))
    # Both the session order and each session's within-session ranking are
    # seeded. Prefixes at caps 10, 20, ... therefore remain nested.
    within = [list(rng.permutation(len(session.observations))) for session in valid_sessions]
    selected: list[list[int]] = [[] for _ in order]
    remaining = cap
    # A session starts with an atomic pair. This intentionally may leave one
    # unused slot when no further complete pair can be placed.
    for slot, index in enumerate(order):
        if remaining < 2:
            break
        take = min(2, len(within[index]))
        selected[slot].extend(within[index][:take])
        remaining -= take
    all_pairs_allocated = remaining >= 0 and all(
        len(selected[slot]) >= 2 for slot in range(len(order))
    )
    if not all_pairs_allocated:
        output = [
            _subset_session(valid_sessions[index], sorted(indexes))
            for slot, index in enumerate(order)
            if len(indexes := selected[slot]) >= 2
        ]
        return replace(curve, sessions=tuple(output))
    while remaining > 0:
        progressed = False
        for slot, index in enumerate(order):
            if len(selected[slot]) < len(within[index]):
                selected[slot].append(within[index][len(selected[slot])])
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            break
    output = [
        _subset_session(valid_sessions[index], sorted(indexes))
        for slot, index in enumerate(order)
        if len(indexes := selected[slot]) >= 2
    ]
    return replace(curve, sessions=tuple(output))


def _two_d(
    curve: LightCurve,
    block_count: int | None,
    per_block_cap: int | None,
    rng: np.random.Generator,
    *,
    seed: int,
    repeat: int,
) -> LightCurve:
    source = merge_observing_blocks(curve) if not curve.blocks else curve
    if len(source.blocks) < 10:
        raise ReliabilitySamplingError(
            "2D conditions require a common cohort of at least 10 blocks"
        )
    count = len(source.blocks) if block_count is None else min(block_count, len(source.blocks))
    chosen = rng.permutation(len(source.blocks))[:count]
    sessions: list[NativeSession] = []
    for block_index in sorted(chosen):
        block = source.blocks[int(block_index)]
        block_curve = replace(source, sessions=block.sessions, blocks=(block,))
        block_rng = _rng(
            source,
            {"kind": "two_d_block", "seed": seed, "scope": block.block_id},
            repeat,
        )
        selected = (
            _cap(block_curve, per_block_cap, block_rng)
            if per_block_cap is not None
            else replace(block_curve, sessions=block.sessions)
        )
        sessions.extend(selected.valid_sessions)
    return replace(
        source,
        sessions=tuple(sessions),
        blocks=tuple(source.blocks[int(i)] for i in sorted(chosen)),
    )


def add_multiplicative_noise(
    curve: LightCurve, sigma: float, *, seed: int = SEEDS[0], repeat: int = 0
) -> LightCurve:
    """Apply positive multiplicative noise and exactly normalise its sample mean."""
    if sigma < 0 or not math.isfinite(float(sigma)):
        raise ReliabilitySamplingError("sigma must be nonnegative")
    rng = _rng(curve, {"kind": "noise", "sigma": float(sigma), "seed": int(seed)}, repeat)
    sessions = []
    for session in curve.sessions:
        if not session.observations:
            sessions.append(session)
            continue
        z = rng.normal(size=len(session.observations))
        factors = np.exp(float(sigma) * z)
        # Normalise against the flux-weighted mean so the realised sample
        # mean, rather than merely the expected lognormal mean, is unchanged.
        fluxes = np.asarray([item.flux for item in session.observations], dtype=np.float64)
        factors /= float(np.sum(fluxes * factors) / np.sum(fluxes))
        sessions.append(
            replace(
                session,
                observations=tuple(
                    replace(item, flux=item.flux * float(factor))
                    for item, factor in zip(session.observations, factors)
                ),
            )
        )
    return replace(curve, sessions=tuple(sessions))


def transform_curve(
    curve: LightCurve,
    condition: Mapping[str, Any] | str,
    *,
    object_id: str | None = None,
    repeat: int = 0,
) -> LightCurve:
    """Select and transform one curve according to a condition dictionary."""
    cfg = _validate_condition(condition)
    if object_id is not None:
        curve = replace(curve, object_id=object_id)
    rng = _rng(curve, cfg, repeat + int(cfg.get("repeat", 0)))
    kind = cfg["kind"]
    if kind == "full":
        output = replace(curve, sessions=curve.valid_sessions)
    elif kind == "point":
        output = _point(curve, cfg["fraction"], rng)
    elif kind == "whole_session":
        output = _whole_session(curve, cfg["fraction"], rng)
    elif kind in ("observation_cap", "session_cap"):
        output = _cap(curve, cfg["cap"], rng)
    else:
        output = _two_d(
            curve,
            cfg["block_count"],
            cfg["per_block_cap"],
            rng,
            seed=cfg["seed"],
            repeat=repeat + int(cfg.get("repeat", 0)),
        )
    sigma = cfg.get("noise_sigma")
    return (
        add_multiplicative_noise(output, sigma, seed=cfg["seed"], repeat=repeat)
        if sigma is not None
        else output
    )


select_observations = transform_curve


def transform(
    sessions: Sequence[NativeSession] | LightCurve,
    condition: Mapping[str, Any] | str,
    object_id: str = "",
    *,
    seed: int | None = None,
    repeat: int = 0,
    period_hours: float | None = None,
) -> tuple[NativeSession, ...]:
    """Runner-facing transform preserving native session/header structure."""
    curve = (
        sessions
        if isinstance(sessions, LightCurve)
        else LightCurve(tuple(sessions), object_id, period_hours)
    )
    cfg = dict(_validate_condition(condition))
    if seed is not None:
        cfg["seed"] = int(seed)
    return transform_curve(
        curve, cfg, object_id=object_id or curve.object_id, repeat=repeat
    ).sessions


def select_holdout(
    curve: LightCurve,
    *,
    minimum_blocks: int = 3,
    minimum_sessions: int = 5,
    minimum_retained_observations: int = 150,
    minimum_withheld_observations: int = 20,
) -> Holdout:
    """Withhold the latest complete merged block, only when eligibility is met."""
    source = merge_observing_blocks(curve) if not curve.blocks else curve
    if len(source.blocks) < minimum_blocks:
        raise ReliabilitySamplingError("curve has too few merged blocks")
    block = source.blocks[-1]
    train_sessions = tuple(
        session for candidate in source.blocks[:-1] for session in candidate.sessions
    )
    train = replace(source, sessions=train_sessions, blocks=source.blocks[:-1])
    withheld = replace(source, sessions=block.sessions, blocks=(block,))
    if (
        len(train.sessions) >= minimum_sessions
        and len(train.observations) >= minimum_retained_observations
        and len(withheld.observations) >= minimum_withheld_observations
    ):
        return Holdout(train, withheld, block)
    raise ReliabilitySamplingError("latest complete block does not meet holdout eligibility")


def condition_metadata(
    curve: LightCurve, condition: Mapping[str, Any] | str, *, selected: LightCurve | None = None
) -> dict[str, Any]:
    """Return auditable counts and condition details; no labels are included."""
    cfg = _validate_condition(condition)
    value = selected or transform_curve(curve, cfg)
    return {
        "condition": cfg,
        "object_id": curve.object_id,
        "native_sessions": curve.native_sessions,
        "native_observations": curve.total_observations,
        "retained_sessions": value.native_sessions,
        "retained_observations": value.total_observations,
        "blocks": len(value.blocks or merge_observing_blocks(value).blocks),
        "header_flags": [session.calibrated for session in value.sessions],
        "padding": False,
    }


metadata = condition_metadata
holdout = select_holdout
make_condition_configs = condition_configs


def sampling_metadata(
    sessions: Sequence[NativeSession] | LightCurve,
    period_hours: float | None = None,
    *,
    object_id: str = "",
) -> dict[str, Any]:
    """Native inventory metadata used before selecting any condition."""
    curve = (
        sessions
        if isinstance(sessions, LightCurve)
        else LightCurve(tuple(sessions), object_id, period_hours)
    )
    merged = merge_observing_blocks(curve)
    observations = curve.observations
    return {
        "object_id": curve.object_id,
        "period_hours": curve.period_hours,
        "native_sessions": curve.native_sessions,
        "native_observations": curve.total_observations,
        "merged_blocks": len(merged.blocks),
        "session_flags": [session.calibrated for session in curve.sessions],
        "time_span_days": (
            max(item.time_jd for item in observations) - min(item.time_jd for item in observations)
        )
        if observations
        else 0.0,
        "geometry": {"sun": bool(observations), "observer": bool(observations)},
    }


def split_holdout(
    sessions: Sequence[NativeSession] | LightCurve,
    *,
    object_id: str = "",
    period_hours: float | None = None,
) -> tuple[tuple[NativeSession, ...], tuple[NativeSession, ...]]:
    """Runner-facing latest-block split, returning fit and withheld sessions."""
    curve = (
        sessions
        if isinstance(sessions, LightCurve)
        else LightCurve(tuple(sessions), object_id, period_hours)
    )
    result = select_holdout(curve)
    return result.train.sessions, result.withheld.sessions


__all__ = [
    "SEEDS",
    "CurveObservation",
    "NativeSession",
    "ObservingBlock",
    "LightCurve",
    "Holdout",
    "ReliabilitySamplingError",
    "read_lc",
    "write_lc",
    "merge_observing_blocks",
    "read_native",
    "write_native",
    "to_epochs",
    "make_condition_config",
    "select_observations",
    "transform_curve",
    "add_multiplicative_noise",
    "transform",
    "condition_configs",
    "select_holdout",
    "split_holdout",
    "condition_metadata",
    "sampling_metadata",
    "metadata",
    "holdout",
    "make_condition_configs",
]
