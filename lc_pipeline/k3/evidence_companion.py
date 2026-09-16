"""Descriptive, post-hoc checks using frozen K3 outputs and native metadata.

These checks do not refit K3 or constitute a new independent test. In particular,
the within-fold reassignment calculation is not a retrained permutation test.
"""
from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np

from .reliability_sampling import LightCurve, merge_observing_blocks


def session_metadata(curve: LightCurve) -> dict:
    spans, counts, cadences, phase_fractions, amplitudes = [], [], [], [], []
    short = []
    for session in curve.valid_sessions:
        times = np.sort([o.time_jd for o in session.observations])
        hours = float((times[-1] - times[0]) * 24)
        spans.append(hours)
        counts.append(len(times))
        flux = np.asarray([o.flux for o in session.observations])
        amplitudes.append(float((np.quantile(flux, .95) - np.quantile(flux, .05)) / flux.mean()))
        if hours <= 12:
            short.append(session)
            differences = np.diff(times) * 24 * 60
            if np.any(differences > 0):
                cadences.append(float(np.median(differences[differences > 0])))
            phases = np.sort(np.remainder((times - times[0]) * 24 / curve.period_hours, 1))
            phase_fractions.append(float(1 - np.diff(np.r_[phases, phases[0] + 1]).max()))
    all_times = np.asarray([o.time_jd for s in curve.valid_sessions for o in s.observations])
    result = {
        "object_id": curve.object_id, "sessions": len(counts), "points": sum(counts),
        "short_sessions_le12h": len(short),
        "utc_observing_dates": int(len(np.unique(np.floor(all_times + .5)))),
        "blocks_30d": len(merge_observing_blocks(curve).blocks),
        "period_hours": curve.period_hours,
        "time_span_days": float(np.ptp(all_times)),
        "median_session_hours": float(np.median(spans)),
        "median_points_per_session": float(np.median(counts)),
        "median_short_session_points": float(np.median([len(s.observations) for s in short])) if short else None,
        "median_short_session_cadence_minutes": float(np.median(cadences)) if cadences else None,
        "median_short_session_phase_span": float(np.median(phase_fractions)) if phase_fractions else None,
        "median_relative_amplitude_90pct": float(np.median(amplitudes)),
        "session_span_hours": spans,
    }
    return result


def short_session_subset(curve: LightCurve, *, session_count: int | None,
                         point_cap: int | None, seed: int) -> LightCurve:
    """Nested sampling of native sessions <=12h, without redefining epochs.

    These sessions approximate nightly visits, not identified local nights.
    A session with fewer points than the cap keeps its original points.
    """
    if session_count is not None and session_count < 1:
        raise ValueError("session_count must be positive")
    if point_cap is not None and point_cap < 2:
        raise ValueError("point_cap must be at least two")
    sessions = [s for s in curve.valid_sessions if
                (max(o.time_jd for o in s.observations) - min(o.time_jd for o in s.observations)) * 24 <= 12]
    if not sessions or (session_count is not None and len(sessions) < session_count):
        raise ValueError("insufficient short native sessions")

    def rng(scope):
        value = f"short-session-v1:{seed}:{curve.object_id}:{scope}".encode()
        return np.random.default_rng(int.from_bytes(hashlib.sha256(value).digest()[:8], "little"))

    order = rng("sessions").permutation(len(sessions))
    chosen = set(order[:session_count] if session_count is not None else order)
    output = []
    for index, session in enumerate(sessions):
        if index not in chosen:
            continue
        order = rng(session.session_id).permutation(len(session.observations))
        keep = sorted(order[:point_cap] if point_cap is not None else order)
        output.append(replace(session, observations=tuple(session.observations[i] for i in keep)))
    return replace(curve, sessions=tuple(output), blocks=())


def angular_error(axes, references):
    a, r = np.asarray(axes, float), np.asarray(references, float)
    if a.ndim != 2 or r.ndim != 2 or a.shape[1] != 3 or r.shape[1] != 3:
        raise ValueError("vectors must have shape (N,3)")
    if not np.isfinite(a).all() or not np.isfinite(r).all():
        raise ValueError("vectors must be finite")
    if np.any(np.linalg.norm(a, axis=1) == 0) or np.any(np.linalg.norm(r, axis=1) == 0):
        raise ValueError("vectors must be nonzero")
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    r = r / np.linalg.norm(r, axis=1, keepdims=True)
    return float(np.rad2deg(np.arccos(np.clip(np.abs(a @ r.T).max(), 0, 1))))


def matching_diagnostic(axes, references, folds, *, draws=10000, seed=20260916):
    """Compare each set with other objects in its own held-out fold.

    No learned model or reference set is changed. The simulated reassignment
    distribution is descriptive; its tail is not a pipeline permutation p-value.
    """
    folds = np.asarray(folds)
    observed, wrong, records = [], [], []
    matrices = []
    for fold in sorted(set(folds)):
        indices = np.flatnonzero(folds == fold)
        if len(indices) < 2:
            raise ValueError("matching check needs two or more objects per fold")
        matrix = np.asarray([[angular_error(axes[i], references[j]) for j in indices] for i in indices])
        correct = np.diag(matrix)
        other = (matrix.sum(axis=0) - correct) / (len(indices) - 1)
        observed.extend(correct)
        wrong.extend(other)
        records.append({"fold": int(fold), "n": len(indices), "matched_mean_deg": float(correct.mean()),
                        "wrong_object_mean_deg": float(other.mean()),
                        "wrong_minus_matched_deg": float(np.mean(other - correct))})
        matrices.append(matrix)
    rng = np.random.default_rng(seed)
    null = np.zeros(draws)
    for matrix in matrices:
        for draw in range(draws):
            null[draw] += matrix[np.arange(len(matrix)), rng.permutation(len(matrix))].sum()
    null /= len(folds)
    return {"role": "post_hoc_output_matching_diagnostic_not_retrained_permutation_test",
            "observed_mean_deg": float(np.mean(observed)), "wrong_object_mean_deg": float(np.mean(wrong)),
            "wrong_minus_matched_deg": float(np.mean(wrong) - np.mean(observed)),
            "reassignment_95pct_range_deg": np.quantile(null, [.025, .975]).tolist(),
            "folds": records, "draws": draws, "seed": seed}, null


def metadata_neighbours(train_features, test_features, train_axes, *, neighbours=3):
    """Training-only z-scales and three nearest training-primary reference axes."""
    train, test = np.asarray(train_features, float), np.asarray(test_features, float)
    axes = np.asarray(train_axes, float)
    if train.ndim != 2 or test.ndim != 2 or train.shape[1] != test.shape[1]:
        raise ValueError("feature dimensions differ")
    if len(train) < neighbours or axes.shape != (len(train), 3):
        raise ValueError("insufficient training objects or malformed axes")
    if not np.isfinite(train).all() or not np.isfinite(test).all() or not np.isfinite(axes).all():
        raise ValueError("metadata must be finite")
    centre = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1e-12] = 1
    distances = np.square((test[:, None] - centre) / scale - (train[None] - centre) / scale).sum(axis=-1)
    order = np.argsort(distances, axis=1, kind="stable")[:, :neighbours]
    return axes[order], order, np.sqrt(np.take_along_axis(distances, order, axis=1)), centre, scale


def exploratory_grid_modes(scores, vectors, neighbours, *, count, separation_deg=15):
    """Post-hoc grid-only mode count sensitivity; does not change deployed K3.

    K=3 must reproduce the released initial (unrefined) modes. Increasing K
    changes proposal-set size, not the model architecture or its training.
    """
    scores, vectors = np.asarray(scores), np.asarray(vectors)
    if count < 2 or count > 5:
        raise ValueError("exploratory counts are fixed to 2,3,4,5")
    maxima = []
    for index, adjacent in enumerate(neighbours):
        adjacent = np.unique(adjacent[(adjacent >= 0) & (adjacent != index)])
        if (np.all(scores[index] >= scores[adjacent]) and
                not np.any((scores[index] == scores[adjacent]) & (adjacent < index))):
            maxima.append(index)
    order = np.lexsort((np.arange(len(scores)), -scores))
    local_order = sorted(maxima, key=lambda i: (-scores[i], i))
    chosen = []
    for index in [*local_order, *order]:
        if index in chosen:
            continue
        if chosen and np.any(np.abs(vectors[chosen] @ vectors[index]) > np.cos(np.deg2rad(separation_deg))):
            continue
        chosen.append(int(index))
        if len(chosen) == count:
            break
    if len(chosen) != count:
        raise ValueError("cannot extract requested separated modes")
    return vectors[chosen], chosen
