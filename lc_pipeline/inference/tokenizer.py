"""
Canonical lightcurve tokenization for neural network input.

This is the SINGLE AUTHORITATIVE tokenizer used by both training and inference.
The training dataset (damit_multiepoch_dataset.py) imports and uses this module
to ensure identical tokenization logic.

Features (13D):
  0: t_norm      - Normalized time [0, 1] within window
  1: dt_norm     - Time delta (normalized by mean delta)
  2: mag_norm    - Robust MAD-normalized brightness
  3-8: geometry  - Set to zero (reserved)
  9: log1p(|rotations|) - Log of absolute rotation count since reference
  10: log(period) - Log of rotation period in hours (normalized)
  11: sin(rotation_phase) - Sine of rotational phase [0, 2π)
  12: cos(rotation_phase) - Cosine of rotational phase [0, 2π)

Input format (DAMIT 8-column):
  [JD, brightness, sun_x, sun_y, sun_z, obs_x, obs_y, obs_z]

Note: Geometry features (3-8) are always set to ZERO by design.
While the model reads sun/observer vectors from input files, these features
provide implicit regularization benefits when zeroed during training and inference.
"""

from typing import List, Optional, Tuple

import numpy as np

N_FEATURES = 13  # 3 temporal + 6 geometry (zeros) + 4 period


def tokenize_window(
    points: np.ndarray,
    *,
    period_hours: Optional[float] = None,
    global_jd_min: Optional[float] = None,
    use_geometry: bool = False,
    ablate_features: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tokenize a window of observations to model input features.

    This is the CANONICAL tokenizer used by both training and inference.
    Any changes here affect both pipelines.

    Args:
        points: (N, 8) array with columns:
                [JD, brightness, sun_x, sun_y, sun_z, obs_x, obs_y, obs_z]
                - JD: Julian Date (time)
                - brightness: relative brightness/flux (typically 0.9-1.1 for asteroids)
                - sun_xyz: unit vector from asteroid to Sun (not used - geometry disabled)
                - obs_xyz: unit vector from asteroid to observer (not used - geometry disabled)
        period_hours: Rotation period in hours (REQUIRED for period features 9-12)
        global_jd_min: Global JD minimum for rotation counting (optional)
        use_geometry: Ignored - geometry features always set to zero (kept for compatibility)
        ablate_features: Optional list of feature names to zero out for ablation tests.
                        Supported values: 'time', 'cadence', 'brightness', 'geometry', 'period'
                        - 'time': zeros feature 0 (t_norm)
                        - 'cadence': zeros feature 1 (dt_norm)
                        - 'brightness': zeros feature 2 (mag_norm)
                        - 'geometry': zeros features 3-8 (already zero, no-op)
                        - 'period': zeros features 9-12 (rotations, log_period, sin/cos phase)

    Returns:
        tokens: (N, 13) feature array
        mask: (N,) boolean array (1 = valid token)

    Features (13D):
        0: t_norm     - Normalized time [0, 1] within window
        1: dt_norm    - Time delta normalized by mean delta
        2: mag_norm   - Robust MAD-normalized brightness
        3-8: geometry - ALWAYS ZERO (unused, provide regularization)
        9: log1p(|rotations|) - Log of absolute rotation count since reference
        10: log(period) - Log of period hours (normalized)
        11-12: sin/cos(rotation_phase) - Rotational phase
    """
    N = len(points)
    if N == 0:
        tokens = np.zeros((0, N_FEATURES), dtype=np.float32)
        mask = np.zeros(0, dtype=np.float32)
        return tokens, mask

    # NO RESAMPLING - use all points as-is
    # Extract columns
    jd = points[:, 0]
    brightness = points[:, 1]
    # Initialize tokens (natural size N, not resampled)
    tokens = np.zeros((N, N_FEATURES), dtype=np.float32)
    mask = np.ones(N, dtype=np.float32)

    # Feature 0: t_norm (0..1 within epoch)
    t_min, t_max = jd.min(), jd.max()
    t_range = t_max - t_min
    if t_range > 1e-8:
        tokens[:, 0] = (jd - t_min) / t_range
    else:
        tokens[:, 0] = 0.0

    # Feature 1: dt_norm (time delta normalized, clipped to [-10, 10])
    if N > 1:
        dt = np.diff(jd)
        dt_mean = np.mean(np.abs(dt)) + 1e-8  # Use abs to handle negative gaps
        dt_norm = dt / dt_mean
        tokens[1:, 1] = np.clip(dt_norm, -10.0, 10.0)
        tokens[0, 1] = 0.0

    # Feature 2: mag_norm - Robust MAD-based normalization
    #
    # Simple per-window normalization using median and MAD (median absolute deviation).
    # This matches the training data preprocessing exactly.
    #
    # Note: The 'brightness' column from DAMIT is relative brightness (flux-like),
    # typically in range [0.9, 1.1]. We normalize it directly without sign inversion
    # since the model learns the pattern regardless of sign convention.

    mag_median = np.median(brightness)
    mag_mad = np.median(np.abs(brightness - mag_median))
    mag_range = np.max(brightness) - np.min(brightness)
    # Use max of MAD and 10% of range as scaling to prevent instability
    mag_scale = max(mag_mad, 0.1 * (mag_range + 1e-8))

    # Normalize and clip to reasonable range [-10, 10]
    tokens[:, 2] = np.clip((brightness - mag_median) / mag_scale, -10.0, 10.0)

    # Features 3-8: Geometry features (set to zero, reserved)
    tokens[:, 3:9] = 0.0

    # Features 9-12: Period-based features
    has_period = period_hours is not None and period_hours > 0
    if has_period:
        period_days = period_hours / 24.0

        # Safety: reject physically impossible periods (< 1.4 min)
        if period_days < 0.001:
            tokens[:, 9:13] = 0.0
        else:
            # Use global JD min if provided, otherwise local min
            jd_ref = global_jd_min if global_jd_min is not None else jd.min()

            # Feature 9: log1p(|rotations|) since reference time
            rotations = (jd - jd_ref) / period_days
            tokens[:, 9] = np.log1p(np.abs(rotations))

            # Feature 10: log(period) (NORMALIZED)
            # Empirical range: period_hours ∈ [2, 2000] → log(period) ∈ [0.69, 7.6]
            # Normalize to approximately [0, 1], clip for out-of-range periods
            log_period = np.log(period_hours + 1e-8)
            tokens[:, 10] = np.clip((log_period - 0.5) / 7.5, -2.0, 2.0)

            # Features 11-12: sin/cos of rotation phase (already normalized to [-1, 1])
            phase = rotations - np.floor(rotations)  # Fractional part [0, 1]
            tokens[:, 11] = np.sin(2 * np.pi * phase)
            tokens[:, 12] = np.cos(2 * np.pi * phase)
    else:
        # No period provided - leave period features as zero
        tokens[:, 9:13] = 0.0

    # Feature ablation (for sanity checks and ablation studies)
    if ablate_features is not None:
        for feature_name in ablate_features:
            if feature_name == "time":
                tokens[:, 0] = 0.0  # t_norm
            elif feature_name == "cadence":
                tokens[:, 1] = 0.0  # dt_norm
            elif feature_name == "brightness":
                tokens[:, 2] = 0.0  # mag_norm
            elif feature_name == "geometry":
                # Already zero, but included for consistency
                tokens[:, 3:9] = 0.0
            elif feature_name == "period":
                tokens[:, 9:13] = 0.0  # rotations_elapsed, log_period, sin/cos phase
            else:
                raise ValueError(
                    f"Unknown ablation feature: {feature_name}. "
                    f"Supported: 'time', 'cadence', 'brightness', 'geometry', 'period'"
                )

    return tokens, mask


def split_into_windows(
    epochs: List[np.ndarray],
    n_windows: int = 8,
    max_gap_days: float = 1.5,
    max_tokens_per_window: int = 256,
) -> List[np.ndarray]:
    """
    Split epochs into observation windows based on time gaps.

    Algorithm:
    1. Flatten all epochs into one continuous time series
    2. Split by time gaps exceeding max_gap_days
    3. Subdivide segments larger than max_tokens_per_window
    4. Select evenly-spaced subset if more than n_windows

    Args:
        epochs: List of (N, 8) arrays
        n_windows: Target number of windows (default 8)
        max_gap_days: Maximum time gap within a window (default 1.0 days)
        max_tokens_per_window: Maximum tokens per window (default 256)

    Returns:
        List of window arrays (each is (M, 8))
    """
    if len(epochs) == 0:
        return []

    # Flatten all epochs into a single time series
    all_points = np.concatenate(epochs, axis=0)

    # CRITICAL: Sort by time! The dataset always sorts data before windowing.
    # This ensures consistent window boundaries and proper gap detection.
    # Unsorted data creates incorrect gaps and breaks the model.
    times = all_points[:, 0]
    sort_idx = np.argsort(times)
    all_points = all_points[sort_idx]
    times = all_points[:, 0]  # Re-extract sorted times

    # Find gaps exceeding threshold to identify window boundaries
    if len(times) > 1:
        time_diffs = np.diff(times)
        gap_indices = np.where(time_diffs > max_gap_days)[0] + 1
    else:
        gap_indices = []

    # Split by gaps
    split_indices = [0] + list(gap_indices) + [len(all_points)]
    segments = []
    for i in range(len(split_indices) - 1):
        start, end = split_indices[i], split_indices[i + 1]
        if end - start >= 3:  # Minimum window size
            segments.append((start, end))

    # Subdivide segments that exceed maximum token count
    windows = []
    for start, end in segments:
        seg_len = end - start
        if seg_len <= max_tokens_per_window:
            windows.append(all_points[start:end])
        else:
            # Divide into smaller chunks
            n_chunks = (seg_len + max_tokens_per_window - 1) // max_tokens_per_window
            chunk_size = (seg_len + n_chunks - 1) // n_chunks
            for j in range(n_chunks):
                c_start = start + j * chunk_size
                c_end = min(start + (j + 1) * chunk_size, end)
                if c_end - c_start >= 3:
                    windows.append(all_points[c_start:c_end])

    # Limit number of windows (keep evenly spaced subset)
    if len(windows) > n_windows:
        indices = np.linspace(0, len(windows) - 1, n_windows, dtype=int)
        windows = [windows[i] for i in indices]

    # If we have fewer windows, pad by repeating
    while len(windows) < n_windows and len(windows) > 0:
        sizes = [len(w) for w in windows]
        largest_idx = np.argmax(sizes)
        windows.append(windows[largest_idx].copy())

    return windows[:n_windows]


def tokenize_lightcurve(
    epochs: List[np.ndarray],
    period_hours: Optional[float] = None,
    n_windows: int = 8,
    tokens_per_window: int = 256,
    use_geometry: bool = False,
    ablate_features: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert raw lightcurve epochs to model input tokens.

    Windows preserve their natural size (no resampling to fixed length).
    Padding is applied to reach max_T across all windows.

    Args:
        epochs: List of (N, 8) arrays [JD, brightness, sun_xyz, obs_xyz]
        period_hours: Rotation period in hours (NOW USED for features 9-12!)
        n_windows: Number of observation windows
        tokens_per_window: Maximum tokens per window (used for padding)

    Returns:
        tokens: (n_windows, max_T, 13) where max_T = max window size
        mask: (n_windows, max_T) float (1 = valid, 0 = padding)
    """
    n_epochs = len(epochs)
    if n_epochs == 0:
        tokens = np.zeros((n_windows, 0, N_FEATURES), dtype=np.float32)
        mask = np.zeros((n_windows, 0), dtype=np.float32)
        return tokens, mask

    # Split into windows using default gap threshold
    windows = split_into_windows(
        epochs,
        n_windows=n_windows,
        max_gap_days=1.0,  # 1-day gap threshold for window boundaries
        max_tokens_per_window=tokens_per_window,
    )

    # Ensure we have the target number of windows (pad if needed)
    while len(windows) < n_windows:
        if len(windows) > 0:
            windows.append(windows[0].copy())
        else:
            # Create empty window if no data
            windows.append(np.zeros((0, 8), dtype=np.float32))

    # Tokenize each window (preserving natural window sizes)
    # Note: global_jd_min is NOT computed across all epochs. Each window uses
    # its own local jd.min() for rotation counting (global_jd_min=None).
    # This matches training behavior in single_epoch_dataset.py, where each
    # epoch passes its own times.min() as the reference. Using a global min
    # would cause Feature 9 (log1p|rotations|) to reach much larger values
    # for later epochs than anything seen during training.
    all_tokens = []
    all_masks = []

    for w in range(n_windows):
        window_points = windows[w]

        window_tokens, window_mask = tokenize_window(
            window_points,
            period_hours=period_hours,
            global_jd_min=None,  # Window-local: matches training behavior
            use_geometry=use_geometry,
            ablate_features=ablate_features,
        )

        all_tokens.append(window_tokens)
        all_masks.append(window_mask)

    # Find max window size for padding
    max_T = max(t.shape[0] for t in all_tokens) if all_tokens else 0
    if max_T == 0:
        max_T = 1  # Avoid zero-size arrays

    # Pad all windows to max_T
    tokens = np.zeros((n_windows, max_T, N_FEATURES), dtype=np.float32)
    mask = np.zeros((n_windows, max_T), dtype=np.float32)

    for w in range(n_windows):
        T = all_tokens[w].shape[0]
        if T > 0:
            tokens[w, :T] = all_tokens[w]
            mask[w, :T] = all_masks[w]

    return tokens, mask
