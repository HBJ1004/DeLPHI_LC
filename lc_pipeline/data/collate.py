#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data preparation functions for light curve analysis pipeline.
"""

import numpy as np
import torch
from typing import List, Tuple, Dict, Any, Optional, Union
import logging
import os
import random  # Add import for random module
import glob

from lc_pipeline.models.utils import lon_lat_to_direction_vector, direction_vector_to_quaternion

# Removed the conditional import for AsteroidDataset as we'll use a string literal for the type hint.

def collate_fn(batch):
    """
    Optimized custom collate function for batches of variable length asteroid data.
    Accepts batches where each item is (features, targets, ids, raw_data_tuple) from AsteroidDataset.
    
    For use with DataLoader to handle variable-length sequences.
    
    Args:
        batch: List of tuples (features, targets, ids, raw_data_tuple) from AsteroidDataset
        
    Returns:
        tuple: (padded_data, targets, ids, lengths)
            - padded_data: Tensor of shape [batch_size, max_len, feat_dim]
            - targets: Tensor of shape [batch_size, target_dim]
            - ids: List of asteroid IDs
            - lengths: Tensor of original sequence lengths
    """
    # Verify batch is not empty
    if not batch:
        raise ValueError("Empty batch passed to collate_fn")
        
    # Check if batch items have the expected structure
    if len(batch[0]) != 4:
        raise ValueError(f"Expected 4 items per batch entry, got {len(batch[0])}. Check dataset __getitem__ return type.")
    
    # Separate batch components - AsteroidDataset returns 4 items
    # We'll ignore raw_data_tuple in this collate_fn as it's not directly used by standard training loops
    # If needed for LS features on-the-fly, a different or modified collate_fn would handle it.
    features, targets, ids, _raw_data_tuples = zip(*batch) # Unpack 4 items, ignore the 4th
    
    # Verify features are tensors
    if not isinstance(features[0], torch.Tensor):
        raise TypeError(f"Expected tensor features, got {type(features[0]).__name__}. Check dataset __getitem__ return type.")
    
    # Get sequence lengths before padding (from features)
    lengths = torch.tensor([f.shape[0] for f in features], dtype=torch.int32)
    
    # Get max sequence length in this batch
    max_len = lengths.max().item()
    
    # Get feature dimension
    feat_dim = features[0].shape[1]
    
    # Pre-allocate zeroed tensor with efficient layout
    padded_data = torch.zeros(
        (len(batch), max_len, feat_dim), 
        dtype=torch.float32, 
        device=features[0].device
    )
    
    # Fill padded tensor with data efficiently (vectorized where possible)
    for i, (seq, length) in enumerate(zip(features, lengths)):
        # This slice assignment is more efficient than looping
        padded_data[i, :length] = seq
    
    # Stack targets into a tensor efficiently
    if isinstance(targets[0], torch.Tensor):
        targets_tensor = torch.stack(targets)
    else:
        # If targets are not already tensors, convert them
        targets_tensor = torch.tensor(targets, dtype=torch.float32)
    
    return padded_data, targets_tensor, ids, lengths


def collate_fn_with_raw(batch):
    """
    Optimized collate function that preserves raw data for LS feature calculation.
    
    Args:
        batch: List of tuples (data, target, id, length, raw_data) from LSFeatureDataset
        
    Returns:
        tuple: (padded_data, targets, ids, lengths, raw_data_list)
    """
    # Separate batch components
    data, targets, ids, orig_lengths, raw_data = zip(*batch)
    
    # Get sequence lengths before padding
    lengths = torch.tensor([d.shape[0] for d in data], dtype=torch.int32)
    
    # Get max sequence length in this batch
    max_len = lengths.max().item()
    
    # Get feature dimension
    feat_dim = data[0].shape[1]
    
    # Create padded data tensor with optimal memory layout
    padded_data = torch.zeros(
        (len(batch), max_len, feat_dim), 
        dtype=torch.float32, 
        device=data[0].device
    )
    
    # Fill padded tensor with data
    for i, (seq, length) in enumerate(zip(data, lengths)):
        padded_data[i, :length] = seq
    
    # Stack targets into a tensor efficiently
    if isinstance(targets[0], torch.Tensor):
        targets_tensor = torch.stack(targets)
    else:
        targets_tensor = torch.tensor(targets, dtype=torch.float32)
    
    return padded_data, targets_tensor, ids, lengths, raw_data


def phase_fold(time, values, period, num_bins=100, smooth=True, return_bin_centers=False):
    """
    Optimized phase-folding for light curves using the given period.
    
    Args:
        time: Time array
        values: Magnitude/brightness array
        period: Rotation period to use for folding
        num_bins: Number of bins in the phase-folded curve
        smooth: Apply smoothing to the phase-folded curve
        return_bin_centers: Return bin centers as well
        
    Returns:
        numpy.ndarray: Phase-folded curve of shape [num_bins]
        or
        tuple: (phase_folded_curve, bin_centers) if return_bin_centers=True
    """
    logger = logging.getLogger(__name__)
    
    # Pre-compute bin centers
    bin_centers = np.linspace(0, 1, num_bins, endpoint=False) + 0.5/num_bins
    _logged_cubic_fallback = False # Flag for logging cubic fallback only once per call
    
    # Validate inputs
    if len(time) < 5 or period <= 0:
        logger.debug(f"phase_fold: Insufficient data points ({len(time)}) or invalid period ({period})")
        # Return empty phase curve if not enough data or invalid period
        if return_bin_centers:
            return np.zeros(num_bins, dtype=np.float32), bin_centers
        return np.zeros(num_bins, dtype=np.float32)
    
    try:
        # Calculate phases with better numerical stability
        safe_period = max(period, 1e-6)
        phases = np.mod(time / safe_period, 1.0)
        
        # Check for invalid phases quickly
        if not np.all(np.isfinite(phases)):
            logger.debug(f"phase_fold: Invalid phases detected for period {period}")
            if return_bin_centers:
                return np.zeros(num_bins, dtype=np.float32), bin_centers
            return np.zeros(num_bins, dtype=np.float32)
            
        # Create phase bins once
        bins = np.linspace(0, 1, num_bins + 1, endpoint=True)
        
        # Separate handling based on data size and smoothing preference
        if smooth and len(time) >= 10:  # Only apply smoothing if enough data points
            # Use numpy operations for better efficiency
            extended_phases = np.concatenate([phases, phases + 1.0])
            extended_values = np.concatenate([values, values])
            
            # Use more efficient sorting
            sorted_indices = np.argsort(extended_phases)
            sorted_phases_raw = extended_phases[sorted_indices]
            sorted_values_raw = extended_values[sorted_indices]

            # Handle duplicate phases by averaging corresponding values
            unique_phases, unique_indices, inverse_indices, counts = np.unique(
                sorted_phases_raw, return_index=True, return_inverse=True, return_counts=True
            )
            
            averaged_values = np.zeros_like(unique_phases, dtype=np.float64) # Use float64 for sum to maintain precision
            np.add.at(averaged_values, inverse_indices, sorted_values_raw) # Sum values for each unique phase
            averaged_values /= counts # Divide by counts to get the average

            # Now use unique_phases and averaged_values for interpolation
            interp_phases = unique_phases
            interp_values = averaged_values.astype(np.float32) # Cast back to float32 if needed
            
            # Interpolate to get values at bin centers
            if len(interp_phases) > 3:  # Need at least 4 points for cubic interpolation
                try:
                    from scipy.interpolate import interp1d
                    
                    interpolator = interp1d(
                        interp_phases, interp_values, 
                        kind='cubic', bounds_error=False, fill_value='extrapolate',
                        assume_sorted=True 
                    )
                    phase_folded = interpolator(bin_centers)
                except Exception as e:
                    if logger.isEnabledFor(logging.DEBUG) and not _logged_cubic_fallback:
                        # Log includes (PID, time_id, values_id) to help trace specific call
                        logger.debug(f"phase_fold: Cubic interpolation failed (UID: {os.getpid()}-{id(time)}-{id(values)}): {e}, falling back to linear. Unique phases: {len(interp_phases)}")
                        _logged_cubic_fallback = True 
                    try:
                        # Fall back to linear interpolation
                        interpolator = interp1d(
                            interp_phases, interp_values,
                            kind='linear', bounds_error=False, fill_value='extrapolate',
                            assume_sorted=True  
                        )
                        phase_folded = interpolator(bin_centers)
                    except Exception as e2:
                        # Log includes (PID, time_id, values_id) to help trace specific call
                        logger.debug(f"phase_fold: Linear interpolation also failed (UID: {os.getpid()}-{id(time)}-{id(values)}): {e2}, falling back to histogram. Unique phases: {len(interp_phases)}")
                        # Use numpy's histogram function (optimized C implementation)
                        weights = values
                        counts = np.ones_like(phases)
                        phase_folded_weighted, _ = np.histogram(phases, bins=bins, weights=weights)
                        bin_count, _ = np.histogram(phases, bins=bins, weights=counts)
                        
                        # Avoid division by zero with more efficient method
                        mask = bin_count > 0
                        phase_folded = np.zeros(num_bins, dtype=np.float32)
                        phase_folded[mask] = phase_folded_weighted[mask] / bin_count[mask]
            else:
                # Use histograms directly for few points (faster than interpolation)
                weights = values
                counts = np.ones_like(phases)
                phase_folded_weighted, _ = np.histogram(phases, bins=bins, weights=weights)
                bin_count, _ = np.histogram(phases, bins=bins, weights=counts)
                
                # Use vectorized operations
                mask = bin_count > 0
                phase_folded = np.zeros(num_bins, dtype=np.float32)
                phase_folded[mask] = phase_folded_weighted[mask] / bin_count[mask]
        else:
            # Simple binning approach with direct numpy operations (no loops)
            weights = values
            counts = np.ones_like(phases)
            phase_folded_weighted, _ = np.histogram(phases, bins=bins, weights=weights)
            bin_count, _ = np.histogram(phases, bins=bins, weights=counts)
            
            # Avoid division by zero with vectorized operations
            mask = bin_count > 0
            phase_folded = np.zeros(num_bins, dtype=np.float32)
            phase_folded[mask] = phase_folded_weighted[mask] / bin_count[mask]
        
        # Clean up result with efficient vectorized operations
        if not np.all(np.isfinite(phase_folded)):
            phase_folded = np.nan_to_num(phase_folded, nan=0.0, posinf=0.0, neginf=0.0)
            
        # Ensure float32 for memory efficiency
        phase_folded = phase_folded.astype(np.float32)
            
        if return_bin_centers:
            return phase_folded, bin_centers
        return phase_folded
        
    except Exception as e:
        logger.debug(f"phase_fold: Exception during phase folding: {e}")
        # Return zeros on any unexpected error
        if return_bin_centers:
            return np.zeros(num_bins, dtype=np.float32), bin_centers
        return np.zeros(num_bins, dtype=np.float32)


def generate_axis_data(
    period_model, 
    dataset: 'AsteroidDataset', # Use string literal for forward reference
    num_bins=100, 
    smooth=True, 
    return_ids=False, 
    use_true_periods=False, # DEPRECATED in favor of true_period_mix_ratio
    use_quaternions=True,
    true_period_mix_ratio: float = 1.0, # Ratio of times to use true period (0.0 = always predict, 1.0 = always true)
    logger: Optional[logging.Logger] = None
):
    """
    Generate phase-folded light curves and corresponding axis targets.

    Args:
        period_model: Trained period prediction model (can be None if true_period_mix_ratio=1.0)
        dataset: AsteroidDataset instance containing raw and processed data
        num_bins: Number of bins for phase folding
        smooth: Apply smoothing to phase-folded curves
        return_ids: Whether to return asteroid IDs
        use_true_periods: DEPRECATED. Use true_period_mix_ratio instead.
        use_quaternions: Whether to use quaternions for axis targets (4D) or direction vectors (3D)
        true_period_mix_ratio: Fraction of samples for which to use the ground truth period.
                               If < 1.0, period_model must be provided for the remainder.
        logger: Optional logger instance

    Returns:
        tuple: (phase_folded_curves, axis_targets, [asteroid_ids])
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if period_model is None and true_period_mix_ratio < 1.0:
        logger.warning("period_model is None, but true_period_mix_ratio < 1.0. Defaulting to using true periods for all samples.")
        true_period_mix_ratio = 1.0
        
    if use_true_periods and true_period_mix_ratio != 1.0:
        logger.warning("'use_true_periods=True' is set, but true_period_mix_ratio is not 1.0. "
                       "Prioritizing true_period_mix_ratio. Effective true_period_mix_ratio will be 1.0.")
        true_period_mix_ratio = 1.0 # Ensure use_true_periods=True forces all true periods

    phase_folded_curves = []
    axis_targets = []
    asteroid_ids = [] # Only populated if return_ids is True
    
    # Determine device for period_model if it exists
    device = None
    if period_model:
        try:
            device = next(period_model.parameters()).device
            period_model.eval() # Ensure model is in eval mode
        except StopIteration: # Model has no parameters
            logger.warning("Period model has no parameters. Cannot determine device or run predictions.")
            period_model = None # Treat as if no model was provided
            if true_period_mix_ratio < 1.0:
                logger.warning("Cannot use predicted periods as model has no parameters. Defaulting to true_period_mix_ratio = 1.0")
                true_period_mix_ratio = 1.0


    for i in range(len(dataset)):
        try:
            # Handle Subset objects appropriately
            is_subset = isinstance(dataset, torch.utils.data.Subset)
            
            if is_subset:
                original_idx = dataset.indices[i]
                actual_dataset = dataset.dataset
                # AsteroidDataset.get_raw_item returns: (raw_time, raw_mag, raw_err), raw_target_info, asteroid_id, source_file_path
                (time_raw, mag_raw, err_raw), raw_target_info, asteroid_id_str, _ = actual_dataset.get_raw_item(original_idx)
            else:
                # Original behavior for non-Subset datasets
                actual_dataset = dataset # For consistent access to period_scale_factor later
                (time_raw, mag_raw, err_raw), raw_target_info, asteroid_id_str, _ = dataset.get_raw_item(i)
                
            if len(time_raw) < 5: # Minimum points for meaningful phase folding
                logger.debug(f"Skipping asteroid {asteroid_id_str}: not enough data points ({len(time_raw)}).")
                continue
                
            # Use actual_dataset to access attributes like period_scale_factor
            true_period = raw_target_info[2] * actual_dataset.period_scale_factor # De-scale the period
            
            period_to_use = true_period # Default to true period

            if period_model and true_period_mix_ratio < 1.0: # Condition to potentially use predicted period
                if random.random() > true_period_mix_ratio: # Stochastic decision
                    try:
                        # Get processed features for period model prediction
                        if is_subset:
                            processed_features_tensor, _, _ = actual_dataset[original_idx]
                        else:
                            processed_features_tensor, _, _ = dataset[i]
                        
                        # Ensure features are on the same device as the model
                        processed_features_tensor = processed_features_tensor.unsqueeze(0).to(device) # Add batch dim
                        
                        # Placeholder for sequence lengths if model requires it (assuming batch_size 1 for prediction)
                        # This might need adjustment based on the period_model's specific input requirements.
                        # Most of my period models take (data, lengths, ls_features)
                        # Assuming lengths are not critical for single-sample inference or are handled inside model.
                        # For robustness, create a lengths tensor.
                        sequence_lengths = torch.tensor([processed_features_tensor.shape[1]], device=device, dtype=torch.long)
                        
                        # Placeholder for LS features - assuming period_model can handle None if not used
                        ls_features_input = None 

                        with torch.no_grad():
                            prediction_output = period_model(processed_features_tensor, sequence_lengths, ls_features_input)
                        
                        # prediction_output from PeriodLogScaleNet is already actual period.
                        # prediction_output from other models (e.g. PeriodLSTMNet) is also actual period (after F.softplus).
                        if isinstance(prediction_output, tuple):
                            # Handle models like PeriodLSTMWithLSPrior when NOT wrapped by PeriodLogScaleNet
                            # (though typically they would be if use_log_scale is True for them)
                            # Assuming the first element is the primary period prediction.
                            period_to_use = prediction_output[0].item()
                        else:
                            period_to_use = prediction_output.item()
                        
                        # Ensure predicted period is positive
                        if period_to_use <= 0:
                            logger.warning(f"Predicted period for {asteroid_id_str} is <= 0 ({period_to_use:.4f}). Using true period {true_period:.4f} instead.")
                            period_to_use = true_period
                            
                    except Exception as e:
                        logger.error(f"Error predicting period for asteroid {asteroid_id_str}: {e}. Defaulting to true period.")
                        period_to_use = true_period
                # Else (random.random() <= true_period_mix_ratio), period_to_use remains true_period
            elif period_model is None and true_period_mix_ratio < 1.0:
                # This case should be caught by the initial check, but as a safeguard:
                logger.debug(f"Period model is None, using true period for {asteroid_id_str} despite mix_ratio < 1.0.")
                period_to_use = true_period


            # Phase fold using the determined period
            folded_curve = phase_fold(time_raw, mag_raw, period_to_use, num_bins, smooth)
            
            if folded_curve is None or np.sum(folded_curve) == 0: # Check if phase folding failed or resulted in all zeros
                logger.debug(f"Skipping asteroid {asteroid_id_str}: phase folding failed or resulted in zero curve for period {period_to_use:.4f}.")
                continue
        
            phase_folded_curves.append(folded_curve)
            
            # Axis target (l, b from raw_target_info)
            true_l, true_b = raw_target_info[0], raw_target_info[1]
            
            if use_quaternions:
                direction_vec = lon_lat_to_direction_vector(lon_deg=true_l, lat_deg=true_b)
                # Ensure direction_vec is suitable for conversion (e.g., correct shape if it's a single vector)
                # lon_lat_to_direction_vector returns a (3,) array for scalar inputs
                # direction_vector_to_quaternion expects a tensor.
                # If true_l, true_b are scalars, direction_vec will be (3,).
                # We need to ensure it's [1, 3] if the quaternion function expects a batch.
                # However, the existing .squeeze().numpy() suggests it might be okay, 
                # let's assume direction_vector_to_quaternion handles single (3,) tensor.
                quaternion = direction_vector_to_quaternion(torch.from_numpy(direction_vec).float())
                axis_targets.append(quaternion.squeeze().numpy())
            else:
                direction_vec = lon_lat_to_direction_vector(lon_deg=true_l, lat_deg=true_b)
                # Squeeze might be okay if lon_lat_to_direction_vector returns (1,3) for scalar inputs, 
                # but it returns (3,) for scalar inputs. Squeeze on (3,) does nothing.
                # The original code had np.array([[true_l, true_b]]) which would make lon_lat_to_direction_vector return (1,3)
                # If we want a (3,) vector, squeeze is not needed here.
                # Let's assume the target is a flat (3,) vector.
                axis_targets.append(direction_vec) # Assuming direction_vec is already the correct shape (3,)
                
            if return_ids:
                asteroid_ids.append(asteroid_id_str)
        except Exception as e:
            logger.error(f"Error processing asteroid at index {i}: {e}. Skipping.")
            continue

    # Log summary statistics
    logger.info(f"Generated axis data for {len(phase_folded_curves)} out of {len(dataset)} asteroids.")
    
    # Handle empty results
    if len(phase_folded_curves) == 0:
        logger.warning("No valid phase-folded curves were generated. Check dataset and parameters.")
        empty_phase_curve = np.zeros((0, num_bins), dtype=np.float32)
        empty_axis_targets = np.zeros((0, 4 if use_quaternions else 3), dtype=np.float32)
        if return_ids:
            return empty_phase_curve, empty_axis_targets, []
        else:
            return empty_phase_curve, empty_axis_targets

    # Convert to numpy arrays
    phase_folded_curves_np = np.array(phase_folded_curves, dtype=np.float32)
    axis_targets_np = np.array(axis_targets, dtype=np.float32)

    if return_ids:
        return phase_folded_curves_np, axis_targets_np, asteroid_ids
    else:
        return phase_folded_curves_np, axis_targets_np


def normalize_time_magnitude(time, magnitude, error=None):
    """
    Optimized normalization of time and magnitude arrays with vectorized operations.
    
    Args:
        time: Time array
        magnitude: Magnitude array
        error: Error array (optional)
        
    Returns:
        tuple: (time_norm, mag_norm, error_norm)
    """
    # Use NumPy's vectorized operations instead of conditionals
    
    # Normalize time to [0, 1] with single vectorized operation
    time_min, time_max = time.min(), time.max()
    time_range = max(time_max - time_min, 1e-8)  # Avoid division by zero
    time_norm = (time - time_min) / time_range
    
    # Normalize magnitude using vectorized operations
    mag_mean = magnitude.mean()
    mag_std = max(magnitude.std(), 1e-8)  # Avoid division by zero
    mag_norm = (magnitude - mag_mean) / mag_std
    
    # Normalize error if provided (vectorized)
    if error is not None and len(error) > 0:
        error_max = max(error.max(), 1e-8)  # Avoid division by zero
        error_norm = error / error_max
        return time_norm, mag_norm, error_norm
    
    return time_norm, mag_norm, None


def augment_light_curve(time, magnitude, error=None, scale_factor=None, time_shift=None, add_noise=False, noise_level=0.01):
    """
    Efficiently augment a light curve with scaling, shifting, and noise using vectorized operations.
    
    Args:
        time: Time array
        magnitude: Magnitude array
        error: Error array (optional)
        scale_factor: Factor to scale the time axis (optional)
        time_shift: Amount to shift the time axis (optional)
        add_noise: Whether to add random noise
        noise_level: Level of noise to add
        
    Returns:
        tuple: (time_aug, mag_aug, error_aug)
    """
    # Create views instead of copies where possible for memory efficiency
    time_aug = time
    mag_aug = magnitude
    error_aug = error
    
    # Apply time scaling with vectorized operation if needed
    if scale_factor is not None and scale_factor > 0:
        # Now we need a copy because we're modifying
        time_aug = time * scale_factor
    
    # Apply time shift with vectorized operation if needed
    if time_shift is not None:
        # Only create a copy if we didn't already in the scaling step
        if scale_factor is None or scale_factor <= 0:
            time_aug = time.copy()
        time_aug += time_shift
    elif scale_factor is None or scale_factor <= 0:
        # If no modifications have been made, we need to create a copy
        time_aug = time.copy()
    
    # Add noise to magnitude with vectorized operation
    if add_noise and noise_level > 0:
        # Create a copy of magnitude since we'll modify it
        mag_aug = magnitude.copy()
        
        # Generate noise vector and add in one operation
        noise = np.random.normal(0, noise_level, size=mag_aug.shape)
        mag_aug += noise
        
        # Update error if provided using vectorized operations
        if error_aug is not None:
            error_aug = np.sqrt(error**2 + noise_level**2)
    else:
        # No noise modification, but we still need a copy
        mag_aug = magnitude.copy()
        if error is not None:
            error_aug = error.copy()
    
    return time_aug, mag_aug, error_aug


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    
    # Generate synthetic light curve
    time = np.linspace(0, 10, 100)
    period = 2.0
    magnitude = np.sin(2 * np.pi * time / period) + 0.1 * np.random.randn(100)
    
    # Phase fold
    phase_folded = phase_fold(time, magnitude, period, num_bins=50)
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.scatter(time, magnitude, alpha=0.7)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Magnitude")
    ax1.set_title("Original Light Curve")
    
    ax2.plot(np.linspace(0, 1, len(phase_folded), endpoint=False), phase_folded)
    ax2.set_xlabel("Phase")
    ax2.set_ylabel("Magnitude")
    ax2.set_title(f"Phase-folded (P={period:.2f})")
    
    plt.tight_layout()
    plt.show() 