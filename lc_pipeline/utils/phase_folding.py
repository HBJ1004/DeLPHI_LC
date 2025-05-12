#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase folding utilities for lightcurve analysis.
"""

import numpy as np
from scipy.interpolate import interp1d


def phase_fold(
    time,
    magnitude,
    period,
    num_bins=100,
    smooth=True,
    error=None,
    reference_time=None
):
    """
    Phase-fold a time series using the given period.
    
    Args:
        time (numpy.ndarray): Time values
        magnitude (numpy.ndarray): Magnitude values
        period (float): Folding period
        num_bins (int): Number of bins for the folded light curve
        smooth (bool): Whether to smooth the folded curve
        error (numpy.ndarray, optional): Error values for the magnitudes
        reference_time (float, optional): Reference time for phase 0
        
    Returns:
        numpy.ndarray: Binned phase-folded light curve of shape [num_bins]
    """
    # Validate inputs
    if len(time) != len(magnitude):
        raise ValueError("time and magnitude must have the same length")
    
    if len(time) == 0:
        # Return empty array for empty inputs
        return np.zeros(num_bins)
    
    # Handle reference time
    if reference_time is None:
        reference_time = time[0]
    
    # Calculate phases (0 to 1)
    phases = ((time - reference_time) / period) % 1.0
    
    # Create uniform phase bins
    bin_edges = np.linspace(0, 1, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    if smooth and len(phases) >= 4:
        try:
            # Use interpolation for a smooth curve
            # Sort by phase for interpolation
            sorted_indices = np.argsort(phases)
            sorted_phases = phases[sorted_indices]
            sorted_mags = magnitude[sorted_indices]
            
            # Create a cyclic version by duplicating with phase+1
            cyclic_phases = np.concatenate([sorted_phases, sorted_phases + 1])
            cyclic_mags = np.concatenate([sorted_mags, sorted_mags])
            
            # Create interpolation function
            interp_func = interp1d(
                cyclic_phases,
                cyclic_mags,
                kind='linear',
                bounds_error=False,
                fill_value='extrapolate'
            )
            
            # Apply to bin centers
            folded_curve = interp_func(bin_centers)
            
        except Exception as e:
            # Fallback to binning if interpolation fails
            print(f"Interpolation failed: {e}, falling back to binning")
            folded_curve = np.zeros(num_bins)
            for i in range(num_bins):
                bin_mask = (phases >= bin_edges[i]) & (phases < bin_edges[i+1])
                if np.any(bin_mask):
                    folded_curve[i] = np.mean(magnitude[bin_mask])
                else:
                    # For empty bins, use neighboring values
                    left_val = folded_curve[i-1] if i > 0 else 0
                    folded_curve[i] = left_val
    else:
        # Use simple binning
        folded_curve = np.zeros(num_bins)
        for i in range(num_bins):
            bin_mask = (phases >= bin_edges[i]) & (phases < bin_edges[i+1])
            if np.any(bin_mask):
                if error is not None:
                    # Weighted average if errors are provided
                    weights = 1.0 / (error[bin_mask]**2 + 1e-10)
                    folded_curve[i] = np.average(magnitude[bin_mask], weights=weights)
                else:
                    folded_curve[i] = np.mean(magnitude[bin_mask])
            else:
                # For empty bins, interpolate from neighbors
                left_idx = (i - 1) % num_bins
                right_idx = (i + 1) % num_bins
                if np.isclose(folded_curve[left_idx], 0) and np.isclose(folded_curve[right_idx], 0):
                    folded_curve[i] = 0
                elif np.isclose(folded_curve[left_idx], 0):
                    folded_curve[i] = folded_curve[right_idx]
                elif np.isclose(folded_curve[right_idx], 0):
                    folded_curve[i] = folded_curve[left_idx]
                else:
                    folded_curve[i] = (folded_curve[left_idx] + folded_curve[right_idx]) / 2
    
    # Normalize if needed (zero mean)
    if not np.all(np.isclose(folded_curve, 0)):
        folded_curve = folded_curve - np.mean(folded_curve)
    
    return folded_curve 