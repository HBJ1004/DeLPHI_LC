#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization utilities for the light curve pipeline.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import seaborn as sns
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
from sklearn.metrics import r2_score
from typing import List, Dict, Optional, Tuple, Any
import logging

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context('notebook')

def setup_figure(figsize=(10, 6), title=None, suptitle=None):
    """Set up a figure with given size and title."""
    fig = plt.figure(figsize=figsize)
    if title:
        fig.suptitle(title, fontsize=16)
    if suptitle:
        plt.title(suptitle, fontsize=14)
    return fig

def add_colorbar(ax, im, label=None):
    """Add a colorbar to a plot."""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    if label:
        cbar.set_label(label)
    return cbar

def save_figure(fig, filename, dpi=300, bbox_inches='tight'):
    """Save a figure to a file."""
    fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
    print(f"Figure saved to {filename}")

##############################################################################
# Period Visualization Functions
##############################################################################

def plot_ls_period_distribution(ls_periods, figsize=(10, 6), bins=30, log_scale=True, 
                                title="Distribution of Lomb-Scargle Periods", 
                                save_path=None):
    """
    Plot the distribution of Lomb-Scargle periods.
    
    Args:
        ls_periods: Array of Lomb-Scargle periods
        figsize: Figure size tuple
        bins: Number of histogram bins
        log_scale: Whether to use log scale for x-axis
        title: Plot title
        save_path: Path to save the figure (optional)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if log_scale and np.all(np.array(ls_periods) > 0):
        ax.set_xscale('log')
        bins = np.logspace(np.log10(min(ls_periods)), np.log10(max(ls_periods)), bins)
    
    ax.hist(ls_periods, bins=bins, alpha=0.7, color='royalblue', edgecolor='black')
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Period (hours)" if not log_scale else "Period (hours, log scale)")
    ax.set_ylabel("Count")
    
    # Add statistical annotations
    stats_text = f"Mean: {np.mean(ls_periods):.2f} h\n"
    stats_text += f"Median: {np.median(ls_periods):.2f} h\n"
    stats_text += f"Min: {np.min(ls_periods):.2f} h\n"
    stats_text += f"Max: {np.max(ls_periods):.2f} h"
    
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig, ax

def plot_phase_folded_lightcurves(time, brightness, periods, period_labels=None, 
                                  num_samples=3, figsize=(12, 10), save_path=None):
    """
    Plot examples of phase-folded lightcurves using different periods.
    
    Args:
        time: List of time arrays, one per lightcurve
        brightness: List of brightness arrays, one per lightcurve
        periods: List of period arrays, where each array contains different period values for the same lightcurve
        period_labels: Labels for different periods (e.g., ["Ground Truth", "Lomb-Scargle", "Predicted"])
        num_samples: Number of lightcurve samples to show
        figsize: Figure size tuple
        save_path: Path to save the figure (optional)
    """
    if not period_labels:
        period_labels = [f"Period {i+1}" for i in range(len(periods[0]))]
    
    n_periods = len(periods[0])
    n_samples = min(num_samples, len(time))
    
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(n_samples, n_periods)
    
    for i in range(n_samples):
        for j in range(n_periods):
            ax = fig.add_subplot(gs[i, j])
            
            # Calculate phase
            period = periods[i][j]
            phase = (time[i] % period) / period
            
            # Sort by phase for better visualization
            sorted_idx = np.argsort(phase)
            sorted_phase = phase[sorted_idx]
            sorted_brightness = brightness[i][sorted_idx]
            
            # Plot two cycles for better visualization
            ax.scatter(sorted_phase, sorted_brightness, s=3, alpha=0.7, color='royalblue')
            ax.scatter(sorted_phase + 1, sorted_brightness, s=3, alpha=0.7, color='royalblue')
            
            ax.set_xlim(0, 2)
            ax.set_xlabel("Phase")
            
            if j == 0:
                ax.set_ylabel("Brightness")
            
            if i == 0:
                ax.set_title(f"{period_labels[j]}\nP = {period:.2f} h", fontsize=12)
            else:
                ax.set_title(f"P = {period:.2f} h", fontsize=12)
    
    plt.tight_layout()
    fig.suptitle("Phase-Folded Lightcurves", fontsize=16, y=1.02)
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig

def plot_period_scatter(
    true_periods: List[float],
    pred_periods: List[float],
    ls_periods: Optional[List[float]] = None,
    title: str = "Predicted vs. True Period",
    save_path: Optional[str] = None,
    logger: Optional[Any] = None,
    **kwargs
):
    """
    Plots a scatter plot of predicted vs. true periods.
    
    Args:
        true_periods: List of ground truth periods
        pred_periods: List of predicted periods
        ls_periods: List of Lomb-Scargle periods (optional)
        title: Plot title
        save_path: Path to save the figure (optional)
        logger: Logger object for error handling (optional)
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate metrics (MAE, RMSE)
    if not true_periods or not pred_periods:
        if logger:
            logger.warning("Cannot calculate metrics for period scatter plot: empty data.")
        mae, rmse = float('nan'), float('nan')
    else:
        # Convert lists to numpy arrays for calculations
        true_periods_np = np.array(true_periods)
        pred_periods_np = np.array(pred_periods)

        try:
            mae = np.mean(np.abs(true_periods_np - pred_periods_np))
            rmse = np.sqrt(np.mean((true_periods_np - pred_periods_np) ** 2))
        except TypeError as e:
            if logger:
                logger.error(f"Error calculating period metrics: {e}")

    # Create scatter plot
    scatter = ax.scatter(true_periods_np, pred_periods_np, alpha=0.7, 
                         c=np.abs(true_periods_np - pred_periods_np), cmap='viridis')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Absolute Error (hours)')
    
    # Add identity line
    min_val = np.min([np.min(true_periods_np), np.min(pred_periods_np)])
    max_val = np.max([np.max(true_periods_np), np.max(pred_periods_np)])
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', label='Identity', alpha=0.7)

    # Add metrics to the plot
    ax.text(0.05, 0.95, f'MAE: {mae:.4f}\nRMSE: {rmse:.4f}',
             transform=ax.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

    # Set labels and title
    ax.set_xlabel('True Period (hours)')
    ax.set_ylabel('Predicted Period (hours)')
    ax.set_title(title, fontsize=14)
    
    # Add legend
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig, ax

def plot_period_error_histogram(true_periods, pred_periods, figsize=(10, 6), bins=30,
                                log_error=True, title="Distribution of Period Prediction Errors",
                                save_path=None, logger=None):
    """
    Create a histogram of period prediction errors.
    
    Args:
        true_periods: Array of ground truth periods
        pred_periods: Array of predicted periods
        figsize: Figure size tuple
        bins: Number of histogram bins
        log_error: Whether to use log(pred/true) as the error measure
        title: Plot title
        save_path: Path to save the figure (optional)
        logger: Logger object for warnings (optional)
    """
    if logger is None:
        logger = logging.getLogger(__name__) # Use standard logging if no logger provided

    fig, ax = plt.subplots(figsize=figsize)

    # --- Input Validation ---
    try:
        # Attempt conversion to numpy array and ensure numeric type
        true_periods_np = np.asarray(true_periods, dtype=float)
        pred_periods_np = np.asarray(pred_periods, dtype=float)

        # Robust Flattening: Ensure data is truly 1D
        true_periods_np = np.ravel(true_periods_np)
        pred_periods_np = np.ravel(pred_periods_np)

        # Re-check dimensions after flattening
        if true_periods_np.ndim != 1 or pred_periods_np.ndim != 1:
            raise ValueError(f"Flattening failed: true_periods dim={true_periods_np.ndim}, pred_periods dim={pred_periods_np.ndim}")

        # Ensure they have the same shape after flattening
        if true_periods_np.shape != pred_periods_np.shape:
            raise ValueError(f"Shape mismatch after flattening: true={true_periods_np.shape}, pred={pred_periods_np.shape}")

    except (ValueError, TypeError) as e:
        logger.error(f"Invalid input for plot_period_error_histogram: {e}. Skipping plot.")
        # Add more detailed debug info
        logger.debug(f"Input types: true={type(true_periods)}, pred={type(pred_periods)}")
        try:
            logger.debug(f"Shape after initial conversion: true={np.asarray(true_periods, dtype=object).shape}, pred={np.asarray(pred_periods, dtype=object).shape}")
            logger.debug(f"Shape after np.ravel: true={np.ravel(np.asarray(true_periods, dtype=float)).shape}, pred={np.ravel(np.asarray(pred_periods, dtype=float)).shape}")
        except Exception as shape_err:
            logger.debug(f"Could not determine shapes for debugging: {shape_err}")
        # Optionally log snippets
        # logger.debug(f"True periods sample: {str(true_periods)[:100]}")
        # logger.debug(f"Pred periods sample: {str(pred_periods)[:100]}")
        plt.close(fig) # Close the empty figure
        return fig, ax # Return the empty figure/axis objects

    # --- Calculation ---
    if log_error:
        # Use log ratio for error, handling possible zeros or negatives
        eps = 1e-8
        # Filter out non-positive values before log
        valid_mask = (true_periods_np > eps) & (pred_periods_np > eps)
        if not np.any(valid_mask):
            logger.warning("No valid positive period pairs for log_error calculation in histogram. Skipping.")
            plt.close(fig)
            return fig, ax
        
        errors = np.log10(pred_periods_np[valid_mask] / true_periods_np[valid_mask])
        xlabel = "log₁₀(Predicted Period / True Period)"
    else:
        errors = pred_periods_np - true_periods_np
        xlabel = "Predicted Period - True Period (hours)"
    
    # Check if errors array is empty after filtering
    if errors.size == 0:
         logger.warning("No errors calculated (possibly due to filtering). Skipping histogram plot.")
         plt.close(fig)
         return fig, ax

    # Create histogram
    ax.hist(errors, bins=bins, alpha=0.7, color='royalblue', edgecolor='black')
    
    # Add vertical line at zero error
    ax.axvline(x=0, color='r', linestyle='--', alpha=0.7)
    
    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title, fontsize=14)
    
    # Add statistical annotations
    stats_text = f"Mean Error: {np.mean(errors):.4f}\n"
    stats_text += f"Median Error: {np.median(errors):.4f}\n"
    stats_text += f"Std Dev: {np.std(errors):.4f}"
    
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig, ax

def plot_period_metrics_vs_epoch(train_loss, val_loss, val_mae=None, val_rmse=None, 
                                figsize=(12, 8), title="Period Model Training Metrics",
                                save_path=None):
    """
    Plot period model training metrics over epochs.
    
    Args:
        train_loss: Array of training loss values per epoch
        val_loss: Array of validation loss values per epoch
        val_mae: Array of validation MAE values per epoch (optional)
        val_rmse: Array of validation RMSE values per epoch (optional)
        figsize: Figure size tuple
        title: Plot title
        save_path: Path to save the figure (optional)
    """
    fig = plt.figure(figsize=figsize)
    
    # Determine number of subplots
    has_mae = val_mae is not None
    has_rmse = val_rmse is not None
    n_plots = 1 + has_mae + has_rmse
    
    gs = GridSpec(n_plots, 1, height_ratios=[1] * n_plots)
    
    # Loss plot
    ax1 = fig.add_subplot(gs[0])
    epochs = range(1, len(train_loss) + 1)
    
    ax1.plot(epochs, train_loss, 'b-', marker='o', markersize=4, label='Training Loss')
    ax1.plot(epochs, val_loss, 'r-', marker='s', markersize=4, label='Validation Loss')
    
    ax1.set_title("Loss vs. Epoch", fontsize=12)
    ax1.set_xlabel("Epoch" if n_plots == 1 else "")
    ax1.set_ylabel("Loss")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # MAE plot
    if has_mae:
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(epochs, val_mae, 'g-', marker='d', markersize=4)
        ax2.set_title("Mean Absolute Error vs. Epoch", fontsize=12)
        ax2.set_xlabel("Epoch" if n_plots == 2 or not has_rmse else "")
        ax2.set_ylabel("MAE (hours)")
        ax2.grid(True, alpha=0.3)
    
    # RMSE plot
    if has_rmse:
        ax3 = fig.add_subplot(gs[2 if has_mae else 1])
        ax3.plot(epochs, val_rmse, 'c-', marker='^', markersize=4)
        ax3.set_title("Root Mean Squared Error vs. Epoch", fontsize=12)
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("RMSE (hours)")
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.suptitle(title, fontsize=16, y=1.02)
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig 