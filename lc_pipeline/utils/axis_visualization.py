#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Axis prediction visualization utilities for the light curve pipeline.
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
import matplotlib.colors as colors
from typing import Optional, List, Any
import os
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

def save_figure(fig, filename, dpi=300, bbox_inches='tight'):
    """Save a figure to a file."""
    fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
    print(f"Figure saved to {filename}")

def quaternion_to_lb(quaternion):
    """
    Convert quaternion representation to ecliptic longitude and latitude.
    
    Args:
        quaternion: Array or list of shape (4,) with quaternion components [w, x, y, z]
        
    Returns:
        tuple: (longitude, latitude) in degrees
    """
    qw, qx, qy, qz = quaternion
    
    # Rotation matrix from quaternion
    r11 = 1 - 2*qy**2 - 2*qz**2
    r21 = 2*qx*qy - 2*qz*qw
    r31 = 2*qx*qz + 2*qy*qw
    
    r32 = 2*qy*qz - 2*qx*qw
    r33 = 1 - 2*qx**2 - 2*qy**2
    
    # Calculate longitude and latitude from rotation matrix
    longitude = np.arctan2(r21, r11)
    latitude = np.arcsin(r31)
    
    # Convert to degrees
    longitude_deg = np.degrees(longitude) % 360
    latitude_deg = np.degrees(latitude)
    
    return longitude_deg, latitude_deg

def direction_vector_to_lb(direction_vector):
    """
    Convert a 3D direction vector (x, y, z) to longitude and latitude.

    Assumes the input vector(s) are normalized or near-normalized.

    Args:
        direction_vector: A numpy array of shape (3,) or (N, 3) representing
                          the direction vector(s) [x, y, z].

    Returns:
        tuple: (longitude, latitude) in degrees. If input is (N, 3),
               returns (lon_array, lat_array) both of shape (N,).
    """
    direction_vector = np.asarray(direction_vector)
    
    if direction_vector.ndim == 1:
        if direction_vector.shape[0] != 3:
            raise ValueError("Single direction vector must have shape (3,)")
        x, y, z = direction_vector
        # Ensure normalization for safety, although asin(z) works for unit vectors
        norm = np.sqrt(x**2 + y**2 + z**2)
        if norm < 1e-8:
             # Handle zero vector case - return (0, 0) or NaN?
             return 0.0, 0.0 
        z = np.clip(z / norm, -1.0, 1.0) # Clip z/norm for asin robustness
        
        longitude = np.arctan2(y, x)
        latitude = np.arcsin(z) # Latitude is the angle from the xy-plane
        
    elif direction_vector.ndim == 2:
        if direction_vector.shape[1] != 3:
            raise ValueError("Direction vector array must have shape (N, 3)")
        x = direction_vector[:, 0]
        y = direction_vector[:, 1]
        z = direction_vector[:, 2]
        
        # Ensure normalization for safety
        norm = np.sqrt(x**2 + y**2 + z**2)
        # Avoid division by zero for zero vectors
        safe_norm = np.where(norm < 1e-8, 1.0, norm)
        z_normalized = np.clip(z / safe_norm, -1.0, 1.0)
        
        longitude = np.arctan2(y, x)
        latitude = np.arcsin(z_normalized)
        
        # Handle zero vectors explicitly if needed (e.g., return NaN or 0)
        longitude = np.where(norm < 1e-8, 0.0, longitude)
        latitude = np.where(norm < 1e-8, 0.0, latitude)
        
    else:
        raise ValueError("Input must be a 1D array (3,) or a 2D array (N, 3)")

    # Convert to degrees
    longitude_deg = np.degrees(longitude)
    # Ensure longitude is in [0, 360)
    longitude_deg = longitude_deg % 360 
    latitude_deg = np.degrees(latitude)
    
    return longitude_deg, latitude_deg

def angular_distance(l1, b1, l2, b2):
    """
    Calculate angular distance between two points on a sphere.
    
    Args:
        l1, b1: Longitude and latitude of first point in degrees
        l2, b2: Longitude and latitude of second point in degrees
        
    Returns:
        float: Angular distance in degrees
    """
    # Convert to radians
    l1_rad, b1_rad = np.radians(l1), np.radians(b1)
    l2_rad, b2_rad = np.radians(l2), np.radians(b2)
    
    # Compute angular distance using haversine formula
    cos_angle = (np.sin(b1_rad) * np.sin(b2_rad) + 
                 np.cos(b1_rad) * np.cos(b2_rad) * np.cos(l1_rad - l2_rad))
    # Clip to avoid numerical errors
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    
    return np.degrees(angle_rad)

def plot_axis_scatter(
    true_longitudes: Optional[List[float]],
    true_latitudes: Optional[List[float]],
    pred_longitudes: Optional[List[float]],
    pred_latitudes: Optional[List[float]],
    epoch: Optional[int] = None,
    title_prefix: str = "",
    save_dir: Optional[str] = None,
    logger: Optional[Any] = None,
    **kwargs
):
    """
    Plots a scatter plot of predicted vs. true axis orientations (longitude/latitude).
    Also plots true vs. predicted kappa values if available.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Ensure inputs are lists before the check. Default to empty list if None.
    # These are the arguments to the function
    safe_true_lon = true_longitudes if true_longitudes is not None else []
    safe_true_lat = true_latitudes if true_latitudes is not None else []
    safe_pred_lon = pred_longitudes if pred_longitudes is not None else []
    safe_pred_lat = pred_latitudes if pred_latitudes is not None else []

    # Use these safe versions in the check
    # Original line 61 (approximately)
    if not all(isinstance(arr, (list, np.ndarray)) and len(arr) > 0 for arr in [safe_true_lon, safe_true_lat, safe_pred_lon, safe_pred_lat]):
        if logger:
            logger.warning(
                "Cannot generate axis scatter plot: input data for lon/lat is invalid or empty (after None check)."
            )
        return

    # Convert to numpy arrays using the safe versions
    # These are now local variables within the function
    np_true_lon = np.array(safe_true_lon)
    np_true_lat = np.array(safe_true_lat)
    np_pred_lon = np.array(safe_pred_lon)
    np_pred_lat = np.array(safe_pred_lat)

    # Extract kappa values if provided in kwargs
    true_kappas = kwargs.get('true_kappas', None)
    pred_kappas = kwargs.get('pred_kappas', None)

    safe_true_kappas = true_kappas if true_kappas is not None else []
    safe_pred_kappas = pred_kappas if pred_kappas is not None else []
    
    np_true_kappas = np.array(safe_true_kappas) if safe_true_kappas else None
    np_pred_kappas = np.array(safe_pred_kappas) if safe_pred_kappas else None

    # Determine number of subplots needed
    num_subplots = 1
    if np_true_kappas is not None and np_pred_kappas is not None and len(np_true_kappas) > 0 and len(np_pred_kappas) > 0:
        num_subplots = 2
        fig_size = (16, 6)
    else:
        fig_size = (8, 6)

    fig, axes = plt.subplots(1, num_subplots, figsize=fig_size, squeeze=False) # squeeze=False to always get an array of axes
    ax1 = axes[0, 0]

    # Scatter plot for longitude and latitude
    # Ensure all coordinate arrays have the same length for plotting
    min_len = min(len(np_true_lon), len(np_true_lat), len(np_pred_lon), len(np_pred_lat))
    if min_len == 0:
        if logger:
            logger.warning("Cannot plot axis scatter: one or more coordinate arrays are empty after processing.")
        plt.close(fig)
        return
        
    np_true_lon = np_true_lon[:min_len]
    np_true_lat = np_true_lat[:min_len]
    np_pred_lon = np_pred_lon[:min_len]
    np_pred_lat = np_pred_lat[:min_len]

    # Calculate angular errors
    errors = angular_distance(np_true_lon, np_true_lat, np_pred_lon, np_pred_lat)
    mae = np.mean(np.abs(errors)) if errors.size > 0 else float('nan')
    median_error = np.median(np.abs(errors)) if errors.size > 0 else float('nan')
    std_error = np.std(errors) if errors.size > 0 else float('nan')
    success_rate_15 = np.mean(errors <= 15) * 100 if errors.size > 0 else float('nan')
    success_rate_30 = np.mean(errors <= 30) * 100 if errors.size > 0 else float('nan')
    success_rate_45 = np.mean(errors <= 45) * 100 if errors.size > 0 else float('nan')

    scatter = ax1.scatter(np_true_lon, np_pred_lon, c=errors, cmap='viridis', alpha=0.6, label='Longitude')
    ax1.scatter(np_true_lat, np_pred_lat, c=errors, cmap='coolwarm', alpha=0.6, marker='x', label='Latitude')
    
    # Add 1:1 line
    min_val_lon = min(np_true_lon.min() if np_true_lon.size > 0 else 0, np_pred_lon.min() if np_pred_lon.size > 0 else 0)
    max_val_lon = max(np_true_lon.max() if np_true_lon.size > 0 else 0, np_pred_lon.max() if np_pred_lon.size > 0 else 0)
    ax1.plot([min_val_lon, max_val_lon], [min_val_lon, max_val_lon], 'r--', lw=2, label='Ideal (Lon)')

    min_val_lat = min(np_true_lat.min() if np_true_lat.size > 0 else 0, np_pred_lat.min() if np_pred_lat.size > 0 else 0)
    max_val_lat = max(np_true_lat.max() if np_true_lat.size > 0 else 0, np_pred_lat.max() if np_pred_lat.size > 0 else 0)
    ax1.plot([min_val_lat, max_val_lat], [min_val_lat, max_val_lat], 'b:', lw=2, label='Ideal (Lat)')

    ax1.set_xlabel("True Value (degrees)")
    ax1.set_ylabel("Predicted Value (degrees)")
    plot_title_ax1 = f"{title_prefix}Axis Orientation Scatter"
    if epoch is not None:
        plot_title_ax1 += f" (Epoch {epoch})"
    
    metrics_text_ax1 = (f"MAE: {mae:.2f}\\nMedian Err: {median_error:.2f}\\nStd Err: {std_error:.2f}\\n"
                        f"SR@15: {success_rate_15:.1f}%\\nSR@30: {success_rate_30:.1f}%\\nSR@45: {success_rate_45:.1f}%")
    ax1.set_title(f"{plot_title_ax1}\\n{metrics_text_ax1}", fontsize=10)
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.7)
    fig.colorbar(scatter, ax=ax1, label='Angular Error (degrees)')

    if num_subplots == 2:
        ax2 = axes[0, 1]
        if np_true_kappas is not None and np_pred_kappas is not None and len(np_true_kappas) > 0 and len(np_pred_kappas) > 0:
            min_kappa_len = min(len(np_true_kappas), len(np_pred_kappas))
            np_true_kappas = np_true_kappas[:min_kappa_len]
            np_pred_kappas = np_pred_kappas[:min_kappa_len]

            kappa_errors = np.abs(np_true_kappas - np_pred_kappas)
            kappa_mae = np.mean(kappa_errors) if kappa_errors.size > 0 else float('nan')
            
            scatter_kappa = ax2.scatter(np_true_kappas, np_pred_kappas, c=kappa_errors, cmap='plasma', alpha=0.6)
            min_k_val = min(np_true_kappas.min() if np_true_kappas.size > 0 else 0, np_pred_kappas.min() if np_pred_kappas.size > 0 else 0)
            max_k_val = max(np_true_kappas.max() if np_true_kappas.size > 0 else 1, np_pred_kappas.max() if np_pred_kappas.size > 0 else 1) # Avoid min=max=0
            ax2.plot([min_k_val, max_k_val], [min_k_val, max_k_val], 'g--', lw=2)
            ax2.set_xlabel("True Kappa")
            ax2.set_ylabel("Predicted Kappa")
            plot_title_ax2 = f"{title_prefix}Kappa Scatter"
            if epoch is not None:
                plot_title_ax2 += f" (Epoch {epoch})"
            ax2.set_title(f"{plot_title_ax2}\\nMAE: {kappa_mae:.2f}", fontsize=10)
            ax2.grid(True, linestyle=':', alpha=0.7)
            fig.colorbar(scatter_kappa, ax=ax2, label='|True Kappa - Pred Kappa|')
        else:
            ax2.text(0.5, 0.5, "Kappa data not available\nor insufficient for plotting.",
                     horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
            ax2.set_title(f"{title_prefix}Kappa Scatter (Data N/A)")


    fig.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
    overall_title = f"{title_prefix}Axis Evaluation Scatter Plots"
    if epoch is not None:
        overall_title += f" - Epoch {epoch}"
    fig.suptitle(overall_title, fontsize=14)
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{title_prefix.lower().replace(' ', '_')}axis_scatter_epoch_{epoch if epoch is not None else 'final'}.png"
        save_path = os.path.join(save_dir, filename)
        try:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
            if logger: logger.info(f"Saved axis scatter plot to {save_path}")
        except Exception as e:
            if logger: logger.error(f"Failed to save axis scatter plot: {e}")
    else:
        if logger: logger.info("No save directory provided for axis scatter plot.")
    
    plt.close(fig)

def plot_angular_error_histogram(angular_errors, figsize=(10, 6), bins=30,
                                threshold=30.0, title="Distribution of Angular Errors",
                                save_path=None):
    """
    Create a histogram of angular errors between predicted and true axes.
    
    Args:
        angular_errors: Array of angular errors in degrees
        figsize: Figure size tuple
        bins: Number of histogram bins
        threshold: Success threshold in degrees
        title: Plot title
        save_path: Path to save the figure (optional)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create histogram
    counts, edges, patches = ax.hist(angular_errors, bins=bins, alpha=0.7, 
                                     color='royalblue', edgecolor='black')
    
    # Color bars based on threshold
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    cmap = plt.cm.RdYlGn_r
    norm = colors.Normalize(0, max(threshold * 2, max(angular_errors)))
    
    for i, patch in enumerate(patches):
        color = cmap(norm(bin_centers[i]))
        patch.set_facecolor(color)
    
    # Add vertical line at threshold
    ax.axvline(x=threshold, color='r', linestyle='--', alpha=0.7)
    
    # Set labels and title
    ax.set_xlabel('Angular Error (degrees)')
    ax.set_ylabel('Count')
    ax.set_title(title, fontsize=14)
    
    # Calculate success rate
    success_rate = 100 * np.mean(np.array(angular_errors) < threshold)
    
    # Add statistical annotations
    stats_text = f"Mean Error: {np.mean(angular_errors):.2f}°\n"
    stats_text += f"Median Error: {np.median(angular_errors):.2f}°\n"
    stats_text += f"Success Rate (<{threshold}°): {success_rate:.2f}%"
    
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig, ax

def plot_axis_sky_map(true_l, true_b, pred_l, pred_b, figsize=(12, 8),
                     title="Sky Map of True and Predicted Axis Orientations",
                     save_path=None):
    """
    Create an Aitoff projection sky map showing true and predicted axis orientations.
    
    Args:
        true_l: Array of ground truth longitudes (degrees)
        true_b: Array of ground truth latitudes (degrees)
        pred_l: Array of predicted longitudes (degrees)
        pred_b: Array of predicted latitudes (degrees)
        figsize: Figure size tuple
        title: Plot title
        save_path: Path to save the figure (optional)
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='aitoff')
    
    # Convert longitude to range [-pi, pi] for Aitoff projection
    true_l_rad = np.radians(true_l)
    true_l_rad = np.where(true_l_rad > np.pi, true_l_rad - 2*np.pi, true_l_rad)
    
    pred_l_rad = np.radians(pred_l)
    pred_l_rad = np.where(pred_l_rad > np.pi, pred_l_rad - 2*np.pi, pred_l_rad)
    
    true_b_rad = np.radians(true_b)
    pred_b_rad = np.radians(pred_b)
    
    # Calculate angular errors
    angular_errors = np.array([angular_distance(true_l[i], true_b[i], pred_l[i], pred_b[i]) 
                              for i in range(len(true_l))])
    
    # Plot true positions
    ax.scatter(true_l_rad, true_b_rad, s=50, marker='o', color='blue', alpha=0.7, label='True')
    
    # Plot predicted positions
    scatter = ax.scatter(pred_l_rad, pred_b_rad, s=50, marker='x', c=angular_errors, 
                        cmap='RdYlGn_r', alpha=0.7, label='Predicted')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, pad=0.1)
    cbar.set_label('Angular Error (degrees)')
    
    # Connect true and predicted points with lines
    for i in range(len(true_l)):
        ax.plot([true_l_rad[i], pred_l_rad[i]], [true_b_rad[i], pred_b_rad[i]], 
                'k-', alpha=0.3, linewidth=0.5)
    
    # Set grid and labels
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Longitude (degrees)')
    ax.set_ylabel('Latitude (degrees)')
    
    # Convert x-axis labels from radians to degrees
    xticks_rad = np.array([-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150]) * np.pi / 180
    xticks_deg = [f"{int(x)}°" for x in [-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150]]
    plt.xticks(xticks_rad, xticks_deg)
    
    # Convert y-axis labels from radians to degrees
    yticks_rad = np.array([-60, -30, 0, 30, 60]) * np.pi / 180
    yticks_deg = [f"{int(y)}°" for y in [-60, -30, 0, 30, 60]]
    plt.yticks(yticks_rad, yticks_deg)
    
    # Add legend
    ax.legend(loc='upper right')
    
    plt.title(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig, ax

def plot_axis_metrics_vs_epoch(train_losses, val_losses, mean_angular_errors=None, 
                              success_rates=None, threshold=30.0, figsize=(12, 10),
                              title="Axis Model Training Metrics", save_path=None):
    """
    Plot axis model training metrics over epochs.
    
    Args:
        train_losses: Array of training loss values per epoch
        val_losses: Array of validation loss values per epoch
        mean_angular_errors: Array of mean angular errors per epoch (optional)
        success_rates: Array of success rates per epoch (optional)
        threshold: Success threshold in degrees
        figsize: Figure size tuple
        title: Plot title
        save_path: Path to save the figure (optional)
    """
    # Determine number of subplots
    has_errors = mean_angular_errors is not None
    has_rates = success_rates is not None
    n_plots = 1 + has_errors + has_rates
    
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(n_plots, 1, height_ratios=[1] * n_plots)
    
    # Loss plot
    ax1 = fig.add_subplot(gs[0])
    epochs = range(1, len(train_losses) + 1)
    
    ax1.plot(epochs, train_losses, 'b-', marker='o', markersize=4, label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', marker='s', markersize=4, label='Validation Loss')
    
    ax1.set_title("Loss vs. Epoch", fontsize=12)
    ax1.set_xlabel("Epoch" if n_plots == 1 else "")
    ax1.set_ylabel("Loss")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Angular error plot
    if has_errors:
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(epochs, mean_angular_errors, 'g-', marker='d', markersize=4)
        ax2.set_title("Mean Angular Error vs. Epoch", fontsize=12)
        ax2.set_xlabel("Epoch" if n_plots == 2 or not has_rates else "")
        ax2.set_ylabel("Mean Angular Error (degrees)")
        ax2.grid(True, alpha=0.3)
    
    # Success rate plot
    if has_rates:
        ax3 = fig.add_subplot(gs[2 if has_errors else 1])
        ax3.plot(epochs, success_rates, 'c-', marker='^', markersize=4)
        ax3.set_title(f"Success Rate (<{threshold}°) vs. Epoch", fontsize=12)
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Success Rate (%)")
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 100)
    
    plt.tight_layout()
    fig.suptitle(title, fontsize=16, y=1.02)
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig

def plot_kappa_distribution(kappa_values, figsize=(10, 6), bins=30,
                           title="Distribution of Predicted Concentration Parameter (κ)",
                           save_path=None):
    """
    Plot the distribution of predicted concentration parameters (κ).
    
    Args:
        kappa_values: Array of concentration parameters
        figsize: Figure size tuple
        bins: Number of histogram bins
        title: Plot title
        save_path: Path to save the figure (optional)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create histogram
    ax.hist(kappa_values, bins=bins, alpha=0.7, color='royalblue', edgecolor='black')
    
    # Add vertical line at mean
    mean_kappa = np.mean(kappa_values)
    ax.axvline(x=mean_kappa, color='r', linestyle='--', alpha=0.7, 
               label=f'Mean κ = {mean_kappa:.2f}')
    
    # Set labels and title
    ax.set_xlabel('Concentration Parameter (κ)')
    ax.set_ylabel('Count')
    ax.set_title(title, fontsize=14)
    
    # Add statistical annotations
    stats_text = f"Mean κ: {mean_kappa:.2f}\n"
    stats_text += f"Median κ: {np.median(kappa_values):.2f}\n"
    stats_text += f"Min κ: {np.min(kappa_values):.2f}\n"
    stats_text += f"Max κ: {np.max(kappa_values):.2f}"
    
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.legend()
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig, ax

def plot_error_vs_kappa(angular_errors, kappa_values, figsize=(10, 6),
                       title="Angular Error vs. Concentration Parameter (κ)",
                       save_path=None):
    """
    Create a scatter plot of angular errors vs. predicted concentration parameters.
    
    Args:
        angular_errors: Array of angular errors in degrees
        kappa_values: Array of concentration parameters
        figsize: Figure size tuple
        title: Plot title
        save_path: Path to save the figure (optional)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create scatter plot
    scatter = ax.scatter(kappa_values, angular_errors, alpha=0.7, c=angular_errors,
                        cmap='RdYlGn_r')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Angular Error (degrees)')
    
    # Calculate correlation
    correlation = np.corrcoef(kappa_values, angular_errors)[0, 1]
    
    # Add best-fit line
    z = np.polyfit(kappa_values, angular_errors, 1)
    p = np.poly1d(z)
    ax.plot(sorted(kappa_values), p(sorted(kappa_values)), 'r--', alpha=0.7,
           label=f'Trend: y = {z[0]:.2f}x + {z[1]:.2f}')
    
    # Set labels and title
    ax.set_xlabel('Concentration Parameter (κ)')
    ax.set_ylabel('Angular Error (degrees)')
    ax.set_title(title, fontsize=14)
    
    # Add correlation text
    corr_text = f"Correlation: {correlation:.4f}"
    if correlation < 0:
        corr_text += "\n(Higher κ → Lower Error)"
    else:
        corr_text += "\n(Higher κ → Higher Error)"
    
    ax.text(0.05, 0.95, corr_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.legend()
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig, ax 