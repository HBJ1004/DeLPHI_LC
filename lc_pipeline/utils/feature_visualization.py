#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature importance and model interpretability visualization utilities for the light curve pipeline.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import seaborn as sns
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import torch
from captum.attr import (
    IntegratedGradients, 
    Occlusion,
    NoiseTunnel,
    Saliency,
    GuidedBackprop,
    visualization as viz
)

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

def plot_feature_importance(feature_names, importances, figsize=(12, 8), 
                            title="Feature Importance", sort=True, save_path=None):
    """
    Plot feature importance for a model.
    
    Args:
        feature_names: List of feature names
        importances: List or array of feature importance values
        figsize: Figure size tuple
        title: Plot title
        sort: Whether to sort features by importance
        save_path: Path to save the figure (optional)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create DataFrame for easier handling
    df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    
    # Sort if requested
    if sort:
        df = df.sort_values('Importance', ascending=False)
    
    # Create horizontal bar plot
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(df)))
    sns.barplot(x='Importance', y='Feature', data=df, ax=ax, palette=colors)
    
    # Set labels and title
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    ax.set_title(title, fontsize=14)
    
    # Add grid lines
    ax.xaxis.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig, ax

def plot_integrated_gradients(model, input_tensor, target_class=None, 
                             n_steps=50, figsize=(15, 8), feature_names=None,
                             title="Integrated Gradients Feature Attribution",
                             save_path=None):
    """
    Visualize feature attribution using Integrated Gradients.
    
    Args:
        model: PyTorch model
        input_tensor: Input tensor to analyze
        target_class: Target class index for classification models
        n_steps: Number of steps for path integral approximation
        figsize: Figure size tuple
        feature_names: List of feature names
        title: Plot title
        save_path: Path to save the figure (optional)
    """
    # Check that input is a tensor
    if not isinstance(input_tensor, torch.Tensor):
        input_tensor = torch.tensor(input_tensor, dtype=torch.float32)
    
    # Make sure input has a batch dimension
    if len(input_tensor.shape) == 1:
        input_tensor = input_tensor.unsqueeze(0)
    
    # Create IntegratedGradients instance
    ig = IntegratedGradients(model)
    
    # Compute attributions
    attributions, approximation_error = ig.attribute(
        input_tensor, 
        target=target_class, 
        n_steps=n_steps,
        return_convergence_delta=True
    )
    
    # Convert to numpy for plotting
    attr = attributions.detach().numpy()
    
    # Get feature names if not provided
    if feature_names is None:
        feature_names = [f"Feature {i+1}" for i in range(attr.shape[1])]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'Feature': feature_names,
        'Attribution': attr[0]
    })
    
    # Sort by absolute attribution magnitude
    df['AbsAttribution'] = np.abs(df['Attribution'])
    df = df.sort_values('AbsAttribution', ascending=False)
    
    # Create color map based on attribution sign
    colors = ['red' if val < 0 else 'green' for val in df['Attribution']]
    
    # Create horizontal bar plot
    sns.barplot(x='Attribution', y='Feature', data=df, ax=ax, palette=colors)
    
    # Set labels and title
    ax.set_xlabel('Attribution Value')
    ax.set_ylabel('Feature')
    ax.set_title(title, fontsize=14)
    
    # Add vertical line at zero
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.7)
    
    # Add grid lines
    ax.xaxis.grid(True, alpha=0.3)
    
    # Add text annotation for approximation error
    error_text = f"Approximation Error: {approximation_error.mean().item():.4f}"
    ax.text(0.05, 0.02, error_text, transform=ax.transAxes, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig, ax, attributions

def plot_occlusion_sensitivity(model, input_tensor, target_class=None, 
                              sliding_window_shapes=(1,), strides=None,
                              figsize=(15, 8), feature_names=None,
                              title="Occlusion Sensitivity Analysis",
                              save_path=None):
    """
    Visualize feature importance using Occlusion sensitivity analysis.
    
    Args:
        model: PyTorch model
        input_tensor: Input tensor to analyze
        target_class: Target class index for classification models
        sliding_window_shapes: Tuple with sliding window shapes
        strides: Tuple with stride values (defaults to window shape)
        figsize: Figure size tuple
        feature_names: List of feature names
        title: Plot title
        save_path: Path to save the figure (optional)
    """
    # Check that input is a tensor
    if not isinstance(input_tensor, torch.Tensor):
        input_tensor = torch.tensor(input_tensor, dtype=torch.float32)
    
    # Make sure input has a batch dimension
    if len(input_tensor.shape) == 1:
        input_tensor = input_tensor.unsqueeze(0)
    
    # Set default strides if not provided
    if strides is None:
        strides = sliding_window_shapes
    
    # Create Occlusion instance
    occlusion = Occlusion(model)
    
    # Compute attributions
    attributions = occlusion.attribute(
        input_tensor,
        target=target_class,
        sliding_window_shapes=sliding_window_shapes,
        strides=strides
    )
    
    # Convert to numpy for plotting
    attr = attributions.detach().numpy()
    
    # Get feature names if not provided
    if feature_names is None:
        feature_names = [f"Feature {i+1}" for i in range(attr.shape[1])]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'Feature': feature_names,
        'Attribution': attr[0]
    })
    
    # Sort by absolute attribution magnitude
    df['AbsAttribution'] = np.abs(df['Attribution'])
    df = df.sort_values('AbsAttribution', ascending=False)
    
    # Create color map based on attribution sign
    colors = ['red' if val < 0 else 'green' for val in df['Attribution']]
    
    # Create horizontal bar plot
    sns.barplot(x='Attribution', y='Feature', data=df, ax=ax, palette=colors)
    
    # Set labels and title
    ax.set_xlabel('Attribution Value')
    ax.set_ylabel('Feature')
    ax.set_title(title, fontsize=14)
    
    # Add vertical line at zero
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.7)
    
    # Add grid lines
    ax.xaxis.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig, ax, attributions

def plot_input_feature_attributions(input_data, attributions, model_type='period', 
                                   figsize=(15, 8), title=None, save_path=None):
    """
    Visualize input feature attributions for time series data.
    
    Args:
        input_data: Original input data (time, brightness, etc.)
        attributions: Attribution values from attribution method
        model_type: Type of model ('period' or 'axis')
        figsize: Figure size tuple
        title: Plot title
        save_path: Path to save the figure (optional)
    """
    # Convert attributions to numpy if it's a tensor
    if isinstance(attributions, torch.Tensor):
        attributions = attributions.detach().numpy()
    
    # Flatten attributions if needed
    if len(attributions.shape) > 2:
        attributions = attributions.reshape(attributions.shape[0], -1)
    
    # Determine number of features based on model type
    if model_type.lower() == 'period':
        n_features = 2  # time and brightness
        feature_names = ['Time', 'Relative Brightness']
    else:  # axis
        # Assuming axis model has phase-folded data
        n_features = 1  # phase-folded brightness
        feature_names = ['Phase-Folded Brightness']
    
    # Create figure
    fig, axes = plt.subplots(n_features, 1, figsize=figsize)
    
    # Handle single axis case
    if n_features == 1:
        axes = [axes]
    
    # Plot each feature
    for i in range(n_features):
        ax = axes[i]
        
        # Extract data for current feature
        if model_type.lower() == 'period':
            if i == 0:  # time
                x_data = input_data[0, 0, :]  # Assuming input shape is [batch, features, time]
                attr_data = attributions[0, :x_data.shape[0]]
                x_label = 'Time'
            else:  # brightness
                x_data = input_data[0, 1, :]
                attr_data = attributions[0, x_data.shape[0]:]
                x_label = 'Relative Brightness'
        else:  # axis
            x_data = np.arange(attributions.shape[1])  # Use index for x-axis
            attr_data = attributions[0, :]
            x_label = 'Phase Bin'
        
        # Create twin axes for data and attributions
        ax2 = ax.twinx()
        
        # Plot original data
        ax.plot(x_data, color='blue', alpha=0.7, label='Original Data')
        ax.set_ylabel(feature_names[i], color='blue')
        ax.tick_params(axis='y', labelcolor='blue')
        
        # Plot attributions
        # Use positive and negative colors
        pos_attr = np.copy(attr_data)
        neg_attr = np.copy(attr_data)
        pos_attr[pos_attr < 0] = 0
        neg_attr[neg_attr > 0] = 0
        
        ax2.bar(range(len(attr_data)), pos_attr, alpha=0.5, color='green', label='Positive Attribution')
        ax2.bar(range(len(attr_data)), neg_attr, alpha=0.5, color='red', label='Negative Attribution')
        
        ax2.set_ylabel('Attribution', color='black')
        ax2.tick_params(axis='y', labelcolor='black')
        
        # Set x-label only for bottom subplot
        if i == n_features - 1:
            ax.set_xlabel(x_label)
        
        # Add legend
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper right')
    
    # Set overall title
    if title is None:
        title = f"Feature Attributions for {model_type.capitalize()} Model"
    plt.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig, axes

def plot_saliency_map(model, input_tensor, target_class=None, 
                     figsize=(15, 6), abs_values=True, 
                     title="Saliency Map Analysis",
                     save_path=None):
    """
    Visualize feature importance using Saliency analysis.
    
    Args:
        model: PyTorch model
        input_tensor: Input tensor to analyze
        target_class: Target class index for classification models
        figsize: Figure size tuple
        abs_values: Whether to show absolute values
        title: Plot title
        save_path: Path to save the figure (optional)
    """
    # Check that input is a tensor
    if not isinstance(input_tensor, torch.Tensor):
        input_tensor = torch.tensor(input_tensor, dtype=torch.float32)
    
    # Make sure input has a batch dimension
    if len(input_tensor.shape) == 1:
        input_tensor = input_tensor.unsqueeze(0)
    
    # Create Saliency instance
    saliency = Saliency(model)
    
    # Compute attributions
    attributions = saliency.attribute(input_tensor, target=target_class, abs=abs_values)
    
    # Convert to numpy for visualization
    attr_np = attributions.detach().numpy()
    
    # Create figure based on input shape
    if len(input_tensor.shape) == 2:  # 1D input (batch, features)
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create bar plot
        ax.bar(range(attr_np.shape[1]), attr_np[0], alpha=0.7)
        
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Saliency')
        ax.set_title(title)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
    elif len(input_tensor.shape) == 3:  # 2D input (batch, channels, length)
        fig, axes = plt.subplots(input_tensor.shape[1], 1, figsize=figsize)
        
        # Handle single channel case
        if input_tensor.shape[1] == 1:
            axes = [axes]
        
        # Plot each channel
        for i in range(input_tensor.shape[1]):
            ax = axes[i]
            ax.plot(attr_np[0, i], alpha=0.7, color='royalblue')
            ax.set_ylabel(f'Channel {i}')
            
            # Set x-label only for bottom subplot
            if i == input_tensor.shape[1] - 1:
                ax.set_xlabel('Sequence Length')
            
            # Add grid
            ax.grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=16)
        
    else:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Unsupported input shape", 
                ha='center', va='center', fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig, attributions 