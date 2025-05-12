#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example code for using the visualization utilities in the colab environment.
This file contains functions that can be copied into colab_run.py to generate
various visualizations for the light curve pipeline.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import torch.nn.functional as F

# Import visualization modules
from lc_pipeline.utils.visualization import (
    plot_ls_period_distribution,
    plot_phase_folded_lightcurves,
    plot_period_scatter,
    plot_period_error_histogram,
    plot_period_metrics_vs_epoch
)
from lc_pipeline.utils.axis_visualization import (
    plot_axis_scatter,
    plot_angular_error_histogram,
    plot_axis_sky_map,
    plot_axis_metrics_vs_epoch,
    plot_kappa_distribution,
    plot_error_vs_kappa,
    angular_distance
)
from lc_pipeline.utils.feature_visualization import (
    plot_feature_importance,
    plot_integrated_gradients,
    plot_occlusion_sensitivity,
    plot_input_feature_attributions,
    plot_saliency_map
)


def generate_period_visualizations(dataset, period_model, training_history, output_dir):
    """
    Generate visualizations for period prediction.
    
    Args:
        dataset: Dataset object containing light curves
        period_model: Trained period prediction model
        training_history: Dictionary with training metrics
        output_dir: Directory to save visualizations
    """
    # Create output directory if it doesn't exist
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # 1. Extract data from dataset
    all_ls_periods = []
    all_true_periods = []
    all_pred_periods = []
    
    # Device for model
    device = next(period_model.parameters()).device
    
    # Put model in eval mode
    period_model.eval()
    
    # Process samples from dataset
    with torch.no_grad():
        for i in range(min(len(dataset), 100)):  # Limit to 100 samples
            sample = dataset[i]
            
            # Extract data
            inputs = sample['inputs'].unsqueeze(0).to(device)
            true_period = sample['period_target'].item()
            ls_period = sample['ls_period'].item()
            
            # Get prediction
            outputs = period_model(inputs)
            pred_period = outputs.item()
            
            # Back-transform if necessary (e.g., from log scale)
            if 'period_transform' in sample:
                true_period = 10**true_period
                pred_period = 10**pred_period
                ls_period = 10**ls_period
            
            all_true_periods.append(true_period)
            all_pred_periods.append(pred_period)
            all_ls_periods.append(ls_period)
    
    # 2. Lomb-Scargle Period Distribution (Figure 2)
    fig2 = plot_ls_period_distribution(
        all_ls_periods,
        bins=30,
        log_scale=True,
        title="Distribution of Lomb-Scargle Periods",
        save_path=os.path.join(figures_dir, "ls_period_distribution.png")
    )
    
    # 3. Phase-Folded Light Curves (Figure 3)
    # Extract a few samples for visualization
    sample_indices = np.random.choice(len(dataset), size=3, replace=False)
    times = []
    brightnesses = []
    periods_list = []
    
    for idx in sample_indices:
        sample = dataset[idx]
        # Extract raw time and brightness data
        time = sample['time']
        brightness = sample['brightness']
        
        # Get different period values
        gt_period = sample['period_target'].item()
        ls_period = sample['ls_period'].item()
        
        # Make predictions
        with torch.no_grad():
            inputs = sample['inputs'].unsqueeze(0).to(device)
            pred_period = period_model(inputs).item()
        
        # Back-transform if necessary
        if 'period_transform' in sample:
            gt_period = 10**gt_period
            pred_period = 10**pred_period
            ls_period = 10**ls_period
        
        times.append(time.numpy())
        brightnesses.append(brightness.numpy())
        periods_list.append([gt_period, ls_period, pred_period])
    
    fig3 = plot_phase_folded_lightcurves(
        times,
        brightnesses,
        periods_list,
        period_labels=["Ground Truth", "Lomb-Scargle", "Predicted"],
        save_path=os.path.join(figures_dir, "phase_folded_lightcurves.png")
    )
    
    # 4. Predicted vs. True Period Scatter Plot (Figure 4)
    fig4 = plot_period_scatter(
        all_true_periods,
        all_pred_periods,
        add_harmonics=True,
        title="Predicted vs. True Period",
        save_path=os.path.join(figures_dir, "period_scatter.png")
    )
    
    # 5. Histogram of Period Prediction Errors (Figure 5)
    fig5 = plot_period_error_histogram(
        all_true_periods,
        all_pred_periods,
        log_error=True,
        title="Distribution of Period Prediction Errors",
        save_path=os.path.join(figures_dir, "period_error_histogram.png")
    )
    
    # 6. Period MAE/RMSE vs. Epoch (Figure 6)
    # Extract training history
    train_losses = training_history.get('train_loss', [])
    val_losses = training_history.get('val_loss', [])
    val_mae = training_history.get('val_mae', [])
    val_rmse = training_history.get('val_rmse', [])
    
    fig6 = plot_period_metrics_vs_epoch(
        train_losses,
        val_losses,
        val_mae,
        val_rmse,
        title="Period Model Training Metrics",
        save_path=os.path.join(figures_dir, "period_metrics_vs_epoch.png")
    )
    
    print(f"Generated period visualizations in {figures_dir}")


def generate_axis_visualizations(dataset, axis_model, axis_training_history, output_dir, threshold=30.0):
    """
    Generate visualizations for axis prediction.
    
    Args:
        dataset: Dataset object containing light curves and axis data
        axis_model: Trained axis prediction model
        axis_training_history: Dictionary with training metrics
        output_dir: Directory to save visualizations
        threshold: Success threshold in degrees (default: 30.0)
    """
    # Create output directory if it doesn't exist
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # 1. Extract data from dataset
    all_true_l = []
    all_true_b = []
    all_pred_l = []
    all_pred_b = []
    all_kappas = []
    
    # Device for model
    device = next(axis_model.parameters()).device
    
    # Put model in eval mode
    axis_model.eval()
    
    # Process samples from dataset
    with torch.no_grad():
        for i in range(min(len(dataset), 100)):  # Limit to 100 samples
            sample = dataset[i]
            
            # Extract data
            inputs = sample['inputs'].unsqueeze(0).to(device)
            true_l = sample['l'].item()
            true_b = sample['b'].item()
            
            # Get prediction (assuming model outputs quaternions or l,b directly)
            outputs = axis_model(inputs)
            
            # Process outputs based on model type
            if outputs.shape[1] > 2:  # Quaternion + kappa model
                # Assuming first 4 elements are quaternion and last is kappa
                quat = outputs[0, :4].cpu().numpy()
                kappa = outputs[0, 4].item() if outputs.shape[1] > 4 else 100.0  # Default kappa if not predicted
                
                # Convert quaternion to l,b
                from lc_pipeline.utils.axis_visualization import quaternion_to_lb
                pred_l, pred_b = quaternion_to_lb(quat)
                
                all_kappas.append(kappa)
            else:  # Direct l,b prediction
                pred_l = outputs[0, 0].item()
                pred_b = outputs[0, 1].item()
            
            all_true_l.append(true_l)
            all_true_b.append(true_b)
            all_pred_l.append(pred_l)
            all_pred_b.append(pred_b)
    
    # Calculate angular errors
    angular_errors = [
        angular_distance(all_true_l[i], all_true_b[i], all_pred_l[i], all_pred_b[i])
        for i in range(len(all_true_l))
    ]
    
    # 8. Predicted vs. True Axis Scatter Plot (Figure 8)
    fig8 = plot_axis_scatter(
        all_true_l,
        all_true_b,
        all_pred_l,
        all_pred_b,
        title="Predicted vs. True Axis Orientation",
        save_path=os.path.join(figures_dir, "axis_scatter.png")
    )
    
    # 9. Histogram of Angular Errors (Figure 9)
    fig9 = plot_angular_error_histogram(
        angular_errors,
        threshold=threshold,
        title="Distribution of Angular Errors",
        save_path=os.path.join(figures_dir, "angular_error_histogram.png")
    )
    
    # 10. Sky Plot of Predicted vs. True Axes (Figure 10)
    fig10 = plot_axis_sky_map(
        all_true_l,
        all_true_b,
        all_pred_l,
        all_pred_b,
        title="Sky Map of True and Predicted Axis Orientations",
        save_path=os.path.join(figures_dir, "axis_sky_map.png")
    )
    
    # 11/12. Mean Angular Error and Success Rate vs. Epoch (Figure 11/12)
    # Extract training history
    train_losses = axis_training_history.get('train_loss', [])
    val_losses = axis_training_history.get('val_loss', [])
    mean_angular_errors = axis_training_history.get('val_angular_error', [])
    success_rates = axis_training_history.get('val_success_rate', [])
    
    fig11 = plot_axis_metrics_vs_epoch(
        train_losses,
        val_losses,
        mean_angular_errors,
        success_rates,
        threshold=threshold,
        title="Axis Model Training Metrics",
        save_path=os.path.join(figures_dir, "axis_metrics_vs_epoch.png")
    )
    
    # 13/14. Kappa distribution and Error vs. Kappa (if available)
    if all_kappas:
        fig13 = plot_kappa_distribution(
            all_kappas,
            title="Distribution of Predicted Concentration Parameter (κ)",
            save_path=os.path.join(figures_dir, "kappa_distribution.png")
        )
        
        fig14 = plot_error_vs_kappa(
            angular_errors,
            all_kappas,
            title="Angular Error vs. Concentration Parameter (κ)",
            save_path=os.path.join(figures_dir, "error_vs_kappa.png")
        )
    
    print(f"Generated axis visualizations in {figures_dir}")


def generate_feature_importance(dataset, period_model, axis_model, output_dir):
    """
    Generate feature importance visualizations for both models.
    
    Args:
        dataset: Dataset object containing light curves
        period_model: Trained period prediction model
        axis_model: Trained axis prediction model
        output_dir: Directory to save visualizations
    """
    # Create output directory if it doesn't exist
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Device for models
    device = next(period_model.parameters()).device
    
    # Put models in eval mode
    period_model.eval()
    axis_model.eval()
    
    # Get a sample for visualization
    sample = dataset[0]
    period_inputs = sample['inputs'].unsqueeze(0).to(device)
    
    # 16. Feature Importance for Period Model
    try:
        # Using integrated gradients
        from captum.attr import IntegratedGradients
        
        # For period model
        ig = IntegratedGradients(period_model)
        period_attr, _ = ig.attribute(period_inputs, n_steps=50, return_convergence_delta=True)
        
        # Convert to numpy
        period_attr_np = period_attr.detach().cpu().numpy()
        
        # Get feature names (assuming 2-channel input: time and brightness)
        num_points = period_inputs.shape[2]
        feature_names = []
        for i in range(num_points):
            feature_names.append(f"Time_{i}")
        for i in range(num_points):
            feature_names.append(f"Brightness_{i}")
        
        # Calculate absolute importance
        feature_importance = np.abs(period_attr_np).sum(axis=0)
        
        # Plot feature importance
        fig16a = plot_feature_importance(
            feature_names,
            feature_importance,
            title="Period Model Feature Importance",
            save_path=os.path.join(figures_dir, "period_feature_importance.png")
        )
        
        # Plot input attributions
        fig16b = plot_input_feature_attributions(
            period_inputs.cpu().numpy(),
            period_attr_np,
            model_type='period',
            title="Period Model Input Feature Attributions",
            save_path=os.path.join(figures_dir, "period_input_attributions.png")
        )
        
        # For axis model (if folded inputs available)
        if 'folded_inputs' in sample:
            axis_inputs = sample['folded_inputs'].unsqueeze(0).to(device)
            
            ig_axis = IntegratedGradients(axis_model)
            axis_attr, _ = ig_axis.attribute(axis_inputs, n_steps=50, return_convergence_delta=True)
            
            # Convert to numpy
            axis_attr_np = axis_attr.detach().cpu().numpy()
            
            # Get feature names (assuming single-channel input: phase-folded brightness)
            axis_feature_names = [f"Phase_{i}" for i in range(axis_inputs.shape[2])]
            
            # Calculate absolute importance
            axis_feature_importance = np.abs(axis_attr_np).sum(axis=0)
            
            # Plot feature importance
            fig16c = plot_feature_importance(
                axis_feature_names,
                axis_feature_importance,
                title="Axis Model Feature Importance",
                save_path=os.path.join(figures_dir, "axis_feature_importance.png")
            )
            
            # Plot input attributions
            fig16d = plot_input_feature_attributions(
                axis_inputs.cpu().numpy(),
                axis_attr_np,
                model_type='axis',
                title="Axis Model Input Feature Attributions",
                save_path=os.path.join(figures_dir, "axis_input_attributions.png")
            )
        
        print(f"Generated feature importance visualizations in {figures_dir}")
        
    except Exception as e:
        print(f"Feature importance visualization failed: {e}")


# Example of how to call these functions from colab_run.py:
"""
#@title Generate Visualizations
from lc_pipeline.utils.visualization_examples import (
    generate_period_visualizations,
    generate_axis_visualizations,
    generate_feature_importance
)

# After training models
generate_period_visualizations(test_dataset, period_model, period_training_history, RESULTS_DIR)
generate_axis_visualizations(test_dataset, axis_model, axis_training_history, RESULTS_DIR)
generate_feature_importance(test_dataset, period_model, axis_model, RESULTS_DIR)
""" 