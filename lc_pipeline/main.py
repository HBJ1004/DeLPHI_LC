#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script for light curve analysis pipeline.

This script orchestrates the entire analysis pipeline.
"""

import os
import sys
import torch
import torch.nn as nn
import argparse
from pathlib import Path
import json
import numpy as np
import time
import logging # Import the standard logging library
import pandas as pd
from torch.utils.data import DataLoader, random_split
from omegaconf import OmegaConf
import glob

from lc_pipeline.config import load_config, save_config, PipelineConfig, validate_config
from lc_pipeline.utils.logging import Logger, Timer
from lc_pipeline.data.datasets import AsteroidDataset, AxisDataset, LSFeatureDataset
from lc_pipeline.data.collate import collate_fn, generate_axis_data
from lc_pipeline.data.cache_utils import (
    get_cached_axis_data, 
    save_axis_data_to_cache,
    get_dataset_identifier,
    get_period_model_identifier
)

# Import model classes
from lc_pipeline.models.period_nets import (
    PeriodLSTMNet, PeriodTransformerNet, PeriodLSTMWithLSPrior, PeriodLogScaleNet
)
from lc_pipeline.models.axis_nets import AxisCNNNet, PhaseAwareTransformerAxis
from lc_pipeline.models.axis_nets import create_axis_model as create_axis_model_factory

# Import training functions
from lc_pipeline.training.train_period import train_period_model, train_enhanced_period_model
from lc_pipeline.training.train_axis import (
    train_axis_model, evaluate_axis_model
)
from lc_pipeline.evaluation import evaluate_period_model, save_evaluation_results

# Import visualization example functions (assuming they are structured for direct use)
# If these cause import errors or need adaptation, we can import individual functions
try:
    from lc_pipeline.utils.visualization_examples import (
        generate_period_visualizations,
        generate_axis_visualizations,
        generate_feature_importance
    )
    # Flag to indicate successful import
    _VIS_EXAMPLES_IMPORTED = True
except ImportError:
    _VIS_EXAMPLES_IMPORTED = False
    # Fallback: Import individual plotting functions if example wrappers aren't found/usable
    from lc_pipeline.utils.visualization import (
        plot_ls_period_distribution, plot_phase_folded_lightcurves,
        plot_period_scatter, plot_period_error_histogram, plot_period_metrics_vs_epoch,
        save_figure
    )
    from lc_pipeline.utils.axis_visualization import (
        plot_axis_scatter, plot_angular_error_histogram, plot_axis_sky_map,
        plot_axis_metrics_vs_epoch, plot_kappa_distribution, plot_error_vs_kappa,
        angular_distance, quaternion_to_lb
    )
    # Attempt to import feature visualization tools safely
    try:
        from captum.attr import IntegratedGradients # Check if captum is installed
        from lc_pipeline.utils.feature_visualization import (
            plot_feature_importance,
            plot_integrated_gradients,
            plot_occlusion_sensitivity,
            plot_input_feature_attributions,
            plot_saliency_map
        )
        CAPTUM_AVAILABLE = True
    except ImportError:
        CAPTUM_AVAILABLE = False
        # Define placeholders if captum is not available to avoid NameError later
        plot_feature_importance = None
        plot_integrated_gradients = None
        plot_occlusion_sensitivity = None
        plot_input_feature_attributions = None
        plot_saliency_map = None

# Global flag for captum availability (checked later)
CAPTUM_AVAILABLE = False

# --- Suppress excessive matplotlib font logging ---
mpl_logger = logging.getLogger('matplotlib.font_manager')
mpl_logger.setLevel(logging.WARNING)

# Helper function to make results JSON serializable
def make_serializable(obj):
    """
    Recursively convert objects to JSON serializable types with performance optimizations.
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON serializable version of the object
    """
    # Use faster type checking
    obj_type = type(obj)
    
    # Handle None and basic types efficiently
    if obj is None or obj_type in (str, int, float, bool):
        return obj
        
    # Handle dictionaries with dict comprehension (faster than loops)
    if obj_type is dict:
        return {k: make_serializable(v) for k, v in obj.items()}
        
    # Handle lists and tuples with list comprehension
    if obj_type in (list, tuple):
        return [make_serializable(elem) for elem in obj]
        
    # Check for numpy types efficiently (avoid repeated module checks)
    if 'numpy' in str(obj_type):
        import numpy as np
        
        # Handle NumPy arrays efficiently with direct conversion
        if isinstance(obj, np.ndarray):
            # Use .tolist() directly for best performance
            return obj.tolist()
            
        # Group similar numpy scalar types together for fewer checks
        if isinstance(obj, (np.number,)):
            # Directly convert to Python scalar
            return obj.item()
                    
        # Fallback for other numpy types
        if hasattr(obj, 'item'):
            return obj.item()
        return str(obj)
            
    # Handle torch tensors (single conditional check)
    if 'torch' in str(obj_type) and hasattr(obj, 'detach'):
        # Convert tensors to lists directly
        try:
            return obj.detach().cpu().numpy().tolist()
        except Exception:
            return str(obj)
            
    # Handle dataclasses and other objects with __dict__
    if hasattr(obj, '__dict__'):
        try:
            # Use dict comprehension with efficient filtering
            return {k: make_serializable(v) for k, v in obj.__dict__.items() 
                    if not k.startswith('_') and not callable(v)}
        except Exception:
            return str(obj)
            
    # Handle objects with to_dict method
    if hasattr(obj, 'to_dict') and callable(obj.to_dict):
        try:
            return make_serializable(obj.to_dict())
        except Exception:
            return str(obj)
            
    # Default for any other types - convert to string
    return str(obj)

def setup_environment(config):
    """
    Set up the environment for training.
    
    Args:
        config: Configuration object
        
    Returns:
        logger: Logger object
    """
    # Create directories
    os.makedirs(config.paths.logs_dir, exist_ok=True)
    os.makedirs(config.paths.models_dir, exist_ok=True)
    os.makedirs(config.paths.results_dir, exist_ok=True)
    os.makedirs(config.paths.figures_dir, exist_ok=True)
    
    # Set up logging
    log_level_to_set = config.logging.log_level
    if hasattr(config, 'debug') and config.debug:
        log_level_to_set = "DEBUG"

    logger = Logger(
        log_dir=config.paths.logs_dir,
        experiment_name=f"lc_pipeline_{config.period_model.model_name}",
        log_level=log_level_to_set,
        log_to_console=config.logging.log_to_console,
        log_to_file=config.logging.log_to_file,
        use_tensorboard=True,
        config=vars(config)
    )
    
    # Set random seed for reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Log configuration
    logger.info(f"Using device: {config.device}")
    
    return logger


def load_data(config, logger):
    """
    Load data for training.
    
    Args:
        config: Configuration object
        logger: Logger object
        
    Returns:
        tuple: (dataset, train_loader, val_loader)
    """
    with Timer("Data loading", logger):
        csv_files = []
        if config.data.use_damit_data:
            damit_dir = Path(config.data.damit_data_dir)
            damit_files = list(damit_dir.glob("*.csv"))
            max_damit = getattr(config.data, 'max_damit_files', None)
            if max_damit and len(damit_files) > max_damit:
                logger.info(f"Limiting to {max_damit} DAMIT files from {len(damit_files)} found.")
                damit_files = damit_files[:max_damit]
            else:
                logger.info(f"Found {len(damit_files)} DAMIT files in {damit_dir}")
            csv_files.extend(damit_files)
        
        if config.data.use_synthetic_data:
            synthetic_dir = Path(config.data.synthetic_data_dir)
            synthetic_files = list(synthetic_dir.glob("*.csv"))
            logger.info(f"Found {len(synthetic_files)} synthetic files in {synthetic_dir}")
            csv_files.extend(synthetic_files)
        
        if not csv_files:
            msg = f"No CSV files found. Searched DAMIT: {config.data.damit_data_dir}, Synthetic: {config.data.synthetic_data_dir}"
            logger.error(msg)
            raise FileNotFoundError(msg)
            
        logger.info(f"Creating dataset with {len(csv_files)} files. Max sequence length: {config.data.max_sequence_length}")
        
        # Set up caching parameters
        use_disk_caching = getattr(config.cache, 'use_disk_caching', True) 
        
        # Correctly determine force_recache, prioritizing config.data.force_rebuild_cache
        force_recache_data = getattr(config.data, 'force_rebuild_cache', False)
        force_recache_cache_config = getattr(config.cache, 'force_recache', False)
        force_recache_top_level = getattr(config, 'force_recache', False) # Legacy check for older configs

        force_recache = force_recache_data or force_recache_cache_config or force_recache_top_level
        
        # Configure cache directory
        asteroid_cache_dir = config.paths.cache_dir / "asteroid_cache"
        if use_disk_caching:
            logger.info(f"Using disk caching for AsteroidDataset: {asteroid_cache_dir}")
            if force_recache: # This 'if' block is important
                logger.info("Force recache is TRUE. Existing cache for AsteroidDataset (if any) will be ignored/overwritten during dataset initialization.")
                
                # DIRECT INTERVENTION: Check for existing cache files and delete them
                os.makedirs(asteroid_cache_dir, exist_ok=True)
                cache_pattern = os.path.join(asteroid_cache_dir, "dataset_cache_*.pt")
                cache_files = glob.glob(cache_pattern)
                if cache_files:
                    logger.info(f"Found {len(cache_files)} existing cache files. Deleting them to force rebuild.")
                    for cache_file in cache_files:
                        try:
                            os.remove(cache_file)
                            logger.info(f"Successfully deleted cache file: {cache_file}")
                        except Exception as e:
                            logger.warning(f"Failed to delete cache file {cache_file}: {e}")
                else:
                    logger.info("No existing cache files found.")
        else:
            logger.info("Disk caching is disabled for AsteroidDataset")
            
        # Create relevant data config snapshot for caching
        data_config_snapshot = {
            'max_sequence_length': config.data.max_sequence_length,
            'use_damit_data': config.data.use_damit_data,
            'use_synthetic_data': config.data.use_synthetic_data,
            # Add any other relevant parameters that affect dataset processing
        }

        dataset = AsteroidDataset(
            csv_files=csv_files,
            config=data_config_snapshot,
            max_sequence_length=config.data.max_sequence_length,
            logger=logger,
            # period_scale is handled by AsteroidDataset if None
            max_files=None, # max_files is already applied to csv_files list if max_damit_files was set
            use_single_file_processing_cache=True,
            cache_dir=asteroid_cache_dir if use_disk_caching else None,
            force_recache=force_recache
        )
        
        train_size = int(len(dataset) * config.data.train_val_ratio)
        val_size = len(dataset) - train_size
        if train_size == 0 or val_size == 0:
            logger.error(f"Train ({train_size}) or Val ({val_size}) split resulted in zero samples. Total dataset size: {len(dataset)}. Check data and ratios.")
            # Fallback to using the whole dataset for both if small, to avoid crash, though this is not ideal for training.
            if len(dataset) > 0:
                 train_dataset, val_dataset = dataset, dataset 
                 logger.warning("Train/Val split is 0. Using full dataset for both. Review data and split ratios.")
            else:
                 raise ValueError("Dataset is empty, cannot create train/val split.")
        else:
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(config.seed) # for reproducible splits
            )
        
        num_workers = getattr(config, 'num_workers', 0) # Default to 0 if not in config
        if os.name == 'nt' and num_workers > 0: # Windows Dataloader often needs num_workers=0
            logger.warning("Running on Windows, setting num_workers to 0 for DataLoader to prevent potential issues. Configure explicitly if needed.")
            num_workers = 0
            
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.period_model.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True if config.device == 'cuda' else False
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.period_model.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True if config.device == 'cuda' else False
        )
        
        logger.info(f"Dataset split: {len(train_dataset)} train, {len(val_dataset)} validation. Num workers: {num_workers}")
    
    return dataset, train_loader, val_loader


def create_period_model(config, logger):
    """
    Create period prediction model.
    
    Args:
        config: Configuration object
        logger: Logger object
        
    Returns:
        model: Period prediction model
    """
    model_name = config.period_model.model_name
    logger.info(f"Creating period model: {model_name}")
    
    if model_name == "PeriodLSTMNet":
        model = PeriodLSTMNet(
            input_dim=config.period_model.input_dim,
            hidden_dim=config.period_model.hidden_dim,
            num_layers=config.period_model.num_layers,
            dropout=config.period_model.dropout
        )
    elif model_name == "PeriodTransformerNet":
        model = PeriodTransformerNet(
            input_dim=config.period_model.input_dim,
            hidden_dim=config.period_model.hidden_dim,
            num_heads=4,
            num_layers=config.period_model.num_layers,
            dropout=config.period_model.dropout
        )
    elif model_name == "PeriodLSTMWithLSPrior":
        model = PeriodLSTMWithLSPrior(
            input_dim=config.period_model.input_dim,
            hidden_dim=config.period_model.hidden_dim,
            num_layers=config.period_model.num_layers,
            dropout=config.period_model.dropout,
            prior_weight=0.5
        )
    else:
        logger.error(f"Unknown period model: {model_name}")
        raise ValueError(f"Unknown period model: {model_name}")
    
    # Apply log-scale if requested
    if config.period_model.use_log_scale:
        model = PeriodLogScaleNet(
            base_model_class=model.__class__,
            min_period=config.period_model.min_period,
            max_period=config.period_model.max_period,
            **{k: v for k, v in model.__dict__.items() if not k.startswith('_')}
        )
    
    # Move model to device
    model = model.to(config.device)
    
    return model


def create_axis_model(config: PipelineConfig, logger: logging.Logger) -> nn.Module:
    """
    Create an axis prediction model based on configuration.
    This function serves as a wrapper to call the model factory in axis_nets.py.
    
    Args:
        config: PipelineConfig object containing model and data configurations.
        logger: Logger instance.
        
    Returns:
        torch.nn.Module: The instantiated axis prediction model.
    """
    model_cfg = config.axis_model
    data_cfg = config.data

    # Prepare kwargs for the factory from the config
    # These will be filtered by the factory based on what the specific model needs
    model_kwargs = {
        # CNN specific (will be ignored by Transformer if not explicitly used)
        'initial_channels': model_cfg.initial_channels,
        'blocks': model_cfg.blocks,
        'kernel': model_cfg.kernel,
        # Transformer specific (will be ignored by CNN)
        'folded_input_features': model_cfg.folded_input_features,
        'raw_input_features': model_cfg.raw_input_features,
        'max_raw_seq_len': model_cfg.max_raw_seq_len,
        'num_heads': model_cfg.num_heads,
        'num_encoder_layers': model_cfg.num_encoder_layers,
        'fusion_strategy': model_cfg.fusion_strategy,
        # Shared / common parameters in config that might be used as kwargs
        'dropout': model_cfg.dropout,
        'use_norm': model_cfg.use_norm,
        # input_features for AxisCNNNet specifically, might not be in model_cfg directly for PAT
        'input_features': getattr(model_cfg, 'input_features', 1) # default if not present
    }

    # Call the centralized model factory
    model = create_axis_model_factory(
        model_name=model_cfg.model_name,
        num_bins=data_cfg.num_axis_bins, # num_bins comes from data config
        hidden_dim=model_cfg.hidden_dim, # Main hidden_dim from axis_model config
        use_quaternions=model_cfg.use_quaternions, # from axis_model config
        logger=logger,
        **model_kwargs
    )
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Created {model_cfg.model_name} with {num_params:,} trainable parameters.")
    
    return model


def train_period_phase(config, logger, dataset, train_loader_override=None, val_loader_override=None):
    """
    Train the period prediction model.
    
    Args:
        config: Configuration object
        logger: Logger object
        dataset: Dataset object
        train_loader_override: Optional DataLoader for training data
        val_loader_override: Optional DataLoader for validation data
        
    Returns:
        period_model: Trained model
        period_predictions: Dictionary of period predictions
    """
    logger.info("Starting period prediction phase")
    device = torch.device(config.device)
    
    # Create dataloaders if not overridden
    if train_loader_override is None and val_loader_override is None:
        train_size = int(len(dataset) * config.data.train_val_ratio)
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size],
                                                  generator=torch.Generator().manual_seed(config.seed))
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.period_model.batch_size,
            shuffle=True,
            collate_fn=collate_fn,  # USE OUR COLLATE FUNCTION
            num_workers=0
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.period_model.batch_size,
            shuffle=False,
            collate_fn=collate_fn,  # USE OUR COLLATE FUNCTION
            num_workers=0
        )
    else:
        train_loader = train_loader_override
        val_loader = val_loader_override
    
    # Perform hyperparameter optimization if enabled
    if config.run_hyperopt:
        from lc_pipeline.training import run_period_optimization, update_config_with_best_params
        
        logger.info("Running hyperparameter optimization for period model")
        
        # Run Optuna optimization
        best_params = run_period_optimization(
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            logger=logger
        )
        
        # Update config with best parameters
        if best_params:
            logger.info(f"Updating period model config with best parameters: {best_params}")
            config = update_config_with_best_params(config, best_params, "period")
            
            # Save updated config
            config_save_path = os.path.join(config.paths.models_dir, "best_period_config.yaml")
            save_config(config, config_save_path)
            logger.info(f"Saved best period model config to {config_save_path}")
        else:
            logger.warning("Hyperparameter optimization did not yield valid results. Using default parameters.")
    
    # Create the model with the (potentially updated) config
    period_model = create_period_model(config, logger)
    
    # Set up checkpoint path
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.join(config.paths.models_dir, f"period_model_{timestamp}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
    
    # Train the model
    logger.info(f"Training period model: {config.period_model.model_name}")
    
    # Choose training function based on model type
    if config.period_model.model_name in ["PhaseAwareTransformerWithLSPrior", "EnhancedPhaseAwareTransformer"]:
        from lc_pipeline.training import train_enhanced_period_model
        
        period_model, training_history = train_enhanced_period_model(
            model=period_model,
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=config.period_model.lr,
            weight_decay=config.period_model.weight_decay,
            num_epochs=config.period_model.epochs,
            device=device,
            patience=config.period_model.patience,
            logger=logger,
            checkpoint_path=checkpoint_path
        )
    else:
        from lc_pipeline.training import train_period_model
        
        period_model, training_history = train_period_model(
            model=period_model,
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=config.period_model.lr,
            weight_decay=config.period_model.weight_decay,
            num_epochs=config.period_model.epochs,
            device=device,
            patience=config.period_model.patience,
            logger=logger,
            checkpoint_path=checkpoint_path,
            period_scale_factor=config.period_model.period_scale_factor,
            use_log_scale=config.period_model.use_log_scale,
            min_period_for_loss=config.period_model.min_period,
            l2_lambda_loss=config.period_model.get("l2_lambda_loss", 0.0)
        )

    # Log best validation metrics
    if 'val_mae' in training_history:
        best_val_mae = min(training_history['val_mae'])
        best_epoch = training_history['val_mae'].index(best_val_mae) + 1
        logger.info(f"Best validation MAE: {best_val_mae:.4f} (Epoch {best_epoch})")
    
    # Load best model from checkpoint
    if os.path.exists(checkpoint_path):
        logger.info(f"Loading best model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        period_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate model
    logger.info("Evaluating period model on validation set")
    val_results = evaluate_period_model(
        model=period_model,
        dataloader=val_loader,
        device=device,
        return_df=True,
        use_true_period_for_metrics=False
    )
    
    # Generate predictions for all data (for use in axis model training)
    logger.info("Generating period predictions for all data")
    
    # Create a dataloader for the entire dataset
    from torch.utils.data import DataLoader
    full_loader = DataLoader(
        dataset,
        batch_size=config.period_model.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    all_results = evaluate_period_model(
        model=period_model,
        dataloader=full_loader,
        device=device,
        return_df=True,
        use_true_period_for_metrics=False
    )
    
    # Create a dictionary mapping asteroid IDs to predicted periods
    period_predictions = {}
    for i, row in all_results.iterrows():
        asteroid_id = row['asteroid_id']
        predicted_period = row['predicted_period']
        period_predictions[asteroid_id] = predicted_period
    
    # Save evaluation results
    os.makedirs(config.paths.results_dir, exist_ok=True)
    results_path = os.path.join(config.paths.results_dir, f"period_results_{timestamp}.json")
    
    # Make results serializable and save
    serializable_results = {
        'config': make_serializable(config),
        'training_history': make_serializable(training_history),
        'validation_results': make_serializable(val_results.to_dict(orient='records')),
        'all_results': make_serializable(all_results.to_dict(orient='records'))
    }
    
    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Saved period model results to {results_path}")
    
    # Generate visualizations if enabled
    if config.logging.save_plots and _VIS_EXAMPLES_IMPORTED:
        try:
            figures_dir = os.path.join(config.paths.figures_dir, f"period_{timestamp}")
            os.makedirs(figures_dir, exist_ok=True)
            
            logger.info(f"Generating period model visualizations in {figures_dir}")
            generate_period_visualizations(
                model=period_model,
                dataset=dataset,
                val_results=val_results,
                training_history=training_history,
                output_dir=figures_dir,
                device=device,
                config=config
            )
            
            # Optionally generate feature importance visualizations
            if CAPTUM_AVAILABLE:
                logger.info("Generating feature importance visualizations")
                generate_feature_importance(
                    model=period_model,
                    dataset=dataset,
                    output_dir=figures_dir,
                    device=device,
                    config=config
                )
            else:
                logger.warning("Captum not available, skipping feature importance visualizations")
        except Exception as e:
            logger.error(f"Error generating period visualizations: {e}")
    
    return period_model, period_predictions


def train_axis_phase(config, logger, main_dataset, period_model, period_predictions):
    """
    Train the axis prediction model.
    
    Args:
        config: Configuration object
        logger: Logger object
        main_dataset: The full AsteroidDataset instance (potentially pre-filtered or split)
        period_model: Trained period prediction model (can be None)
        period_predictions: Predictions from the period model (currently not directly used here if data is regenerated)
        
    Returns:
        tuple: (axis_model, axis_training_history)
    """
    logger.info("Starting axis prediction model training phase...")
    device = config.device
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Create Axis Model
    axis_model = create_axis_model(config, logger)
    axis_model = axis_model.to(device)

    checkpoint_path = os.path.join(config.paths.models_dir, f"{config.axis_model.model_name}_axis_best.pth")

    # Data for Axis Model Training
    # If not using curriculum, we prepare dataloaders here.
    # If using curriculum, train_axis_with_curriculum will handle data generation and dataloaders internally.
    if not config.axis_model.use_curriculum:
        logger.info("Preparing data for standard axis model training (non-curriculum).")
        # Generate axis data once using either true periods or predictions from period_model
        # The period_model passed here is used by generate_axis_data if use_true_periods is False.
        # If use_true_periods is True, period_model is ignored by generate_axis_data for period determination.
        axis_data_features, axis_data_targets, _ = generate_axis_data(
            period_model=period_model,
            dataset=main_dataset, 
            num_bins=config.data.num_axis_bins, # from data config
            smooth=config.data.smooth_axis_data, # from data config
            use_true_periods=config.axis_model.use_true_periods, # from axis_model config
            use_quaternions=config.axis_model.use_quaternions, # from axis_model config
            true_period_mix_ratio=1.0 if config.axis_model.use_true_periods else 0.0, # simple mapping
            logger=logger
        )

        if axis_data_features is None or len(axis_data_features) == 0:
            logger.error("Failed to generate axis data for standard training. Aborting axis phase.")
            return axis_model, {}

        axis_dataset_full = AxisDataset(axis_data_features, axis_data_targets, use_quaternions=config.axis_model.use_quaternions)
        
        # Split into training and validation sets
        train_size = int(config.data.train_val_ratio * len(axis_dataset_full))
        val_size = len(axis_dataset_full) - train_size
        
        if val_size == 0 and len(axis_dataset_full) > 0: # Ensure val_size is not zero if dataset is not empty
            logger.warning("Validation set size is 0. Using a small fraction of training data for validation.")
            if train_size > 1: # Ensure there's enough to split
                val_size = max(1, int(0.1 * train_size)) # Use 10% of train for val, or at least 1 sample
                train_size = len(axis_dataset_full) - val_size
            else: # Not enough data to create a val set, use train set for val (not ideal)
                logger.warning("Not enough data to create a validation set. Using training set for validation. THIS IS NOT RECOMMENDED.")
                train_dataset_split, val_dataset_split = axis_dataset_full, axis_dataset_full
        
        if train_size > 0 and val_size > 0:
            train_dataset_split, val_dataset_split = random_split(axis_dataset_full, [train_size, val_size])
        elif train_size > 0: # Only training data
            logger.warning("Only training data available after split. Validation will be skipped or use training data.")
            train_dataset_split = axis_dataset_full
            val_dataset_split = train_dataset_split # Use training data for validation (with warning)
        else:
            logger.error("No data available for training/validation after splitting. Aborting axis phase.")
            return axis_model, {}

        logger.info(f"Axis training set size: {len(train_dataset_split)}, Validation set size: {len(val_dataset_split)}")

        train_loader_axis = DataLoader(
            train_dataset_split,
            batch_size=config.axis_model.batch_size,
            shuffle=True,
            collate_fn=collate_fn, # Standard collate_fn should work for AxisDataset output
            num_workers=0 # Keep num_workers as 0 for simplicity or make configurable
        )
        val_loader_axis = DataLoader(
            val_dataset_split,
            batch_size=config.axis_model.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
    else:
        # For curriculum learning, loaders are not prepared here.
        # train_axis_with_curriculum handles data generation and loaders internally.
        train_loader_axis = None # Not used
        val_loader_axis = None   # Not used
        logger.info("Curriculum learning enabled for axis model. Data generation will be handled by train_axis_with_curriculum.")
    
    # Choose training function based on whether curriculum learning is enabled
    if config.axis_model.use_curriculum:
        from lc_pipeline.training.train_axis import train_axis_with_curriculum
        
        # Ensure period_model is in eval mode before passing to curriculum training if it exists
        if period_model:
            period_model.eval()

        axis_model, training_history = train_axis_with_curriculum(
            model=axis_model,
            dataset=main_dataset, # Pass the main AsteroidDataset
            period_model=period_model, # Pass the trained period model
            device=device,
            num_bins=config.data.num_axis_bins, # From data config typically for phase folding
            epochs=config.axis_model.epochs,
            batch_size=config.axis_model.batch_size,
            lr=config.axis_model.lr,
            weight_decay=config.axis_model.weight_decay,
            patience=config.axis_model.patience,
            checkpoint_path=checkpoint_path,
            use_quaternions=config.axis_model.use_quaternions,
            # Pass the teacher forcing schedule from config
            teacher_forcing_schedule=config.axis_model.teacher_forcing_schedule,
            # Fallback curriculum_epochs from config (train_axis_with_curriculum will use if schedule is None)
            curriculum_epochs=config.axis_model.curriculum_epochs,
            logger=logger
        )
    else:
        from lc_pipeline.training.train_axis import train_axis_model
        
        if not train_loader_axis or not val_loader_axis:
            logger.error("DataLoaders for standard axis training are not available. Aborting.")
            return axis_model, {}
            
        axis_model, training_history = train_axis_model(
            model=axis_model,
            train_loader=train_loader_axis,
            val_loader=val_loader_axis,
            device=device,
            epochs=config.axis_model.epochs,
            lr=config.axis_model.lr,
            weight_decay=config.axis_model.weight_decay,
            patience=config.axis_model.patience,
            checkpoint_path=checkpoint_path,
            logger=logger
        )
    
    # Log best validation metrics
    if 'val_angular_error' in training_history:
        best_val_error = min(training_history['val_angular_error'])
        best_epoch = training_history['val_angular_error'].index(best_val_error) + 1
        logger.info(f"Best validation angular error: {best_val_error:.4f} (Epoch {best_epoch})")
    
    # Load best model from checkpoint
    if os.path.exists(checkpoint_path):
        logger.info(f"Loading best model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        axis_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate model
    logger.info("Evaluating axis model on validation set")
    from lc_pipeline.training import evaluate_axis_model
    
    val_metrics = evaluate_axis_model(
        model=axis_model,
        dataloader=val_loader_axis,
        device=device,
        quaternions=config.axis_model.use_quaternions
    )
    
    # Log evaluation metrics
    logger.info(f"Axis model validation metrics: {val_metrics}")
    
    # Save evaluation results
    os.makedirs(config.paths.results_dir, exist_ok=True)
    results_path = os.path.join(config.paths.results_dir, f"axis_results_{timestamp}.json")
    
    # Make results serializable and save
    serializable_results = {
        'config': make_serializable(config),
        'training_history': make_serializable(training_history),
        'validation_metrics': make_serializable(val_metrics)
    }
    
    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Saved axis model results to {results_path}")
    
    # Generate visualizations if enabled
    if config.logging.save_plots and _VIS_EXAMPLES_IMPORTED:
        try:
            figures_dir = os.path.join(config.paths.figures_dir, f"axis_{timestamp}")
            os.makedirs(figures_dir, exist_ok=True)
            
            logger.info(f"Generating axis model visualizations in {figures_dir}")
            generate_axis_visualizations(
                model=axis_model,
                dataloader=val_loader_axis,
                training_history=training_history,
                output_dir=figures_dir,
                device=device,
                config=config
            )
        except Exception as e:
            logger.error(f"Error generating axis visualizations: {e}")
    
    return axis_model, training_history


def log_memory_usage(logger=None):
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        logger.info(f"GPU Memory: {allocated:.2f}MB allocated, {reserved:.2f}MB reserved")


def main(config_path=None, debug=False):
    """
    Main function for the light curve analysis pipeline.
    
    Args:
        config_path: Path to configuration file (optional)
        debug: Whether to enable debug logging (optional)
        
    Returns:
        tuple: (period_model, axis_model, period_results, axis_results)
    """
    # Load configuration
    if isinstance(config_path, str):
        config = load_config(config_path)
    elif isinstance(config_path, PipelineConfig):
        config = config_path  # Allow passing a config object directly
    else:
        config = load_config()
    
    start_time = time.time() # Define start_time here
    
    # Set debug flag
    if debug:
        config.debug = True
        
    # Set up environment
    logger = setup_environment(config)
    validate_config(config, logger)
    log_memory_usage(logger)
    
    # Apply test mode adjustments if enabled
    if config.test_mode:
        logger.info("Running in TEST MODE - Using reduced dataset and iterations")
        # Log original values before modification
        original_max_damit = config.data.max_damit_files if hasattr(config.data, "max_damit_files") else 2500
        logger.info(f"Original params: max_damit_files={original_max_damit}, periods_epochs={config.period_model.epochs}, axis_epochs={config.axis_model.epochs}")
        
        # Override parameters for test mode
        # Reduce dataset sizes - ensure this is respected in load_data
        config.data.max_damit_files = min(original_max_damit or 2500, 10)
        
        # Make max_damit_files absolute by forcing it to 10 for test mode
        config.data.max_damit_files = 10
        
        # Reduce training iterations
        config.period_model.epochs = min(config.period_model.epochs, 3)
        config.axis_model.epochs = min(config.axis_model.epochs, 3)
        config.period_model.optuna_trials = min(config.period_model.optuna_trials, 2)
        config.period_model.optuna_epochs = min(config.period_model.optuna_epochs, 2)
        
        # Reduce patience for faster early stopping
        config.period_model.patience = 2
        config.axis_model.patience = 2
        
        # Log modified values
        logger.info(f"TEST MODE params: max_damit_files={config.data.max_damit_files}, period_epochs={config.period_model.epochs}, "
                  f"axis_epochs={config.axis_model.epochs}, optuna_trials={config.period_model.optuna_trials}")
    
    try:
        # Save the configuration for reference
        save_config(config, os.path.join(config.paths.results_dir, "config.yaml"))
        
        # Load data
        dataset, train_loader, val_loader = load_data(config, logger)
        log_memory_usage(logger)
        
        # Train period model if requested
        period_model = None
        period_predictions = None
        period_results = {}
        if config.run_period_training:
            period_model, period_predictions = train_period_phase(
                config, logger, dataset, train_loader, val_loader
            )
        
        # Train axis model if requested
        axis_model = None
        axis_results = {}
        if config.run_axis_training:
            axis_model, axis_results = train_axis_phase(
                config, logger, dataset, period_model, period_predictions
            )
        
        # Save final results
        results = {
            "period_results": period_results,
            "axis_results": axis_results
        }
        serializable_results = make_serializable(results) # Apply the helper function
        with open(os.path.join(config.paths.results_dir, "results.json"), "w") as f:
            json.dump(serializable_results, f, indent=4) # Dump the sanitized dictionary
        
        logger.info("Pipeline execution completed")
        
        # --- Generate Visualizations ---
        logger.info("Attempting to generate visualizations...")
        figures_path = Path(config.paths.figures_dir)
        run_timestamp = logger.get_run_timestamp() # Get the timestamp from the logger

        # Consolidate data for plotting
        period_model_to_plot = period_model if 'period_model' in locals() else None
        axis_model_to_plot = axis_model if 'axis_model' in locals() else None
        
        period_results_for_plot = period_results if 'period_results' in locals() else {}
        axis_results_for_plot = axis_results if 'axis_results' in locals() else {}

        # Ensure datasets are available for plotting if they were loaded
        # These might be Subsets, so we might need .dataset attribute for some plots
        train_ds_period_plot = train_dataset if 'train_dataset' in locals() else None
        val_ds_period_plot = val_dataset if 'val_dataset' in locals() else None
        test_ds_period_plot = test_dataset if 'test_dataset' in locals() else None
        
        train_axis_ds_plot = axis_dataset_full if 'axis_dataset_full' in locals() else None
        val_axis_ds_plot = val_axis_dataset if 'val_axis_dataset' in locals() else None
        test_axis_ds_plot = test_axis_dataset if 'test_axis_dataset' in locals() else None # From main axis training
        
        # DataLoaders for feature importance or other plots needing iteration
        test_period_dl_plot = test_loader if 'test_loader' in locals() else None
        test_axis_dl_plot = test_axis_loader if 'test_axis_loader' in locals() else None

        if _VIS_EXAMPLES_IMPORTED:
            logger.info("Using visualization example wrappers for plotting.")
            if period_model_to_plot and period_results_for_plot.get('training_history') and period_results_for_plot.get('test_metrics'):
                try:
                    logger.info("Calling generate_period_visualizations wrapper...")
                    generate_period_visualizations(
                        model=period_model_to_plot,
                        train_dataset=train_ds_period_plot,
                        val_dataset=val_ds_period_plot,
                        test_dataset=test_ds_period_plot,
                        training_history=period_results_for_plot['training_history'], 
                        test_metrics=period_results_for_plot['test_metrics'],
                        config=config,
                        logger=logger,
                        output_dir=figures_path,
                        run_timestamp=run_timestamp
                    )
                except Exception as e:
                    logger.error(f"Error generating period visualizations using example wrapper: {e}", exc_info=True)
            else:
                logger.warning("Skipping period example visualizations due to missing model or results.")

            if axis_model_to_plot and axis_results_for_plot.get('training_history') and axis_results_for_plot.get('test_metrics'):
                try:
                    logger.info("Calling generate_axis_visualizations wrapper...")
                    # Ensure test_axis_ds_plot is an AxisDataset instance for the wrapper
                    current_test_axis_ds_for_plot = test_axis_ds_plot
                    if isinstance(test_axis_dl_plot, DataLoader) and hasattr(test_axis_dl_plot, 'dataset'):
                        current_test_axis_ds_for_plot = test_axis_dl_plot.dataset # Prefer dataset from loader if available

                    if current_test_axis_ds_for_plot:
                        generate_axis_visualizations(
                            model=axis_model_to_plot,
                            test_dataset=current_test_axis_ds_for_plot, 
                            training_history=axis_results_for_plot['training_history'],
                            test_metrics=axis_results_for_plot['test_metrics'],
                            config=config,
                            logger=logger,
                            output_dir=figures_path,
                            run_timestamp=run_timestamp
                            # period_predictions_test # This might be needed if not part of test_metrics
                        )
                    else:
                        logger.warning("Skipping axis example visualizations as test_axis_dataset was not readily available.")
                except Exception as e:
                    logger.error(f"Error generating axis visualizations using example wrapper: {e}", exc_info=True)
            else:
                logger.warning("Skipping axis example visualizations due to missing model or results.")
            
            if CAPTUM_AVAILABLE and period_model_to_plot and test_period_dl_plot: 
                 try:
                    logger.info("Calling generate_feature_importance wrapper...")
                    generate_feature_importance(
                        model=period_model_to_plot,
                        dataloader=test_period_dl_plot, 
                        config=config,
                        logger=logger,
                        device=config.device,
                        output_dir=figures_path,
                        run_timestamp=run_timestamp
                    )
                 except Exception as e:
                    logger.error(f"Error generating feature importance visualizations using example wrapper: {e}", exc_info=True)
            elif CAPTUM_AVAILABLE:
                logger.warning("Skipping feature importance example visualization due to missing period model or test loader.")

        else: # Fallback to individual plotting functions
            logger.info("Using individual plotting functions as example wrappers were not imported.")
            
            # Period Model Visualizations
            if period_model_to_plot and period_results_for_plot:
                logger.info("Generating individual period model visualizations...")
                try:
                    period_train_hist = period_results_for_plot.get('training_history')
                    if period_train_hist and isinstance(period_train_hist, list) and len(period_train_hist) > 0:
                        # The history from train_period_model is a list: [model_config_dict, history_dict]
                        actual_history_dict = period_train_hist[1] if len(period_train_hist) > 1 and isinstance(period_train_hist[1], dict) else period_train_hist[0]
                        plot_period_metrics_vs_epoch(
                            actual_history_dict,
                            figures_path / f"period_metrics_{run_timestamp}.png", 
                            logger=logger
                        )
                    
                    period_test_metrics = period_results_for_plot.get('test_metrics', {})
                    if 'predictions_df_path' in period_test_metrics:
                        preds_df = pd.read_csv(period_test_metrics['predictions_df_path'])
                        y_true_period_test = preds_df['true_period'].values
                        y_pred_period_test = preds_df['pred_period'].values
                        
                        plot_period_scatter(
                            y_true_period_test, y_pred_period_test, 
                            str(figures_path / f"period_scatter_test_{run_timestamp}.png"), # Ensure path is string
                            logger=logger,
                            title=f"Period Prediction Scatter (Test Set) - {run_timestamp}"
                        )
                        plot_period_error_histogram(
                            y_true_period_test, y_pred_period_test, 
                            str(figures_path / f"period_error_hist_test_{run_timestamp}.png"), # Ensure path is string
                            logger=logger,
                            title=f"Period Prediction Error Histogram (Test Set) - {run_timestamp}"
                        )
                    else:
                        logger.warning("Skipping period scatter/histogram as test predictions_df_path was not found.")
                    # Consider plot_ls_period_distribution(dataset.get_all_periods(), figures_path / f"ls_dist_{run_timestamp}.png", logger) if relevant dataset available

                except Exception as e:
                    logger.error(f"Error during individual period model visualizations: {e}", exc_info=True)

            # Axis Model Visualizations
            if axis_model_to_plot and axis_results_for_plot:
                logger.info("Generating individual axis model visualizations...")
                try:
                    axis_train_hist = axis_results_for_plot.get('training_history')
                    if axis_train_hist and isinstance(axis_train_hist, list) and len(axis_train_hist) > 0:
                        actual_history_dict = axis_train_hist[1] if len(axis_train_hist) > 1 and isinstance(axis_train_hist[1], dict) else axis_train_hist[0]
                        plot_axis_metrics_vs_epoch(
                            actual_history_dict, 
                            figures_path / f"axis_metrics_{run_timestamp}.png", 
                            logger=logger
                        )

                    axis_test_metrics = axis_results_for_plot.get('test_metrics', {})
                    pred_dirs = axis_test_metrics.get('predictions_dir') # List of lists/arrays
                    true_dirs = axis_test_metrics.get('targets_dir')   # List of lists/arrays
                    pred_kappas = axis_test_metrics.get('pred_concentrations') # List of lists/arrays or flat list

                    if pred_dirs and true_dirs:
                        pred_dirs_np = np.array(pred_dirs)
                        true_dirs_np = np.array(true_dirs)
                        
                        # Convert direction vectors to (lon, lat) degrees for plotting
                        true_lons, true_lats = [], []
                        for vec in true_dirs_np:
                            lon, lat = direction_vector_to_lon_lat(torch.tensor(vec).float()) # Assuming this utility exists
                            true_lons.append(lon.item())
                            true_lats.append(lat.item())
                        
                        pred_lons, pred_lats = [], []
                        for vec in pred_dirs_np:
                            lon, lat = direction_vector_to_lon_lat(torch.tensor(vec).float()) # Assuming this utility exists
                            pred_lons.append(lon.item())
                            pred_lats.append(lat.item())

                        angular_errors = axis_test_metrics.get('angular_errors_per_sample')
                        if not angular_errors and len(pred_dirs_np) == len(true_dirs_np):
                             angular_errors = [angular_distance(true_dirs_np[i], pred_dirs_np[i], use_numpy=True) for i in range(len(true_dirs_np))]

                        if angular_errors:
                            plot_angular_error_histogram(
                                np.array(angular_errors), 
                                str(figures_path / f"axis_error_hist_test_{run_timestamp}.png"), 
                                logger=logger,
                                title=f"Axis Angular Error Histogram (Test Set) - {run_timestamp}"
                            )
                        
                        plot_axis_scatter(
                            true_lons, true_lats, pred_lons, pred_lats, 
                            str(figures_path / f"axis_scatter_test_{run_timestamp}.png"), 
                            logger=logger,
                            title=f"Axis Prediction Scatter (Test Set) - {run_timestamp}"
                        )
                        plot_axis_sky_map(
                            true_lons, true_lats, "True Axes", 
                            str(figures_path / f"axis_sky_map_true_test_{run_timestamp}.png"), 
                            logger=logger
                        )
                        plot_axis_sky_map(
                            pred_lons, pred_lats, "Predicted Axes", 
                            str(figures_path / f"axis_sky_map_pred_test_{run_timestamp}.png"), 
                            logger=logger
                        )

                        if pred_kappas:
                            pred_kappas_np = np.array(pred_kappas).flatten()
                            plot_kappa_distribution(
                                pred_kappas_np, 
                                str(figures_path / f"axis_kappa_dist_test_{run_timestamp}.png"), 
                                logger=logger,
                                title=f"Axis Kappa Distribution (Test Set) - {run_timestamp}"
                            )
                            if angular_errors and len(angular_errors) == len(pred_kappas_np):
                                plot_error_vs_kappa(
                                    np.array(angular_errors), 
                                    pred_kappas_np, 
                                    str(figures_path / f"axis_error_vs_kappa_test_{run_timestamp}.png"), 
                                    logger=logger,
                                    title=f"Axis Error vs Kappa (Test Set) - {run_timestamp}"
                                )
                    else:
                        logger.warning("Skipping some axis plots as test predictions_dir/targets_dir were not found.")
                    
                    # Plotting phase-folded lightcurves for a few test examples
                    if test_axis_ds_plot and hasattr(test_axis_ds_plot, 'get_raw_item_for_plotting'): # Needs specific method
                        logger.info("Attempting to plot some phase-folded test lightcurves...")
                        num_to_plot = min(5, len(test_axis_ds_plot))
                        for i in range(num_to_plot):
                            try:
                                # This method needs to exist and return: (raw_time, raw_mag, period_used, true_axis_vector, asteroid_id)
                                plot_data = test_axis_ds_plot.get_raw_item_for_plotting(i) 
                                raw_time, raw_mag, period_used, true_axis_vector, asteroid_id = plot_data
                                plot_phase_folded_lightcurves(
                                    raw_time, raw_mag, period_used, 
                                    num_bins=config.data.num_axis_bins,
                                    title=f"Test Sample {asteroid_id} Phase Folded (P={period_used:.2f}h)",
                                    save_path=str(figures_path / f"phase_folded_test_sample_{asteroid_id}_{run_timestamp}.png"),
                                    logger=logger
                                )
                            except Exception as e_plot_lc:
                                logger.error(f"Could not plot phase-folded lightcurve for sample {i}: {e_plot_lc}", exc_info=True)
                    elif test_axis_loader: # Fallback to iterate loader if dataset method not available
                        logger.info("Attempting to plot phase-folded lightcurves by iterating test_axis_loader (less direct)...")
                        # This is more complex as it requires matching predictions/periods to raw data if not in dataset object
                        logger.warning("Plotting phase-folded LCs from loader directly is complex and not fully implemented here. Needs careful data tracking.")

                except ImportError as ie:
                    logger.error(f"ImportError during axis visualizations, possibly missing a utility like direction_vector_to_lon_lat: {ie}")
                except Exception as e:
                    logger.error(f"Error during individual axis model visualizations: {e}", exc_info=True)

            # Feature Importance Visualizations (if Captum available and functions are imported)
            if CAPTUM_AVAILABLE and period_model_to_plot and test_period_dl_plot:
                logger.info("Generating individual feature importance visualizations for period model...")
                # Check for specific plot functions for safety, as CAPTUM_AVAILABLE is broad
                if plot_integrated_gradients:
                    try:
                        logger.info("Attempting Integrated Gradients plot for period model.")
                        # This call needs to be adapted to how plot_integrated_gradients is defined.
                        # It might require specific target handling for regression.
                        # plot_integrated_gradients(
                        # period_model_to_plot, 
                        # test_period_dl_plot, 
                        # config.device, 
                        # save_path_prefix=str(figures_path / f"period_feat_imp_ig_{run_timestamp}"),
                        # logger=logger
                        # )
                        logger.warning("Call to plot_integrated_gradients is commented out pending review of its exact signature and requirements for regression.")
                    except Exception as e:
                        logger.error(f"Error plotting Integrated Gradients: {e}", exc_info=True)
                # Add calls to other captum plots like plot_occlusion_sensitivity if they are imported and ready
            elif not CAPTUM_AVAILABLE and (period_model_to_plot and period_results_for_plot):
                 logger.info("Captum library not available. Skipping feature importance plots.")

        # --- End of Visualization Generation ---

        # Save the full configuration used for this run (including any hyperopt updates)
        run_summary = {
            "config": make_serializable(config),
            "period_model_results": period_results_for_plot,
            "axis_model_results": axis_results_for_plot,
            "run_timestamp": run_timestamp
        }
        logger.info(f"Full results summary data: {run_summary}")

        return period_model, axis_model, period_results, axis_results
    
    except Exception as e:
        if logger:
            logger.critical(f"Unhandled exception in pipeline: {e}")
        else:
            print(f"CRITICAL ERROR: Unhandled exception in pipeline: {e}")
            import traceback
            traceback.print_exc()
        # Attempt to close logger gracefully
        if logger:
            logger.info(f"Pipeline finished in {time.time() - start_time:.4f} seconds")
            logger.close()
        sys.exit(1) # Exit with error code
    finally:
        # Ensure logger is closed even if errors occurred
        if logger:
            logger.info(f"Pipeline finished in {time.time() - start_time:.4f} seconds")
            logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Light curve analysis pipeline")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable even more detailed debug logging"
    )
    
    args = parser.parse_args()
    try:
        main(args.config, debug=args.debug)
    except Exception as e:
        print(f"Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 