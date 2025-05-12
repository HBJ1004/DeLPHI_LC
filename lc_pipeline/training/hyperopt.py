#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hyperparameter optimization module for the light curve analysis pipeline.
Uses Optuna for hyperparameter search of both period and axis models.
"""

import os
import logging
import torch
import optuna
import numpy as np
from typing import Dict, Any, Optional, Union, Tuple, List
from torch.utils.data import DataLoader

from lc_pipeline.models.period_nets import (
    PeriodLSTMNet, PeriodTransformerNet, PeriodLSTMWithLSPrior, PeriodLogScaleNet
)
from lc_pipeline.models.axis_nets import AxisCNNNet, PhaseAwareTransformerAxis
from lc_pipeline.training.train_period import train_period_model
from lc_pipeline.training.train_axis import train_axis_model
from lc_pipeline.data.collate import generate_axis_data
from lc_pipeline.evaluation import evaluate_period_model, evaluate_axis_model
from lc_pipeline.data.datasets import AxisDataset


def objective_period(
    trial: optuna.Trial,
    config: Any,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    logger: Optional[logging.Logger] = None
) -> float:
    """
    Objective function for Optuna hyperparameter optimization of the period model.
    
    Args:
        trial: Optuna trial object
        config: Configuration object
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to run on
        logger: Logger object
        
    Returns:
        float: Metric to optimize (lower is better for period MAE)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Use a convenience variable for period model config
    pm_config = config.period_model
    
    # Set the model name based on the config
    model_name = getattr(pm_config, 'model_name', 'PeriodLSTMWithLSPrior') # Default if not found
    
    # Suggest hyperparameters from the search space
    hidden_dim = trial.suggest_int(
        "hidden_dim", 
        getattr(pm_config, 'hidden_dim_range', [64, 256])[0], 
        getattr(pm_config, 'hidden_dim_range', [64, 256])[1],
        log=True
    )
    num_layers = trial.suggest_int(
        "num_layers", 
        getattr(pm_config, 'num_layers_range', [2, 5])[0], 
        getattr(pm_config, 'num_layers_range', [2, 5])[1]
    )
    dropout = trial.suggest_float(
        "dropout", 
        getattr(pm_config, 'dropout_range', [0.1, 0.5])[0], 
        getattr(pm_config, 'dropout_range', [0.1, 0.5])[1]
    )
    lr = trial.suggest_float(
        "lr", 
        getattr(pm_config, 'lr_range', [0.0001, 0.01])[0], 
        getattr(pm_config, 'lr_range', [0.0001, 0.01])[1],
        log=True
    )
    weight_decay = trial.suggest_float(
        "weight_decay", 
        getattr(pm_config, 'weight_decay_range', [1e-05, 0.001])[0], 
        getattr(pm_config, 'weight_decay_range', [1e-05, 0.001])[1],
        log=True
    )
    
    # Additional model-specific parameters
    input_dim = getattr(pm_config, 'input_dim', 17) # Default if not found
    
    # Create the model based on the model_name
    if model_name in ["PeriodLSTMNet", "lstm"]:
        model = PeriodLSTMNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        ).to(device)
    elif model_name in ["PeriodTransformerNet", "transformer"]:
        num_heads = trial.suggest_int("num_heads", 1, 8)
        model = PeriodTransformerNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            num_heads=num_heads
        ).to(device)
    elif model_name in ["PeriodLSTMWithLSPrior", "lstm_with_ls"]:
        prior_weight = trial.suggest_float("prior_weight", 0.0, 1.0)
        model = PeriodLSTMWithLSPrior(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            prior_weight=prior_weight
        ).to(device)
    elif model_name in ["PeriodLogScaleNet", "log_scale"]:
        model = PeriodLogScaleNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        ).to(device)
    else:
        logger.error(f"Unknown period model type: {model_name}")
        raise ValueError(f"Unknown period model type: {model_name}")
    
    # Train the model for a limited number of epochs
    _, training_history = train_period_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=lr,
        weight_decay=weight_decay,
        num_epochs=getattr(pm_config, 'optuna_epochs', 10),
        device=device,
        patience=5,  # Use shorter patience for optimization
        logger=logger,
        debug=True,
        checkpoint_path=None,  # Don't save intermediate checkpoints
        period_scale_factor=getattr(pm_config, 'period_scale_factor', 50.0), # Added getattr
        use_log_scale=getattr(pm_config, 'use_log_scale', False), # Added getattr
        min_period_for_loss=getattr(pm_config, 'min_period', 2.0), # Added getattr
        l2_lambda_loss=getattr(pm_config, 'l2_lambda_loss', 0.0) # Changed from .get to getattr
    )
    
    # Extract the best validation metric (typically MAE) from the training history
    best_val_metric = min(training_history.get('val_mae', [float('inf')]))
    
    # Log the trial results
    logger.info(f"Trial {trial.number}: Hidden dim={hidden_dim}, Num layers={num_layers}, "
                f"Dropout={dropout}, LR={lr}, Weight decay={weight_decay}, "
                f"Best val MAE={best_val_metric:.4f}")
    
    return best_val_metric


def objective_axis(
    trial: optuna.Trial,
    config: Any,
    train_loader: DataLoader,
    val_loader: DataLoader,
    period_model: torch.nn.Module,
    device: torch.device,
    logger: Optional[logging.Logger] = None
) -> float:
    """
    Objective function for Optuna hyperparameter optimization of the axis model.
    
    Args:
        trial: Optuna trial object
        config: Configuration object
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        period_model: Trained period model
        device: Device to run on
        logger: Logger object
        
    Returns:
        float: Metric to optimize (lower is better for angular error)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Set the model name based on the config
    model_name = config.axis_model.model_name
    axis_model_config = config.axis_model # Convenience variable

    # Suggest hyperparameters from the search space using getattr for safety
    hidden_dim = trial.suggest_int(
        "hidden_dim",
        *getattr(axis_model_config, 'hidden_dim_range', [64, 128]), # Default: [64, 128]
        log=True
    )
    dropout = trial.suggest_float(
        "dropout",
        *getattr(axis_model_config, 'dropout_range', [0.1, 0.5])    # Default: [0.1, 0.5]
    )
    lr = trial.suggest_float(
        "lr",
        *getattr(axis_model_config, 'lr_range', [0.0001, 0.01]),   # Default: [0.0001, 0.01]
        log=True
    )
    weight_decay = trial.suggest_float(
        "weight_decay",
        *getattr(axis_model_config, 'weight_decay_range', [1e-5, 1e-3]), # Default: [1e-5, 1e-3]
        log=True
    )
    
    # Create the model based on the model_name
    if model_name in ["AxisCNNNet", "cnn"]:
        blocks_range = getattr(axis_model_config, 'blocks_range', [2, 5]) # Default: [2, 5]
        blocks = trial.suggest_int("blocks", blocks_range[0], blocks_range[1])
        
        initial_channels_range = getattr(axis_model_config, 'initial_channels_range', [16, 64]) # Default: [16, 64]
        initial_channels = trial.suggest_int(
            "initial_channels", 
            initial_channels_range[0], 
            initial_channels_range[1],
            log=True
        )
        
        kernel_config_range = getattr(axis_model_config, 'kernel_range', [3, 7]) # Default: [3, 7]
        default_kernel_val = getattr(axis_model_config, 'kernel', 5) # Default single kernel value

        possible_kernels = []
        if len(kernel_config_range) == 2 and kernel_config_range[0] <= kernel_config_range[1]:
            # If range is [min, max], generate odd numbers in that range
            for k_val in range(kernel_config_range[0], kernel_config_range[1] + 1):
                if k_val % 2 != 0:
                    possible_kernels.append(k_val)
        
        if not possible_kernels: # If range was invalid or yielded no odd kernels
            # Fallback to common odd kernels, or the single default if absolutely necessary
            possible_kernels = [k for k in [3, 5, 7] if hasattr(axis_model_config, 'kernel') and k == default_kernel_val] 
            if not possible_kernels: # If default_kernel_val isn't in [3,5,7] or not defined
                 possible_kernels = [3, 5, 7] # Absolute fallback list
            if default_kernel_val not in possible_kernels and hasattr(axis_model_config, 'kernel'): # Add default if it's not there yet
                # This case is tricky, if default is e.g. 4, and range was bad, do we offer 4?
                # Forcing categorical, so it should be one of the choices.
                # Let's ensure the default_kernel_val (if odd) is an option if no range worked.
                if default_kernel_val % 2 != 0 and default_kernel_val not in possible_kernels:
                    possible_kernels.append(default_kernel_val)
                    possible_kernels.sort()
        
        # Remove duplicates and sort, ensure at least one option
        possible_kernels = sorted(list(set(possible_kernels)))
        if not possible_kernels:
            possible_kernels = [default_kernel_val] # Ensure at least the default value is an option

        kernel = trial.suggest_categorical("kernel", possible_kernels)

        model = AxisCNNNet(
            input_features=1,  # Will be dynamically adjusted by model if needed
            hidden_dim=hidden_dim,
            blocks=blocks,
            initial_channels=initial_channels,
            kernel=kernel,
            dropout=dropout,
            num_bins=config.data.num_axis_bins,
            use_norm=getattr(axis_model_config, 'use_norm', True),
            use_quaternions=getattr(axis_model_config, 'use_quaternions', True)
        ).to(device)
    elif model_name in ["PhaseAwareTransformerAxis", "transformer"]:
        # Assuming PhaseAwareTransformerAxis might also have num_layers and num_heads ranges
        num_layers_range = getattr(axis_model_config, 'num_layers_range', [1, 5]) # Example default
        num_layers = trial.suggest_int("num_layers", num_layers_range[0], num_layers_range[1])
        
        num_heads_range = getattr(axis_model_config, 'num_heads_range', [1, 8]) # Example default
        num_heads = trial.suggest_int("num_heads", num_heads_range[0], num_heads_range[1])
        
        model = PhaseAwareTransformerAxis(
            input_features=1,  # Will be dynamically adjusted by model if needed
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            num_heads=num_heads,
            num_bins=config.data.num_axis_bins,
            use_quaternions=getattr(axis_model_config, 'use_quaternions', True)
        ).to(device)
    else:
        logger.error(f"Unknown axis model type: {model_name}")
        raise ValueError(f"Unknown axis model type: {model_name}")
    
    # Since we're optimizing the axis model, we need to generate axis training data
    # using the trained period model
    logger.info(f"Generating axis data using period model for trial {trial.number}")
    
    # Generate phase-folded lightcurves
    try:
        # Use the train dataset from the dataloader
        # These are expected to be AsteroidDataset instances (or subsets of it)
        train_subset = train_loader.dataset # This is a torch.utils.data.Subset
        val_subset = val_loader.dataset   # This is a torch.utils.data.Subset
        
        # Generate axis data (numpy arrays)
        # For the training set
        axis_data_train_np, axis_targets_train_np, _ = generate_axis_data(
            dataset=train_subset, # Pass the Subset directly, generate_axis_data will handle it
            period_model=period_model,
            num_bins=config.data.num_axis_bins,
            smooth=config.data.smooth_axis_data,
            use_true_periods=getattr(config.axis_model, 'use_true_periods_in_hyperopt', config.axis_model.use_true_periods),
            use_quaternions=config.axis_model.use_quaternions,
            return_ids=True
        )
        
        # For the validation set
        axis_data_val_np, axis_targets_val_np, _ = generate_axis_data(
            dataset=val_subset, # Pass the Subset directly, generate_axis_data will handle it
            period_model=period_model,
            num_bins=config.data.num_axis_bins,
            smooth=config.data.smooth_axis_data,
            use_true_periods=getattr(config.axis_model, 'use_true_periods_in_hyperopt', config.axis_model.use_true_periods),
            use_quaternions=config.axis_model.use_quaternions,
            return_ids=True
        )

        if axis_data_train_np is None or len(axis_data_train_np) == 0 or \
           axis_data_val_np is None or len(axis_data_val_np) == 0:
            logger.error("Generated axis data numpy arrays are empty or None")
            return float('inf')

        # Create AxisDataset instances
        # The AxisDataset constructor handles quaternion conversion based on its own use_quaternions flag
        # which should align with config.axis_model.use_quaternions used in generate_axis_data
        axis_train_dataset_obj = AxisDataset(
            phase_curves=axis_data_train_np, 
            axis_targets=axis_targets_train_np,
            use_quaternions=config.axis_model.use_quaternions 
        )
        axis_val_dataset_obj = AxisDataset(
            phase_curves=axis_data_val_np, 
            axis_targets=axis_targets_val_np,
            use_quaternions=config.axis_model.use_quaternions
        )

        if len(axis_train_dataset_obj) == 0 or len(axis_val_dataset_obj) == 0:
            logger.error("Created AxisDataset objects are empty.")
            return float('inf')
            
        # Create dataloaders for axis training
        axis_train_loader = DataLoader(
            axis_train_dataset_obj, # Pass the created AxisDataset
            batch_size=config.axis_model.batch_size,
            shuffle=True,
            num_workers=0 # Keep as 0 for simplicity in hyperopt unless profiling shows bottleneck
        )
        
        axis_val_loader = DataLoader(
            axis_val_dataset_obj, # Pass the created AxisDataset
            batch_size=config.axis_model.batch_size,
            shuffle=False,
            num_workers=0 # Keep as 0
        )
        
    except Exception as e:
        logger.error(f"Error generating axis data: {e}")
        # Return a high value to penalize this trial
        return float('inf')
    
    # Train the model for a limited number of epochs
    # Unpack the model and history dictionary from the tuple returned by train_axis_model
    _, training_history_dict = train_axis_model(
        model=model,
        train_loader=axis_train_loader,
        val_loader=axis_val_loader,
        device=device,
        epochs=config.axis_model.optuna_epochs,
        lr=lr,
        weight_decay=weight_decay,
        patience=5,  # Use shorter patience for optimization
        checkpoint_path=None,  # Don't save intermediate checkpoints
        logger=logger
    )
    
    # Extract the best validation metric (typically angular error) from the training history
    # Use the unpacked training_history_dict here
    if 'val_mean_angular_error' in training_history_dict: # Check for the correct key
        # Ensure the list is not empty before calling min
        val_errors = training_history_dict['val_mean_angular_error']
        best_val_metric = min(val_errors) if val_errors else float('inf')
    elif 'val_angular_error' in training_history_dict: # Fallback to older key if present
        val_errors = training_history_dict['val_angular_error']
        best_val_metric = min(val_errors) if val_errors else float('inf')
    else:
        # If no angular error, use validation loss
        val_losses = training_history_dict.get('val_loss', [float('inf')])
        best_val_metric = min(val_losses) if val_losses else float('inf')
    
    # Log the trial results
    if model_name in ["AxisCNNNet", "cnn"]:
        logger.info(f"Trial {trial.number}: Hidden dim={hidden_dim}, Blocks={blocks}, "
                    f"Initial channels={initial_channels}, Kernel={kernel}, Dropout={dropout}, "
                    f"LR={lr}, Weight decay={weight_decay}, "
                    f"Best val angular error={best_val_metric:.4f}")
    else:
        logger.info(f"Trial {trial.number}: Hidden dim={hidden_dim}, Num layers={num_layers}, "
                    f"Num heads={num_heads}, Dropout={dropout}, "
                    f"LR={lr}, Weight decay={weight_decay}, "
                    f"Best val angular error={best_val_metric:.4f}")
    
    return best_val_metric


def run_period_optimization(
    config: Any,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Run Optuna hyperparameter optimization for the period model.
    
    Args:
        config: Configuration object
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to run on
        logger: Logger object
        
    Returns:
        Dict[str, Any]: Dictionary with best hyperparameters
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f"Starting period model hyperparameter optimization with {config.period_model.optuna_trials} trials")
    
    # Create an Optuna study
    study_name = f"period_model_{config.period_model.model_name}"
    try:
        # Use SQLite storage if a models_dir is provided
        if hasattr(config.paths, 'models_dir') and config.paths.models_dir:
            os.makedirs(config.paths.models_dir, exist_ok=True)
            storage_path = os.path.join(config.paths.models_dir, f"{study_name}.db")
            storage = f"sqlite:///{storage_path}"
        else:
            storage = None
        
        study = optuna.create_study(
            direction="minimize",  # Minimize MAE
            study_name=study_name,
            storage=storage,
            load_if_exists=True
        )
    except Exception as e:
        logger.warning(f"Failed to create Optuna study with storage: {e}")
        # Fall back to in-memory storage
        study = optuna.create_study(
            direction="minimize",
            study_name=study_name
        )
    
    # Define the objective function with fixed parameters
    objective = lambda trial: objective_period(
        trial, config, train_loader, val_loader, device, logger
    )
    
    # Run the optimization
    try:
        study.optimize(objective, n_trials=config.period_model.optuna_trials)
    except Exception as e:
        logger.error(f"Error during period optimization: {e}")
        # Return empty dict if optimization failed
        return {}
    
    # Get the best hyperparameters
    best_params = study.best_params
    best_value = study.best_value
    
    logger.info(f"Best period model hyperparameters found: {best_params}")
    logger.info(f"Best period model validation MAE: {best_value:.4f}")
    
    # Return the best hyperparameters
    return best_params


def run_axis_optimization(
    config: Any,
    train_loader: DataLoader,
    val_loader: DataLoader,
    period_model: torch.nn.Module,
    device: torch.device,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Run Optuna hyperparameter optimization for the axis model.
    
    Args:
        config: Configuration object
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        period_model: Trained period model
        device: Device to run on
        logger: Logger object
        
    Returns:
        Dict[str, Any]: Dictionary with best hyperparameters
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f"Starting axis model hyperparameter optimization with {config.axis_model.optuna_trials} trials")
    
    # Create an Optuna study
    study_name = f"axis_model_{config.axis_model.model_name}"
    try:
        # Use SQLite storage if a models_dir is provided
        if hasattr(config.paths, 'models_dir') and config.paths.models_dir:
            os.makedirs(config.paths.models_dir, exist_ok=True)
            storage_path = os.path.join(config.paths.models_dir, f"{study_name}.db")
            storage = f"sqlite:///{storage_path}"
        else:
            storage = None
        
        study = optuna.create_study(
            direction="minimize",  # Minimize angular error
            study_name=study_name,
            storage=storage,
            load_if_exists=True
        )
    except Exception as e:
        logger.warning(f"Failed to create Optuna study with storage: {e}")
        # Fall back to in-memory storage
        study = optuna.create_study(
            direction="minimize",
            study_name=study_name
        )
    
    # Define the objective function with fixed parameters
    objective = lambda trial: objective_axis(
        trial, config, train_loader, val_loader, period_model, device, logger
    )
    
    # Run the optimization
    try:
        study.optimize(objective, n_trials=config.axis_model.optuna_trials)
    except Exception as e:
        logger.error(f"Error during axis optimization: {e}")
        # Return empty dict if optimization failed
        return {}
    
    # Get the best hyperparameters
    best_params = study.best_params
    best_value = study.best_value
    
    logger.info(f"Best axis model hyperparameters found: {best_params}")
    logger.info(f"Best axis model validation angular error: {best_value:.4f}")
    
    # Return the best hyperparameters
    return best_params


def update_config_with_best_params(
    config: Any,
    best_params: Dict[str, Any],
    model_type: str
) -> Any:
    """
    Update the configuration object with the best hyperparameters found by Optuna.
    
    Args:
        config: Configuration object
        best_params: Dictionary with best hyperparameters
        model_type: Either "period" or "axis"
        
    Returns:
        Any: Updated configuration object
    """
    if not best_params:
        return config
    
    # Get the model config based on the model type
    if model_type.lower() == "period":
        model_config = config.period_model
    elif model_type.lower() == "axis":
        model_config = config.axis_model
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Update the model config with the best hyperparameters
    for param_name, param_value in best_params.items():
        if hasattr(model_config, param_name):
            setattr(model_config, param_name, param_value)
    
    return config 