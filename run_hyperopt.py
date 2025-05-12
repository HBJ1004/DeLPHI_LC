#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone script to run hyperparameter optimization for asteroid lightcurve analysis models.
This script can be used to run Optuna search locally without using colab_run.py.
"""

import os
import sys
import torch
import argparse
import logging
import time
from pathlib import Path
import json

from lc_pipeline.config import load_config, save_config, PipelineConfig
from lc_pipeline.data.datasets import AsteroidDataset, AxisDataset
from lc_pipeline.data.collate import collate_fn, generate_axis_data
from torch.utils.data import DataLoader, random_split

from lc_pipeline.training import (
    run_period_optimization,
    run_axis_optimization,
    update_config_with_best_params,
    train_period_model,
    train_axis_model
)

from lc_pipeline.models.period_nets import (
    PeriodLSTMNet, PeriodTransformerNet, PeriodLSTMWithLSPrior
)

from lc_pipeline.models.axis_nets import AxisCNNNet, PhaseAwareTransformerAxis
from lc_pipeline.utils.logging import Logger


# Renamed from load_data_for_hyperopt to avoid clash when imported by run_pipeline.py
def _standalone_load_data_for_hyperopt(config, logger):
    """Load and prepare datasets for hyperopt."""
    all_csv_files = []
    
    def collect_csv_files(directory_path, file_list):
        if not os.path.isdir(directory_path):
            logger.warning(f"Directory not found: {directory_path}, skipping.")
            return
        for item in os.listdir(directory_path):
            if item.lower().endswith('.csv'):
                full_path = os.path.join(directory_path, item)
                file_list.append(full_path)

    if config.data.use_synthetic_data:
        synthetic_dir = str(config.data.synthetic_data_dir) # Ensure it's a string
        logger.info(f"Collecting CSV files from synthetic data directory: {synthetic_dir}")
        collect_csv_files(synthetic_dir, all_csv_files)
        
    if config.data.use_damit_data:
        damit_dir = str(config.data.damit_data_dir) # Ensure it's a string
        logger.info(f"Collecting CSV files from DAMIT data directory: {damit_dir}")
        collect_csv_files(damit_dir, all_csv_files)

    if not all_csv_files:
        logger.error("No CSV files found in the specified data directories. Cannot proceed with hyperopt.")
        raise ValueError("No CSV files found for dataset creation.")

    logger.info(f"Found {len(all_csv_files)} CSV files for hyperopt.")
    
    max_files = config.data.max_damit_files if hasattr(config.data, 'max_damit_files') else None
    
    logger.info(f"Using max_damit_files: {max_files} for hyperopt if applicable (passed to dataset).")
    
    dataset = AsteroidDataset(
        csv_files=all_csv_files, # Corrected: pass list of file paths
        max_sequence_length=config.data.max_sequence_length,
        max_files=max_files, # max_files will be applied internally by AsteroidDataset
        logger=logger
    )
    
    logger.info(f"Dataset loaded with {len(dataset)} samples for hyperopt")
    
    # Split dataset
    # Using train_val_ratio for the combined size of train+val, then val_ratio for val within that.
    # This interpretation might differ from lc_pipeline.main.load_data if it uses val_ratio from total.
    # For consistency, ensure data splitting logic is the same or clearly defined for each context.
    
    # Assuming config.data.val_ratio is the proportion of the *total dataset* for validation
    # and train_val_ratio is for the training+validation set from which val_ratio is taken.
    # Let's make it simpler: train_ratio, val_ratio, test_ratio that sum to 1.
    # If config only has train_val_ratio and val_ratio (of train_val_set):
    
    total_size = len(dataset)
    if hasattr(config.data, 'test_ratio'):
        test_size = int(config.data.test_ratio * total_size)
        train_val_size = total_size - test_size
        val_size = int(config.data.val_ratio * train_val_size) # val_ratio of the (train+val) part
        train_size = train_val_size - val_size
    else: # Fallback if test_ratio is not defined, split total into train and val
        val_size = int(config.data.val_ratio * total_size) # val_ratio of total dataset
        train_size = total_size - val_size
        test_size = 0 # No test set in this case for hyperopt script if not specified

    if train_size + val_size + test_size != total_size: # Adjust train_size to account for rounding
        train_size = total_size - val_size - test_size

    logger.info(f"Splitting dataset for hyperopt: train ({train_size}), val ({val_size}), test ({test_size})")
    
    if train_size <= 0 or val_size <= 0:
        logger.error(f"Train ({train_size}) or Val ({val_size}) split resulted in zero or negative samples. Total: {total_size}")
        raise ValueError("Cannot perform hyperopt with zero samples in train or validation set.")

    splits = [train_size, val_size]
    if test_size > 0:
        splits.append(test_size)
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, 
            splits,
            generator=torch.Generator().manual_seed(config.seed)
        )
    else:
        train_dataset, val_dataset = random_split(
            dataset, 
            splits,
            generator=torch.Generator().manual_seed(config.seed)
        )
        test_dataset = None # Explicitly set to None
    
    # Create dataloaders
    # Use batch_size from period_model config as a default, can be overridden by Optuna
    batch_size_optuna = config.period_model.batch_size 

    # Determine num_workers, considering OS
    num_workers_config = getattr(config, 'num_workers', 0)
    if os.name == 'nt' and num_workers_config > 0:
        logger.warning("Running on Windows, setting num_workers to 0 for DataLoader in hyperopt script.")
        num_workers_final = 0
    else:
        num_workers_final = num_workers_config

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_optuna, # Optuna might vary this per trial
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers_final
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size_optuna, # Use same batch size for val during optuna trials
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers_final
    )
    
    # test_loader might not be used directly in hyperopt but good to have if data is split
    test_loader = None
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size_optuna,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers_final
        )
    
    return dataset, train_loader, val_loader, test_loader # test_loader can be None


# Renamed from optimize_period_model_standalone to avoid clash
def _standalone_optimize_period_model(config, train_loader, val_loader, logger):
    """Run Optuna hyperparameter optimization for period model (standalone context)."""
    logger.info("Starting period model hyperparameter optimization (standalone)")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and config.device == 'cuda' else "cpu")
    logger.info(f"Using device: {device}")
    
    # Run optimization
    best_params = run_period_optimization(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        logger=logger
    )
    
    # Log best parameters
    if best_params:
        logger.info(f"Best period model parameters: {best_params}")
        # Update config with best parameters
        config = update_config_with_best_params(config, best_params, "period")
    else:
        logger.warning("Period model optimization did not return best parameters.")

    # Save updated config
    config_save_path = os.path.join(config.paths.models_dir, "best_period_config_hyperopt_script.yaml")
    save_config(config, config_save_path) # save_config needs PipelineConfig
    logger.info(f"Saved best period model config from hyperopt script to {config_save_path}")
    
    return config, best_params


# Renamed from create_period_model_standalone
def _standalone_create_period_model(config, device, logger): # Added logger
    """Create period model with the given configuration (standalone context)."""
    # Create model based on model name
    model_name = config.period_model.model_name
    logger.info(f"Creating period model ({model_name}) for standalone script")

    # Hyperparameters from config (could be updated by Optuna via update_config_with_best_params)
    input_dim = config.period_model.input_dim
    hidden_dim = config.period_model.hidden_dim
    num_layers = config.period_model.num_layers
    dropout = config.period_model.dropout
    # Model-specific params that Optuna might tune
    num_heads = getattr(config.period_model, 'num_heads', 8)
    prior_weight = getattr(config.period_model, 'prior_weight', 0.5)

    if model_name in ["PeriodLSTMNet", "lstm"]:
        model = PeriodLSTMNet(
            input_dim=input_dim, hidden_dim=hidden_dim,
            num_layers=num_layers, dropout=dropout
        ).to(device)
    elif model_name in ["PeriodTransformerNet", "transformer"]:
        model = PeriodTransformerNet(
            input_dim=input_dim, hidden_dim=hidden_dim,
            num_layers=num_layers, dropout=dropout, num_heads=num_heads
        ).to(device)
    elif model_name in ["PeriodLSTMWithLSPrior", "lstm_with_ls"]:
        model = PeriodLSTMWithLSPrior(
            input_dim=input_dim, hidden_dim=hidden_dim,
            num_layers=num_layers, dropout=dropout, prior_weight=prior_weight
        ).to(device)
    else:
        logger.error(f"Unknown period model type for standalone script: {model_name}")
        raise ValueError(f"Unknown period model type: {model_name}")
    
    return model


# Renamed from train_optimized_period_model_standalone
def _standalone_train_optimized_period_model(config, train_loader, val_loader, logger):
    """Train period model with optimized hyperparameters (standalone context)."""
    logger.info("Training period model with optimized hyperparameters (standalone)")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and config.device == 'cuda' else "cpu")
    
    # Create model using the potentially updated config
    model = _standalone_create_period_model(config, device, logger) # Pass logger
    
    # Set checkpoint path
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    checkpoint_path = os.path.join(config.paths.models_dir, f"period_model_optimized_standalone_{timestamp}.pt")
    
    # Train model
    # It returns (trained_model_object, history_dictionary)
    final_model, training_history_dict_from_train = train_period_model(
        model=model, # Pass the newly created model
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=config.period_model.lr,
        weight_decay=config.period_model.weight_decay,
        num_epochs=config.period_model.epochs, # Use full epochs from config
        device=device,
        patience=config.period_model.patience,
        logger=logger,
        checkpoint_path=checkpoint_path # Allow saving checkpoint for this full run
    )
    
    # Prepare a sanitized history for JSON serialization
    sanitized_history_for_json = {
        key: training_history_dict_from_train.get(key, []) 
        for key in ['train_loss', 'val_loss', 'val_mae', 'val_rmse']
    }
    # You can add other expected scalar/list-of-scalar keys here if your history_dict contains more.

    history_path = os.path.join(config.paths.results_dir, f"period_training_history_standalone_{timestamp}.json")
    with open(history_path, 'w') as f:
        json.dump(sanitized_history_for_json, f, indent=2) # Dump the sanitized history
    
    logger.info(f"Saved standalone training history to {history_path}")
    
    return final_model, checkpoint_path


# Renamed from optimize_axis_model_standalone
def _standalone_optimize_axis_model(config, train_loader_axis, val_loader_axis, period_model, logger):
    """Run Optuna hyperparameter optimization for axis model (standalone context)."""
    logger.info("Starting axis model hyperparameter optimization (standalone)")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and config.device == 'cuda' else "cpu")
    logger.info(f"Using device: {device}")
    
    # Run optimization
    best_params = run_axis_optimization(
        config=config,
        train_loader=train_loader_axis,
        val_loader=val_loader_axis,
        period_model=period_model,
        device=device,
        logger=logger
    )
    
    # Log best parameters
    if best_params:
        logger.info(f"Best axis model parameters from standalone script: {best_params}")
        # Update config with best parameters for axis model
        config = update_config_with_best_params(config, best_params, "axis")
    else:
        logger.warning("Axis model optimization did not return best parameters.")

    # Save updated config for axis model
    axis_config_save_path = os.path.join(config.paths.models_dir, "best_axis_config_hyperopt_script.yaml")
    save_config(config, axis_config_save_path) # save_config needs PipelineConfig
    logger.info(f"Saved best axis model config from hyperopt script to {axis_config_save_path}")

    return config, best_params


# Renamed from create_axis_model_standalone
def _standalone_create_axis_model(config, device, logger): # Added logger
    """Create axis model with the given configuration (standalone context)."""
    # Create model based on model name
    model_name = config.axis_model.model_name
    logger.info(f"Creating axis model ({model_name}) for standalone script")

    # Removed outdated/unused Hyperparameters from config:
    # input_channels = config.axis_model.input_channels # This was causing an AttributeError
    # hidden_dims_cnn = config.axis_model.hidden_dims_cnn
    # kernel_sizes = config.axis_model.kernel_sizes
    # output_dim_cnn = config.axis_model.output_dim_cnn
    
    # Parameters for PhaseAwareTransformerAxis (might need review if this model is used)
    # sequence_length = config.data.max_sequence_length 
    # num_layers_transformer = getattr(config.axis_model, 'num_layers_transformer', 2)
    # hidden_dim_transformer = getattr(config.axis_model, 'hidden_dim_transformer', 128)
    # num_heads_transformer = getattr(config.axis_model, 'num_heads_transformer', 4)
    # dropout_transformer = getattr(config.axis_model, 'dropout_transformer', 0.1)

    if model_name in ["AxisCNNNet", "cnn"]:
        model = AxisCNNNet(
            input_features=1, # Assuming folded light curve is [batch, num_bins] -> [batch, num_bins, 1]
            hidden_dim=config.axis_model.hidden_dim,
            blocks=config.axis_model.blocks,
            initial_channels=config.axis_model.initial_channels,
            kernel=config.axis_model.kernel,
            dropout=config.axis_model.dropout,
            num_bins=config.data.num_axis_bins, # Use num_axis_bins from data config
            use_norm=config.axis_model.use_norm,
            use_quaternions=config.axis_model.use_quaternions
        ).to(device)
    elif model_name in ["PhaseAwareTransformerAxis", "phase_transformer"]:
        logger.warning(f"PhaseAwareTransformerAxis selected in standalone script. Its parameterization might need review.")
        # Placeholder/example parameterization, ensure it matches model definition and config intentions
        # The original `input_dim=sequence_length * input_channels` was problematic.
        # Assuming PhaseAwareTransformerAxis expects features per bin after an embedding layer.
        # The `input_dim` for the transformer's embedding layer is often a hyperparameter itself.
        # For now, using num_axis_bins as sequence length and a placeholder for feature/embedding dim.
        
        # Determine output_dim based on quaternion usage
        output_dim = 4 if config.axis_model.use_quaternions else 6 

        model = PhaseAwareTransformerAxis(
            input_dim=config.data.num_axis_bins, # This is likely sequence_length for transformer
                                                # The actual feature dimension per bin might be 1, then embedded.
                                                # Or this input_dim is the embedding dim itself.
                                                # Consult PhaseAwareTransformerAxis definition.
            hidden_dim=getattr(config.axis_model, 'hidden_dim_transformer', 128),
            num_layers=getattr(config.axis_model, 'num_layers_transformer', 2),
            num_heads=getattr(config.axis_model, 'num_heads_transformer', 4),
            dropout=getattr(config.axis_model, 'dropout_transformer', 0.1),
            output_dim=output_dim 
        ).to(device)
    else:
        logger.error(f"Unknown axis model type for standalone script: {model_name}")
        raise ValueError(f"Unknown axis model type: {model_name}")
    
    return model


# Renamed from train_optimized_axis_model_standalone
def _standalone_train_optimized_axis_model(config, train_loader_period, val_loader_period, period_model, logger):
    """Train axis model with optimized hyperparameters (standalone context)."""
    logger.info("Training axis model with optimized hyperparameters (standalone)")
    
    device = torch.device("cuda" if torch.cuda.is_available() and config.device == 'cuda' else "cpu")
    
    # 1. Get Subsets from the period data loaders
    train_subset_period = train_loader_period.dataset
    val_subset_period = val_loader_period.dataset
    logger.info(f"Original train subset size: {len(train_subset_period)}, val subset size: {len(val_subset_period)}")

    # 2. Generate axis data using the provided period_model
    logger.info("Generating axis data for final training run...")
    axis_data_train_np, axis_targets_train_np, _ = generate_axis_data(
        dataset=train_subset_period,
        period_model=period_model, # Use the passed (trained or optimized) period model
        num_bins=config.data.num_axis_bins,
        smooth=config.data.smooth_axis_data,
        use_true_periods=config.axis_model.use_true_periods, # Use config setting for final train
        use_quaternions=config.axis_model.use_quaternions,
        return_ids=True
    )
    axis_data_val_np, axis_targets_val_np, _ = generate_axis_data(
        dataset=val_subset_period,
        period_model=period_model,
        num_bins=config.data.num_axis_bins,
        smooth=config.data.smooth_axis_data,
        use_true_periods=config.axis_model.use_true_periods,
        use_quaternions=config.axis_model.use_quaternions,
        return_ids=True
    )

    if axis_data_train_np is None or len(axis_data_train_np) == 0 or \
       axis_data_val_np is None or len(axis_data_val_np) == 0:
        logger.error("Failed to generate sufficient axis data for final training. Aborting axis model training.")
        return None, None # Or raise an error

    # 3. Create AxisDataset instances
    logger.info(f"Creating AxisDatasets: Train size {len(axis_data_train_np)}, Val size {len(axis_data_val_np)}")
    final_train_axis_dataset = AxisDataset(
        phase_curves=axis_data_train_np,
        axis_targets=axis_targets_train_np,
        use_quaternions=config.axis_model.use_quaternions
    )
    final_val_axis_dataset = AxisDataset(
        phase_curves=axis_data_val_np,
        axis_targets=axis_targets_val_np,
        use_quaternions=config.axis_model.use_quaternions
    )

    if len(final_train_axis_dataset) == 0:
        logger.error("Final training AxisDataset is empty. Aborting.")
        return None, None

    # 4. Create new DataLoaders for AxisDataset
    # Use batch_size from axis_model config for this final training
    final_train_axis_loader = DataLoader(
        final_train_axis_dataset,
        batch_size=config.axis_model.batch_size,
        shuffle=True,
        # No collate_fn needed for AxisDataset as it yields tensors directly
        num_workers=getattr(config, 'num_workers', 0)
    )
    # Val loader might be empty if all data went to train and no val_ratio for axis phase specifically
    final_val_axis_loader = None
    if len(final_val_axis_dataset) > 0:
        final_val_axis_loader = DataLoader(
            final_val_axis_dataset,
            batch_size=config.axis_model.batch_size,
            shuffle=False,
            num_workers=getattr(config, 'num_workers', 0)
        )
    else:
        logger.warning("Final validation AxisDataset is empty. Validation will be skipped during final axis training.")
        # Pass an empty loader or handle in train_axis_model if val_loader is None
        # For now, train_axis_model should handle val_loader being effectively empty or None.
        # Creating a loader with an empty dataset might cause issues, let's pass None if empty.
        final_val_axis_loader = val_loader_period # Fallback to original val_loader to avoid None if train_axis_model expects a loader
                                                # This is not ideal. train_axis_model should gracefully handle an empty or None val_loader.
                                                # For now, let's ensure train_axis_model can handle this.
                                                # Better: pass a loader with an empty AxisDataset if val_axis_dataset is empty but loader required.
        # Let's assume train_axis_model can handle val_loader being an empty loader.
        # So if final_val_axis_dataset is empty, final_val_axis_loader will also be empty.


    # Create axis model using the potentially updated config
    model = _standalone_create_axis_model(config, device, logger)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    checkpoint_path = os.path.join(config.paths.models_dir, f"axis_model_optimized_standalone_{timestamp}.pt")
    
    logger.info(f"Training final optimized axis model with: epochs={config.axis_model.epochs}, lr={config.axis_model.lr}, wd={config.axis_model.weight_decay}")
    
    # 6. Pass new DataLoaders to train_axis_model
    final_model, training_history_dict_from_train = train_axis_model(
        model=model,
        train_loader=final_train_axis_loader, # Use the new axis data loader
        val_loader=final_val_axis_loader if final_val_axis_loader and len(final_val_axis_dataset) > 0 else None, # Pass None if val set is empty
        device=device,
        epochs=config.axis_model.epochs, 
        lr=config.axis_model.lr,
        weight_decay=config.axis_model.weight_decay,
        patience=config.axis_model.patience,
        checkpoint_path=checkpoint_path,
        logger=logger
    )
    
    # Prepare a sanitized history for JSON serialization
    # Common keys for axis model history might be: train_loss, val_loss, val_angular_error, val_mse_vec, etc.
    # Adjust keys based on what train_axis_model actually returns in its history dict.
    expected_axis_history_keys = [
        'train_loss', 'val_loss', 'val_angular_error', 'val_mse_vec1', 'val_mse_vec2', 
        'val_vmf_loss1', 'val_vmf_loss2', 'val_combined_loss', 'val_quat_angle_diff'
    ]
    sanitized_history_for_json = {
        key: training_history_dict_from_train.get(key, []) 
        for key in expected_axis_history_keys
        if training_history_dict_from_train.get(key) is not None # Only include keys actually present
    }

    # Save training history
    history_path = os.path.join(config.paths.results_dir, f"axis_training_history_standalone_{timestamp}.json")
    with open(history_path, 'w') as f:
        json.dump(sanitized_history_for_json, f, indent=2)
    
    logger.info(f"Saved standalone training history to {history_path}")
    
    return final_model, checkpoint_path


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run hyperparameter optimization for asteroid lightcurve models (standalone)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--model", type=str, choices=["period", "axis", "both"], default="both", 
                        help="Which model to optimize")
    parser.add_argument("--train-after", action="store_true", help="Train model with optimized hyperparameters after search")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (overrides config)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory in config")
    parser.add_argument("--max-files", type=int, default=None, help="Max data files (overrides config)")
    parser.add_argument("--hyperopt-trials", type=int, default=None, help="Number of Optuna trials (overrides config)")
    parser.add_argument("--hyperopt-epochs", type=int, default=None, help="Epochs per Optuna trial (overrides config)")

    args = parser.parse_args()
    
    # Load config - returns PipelineConfig
    pipeline_config_obj = load_config(args.config)
    
    # --- Override config with args --- 
    if args.seed is not None:
        pipeline_config_obj.seed = args.seed
    if args.debug:
        pipeline_config_obj.logging.log_level = "DEBUG"
    if args.output_dir:
        pipeline_config_obj.paths.base_dir = args.output_dir
        # Update other paths based on new base_dir if they are relative
        pipeline_config_obj.paths.models_dir = os.path.join(args.output_dir, "models")
        pipeline_config_obj.paths.results_dir = os.path.join(args.output_dir, "results")
        pipeline_config_obj.paths.figures_dir = os.path.join(args.output_dir, "figures")
        pipeline_config_obj.paths.logs_dir = os.path.join(args.output_dir, "logs")
    if args.max_files is not None:
        pipeline_config_obj.data.max_damit_files = args.max_files
    if args.hyperopt_trials is not None:
        pipeline_config_obj.period_model.optuna_trials = args.hyperopt_trials
        pipeline_config_obj.axis_model.optuna_trials = args.hyperopt_trials # Assuming same for axis
    if args.hyperopt_epochs is not None:
        pipeline_config_obj.period_model.optuna_epochs = args.hyperopt_epochs
        pipeline_config_obj.axis_model.optuna_epochs = args.hyperopt_epochs # Assuming same for axis

    # Apply test_mode defaults if enabled in config and not overridden by CLI args
    if hasattr(pipeline_config_obj, 'test_mode') and pipeline_config_obj.test_mode:
        TEST_MODE_OPTUNA_TRIALS = 1
        TEST_MODE_OPTUNA_EPOCHS = 2 # Epochs per trial during Optuna
        TEST_MODE_MAIN_EPOCHS = 2   # Epochs for final training after Optuna

        print(f"INFO: run_hyperopt.py: test_mode is True in config.")

        # Override Optuna trials if not set by CLI
        if args.hyperopt_trials is None:
            pipeline_config_obj.period_model.optuna_trials = TEST_MODE_OPTUNA_TRIALS
            pipeline_config_obj.axis_model.optuna_trials = TEST_MODE_OPTUNA_TRIALS
            print(f"INFO: run_hyperopt.py: test_mode overriding Optuna trials to {TEST_MODE_OPTUNA_TRIALS}")
        else:
            print(f"INFO: run_hyperopt.py: test_mode not overriding Optuna trials, using CLI value: {args.hyperopt_trials}")

        # Override Optuna epochs per trial if not set by CLI
        if args.hyperopt_epochs is None:
            pipeline_config_obj.period_model.optuna_epochs = TEST_MODE_OPTUNA_EPOCHS
            pipeline_config_obj.axis_model.optuna_epochs = TEST_MODE_OPTUNA_EPOCHS
            print(f"INFO: run_hyperopt.py: test_mode overriding Optuna epochs per trial to {TEST_MODE_OPTUNA_EPOCHS}")
        else:
            print(f"INFO: run_hyperopt.py: test_mode not overriding Optuna epochs, using CLI value: {args.hyperopt_epochs}")
        
        # Also override main training epochs for the --train-after phase in test_mode
        # These are separate from optuna_epochs (epochs *per trial*)
        # No direct CLI arg in run_hyperopt.py for these, so test_mode always overrides if active.
        pipeline_config_obj.period_model.epochs = TEST_MODE_MAIN_EPOCHS
        pipeline_config_obj.axis_model.epochs = TEST_MODE_MAIN_EPOCHS
        print(f"INFO: run_hyperopt.py: test_mode overriding final training epochs to {TEST_MODE_MAIN_EPOCHS}")

    # Create directories if they don't exist
    os.makedirs(pipeline_config_obj.paths.logs_dir, exist_ok=True)
    os.makedirs(pipeline_config_obj.paths.models_dir, exist_ok=True)
    os.makedirs(pipeline_config_obj.paths.results_dir, exist_ok=True)
    
    # Set up logging using Logger class
    # Convert PipelineConfig to dict for Logger if needed, or ensure Logger handles it.
    # Assuming Logger.config is for W&B/Tensorboard hparams, so pass main config dict.
    main_config_for_logger = pipeline_config_obj.__dict__ # Or a better to_dict() if PipelineConfig has one

    logger = Logger(
        log_dir=pipeline_config_obj.paths.logs_dir,
        experiment_name=f"hyperopt_standalone_{pipeline_config_obj.period_model.model_name}_{args.model}",
        log_level=pipeline_config_obj.logging.log_level,
        log_to_console=True, # Default to True for script
        log_to_file=True,    # Default to True for script
        config=main_config_for_logger
    )
    
    # Set random seed
    torch.manual_seed(pipeline_config_obj.seed)
    
    # Load data using the renamed function
    dataset, train_loader, val_loader, test_loader = _standalone_load_data_for_hyperopt(pipeline_config_obj, logger)
    
    period_model_trained = None # To store the period model if trained

    # Run hyperparameter optimization for period model
    if args.model in ["period", "both"]:
        pipeline_config_obj, best_period_params = _standalone_optimize_period_model(
            pipeline_config_obj, train_loader, val_loader, logger
        )
        
        # Train with best parameters if requested
        if args.train_after:
            period_model_trained, period_checkpoint = _standalone_train_optimized_period_model(
                pipeline_config_obj, train_loader, val_loader, logger
            )
        elif args.model == "both": # If optimizing both and not training after, still need a period model for axis opt
            logger.info("Creating period model with (potentially optimized) params for axis optimization.")
            device = torch.device("cuda" if torch.cuda.is_available() and pipeline_config_obj.device == 'cuda' else "cpu")
            period_model_trained = _standalone_create_period_model(pipeline_config_obj, device, logger)
            # Optionally, a quick train here if axis_opt absolutely needs a trained period model
            # For now, assuming axis_opt can work with an initialized (but not fully trained) period model
            # or that it will use true periods if period_model is just initialized.

    # Optimize axis model if requested (requires a period model instance)
    if args.model in ["axis", "both"]:
        if not period_model_trained: # Ensure period model is available
            logger.warning("Period model not trained or loaded. Axis optimization might rely on a default or dummy period model if run_axis_optimization requires it.")
            # Attempt to create a default period model if None
            # This is a fallback - ideally, period_model_trained should be set.
            device_for_fallback = torch.device("cuda" if torch.cuda.is_available() and pipeline_config_obj.device == 'cuda' else "cpu")
            period_model_trained = _standalone_create_period_model(pipeline_config_obj, device_for_fallback, logger)


        logger.info(f"Proceeding to axis optimization. Period model type: {type(period_model_trained)}")
        
        pipeline_config_obj, best_axis_params = _standalone_optimize_axis_model(
            pipeline_config_obj, 
            train_loader, 
            val_loader,   
            period_model_trained, 
            logger
        )
        
        if args.train_after:
            logger.info(f"Training axis model with optimized hyperparameters (following {args.model} model optimization).")
            _standalone_train_optimized_axis_model(
                 pipeline_config_obj, 
                 train_loader, # Pass original train_loader (for AsteroidDataset)
                 val_loader,   # Pass original val_loader (for AsteroidDataset)
                 period_model_trained, # Pass the trained/optimized period model
                 logger
            )

    logger.info("Standalone hyperparameter optimization script finished.")
    if 'logger' in locals() and logger is not None:
        logger.close()


if __name__ == "__main__":
    main() 