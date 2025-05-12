#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline runner for asteroid lightcurve analysis.

This script provides a single entry point for running the entire pipeline with various options,
including hyperparameter optimization and model training.
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
from lc_pipeline.main import main as run_main_pipeline
from lc_pipeline.utils.logging import Logger

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Asteroid Lightcurve Pipeline Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config file")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override the output directory in config")
    
    # Pipeline control
    parser.add_argument("--mode", type=str, choices=["full", "period", "axis", "hyperopt"],
                        default="full", help="Pipeline mode to run")
    
    # Hyperopt options
    parser.add_argument("--hyperopt-model", type=str, choices=["period", "axis", "both"],
                        default="both", help="Which model to optimize in hyperopt mode")
    parser.add_argument("--hyperopt-trials", type=int, default=None,
                        help="Number of Optuna trials to run (overrides config)")
    parser.add_argument("--hyperopt-epochs", type=int, default=None,
                        help="Epochs per trial in hyperopt (overrides config)")
    parser.add_argument("--train-after-hyperopt", action="store_true",
                        help="Train model with optimized hyperparameters")
    
    # Data options
    parser.add_argument("--max-files", type=int, default=None,
                        help="Maximum number of data files to use (overrides config)")
    parser.add_argument("--test-mode", action="store_true",
                        help="Run in test mode with reduced dataset and iterations")
    
    # Training options
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of epochs for training (overrides config)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size for training (overrides config)")
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="Learning rate for training (overrides config)")
    
    # Device options
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"],
                        help="Device to use (overrides config)")
    
    # Misc options
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (overrides config)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    parser.add_argument("--no-plots", action="store_true",
                        help="Disable plot generation")
    
    return parser.parse_args()


def update_config_from_args(config, args):
    """Update configuration with command-line arguments."""
    # Override output directories if specified
    if args.output_dir:
        base_dir = args.output_dir
        config.paths.base_dir = base_dir
        config.paths.models_dir = os.path.join(base_dir, "models")
        config.paths.results_dir = os.path.join(base_dir, "results")
        config.paths.figures_dir = os.path.join(base_dir, "figures")
        config.paths.logs_dir = os.path.join(base_dir, "logs")
    
    # Override data options
    if args.max_files:
        config.data.max_damit_files = args.max_files
    
    # Override training options
    if args.epochs:
        config.period_model.epochs = args.epochs
        config.axis_model.epochs = args.epochs
    
    if args.batch_size:
        config.period_model.batch_size = args.batch_size
        config.axis_model.batch_size = args.batch_size
    
    if args.learning_rate:
        config.period_model.lr = args.learning_rate
        config.axis_model.lr = args.learning_rate
    
    # Override hyperopt options
    if args.hyperopt_trials:
        config.period_model.optuna_trials = args.hyperopt_trials
        config.axis_model.optuna_trials = args.hyperopt_trials
    
    if args.hyperopt_epochs:
        config.period_model.optuna_epochs = args.hyperopt_epochs
        config.axis_model.optuna_epochs = args.hyperopt_epochs
    
    # Override device options
    if args.device:
        config.device = args.device
    
    # Override misc options
    if args.seed:
        config.seed = args.seed
    
    if args.test_mode:
        config.test_mode = True
    
    if args.no_plots:
        config.logging.save_plots = False
    
    # Set pipeline workflow options based on mode
    if args.mode == "period":
        config.run_period_training = True
        config.run_axis_training = False
        config.run_hyperopt = False
    
    elif args.mode == "axis":
        config.run_period_training = True  # Still need period model for axis training
        config.run_axis_training = True
        config.run_hyperopt = False
    
    elif args.mode == "hyperopt":
        config.run_hyperopt = True
        
        # Set which models to run hyperopt for
        if args.hyperopt_model == "period":
            config.run_period_training = True
            config.run_axis_training = False
        elif args.hyperopt_model == "axis":
            config.run_period_training = True  # Still need period model for axis training
            config.run_axis_training = True
        elif args.hyperopt_model == "both":
            config.run_period_training = True
            config.run_axis_training = True
    
    # Debug mode
    if args.debug:
        config.logging.log_level = "DEBUG"
    
    # Apply test mode overrides AFTER all other CLI args and mode settings
    if config.test_mode:
        TEST_EPOCHS = 2
        TEST_OPTUNA_EPOCHS = 2 # Can be same or different from main training test epochs
        TEST_OPTUNA_TRIALS = 1
        
        # Standard logging for early messages before full logger setup
        print(f"INFO: Test mode enabled. Overriding epochs and potentially Optuna trials.")

        config.period_model.epochs = TEST_EPOCHS
        config.axis_model.epochs = TEST_EPOCHS
        print(f"INFO: Test mode: Main training epochs set to {TEST_EPOCHS}")

        config.period_model.optuna_epochs = TEST_OPTUNA_EPOCHS
        config.axis_model.optuna_epochs = TEST_OPTUNA_EPOCHS
        print(f"INFO: Test mode: Optuna trial epochs set to {TEST_OPTUNA_EPOCHS}")

        # Only override optuna_trials if the user hasn't explicitly set them via CLI
        if args.hyperopt_trials is None:
            config.period_model.optuna_trials = TEST_OPTUNA_TRIALS
            config.axis_model.optuna_trials = TEST_OPTUNA_TRIALS
            print(f"INFO: Test mode: Optuna trials set to {TEST_OPTUNA_TRIALS} (since --hyperopt-trials not specified).")
        else:
            print(f"INFO: Test mode: Optuna trials kept at user-specified {args.hyperopt_trials}.")

    return config


def run_hyperopt_mode(config, logger, args_for_hyperopt):
    """Run hyperparameter optimization using Optuna."""
    # Import the correctly named standalone functions from run_hyperopt.py
    from run_hyperopt import (
        _standalone_load_data_for_hyperopt, 
        _standalone_optimize_period_model,
        _standalone_train_optimized_period_model,
        _standalone_create_period_model,
        _standalone_optimize_axis_model,
        _standalone_train_optimized_axis_model,
        _standalone_create_axis_model
    )
    # These are fine as they are from the main training package
    from lc_pipeline.training import run_axis_optimization, update_config_with_best_params, train_period_model
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu")
    logger.info(f"Using device: {device}")
    
    # Set random seed
    torch.manual_seed(config.seed)
    
    # Load data using the correctly named function from run_hyperopt
    logger.info("Loading data for hyperparameter optimization via run_hyperopt module")
    dataset, train_loader, val_loader, test_loader = _standalone_load_data_for_hyperopt(config, logger) 
    
    period_model = None
    
    # Run period model hyperparameter optimization if requested
    if config.run_period_training and args_for_hyperopt.hyperopt_model in ["period", "both"]:
        logger.info("Running period model hyperparameter optimization")
        # Use the correctly named function from run_hyperopt
        config, best_period_params = _standalone_optimize_period_model(config, train_loader, val_loader, logger)
        
        # Train with best parameters if requested
        if args_for_hyperopt.train_after_hyperopt:
            logger.info("Training period model with optimized hyperparameters")
            # Use the correctly named function from run_hyperopt
            period_model, period_checkpoint = _standalone_train_optimized_period_model(config, train_loader, val_loader, logger)
        else:
            # Create period model with best params for axis optimization
            logger.info("Creating period model with best hyperparameters")
            # Use the correctly named function from run_hyperopt
            period_model = _standalone_create_period_model(config, device, logger)
            
            # Quick training for axis optimization
            if config.run_axis_training:
                logger.info("Quick training of period model for axis optimization")
                # This train_period_model is from lc_pipeline.training, which is fine.
                train_period_model(
                    model=period_model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    config=config.period_model,
                    logger=logger,
                    debug=args_for_hyperopt.debug,
                    checkpoint_path=None
                )
    
    # Run axis model hyperparameter optimization if requested
    if config.run_axis_training and args_for_hyperopt.hyperopt_model in ["axis", "both"]:
        if period_model is None and config.run_period_training:
            logger.info("Period model not trained or created; creating and training a default one for axis optimization.")
            # Use the correctly named function from run_hyperopt
            period_model = _standalone_create_period_model(config, device, logger)
            # This train_period_model is from lc_pipeline.training, which is fine.
            train_period_model(
                model=period_model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                config=config.period_model,
                logger=logger,
                debug=args_for_hyperopt.debug,
                checkpoint_path=None
            )
        elif period_model is None and not config.run_period_training:
             logger.error("Axis hyperopt requires a period model, but period training was skipped and no model was loaded/created.")
             return config # or raise an error

        # Run axis optimization
        logger.info("Running axis model hyperparameter optimization")
        # Instead of using run_axis_optimization directly, use the standalone wrapper
        # if it performs additional steps like saving config or specific logging.
        # The run_hyperopt.py script now has _standalone_optimize_axis_model which wraps run_axis_optimization.
        config, best_axis_params = _standalone_optimize_axis_model(
            config=config,
            train_loader_axis=train_loader, 
            val_loader_axis=val_loader,    
            period_model=period_model,
            logger=logger
        )
        
        # Update config with best parameters (this is now done inside _standalone_optimize_axis_model)
        # if best_axis_params: 
        #     logger.info(f"Best axis model parameters: {best_axis_params}")
        #     config = update_config_with_best_params(config, best_axis_params, "axis")
        # 
        #     # Save updated config
        #     config_save_path = os.path.join(config.paths.models_dir, "best_axis_config.yaml")
        #     save_config(config, config_save_path) 
        #     logger.info(f"Saved best axis model config to {config_save_path}")
        # else:
        #     logger.warning("Axis hyperparameter optimization did not return best parameters.")

        # Train with best parameters if requested
        if args_for_hyperopt.train_after_hyperopt:
            logger.info("Training axis model with optimized hyperparameters after hyperopt mode")
            # If we optimized axis, we should now train axis model with its best params.
            # This assumes that if hyperopt_model was "both", the period model is already handled,
            # and config has best period_params. Now config has best axis_params too.
            # We need a standalone axis training function if the full pipeline run_main_pipeline
            # is too much or doesn't fit the flow here.
            # For now, using _standalone_train_optimized_axis_model seems correct.
            
            if period_model is None:
                logger.error("Cannot train optimized axis model: period_model is None. This should have been created or trained earlier.")
                # Potentially create a default one here if that's desired, or raise error
            else:
                _standalone_train_optimized_axis_model(
                    config, # This config should have best_axis_params applied by _standalone_optimize_axis_model
                    train_loader, # This is train_loader_period (original AsteroidDataset loader)
                    val_loader,   # This is val_loader_period (original AsteroidDataset loader)
                    period_model, # Pass the period_model instance
                    logger        # Pass the logger instance
                )
            # The original call to run_main_pipeline(config, debug=args_for_hyperopt.debug) 
            # might be for when hyperopt is done for BOTH and then you run the whole thing.
            # If hyperopt was only for 'axis', then only axis should be trained here.
            # The logic in run_hyperopt.py's main() is more granular and might be a better model.
            # For now, this is a targeted training of axis model.
    
    # Save final config
    final_config_path = os.path.join(config.paths.results_dir, "final_config_hyperopt.yaml") 
    save_config(config, final_config_path) 
    logger.info(f"Saved final hyperopt config to {final_config_path}")
    
    return config


if __name__ == "__main__":
    # --- Early basic logging setup for diagnostics ---
    temp_log_file = Path("early_pipeline_debug.log")
    logging.basicConfig(level=logging.DEBUG, 
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(temp_log_file, mode='w'),
                            logging.StreamHandler(sys.stdout) # Also try to force console output
                        ])
    early_logger = logging.getLogger("EARLY_INIT")
    early_logger.info(f"--- Early diagnostic logging started. Output to console and {temp_log_file.resolve()} ---")
    # --- End of early basic logging setup ---

    # Parse command-line arguments
    early_logger.debug("Parsing command-line arguments...")
    args = parse_args()
    early_logger.debug(f"Arguments parsed: {args}")
    
    # Load configuration
    early_logger.debug(f"Loading base configuration from: {args.config}")
    pipeline_config_obj = load_config(args.config)
    early_logger.debug(f"Base configuration loaded. Type: {type(pipeline_config_obj)}")
    
    # Update configuration with command-line arguments
    early_logger.debug("Updating configuration from command-line arguments...")
    pipeline_config_obj = update_config_from_args(pipeline_config_obj, args)
    early_logger.debug("Configuration updated from arguments.")

    # Create directories
    try:
        early_logger.debug(f"Ensuring logs directory exists: {pipeline_config_obj.paths.logs_dir}")
        os.makedirs(pipeline_config_obj.paths.logs_dir, exist_ok=True)
        early_logger.debug(f"Ensuring models directory exists: {pipeline_config_obj.paths.models_dir}")
        os.makedirs(pipeline_config_obj.paths.models_dir, exist_ok=True)
        early_logger.debug(f"Ensuring results directory exists: {pipeline_config_obj.paths.results_dir}")
        os.makedirs(pipeline_config_obj.paths.results_dir, exist_ok=True)
        early_logger.debug(f"Ensuring figures directory exists: {pipeline_config_obj.paths.figures_dir}")
        os.makedirs(pipeline_config_obj.paths.figures_dir, exist_ok=True)
        early_logger.debug("All required directories checked/created.")
    except Exception as e_dir:
        early_logger.error(f"CRITICAL: Failed to create essential directories: {e_dir}", exc_info=True)
        sys.exit(1) # Exit if we can't even make directories
    
    # Set up logging using the Logger class
    early_logger.debug("Initializing main Logger class...")
    logger_config_dict = pipeline_config_obj.logging.__dict__
    main_config_for_logger = pipeline_config_obj.__dict__

    logger = Logger(
        log_dir=pipeline_config_obj.paths.logs_dir,
        experiment_name=f"lc_pipeline_runner_{pipeline_config_obj.period_model.model_name}_{args.mode}",
        log_level=pipeline_config_obj.logging.log_level,
        log_to_console=pipeline_config_obj.logging.log_to_console,
        log_to_file=pipeline_config_obj.logging.log_to_file,
        config=main_config_for_logger
    )
    
    # Save initial config
    initial_config_path = os.path.join(pipeline_config_obj.paths.results_dir, "initial_config.yaml")
    save_config(pipeline_config_obj, initial_config_path)
    logger.info(f"Saved initial config to {initial_config_path}")
    
    try:
        # Record start time
        start_time = time.time()
        
        # Run the appropriate mode
        if args.mode == "hyperopt":
            logger.info("Running hyperparameter optimization mode")
            pipeline_config_obj = run_hyperopt_mode(pipeline_config_obj, logger, args)
        else:
            logger.info(f"Running pipeline in {args.mode} mode")
            run_main_pipeline(pipeline_config_obj, debug=args.debug)
        
        # Record end time and log duration
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"Pipeline completed in {duration:.2f} seconds ({duration/60:.2f} minutes)")
        
        # Log the actual resource usage
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            logger.info(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")
            
            cpu_percent = process.cpu_percent(interval=1.0)
            if cpu_percent is not None:
                 logger.info(f"CPU usage: {cpu_percent:.2f}%")
            
            if torch.cuda.is_available():
                gpu_memory_allocated = torch.cuda.memory_allocated() / 1024 / 1024
                gpu_memory_reserved = torch.cuda.memory_reserved() / 1024 / 1024
                logger.info(f"GPU memory: {gpu_memory_allocated:.2f}MB allocated / {gpu_memory_reserved:.2f}MB reserved")
        except ImportError:
            logger.info("Psutil not available, skipping resource usage logging")
        except Exception as e_res:
            logger.warning(f"Could not log resource usage: {e_res}")

        logger.info("Pipeline execution completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if 'logger' in locals() and logger is not None:
            logger.close() 