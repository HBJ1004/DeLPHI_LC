#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration module for the light curve analysis pipeline.
Uses OmegaConf for structured configuration management.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from omegaconf import OmegaConf

@dataclass
class DataConfig:
    # Data sources
    use_synthetic_data: bool = False
    use_damit_data: bool = True
    max_damit_files: Optional[int] = 2500  # Limit number of DAMIT files used
    synthetic_data_dir: str = "lc_sample/"
    damit_data_dir: str = "DAMIT_csv/"
    force_rebuild_cache: bool = False  # Force rebuild of asteroid dataset cache
    
    # Dataset parameters
    max_sequence_length: int = 200
    train_val_ratio: float = 0.8
    val_ratio: float = 0.2
    
    # Phase-folding configuration
    num_axis_bins: int = 100
    smooth_axis_data: bool = True
    
    # Augmentation
    use_augmentation: bool = False
    augmentation_factors: List[float] = field(default_factory=lambda: [0.8, 1.2])
    noise_level: float = 0.01

@dataclass
class PeriodModelConfig:
    # Model selection
    model_name: str = "PeriodLSTMWithLSPrior"  # Options: PeriodLSTMNet, PeriodTCNNet, PeriodTransformerNet, etc.
    loss_function: str = "ClipPeriodLoss"  # Options: "ClipPeriodLoss", "LogSpacePeriodMAELoss"
    input_dim: int = 17
    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.2
    
    # LS Prior settings
    use_ls_prior: bool = True
    
    # Log-scale settings
    use_log_scale: bool = False
    min_period: float = 2.0
    max_period: float = 100.0
    
    # Period scale factor
    period_scale_factor: Optional[float] = None  # Set to None for auto-calculation
    
    # Training parameters
    batch_size: int = 64
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10
    
    # Hyperparameter search
    optuna_trials: int = 20
    optuna_epochs: int = 10
    # Hyperparameter search space bounds
    hidden_dim_range: List[int] = field(default_factory=lambda: [32, 256])
    num_layers_range: List[int] = field(default_factory=lambda: [1, 5])
    dropout_range: List[float] = field(default_factory=lambda: [0.0, 0.5])
    lr_range: List[float] = field(default_factory=lambda: [1e-5, 1e-2])
    weight_decay_range: List[float] = field(default_factory=lambda: [1e-6, 1e-3])
    mc_dropout_samples: int = 0 # Number of MC Dropout samples for uncertainty estimation (0 to disable)

@dataclass
class AxisModelConfig:
    # Model selection
    model_name: str = "AxisCNNNet"  # Options: AxisCNNNet, PhaseAwareTransformerAxis, etc.
    
    # --- Parameters for AxisCNNNet ---
    blocks: int = 3
    initial_channels: int = 32
    kernel: int = 5
    # hidden_dim is shared
    # dropout is shared
    # use_norm is shared

    # --- Parameters for PhaseAwareTransformerAxis ---
    folded_input_features: int = 1
    # num_bins is in DataConfig (config.data.num_axis_bins)
    raw_input_features: int = 2 # Example: flux, time_delta_obs
    max_raw_seq_len: int = 500 # Max length for raw time series features
    num_heads: int = 4
    num_encoder_layers: int = 3
    fusion_strategy: str = "concat" # Options: "concat", "cross_attention"
    pat_use_cyclic_pe_folded: bool = False # Use Cyclic Positional Encoding for folded LC in PAT

    # --- Shared Model Parameters ---
    hidden_dim: int = 128
    dropout: float = 0.2
    use_norm: bool = True # Primarily for CNN, but can be a general flag
    
    # Training parameters
    batch_size: int = 64
    epochs: int = 60
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10
    
    # Curriculum learning parameters
    use_curriculum: bool = False
    # Defines a teacher forcing schedule: list of [epoch_end, mix_ratio] pairs.
    # e.g., [[10, 1.0], [20, 0.5], [60, 0.0]] means:
    # - Epochs 0-9: mix_ratio = 1.0 (all true periods)
    # - Epochs 10-19: mix_ratio = 0.5
    # - Epochs 20-59: mix_ratio = 0.0 (all predicted periods)
    # This is used if teacher_forcing_decay_type below is None.
    teacher_forcing_schedule: Optional[List[List[Union[int, float]]]] = None
    
    # Fallback for staged curriculum if teacher_forcing_schedule and teacher_forcing_decay_type are None.
    # List of 3 integers: [stage1_end_epoch, stage2_end_epoch, total_curriculum_epochs]
    # Example: [20, 40, 60] means stage 1 (100% true) up to epoch 19,
    # stage 2 (e.g., 50% true) up to epoch 39, stage 3 (0% true) up to epoch 59.
    curriculum_epochs: Optional[List[int]] = field(default_factory=lambda: [20, 40, 60])

    # Continuous decay for teacher forcing ratio. Overrides teacher_forcing_schedule and curriculum_epochs if set.
    teacher_forcing_decay_type: Optional[str] = None  # Options: "linear", "exponential", None
    teacher_forcing_initial_ratio: float = 1.0
    teacher_forcing_final_ratio: float = 0.0
    teacher_forcing_decay_epochs: Optional[int] = None # Num epochs for decay from initial to final ratio
    
    # Special flags
    use_true_periods: bool = False  # Use ground truth period for training
    use_quaternions: bool = True    # Use quaternion representation for rotation
    
    # Hyperparameter search
    optuna_trials: int = 20
    optuna_epochs: int = 10
    # Hyperparameter search space bounds
    hidden_dim_range: List[int] = field(default_factory=lambda: [32, 256])
    blocks_range: List[int] = field(default_factory=lambda: [2, 6])
    initial_channels_range: List[int] = field(default_factory=lambda: [16, 64])
    kernel_range: List[int] = field(default_factory=lambda: [3, 7])
    dropout_range: List[float] = field(default_factory=lambda: [0.0, 0.5])
    lr_range: List[float] = field(default_factory=lambda: [1e-5, 1e-2])
    weight_decay_range: List[float] = field(default_factory=lambda: [1e-6, 1e-3])
    mc_dropout_samples: int = 0 # Number of MC Dropout samples for uncertainty estimation (0 to disable)

@dataclass
class PathConfig:
    # Directory paths
    base_dir: str = "."
    models_dir: str = "models"
    results_dir: str = "results"
    figures_dir: str = "figures"
    logs_dir: str = "logs"
    cache_dir: str = "data_cache"  # Base directory for caching datasets

@dataclass
class CacheConfig:
    # Dataset caching configuration
    use_disk_caching: bool = True  # Enable disk-based caching
    force_recache: bool = False  # Force regeneration of cache
    asteroid_dataset_cache_dir: str = "asteroid_dataset"  # Relative to cache_dir
    axis_dataset_cache_dir: str = "axis_dataset"  # Relative to cache_dir

@dataclass
class LoggingConfig:
    # Logging configuration
    log_level: str = "INFO"
    log_to_file: bool = True
    log_to_console: bool = True
    
    # Email notifications
    enable_email: bool = False
    email_recipient: str = ""
    
    # Visualization
    save_plots: bool = True
    interactive_plots: bool = False

@dataclass
class HyperOptPeriodModelConfig:
    optuna_trials: int = 20
    optuna_epochs: int = 10
    hidden_dim_range: List[int] = field(default_factory=lambda: [64, 256])
    num_layers_range: List[int] = field(default_factory=lambda: [2, 5])
    dropout_range: List[float] = field(default_factory=lambda: [0.1, 0.5])
    lr_range: List[float] = field(default_factory=lambda: [0.0001, 0.01]) # log scale
    weight_decay_range: List[float] = field(default_factory=lambda: [1e-5, 1e-3]) # log scale
    prior_weight_range: Optional[List[float]] = field(default_factory=lambda: [0.1, 0.9]) # For PeriodLSTMWithLSPrior

@dataclass
class HyperOptAxisModelConfig:
    optuna_trials: int = 15
    optuna_epochs: int = 15
    hidden_dim_range: List[int] = field(default_factory=lambda: [64, 128])
    blocks_range: List[int] = field(default_factory=lambda: [2, 5])
    initial_channels_range: List[int] = field(default_factory=lambda: [16, 64])
    kernel_range: List[int] = field(default_factory=lambda: [3, 7]) # suggest_int with step=2 if bounds are odd and min < max
    dropout_range: List[float] = field(default_factory=lambda: [0.1, 0.5])
    lr_range: List[float] = field(default_factory=lambda: [0.0001, 0.01]) # log scale
    weight_decay_range: List[float] = field(default_factory=lambda: [1e-5, 1e-3]) # log scale
    # Add other specific ranges for axis model if needed, e.g., num_layers_range, num_heads_range for transformers

@dataclass
class HyperOptConfig:
    run_period_hyperopt: bool = True
    run_axis_hyperopt: bool = True
    train_after_hyperopt: bool = True
    # Nested configurations for period and axis model hyperparameter ranges
    period_model: HyperOptPeriodModelConfig = field(default_factory=HyperOptPeriodModelConfig)
    axis_model: HyperOptAxisModelConfig = field(default_factory=HyperOptAxisModelConfig)

@dataclass
class PipelineConfig:
    # General pipeline configuration
    seed: int = 42
    device: str = "cuda"  # Will be automatically set to CPU if CUDA is not available
    test_mode: bool = False  # Flag for running in test mode with reduced dataset and iterations
    
    # Workflow steps to run
    run_period_training: bool = True
    run_axis_training: bool = True
    run_hyperopt: bool = True
    run_fine_tuning: bool = True
    run_evaluation: bool = True
    
    # Component configs
    data: DataConfig = field(default_factory=DataConfig)
    period_model: PeriodModelConfig = field(default_factory=PeriodModelConfig)
    axis_model: AxisModelConfig = field(default_factory=AxisModelConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    hyperopt: HyperOptConfig = field(default_factory=HyperOptConfig)
    
    # Convenience properties
    force_recache: bool = False  # Global override for all caching
    
    # Override model selection based on Period model name
    def __post_init__(self):
        # Automatically set flags based on the selected period model name
        ls_prior_models = [
            'PeriodLSTMWithLSPrior', 
            'PhaseAwareTransformerWithLSPrior', 
            'EnhancedPhaseAwareTransformer'
        ]
        log_scale_models = [
            'PeriodLSTMLogScale', 
            'PhaseAwareTransformerLogScale', 
            'EnhancedPhaseAwareTransformer'
        ]
        
        # Update period model config based on model name
        self.period_model.use_ls_prior = self.period_model.model_name in ls_prior_models
        self.period_model.use_log_scale = self.period_model.model_name in log_scale_models


def load_config(config_path: Optional[str] = None) -> PipelineConfig:
    """
    Load configuration from a YAML file or create default configuration.
    
    Args:
        config_path: Path to a YAML configuration file (optional)
    
    Returns:
        PipelineConfig object with configuration
    """
    # Create default config
    default_conf = OmegaConf.structured(PipelineConfig())
    
    # If config path is provided, load and merge with defaults
    if config_path and os.path.exists(config_path):
        user_conf = OmegaConf.load(config_path)
        conf = OmegaConf.merge(default_conf, user_conf)
    else:
        conf = default_conf
    
    # Parse OmegaConf to PipelineConfig
    config = OmegaConf.to_object(conf)
    
    # Auto-set device if CUDA is not available
    import torch
    if not torch.cuda.is_available() and config.device == "cuda":
        config.device = "cpu"
        print("CUDA not available, using CPU instead.")
    
    return config


def save_config(config: PipelineConfig, file_path: str) -> None:
    """
    Save configuration to a YAML file.
    
    Args:
        config: PipelineConfig object
        file_path: Path to save the configuration file
    """
    conf = OmegaConf.structured(config)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Save to file
    with open(file_path, 'w') as f:
        OmegaConf.save(conf, f)


def validate_config(config, logger=None):
    """Validate configuration and set defaults for missing values."""
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)
    required_attrs = {
        'period_model': ['model_name', 'use_log_scale'],
        'axis_model': ['model_name', 'use_quaternions'],
        'data': ['max_sequence_length', 'num_axis_bins']
    }
    for section, attrs in required_attrs.items():
        if not hasattr(config, section):
            logger.error(f"Missing config section: {section}")
            raise ValueError(f"Missing config section: {section}")
        for attr in attrs:
            if not hasattr(getattr(config, section), attr):
                logger.warning(f"Missing {section}.{attr} in config. Using default if available.")


if __name__ == "__main__":
    # Example: Print the default configuration
    config = load_config()
    print(OmegaConf.to_yaml(OmegaConf.structured(config))) 