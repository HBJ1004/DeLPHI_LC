"""
Training module for light curve analysis pipeline.
"""

from .train_period import train_period_model, train_enhanced_period_model
from .train_axis import train_axis_model, evaluate_axis_model, train_axis_with_curriculum
from .hyperopt import (
    run_period_optimization,
    run_axis_optimization,
    update_config_with_best_params
)

__all__ = [
    "train_period_model",
    "train_enhanced_period_model",
    "train_axis_model",
    "evaluate_axis_model",
    "train_axis_with_curriculum",
    "run_period_optimization",
    "run_axis_optimization",
    "update_config_with_best_params"
] 