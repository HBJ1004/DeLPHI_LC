#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training utilities for the light curve analysis pipeline.
"""

import os
import numpy as np
import torch
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union, List, Callable


class EarlyStopping:
    """
    Early stopping mechanism to prevent overfitting.
    
    Monitors a validation metric and stops training when the metric
    stops improving for a specified number of epochs.
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
        verbose: bool = False,
        save_path: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs with no improvement after which
                      training will be stopped.
            min_delta: Minimum change in the monitored quantity to qualify
                      as an improvement.
            mode: 'min' for metrics that are better when lower (e.g., loss),
                 'max' for metrics that are better when higher (e.g., accuracy).
            verbose: Whether to print messages about early stopping.
            save_path: Path to save the best model. If None, model is not saved.
            logger: Logger for log messages. If None, print is used.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        # Store save_path as string to avoid Path issues with Drive in Colab
        self.save_path = str(save_path) if save_path else None
        self.logger = logger
        
        # Initialize counter and best score
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        # Create save directory if needed
        if self.save_path:
            # Use Path only for directory creation, then convert back to string
            save_dir = Path(self.save_path).parent
            save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set comparison function based on mode
        if self.mode == "min":
            self._is_better = lambda score, best: score < best - self.min_delta
        elif self.mode == "max":
            self._is_better = lambda score, best: score > best + self.min_delta
        else:
            raise ValueError(f"Mode {mode} is not supported. Use 'min' or 'max'.")
    
    def __call__(self, score: float, model: torch.nn.Module) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation metric value
            model: Current model state
            
        Returns:
            bool: True if training should stop, False otherwise
        """
        # Initialize best score on first call
        if self.best_score is None:
            self.best_score = score
            self._save_model(model, score)
            return False
        
        # Check if current score is better
        if self._is_better(score, self.best_score):
            # Reset counter and update best score
            self.best_score = score
            self.counter = 0
            self._save_model(model, score)
        else:
            # Increment counter
            self.counter += 1
            if self.verbose:
                self._log(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            
            # Check if patience exceeded
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    self._log(f"Early stopping triggered after {self.counter} epochs without improvement")
        
        return self.early_stop
    
    def _save_model(self, model: torch.nn.Module, score: float) -> None:
        """Save model state dictionary to disk."""
        if self.save_path:
            try:
                # ADDED DEBUG LOG
                if self.logger:
                    self.logger.debug(f"[EarlyStopping._save_model] Attempting to save model. Save path: '{self.save_path}', Type: {type(self.save_path)}")
                else: # Fallback if logger not present but verbose is
                    print(f"[EarlyStopping._save_model DEBUG] Attempting to save model. Save path: '{self.save_path}', Type: {type(self.save_path)}")

                # Ensure save_path is a string for torch.save
                save_path_str = str(self.save_path)
                
                # Save only the model state dict
                torch.save(model.state_dict(), save_path_str)
                
                if self.verbose:
                    score_str = f"{score:.6f}"
                    self._log(f"Model state dict saved with {self.mode} score: {score_str} to {save_path_str}")
            except Exception as e:
                self._log(f"Failed to save model state dict: {str(e)}", level="error")
                # Add more detailed error info for debugging
                import traceback
                self._log(f"Error details: {traceback.format_exc()}", level="error")
    
    def load_best_model(self, model: torch.nn.Module) -> Optional[float]:
        """
        Load the best model state dictionary from disk.
        
        Args:
            model: Model instance to load weights into
            
        Returns:
            float: Best score tracked by the instance or None if loading failed
        """
        if not self.save_path or not os.path.exists(str(self.save_path)):
            self._log("No saved model checkpoint found at specified path", level="warning")
            return None
        
        try:
            # Load the state dictionary (should work with weights_only=True default)
            # Explicitly load to CPU to ensure portability, model can be moved to device later.
            # Ensure path is a string
            path_to_load = str(self.save_path)
            if self.logger:
                self.logger.debug(f"[EarlyStopping] Attempting to load checkpoint from: '{path_to_load}'")
            
            state_dict = torch.load(path_to_load, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
            
            if self.verbose:
                # Use the internally tracked best_score
                score_str = f"{self.best_score:.6f}" if self.best_score is not None else "N/A"
                self._log(f"Loaded best model state dict from {path_to_load}. Best score tracked: {score_str}")
            
            # Return the best score tracked by this instance
            return self.best_score
        except Exception as e:
            self._log(f"Failed to load model state dict from {self.save_path}: {str(e)}", level="error")
            # Add specific check for the weights_only issue for better logging
            if "weights_only load failed" in str(e):
                 self._log("This might be due to the file containing non-tensor data. Ensure only state_dict is saved.", level="error")
            # Add more detailed error info for debugging
            import traceback
            self._log(f"Error details: {traceback.format_exc()}", level="error")
            return None
    
    def _log(self, message: str, level: str = "info") -> None:
        """Log a message."""
        if self.logger:
            getattr(self.logger, level)(message)
        elif self.verbose:
            print(message)


class MetricTracker:
    """
    Track metrics during training.
    
    Keeps a history of metrics for plotting and analysis.
    """
    
    def __init__(self, metrics: List[str]):
        """
        Initialize tracker.
        
        Args:
            metrics: List of metric names to track
        """
        self.metrics = metrics
        self.history = {metric: [] for metric in metrics}
        self.best_values = {metric: None for metric in metrics}
        self.best_epochs = {metric: None for metric in metrics}
    
    def update(self, epoch: int, **kwargs) -> None:
        """
        Update metrics for current epoch.
        
        Args:
            epoch: Current epoch number
            **kwargs: Metric values (key=metric name, value=metric value)
        """
        for metric, value in kwargs.items():
            if metric in self.metrics:
                # Add to history
                self.history[metric].append(value)
                
                # Update best value if better or first epoch
                if self.best_values[metric] is None or value < self.best_values[metric]:
                    self.best_values[metric] = value
                    self.best_epochs[metric] = epoch
    
    def get_best(self, metric: str) -> tuple:
        """
        Get best value and epoch for a metric.
        
        Args:
            metric: Metric name
            
        Returns:
            tuple: (best_value, best_epoch)
        """
        if metric not in self.metrics:
            raise ValueError(f"Metric {metric} not tracked")
        
        return self.best_values[metric], self.best_epochs[metric]
    
    def get_history(self, metric: str) -> List[float]:
        """
        Get history of a metric.
        
        Args:
            metric: Metric name
            
        Returns:
            List[float]: History of metric values
        """
        if metric not in self.metrics:
            raise ValueError(f"Metric {metric} not tracked")
        
        return self.history[metric] 