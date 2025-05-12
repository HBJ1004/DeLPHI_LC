#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Logging utilities for the light curve analysis pipeline.
"""

import os
import logging
from pathlib import Path
import sys
import time
import json
from datetime import datetime
from typing import Optional, Dict, Any, Union, List
import inspect

# Optional imports for visualization
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Optional imports for TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

# Optional imports for Weights & Biases
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# Helper function to convert config objects to plain dictionaries
def _config_to_dict(obj, depth=0, max_depth=10):
    """
    Recursively convert configuration objects to serializable dictionaries.
    Handles dataclasses, OmegaConf objects, and other complex types.
    
    Args:
        obj: The object to convert
        depth: Current recursion depth
        max_depth: Maximum recursion depth to prevent infinite recursion
        
    Returns:
        A serializable dictionary, list, or primitive type
    """
    # Prevent infinite recursion
    if depth > max_depth:
        return "Recursion limit exceeded"
    
    if obj is None:
        return None
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, (list, tuple)):
        return [_config_to_dict(item, depth + 1, max_depth) for item in obj]
    elif isinstance(obj, dict):
        return {str(k): _config_to_dict(v, depth + 1, max_depth) for k, v in obj.items()}
    elif hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
        # Handle objects with custom to_dict method
        try:
            return _config_to_dict(obj.to_dict(), depth + 1, max_depth)
        except Exception as e:
            return f"Error in to_dict: {str(e)}"
    elif hasattr(obj, "__dict__"):
        # Handle objects with __dict__ (dataclasses, custom classes)
        result = {}
        for k, v in obj.__dict__.items():
            if not k.startswith("_") and not callable(v):
                try:
                    result[k] = _config_to_dict(v, depth + 1, max_depth)
                except Exception as e:
                    result[k] = f"Error: {str(e)}"
        return result
    elif hasattr(obj, "__dataclass_fields__"):
        # Handle dataclasses directly
        result = {}
        for k in obj.__dataclass_fields__:
            try:
                result[k] = _config_to_dict(getattr(obj, k), depth + 1, max_depth)
            except Exception as e:
                result[k] = f"Error: {str(e)}"
        return result
    elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, bytearray)):
        # Handle other iterables
        try:
            return [_config_to_dict(item, depth + 1, max_depth) for item in obj]
        except Exception:
            pass  # Fall through to str conversion
            
    # For pytorch tensors, handle specially
    if hasattr(obj, "tolist"):
        try:
            if hasattr(obj, "shape") and len(obj.shape) <= 2:  # Only convert small tensors
                return obj.tolist()
            else:
                return f"Tensor with shape {obj.shape}"
        except Exception:
            pass
            
    # Try to convert to string as fallback
    try:
        return str(obj)
    except Exception as e:
        return f"Unserializable Object: {str(e)}"


class Logger:
    """
    Unified logging interface for the pipeline.
    
    Handles console logging, file logging, and experiment tracking
    with TensorBoard and Weights & Biases.
    """
    
    def __init__(
        self,
        log_dir: str,
        experiment_name: Optional[str] = None,
        log_level: str = "INFO",
        log_to_console: bool = True,
        log_to_file: bool = True,
        use_tensorboard: bool = False,
        use_wandb: bool = False,
        config: Optional[Dict[str, Any]] = None,
        email_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the logger.

        Args:
            log_dir: Directory for log files
            experiment_name: Name of the experiment
            log_level: Logging level
            log_to_console: Whether to log to console
            log_to_file: Whether to log to file
            use_tensorboard: Whether to use TensorBoard
            use_wandb: Whether to use Weights & Biases
            config: Configuration dictionary for experiment tracking
            email_config: Email notification configuration
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Set up experiment name
        self.experiment_name = experiment_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Set up Python logger
        self.logger = logging.getLogger(self.experiment_name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        self.logger.handlers = []  # Clear existing handlers

        # Log formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Console handler
        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # File handler
        if log_to_file:
            log_file = self.log_dir / f"{self.experiment_name}.log"
            # Explicitly set mode='w' to overwrite the log file on each run
            file_handler = logging.FileHandler(log_file, mode='w')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        # --- Process and Save Configuration ---
        processed_config = None
        config_file_path = None # Initialize path variable

        if config is not None:
            # Try to convert/save the config
            try:
                # Always run through _config_to_dict to handle nested objects
                processed_config = _config_to_dict(config)
                log_msg_prefix = "Configuration serialized and saved to"

                # Save the processed config dict
                config_file_path = self.log_dir / f"{self.experiment_name}_config.json"
                with open(config_file_path, 'w') as f:
                    json.dump(processed_config, f, indent=2)
                # Log only *after* successful save
                self.info(f"{log_msg_prefix} {config_file_path}")

            except Exception as e:
                self.warning(f"Failed to process/save configuration: {e}")
                # Create a basic config dict for W&B to prevent errors
                processed_config = {"error": "Configuration could not be processed", "experiment": self.experiment_name}
        else:
             # No config provided
             processed_config = {"status": "No config provided", "experiment": self.experiment_name}

        # --- Setup Tracking Systems ---

        # TensorBoard writer
        self.tensorboard_writer = None
        if use_tensorboard and TENSORBOARD_AVAILABLE:
            tb_dir = self.log_dir / "tensorboard" / self.experiment_name
            tb_dir.mkdir(parents=True, exist_ok=True)
            self.tensorboard_writer = SummaryWriter(log_dir=str(tb_dir))
        elif use_tensorboard and not TENSORBOARD_AVAILABLE:
            self.warning("TensorBoard requested but not available. Please install tensorboard.")

        # Weights & Biases
        self.wandb_run = None
        if use_wandb and WANDB_AVAILABLE:
            self.wandb_run = wandb.init(
                project="light_curve_analysis",
                name=self.experiment_name,
                config=processed_config, # Use the processed config
                dir=str(self.log_dir / "wandb"),
                reinit=True
            )
        elif use_wandb and not WANDB_AVAILABLE:
            self.warning("Weights & Biases requested but not available. Please install wandb.")

        # Email configuration
        self.email_config = email_config

        # Log initial message
        self.info(f"Logger initialized: {self.experiment_name}")
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str, exc_info: bool = False):
        """Log error message and flush all handlers."""
        self.logger.error(message, exc_info=exc_info)
        for handler in self.logger.handlers:
            handler.flush()
    
    def critical(self, message: str):
        """Log critical message."""
        self.logger.critical(message)
    
    def exception(self, message: str):
        """
        Log an exception with full traceback and flush all handlers immediately.
        This ensures the error is written to disk even if the program crashes.
        
        Args:
            message: Error message to log
        """
        # Use Python's built-in exception logging
        self.logger.exception(message)
        
        # Immediately flush to ensure it's written to disk
        for handler in self.logger.handlers:
            handler.flush()
        
        # If using other logging systems, flush them too
        if self.tensorboard_writer:
            self.tensorboard_writer.flush()
        
        # If debug mode is enabled, log additional system information
        if logging.getLogger().level <= logging.DEBUG:
            try:
                import psutil
                import torch
                import platform
                import sys
                
                memory_info = psutil.virtual_memory()
                self.logger.debug(f"System RAM: {memory_info.percent}% used, "
                                 f"{memory_info.available / (1024**3):.2f}GB available")
                
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        self.logger.debug(f"CUDA device {i}: "
                                          f"{torch.cuda.get_device_name(i)}, "
                                          f"{torch.cuda.memory_allocated(i) / (1024**3):.2f}GB allocated, "
                                          f"{torch.cuda.memory_reserved(i) / (1024**3):.2f}GB reserved")
                        
                self.logger.debug(f"Python version: {sys.version}")
                self.logger.debug(f"Platform: {platform.platform()}")
                
                # Get calling function info
                frame = inspect.currentframe().f_back
                if frame:
                    func_name = frame.f_code.co_name
                    filename = frame.f_code.co_filename
                    lineno = frame.f_lineno
                    self.logger.debug(f"Exception occurred in {filename}:{func_name}:{lineno}")
                    
                    # If a PyTorch tensor shape error, try to get tensor details
                    if "shape" in message and "tensor" in message.lower():
                        # Inspect the local variables
                        for name, var in frame.f_locals.items():
                            if isinstance(var, torch.Tensor):
                                self.logger.debug(f"Tensor '{name}' shape: {var.shape}, dtype: {var.dtype}, device: {var.device}")
                
            except Exception as e:
                self.logger.debug(f"Failed to gather system info: {str(e)}")
    
    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        prefix: str = ""
    ):
        """
        Log metrics to all enabled tracking systems.
        
        Args:
            metrics: Dictionary of metric names and values, can include nested dictionaries
            step: Training step or epoch
            prefix: Prefix for metric names
        """
        # Process metrics to handle nested dictionaries and add prefix
        processed_metrics = {}
        
        def process_dict(d, current_prefix=""):
            for k, v in d.items():
                new_key = f"{current_prefix}/{k}" if current_prefix else k
                if isinstance(v, dict):
                    # Recursively process nested dictionaries
                    process_dict(v, new_key)
                else:
                    # Add scalar metric
                    try:
                        # Convert to float if possible (TensorBoard requires scalars)
                        processed_metrics[new_key] = float(v)
                    except (TypeError, ValueError):
                        # Skip non-scalar values
                        self.debug(f"Skipping non-scalar metric: {new_key}")
        
        process_dict(metrics)
        
        # Add global prefix if provided
        if prefix:
            processed_metrics = {f"{prefix}/{k}": v for k, v in processed_metrics.items()}
        
        # Log to TensorBoard
        if self.tensorboard_writer:
            for name, value in processed_metrics.items():
                self.tensorboard_writer.add_scalar(name, value, step)
        
        # Log to Weights & Biases
        if self.wandb_run:
            self.wandb_run.log(processed_metrics, step=step)
        
        # Log to Python logger (for important metrics)
        if step is not None:
            metrics_str = ", ".join([f"{k}: {v:.6f}" for k, v in processed_metrics.items()])
            self.info(f"Step {step}: {metrics_str}")
        else:
            metrics_str = ", ".join([f"{k}: {v:.6f}" for k, v in processed_metrics.items()])
            self.info(f"Metrics: {metrics_str}")
    
    def log_figure(
        self,
        figure: Union[plt.Figure, Figure],
        name: str,
        step: Optional[int] = None
    ):
        """
        Log matplotlib figure to all enabled tracking systems.
        
        Args:
            figure: Matplotlib figure
            name: Figure name
            step: Training step or epoch
        """
        if not MATPLOTLIB_AVAILABLE:
            self.warning("Matplotlib not available. Figure not logged.")
            return
        
        # Save figure to file
        figure_dir = self.log_dir / "figures"
        figure_dir.mkdir(parents=True, exist_ok=True)
        
        if step is not None:
            figure_file = figure_dir / f"{name}_{step:06d}.png"
        else:
            figure_file = figure_dir / f"{name}.png"
        
        figure.savefig(figure_file, dpi=300, bbox_inches='tight')
        
        # Log to TensorBoard
        if self.tensorboard_writer:
            self.tensorboard_writer.add_figure(name, figure, step)
        
        # Log to Weights & Biases
        if self.wandb_run:
            self.wandb_run.log({name: wandb.Image(str(figure_file))}, step=step)
        
        self.debug(f"Figure saved: {figure_file}")
    
    def log_histogram(
        self,
        values,
        name: str,
        step: Optional[int] = None
    ):
        """
        Log histogram to all enabled tracking systems.
        
        Args:
            values: Values to plot in histogram
            name: Histogram name
            step: Training step or epoch
        """
        # Log to TensorBoard
        if self.tensorboard_writer:
            self.tensorboard_writer.add_histogram(name, values, step)
        
        # Log to Weights & Biases
        if self.wandb_run:
            self.wandb_run.log({name: wandb.Histogram(values)}, step=step)
    
    def log_model(
        self,
        model,
        model_name: str,
        metrics: Optional[Dict[str, float]] = None
    ):
        """
        Log model to Weights & Biases.
        
        Args:
            model: PyTorch model
            model_name: Model name
            metrics: Optional metrics to associate with the model
        """
        if self.wandb_run:
            # Save model
            model_dir = self.log_dir / "models"
            model_dir.mkdir(parents=True, exist_ok=True)
            model_file = model_dir / f"{model_name}.pt"
            
            import torch
            torch.save(model.state_dict(), model_file)
            
            # Log model to W&B
            artifact = wandb.Artifact(name=model_name, type="model")
            artifact.add_file(str(model_file))
            
            # Add metrics as metadata
            if metrics:
                for k, v in metrics.items():
                    artifact.metadata[k] = v
            
            self.wandb_run.log_artifact(artifact)
            self.info(f"Model saved and logged to W&B: {model_file}")
    
    def send_email_notification(
        self,
        subject: str,
        message: str,
        metrics: Optional[Dict[str, float]] = None
    ):
        """
        Send email notification.
        
        Args:
            subject: Email subject
            message: Email message
            metrics: Optional metrics to include in email
        """
        if not self.email_config:
            self.warning("Email configuration not provided. Skipping email notification.")
            return
        
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            # Create email
            email = MIMEMultipart()
            email["From"] = self.email_config.get("sender", "")
            email["To"] = self.email_config.get("recipient", "")
            email["Subject"] = f"[{self.experiment_name}] {subject}"
            
            # Add metrics to message if provided
            body = message
            if metrics:
                metrics_str = "\n".join([f"{k}: {v:.6f}" for k, v in metrics.items()])
                body += f"\n\nMetrics:\n{metrics_str}"
            
            # Add timestamp
            body += f"\n\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            # Add message to email
            email.attach(MIMEText(body, "plain"))
            
            # Connect to SMTP server and send email
            with smtplib.SMTP(
                self.email_config.get("smtp_server", ""),
                self.email_config.get("smtp_port", 587)
            ) as server:
                server.starttls()
                server.login(
                    self.email_config.get("username", ""),
                    self.email_config.get("password", "")
                )
                server.send_message(email)
            
            self.info(f"Email notification sent: {subject}")
            
        except Exception as e:
            self.error(f"Failed to send email notification: {e}")
    
    def close(self):
        """Close logger and all tracking systems."""
        # Close TensorBoard writer
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
        
        # Close Weights & Biases run
        if self.wandb_run:
            self.wandb_run.finish()
        
        self.info(f"Logger closed: {self.experiment_name}")


class Timer:
    """
    Simple timer for profiling code execution.
    
    Can be used as a context manager or standalone.
    """
    
    def __init__(self, name: str = "Timer", logger: Optional[Logger] = None):
        """
        Initialize timer.
        
        Args:
            name: Timer name
            logger: Optional logger for logging timing results
        """
        self.name = name
        self.logger = logger
        self.start_time = None
        self.end_time = None
        self.elapsed = None
    
    def __enter__(self):
        """Start timer when entering context."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timer when exiting context and log result."""
        self.stop()
        self.log()
    
    def start(self):
        """Start timer."""
        self.start_time = time.time()
        return self
    
    def stop(self):
        """Stop timer and return elapsed time."""
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time
        return self.elapsed
    
    def log(self):
        """Log timing result if logger is provided."""
        if self.logger and self.elapsed is not None:
            self.logger.info(f"{self.name}: {self.elapsed:.4f} seconds")
        return self.elapsed


def log_tensor_stats(tensor, name="tensor", logger=None):
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)
    try:
        if tensor is None:
            logger.debug(f"{name} is None")
            return
        logger.debug(f"{name} shape: {getattr(tensor, 'shape', None)}, dtype: {getattr(tensor, 'dtype', None)}, "
                     f"device: {getattr(tensor, 'device', None)}, "
                     f"min: {getattr(tensor, 'min', lambda: None)()}, max: {getattr(tensor, 'max', lambda: None)()}")
    except Exception as e:
        logger.debug(f"Could not log {name} stats: {e}")

def log_memory_usage(logger=None):
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)
    import torch
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        logger.info(f"GPU Memory: {allocated:.2f}MB allocated, {reserved:.2f}MB reserved")


if __name__ == "__main__":
    # Example usage
    logger = Logger(
        log_dir="logs",
        experiment_name="example_run",
        log_level="INFO",
        log_to_console=True,
        log_to_file=True,
        use_tensorboard=False,
        use_wandb=False,
        config={"learning_rate": 0.001, "batch_size": 64}
    )
    
    # Log messages
    logger.info("Starting experiment")
    logger.debug("Debug information")
    logger.warning("Warning message")
    
    # Log metrics
    metrics = {"loss": 0.123, "accuracy": 0.987}
    logger.log_metrics(metrics, step=1, prefix="train")
    
    # Close logger
    logger.close()
    
    # Use timer as context manager
    with Timer("Example operation", logger):
        # Simulate work
        time.sleep(1)
    
    # Use timer standalone
    timer = Timer("Standalone timer", logger)
    timer.start()
    # Simulate work
    time.sleep(0.5)
    timer.stop()
    timer.log() 