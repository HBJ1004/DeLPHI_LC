#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Environment utilities for light curve analysis pipeline.
Handles detection and setup for various execution environments.
"""

import os
import sys
import platform
import logging
import torch
import numpy as np
import random
from typing import Dict, Any, Optional, Tuple, Union


def is_colab() -> bool:
    """Check if code is running in Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return False


def is_kaggle() -> bool:
    """Check if code is running in Kaggle."""
    return 'KAGGLE_KERNEL_RUN_TYPE' in os.environ


def is_notebook() -> bool:
    """Check if code is running in a Jupyter notebook."""
    try:
        from IPython import get_ipython
        if get_ipython() is None:
            return False
        if 'IPKernelApp' not in get_ipython().config:
            return False
        return True
    except ImportError:
        return False


def detect_tpu() -> Optional[object]:
    """Detect and return TPU if available."""
    if 'COLAB_TPU_ADDR' in os.environ:
        try:
            import torch_xla.core.xla_model as xm
            return xm.xla_device()
        except ImportError:
            print("TPU environment detected but PyTorch XLA not installed. Run:")
            print("!pip install torch_xla")
            return None
    return None


def setup_environment(config: Dict[str, Any], logger: logging.Logger) -> Dict[str, Any]:
    """
    Set up the environment for training/testing based on config.
    Automatically detects and configures for Colab/TPU/GPU environments.
    
    Args:
        config: Configuration dictionary
        logger: Logger instance
        
    Returns:
        dict: Environment information with device selection
    """
    # Log environment info
    env_info = {
        "platform": platform.platform(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "is_colab": is_colab(),
        "is_kaggle": is_kaggle(),
        "is_notebook": is_notebook(),
    }
    
    for key, value in env_info.items():
        logger.info(f"{key}: {value}")
    
    # Set random seeds for reproducibility
    seed = config["general"]["random_seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Determine device automatically
    device = "cpu"  # Default
    
    # Try to detect and use TPU first
    tpu_device = detect_tpu()
    if tpu_device is not None:
        device = tpu_device
        logger.info("TPU detected and configured")
    # If no TPU, try to use GPU
    elif config["general"]["use_cuda"] and torch.cuda.is_available():
        cuda_id = config['general']['cuda_device']
        device = f"cuda:{cuda_id}"
        
        # Log GPU information
        logger.info(f"GPU: {torch.cuda.get_device_name(cuda_id)}")
        logger.info(f"CUDA: {torch.version.cuda}")
        logger.info(f"CUDNN: {torch.backends.cudnn.version()}")
        
        # Enable TF32 on Ampere GPUs for faster performance
        if torch.cuda.get_device_capability(cuda_id)[0] >= 8:  # Ampere or newer
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
            logger.info("TF32 enabled on Ampere+ GPU for improved performance")
            
        # Set CUDA optimizations
        torch.backends.cudnn.benchmark = config["general"].get("cudnn_benchmark", True)
        torch.backends.cudnn.deterministic = config["general"].get("cudnn_deterministic", False)
        
        if is_colab():
            # In Colab, we want to make sure we're utilizing the GPU effectively
            logger.info("Colab GPU detected, enabling performance optimizations")
            torch.backends.cudnn.benchmark = True
    else:
        logger.info("Using CPU for computation")
    
    # Set up torch.multiprocessing for data loading
    if "num_workers" in config["general"]:
        # Auto-detect number of workers if set to -1
        if config["general"]["num_workers"] == -1:
            import multiprocessing
            config["general"]["num_workers"] = multiprocessing.cpu_count()
        
        torch.set_num_threads(config["general"]["num_workers"])
        logger.info(f"Set PyTorch to use {config['general']['num_workers']} threads")
    
    # Enable anomaly detection in debug mode
    if config["general"]["debug"]:
        torch.autograd.set_detect_anomaly(True)
        logger.info("PyTorch anomaly detection enabled")
    
    # Return environment info with selected device
    return {"device": device, "env_info": env_info}


def get_memory_usage(device: Union[str, torch.device]) -> Dict[str, float]:
    """
    Get memory usage statistics for the given device.
    
    Args:
        device: PyTorch device
        
    Returns:
        dict: Memory usage statistics in MB
    """
    memory_stats = {"available": 0, "used": 0, "total": 0}
    
    if isinstance(device, str) and device.startswith("cuda"):
        # GPU memory stats
        try:
            if torch.cuda.is_available():
                cuda_id = int(device.split(":")[-1]) if ":" in device else 0
                memory_stats["total"] = torch.cuda.get_device_properties(cuda_id).total_memory / 1024**2
                memory_stats["used"] = torch.cuda.memory_allocated(cuda_id) / 1024**2
                memory_stats["reserved"] = torch.cuda.memory_reserved(cuda_id) / 1024**2
                memory_stats["available"] = memory_stats["total"] - memory_stats["reserved"]
        except Exception as e:
            print(f"Error getting CUDA memory stats: {e}")
    elif str(device) == "xla":
        # TPU memory stats not directly accessible through PyTorch
        memory_stats["note"] = "TPU memory statistics not available through PyTorch API"
    else:
        # CPU memory stats
        try:
            import psutil
            vm = psutil.virtual_memory()
            memory_stats["total"] = vm.total / 1024**2
            memory_stats["available"] = vm.available / 1024**2
            memory_stats["used"] = vm.used / 1024**2
            memory_stats["percent"] = vm.percent
        except ImportError:
            memory_stats["note"] = "Install psutil for CPU memory statistics"
    
    return memory_stats 