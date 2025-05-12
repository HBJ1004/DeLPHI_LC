#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup utilities for running lc_pipeline in Google Colab.

This module provides simplified functions for setting up the Colab environment
to run the asteroid lightcurve pipeline.
"""

import os
import sys
import logging
import importlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def find_project_dir():
    """
    Find project directory containing lc_pipeline package
    
    Returns:
        str or None: Path to the project directory if found, None otherwise
    """
    candidates = [
        "/content/drive/MyDrive/Colab Notebooks/asteroid_lightcurve_pipeline",
        "/content/drive/MyDrive/asteroid_lightcurve_pipeline",
        "/content/drive/MyDrive/Colab Notebooks/lc_pipeline",
        "/content/drive/MyDrive/lc_pipeline",
        "/content/drive/MyDrive/Colab Notebooks"
    ]
    
    for path in candidates:
        if os.path.exists(path) and os.path.exists(os.path.join(path, "lc_pipeline")):
            return path
    
    return None

def setup_colab():
    """
    Set up the Colab environment for running lc_pipeline.
    
    This handles all the necessary directory structure setup
    and environment checks to ensure the pipeline runs correctly in Colab.
    
    Returns:
        bool: True if setup was successful, False otherwise
    """
    try:
        # Check if we're in Colab
        try:
            import google.colab
            is_colab = True
            logger.info("Running in Google Colab environment")
        except ImportError:
            is_colab = False
            logger.warning("Not running in Google Colab. Setup may not be needed.")
            
        # Ensure project directory is in path
        project_dir = find_project_dir()
        if project_dir:
            if project_dir not in sys.path:
                sys.path.insert(0, project_dir)
                logger.info(f"Added {project_dir} to Python path")
        else:
            logger.warning("Could not find project directory")
            
        # Create required directories
        required_dirs = ["logs", "models", "results", "figures"]
        base_dir = project_dir if project_dir else "/content"
        
        for dir_name in required_dirs:
            dir_path = os.path.join(base_dir, dir_name)
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Ensured directory exists: {dir_path}")
        
        # Check CUDA availability
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "unknown"
            logger.info(f"CUDA is available. Device: {device_name}")
            
            # Test with a simple tensor operation
            try:
                x = torch.ones(1, device='cuda')
                y = x * 2
                del x, y
                logger.info("CUDA tensor operations test passed")
            except Exception as e:
                logger.warning(f"CUDA tensor operation test failed: {e}")
        else:
            logger.warning("CUDA is not available. Using CPU only.")
            
        # Apply minimal patches to ensure model compatibility without complex version manipulations
        try:
            # Import utility functions directly rather than through compatibility layer
            # Standardize device naming
            def standardize_device_name(device_str):
                """Standardize device naming between environments"""
                if device_str == "cuda" and torch.cuda.is_available():
                    return "cuda:0"
                return device_str
            
            # Set the global utility
            setattr(sys.modules[__name__], 'standardize_device_name', standardize_device_name)
            logger.info("Set up device name standardization")
                
            # Print success message
            logger.info("Simplified Colab environment setup successful!")
            
            # Print usage instructions
            print("\n========== lc_pipeline Colab Setup ==========")
            print("✅ Environment successfully configured for Colab")
            print("\nNext steps:")
            print("1. Import core modules:")
            print("   from lc_pipeline.config import load_config")
            print("   from lc_pipeline.main import main")
            print("\n2. Load config:")
            print("   config = load_config('your_config.yaml')")
            print("\n3. Run pipeline:")
            print("   results = main(config)")
            print("=============================================\n")
            
            return True
        except Exception as e:
            logger.error(f"Error in model compatibility setup: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            logger.warning("Continuing with partial setup")
            return False
    
    except Exception as e:
        logger.error(f"Error setting up Colab environment: {e}")
        import traceback
        traceback.print_exc()
        return False

def download_sample_data(dataset_size="small", output_dir=None):
    """
    Download sample asteroid lightcurve data for testing the pipeline.
    
    This is a wrapper around the download_data.py script.
    
    Args:
        dataset_size: Size of the dataset to download ("small", "medium", or "full")
        output_dir: Directory to save the data to (default: None, uses default)
        
    Returns:
        str: Path to the downloaded data
    """
    try:
        from lc_pipeline.download_data import download_and_extract_data
        
        # Default output directory is /content/data in Colab
        if output_dir is None:
            output_dir = "/content/data"
            os.makedirs(output_dir, exist_ok=True)
            
        # Download and extract the data
        success = download_and_extract_data(dataset_size, output_dir)
        
        if success:
            logger.info(f"Successfully downloaded {dataset_size} dataset to {output_dir}")
            return output_dir
        else:
            logger.error("Failed to download sample data")
            return None
            
    except Exception as e:
        logger.error(f"Error downloading sample data: {e}")
        import traceback
        traceback.print_exc()
        return None

def download_pretrained_models(models="all", output_dir=None):
    """
    Download pretrained models for the pipeline.
    
    This is a wrapper around the download_models.py script.
    
    Args:
        models: Names of models to download (default: "all")
        output_dir: Directory to save the models to (default: None, uses default)
        
    Returns:
        str: Path to the downloaded models
    """
    try:
        from lc_pipeline.download_models import download_models
        
        # Default output directory is /content/models in Colab
        if output_dir is None:
            output_dir = "/content/models"
            os.makedirs(output_dir, exist_ok=True)
            
        # Handle "all" case
        if models == "all":
            models = ["all"]
            
        # Convert string to list if needed
        if isinstance(models, str):
            models = [models]
            
        # Download the models
        success, paths = download_models(models, output_dir)
        
        if success:
            logger.info(f"Successfully downloaded models to {output_dir}")
            return output_dir
        else:
            logger.error("Failed to download pretrained models")
            return None
            
    except Exception as e:
        logger.error(f"Error downloading pretrained models: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # If this script is run directly, set up the Colab environment
    setup_colab() 