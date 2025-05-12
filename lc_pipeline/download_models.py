#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model downloader utility for the light curve analysis pipeline.

This script downloads pre-trained models for quick starting with the pipeline.
"""

import os
import sys
import argparse
import logging
import requests
import hashlib
from pathlib import Path
from tqdm.auto import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('lc_pipeline.download_models')

# Pre-trained model URLs - these are example placeholders, replace with actual model links
PRETRAINED_MODELS = {
    'period_lstm': {
        'url': 'https://example.com/period_lstm_model.pt',  # Replace with actual URL
        'md5': 'abcdef0123456789abcdef0123456789', # Replace with actual checksum
        'size': '50MB'
    },
    'period_transformer': {
        'url': 'https://example.com/period_transformer_model.pt',  # Replace with actual URL
        'md5': '0123456789abcdef0123456789abcdef', # Replace with actual checksum
        'size': '80MB'
    },
    'axis_cnn': {
        'url': 'https://example.com/axis_cnn_model.pt',  # Replace with actual URL
        'md5': '9876543210fedcba9876543210fedcba', # Replace with actual checksum
        'size': '30MB'
    }
}

# Default download location
DEFAULT_MODEL_DIR = os.path.join(os.path.expanduser('~'), 'lc_pipeline_models')


def calculate_md5(file_path: str) -> str:
    """
    Calculate MD5 hash of a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        str: MD5 hash
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def verify_checksum(file_path: str, expected_md5: str) -> bool:
    """
    Verify the MD5 checksum of a file.
    
    Args:
        file_path: Path to the file
        expected_md5: Expected MD5 hash
        
    Returns:
        bool: True if the checksum matches, False otherwise
    """
    actual_md5 = calculate_md5(file_path)
    matches = actual_md5.lower() == expected_md5.lower()
    
    if matches:
        logger.info(f"Checksum verification successful: {file_path}")
    else:
        logger.error(f"Checksum verification failed for {file_path}")
        logger.error(f"  Expected: {expected_md5}")
        logger.error(f"  Actual: {actual_md5}")
    
    return matches


def download_file(url: str, target_path: str) -> bool:
    """
    Download a file from a URL with progress bar.
    
    Args:
        url: URL to download from
        target_path: Path to save the file to
        
    Returns:
        bool: True if download was successful, False otherwise
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 KB
        
        logger.info(f"Downloading from {url} to {target_path}")
        
        with open(target_path, 'wb') as file, tqdm(
            desc="Downloading",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for data in response.iter_content(block_size):
                file.write(data)
                progress_bar.update(len(data))
                
        logger.info(f"Download completed: {target_path}")
        return True
    except Exception as e:
        logger.error(f"Download failed: {str(e)}")
        return False


def download_model(model_name: str, output_dir: str, force: bool = False) -> bool:
    """
    Download a pre-trained model.
    
    Args:
        model_name: Name of the model to download
        output_dir: Directory to save the model to
        force: Whether to force download even if the file exists
        
    Returns:
        bool: True if successful, False otherwise
    """
    if model_name not in PRETRAINED_MODELS:
        logger.error(f"Invalid model name: {model_name}. Valid options: {', '.join(PRETRAINED_MODELS.keys())}")
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get model information
    model_info = PRETRAINED_MODELS[model_name]
    url = model_info['url']
    expected_md5 = model_info['md5']
    
    # Determine target path
    model_filename = os.path.basename(url)
    target_path = os.path.join(output_dir, model_filename)
    
    # Check if the file already exists and has the correct checksum
    if os.path.exists(target_path) and not force:
        logger.info(f"Model file already exists: {target_path}")
        
        # Verify checksum
        if verify_checksum(target_path, expected_md5):
            logger.info("Existing file is valid, skipping download")
            return True
        else:
            logger.warning("Existing file has incorrect checksum, re-downloading")
    
    # Download the model
    if not download_file(url, target_path):
        return False
    
    # Verify the downloaded file
    if not verify_checksum(target_path, expected_md5):
        return False
    
    # Create a symbolic link for easier access
    link_name = f"{model_name}_model.pt"
    link_path = os.path.join(output_dir, link_name)
    
    try:
        # Remove existing link if it exists
        if os.path.exists(link_path):
            os.remove(link_path)
        
        # Create symbolic link on Unix-like systems
        if os.name == 'posix':
            os.symlink(target_path, link_path)
            logger.info(f"Created symbolic link: {link_path} -> {target_path}")
        # Copy file on Windows (which may not support symlinks)
        else:
            import shutil
            shutil.copy2(target_path, link_path)
            logger.info(f"Created copy: {link_path}")
    except Exception as e:
        logger.warning(f"Failed to create link/copy: {str(e)}")
    
    return True


def main():
    """Main function for the model downloader."""
    parser = argparse.ArgumentParser(description="Download pre-trained models for the pipeline")
    parser.add_argument("--models", type=str, nargs='+', 
                        choices=list(PRETRAINED_MODELS.keys()) + ['all'],
                        default=['all'],
                        help="Names of models to download (default: all)")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_MODEL_DIR,
                        help=f"Directory to save the models to (default: {DEFAULT_MODEL_DIR})")
    parser.add_argument("--force", action="store_true",
                        help="Force download even if models already exist")
    
    args = parser.parse_args()
    
    # Display welcome message
    print("""
    ╔════════════════════════════════════════════════╗
    ║                                                ║
    ║        ASTEROID LIGHTCURVE MODEL DOWNLOAD      ║
    ║                                                ║
    ╚════════════════════════════════════════════════╝
    """)
    
    # Determine which models to download
    models_to_download = list(PRETRAINED_MODELS.keys()) if 'all' in args.models else args.models
    
    print(f"Models to download: {', '.join(models_to_download)}")
    print(f"Download location: {args.output_dir}")
    
    # Download each model
    success = True
    for model_name in models_to_download:
        print(f"\nDownloading {model_name} model ({PRETRAINED_MODELS[model_name]['size']})...")
        
        model_success = download_model(model_name, args.output_dir, args.force)
        success = success and model_success
        
        if model_success:
            print(f"✅ {model_name} model downloaded successfully")
        else:
            print(f"❌ {model_name} model download failed")
    
    if success:
        # Print success message with instructions
        print("\n✅ All requested models downloaded successfully!")
        print(f"\nModels location: {args.output_dir}")
        print("\nTo use these models in the pipeline, update your config.yaml or use the command line:")
        print(f"  python -m lc_pipeline.main --model-dir={args.output_dir}")
        return 0
    else:
        print("\n❌ Some models failed to download. Please check the logs for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 