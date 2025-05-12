#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data downloader utility for the light curve analysis pipeline.

This script downloads sample asteroid light curve data for testing the pipeline.
"""

import os
import sys
import argparse
import logging
import requests
import zipfile
import io
from pathlib import Path
from tqdm.auto import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('lc_pipeline.download_data')

# Sample data URLs - these are example placeholders, replace with actual data links
SAMPLE_DATA_URLS = {
    'small': 'https://example.com/asteroid_data_small.zip',  # Replace with actual URL
    'medium': 'https://example.com/asteroid_data_medium.zip',  # Replace with actual URL
    'full': 'https://example.com/asteroid_data_full.zip'  # Replace with actual URL
}

# Default download location
DEFAULT_DATA_DIR = os.path.join(os.path.expanduser('~'), 'lc_pipeline_data')


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
                
        return True
    except Exception as e:
        logger.error(f"Download failed: {str(e)}")
        return False


def extract_zip(zip_path: str, extract_to: str) -> bool:
    """
    Extract a ZIP file with progress indication.
    
    Args:
        zip_path: Path to ZIP file
        extract_to: Directory to extract to
        
    Returns:
        bool: True if extraction was successful, False otherwise
    """
    try:
        logger.info(f"Extracting {zip_path} to {extract_to}")
        
        # Get total size for progress bar
        total_size = os.path.getsize(zip_path)
        extracted_size = 0
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Count total files for progress
            total_files = len(zip_ref.infolist())
            
            # Extract with progress bar
            with tqdm(total=total_files, desc="Extracting") as progress_bar:
                for file in zip_ref.infolist():
                    zip_ref.extract(file, extract_to)
                    progress_bar.update(1)
                    
        logger.info(f"Successfully extracted {total_files} files")
        return True
    except Exception as e:
        logger.error(f"Extraction failed: {str(e)}")
        return False


def download_and_extract_data(dataset_size: str, output_dir: str, keep_zip: bool = False) -> bool:
    """
    Download and extract a dataset.
    
    Args:
        dataset_size: Size of the dataset ('small', 'medium', or 'full')
        output_dir: Directory to save the dataset to
        keep_zip: Whether to keep the downloaded ZIP file
        
    Returns:
        bool: True if successful, False otherwise
    """
    if dataset_size not in SAMPLE_DATA_URLS:
        logger.error(f"Invalid dataset size: {dataset_size}. Valid options: {', '.join(SAMPLE_DATA_URLS.keys())}")
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Download the dataset
    url = SAMPLE_DATA_URLS[dataset_size]
    zip_path = os.path.join(output_dir, f"asteroid_data_{dataset_size}.zip")
    
    if not download_file(url, zip_path):
        return False
    
    # Extract the dataset
    extract_success = extract_zip(zip_path, output_dir)
    
    # Remove the ZIP file if requested
    if not keep_zip and extract_success:
        try:
            os.remove(zip_path)
            logger.info(f"Removed ZIP file: {zip_path}")
        except Exception as e:
            logger.warning(f"Failed to remove ZIP file: {str(e)}")
    
    return extract_success


def main():
    """Main function for the data downloader."""
    parser = argparse.ArgumentParser(description="Download asteroid light curve data for the pipeline")
    parser.add_argument("--size", type=str, choices=["small", "medium", "full"], default="small",
                        help="Size of dataset to download (small: ~100MB, medium: ~500MB, full: ~2GB)")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_DATA_DIR,
                        help=f"Directory to save the data to (default: {DEFAULT_DATA_DIR})")
    parser.add_argument("--keep-zip", action="store_true",
                        help="Keep the downloaded ZIP file after extraction")
    
    args = parser.parse_args()
    
    # Display welcome message
    print("""
    ╔════════════════════════════════════════════════╗
    ║                                                ║
    ║        ASTEROID LIGHTCURVE DATA DOWNLOAD       ║
    ║                                                ║
    ╚════════════════════════════════════════════════╝
    """)
    
    # Download and extract the data
    success = download_and_extract_data(args.size, args.output_dir, args.keep_zip)
    
    if success:
        # Print success message with instructions
        print("\n✅ Data download and extraction completed successfully!")
        print(f"\nData location: {args.output_dir}")
        print("\nTo use this data in the pipeline, update your config.yaml or use the --data-dir option:")
        print(f"  python -m lc_pipeline.main --data-dir={args.output_dir}")
        return 0
    else:
        print("\n❌ Data download or extraction failed. Please check the logs for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 