#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Caching utilities for datasets in the light curve analysis pipeline.
"""

import os
import torch
import numpy as np
import hashlib
import json
import logging
from typing import Tuple, List, Dict, Any, Optional, Union


def get_cached_axis_data(
    dataset_identifier: str,
    indices_to_process: List[int],
    period_model_identifier: Optional[str] = None,
    use_predicted_period: bool = False,
    config_snapshot: Dict[str, Any] = None,
    cache_dir: str = "data_cache/axis_dataset",
    force_recache: bool = False,
    logger: Optional[Any] = None
) -> Optional[Tuple[np.ndarray, np.ndarray, List[str], List[float]]]:
    """
    Attempt to load cached axis data generated from phase-folding.
    
    Args:
        dataset_identifier: Identifier for the source dataset (e.g., hash of CSV files)
        indices_to_process: List of indices from the source dataset to process
        period_model_identifier: Identifier for the period model used (e.g., path/hash)
        use_predicted_period: Whether predicted periods were used for phase-folding
        config_snapshot: Dictionary of relevant configuration parameters
        cache_dir: Directory for cache files
        force_recache: If True, ignore cache and force reprocessing
        logger: Logger instance
        
    Returns:
        If cache hit: Tuple of (phase_folded_curves, axis_vectors, successful_ids, periods_used)
        If cache miss: None
    """
    log = logger or logging.getLogger(__name__)
    
    if not cache_dir:
        log.debug("Cache directory not specified, skipping cache lookup.")
        return None
        
    cache_filename = _get_axis_cache_filename(
        dataset_identifier, 
        indices_to_process, 
        period_model_identifier, 
        use_predicted_period,
        config_snapshot
    )
    
    cache_path = os.path.join(cache_dir, cache_filename)
    log.info(f"Axis data cache path: {cache_path}")
    
    if force_recache:
        log.info("Force recache requested, skipping cache lookup.")
        return None
        
    if not os.path.exists(cache_path):
        log.info("No cached axis data found. Will generate fresh data.")
        return None
        
    try:
        log.info(f"Loading cached axis data from {cache_path}")
        cached_data = torch.load(cache_path)
        
        phase_folded_curves = cached_data['phase_folded_curves']
        axis_vectors = cached_data['axis_vectors']
        successful_ids = cached_data['successful_ids']
        
        # For backward compatibility, check if periods_used exists, otherwise use placeholders
        periods_used = cached_data.get('periods_used', [0.0] * len(phase_folded_curves))
        
        log.info(f"Successfully loaded cached axis data: {len(phase_folded_curves)} samples")
        
        return phase_folded_curves, axis_vectors, successful_ids, periods_used
    except Exception as e:
        log.warning(f"Failed to load cached axis data: {e}")
        return None


def save_axis_data_to_cache(
    data_tuple: Tuple[np.ndarray, np.ndarray, List[str], Optional[List[float]]],
    dataset_identifier: str,
    indices_to_process: List[int],
    period_model_identifier: Optional[str] = None,
    use_predicted_period: bool = False,
    config_snapshot: Dict[str, Any] = None,
    cache_dir: str = "data_cache/axis_dataset",
    logger: Optional[Any] = None
) -> bool:
    """
    Save generated axis data to cache.
    
    Args:
        data_tuple: Tuple of (phase_folded_curves, axis_vectors, successful_ids) or 
                   (phase_folded_curves, axis_vectors, successful_ids, periods_used)
        dataset_identifier: Identifier for the source dataset
        indices_to_process: List of indices that were processed
        period_model_identifier: Identifier for the period model used 
        use_predicted_period: Whether predicted periods were used
        config_snapshot: Dictionary of relevant configuration parameters
        cache_dir: Directory for cache files
        logger: Logger instance
        
    Returns:
        bool: True if successfully cached, False otherwise
    """
    log = logger or logging.getLogger(__name__)
    
    if not cache_dir:
        log.debug("Cache directory not specified, skipping caching.")
        return False
        
    # Unpack data tuple with handling for optional periods_used
    if len(data_tuple) == 3:
        phase_folded_curves, axis_vectors, successful_ids = data_tuple
        periods_used = [0.0] * len(phase_folded_curves)  # Placeholder
    elif len(data_tuple) == 4:
        phase_folded_curves, axis_vectors, successful_ids, periods_used = data_tuple
    else:
        log.error(f"Invalid data_tuple length: {len(data_tuple)}. Expected 3 or 4 elements.")
        return False
    
    cache_filename = _get_axis_cache_filename(
        dataset_identifier, 
        indices_to_process, 
        period_model_identifier, 
        use_predicted_period,
        config_snapshot
    )
    
    cache_path = os.path.join(cache_dir, cache_filename)
    
    try:
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
            
        data_to_cache = {
            'phase_folded_curves': phase_folded_curves,
            'axis_vectors': axis_vectors,
            'successful_ids': successful_ids,
            'periods_used': periods_used,
            # Store metadata about how this cache was created
            'metadata': {
                'dataset_identifier': dataset_identifier,
                'indices_processed': indices_to_process,
                'period_model_identifier': period_model_identifier,
                'use_predicted_period': use_predicted_period,
                'config_snapshot': config_snapshot,
                'cache_version': 'v1'
            }
        }
        
        log.info(f"Saving axis data to cache: {cache_path}")
        torch.save(data_to_cache, cache_path)
        log.info(f"Successfully cached axis data ({len(phase_folded_curves)} samples)")
        return True
    except Exception as e:
        log.warning(f"Failed to cache axis data: {e}")
        return False


def get_dataset_identifier(dataset: Any) -> str:
    """
    Generate a consistent identifier for a dataset.
    
    For AsteroidDataset, this is based on the CSV files.
    For a Subset of AsteroidDataset, this is based on the wrapped dataset + indices.
    
    Args:
        dataset: The dataset object (AsteroidDataset or Subset)
        
    Returns:
        str: A unique identifier string for the dataset
    """
    from torch.utils.data import Subset
    
    # If it's a Subset, get the base dataset
    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        hasher = hashlib.md5()
        
        # Hash base dataset identifier
        base_id = get_dataset_identifier(base_dataset)
        hasher.update(base_id.encode('utf-8'))
        
        # Hash the subset indices (sorted)
        indices = sorted(dataset.indices)
        hasher.update(json.dumps(indices).encode('utf-8'))
        
        return f"subset_{hasher.hexdigest()}"
    
    # For AsteroidDataset, use csv files
    elif hasattr(dataset, 'csv_files_original_list'):
        hasher = hashlib.md5()
        
        # Hash the sorted list of CSV files
        for f_path in sorted(dataset.csv_files_original_list):
            hasher.update(f_path.encode('utf-8'))
            
        return f"asteroid_dataset_{hasher.hexdigest()}"
    
    # For other types, use object id
    else:
        return f"dataset_{id(dataset)}"


def get_period_model_identifier(period_model: Optional[Any]) -> Optional[str]:
    """
    Generate a consistent identifier for a period model.
    
    Based on the model's state dict.
    
    Args:
        period_model: The period model or None
        
    Returns:
        Optional[str]: An identifier string for the period model or None
    """
    if period_model is None:
        return None
        
    hasher = hashlib.md5()
    
    # If the model has a path attribute, use that
    model_path = getattr(period_model, 'save_path', None)
    if model_path:
        hasher.update(str(model_path).encode('utf-8'))
        return f"period_model_path_{hasher.hexdigest()}"
    
    # Otherwise, hash a representation of the state dict parameters
    try:
        # Get state dict and extract parameter shapes and non-zero count as a fingerprint
        # (Full parameters would be too large to hash efficiently)
        state_dict = period_model.state_dict()
        param_info = {}
        
        for key, param in state_dict.items():
            if hasattr(param, 'shape') and hasattr(param, 'nonzero'):
                param_info[key] = {
                    'shape': list(param.shape),
                    'nonzeros': param.nonzero().numel()
                }
        
        # Hash this parameter fingerprint
        hasher.update(json.dumps(param_info, sort_keys=True).encode('utf-8'))
        return f"period_model_params_{hasher.hexdigest()}"
    except Exception:
        # Fallback to model id if state_dict fails
        return f"period_model_{id(period_model)}"


def _get_axis_cache_filename(
    dataset_identifier: str,
    indices_to_process: List[int],
    period_model_identifier: Optional[str],
    use_predicted_period: bool,
    config_snapshot: Dict[str, Any]
) -> str:
    """
    Generate a unique filename for axis data cache.
    
    Args:
        dataset_identifier: Identifier for the source dataset
        indices_to_process: List of indices to process
        period_model_identifier: Identifier for the period model
        use_predicted_period: Whether predicted periods were used
        config_snapshot: Dictionary of configuration parameters
        
    Returns:
        str: Cache filename
    """
    hasher = hashlib.md5()
    
    # Hash dataset identifier
    hasher.update(dataset_identifier.encode('utf-8'))
    
    # Hash indices to process (sorted)
    indices_str = json.dumps(sorted(indices_to_process))
    hasher.update(indices_str.encode('utf-8'))
    
    # Hash period model identifier if used
    if use_predicted_period and period_model_identifier is not None:
        hasher.update(period_model_identifier.encode('utf-8'))
    
    # Add flag for whether predicted periods were used
    hasher.update(str(use_predicted_period).encode('utf-8'))
    
    # Hash relevant configuration parameters
    if config_snapshot:
        # Sort keys for consistent hash
        config_str = json.dumps(sorted(config_snapshot.items()))
        hasher.update(config_str.encode('utf-8'))
    
    # Add version for cache format
    cache_version = "v1"
    hasher.update(cache_version.encode('utf-8'))
    
    period_type = "predicted" if use_predicted_period else "original"
    return f"axis_data_{period_type}_{hasher.hexdigest()}.pt" 