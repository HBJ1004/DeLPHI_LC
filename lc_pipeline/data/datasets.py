#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset classes for light curve analysis pipeline.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional, Union, Any
import logging
from tqdm import tqdm
import multiprocessing
import traceback
import hashlib # Added for cache filename generation
import json # Added for config hashing


class AsteroidDataset(Dataset):
    """
    Dataset for asteroid light curves.
    
    Loads and preprocesses asteroid light curve data from CSV files.
    Each item represents one asteroid with time series data.
    """
    
    POSSIBLE_MAG_COLS = ['magnitude', 'mag', 'brightness', 'flux']
    POSSIBLE_ERR_COLS = ['error', 'err', 'e_mag', 'flux_err']
    POSSIBLE_TIME_COLS = ['time', 'jd', 'mjd']

    def __init__(
        self, 
        csv_files: List[str], 
        config: Optional[Dict[str, Any]] = None, # Expecting relevant parts of data config for hashing
        max_sequence_length: int = 200, 
        period_scale: Optional[float] = None,
        logger: Optional[Any] = None,
        max_files: Optional[int] = None,
        num_workers: int = 4,
        use_single_file_processing_cache: bool = True, # For in-memory cache of individual file processing results during parallel loading
        cache_dir: Optional[str] = "data_cache/asteroid_dataset", # For disk-based full dataset cache
        force_recache: bool = False # For disk-based full dataset cache
    ):
        self.csv_files_original_list = sorted(list(set(csv_files))) # Store sorted unique list for cache key
        self.csv_files = self.csv_files_original_list[:max_files] if max_files else self.csv_files_original_list
        
        # Ensure self.max_sequence_length is set from the direct parameter first.
        self.max_sequence_length = max_sequence_length 
        
        # Now, self.config_snapshot can safely reference self.max_sequence_length if needed.
        self.config_snapshot = self._get_relevant_config_snapshot(config)
        
        self.logger = logger if logger else logging.getLogger(__name__)
        self.num_workers = num_workers
        self.use_single_file_processing_cache = use_single_file_processing_cache # For old in-memory cache
        self._cache = {} # Old in-memory cache for individual file processing results

        self.cache_dir = cache_dir
        self.force_recache = force_recache
        self.cache_filename = None # Will be set by _get_cache_filename
        self.cache_path = None # Will be set if cache_dir is provided

        if self.cache_dir:
            self.cache_filename = self._get_cache_filename()
            self.cache_path = os.path.join(self.cache_dir, self.cache_filename)
            self.logger.info(f"AsteroidDataset cache path: {self.cache_path}")
            if not self.force_recache and os.path.exists(self.cache_path):
                self.logger.info(f"Attempting to load AsteroidDataset from cache: {self.cache_path}")
                try:
                    cached_data = torch.load(self.cache_path, map_location=torch.device('cpu'), weights_only=False)
                    self.features = cached_data['features']
                    self.targets = cached_data['targets']
                    self.ids = cached_data['ids']
                    self.source_files = cached_data['source_files']
                    self.raw_data_cache = cached_data['raw_data_cache']
                    # period_scale is determined after loading, so load what was used during caching
                    self.period_scale_factor = cached_data.get('period_scale_factor', 50.0) 
                    self.logger.info(f"Successfully loaded AsteroidDataset from cache. Loaded {len(self.features)} items.")
                    self.logger.info(f"Using cached period_scale_factor: {self.period_scale_factor}")
                    # Skip further initialization if cache is loaded
                    if not self.features: # Should not happen if cache was valid
                        self.logger.warning("Cache loaded but no features found. Proceeding to re-process.")
                        self._initialize_and_process_data(period_scale)
                    return 
                except Exception as e:
                    self.logger.warning(f"Failed to load AsteroidDataset from cache: {e}. Re-processing.")
                    self._initialize_and_process_data(period_scale)
                    return

        # If cache not used, not found, or failed to load, or force_recache is True
        self._initialize_and_process_data(period_scale)

    def _get_relevant_config_snapshot(self, config_dict: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Extracts data-processing relevant parts of a config for cache hashing."""
        snapshot = {}
        if config_dict:
            # Prefer max_sequence_length from the passed config_dict if it exists
            snapshot['max_sequence_length'] = config_dict.get('max_sequence_length', self.max_sequence_length)
            snapshot['default_error'] = config_dict.get('default_error_value', 0.1) # If it influences processing
            # Add other relevant keys from your DataConfig that change preprocessing
            # For now, keeping it simple. Add more as needed.
        else:
            # If config_dict is None, use the instance's max_sequence_length
            snapshot['max_sequence_length'] = self.max_sequence_length
        
        # Ensure max_sequence_length is always in the snapshot (redundant if logic above is correct, but safe)
        if 'max_sequence_length' not in snapshot:
             snapshot['max_sequence_length'] = self.max_sequence_length
        return snapshot

    def _get_cache_filename(self) -> str:
        """Generates a unique filename for the dataset cache."""
        hasher = hashlib.md5()
        
        # Hash the list of CSV files
        for f_path in self.csv_files_original_list: # Use the full original list for consistent hashing
            hasher.update(f_path.encode('utf-8'))
            
        # Hash relevant configuration parameters
        # Sort the config items to ensure consistent hash
        sorted_config_items = sorted(self.config_snapshot.items())
        hasher.update(json.dumps(sorted_config_items).encode('utf-8'))
        
        # Add a version to the cache key in case processing logic changes
        cache_version = "v1" 
        hasher.update(cache_version.encode('utf-8'))
        
        return f"dataset_cache_{hasher.hexdigest()}.pt"

    def _initialize_and_process_data(self, period_scale: Optional[float]):
        """Helper to contain the original data loading and processing logic."""
        if not self.csv_files:
            error_msg = "No input CSV files provided."
            self.logger.error(error_msg)
            raise ValueError(error_msg)
            
        valid_files_exist = any(os.path.exists(f) for f in self.csv_files)
        if not valid_files_exist:
            error_msg = f"None of the {len(self.csv_files)} specified CSV files exist."
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        self.features = []
        self.targets = []
        self.ids = []
        self.source_files = []
        self.raw_data_cache = []

        self._load_and_preprocess_data()
        
        if not self.features:
            error_msg = f"No valid data was loaded and processed from the {len(self.csv_files)} input files."
            self.logger.error(error_msg)
            # Do not raise ValueError here if we want to allow empty datasets from valid but empty/filtered files
            # For now, keeping original behavior:
            raise ValueError(error_msg) 
        
        if period_scale is None:
            periods = np.array([target[2] for target in self.targets if target is not None and len(target) == 3 and target[2] > 0])
            if periods.size > 0:
                self.period_scale_factor = np.percentile(periods, 95)
            else:
                self.period_scale_factor = 50.0 # Default fallback
                self.logger.warning(f"No valid periods in dataset, using default period_scale_factor={self.period_scale_factor}")
        else:
            self.period_scale_factor = period_scale
        self.logger.info(f"Using period_scale_factor: {self.period_scale_factor}")

        # Save to cache if processed and cache_dir is set
        if self.cache_dir and self.cache_path and (self.force_recache or not os.path.exists(self.cache_path)):
            self.logger.info(f"Saving AsteroidDataset to cache: {self.cache_path}")
            try:
                if not os.path.exists(self.cache_dir):
                    os.makedirs(self.cache_dir, exist_ok=True)
                data_to_cache = {
                    'features': self.features,
                    'targets': self.targets,
                    'ids': self.ids,
                    'source_files': self.source_files,
                    'raw_data_cache': self.raw_data_cache,
                    'period_scale_factor': self.period_scale_factor # Save the determined scale factor
                }
                torch.save(data_to_cache, self.cache_path)
                self.logger.info(f"Successfully saved AsteroidDataset to cache.")
            except Exception as e:
                self.logger.warning(f"Failed to save AsteroidDataset to cache: {e}")

    def _log(self, level: str, message: str):
        if self.logger:
            getattr(self.logger, level.lower(), self.logger.info)(message)
        else:
            print(f"{level.upper()}: {message}")
    
    @staticmethod
    def _calculate_features_static(time_np: np.ndarray, mag_np: np.ndarray, err_np: np.ndarray) -> np.ndarray:
        # Normalize time to [0, 1] range
        time_min, time_max = time_np.min(), time_np.max()
        time_norm = (time_np - time_min) / (time_max - time_min + 1e-6)

        # Normalize magnitude (center and scale)
        mag_mean, mag_std = mag_np.mean(), mag_np.std()
        mag_norm = (mag_np - mag_mean) / (mag_std + 1e-6)

        # Scale error by magnitude std dev
        err_scaled = err_np / (mag_std + 1e-6)

        # Calculate delta features
        dt = np.diff(time_np, prepend=time_np[0])
        dmag = np.diff(mag_np, prepend=mag_np[0])
        dt_norm = (dt - dt.mean()) / (dt.std() + 1e-6)
        dmag_dt = np.divide(dmag, dt, out=np.zeros_like(dmag, dtype=float), where=dt!=0)

        # Calculate rolling window features (using pandas for convenience)
        df_temp = pd.DataFrame({'mag': mag_np})
        roll_mean_5 = df_temp['mag'].rolling(window=5, min_periods=1, center=True).mean().values
        roll_std_5 = df_temp['mag'].rolling(window=5, min_periods=1, center=True).std().fillna(0).values
        roll_mean_10 = df_temp['mag'].rolling(window=10, min_periods=1, center=True).mean().values
        roll_std_10 = df_temp['mag'].rolling(window=10, min_periods=1, center=True).std().fillna(0).values

        # Time Encoding Features
        time_sin_2pi = np.sin(2 * np.pi * time_norm)
        time_cos_2pi = np.cos(2 * np.pi * time_norm)
        time_sin_4pi = np.sin(4 * np.pi * time_norm)
        time_cos_4pi = np.cos(4 * np.pi * time_norm)
        time_sin_8pi = np.sin(8 * np.pi * time_norm)
        time_cos_8pi = np.cos(8 * np.pi * time_norm)
        time_squared = time_norm**2

        return np.stack([
            time_norm, mag_norm, err_scaled, dt_norm, dmag, dmag_dt,
            roll_mean_5, roll_std_5, roll_mean_10, roll_std_10,
            time_sin_2pi, time_cos_2pi, time_sin_4pi, time_cos_4pi,
            time_sin_8pi, time_cos_8pi, time_squared
        ], axis=-1).astype(np.float32)

    def _load_and_preprocess_data(self):
        self.logger.info(f"Starting to load and preprocess {len(self.csv_files)} CSV files...")
        required_cols_damit = ['time', 'relative_brightness', 'l', 'b', 'rot_per']
        
        if self.num_workers > 0 and len(self.csv_files) >= 10:
            self._load_data_parallel()
        else:
            self.logger.info(f"Using sequential data loading ({len(self.csv_files)} files <= 10 or num_workers <= 0).")
            self._load_data_sequential()
        
        if len(self.features) == 0 and len(self.csv_files) > 0:
             self.logger.error("No files were successfully processed. Check data integrity and required columns.")

    def _load_data_sequential(self):
        """Load and process data sequentially"""
        required_cols_damit = ['time', 'relative_brightness', 'l', 'b', 'rot_per']
        success_count = 0
        for i, file_path in enumerate(self.csv_files):
            if i % 100 == 0 and i > 0:
                self.logger.info(f"Processed {i}/{len(self.csv_files)} files...")
            
            # Determine the logger identifier to pass
            logger_identifier = getattr(self.logger, 'experiment_name', None)
            if logger_identifier is None:
                logger_identifier = getattr(self.logger, 'name', 'dataset_processing_sequential')

            result = AsteroidDataset._process_single_file_static(
                file_path, 
                self.max_sequence_length, 
                logger_identifier
            )

            if result is not None:
                features_tensor, target_tensor, metadata_tuple, raw_data_tuple = result
                if features_tensor is not None:
                    self.features.append(features_tensor)
                    self.targets.append(target_tensor)
                    
                    if metadata_tuple is not None:
                        asteroid_id, _ = metadata_tuple # file_path is already known
                        self.ids.append(asteroid_id)
                        self.source_files.append(file_path)
                    
                    if raw_data_tuple is not None:
                        self.raw_data_cache.append(raw_data_tuple)
                    else: # Should not happen if features_tensor is not None
                        self.raw_data_cache.append((np.array([]), np.array([]), np.array([])))
                        
                    success_count += 1
            else:
                # Ensure placeholder for raw_data_cache if processing failed entirely
                self.raw_data_cache.append((np.array([]), np.array([]), np.array([])))
        
        self.logger.info(f"Successfully loaded and processed {success_count} files sequentially.")

    def _load_data_parallel(self):
        """Load and process data in parallel using multiprocessing.Pool"""
        success_count = 0
        results_from_pool = []
        
        try:
            num_procs = self.num_workers if self.num_workers > 0 else multiprocessing.cpu_count()
            self.logger.info(f"Starting parallel data loading with {num_procs} workers.")
            
            # Determine the logger identifier to pass
            logger_identifier = getattr(self.logger, 'experiment_name', None)
            if logger_identifier is None:
                logger_identifier = getattr(self.logger, 'name', 'dataset_processing_parallel')
            
            map_args = [(fp, self.max_sequence_length, logger_identifier) for fp in self.csv_files]
            
            with multiprocessing.Pool(processes=num_procs) as pool:
                results_from_pool = list(tqdm(pool.starmap(AsteroidDataset._process_single_file_static, map_args), 
                                              total=len(self.csv_files), 
                                              desc="Processing files in parallel"))
                                    
        except Exception as pool_exc:
            self.logger.error(f"Error during parallel processing pool execution: {pool_exc}", exc_info=True)
            self.logger.warning("Falling back to sequential processing due to pool error.")
            self._load_data_sequential() 
            return 

        temp_features = []
        temp_targets = []
        temp_ids = []
        temp_source_files = []
        temp_raw_data_cache = []
        
        for i, result in enumerate(results_from_pool):
            original_file_path = self.csv_files[i] # Get original file path for caching key
            
            # Logic for populating self._cache (the in-memory cache for individual file results from _process_single_file_static)
            # This can be kept if it helps avoid re-processing single files during _load_data_parallel,
            # especially if _load_data_parallel itself is called multiple times or if the full disk cache is missed.
            if self.use_single_file_processing_cache and result is not None: 
                 self._cache[original_file_path] = result # Store raw result from _process_single_file_static

            if result is not None:
                features_tensor, target_tensor, metadata_tuple, raw_data_tuple = result
                if features_tensor is not None: # Check if processing was successful for this file
                    temp_features.append(features_tensor)
                    temp_targets.append(target_tensor)
                    
                    if metadata_tuple is not None:
                        asteroid_id, file_path_from_meta = metadata_tuple 
                        temp_ids.append(asteroid_id)
                        temp_source_files.append(file_path_from_meta) # Should match original_file_path
                    
                    if raw_data_tuple is not None:
                        temp_raw_data_cache.append(raw_data_tuple)
                    else: # Placeholder if raw_data_tuple somehow None despite successful processing
                        temp_raw_data_cache.append((np.array([]), np.array([]), np.array([])))

                    if self.use_single_file_processing_cache and metadata_tuple:
                        self._cache[original_file_path] = (features_tensor, target_tensor, metadata_tuple, raw_data_tuple)
                            
                    success_count += 1
                else: # Result is not None, but features_tensor is None (should not happen based on _process_single_file_static logic)
                    self.logger.warning(f"Processing returned None for features in {original_file_path} in worker process.")
                    temp_raw_data_cache.append((np.array([]), np.array([]), np.array([])))
            else:
                self.logger.warning(f"Processing failed for {original_file_path} in worker process (result was None).")
                temp_raw_data_cache.append((np.array([]), np.array([]), np.array([])))
        
        self.features = temp_features
        self.targets = temp_targets
        self.ids = temp_ids
        self.source_files = temp_source_files
        self.raw_data_cache = temp_raw_data_cache
        
        self.logger.info(f"Successfully loaded and processed {success_count} files using parallel processing.")

    @staticmethod
    def _process_single_file_static(file_path: str, max_sequence_length: int, logger_name: Optional[str]):
        """Process a single CSV file. Static method for multiprocessing compatibility."""
        logger = logging.getLogger(logger_name if logger_name else __name__ + ".worker")
        required_cols_damit = ['time', 'relative_brightness', 'l', 'b', 'rot_per']
        
        try:
            asteroid_id = os.path.basename(file_path).split('.')[0]
            # Load only necessary columns
            df = pd.read_csv(file_path, usecols=lambda col: col in required_cols_damit)
            
            if df.empty or not all(col in df.columns for col in required_cols_damit):
                logger.warning(f"Skipping {file_path}: missing required DAMIT columns or empty.")
                return None # Indicates failure for this file

            time_np = df['time'].values
            mag_np = df['relative_brightness'].values
            err_np = df.get('error', pd.Series(0.01, index=df.index)).values # Default error
            
            first_row = df.iloc[0]
            target_values = [first_row['l'], first_row['b'], first_row['rot_per']]
            raw_data = (time_np.copy(), mag_np.copy(), err_np.copy()) # Store raw before truncation
            
            if len(time_np) > max_sequence_length:
                indices = np.linspace(0, len(time_np) - 1, max_sequence_length, dtype=int)
                time_np = time_np[indices]
                mag_np = mag_np[indices]
                err_np = err_np[indices]
            
            feature_array = AsteroidDataset._calculate_features_static(time_np, mag_np, err_np)
            
            return (
                torch.tensor(feature_array, dtype=torch.float32), 
                torch.tensor(target_values, dtype=torch.float32),
                (asteroid_id, file_path), # Metadata tuple
                raw_data 
            )
            
        except Exception as e:
            # Ensure sys and traceback are available if this is run in a worker
            import sys
            import traceback # Ensure traceback is imported here too for the worker context
            error_msg = f"Error processing file {file_path} in worker (_process_single_file_static): {e}"
            # Use the passed logger_name to get a logger instance within the worker, or default
            worker_logger_name = logger_name if logger_name else __name__ + ".worker_static"
            current_logger = logging.getLogger(worker_logger_name)
            current_logger.error(error_msg) # Log to the named logger
            # current_logger.debug(f"Traceback for {file_path} in worker: {traceback.format_exc()}") # Already doing this, but stderr is more direct for crashes
            
            print(f"!!! WORKER _process_single_file_static ERROR !!!\\n{error_msg}", file=sys.stderr)
            print(f"--- Traceback from _process_single_file_static worker (file: {file_path}) ---", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            print(f"--- End Traceback from _process_single_file_static worker (file: {file_path}) ---", file=sys.stderr)
            sys.stderr.flush() # Ensure it gets printed
            return None # Indicates failure for this file

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        """
        Get data item and target by index.
        
        Args:
            idx: Index of the item
            
        Returns:
            tuple: (features, target, asteroid_id, raw_data_tuple)
        """
        try:
            # Retrieve cached processed data if available
            features, target, asteroid_id, raw_data_tuple = self._get_cached_item(idx)
            
            # Perform validation checks on tensors
            if not isinstance(features, torch.Tensor):
                raise TypeError(f"Features must be a tensor, got {type(features).__name__}")
            
            if features.dim() != 2:
                raise ValueError(f"Features tensor must be 2D [seq_len, feat_dim], got shape {features.shape}")
            
            if torch.isnan(features).any():
                self.logger.warning(f"NaN values detected in features for asteroid {asteroid_id}")
                # Replace NaNs with zeros to prevent propagation
                features = torch.nan_to_num(features, nan=0.0)
                
            if torch.isinf(features).any():
                self.logger.warning(f"Infinite values detected in features for asteroid {asteroid_id}")
                # Replace infinities with large values to prevent errors
                features = torch.nan_to_num(features, posinf=1e9, neginf=-1e9)
            
            # Validate target tensor
            if isinstance(target, torch.Tensor):
                if torch.isnan(target).any() or torch.isinf(target).any():
                    self.logger.warning(f"NaN/Inf values detected in target for asteroid {asteroid_id}")
                    target = torch.nan_to_num(target, nan=0.0, posinf=1e9, neginf=-1e9)
            
            return features, target, asteroid_id, raw_data_tuple
        except Exception as e:
            # Enhanced error reporting for worker crashes
            error_msg = f"Error retrieving item {idx} in AsteroidDataset.__getitem__: {e}"
            self.logger.error(error_msg)
            # Ensure sys and traceback are available if this is run in a worker
            # In a typical setup, they should be, but being explicit for debugging.
            import sys
            import traceback
            print(f"!!! WORKER __getitem__ ERROR !!!\\n{error_msg}", file=sys.stderr)
            print("--- Traceback from __getitem__ worker ---", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            print("--- End Traceback from __getitem__ worker ---", file=sys.stderr)
            sys.stderr.flush() # Ensure it gets printed

            # Return reasonable fallback tensors
            # Make fallback feature dimension safer
            try:
                # Attempt to use self.feature_dim if available and valid
                feature_dim_fallback = self.feature_dim if hasattr(self, 'feature_dim') and isinstance(self.feature_dim, int) and self.feature_dim > 0 else 10 # Default to 10 if not available/valid
            except Exception:
                feature_dim_fallback = 10 # Absolute fallback dimension
            
            fallback_features = torch.zeros((1, feature_dim_fallback), dtype=torch.float32)
            
            try:
                # Attempt to use len(self.target_keys) if available
                target_len_fallback = len(self.target_keys) if hasattr(self, 'target_keys') and isinstance(self.target_keys, list) else 3 # Default to 3 if not available
            except Exception:
                target_len_fallback = 3 # Absolute fallback length

            fallback_target = torch.zeros((target_len_fallback), dtype=torch.float32)
            fallback_id = f"error__{idx}"
            fallback_raw_data = (np.array([]), np.array([]), np.array([]))
            self.logger.error(f"Returning fallback tensors: features shape {fallback_features.shape}, target shape {fallback_target.shape}")
            return fallback_features, fallback_target, fallback_id, fallback_raw_data

    def get_raw_item(self, idx: int) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], List[float], str, str]:
        if idx < len(self.raw_data_cache):
            raw_time, raw_mag, raw_err = self.raw_data_cache[idx]
        else: # Should not happen if lists are kept in sync
            self.logger.warning(f"Index {idx} out of bounds for raw_data_cache (len {len(self.raw_data_cache)}). Returning empty arrays.")
            raw_time, raw_mag, raw_err = np.array([]), np.array([]), np.array([])
            
        raw_target_info = self.targets[idx].tolist() 
        return (raw_time, raw_mag, raw_err), raw_target_info, self.ids[idx], self.source_files[idx]

    def get_processed_features(self, idx: int) -> torch.Tensor:
        return self.features[idx]

    def _get_cached_item(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Get cached item or compute and return it.
        
        Args:
            idx: Index of the item
            
        Returns:
            tuple: (features, target, asteroid_id, raw_data_tuple)
        """
        # Simply retrieve from existing storage
        features = self.features[idx]
        target = self.targets[idx].clone()
        asteroid_id = self.ids[idx]
        
        raw_data_tuple = (np.array([]), np.array([]), np.array([])) # Default empty tuple
        if idx < len(self.raw_data_cache):
            raw_data_tuple = self.raw_data_cache[idx]
        else:
            self.logger.warning(f"Index {idx} out of bounds for raw_data_cache (len {len(self.raw_data_cache)}). Using empty raw_data.")
            
        return features, target, asteroid_id, raw_data_tuple


class AxisDataset(Dataset):
    """
    Dataset for axis prediction using phase-folded light curves.
    
    Used in the third phase of the pipeline to predict rotation axis
    from phase-folded curves.
    """
    
    def __init__(
        self, 
        phase_curves: np.ndarray, 
        axis_targets: np.ndarray,
        use_quaternions: bool = True
    ):
        """
        Initialize the axis dataset.
        
        Args:
            phase_curves: Array of phase-folded light curves [n_samples, n_bins]
            axis_targets: Array of axis targets [n_samples, 3] or [n_samples, 4]
            use_quaternions: Whether to use quaternions for targets (4D) or direction vectors (3D)
        """
        self.phase_curves = phase_curves
        self.use_quaternions = use_quaternions
        
        # Check if we need to convert between quaternions and direction vectors
        if not use_quaternions and axis_targets.shape[1] == 4:
            # Convert quaternions to direction vectors
            from lc_pipeline.models.utils import quaternion_to_direction_vector
            direction_vectors = np.zeros((len(axis_targets), 3))
            for i, quat in enumerate(axis_targets):
                direction_vectors[i] = quaternion_to_direction_vector(quat)
            self.axis_targets = direction_vectors
            print(f"Converted {len(axis_targets)} quaternions to direction vectors")
        elif use_quaternions and axis_targets.shape[1] == 3:
            # Convert direction vectors to quaternions
            from lc_pipeline.models.utils import direction_vector_to_quaternion
            quaternions = np.zeros((len(axis_targets), 4))
            for i, vec in enumerate(axis_targets):
                quaternions[i] = direction_vector_to_quaternion(vec)
            self.axis_targets = quaternions
            print(f"Converted {len(axis_targets)} direction vectors to quaternions")
        else:
            self.axis_targets = axis_targets
            
        print(f"AxisDataset initialized with {len(self.phase_curves)} samples, target shape: {self.axis_targets.shape}, use_quaternions: {use_quaternions}")
    
    def __len__(self):
        return len(self.phase_curves)
    
    def __getitem__(self, idx):
        return self.phase_curves[idx], self.axis_targets[idx]
    
    def get_raw_item(self, idx):
        """
        Get the raw data for an item.
        
        Args:
            idx: Index of the item
            
        Returns:
            tuple: (phase_curve, axis_target)
        """
        return self.phase_curves[idx].numpy(), self.axis_targets[idx].numpy()


class LSFeatureDataset(Dataset):
    """
    Dataset that includes Lomb-Scargle prior features.
    
    Wraps an AsteroidDataset and computes LS features on-the-fly.
    """
    
    def __init__(
        self, 
        base_dataset: AsteroidDataset, 
        min_period: float = 2.0, 
        max_period: float = 100.0,
        num_features: int = 2
    ):
        """
        Initialize the LS feature dataset.
        
        Args:
            base_dataset: Base AsteroidDataset to wrap
            min_period: Minimum period to consider in LS periodogram
            max_period: Maximum period to consider in LS periodogram
            num_features: Number of LS features to compute
        """
        self.base_dataset = base_dataset
        self.min_period = min_period
        self.max_period = max_period
        self.num_features = num_features
        
        # Try to import astropy
        try:
            from astropy.timeseries import LombScargle
            self.LombScargle = LombScargle
        except ImportError:
            raise ImportError("astropy is required for LSFeatureDataset. "
                             "Install it with 'pip install astropy'.")
    
    def __len__(self):
        """Return the number of samples."""
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        """
        Get item with Lomb-Scargle features added.
        
        Returns:
            tuple: (features, target, id, length, (time_raw, mag_raw, err_raw))
        """
        # Get base features from dataset
        features, target, id_val = self.base_dataset[idx]
        
        # Get raw data for LS calculation
        (time_raw, mag_raw, err_raw), _, _, _ = self.base_dataset.get_raw_item(idx)
        
        # Calculate LS features
        ls_features = self._get_ls_features(time_raw, mag_raw, err_raw)
        
        # Return with LS features
        return features, target, id_val, len(features), (time_raw, mag_raw, err_raw)
    
    def _get_ls_features(self, time_raw, mag_raw, err_raw):
        """
        Calculate Lomb-Scargle period features.
        
        Args:
            time_raw: Raw time array
            mag_raw: Raw magnitude array
            err_raw: Raw error array (can be None)
            
        Returns:
            list: LS features [best_period, best_power, ...]
        """
        try:
            # Convert inputs to numpy arrays if needed
            time = np.asarray(time_raw)
            mag = np.asarray(mag_raw)
            err = np.asarray(err_raw) if err_raw is not None else None
            
            # Create LS periodogram
            ls = self.LombScargle(time, mag, dy=err)
            
            # Calculate frequency range
            min_freq = 1/(2*self.max_period)
            max_freq = 1/(self.min_period)
            
            # Compute periodogram
            freq, power = ls.autopower(
                minimum_frequency=min_freq,
                maximum_frequency=max_freq,
                samples_per_peak=10
            )
            
            if len(freq) == 0:
                return [0.0] * self.num_features
            
            # Get best peak
            best_idx = np.argmax(power)
            best_freq = freq[best_idx]
            best_power = power[best_idx]
            
            # Convert to period
            best_period = 1.0 / best_freq if best_freq > 1e-9 else self.max_period
            
            # Basic features
            features = [best_period, best_power]
            
            # Add more features if requested
            if self.num_features > 2:
                # Find secondary peaks
                # Mask out region around the primary peak
                window = len(freq) // 20  # 5% window
                mask = np.ones_like(power, dtype=bool)
                lower_idx = max(0, best_idx - window)
                upper_idx = min(len(freq), best_idx + window)
                mask[lower_idx:upper_idx] = False
                
                # Find secondary peak
                if np.any(mask) and np.any(power[mask]):
                    sec_idx = np.argmax(power[mask])
                    sec_freq = freq[mask][sec_idx]
                    sec_power = power[mask][sec_idx]
                    sec_period = 1.0 / sec_freq if sec_freq > 1e-9 else self.max_period
                    
                    # Add secondary peak features
                    features.extend([sec_period, sec_power])
                else:
                    # No valid secondary peak
                    features.extend([0.0, 0.0])
            
            # Truncate or pad to desired feature count
            features = features[:self.num_features]
            while len(features) < self.num_features:
                features.append(0.0)
            
            return features
        
        except Exception as e:
            print(f"LS feature calculation error: {e}")
            return [0.0] * self.num_features 