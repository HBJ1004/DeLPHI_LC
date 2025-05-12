# Light Curve Analysis Pipeline - Colab Script
# This script is designed to run the asteroid lightcurve pipeline in Google Colab

#@title Mount Google Drive
from google.colab import drive
drive.mount('/content/drive',force_remount=True)

#@title Install Required Dependencies
# This block MUST run before any lc_pipeline imports
print("Installing and configuring dependencies...")

# Install PyTorch with CUDA support
!pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

# Install required dependencies from requirements.txt
!pip install -r "/content/drive/MyDrive/Colab Notebooks/colab_requirements.txt"

# Add project directory to Python path
import os
import sys

# Search for the project directory
project_candidates = [
    "/content/drive/MyDrive/Colab Notebooks/asteroid_lightcurve_pipeline",
    "/content/drive/MyDrive/asteroid_lightcurve_pipeline",
    "/content/drive/MyDrive/Colab Notebooks/lc_pipeline",
    "/content/drive/MyDrive/lc_pipeline",
    "/content/drive/MyDrive/Colab Notebooks"
]

project_dir = None
for path in project_candidates:
    if os.path.exists(path) and os.path.exists(os.path.join(path, "lc_pipeline")):
        project_dir = path
        print(f"[Initial Setup] Found project directory: {project_dir}")
        break

if project_dir:
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)
        print(f"[Initial Setup] Added '{project_dir}' to sys.path for 'lc_pipeline' import.")
else:
    print("[Initial Setup] Could not find project directory containing 'lc_pipeline'. Pipeline might not run correctly if PROJECT_DIR is essential and remains None.")

# Import our compatibility layer and run setup
from lc_pipeline.colab_setup import setup_colab

# Run the setup with automatic environment patching
setup_colab()

# The following is adapted from colab_ready/colab_run.py
# starting from its import section and modified to use the 'project_dir' found above.

#@title Set Up Full Environment & Global Configurations
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from datetime import datetime
import gc
import shutil
import logging # logging module is built-in
import psutil
import pickle
import glob
import yaml
import traceback # Added for detailed error logging
import torch.multiprocessing as mp # Import multiprocessing

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__) # Main logger for this combined script

# Use the project_dir found by the initial setup from the original colab_run.py preamble
PROJECT_DIR = project_dir # project_dir is from the preamble of this script
if PROJECT_DIR is None:
    logger.error("CRITICAL: project_dir (and thus PROJECT_DIR) was not found or set by the initial setup. Many subsequent operations will likely fail. Please ensure the project path is correctly identified in the 'Search for the project directory' section.")
    # Optionally, raise an error or exit:
    # raise ValueError("PROJECT_DIR is not set. Cannot continue.")
else:
    logger.info(f"Using dynamically determined PROJECT_DIR for pipeline operations: {PROJECT_DIR}")

def verify_environment():
    """Verify the environment is correctly set up before execution"""
    checks = []
    results = {}
    
    # Check Google Drive mount
    drive_dir = "/content/drive"
    drive_mounted = os.path.exists(drive_dir) and os.path.isdir(drive_dir)
    checks.append(drive_mounted)
    results["drive_mounted"] = drive_mounted
    logger.info(f"Drive mounted: {drive_mounted}")
    
    if not drive_mounted:
        logger.error("Google Drive not mounted - run the mount cell first")
        return False, results
    
    # Check project directory exists
    # PROJECT_DIR is now set from the dynamic project_dir
    if PROJECT_DIR is None: # Add a check here
        project_exists = False
        logger.error("PROJECT_DIR is None, cannot check if project directory exists.")
    else:
        project_exists = os.path.exists(PROJECT_DIR) and os.path.isdir(PROJECT_DIR)
    checks.append(project_exists)
    results["project_exists"] = project_exists
    logger.info(f"Project directory exists: {project_exists} (Path: {PROJECT_DIR})")
    
    # Check CUDA if requested
    # Assuming CUDA is always requested if available for this pipeline
    cuda_available = torch.cuda.is_available()
    checks.append(cuda_available) # Simplified this check slightly
    results["cuda_check"] = cuda_available 
    results["cuda_available"] = cuda_available
    logger.info(f"CUDA available: {cuda_available}")
    
    # Overall check result
    environment_valid = all(checks)
    logger.info(f"Environment verification: {'PASSED' if environment_valid else 'FAILED'}")
    
    return environment_valid, results

# Verify environment before continuing
env_ok, env_results = verify_environment()
if not env_ok:
    logger.error("Environment verification failed! Please check the logs for details.")
    logger.error(f"Verification results: {env_results}")
    logger.warning("Continuing despite environment issues - expect potential failures.")
    # If PROJECT_DIR is None and project_exists is False, this is a critical failure point.

# Import compatibility module and apply fixes
# Note: setup_colab() from lc_pipeline.colab_setup was already called.
# model_compatibility.py might be part of lc_pipeline.colab_setup or offer additional patches.
# The colab_ready script specifically calls apply_all_patches, ensure_directories.
try:
    from lc_pipeline.models.model_compatibility import apply_all_patches, ensure_directories, standardize_device_name
    if PROJECT_DIR: # Only run if PROJECT_DIR is set
        ensure_directories(PROJECT_DIR) # This uses PROJECT_DIR
        logger.info(f"Ensured standard directories exist under {PROJECT_DIR}")
    else:
        logger.warning("PROJECT_DIR is not set, skipping ensure_directories.")
    apply_all_patches()
    logger.info("Applied model compatibility patches successfully.")
except ImportError as e:
    logger.warning(f"Could not import or run parts of lc_pipeline.models.model_compatibility: {str(e)}")
    logger.warning("Some features or compatibility layers may not work correctly.")
except Exception as e:
    logger.error(f"Error during model_compatibility setup: {str(e)}")


# Import main modules from lc_pipeline
try:
    from lc_pipeline.config import load_config # For loading config if not generated by this script
    from lc_pipeline.data.datasets import AsteroidDataset, AxisDataset # Added AxisDataset
    from lc_pipeline.data.collate import generate_axis_data, collate_fn as imported_collate_fn # Added generate_axis_data
    from lc_pipeline.models.period_nets import PeriodLSTMNet, PeriodTransformerNet, PeriodLSTMWithLSPrior # Add other models if used
    from lc_pipeline.models.axis_nets import AxisCNNNet, PhaseAwareTransformerAxis # Add other models if used
    from lc_pipeline.models.utils import direction_vector_to_quaternion # and other model utils
    import lc_pipeline.evaluation as evaluation
    from lc_pipeline.training import train_period_model, train_axis_model # If these are the main training functions
    from lc_pipeline.losses import VMFLoss, GeodesicVMFCombinedLoss # Import specific losses
    # The custom collate_fn is usually defined in data.collate or similar
    # from lc_pipeline.data.collate import collate_fn as imported_collate_fn # Assuming it's here - ALREADY PRESENT

    # Imports for Hyperparameter Optimization
    import optuna 
    from lc_pipeline.training.hyperopt import (
        run_period_optimization,
        run_axis_optimization,
        # update_config_with_best_params # This is used internally by main.py, we'll update config dict directly
    )

    logger.info("Successfully imported core lc_pipeline modules and hyperopt components.")
except ImportError as e:
    logger.error(f"Failed to import one or more required lc_pipeline modules: {str(e)}")
    logger.error("Please check that all dependencies are installed and the lc_pipeline package structure is correct and accessible in sys.path.")
    raise # Re-raise for critical failure

# Set multiprocessing start method for CUDA compatibility with DataLoader workers
try:
    if torch.cuda.is_available():
        mp.set_start_method('spawn', force=True)
        logger.info("Set multiprocessing start method to 'spawn' for CUDA.")
except RuntimeError as e:
    logger.warning(f"Could not set multiprocessing start method to 'spawn': {e}. This might be an issue if num_workers > 0 with CUDA.")

# Check CUDA availability and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if 'standardize_device_name' in globals() or 'standardize_device_name' in locals():
    try:
        device = standardize_device_name(str(device)) # standardize_device_name expects a string
        logger.info(f"Using standardized device: {device}")
    except Exception as e:
        logger.warning(f"Failed to standardize device name, using raw device: {device}. Error: {e}")
else:
    logger.info(f"Using device: {device} (standardize_device_name not found/run).")


# --- Dataset Parameters (defaults, overridden by config.yaml if loaded) ---
MAX_FILES = 100
TRAIN_VAL_RATIO = 0.8

# --- Model Parameters (defaults, overridden by config.yaml if loaded) ---
# These are illustrative; actual values will come from config.
DEFAULT_PERIOD_MODEL_PARAMS = {
    "model_name": "PeriodLSTMWithLSPrior", "input_dim": 17, "hidden_dim": 128,
    "num_layers": 3, "dropout": 0.2, "batch_size": 64, "epochs": 10, "lr": 0.001
}
DEFAULT_AXIS_MODEL_PARAMS = {
    "model_name": "AxisCNNNet", "input_features":1, "blocks": 3, "hidden_dim": 128, "dropout": 0.2,
    "batch_size": 64, "epochs": 20, "lr": 0.001, "num_bins": 100, "use_quaternions": True
}

# --- Paths (derived from PROJECT_DIR) ---
if PROJECT_DIR:
    DAMIT_PATH = os.path.join(PROJECT_DIR, "DAMIT_csv")
    MODELS_DIR = os.path.join(PROJECT_DIR, "models")
    RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
    FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")
    LOGS_DIR = os.path.join(PROJECT_DIR, "logs")
    CONFIG_FILE_PATH = os.path.join(PROJECT_DIR, "config.yaml") # Default config file path
else:
    logger.error("PROJECT_DIR is not set. Cannot define data and output paths. Pipeline will likely fail.")
    # Define them as None or relative to CWD to avoid NameErrors, but they won't be correct.
    DAMIT_PATH, MODELS_DIR, RESULTS_DIR, FIGURES_DIR, LOGS_DIR, CONFIG_FILE_PATH = [None]*6

# Ensure directories exist if PROJECT_DIR was set
if PROJECT_DIR:
    for dir_path in [MODELS_DIR, RESULTS_DIR, FIGURES_DIR, LOGS_DIR]:
        if dir_path: os.makedirs(dir_path, exist_ok=True)
    logger.info(f"Ensured output directories exist under {PROJECT_DIR}")

# --- Helper Functions (from colab_ready) ---
def ensure_model_on_device(model, target_device):
    """Ensure model is on the correct device"""
    if not hasattr(model, 'parameters'):
        logger.warning("ensure_model_on_device: input is not a PyTorch model.")
        return model
    try:
        current_device = next(model.parameters()).device
        if str(current_device) != str(target_device):
            logger.info(f"Moving model from {current_device} to {target_device}")
            return model.to(target_device)
    except StopIteration:
        logger.warning("ensure_model_on_device: Model has no parameters.")
    except Exception as e:
        logger.error(f"Error moving model to device {target_device}: {e}")
    return model

def ensure_tensor_on_device(tensor, target_device):
    """Ensure tensor is on the specified device"""
    if isinstance(tensor, torch.Tensor) and tensor.device != torch.device(target_device):
        return tensor.to(target_device)
    return tensor

#@title Load Config and Refine Logger
# Config path defined above using PROJECT_DIR
config = {}
try:
    if CONFIG_FILE_PATH and os.path.exists(CONFIG_FILE_PATH):
        with open(CONFIG_FILE_PATH, 'r') as file:
            config = yaml.safe_load(file)
        logger.info(f"Loaded configuration from {CONFIG_FILE_PATH}")
    else:
        logger.warning(f"Config file not found at {CONFIG_FILE_PATH} (or path is None). Using default parameters where applicable and hoping for the best.")
        # Populate with some defaults if needed, or rely on pipeline's internal defaults.
        # This script will now primarily use config object; ensure it has necessary top-level keys.
        config = { # Basic structure
            "data": {"train_val_ratio": TRAIN_VAL_RATIO, "val_ratio": 0.15, "test_ratio": 0.15, "normalize": True, "seed": 42},
            "period_model": DEFAULT_PERIOD_MODEL_PARAMS.copy(),
            "axis_model": DEFAULT_AXIS_MODEL_PARAMS.copy(),
            "pipeline": {"phases_to_run": ["period_train", "period_eval", "axis_train", "axis_eval"]},
            "seed": 42 # Top level seed
        }
        if PROJECT_DIR and CONFIG_FILE_PATH: # Try to save a default if we have a path
             os.makedirs(os.path.dirname(CONFIG_FILE_PATH), exist_ok=True)
             with open(CONFIG_FILE_PATH, 'w') as file:
                 yaml.dump(config, file)
             logger.info(f"Created a default configuration at {CONFIG_FILE_PATH}")

except Exception as e:
    logger.error(f"Error loading or creating default config: {e}. Proceeding with an empty/default config.")
    config = {"seed": 42} # Minimal config

# Ensure seed exists
if 'seed' not in config: config['seed'] = 42
torch.manual_seed(config['seed'])
np.random.seed(config['seed'])
logger.info(f"Set random seed to {config['seed']}")

# Refine logging based on potential config settings (e.g., log_level from config)
# The basicConfig is already set. For file logging specific to this run:
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Always use a local log path for stability during the run in Colab
local_log_dir = "/content/logs"
os.makedirs(local_log_dir, exist_ok=True)
log_filename = f"pipeline_colab_{timestamp}.log"
log_filepath = os.path.join(local_log_dir, log_filename)

# Remove existing root handlers if any were added by basicConfig
# and add our specific file handler + console handler
root_logger = logging.getLogger() # Get the root logger
for hdlr in root_logger.handlers[:]: # Iterate over a copy
    root_logger.removeHandler(hdlr) # Remove all handlers

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# File Handler (to local Colab path)
file_handler = logging.FileHandler(log_filepath)
log_level_from_config = config.get("logging", {}).get("level", "INFO").upper()
file_handler.setLevel(getattr(logging, log_level_from_config, logging.INFO))
file_handler.setFormatter(formatter)
root_logger.addHandler(file_handler) # Add the new file handler

# Console Handler
console_handler = logging.StreamHandler() # Defaults to sys.stderr
console_handler.setLevel(getattr(logging, log_level_from_config, logging.INFO))
console_handler.setFormatter(formatter)
root_logger.addHandler(console_handler) # Add the new console handler

root_logger.setLevel(logging.DEBUG) # Root logger captures all, handlers filter
logger.info(f"Detailed logging to: {log_filepath} (local Colab path)")
logger.info(f"Original LOGS_DIR on Drive: {LOGS_DIR if LOGS_DIR else 'Not defined'}")
print(f"[DEBUG PRINT] Log file for this run will be written to (local Colab path): {log_filepath}") # Explicit print


#@title Load and Preprocess Data
# System memory check (from colab_ready)
try:
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    logger.info(f"Available system memory: {available_memory_gb:.2f} GB")
    # Further logic to adjust batch sizes based on memory can be added here if needed,
    # referencing and modifying the 'config' object.
except Exception as e:
    logger.warning(f"Could not check system memory: {e}")

# Load data using AsteroidDataset
logger.info(f"Attempting to load data from DAMIT_PATH: {DAMIT_PATH}")
if not DAMIT_PATH or not os.path.exists(DAMIT_PATH):
    logger.error(f"DAMIT_PATH ('{DAMIT_PATH}') is not valid. Cannot load data.")
    # Consider raising an error or exiting
    raise FileNotFoundError(f"DAMIT_PATH ('{DAMIT_PATH}') not found or not set.")

csv_files = glob.glob(f"{DAMIT_PATH}/*.csv")
if not csv_files:
    logger.error(f"No CSV files found in {DAMIT_PATH}. Cannot proceed with data loading.")
    raise FileNotFoundError(f"No CSV files in {DAMIT_PATH}.")

logger.info(f"Found {len(csv_files)} CSV files in {DAMIT_PATH}")

# Dataset configuration
dataset_config = config.get('data', {})
max_seq_len = dataset_config.get('max_sequence_length', 200) # from AsteroidDataset defaults
use_cache_dataset = dataset_config.get('use_disk_caching', True) # from AsteroidDataset defaults for 'use_cache'

try:
    # Determine cache_dir based on use_disk_caching config
    # use_cache_dataset was previously defined as: dataset_config.get('use_disk_caching', True)
    use_disk_caching = dataset_config.get('use_disk_caching', True)
    actual_cache_dir = None
    if use_disk_caching:
        # Default matches AsteroidDataset's constructor if not in config
        cache_dir_from_config = dataset_config.get('cache_dir', "data_cache/asteroid_dataset")
        if PROJECT_DIR and not os.path.isabs(cache_dir_from_config):
            actual_cache_dir = os.path.join(PROJECT_DIR, cache_dir_from_config)
        else:
            actual_cache_dir = cache_dir_from_config
    
    # Get max_files from config or global MAX_FILES (defined earlier in the script)
    # MAX_FILES is set at line 191 in the original script
    current_max_files = dataset_config.get('max_damit_files', MAX_FILES) # Use max_damit_files from config
    logger.info(f"Determined current_max_files: {current_max_files} (from 'max_damit_files' in config or script default MAX_FILES={MAX_FILES})")

    dataset = AsteroidDataset(
        csv_files=csv_files,  # Pass the full list from glob
        config=dataset_config,  # For cache key generation
        max_sequence_length=max_seq_len,
        logger=logger,
        max_files=current_max_files, # Let AsteroidDataset handle slicing based on this
        num_workers=0, # Force to 0 to ensure no multiprocessing during AsteroidDataset init
        use_single_file_processing_cache=dataset_config.get('use_single_file_processing_cache', True), # Default from AsteroidDataset
        cache_dir=actual_cache_dir,
        force_recache=dataset_config.get('force_recache', False) # Default from AsteroidDataset
    )
    logger.info(f"AsteroidDataset instantiation successful. Number of items: {len(dataset)}. Preprocessing workers forced to 0.")

except Exception as e:
    logger.error(f"Error loading AsteroidDataset: {e}")
    logger.error(traceback.format_exc()) # Add traceback for more details
    raise

# Data splitting
train_share = dataset_config.get('train_val_ratio', 0.7) # Using train_val_ratio as main training share
val_share = dataset_config.get('val_ratio', 0.15)
# test_share is implicitly 1 - train_share - val_share

if 'train_val_ratio' in dataset_config:
    logger.info(f"Using train_val_ratio from config.yaml's data section: {dataset_config['train_val_ratio']}")
elif 'data' in config and isinstance(config['data'], dict) and 'train_val_ratio' in config['data']:
    logger.info(f"Using train_val_ratio from config.yaml's data section (via config object): {config['data']['train_val_ratio']}")
else:
    logger.info(f"train_val_ratio not in config.yaml data section, using script default: 0.7 (global TRAIN_VAL_RATIO was {TRAIN_VAL_RATIO})")

if 'val_ratio' in dataset_config:
    logger.info(f"Using val_ratio from config.yaml's data section: {dataset_config['val_ratio']}")
elif 'data' in config and isinstance(config['data'], dict) and 'val_ratio' in config['data']:
    logger.info(f"Using val_ratio from config.yaml's data section (via config object): {config['data']['val_ratio']}")
else:
    logger.info(f"val_ratio not in config.yaml data section, using script default: 0.15")


# Ensure ratios sum to <= 1
if train_share + val_share > 1.0:
    logger.warning(f"Train ({train_share}) + Val ({val_share}) ratios > 1. Adjusting val_share.")
    val_share = 1.0 - train_share
    if val_share < 0: # if train_share was > 1
        train_share = 0.8
        val_share = 0.1 # fallback
logger.info(f"Data split ratios: Train_Dev={train_share}, Validation={val_share}")


num_total = len(dataset)
train_size = int(train_share * num_total)
val_size = int(val_share * num_total)
test_size = num_total - train_size - val_size

if test_size < 0: # If rounding caused issues or val_share was too high
    test_size = 0
    val_size = num_total - train_size # Adjust val_size
    if val_size < 0: # If train_size was too high
        train_size = int(0.8 * num_total) # Fallback
        val_size = num_total - train_size


train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size]
)
logger.info(f"Data split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

if len(test_dataset) == 0 and num_total > 0:
    logger.critical("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    logger.critical("CRITICAL: Test dataset has 0 samples. This will lead to NaN evaluation metrics.")
    logger.critical(f"This is because train_share ({train_share:.2f}) + val_share ({val_share:.2f}) sums to {(train_share + val_share):.2f}, leaving no data for the test set.")
    logger.critical("Please adjust 'train_val_ratio' and 'val_ratio' in your 'config.yaml' under the 'data' section "
                    "so their sum is less than 1.0 (e.g., train_val_ratio: 0.7, val_ratio: 0.15 for a 0.15 test share).")
    logger.critical("Alternatively, ensure your config.yaml is loaded correctly and these values are set as intended.")
    logger.critical("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


# Define our_collate_fn_colab_version first as it's used by the wrappers
def our_collate_fn_colab_version(batch, strip_metadata=False, target_device_for_conversion=None):
    # This is the complex collate function from colab_ready/colab_run.py (lines 378-518)
    # It handles padding, quaternion conversion, etc.
    # Ensure 'direction_vector_to_quaternion' is imported.
    # target_device_for_conversion is added to ensure tensors are on the right device before stacking for quaternions.
    # Default to CPU if not specified for safety during collation.
    target_device_for_conversion = target_device_for_conversion or torch.device("cpu")
    try:
        # logger.debug(f"Collating batch with {len(batch)} items. Strip: {strip_metadata}")
        if not batch: # Early exit if batch is empty
            logger.warning("Empty batch received by collate_fn")
            empty_tensor = torch.zeros(0, dtype=torch.float32)
            return (empty_tensor, empty_tensor) if strip_metadata else (empty_tensor, empty_tensor, [], torch.tensor([]))

        valid_items = [item for item in batch if item is not None and isinstance(item, (tuple, list)) and len(item) >= 2]
        if not valid_items:
            logger.warning("No valid items in batch after filtering")
            empty_tensor = torch.zeros(0, dtype=torch.float32)
            return (empty_tensor, empty_tensor) if strip_metadata else (empty_tensor, empty_tensor, [], torch.tensor([]))

        features_list = []
        targets_raw_list = []
        ids_list = []
        raw_data_lengths = []


        for item in valid_items:
            feature_data = item[0]
            target_data = item[1]
            id_data = item[2] if len(item) > 2 else -1 # Default ID

            if not isinstance(feature_data, torch.Tensor): feature_data = torch.tensor(feature_data, dtype=torch.float32)
            if not isinstance(target_data, torch.Tensor): target_data = torch.tensor(target_data, dtype=torch.float32)
            
            features_list.append(feature_data)
            targets_raw_list.append(target_data)
            ids_list.append(id_data)
            raw_data_lengths.append(feature_data.shape[0])

        lengths_tensor = torch.tensor(raw_data_lengths, dtype=torch.int32)
        max_len = lengths_tensor.max().item() if lengths_tensor.numel() > 0 else 0
        
        batch_size = len(valid_items)
        # Determine feature dimension dynamically
        feat_dim = features_list[0].shape[1] if features_list and features_list[0].dim() > 1 else 1
        is_1d_feature = features_list[0].dim() == 1 if features_list else False

        if max_len == 0 and batch_size > 0 : # Batch of empty features
             padded_data = torch.zeros((batch_size, 0) if is_1d_feature else (batch_size, 0, feat_dim), dtype=torch.float32)
        elif batch_size == 0 : # No valid items
             padded_data = torch.zeros((0,0) if is_1d_feature else (0,0,feat_dim), dtype=torch.float32)
        else:
            if is_1d_feature: # Features are [seq_len]
                 padded_data = torch.zeros((batch_size, max_len), dtype=features_list[0].dtype)
            else: # Features are [seq_len, feat_dim]
                 padded_data = torch.zeros((batch_size, max_len, feat_dim), dtype=features_list[0].dtype)

            for i, seq_tensor in enumerate(features_list):
                current_len = seq_tensor.shape[0]
                if current_len > 0:
                    if is_1d_feature:
                        padded_data[i, :current_len] = seq_tensor
                    else:
                        padded_data[i, :current_len, :] = seq_tensor
        
        # Process targets
        if strip_metadata: # Typically for AxisNet, expecting (l,b) to convert to quaternion
            # Ensure targets are (N, 2) for l,b before conversion, or (N,3) if already vectors
            # This part from colab_ready was specific.
            if targets_raw_list and targets_raw_list[0].dim() > 0 and targets_raw_list[0].shape[0] >= 2:
                stacked_targets = torch.stack(targets_raw_list).to(target_device_for_conversion) # Move to device for trig ops
                if stacked_targets.shape[1] == 2: # Assume (l,b) degrees
                    true_l_deg, true_b_deg = stacked_targets[:, 0], stacked_targets[:, 1]
                    l_rad, b_rad = torch.deg2rad(true_l_deg), torch.deg2rad(true_b_deg)
                    x = torch.cos(b_rad) * torch.cos(l_rad)
                    y = torch.cos(b_rad) * torch.sin(l_rad)
                    z = torch.sin(b_rad)
                    direction_vectors = torch.stack([x, y, z], dim=1)
                elif stacked_targets.shape[1] == 3: # Assume already direction vectors
                    direction_vectors = stacked_targets
                else: # Fallback, should not happen with correct data
                    logger.warning(f"Unexpected target shape for quaternion conversion: {stacked_targets.shape}. Using identity quaternions.")
                    quaternions = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * batch_size, device=target_device_for_conversion)
                    return padded_data.to(target_device_for_conversion), quaternions # Ensure padded_data is on same device

                quaternions_list = []
                for vec in direction_vectors:
                    try:
                        # direction_vector_to_quaternion needs to be available
                        quat = direction_vector_to_quaternion(vec) # vec is already on target_device_for_conversion
                        quaternions_list.append(quat)
                    except Exception as e_quat:
                        logger.error(f"Error converting vector to quaternion: {e_quat}. Using identity.")
                        quaternions_list.append(torch.tensor([1.0, 0.0, 0.0, 0.0], device=target_device_for_conversion))
                final_targets = torch.stack(quaternions_list) if quaternions_list else torch.zeros((batch_size, 4), device=target_device_for_conversion)
                return padded_data.to(target_device_for_conversion), final_targets
            else: # Targets not suitable for quaternion conversion, or empty
                 logger.warning("Targets in strip_metadata mode are not suitable for quaternion conversion or empty. Returning raw stacked targets or zeros.")
                 final_targets = torch.stack(targets_raw_list).to(target_device_for_conversion) if targets_raw_list else torch.zeros((batch_size, 4), device=target_device_for_conversion) # Default to 4 for quaternion if empty
            return padded_data.to(target_device_for_conversion), final_targets
        else: # For PeriodNet or when metadata is needed
            try:
                targets_tensor = torch.stack(targets_raw_list)
            except RuntimeError as e_stack: # Handle varying target sizes if not already uniform
                if any(t.shape != targets_raw_list[0].shape for t in targets_raw_list):
                    logger.warning(f"Targets have varying shapes, cannot stack directly: {e_stack}. Returning as list.")
                    targets_tensor = targets_raw_list # Return as list of tensors
                else: # Other stacking error
                    raise
            # Ensure padded_data is on the correct device before returning
            return padded_data.to(target_device_for_conversion), targets_tensor, ids_list, lengths_tensor

    except Exception as e_collate:
        # Enhanced error reporting for worker crashes
        error_msg = f"Major error in collate function (our_collate_fn_colab_version): {e_collate}"
        logger.error(error_msg)
        print(f"!!! WORKER COLLATE ERROR !!!\\n{error_msg}", file=sys.stderr)
        import traceback
        print("--- Traceback from collate_fn worker ---", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        print("--- End Traceback from collate_fn worker ---", file=sys.stderr)
        sys.stderr.flush() # Ensure it gets printed

        # Fallback to empty tensors
        empty_tensor = torch.zeros(0, dtype=torch.float32)
        ids_empty, lengths_empty = [], torch.tensor([])
        return (empty_tensor, empty_tensor) if strip_metadata else (empty_tensor, empty_tensor, ids_empty, lengths_empty)


# Define wrapper collate functions at a scope where 'device' is accessible
# or pass it explicitly if these were moved to a separate module.
# For now, assume 'device' is globally accessible in this script context when these are called.
def collate_fn_period_wrapper(batch):
    return our_collate_fn_colab_version(batch, strip_metadata=False, target_device_for_conversion=device)

def collate_fn_axis_wrapper(batch):
    return our_collate_fn_colab_version(batch, strip_metadata=True, target_device_for_conversion=device)


# DataLoaders
dataloader_params = config.get('dataloader', {"num_workers": 2, "pin_memory": torch.cuda.is_available()})

# Define base_num_workers based on config or a default.
# This will be used for CPU, or a different value (4) will be used for CUDA.
base_num_workers_config = dataloader_params.get('num_workers', 2) # Default to 2 if not in config's dataloader section
logger.info(f"Base num_workers from config/default: {base_num_workers_config}")

# Re-enable num_workers > 0 as requested.
# Use base_num_workers_config for CPU, and a fixed value (4) for CUDA as often recommended for Colab.
# num_workers_loader = base_num_workers_config if str(device) == 'cpu' else 4
# Given OOM with 4 workers on T4 High RAM, let's cap CUDA workers at 2 or base_num_workers_config, whichever is lower.
# num_workers_loader = base_num_workers_config if str(device) == 'cpu' else min(base_num_workers_config, 2)
# Further reducing to 1 worker for CUDA due to persistent OOM/worker crashes.
# num_workers_loader = base_num_workers_config if str(device) == 'cpu' else 1
# Trying 1 worker for DataLoader on CUDA again, after ensuring AsteroidDataset.num_workers is 0.
# num_workers_loader = base_num_workers_config if str(device) == 'cpu' else 1
# Reverting to 0 workers as all attempts with >0 workers on CUDA have failed.
num_workers_loader = 0

logger.info(f"Setting num_workers_loader to: {num_workers_loader} (reverted to 0 due to persistent worker crashes even with 1 worker on CUDA)")

# If device is CUDA and collate_fn moves to CUDA, pin_memory should be False.
# Our collate_fn (our_collate_fn_colab_version) uses target_device_for_conversion=device.
pin_memory_loader = dataloader_params.get('pin_memory', True) if str(device) == 'cpu' else False
logger.info(f"DataLoader pin_memory set to: {pin_memory_loader} (False if device is CUDA, as collate_fn handles device placement).")

# Period Model DataLoaders
period_batch_size = config.get('period_model', {}).get('batch_size', DEFAULT_PERIOD_MODEL_PARAMS['batch_size'])
train_loader_period = DataLoader(train_dataset, batch_size=period_batch_size, shuffle=True,
                                 collate_fn=collate_fn_period_wrapper, # Use new wrapper
                                 num_workers=num_workers_loader, pin_memory=pin_memory_loader, persistent_workers=num_workers_loader > 0)
val_loader_period = DataLoader(val_dataset, batch_size=period_batch_size, shuffle=False,
                               collate_fn=collate_fn_period_wrapper, # Use new wrapper
                               num_workers=num_workers_loader, pin_memory=pin_memory_loader, persistent_workers=num_workers_loader > 0)
test_loader_period = DataLoader(test_dataset, batch_size=period_batch_size, shuffle=False,
                                collate_fn=collate_fn_period_wrapper, # Use new wrapper
                                num_workers=num_workers_loader, pin_memory=pin_memory_loader, persistent_workers=num_workers_loader > 0)
logger.info(f"Period DataLoaders created. Batch size: {period_batch_size}, Num workers: {num_workers_loader}")

# Axis Model DataLoaders (use strip_metadata=True)
axis_batch_size = config.get('axis_model', {}).get('batch_size', DEFAULT_AXIS_MODEL_PARAMS['batch_size'])
train_loader_axis = DataLoader(train_dataset, batch_size=axis_batch_size, shuffle=True,
                               collate_fn=collate_fn_axis_wrapper, # Use new wrapper
                               num_workers=num_workers_loader, pin_memory=pin_memory_loader, persistent_workers=num_workers_loader > 0)
val_loader_axis = DataLoader(val_dataset, batch_size=axis_batch_size, shuffle=False,
                             collate_fn=collate_fn_axis_wrapper, # Use new wrapper
                             num_workers=num_workers_loader, pin_memory=pin_memory_loader, persistent_workers=num_workers_loader > 0)
test_loader_axis = DataLoader(test_dataset, batch_size=axis_batch_size, shuffle=False,
                              collate_fn=collate_fn_axis_wrapper, # Use new wrapper
                              num_workers=num_workers_loader, pin_memory=pin_memory_loader, persistent_workers=num_workers_loader > 0)
logger.info(f"Axis DataLoaders created. Batch size: {axis_batch_size}, Num workers: {num_workers_loader}")


# --- HYPERPARAMETER OPTIMIZATION ---
best_period_params_from_hyperopt = None
best_axis_params_from_hyperopt = None
SKIP_MAIN_TRAINING = False # Default to False

if config.get('run_hyperopt', False):
    logger.info("Hyperparameter optimization phase started based on config.run_hyperopt=True.")
    
    # --- Period Model Hyperparameter Optimization ---
    run_period_hyperopt_config = config.get('hyperopt', {}).get('run_period_hyperopt', True) # More granular control
    if run_period_hyperopt_config:
        logger.info("Running hyperparameter optimization for Period Model...")
        try:
            # Ensure config passed to run_period_optimization has the necessary sub-fields like
            # config.period_model.optuna_trials, config.period_model.optuna_epochs, and *_range fields
            # The function run_period_optimization is expected to handle its own Optuna study creation.
            
            # Retrieve necessary params from config for run_period_optimization
            # These are used by the objective function called within run_period_optimization
            num_optuna_trials_period = config.get('period_model', {}).get('optuna_trials', 20) # Default from PeriodModelConfig
            # optuna_epochs_period = config.get('period_model', {}).get('optuna_epochs', 10) # Used by objective

            current_best_params_period = run_period_optimization(
                config=config, 
                train_loader=train_loader_period,
                val_loader=val_loader_period,
                device=device,
                logger=logger
                # n_trials is implicitly handled by run_period_optimization using config.period_model.optuna_trials
            )
            if current_best_params_period:
                logger.info(f"Best period model hyperparameters from Optuna: {current_best_params_period}")
                best_period_params_from_hyperopt = current_best_params_period
                
                logger.info("Updating 'config.period_model' with best hyperparameters from Optuna for subsequent operations.")
                for param_name, param_value in current_best_params_period.items():
                    config['period_model'][param_name] = param_value # Update the main config dict
                
                if MODELS_DIR: # Save the updated config section or full config
                    config_save_path = os.path.join(MODELS_DIR, f"best_period_config_colab_hyperopt_{timestamp}.yaml")
                    try:
                        # Save only the period_model part or the whole config
                        with open(config_save_path, 'w') as f_conf_save:
                            yaml.dump({'period_model': config['period_model']}, f_conf_save)
                        logger.info(f"Saved best period parameters to {config_save_path}")
                    except Exception as e_conf_save:
                        logger.error(f"Could not save updated period model config: {e_conf_save}")
            else:
                logger.warning("Period model hyperparameter optimization did not return best parameters.")
        except Exception as e_period_hyperopt:
            logger.error(f"Error during period model hyperparameter optimization: {e_period_hyperopt}", exc_info=True)
    else:
        logger.info("Skipping period model hyperparameter optimization based on config.hyperopt.run_period_hyperopt.")

    # --- Axis Model Hyperparameter Optimization ---
    # Depends on a period_model. We need to decide which period_model to use:
    # 1. The one just trained if main period training ran before hyperopt block (not current flow).
    # 2. A freshly created one using default or Optuna-optimized period_model params.
    # For this integration, let's create/load a period model before axis hyperopt.
    # It should use the potentially updated period_model config.

    run_axis_hyperopt_config = config.get('hyperopt', {}).get('run_axis_hyperopt', True)
    if run_axis_hyperopt_config:
        logger.info("Preparing for Axis Model Hyperparameter Optimization...")
        # Create/Load a period model instance to be used by objective_axis
        # This model should reflect the latest period_model config (possibly updated by period hyperopt)
        temp_period_model_config_for_axis_hyperopt = config.get('period_model', DEFAULT_PERIOD_MODEL_PARAMS.copy())
        logger.info(f"Creating a temporary period model ({temp_period_model_config_for_axis_hyperopt.get('model_name')}) for axis hyperopt using current config.")
        
        # Re-use model creation logic for period_model
        temp_period_model_name = temp_period_model_config_for_axis_hyperopt.get('model_name', 'PeriodLSTMWithLSPrior')
        temp_common_period_params = {
            "input_dim": temp_period_model_config_for_axis_hyperopt.get('input_dim', 17),
            "hidden_dim": temp_period_model_config_for_axis_hyperopt.get('hidden_dim', 128),
            "num_layers": temp_period_model_config_for_axis_hyperopt.get('num_layers', 3),
            "dropout": temp_period_model_config_for_axis_hyperopt.get('dropout', 0.2),
        }
        if temp_period_model_name in ['transformer', 'PeriodTransformerNet']:
            temp_period_model_arch = PeriodTransformerNet(**temp_common_period_params, num_heads=temp_period_model_config_for_axis_hyperopt.get('num_heads', 4))
        elif temp_period_model_name in ['lstm', 'PeriodLSTMNet']:
            temp_period_model_arch = PeriodLSTMNet(**temp_common_period_params)
        elif temp_period_model_name in ['lstm_with_ls', 'PeriodLSTMWithLSPrior']:
            temp_period_model_arch = PeriodLSTMWithLSPrior(**temp_common_period_params, prior_weight=temp_period_model_config_for_axis_hyperopt.get('prior_weight', 0.5))
        else:
            logger.warning(f"Unknown temp period model type for axis hyperopt: {temp_period_model_name}. Defaulting to PeriodLSTMNet.")
            temp_period_model_arch = PeriodLSTMNet(**temp_common_period_params)
        
        period_model_for_axis_hyperopt = ensure_model_on_device(temp_period_model_arch, device)
        
        # Optionally, train this temporary period model for a few epochs if not using a pre-trained one
        # For now, assume objective_axis can use an initialized (or pre-loaded from a path) period model.
        # If it needs to be trained, that logic would go here or be part of objective_axis.
        # The `main.py` workflow implies a period model is available. If colab_run might not have trained one yet,
        # we might need to load the "final_period_model_path" if it exists from a previous step, or train one.
        # For now, let's assume it can use an initialized one.
        # It might be better if `objective_axis` can take a path to a period model or trains one briefly.
        # Given `main.py` structure, `period_model` passed to `run_axis_optimization` is likely a trained one.
        # In `colab_run.py`, `period_model` (the main one) might not be trained yet if hyperopt runs first.
        # Safest is to use the model trained in the main period training block if it ran.
        # If hyperopt runs *before* main training, this temp model should be trained.
        
        # Let's use 'period_model' if it was defined and trained by the main training block,
        # otherwise the temp_period_model (which is only initialized).
        # This part is tricky due to execution order. For now, objective_axis in hyperopt.py
        # might just use it for inference, so an initialized model might be okay if it loads weights.
        # The run_axis_optimization in `lc_pipeline.training.hyperopt` takes `period_model` as arg.

        # If `period_model` from main training block is available, use it. Otherwise, use the temp initialized one.
        # This requires `period_model` to be defined before this hyperopt block, which it is if period training ran.
        # This script's flow: Data -> HyperOpt (uses existing dataloaders) -> MainTrain -> MainEval
        # So, when hyperopt runs, `period_model` is not yet the fully trained main one.
        # We must provide a capable period_model to run_axis_optimization.
        # Let's assume `period_model_for_axis_hyperopt` (initialized with best period params if avail) is passed.
        # If it needs to be trained, `objective_axis` itself or `run_axis_optimization` should handle it.
        # The current `objective_axis` in `hyperopt.py` expects a `period_model` for inference.
        # It does NOT train the passed `period_model`. So, `period_model_for_axis_hyperopt` should be trained.

        # Simplification for now: if `period_model` (the main one for the script) exists, use it.
        # Otherwise, this indicates a logic flaw if axis_hyperopt relies on a trained one.
        # The `early_pipeline_debug.log` showed `main.py` trains a default period model *before* axis hyperopt.
        # We should replicate that if `period_model` isn't already trained by a previous Colab cell.
        # For this merged script, if run_period_training is True, period_model will be trained *after* this block.
        # So, we MUST train a temporary period model here for axis hyperopt.

        logger.info("Training a temporary period model for axis hyperparameter optimization...")
        # Use a shorter number of epochs for this temporary model
        temp_epochs = config.get('period_model', {}).get('optuna_epochs', 10) # Use optuna_epochs as a proxy
        train_period_model( # This will train period_model_for_axis_hyperopt
            model=period_model_for_axis_hyperopt,
            train_loader=train_loader_period,
            val_loader=val_loader_period,
            device=device,
            num_epochs=temp_epochs, # Short training
            learning_rate=temp_period_model_config_for_axis_hyperopt.get('lr', 0.001),
            weight_decay=temp_period_model_config_for_axis_hyperopt.get('weight_decay', 1e-5),
            patience=5, logger=logger, checkpoint_path=None, debug=True
        )
        logger.info("Temporary period model training for axis hyperopt complete.")


        logger.info("Running hyperparameter optimization for Axis Model...")
        try:
            current_best_params_axis = run_axis_optimization(
                config=config,
                train_loader=train_loader_axis, # These are from AsteroidDataset
                val_loader=val_loader_axis,   # Objective_axis makes AxisDataset from these
                period_model=period_model_for_axis_hyperopt, # Use the just-trained temporary period model
                device=device,
                logger=logger
            )
            if current_best_params_axis:
                logger.info(f"Best axis model hyperparameters from Optuna: {current_best_params_axis}")
                best_axis_params_from_hyperopt = current_best_params_axis
                
                logger.info("Updating 'config.axis_model' with best hyperparameters from Optuna.")
                for param_name, param_value in current_best_params_axis.items():
                     config['axis_model'][param_name] = param_value
                
                if MODELS_DIR:
                    config_save_path = os.path.join(MODELS_DIR, f"best_axis_config_colab_hyperopt_{timestamp}.yaml")
                    try:
                        with open(config_save_path, 'w') as f_conf_save:
                            yaml.dump({'axis_model': config['axis_model']}, f_conf_save)
                        logger.info(f"Saved best axis parameters to {config_save_path}")
                    except Exception as e_conf_save:
                        logger.error(f"Could not save updated axis model config: {e_conf_save}")
            else:
                logger.warning("Axis model hyperparameter optimization did not return best parameters.")
        except Exception as e_axis_hyperopt:
            logger.error(f"Error during axis model hyperparameter optimization: {e_axis_hyperopt}", exc_info=True)
    else:
        logger.info("Skipping axis model hyperparameter optimization based on config.hyperopt.run_axis_hyperopt.")

    # Determine if main training should be skipped
    # Default train_after_hyperopt to True if 'hyperopt' section or the key itself is missing
    hyperopt_config_section = config.get('hyperopt', {}) # Get the hyperopt section, or empty dict
    train_after_hyperopt = hyperopt_config_section.get('train_after_hyperopt', True) 

    if not train_after_hyperopt:
        logger.info("Configuration 'hyperopt.train_after_hyperopt' is False. Main training and evaluation phases will be skipped.")
        SKIP_MAIN_TRAINING = True
    else:
        logger.info("Proceeding with main training/evaluation phases using potentially updated config from hyperopt.")
        SKIP_MAIN_TRAINING = False
else:
    logger.info("run_hyperopt is False in config. Skipping hyperparameter optimization phase.")
    SKIP_MAIN_TRAINING = False


#@title Train Period Model
# This section will now use config['period_model'] which might have been updated by Optuna.
if not SKIP_MAIN_TRAINING:
    period_model_config = config.get('period_model', DEFAULT_PERIOD_MODEL_PARAMS.copy()) # Use copy to avoid modifying default
    # Update with Optuna results if they were better and we want to use them
    # This is already done above by directly modifying config['period_model']
    logger.info(f"Final period_model_config for main training: {period_model_config}")

    logger.info(f"Initializing period model for main training: {period_model_config.get('model_name')}")

    # Select and initialize period model
    # (Code from colab_ready lines 631-656, adapted for config structure)
    # Ensure these models are imported: PeriodLSTMNet, PeriodTransformerNet, PeriodLSTMWithLSPrior
    period_model_name = period_model_config.get('model_name', 'PeriodLSTMWithLSPrior')
    common_period_params = {
        "input_dim": period_model_config.get('input_dim', 17),
        "hidden_dim": period_model_config.get('hidden_dim', 128),
        "num_layers": period_model_config.get('num_layers', 3),
        "dropout": period_model_config.get('dropout', 0.2),
    }
    if period_model_name in ['transformer', 'PeriodTransformerNet']:
        period_model_arch = PeriodTransformerNet(**common_period_params, num_heads=period_model_config.get('num_heads', 4))
    elif period_model_name in ['lstm', 'PeriodLSTMNet']:
        period_model_arch = PeriodLSTMNet(**common_period_params)
    elif period_model_name in ['lstm_with_ls', 'PeriodLSTMWithLSPrior']:
        period_model_arch = PeriodLSTMWithLSPrior(**common_period_params, prior_weight=period_model_config.get('prior_weight', 0.5))
    else:
        logger.error(f"Unknown period model type: {period_model_name}. Defaulting to PeriodLSTMNet.")
        period_model_arch = PeriodLSTMNet(**common_period_params)
    period_model = ensure_model_on_device(period_model_arch, device)

    # Optimizer for period model
    optimizer_period = torch.optim.Adam(period_model.parameters(), lr=period_model_config.get('lr', 0.001), weight_decay=period_model_config.get('weight_decay', 1e-5))

    logger.info("Starting main period model training...")
    # train_period_model is imported from lc_pipeline.training
    # Ensure train_period_model's signature matches these arguments
    period_checkpoint_dir = os.path.join(MODELS_DIR, "period_checkpoints") if MODELS_DIR else None
    if period_checkpoint_dir: os.makedirs(period_checkpoint_dir, exist_ok=True)

    period_training_history = train_period_model(
        model=period_model,
        train_loader=train_loader_period,
        val_loader=val_loader_period,
        device=device,
        num_epochs=period_model_config.get('epochs', 10),
        learning_rate=period_model_config.get('lr', 0.001),
        weight_decay=period_model_config.get('weight_decay', 1e-5),
        patience=period_model_config.get('early_stopping_patience', 10),
        logger=logger,
        checkpoint_path=period_checkpoint_dir, # Renamed from checkpoint_dir
        debug=config.get('debug_mode', False) # Added debug, sourcing from main config or defaulting to False
    )

    final_period_model_path = None
    if MODELS_DIR:
        final_period_model_path = os.path.join(MODELS_DIR, f"period_model_final_{timestamp}.pt")
        torch.save(period_model.state_dict(), final_period_model_path)
        logger.info(f"Saved final period model to {final_period_model_path}")
else:
    logger.info("Skipping main period model training as per hyperopt configuration.")
    period_model = None # Ensure period_model is None if not trained
    final_period_model_path = None
    period_training_history = {}


#@title Train Axis Model
if not SKIP_MAIN_TRAINING:
    if period_model is None: # Check if period model was trained or loaded
        logger.warning("Period model is not available (likely skipped training). Axis model training cannot proceed without it.")
    else:
        axis_model_config = config.get('axis_model', DEFAULT_AXIS_MODEL_PARAMS.copy())
        # Update with Optuna results if available
        # This is already done above by directly modifying config['axis_model']
        logger.info(f"Final axis_model_config for main training: {axis_model_config}")

        logger.info(f"Initializing axis model for main training: {axis_model_config.get('model_name')}")

        # Select and initialize axis model
        # (Code from colab_ready lines 721-745, adapted)
        # Ensure AxisCNNNet, PhaseAwareTransformerAxis are imported
        axis_model_name = axis_model_config.get('model_name', 'AxisCNNNet')
        common_axis_params = {
            "num_bins": axis_model_config.get('num_bins', config.get('data',{}).get('num_phase_bins',100)),
            "input_features": axis_model_config.get('input_features', 1), # Default to 1 if not specified
            "hidden_dim": axis_model_config['hidden_dim'],
            "dropout": axis_model_config.get('dropout', 0.2),
            "use_norm": axis_model_config.get('use_norm', True),
            "use_quaternions": axis_model_config.get('use_quaternions', True)
        }
        if axis_model_name in ['cnn', 'AxisCNNNet']:
            axis_model_arch = AxisCNNNet(**common_axis_params,
                                         blocks=axis_model_config.get('blocks', 3),
                                         initial_channels=axis_model_config.get('initial_channels', 32),
                                         kernel=axis_model_config.get('kernel_size', 5) # kernel_size changed to kernel, logger removed
                                        )
        elif axis_model_name in ['transformer', 'PhaseAwareTransformerAxis']:
            axis_model_arch = PhaseAwareTransformerAxis(**common_axis_params,
                                                        num_layers=axis_model_config.get('num_layers', 3),
                                                        num_heads=axis_model_config.get('num_heads', 4),
                                                        output_dim = 5 if common_axis_params["use_quaternions"] else 3
                                                        )
        else:
            logger.error(f"Unknown axis model type: {axis_model_name}. Defaulting to AxisCNNNet.")
            # Fallback default, ensure params match AxisCNNNet
            common_axis_params["input_features"] = axis_model_config.get('input_features', 1)
            axis_model_arch = AxisCNNNet(**common_axis_params, blocks=3, initial_channels=32, kernel=5, output_dim = 5 if common_axis_params["use_quaternions"] else 3)

        axis_model = ensure_model_on_device(axis_model_arch, device)

        # Optimizer for axis model
        optimizer_axis = torch.optim.Adam(axis_model.parameters(), lr=axis_model_config.get('lr', 0.001), weight_decay=axis_model_config.get('weight_decay', 1e-5))

        # Determine loss function for Axis Model
        # Based on colab_ready logic (lines 806-822)
        axis_criterion = None
        try:
            # Temporarily set model to eval mode for test inference if it's not None
            if axis_model: axis_model.eval()
            sample_batch_data, _ = next(iter(train_loader_axis)) # Get a sample batch
            sample_batch_data = ensure_tensor_on_device(sample_batch_data, device)
            
            if sample_batch_data.numel() > 0 and axis_model:
                with torch.no_grad():
                    sample_output = axis_model(sample_batch_data)
                output_dim = sample_output.shape[-1] # Usually shape is [batch, output_dim]
                logger.info(f"Axis model sample output dimension: {output_dim}")

                if axis_model_config.get('use_quaternions', True):
                    # output_dim 5 for quat+kappa, 4 for just quat.
                    # GeodesicVMFCombinedLoss handles both (kappa optional in its forward)
                    logger.info("Using GeodesicVMFCombinedLoss for quaternion-based axis model.")
                    axis_criterion = GeodesicVMFCombinedLoss(
                        geodesic_weight=axis_model_config.get('loss_geodesic_weight', 0.7),
                        vmf_weight=axis_model_config.get('loss_vmf_weight', 0.3)
                    )
                else: # Direction vector output (typically 3D)
                    logger.info("Using VMFLoss for direction vector-based axis model.")
                    axis_criterion = VMFLoss() # VMFLoss might expect (preds, targets, kappas)
                                          # if model outputs kappas for direction vector.
                                          # If model outputs only 3D vector, VMFLoss needs to handle that.
            else: # Fallback if no sample data or model
                logger.warning("Could not determine axis model output dimension. Defaulting loss function.")
                if axis_model_config.get('use_quaternions', True): axis_criterion = GeodesicVMFCombinedLoss()
                else: axis_criterion = VMFLoss()
            if axis_model: axis_model.train() # Set back to train mode

        except Exception as e_loss_select:
            logger.error(f"Error selecting axis loss function: {e_loss_select}. Defaulting based on config.")
            if axis_model_config.get('use_quaternions', True): axis_criterion = GeodesicVMFCombinedLoss()
            else: axis_criterion = VMFLoss()
        logger.info(f"Axis model criterion: {type(axis_criterion).__name__}")


        logger.info("Starting main axis model training...")
        axis_checkpoint_dir = os.path.join(MODELS_DIR, "axis_checkpoints") if MODELS_DIR else None
        axis_checkpoint_file_path = None
        if axis_checkpoint_dir:
            os.makedirs(axis_checkpoint_dir, exist_ok=True)
            axis_checkpoint_file_path = os.path.join(axis_checkpoint_dir, "best_axis_model_checkpoint.pt")

        # train_axis_model is imported from lc_pipeline.training
        axis_training_history = train_axis_model(
            model=axis_model,
            train_loader=train_loader_axis,
            val_loader=val_loader_axis,
            device=device,
            epochs=axis_model_config.get('epochs', 20),
            lr=axis_model_config.get('lr', 0.001),
            weight_decay=axis_model_config.get('weight_decay', 1e-5),
            patience=axis_model_config.get('early_stopping_patience', 10),
            logger=logger,
            checkpoint_path=axis_checkpoint_file_path # Pass the full file path
        )

        final_axis_model_path = None
        if MODELS_DIR:
            final_axis_model_path = os.path.join(MODELS_DIR, f"axis_model_final_{timestamp}.pt")
            torch.save(axis_model.state_dict(), final_axis_model_path)
            logger.info(f"Saved final axis model to {final_axis_model_path}")
else:
    logger.info("Skipping main axis model training as per hyperopt configuration.")
    axis_model = None # Ensure axis_model is None if not trained
    final_axis_model_path = None
    axis_training_history = {}


#@title Evaluate Models
if not SKIP_MAIN_TRAINING: # Or, if evaluation should always run if models exist from hyperopt.
                           # For now, let's tie it to SKIP_MAIN_TRAINING.
    if period_model and test_loader_period: # Check if model and loader exist
        logger.info("Evaluating Period Model on Test Set...")
        # Ensure evaluate_period_model signature matches
        period_eval_config = config.get('evaluation', {}).get('period_model', {})
        # period_model_config is already defined (e.g., config.get('period_model', DEFAULT_PERIOD_MODEL_PARAMS))

        # Extract parameters for evaluate_period_model from period_model_config and period_eval_config
        # Defaults from evaluate_period_model definition used if not in configs
        period_scale_factor_eval = period_model_config.get('period_scale_factor', 50.0)
        use_log_scale_eval = period_model_config.get('use_log_scale', False)
        min_period_eval = period_model_config.get('min_period', 2.0)
        max_period_eval = period_model_config.get('max_period', 100.0)

        return_predictions_eval = True # Force True to ensure metrics are calculated
        mc_dropout_samples_eval = period_eval_config.get('mc_dropout_samples', 0)

        logger.debug(f"Period evaluation params: scale_factor={period_scale_factor_eval}, log_scale={use_log_scale_eval}, "
                     f"min_period={min_period_eval}, max_period={max_period_eval}, "
                     f"return_predictions={return_predictions_eval}, mc_samples={mc_dropout_samples_eval}")

        period_test_results = evaluation.evaluate_period_model(
            model=period_model,
            dataloader=test_loader_period,
            device=device,
            logger=logger,
            period_scale_factor=period_scale_factor_eval,
            use_log_scale=use_log_scale_eval,
            min_period=min_period_eval,
            max_period=max_period_eval,
            return_predictions=return_predictions_eval,
            mc_dropout_samples=mc_dropout_samples_eval
        )
        print("Period Model Test Results:")
        for metric, value in period_test_results.items(): print(f"{metric}: {value}")
    else:
        logger.info("Skipping Period Model evaluation (model or dataloader not available).")
        period_test_results = {} # Empty results


    if axis_model and test_loader_axis: # Check if model and loader exist
        logger.info("Evaluating Axis Model on Test Set...")
        axis_eval_config = config.get('evaluation', {}).get('axis_model', {})
        # axis_model_config is already defined

        # Extract parameters for evaluate_axis_model (if it also changes)
        # For now, assume evaluate_axis_model still takes config and eval_config as before
        # If it changes, similar modifications will be needed.

        # Check signature of evaluate_axis_model: expects return_predictions, mc_dropout_samples
        return_predictions_axis_eval = True # Force True to ensure metrics are calculated
        mc_dropout_samples_axis_eval = axis_eval_config.get('mc_dropout_samples', 0)

        logger.debug(f"Axis evaluation params: return_predictions={return_predictions_axis_eval}, mc_samples={mc_dropout_samples_axis_eval}")

        axis_test_results = evaluation.evaluate_axis_model(
            model=axis_model,
            dataloader=test_loader_axis,
            device=device,
            logger=logger,
            # config=axis_model_config, # Assuming this might also change or be unused
            # eval_config=axis_eval_config # Assuming this might also change or be unused
            return_predictions=return_predictions_axis_eval,
            mc_dropout_samples=mc_dropout_samples_axis_eval
        )
        print("Axis Model Test Results:")
        for metric, value in axis_test_results.items(): print(f"{metric}: {value}")
    else:
        logger.info("Skipping Axis Model evaluation (model or dataloader not available).")
        axis_test_results = {} # Empty results
else:
    logger.info("Skipping model evaluation as per hyperopt configuration.")
    period_test_results = {}
    axis_test_results = {}


#@title Generate Visualizations
# Visualizations might still be useful if hyperopt ran and models were created,
# even if main training was skipped. Let's make this conditional on models existing.
if final_period_model_path or final_axis_model_path: # Check if any model was saved
    if FIGURES_DIR:
        try:
            logger.info("Generating training history plots...")
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            if period_training_history and 'train_loss' in period_training_history and 'val_loss' in period_training_history:
                plt.plot(period_training_history['train_loss'], label='Train Loss')
                plt.plot(period_training_history['val_loss'], label='Val Loss')
            plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Period Model Training'); plt.legend()

            plt.subplot(1, 2, 2)
            if axis_training_history and 'train_loss' in axis_training_history and 'val_loss' in axis_training_history:
                plt.plot(axis_training_history['train_loss'], label='Train Loss')
                plt.plot(axis_training_history['val_loss'], label='Val Loss')
            plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Axis Model Training'); plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(FIGURES_DIR, f"training_history_{timestamp}.png"))
            plt.show()
            plt.close() # Close figure

            # Add more advanced visualizations from lc_pipeline.visualization if they exist
            # e.g., evaluation.plot_period_results(period_test_results, period_training_history, FIGURES_DIR, timestamp)
            # evaluation.plot_axis_results(axis_test_results, axis_training_history, FIGURES_DIR, timestamp)

            # Example: Plot sample raw lightcurves (if test_dataset allows easy raw access)
            # This part needs careful adaptation based on how AsteroidDataset provides raw items.
            # The colab_ready version (lines 1074-1092) is complex.
            # Assuming a simplified version for now, or rely on dedicated plotting functions from your pipeline.
            logger.info("Skipping complex raw lightcurve plotting from colab_ready for now. Implement using lc_pipeline.visualization if needed.")

        except Exception as e_viz:
            logger.error(f"Error during visualization generation: {e_viz}")
    else:
        logger.warning("FIGURES_DIR not defined. Skipping visualization saving.")


#@title Save Results Summary
# (Code from colab_ready lines 1241-1347, for make_serializable and saving)
# This requires json
import json # Ensure json is imported here if not earlier

results_summary_data = {
    "run_timestamp": timestamp,
    "project_dir": PROJECT_DIR,
    "config_used": make_serializable_colab_version(config), # Serialize config as well
    "period_model_results": {
        "best_hyperopt_params": make_serializable_colab_version(best_period_params_from_hyperopt),
        "model_path": final_period_model_path,
        "training_history": make_serializable_colab_version(period_training_history),
        "test_metrics": make_serializable_colab_version(period_test_results)
    },
    "axis_model_results": {
        "best_hyperopt_params": make_serializable_colab_version(best_axis_params_from_hyperopt),
        "model_path": final_axis_model_path,
        "training_history": make_serializable_colab_version(axis_training_history),
        "test_metrics": make_serializable_colab_version(axis_test_results)
    },
    "run_flags": { # Added a section for run flags
        "run_hyperopt_enabled": config.get('run_hyperopt', False),
        "run_period_hyperopt_setting": config.get('hyperopt', {}).get('run_period_hyperopt', True),
        "run_axis_hyperopt_setting": config.get('hyperopt', {}).get('run_axis_hyperopt', True),
        "train_after_hyperopt_setting": config.get('hyperopt', {}).get('train_after_hyperopt', True),
        "main_training_skipped_due_to_hyperopt_config": SKIP_MAIN_TRAINING
    }
}

if RESULTS_DIR:
    # No need to call make_serializable_colab_version on the entire results_summary_data 
    # if its components are already made serializable during its construction.
    # However, the function is robust, so calling it on the whole dict is also fine.
    # For clarity, let's ensure components are serialized as above.
    summary_path = os.path.join(RESULTS_DIR, f"run_summary_{timestamp}.json")
    try:
        with open(summary_path, 'w') as f:
            json.dump(results_summary_data, f, indent=4) # Directly dump the dict with serialized components
        logger.info(f"Saved run summary to {summary_path}")
    except Exception as e_json:
        logger.error(f"Error saving JSON summary: {e_json}")
else:
    logger.warning("RESULTS_DIR not defined. Skipping saving of run summary JSON.")


#@title End of Pipeline Execution
logger.info("Main pipeline execution script (colab_run.py adapted from colab_ready) has completed.")
if 'log_filepath' in locals(): # log_filepath is now always local
    logger.info(f"Primary log file for this run (local): {log_filepath}")
    if LOGS_DIR: # If user's LOGS_DIR on Drive is defined, suggest copying
        try:
            final_log_destination = os.path.join(LOGS_DIR, os.path.basename(log_filepath))
            shutil.copy(log_filepath, final_log_destination)
            logger.info(f"Copied local log file to Drive: {final_log_destination}")
        except Exception as e_copy_log:
            logger.error(f"Failed to copy log file from {log_filepath} to {LOGS_DIR}: {e_copy_log}")
    else:
        logger.info(f"LOGS_DIR not defined on Drive. Local log at: {log_filepath}")

print("======================================================================")
print("✅ Asteroid Lightcurve Pipeline (Colab Adapted Script) Execution Finished.")
print(f"Timestamp: {timestamp}")
if PROJECT_DIR:
    print(f"Project Directory: {PROJECT_DIR}")
    if MODELS_DIR: print(f"Models saved in: {MODELS_DIR}")
    if RESULTS_DIR: print(f"Results/Summaries saved in: {RESULTS_DIR}")
    if FIGURES_DIR: print(f"Figures saved in: {FIGURES_DIR}")
    if LOGS_DIR and 'log_filepath' in locals(): 
        print(f"Logs (local copy at {local_log_dir}, attempted copy to Drive): {LOGS_DIR}/{os.path.basename(log_filepath) if LOGS_DIR else log_filepath}")
    else: # If LOGS_DIR was not defined, just show the local path
        print(f"Logs (local Colab path only): {log_filepath}")
print("======================================================================")

# Checkpointing and PipelineTracker from colab_ready (lines 1451 onwards)
# These are more advanced features for resumable pipelines; include them for completeness

def save_run_checkpoint(stage_name_str, data_dict, checkpoint_base_dir=None):
    if checkpoint_base_dir is None: checkpoint_base_dir = MODELS_DIR # Default
    if not checkpoint_base_dir:
        logger.warning("Cannot save checkpoint, base directory not defined.")
        return None
    
    os.makedirs(checkpoint_base_dir, exist_ok=True)
    chkpt_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_dict['checkpoint_timestamp'] = chkpt_timestamp
    data_dict['checkpoint_stage'] = stage_name_str
    
    file_path = os.path.join(checkpoint_base_dir, f"run_checkpoint_{stage_name_str}_{chkpt_timestamp}.pkl")
    try:
        with open(file_path, 'wb') as f: pickle.dump(data_dict, f)
        logger.info(f"Saved run checkpoint for stage '{stage_name_str}' to {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Error saving run checkpoint: {e}")
        return None

# Example usage of checkpointing could be:
# save_run_checkpoint("period_training_complete", {"model_path": final_period_model_path, "history": period_training_history})
# save_run_checkpoint("axis_training_complete", {"model_path": final_axis_model_path, "history": axis_training_history})

# logger.info("colab_run.py script finished.") # This line was duplicated, removing one instance.

# --- Helper function for JSON serialization (from lc_pipeline/main.py) ---
def make_serializable_colab_version(obj):
    """
    Recursively convert objects to JSON serializable types with performance optimizations.
    Needed for saving results_summary_data.
    Args:
        obj: Object to convert
    Returns:
        JSON serializable version of the object
    """
    obj_type = type(obj)
    if obj is None or obj_type in (str, int, float, bool):
        return obj
    if obj_type is dict:
        return {str(k): make_serializable_colab_version(v) for k, v in obj.items()} # Ensure keys are strings
    if obj_type in (list, tuple, set): # Added set here
        return [make_serializable_colab_version(elem) for elem in obj]
    
    # Using string checks for numpy/torch can be brittle if module names change
    # but often works for scripts. isinstance is safer if modules are always imported.
    # Assuming numpy and torch are imported as np and torch respectively.
    if 'numpy' in str(obj_type) or isinstance(obj, np.number) or isinstance(obj, np.ndarray):
        # import numpy as np # Numpy should be imported globally already
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.number):
            return obj.item() # Converts numpy numbers to python native types
        return str(obj) # Fallback for other numpy types
            
    if 'torch' in str(obj_type) and hasattr(obj, 'detach'):
        # import torch # Torch should be imported globally already
        try:
            return obj.detach().cpu().numpy().tolist()
        except Exception:
            return str(obj) # Fallback for torch types
            
    if hasattr(obj, '__dict__'):
        try:
            return {str(k): make_serializable_colab_version(v) for k, v in obj.__dict__.items() 
                    if not k.startswith('_') and not callable(v)}
        except Exception:
            return str(obj)
            
    if hasattr(obj, 'to_dict') and callable(obj.to_dict):
        try:
            return make_serializable_colab_version(obj.to_dict())
        except Exception:
            return str(obj)
    
    # Final fallback for any other types
    try:
        return str(obj)
    except Exception:
        return f"<unserializable object of type {obj_type}>"

# logger.info("colab_run.py script finished.") # This line was duplicated, removing one instance.
