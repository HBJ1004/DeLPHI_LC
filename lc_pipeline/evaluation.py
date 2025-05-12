import os
import torch
import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- Debugging Utilities ---
def log_tensor_stats(tensor, name="tensor", logger=None):
    if logger is None:
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

def check_tensor_validity(tensor, name="tensor", logger=None):
    if logger is None:
        logger = logging.getLogger(__name__)
    has_nan = torch.isnan(tensor).any() if hasattr(tensor, 'isnan') else False
    has_inf = torch.isinf(tensor).any() if hasattr(tensor, 'isinf') else False
    if has_nan or has_inf:
        logger.warning(f"{name} contains NaN: {has_nan} or Inf: {has_inf}")
        return False
    return True

def activate_dropout_layers(model: torch.nn.Module):
    """Activates dropout layers in a model for MC Dropout inference."""
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train() # Activates this specific dropout layer

def evaluate_period_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    period_scale_factor: float = 50.0,
    use_log_scale: bool = False,
    min_period: float = 2.0,
    max_period: float = 100.0,
    return_predictions: bool = False,
    logger: Any = None,
    mc_dropout_samples: int = 0  # Number of MC Dropout samples for uncertainty
) -> Dict[str, Any]:
    """
    Evaluate a period prediction model.
    
    Args:
        model: Trained period prediction model
        dataloader: DataLoader for evaluation
        device: Device to use for evaluation
        period_scale_factor: Scale factor for periods (for non-log-scale models)
        use_log_scale: Whether the model uses log-scale output
        min_period: Minimum period value (for log-scale models)
        max_period: Maximum period value (for log-scale models)
        return_predictions: Whether to return predictions and targets
        logger: Logger instance
        mc_dropout_samples: Number of MC Dropout samples. If > 0, MC Dropout is performed.
    
    Returns:
        Dictionary of evaluation metrics, including 'mc_std_dev' if mc_dropout_samples > 0.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    model = model.to(device)
    # model.eval() will be called inside the loop if not doing MC, or before activating dropout for MC
    
    all_predictions = [] # For return_predictions=True, stores final (mean) predictions
    all_mc_std_devs = [] # For return_predictions=True, stores MC std devs
    targets_collected = []
    ids_collected = []
    
    if use_log_scale:
        from .models.period_nets import LogScaleTransformer
        transformer = LogScaleTransformer(min_period, max_period)
    
    # No torch.no_grad() here if we need dropout active for MC samples
    # It will be applied per-sample if not doing MC, or after MC loop

    for batch_idx, batch in enumerate(dataloader):
        # --- DEBUG: Log batch input stats ---
        if len(batch) == 4:
                data, target_full, batch_ids, lengths = batch
        elif len(batch) == 5:
                data, target_full, batch_ids, lengths, _ = batch
        else:
            logger.warning(f"Unexpected batch format with {len(batch)} elements in batch {batch_idx}. Skipping.")
            continue
        log_tensor_stats(data, f"period_batch_data[{batch_idx}]", logger)
        log_tensor_stats(target_full, f"period_batch_target[{batch_idx}]", logger)
        if not check_tensor_validity(data, f"period_batch_data[{batch_idx}]", logger):
            logger.warning(f"NaN/Inf in input data for batch {batch_idx}. Skipping batch.")
            # Need to append NaNs if return_predictions is True to keep alignment
            if return_predictions:
                num_in_batch = data.size(0)
                all_predictions.append(np.full((num_in_batch, 1), np.nan))
                if mc_dropout_samples > 0:
                    all_mc_std_devs.append(np.full((num_in_batch, 1), np.nan))
                targets_collected.append(target_full[:, 2:3].numpy())
                ids_collected.extend(batch_ids)
            continue # Added continue here as it was missing if return_predictions is False
            
        # --- DEBUG: Check device consistency ---
        if hasattr(data, 'device') and model.parameters():
            model_device = next(model.parameters()).device
            if data.device != model_device:
                logger.warning(f"Device mismatch: data on {data.device}, model on {model_device}")
            
            data = data.to(device)
            if isinstance(lengths, torch.Tensor):
                lengths = lengths.to(device)
            else:
                try:
                    lengths = torch.tensor(lengths, dtype=torch.long).to(device)
                except Exception as e:
                    if logger:
                        logger.error(f"Failed to convert lengths to tensor on device {device} for batch {batch_idx}: {e}")
                    continue
        
        target_period_batch = target_full[:, 2:3].numpy() # [batch_size, 1]
        
        batch_predictions_np = None
        batch_mc_std_dev_np = None

        if mc_dropout_samples > 0:
            model.eval() # Overall eval mode
            activate_dropout_layers(model) # Activate dropout layers
            
            mc_preds_for_batch = [] # List to store MC predictions for each item in the batch
            for _ in range(mc_dropout_samples):
                try:
                    output = model(data, lengths)
                    if isinstance(output, tuple):
                        output = output[0]
                    # Ensure output is [batch_size, 1]
                    if len(output.shape) > 1 and output.shape[1] > 1:
                        output = output[:, 0:1]
                    
                    if use_log_scale:
                        pred_linear_sample = transformer.from_log_scale(output.cpu().detach().numpy())
                    else:
                        scale_factor = 1.0 if period_scale_factor is None else period_scale_factor
                        pred_linear_sample = output.cpu().detach().numpy() * scale_factor
                    
                    if len(pred_linear_sample.shape) == 1:
                        pred_linear_sample = pred_linear_sample.reshape(-1, 1)
                    mc_preds_for_batch.append(pred_linear_sample)
                except Exception as e:
                    if logger:
                        logger.error(f"Error during MC sample generation for batch {batch_idx}: {e}")
                    # Append NaNs for this sample if an error occurs
                    mc_preds_for_batch.append(np.full((data.size(0), 1), np.nan))
                    break # Stop MC sampling for this batch if one sample fails
            
            if mc_preds_for_batch:
                mc_preds_tensor = np.stack(mc_preds_for_batch, axis=0) # [num_samples, batch_size, 1]
                batch_predictions_np = np.nanmean(mc_preds_tensor, axis=0) # [batch_size, 1]
                batch_mc_std_dev_np = np.nanstd(mc_preds_tensor, axis=0)   # [batch_size, 1]
            else: # All MC samples failed or no samples run
                batch_predictions_np = np.full((data.size(0), 1), np.nan)
                batch_mc_std_dev_np = np.full((data.size(0), 1), np.nan)

        else: # Standard evaluation (no MC Dropout)
            model.eval()
            with torch.no_grad():
                try:
                    output = model(data, lengths)
                    if isinstance(output, tuple):
                        output = output[0]
                        # Ensure output is [batch_size, 1]
                    if len(output.shape) > 1 and output.shape[1] > 1:
                            output = output[:, 0:1]
                    
                    if use_log_scale:
                        pred_linear = transformer.from_log_scale(output.cpu().numpy())
                    else:
                        scale_factor = 1.0 if period_scale_factor is None else period_scale_factor
                        pred_linear = output.cpu().numpy() * scale_factor
                    
                    # Ensure pred_linear is a 2D numpy array of shape (N, 1)
                    current_preds_np = np.array(pred_linear, copy=False) # Ensure it's an array, avoid copy if already array
                    if current_preds_np.ndim == 0: # Scalar
                        current_preds_np = current_preds_np.reshape(1, 1)
                    elif current_preds_np.ndim == 1: # 1D array [N]
                        current_preds_np = current_preds_np.reshape(-1, 1)
                    # If it's already 2D [N,1], it's fine.
                    # Model output should have been sliced to [batch_size, 1] already if wider.
                    batch_predictions_np = current_preds_np

                except Exception as e:
                    if logger:
                        logger.error(f"Error during standard model evaluation for batch {batch_idx}: {e}")
                    batch_predictions_np = np.full((data.size(0), 1), np.nan)
        # --- DEBUG: Log output stats ---
        log_tensor_stats(output, f"period_model_output[{batch_idx}]", logger)
        if not check_tensor_validity(output, f"period_model_output[{batch_idx}]", logger):
            logger.warning(f"NaN/Inf in model output for batch {batch_idx}")
        
        if return_predictions:
            all_predictions.append(batch_predictions_np)
            if mc_dropout_samples > 0:
                all_mc_std_devs.append(batch_mc_std_dev_np)
            targets_collected.append(target_period_batch)
            ids_collected.extend(batch_ids)
    
    # After loop, ensure model is back in eval mode without activated dropout for safety
    model.eval()

    if not all_predictions and return_predictions: # Only if return_predictions, otherwise this list is empty
        logger.warning("No predictions were made during evaluation (return_predictions=True).")
        # Fallback based on whether we expect mc_std_devs or not
        metrics = {
            'mae': float('nan'), 'rmse': float('nan'),
            'mean_relative_error': float('nan'), 'median_relative_error': float('nan'),
            'predictions': [], 'targets': [], 'ids': []
        }
        if mc_dropout_samples > 0:
            metrics['mc_std_devs'] = []
            metrics['mean_mc_std_dev'] = float('nan')
        return metrics
    
    # If not returning predictions, we need to compute metrics internally based on collected results
    # This part of the original code needs to be adapted to use the batch_predictions_np, etc. that are now generated per batch.
    # For simplicity, if not return_predictions, the original logic path for concatenating `predictions` and `targets` will be used.
    # However, `predictions` and `targets` in the original code were populated inside the loop. 
    # Let's adapt to use `all_predictions` and `targets_collected` if `return_predictions` is True, 
    # or we assume the metrics are calculated based on what `all_predictions` would contain.

    # This logic assumes all_predictions, targets_collected, etc. are the primary source if return_predictions is true
    # If return_predictions is False, metrics will be calculated but not these lists.
    # The original structure had `predictions` list populated. Let's rename `all_predictions` to `final_predictions_list`
    # to avoid confusion with the old `predictions` list used for metric calculation later.

    final_predictions_list = all_predictions # if return_predictions else some_other_list_for_metrics
    final_targets_list = targets_collected   # if return_predictions else ...
    final_ids_list = ids_collected           # if return_predictions else ...
    final_mc_std_devs_list = all_mc_std_devs # if return_predictions and mc_dropout_samples > 0 else ...

    if not final_predictions_list: # Check if any batch succeeded if return_predictions was true
                                  # Or if this path is taken when return_predictions=False and no batches processed for metric arrays
        logger.warning("No valid predictions were made during the evaluation.")
        metrics = {
            'mae': float('nan'), 'rmse': float('nan'),
            'mean_relative_error': float('nan'), 'median_relative_error': float('nan')
        }
        if return_predictions:
            metrics.update({'predictions': [], 'targets': [], 'ids': []})
        if mc_dropout_samples > 0:
            if return_predictions:
                metrics['mc_std_devs'] = []
            metrics['mean_mc_std_dev'] = float('nan')
        return metrics

    predictions_concat = np.concatenate(final_predictions_list)
    targets_concat = np.concatenate(final_targets_list)
    mc_std_devs_concat = np.concatenate(final_mc_std_devs_list) if mc_dropout_samples > 0 and final_mc_std_devs_list else None

    # ... (rest of the filtering and metric calculation logic remains largely the same, 
    #      but will now operate on predictions_concat, targets_concat) ...
    # ... Ensure to also handle mc_std_devs_concat for filtering and potential aggregated metrics ...

    valid_mask = (
        ~np.isnan(predictions_concat).any(axis=1) &
        ~np.isnan(targets_concat).any(axis=1) &
        ~np.isinf(predictions_concat).any(axis=1) &
        ~np.isinf(targets_concat).any(axis=1)
    )
    if mc_std_devs_concat is not None:
        valid_mask &= (~np.isnan(mc_std_devs_concat).any(axis=1) & ~np.isinf(mc_std_devs_concat).any(axis=1))
    
    if not np.any(valid_mask):
        logger.warning("No valid predictions after filtering NaN/Inf values.")
        metrics = {
            'mae': float('nan'), 'rmse': float('nan'),
            'mean_relative_error': float('nan'), 'median_relative_error': float('nan')
        }
        if return_predictions:
            metrics.update({'predictions': [], 'targets': [], 'ids': []})
            if mc_dropout_samples > 0:
                 metrics['mc_std_devs'] = []
        if mc_dropout_samples > 0:
            metrics['mean_mc_std_dev'] = float('nan')
        return metrics
    
    predictions_filtered = predictions_concat[valid_mask]
    targets_filtered = targets_concat[valid_mask]
    ids_filtered = [final_ids_list[i] for i in range(len(final_ids_list)) if i < len(valid_mask) and valid_mask[i]]
    mc_std_devs_filtered = mc_std_devs_concat[valid_mask] if mc_std_devs_concat is not None else None

    if predictions_filtered.shape[0] == 0:
        logger.warning("All predictions were filtered out. No data for metrics.")
        # Same return as above
        metrics = {
            'mae': float('nan'), 'rmse': float('nan'),
            'mean_relative_error': float('nan'), 'median_relative_error': float('nan')
        }
        if return_predictions:
            metrics.update({'predictions': predictions_filtered.tolist() if isinstance(predictions_filtered, np.ndarray) else [], 
                            'targets': targets_filtered.tolist() if isinstance(targets_filtered, np.ndarray) else [], 
                            'ids': ids_filtered})
            if mc_dropout_samples > 0 and mc_std_devs_filtered is not None:
                 metrics['mc_std_devs'] = mc_std_devs_filtered.tolist()
        if mc_dropout_samples > 0:
            metrics['mean_mc_std_dev'] = np.nanmean(mc_std_devs_filtered) if mc_std_devs_filtered is not None and mc_std_devs_filtered.size > 0 else float('nan')
        return metrics
    
    # Ensure shapes match by flattening if needed
    if predictions_filtered.shape != targets_filtered.shape:
        logger.warning(f"Shape mismatch: predictions {predictions_filtered.shape} vs targets {targets_filtered.shape}. Attempting to reconcile.")
        try:
            if predictions_filtered.shape[1] > 1 and targets_filtered.shape[1] == 1:
                predictions_filtered = predictions_filtered[:, 0:1] # Take first column
            elif targets_filtered.shape[1] > 1 and predictions_filtered.shape[1] == 1:
                targets_filtered = targets_filtered[:, 0:1]
            
            if predictions_filtered.shape[0] != targets_filtered.shape[0]: # Mismatch in num samples
                raise ValueError("Number of samples in predictions and targets do not match after filtering.")

            # If still mismatched, try flattening only if one is 1D and other is 2D [N,1]
            if len(predictions_filtered.shape) == 2 and predictions_filtered.shape[1] == 1 and len(targets_filtered.shape) == 1:
                predictions_filtered = predictions_filtered.ravel()
            elif len(targets_filtered.shape) == 2 and targets_filtered.shape[1] == 1 and len(predictions_filtered.shape) == 1:
                targets_filtered = targets_filtered.ravel()
            
            # Final check for compatibility with sklearn metrics
            if predictions_filtered.ndim > 1 and predictions_filtered.shape[1] > 1:
                 predictions_filtered = predictions_filtered[:,0] # Default to first column if still ambiguous
            if targets_filtered.ndim > 1 and targets_filtered.shape[1] > 1:
                 targets_filtered = targets_filtered[:,0]

        except Exception as reshape_e:
            logger.error(f"Could not reconcile shapes for metric calculation: {reshape_e}")
            # Return NaN metrics
            metrics = {
                'mae': float('nan'), 'rmse': float('nan'),
                'mean_relative_error': float('nan'), 'median_relative_error': float('nan')
            }
            if return_predictions:
                metrics.update({'predictions': predictions_filtered.tolist() if isinstance(predictions_filtered, np.ndarray) else [], 
                                'targets': targets_filtered.tolist() if isinstance(targets_filtered, np.ndarray) else [], 
                                'ids': ids_filtered})
                if mc_dropout_samples > 0 and mc_std_devs_filtered is not None:
                    metrics['mc_std_devs'] = mc_std_devs_filtered.tolist()
            if mc_dropout_samples > 0:
                metrics['mean_mc_std_dev'] = np.nanmean(mc_std_devs_filtered) if mc_std_devs_filtered is not None and mc_std_devs_filtered.size > 0 else float('nan')
            return metrics
    
    # Calculate metrics
    try:
        # Ensure inputs are numpy arrays and not empty, although earlier checks should handle emptiness.
        if not (isinstance(targets_filtered, np.ndarray) and targets_filtered.size > 0 and
                isinstance(predictions_filtered, np.ndarray) and predictions_filtered.size > 0):
            logger.warning("Inputs to metric calculation are not valid numpy arrays or are empty. "
                           f"targets_filtered type: {type(targets_filtered)}, size: {getattr(targets_filtered, 'size', 'N/A')}. "
                           f"predictions_filtered type: {type(predictions_filtered)}, size: {getattr(predictions_filtered, 'size', 'N/A')}")
            raise ValueError("Invalid or empty inputs for metric calculation despite earlier checks.")

        logger.debug(f"Calculating metrics. predictions_filtered shape: {predictions_filtered.shape}, dtype: {predictions_filtered.dtype}. "
                     f"targets_filtered shape: {targets_filtered.shape}, dtype: {targets_filtered.dtype}.")
        
        mae = mean_absolute_error(targets_filtered, predictions_filtered)
        rmse = np.sqrt(mean_squared_error(targets_filtered, predictions_filtered))
    
        # Ensure targets_filtered is not zero for division, adding epsilon handles this.
        # Also check for emptiness again before division, though covered by initial check in try.
        if targets_filtered.size > 0: # Redundant if initial check in try is robust, but safe
            relative_errors = np.abs(targets_filtered - predictions_filtered) / (np.abs(targets_filtered) + 1e-9)
            mean_relative_error = np.mean(relative_errors)
            median_relative_error = np.median(relative_errors)
        else: 
            logger.warning("targets_filtered is empty before relative error calculation, this should have been caught by the initial check in the try block.")
            mean_relative_error = float('nan')
            median_relative_error = float('nan')

    except Exception as e_metric_calc:
        logger.error(f"ERROR during metric calculation: {e_metric_calc}")
        # Log details of the arrays that might have caused the issue.
        # Check if they are defined before trying to access shape/dtype.
        pred_shape = getattr(predictions_filtered, 'shape', 'N/A') if 'predictions_filtered' in locals() else 'N/A (undefined)'
        pred_dtype = getattr(predictions_filtered, 'dtype', 'N/A') if 'predictions_filtered' in locals() else 'N/A (undefined)'
        targ_shape = getattr(targets_filtered, 'shape', 'N/A') if 'targets_filtered' in locals() else 'N/A (undefined)'
        targ_dtype = getattr(targets_filtered, 'dtype', 'N/A') if 'targets_filtered' in locals() else 'N/A (undefined)'

        logger.error(f"Data potentially causing error: predictions_filtered (shape {pred_shape}, dtype {pred_dtype}), "
                     f"targets_filtered (shape {targ_shape}, dtype {targ_dtype})")
        
        mae = float('nan')
        rmse = float('nan')
        mean_relative_error = float('nan')
        median_relative_error = float('nan')
        
    results = {
        'mae': mae,
        'rmse': rmse,
        'mean_relative_error': mean_relative_error,
        'median_relative_error': median_relative_error
    }

    if mc_dropout_samples > 0 and mc_std_devs_filtered is not None:
        results['mean_mc_std_dev'] = np.nanmean(mc_std_devs_filtered)
        if return_predictions:
            results['mc_std_devs'] = mc_std_devs_filtered.tolist() # List of std_devs per prediction
        
        if return_predictions:
            results['predictions'] = predictions_filtered.tolist()
            results['targets'] = targets_filtered.tolist()
            results['ids'] = ids_filtered
    
    return results

def calculate_harmonic_success(predictions, targets, threshold=0.05):
    """
    Calculate harmonic success rates for period predictions.
    
    Args:
        predictions: Period predictions
        targets: Ground truth periods
        threshold: Relative error threshold for success
        
    Returns:
        Dictionary of success rates for different harmonics
    """
    # Ensure inputs are numpy arrays
    if isinstance(predictions, list):
        predictions = np.array(predictions)
    if isinstance(targets, list):
        targets = np.array(targets)
        
    # Ensure correct shapes
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
    if targets.ndim == 1:
        targets = targets.reshape(-1, 1)
        
    # Check valid dimensions
    if predictions.shape[0] != targets.shape[0]:
        raise ValueError(f"Predictions and targets have different batch sizes: {predictions.shape[0]} vs {targets.shape[0]}")
        
    # Initialize stats
    harmonic_errors = {'exact': [], 'half': [], 'double': [], 'third': [], 'triple': []}
    best_harmonic_errors = []
    
    for i in range(len(targets)):
        # Extract values safely
        try:
            true_p = float(targets[i, 0])
            pred_p = float(predictions[i, 0])
        except (IndexError, ValueError) as e:
            # Skip problematic entries
            continue
            
        if true_p <= 1e-6:
            continue  # Skip invalid targets
        
        exact_err = np.abs(pred_p - true_p) / true_p
        half_err = np.abs(pred_p - true_p / 2) / true_p
        double_err = np.abs(pred_p - true_p * 2) / true_p
        third_err = np.abs(pred_p - true_p / 3) / true_p
        triple_err = np.abs(pred_p - true_p * 3) / true_p
        
        harmonic_errors['exact'].append(exact_err)
        harmonic_errors['half'].append(half_err)
        harmonic_errors['double'].append(double_err)
        harmonic_errors['third'].append(third_err)
        harmonic_errors['triple'].append(triple_err)
        best_harmonic_errors.append(min(exact_err, half_err, double_err, third_err, triple_err))
    
    # Calculate success rates (within threshold relative error)
    harmonic_success = {k: np.mean(np.array(v) < threshold) if v else 0.0 
                         for k, v in harmonic_errors.items()}
    harmonic_success['any'] = np.mean(np.array(best_harmonic_errors) < threshold) if best_harmonic_errors else 0.0
    
    return harmonic_success

def evaluate_axis_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    return_predictions: bool = False,
    logger: Any = None,
    mc_dropout_samples: int = 0  # Number of MC Dropout samples for uncertainty
) -> Dict[str, Any]:
    """
    Evaluate an axis prediction model.
    
    Args:
        model: Trained axis prediction model
        dataloader: DataLoader for evaluation
        device: Device to use for evaluation
        return_predictions: Whether to return predictions and targets
        logger: Logger instance
        mc_dropout_samples: Number of MC Dropout samples. If > 0, MC Dropout is performed.
    
    Returns:
        Dictionary of evaluation metrics, including MC uncertainty if mc_dropout_samples > 0.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    from .models.utils import quaternion_to_direction_vector # Keep direct import for clarity
    
    model = model.to(device)
    # model.eval() handled per-case below
    
    # For storing results across all batches
    all_final_predictions_dir = [] # Mean direction vector from MC samples or single pred
    all_targets_dir = []
    all_pred_concentrations = [] # From model's direct output (kappa)
    all_mc_uncertainties_axis = [] # e.g., spherical std dev or mean angular spread from MC samples

    # Initial detection of model/target types (can be None initially)
    model_outputs_quaternions = None
    targets_are_quaternions = None

    for batch_idx, (data, target_batch) in enumerate(dataloader):
        # --- DEBUG: Log batch input stats ---
        log_tensor_stats(data, f"axis_batch_data[{batch_idx}]", logger)
        log_tensor_stats(target_batch, f"axis_batch_target[{batch_idx}]", logger)
        if not check_tensor_validity(data, f"axis_batch_data[{batch_idx}]", logger):
            logger.warning(f"NaN/Inf in input data for batch {batch_idx}. Skipping.")
            # Handle appending NaNs if return_predictions is True to maintain alignment
            if return_predictions:
                num_in_batch = data.size(0)
                all_final_predictions_dir.append(np.full((num_in_batch, 3), np.nan))
                all_targets_dir.append(np.full((num_in_batch, 3), np.nan)) # Store original targets later
                if mc_dropout_samples > 0:
                    all_mc_uncertainties_axis.append(np.full((num_in_batch, 1), np.nan))
                all_pred_concentrations.append(np.full((num_in_batch, 1), np.nan))
            continue
        if not check_tensor_validity(target_batch, f"axis_batch_target[{batch_idx}]", logger):
            logger.warning(f"NaN/Inf in target data for batch {batch_idx}. Skipping.")
            # Handle appending NaNs if return_predictions is True to maintain alignment
            if return_predictions:
                num_in_batch = data.size(0)
                all_final_predictions_dir.append(np.full((num_in_batch, 3), np.nan))
                all_targets_dir.append(np.full((num_in_batch, 3), np.nan)) # Store original targets later
                if mc_dropout_samples > 0:
                    all_mc_uncertainties_axis.append(np.full((num_in_batch, 1), np.nan))
                all_pred_concentrations.append(np.full((num_in_batch, 1), np.nan))
            continue

        # --- DEBUG: Check device consistency ---
        if hasattr(data, 'device') and model.parameters():
            model_device = next(model.parameters()).device
            if data.device != model_device:
                logger.warning(f"Device mismatch: data on {data.device}, model on {model_device}")

        data, target_batch_dev = data.to(device), target_batch.to(device)
        
        # --- Determine target format (once per eval run) ---
        if targets_are_quaternions is None:
            targets_are_quaternions = target_batch_dev.shape[1] == 4
            logger.info(f"Detected target format: {'quaternion' if targets_are_quaternions else 'direction vector'}")

        # --- Process Targets --- 
        current_batch_targets_dir_np = None
        if targets_are_quaternions:
            target_dir_batch = quaternion_to_direction_vector(target_batch_dev)
        else:
            target_dir_batch = target_batch_dev
        current_batch_targets_dir_np = torch.nn.functional.normalize(target_dir_batch, p=2, dim=1).cpu().numpy()
        
        batch_mean_pred_dir_np = None
        batch_pred_concentration_np = None
        batch_mc_uncertainty_val = None

        # --- Perform Prediction (Standard or MC Dropout) ---
        if mc_dropout_samples > 0:
            model.eval() # Overall eval mode
            activate_dropout_layers(model) # Activate dropout layers for MC sampling
            
            mc_sample_dirs_for_batch = [] # List of [batch_size, 3] direction vectors from MC
            mc_sample_concentrations_for_batch = [] # List of [batch_size, 1] concentrations from MC

            for i_sample in range(mc_dropout_samples):
                try:
                    output = model(data) # [batch_size, 4 or 5]
                    if torch.isnan(output).any() or torch.isinf(output).any():
                        logger.warning(f"NaN/Inf in MC sample {i_sample} output for batch {batch_idx}. Skipping sample.")
                        # Add NaNs for this sample for all items in batch
                        mc_sample_dirs_for_batch.append(np.full((data.size(0), 3), np.nan))
                        mc_sample_concentrations_for_batch.append(np.full((data.size(0),1), np.nan))
                        continue # Skip to next sample if output is invalid
                
                    # --- Determine model output format (once per eval run, if not already) ---
                    if model_outputs_quaternions is None:
                        model_outputs_quaternions = output.shape[1] == 5 or output.shape[1] == 4
                        logger.info(f"Detected model output format: {'quaternion+kappa' if output.shape[1] == 5 else ('quaternion' if output.shape[1] == 4 else ('direction+kappa' if (not model_outputs_quaternions and output.shape[1] == 4) else 'direction'))}") # Adjusted logger for clarity
                    
                    pred_dir_sample, pred_concentration_sample = None, None
                    if model_outputs_quaternions:
                        pred_quat_sample = output[:, :4]
                        pred_dir_sample = quaternion_to_direction_vector(pred_quat_sample)
                        if output.shape[1] == 5: # Quaternion + log_kappa
                            pred_concentration_sample = torch.exp(output[:, 4:5])
                        # If output.shape[1] == 4 and model_outputs_quaternions is True, it's just quaternion.
                    else: # Direction vector output
                        pred_dir_sample = output[:, :3]
                        if output.shape[1] == 4: # Direction + log_kappa
                            pred_concentration_sample = torch.exp(output[:, 3:4])
                    
                    mc_sample_dirs_for_batch.append(torch.nn.functional.normalize(pred_dir_sample, p=2, dim=1).cpu().detach().numpy())
                    if pred_concentration_sample is not None:
                        mc_sample_concentrations_for_batch.append(pred_concentration_sample.cpu().detach().numpy())
                    else:
                        mc_sample_concentrations_for_batch.append(np.full((data.size(0),1), np.nan))

                except Exception as e_mc_sample:
                    logger.error(f"Error during MC sample {i_sample} for batch {batch_idx}: {e_mc_sample}")
                    mc_sample_dirs_for_batch.append(np.full((data.size(0), 3), np.nan))
                    mc_sample_concentrations_for_batch.append(np.full((data.size(0),1), np.nan))
                    # Optionally break if one sample fails severely, or continue to get as many as possible
            if mc_sample_dirs_for_batch: # if at least one sample was attempted
                mc_sample_dirs_stacked = np.stack(mc_sample_dirs_for_batch, axis=0) # [num_samples, batch_size, 3]
                # Mean direction vector
                batch_mean_pred_dir_np = np.nanmean(mc_sample_dirs_stacked, axis=0) # [batch_size, 3]
                batch_mean_pred_dir_np = batch_mean_pred_dir_np / (np.linalg.norm(batch_mean_pred_dir_np, axis=1, keepdims=True) + 1e-9) # Normalize mean

                # More robust: Mean angular distance to the mean vector
                angular_distances_to_mean = [] # list of arrays, each [batch_size]
                for s_idx in range(mc_sample_dirs_stacked.shape[0]):
                    sample_dirs = mc_sample_dirs_stacked[s_idx, :, :] # [batch_size, 3]
                    # Dot product: sum(A*B, axis=1) -> [batch_size]
                    cos_theta = np.sum(sample_dirs * batch_mean_pred_dir_np, axis=1)
                    cos_theta = np.clip(cos_theta, -1.0, 1.0) # Ensure valid input for arccos
                    angles_rad = np.arccos(cos_theta) # [batch_size]
                    angular_distances_to_mean.append(angles_rad)
                
                if angular_distances_to_mean:
                    angular_distances_stacked = np.stack(angular_distances_to_mean, axis=0) # [num_samples, batch_size]
                    batch_mc_uncertainty_val = np.degrees(np.nanmean(angular_distances_stacked, axis=0)) # Mean angular distance in degrees [batch_size]
                else:
                    batch_mc_uncertainty_val = np.full(data.size(0), np.nan)

                # Mean predicted concentration from MC samples
                if mc_sample_concentrations_for_batch and not all(c is None for c in mc_sample_concentrations_for_batch):
                    mc_concentrations_stacked = np.stack([c for c in mc_sample_concentrations_for_batch if c is not None], axis=0)
                    batch_pred_concentration_np = np.nanmean(mc_concentrations_stacked, axis=0) # [batch_size, 1]
                else:
                    batch_pred_concentration_np = np.full((data.size(0), 1), np.nan)
            else: # All MC samples failed or list empty
                batch_mean_pred_dir_np = np.full((data.size(0), 3), np.nan)
                batch_mc_uncertainty_val = np.full(data.size(0), np.nan)
                batch_pred_concentration_np = np.full((data.size(0), 1), np.nan)
        else: # Standard evaluation (no MC Dropout)
            model.eval()
            with torch.no_grad():
                try:
                    output = model(data)
                    if torch.isnan(output).any() or torch.isinf(output).any():
                        raise ValueError("NaN/Inf in model output during standard eval.")

                    if model_outputs_quaternions is None:
                        model_outputs_quaternions = output.shape[1] == 5 or output.shape[1] == 4
                        logger.info(f"Detected model output format: {'quaternion+kappa' if output.shape[1] == 5 else ('quaternion' if output.shape[1] == 4 else 'direction+kappa')}")

                    pred_dir_single, pred_concentration_single = None, None
                    if model_outputs_quaternions:
                        pred_quat_single = output[:,:4]
                        pred_dir_single = quaternion_to_direction_vector(pred_quat_single)
                        if output.shape[1] == 5:
                            pred_concentration_single = torch.exp(output[:,4:5])
                    else:
                        pred_dir_single = output[:,:3]
                        if output.shape[1] == 4:
                            pred_concentration_single = torch.exp(output[:,3:4])
                    
                    # Robust shaping for batch_mean_pred_dir_np
                    if pred_dir_single is not None:
                        current_pred_dir_np = torch.nn.functional.normalize(pred_dir_single, p=2, dim=1).cpu().numpy()
                        if current_pred_dir_np.ndim == 0: # Should not happen for vectors, but defensive
                            current_pred_dir_np = current_pred_dir_np.reshape(1, -1) # Attempt to make it (1, D)
                        elif current_pred_dir_np.ndim == 1: # e.g. single vector [D] for batch_size=1
                            current_pred_dir_np = current_pred_dir_np.reshape(1, -1) # Make it (1, D)
                        batch_mean_pred_dir_np = current_pred_dir_np
                    else: # Should not happen if model works
                        batch_mean_pred_dir_np = np.full((data.size(0), 3), np.nan)

                    if pred_concentration_single is not None:
                        batch_pred_concentration_np = pred_concentration_single.cpu().numpy()
                        if batch_pred_concentration_np.ndim == 0:
                             batch_pred_concentration_np = batch_pred_concentration_np.reshape(1,1)
                        elif batch_pred_concentration_np.ndim == 1:
                             batch_pred_concentration_np = batch_pred_concentration_np.reshape(-1,1)
                    else:
                        batch_pred_concentration_np = np.full((data.size(0),1), np.nan)

                except Exception as e_std_eval:
                    logger.error(f"Error during standard axis model evaluation for batch {batch_idx}: {e_std_eval}")
                    batch_mean_pred_dir_np = np.full((data.size(0), 3), np.nan)
                    batch_pred_concentration_np = np.full((data.size(0), 1), np.nan)

        # --- DEBUG: Log output stats ---
        log_tensor_stats(output, f"axis_model_output[{batch_idx}]", logger)
        if not check_tensor_validity(output, f"axis_model_output[{batch_idx}]", logger):
            logger.warning(f"NaN/Inf in model output for batch {batch_idx}")

        # Store results for the batch
        all_final_predictions_dir.append(batch_mean_pred_dir_np)
        all_targets_dir.append(current_batch_targets_dir_np)
        all_pred_concentrations.append(batch_pred_concentration_np)
        if mc_dropout_samples > 0:
            # Reshape batch_mc_uncertainty_val if it's 1D [batch_size] to [batch_size, 1]
            if batch_mc_uncertainty_val is not None and batch_mc_uncertainty_val.ndim == 1:
                batch_mc_uncertainty_val = batch_mc_uncertainty_val.reshape(-1, 1)
            all_mc_uncertainties_axis.append(batch_mc_uncertainty_val if batch_mc_uncertainty_val is not None else np.full((data.size(0),1), np.nan))

    # After loop, ensure model is back in strict eval mode
    model.eval()

    if not all_final_predictions_dir:
        logger.warning("No valid axis predictions were made during evaluation.")
        metrics = {'mean_angular_error': float('nan'), 'median_angular_error': float('nan'), 'success_rate_30deg': 0.0}
        if return_predictions:
            metrics.update({'predictions_dir': [], 'targets_dir': [], 'pred_concentrations': []})
        if mc_dropout_samples > 0:
            if return_predictions:
                metrics['mc_uncertainties_axis'] = []
            metrics['mean_mc_angular_uncertainty'] = float('nan')
        return metrics

    # Concatenate all batch results
    final_preds_dir_concat = np.concatenate(all_final_predictions_dir)
    final_targets_dir_concat = np.concatenate(all_targets_dir)
    pred_concentrations_concat = np.concatenate(all_pred_concentrations)
    mc_uncertainties_concat = np.concatenate(all_mc_uncertainties_axis) if mc_dropout_samples > 0 and all_mc_uncertainties_axis else None

    # Filter out NaN/Inf from concatenated results
    valid_mask = ~np.isnan(final_preds_dir_concat).any(axis=1) & \
                 ~np.isinf(final_preds_dir_concat).any(axis=1) & \
                 ~np.isnan(final_targets_dir_concat).any(axis=1) & \
                 ~np.isinf(final_targets_dir_concat).any(axis=1)
    if pred_concentrations_concat is not None:
         valid_mask &= (~np.isnan(pred_concentrations_concat).any(axis=1) & ~np.isinf(pred_concentrations_concat).any(axis=1))
    if mc_uncertainties_concat is not None:
        valid_mask &= (~np.isnan(mc_uncertainties_concat).any(axis=1) & ~np.isinf(mc_uncertainties_concat).any(axis=1))

    if not np.any(valid_mask):
        logger.warning("No valid axis predictions after filtering NaN/Inf.")
        # Same return structure as above for no predictions
        metrics = {'mean_angular_error': float('nan'), 'median_angular_error': float('nan'), 'success_rate_30deg': 0.0}
        if return_predictions:
            metrics.update({'predictions_dir': [], 'targets_dir': [], 'pred_concentrations': []})
        if mc_dropout_samples > 0:
            if return_predictions:
                metrics['mc_uncertainties_axis'] = []
            metrics['mean_mc_angular_uncertainty'] = float('nan')
        return metrics

    preds_filt = final_preds_dir_concat[valid_mask]
    targets_filt = final_targets_dir_concat[valid_mask]
    concentrations_filt = pred_concentrations_concat[valid_mask] if pred_concentrations_concat is not None else None
    mc_uncertainties_filt = mc_uncertainties_concat[valid_mask] if mc_uncertainties_concat is not None else None

    if preds_filt.shape[0] == 0:
        logger.warning("All axis predictions filtered out. No data for metrics.")
        # Same return structure
        metrics = {'mean_angular_error': float('nan'), 'median_angular_error': float('nan'), 'success_rate_30deg': 0.0}
        if return_predictions:
            metrics.update({'predictions_dir': [], 'targets_dir': [], 'pred_concentrations': []})
        if mc_dropout_samples > 0:
            if return_predictions:
                metrics['mc_uncertainties_axis'] = []
            metrics['mean_mc_angular_uncertainty'] = float('nan')
        return metrics

    # Calculate axis metrics using the filtered data
    metrics = calculate_axis_metrics(preds_filt, targets_filt, concentrations_filt, logger)
    
    if mc_dropout_samples > 0 and mc_uncertainties_filt is not None:
        metrics['mean_mc_angular_uncertainty'] = np.nanmean(mc_uncertainties_filt)
        if return_predictions:
            metrics['mc_uncertainties_axis'] = mc_uncertainties_filt.tolist()
    
    if return_predictions:
        metrics['predictions_dir'] = preds_filt.tolist()
        metrics['targets_dir'] = targets_filt.tolist()
        if concentrations_filt is not None:
            metrics['pred_concentrations'] = concentrations_filt.tolist()
           
    return metrics

def calculate_axis_metrics(outputs, targets, concentrations=None, logger=None):
    """
    Calculate axis prediction metrics.
    
    Args:
        outputs: Direction vector predictions
        targets: Direction vector targets
        concentrations: Concentration parameters (if available)
        logger: Logger instance for error reporting
        
    Returns:
        Dictionary of metrics
    """
    # Initialize default logger if not provided, for safety in this isolated function
    if logger is None:
        logger = logging.getLogger(__name__) # Use module-level logger
        # Configure it minimally if it has no handlers (e.g., if this module is used standalone)
        if not logger.hasHandlers():
            logger.addHandler(logging.StreamHandler()) # Add a basic handler
            logger.setLevel(logging.WARNING) # Default to WARNING if not configured

    try: # Wrap the core logic in a try-except block
        # Filter valid outputs/targets
        valid_mask = (
            ~np.isnan(outputs).any(axis=1) &
            ~np.isnan(targets).any(axis=1) &
            ~np.isinf(outputs).any(axis=1) &
            ~np.isinf(targets).any(axis=1)
        )
        
        if not np.any(valid_mask):
            logger.warning("No valid data points after filtering in calculate_axis_metrics.")
            return {
                'mean_angular_error': float('nan'),
                'median_angular_error': float('nan'),
                'success_rate_10deg': 0.0,
                'success_rate_30deg': 0.0,
                'mean_concentration': float('nan'),
                'mean_uncertainty_degrees': float('nan')
            }
        
        # Apply mask
        outputs_filtered = outputs[valid_mask]
        targets_filtered = targets[valid_mask]
        
        # Ensure unit vectors
        outputs_norm = np.linalg.norm(outputs_filtered, axis=1, keepdims=True)
        targets_norm = np.linalg.norm(targets_filtered, axis=1, keepdims=True)
        outputs_normalized = outputs_filtered / (outputs_norm + 1e-8)
        targets_normalized = targets_filtered / (targets_norm + 1e-8)
        
        # Calculate cosine similarity
        cosine_sim = np.sum(outputs_normalized * targets_normalized, axis=1)
        cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
        
        # Calculate angular difference
        angular_diff_rad = np.arccos(cosine_sim)
        angular_diff_deg = np.rad2deg(angular_diff_rad)
        
        # Calculate metrics
        mean_angular_error = np.mean(angular_diff_deg)
        median_angular_error = np.median(angular_diff_deg)
        success_rate_10deg = np.mean(angular_diff_deg < 10.0)
        success_rate_30deg = np.mean(angular_diff_deg < 30.0)
        
        results = {
            'mean_angular_error': float(mean_angular_error),
            'median_angular_error': float(median_angular_error),
            'success_rate_10deg': float(success_rate_10deg),
            'success_rate_30deg': float(success_rate_30deg),
        }
        
        # Add concentration metrics if available
        if concentrations is not None and len(concentrations) > 0:
            # Ensure concentrations is a NumPy array before applying the mask.
            # It's passed as an argument, might be a list of arrays or a single array.
            # If it's a list from evaluate_axis_model, it should be concatenated and filtered there.
            # Here, we assume 'concentrations' is already a filtered NumPy array matching outputs_filtered.
            if isinstance(concentrations, list): # Basic check, evaluate_axis_model should pass a filtered array
                if len(concentrations) == len(valid_mask) and all(isinstance(c, np.ndarray) for c in concentrations):
                     # Attempt to filter if it looks like a list of per-batch arrays that wasn't fully processed
                     temp_concentrations_list = [c[valid_mask[sum(len(cv) for cv in concentrations[:i]):sum(len(cv) for cv in concentrations[:i+1])]] for i,c in enumerate(concentrations) if c.size > 0]
                     if temp_concentrations_list:
                         concentrations_filtered = np.concatenate(temp_concentrations_list)
                     else:
                         concentrations_filtered = np.array([]) # Empty if all sub-arrays were empty or filtering failed
                else: # If it's a list but not matching, it's problematic
                     logger.warning("Concentrations passed as a list with mismatching structure for filtering.")
                     concentrations_filtered = np.array([]) # Cannot reliably filter
            elif isinstance(concentrations, np.ndarray):
                if concentrations.shape[0] == np.sum(valid_mask): # Already filtered
                    concentrations_filtered = concentrations
                elif concentrations.shape[0] == len(valid_mask): # Not filtered yet
                    concentrations_filtered = concentrations[valid_mask]
                else: # Shape mismatch
                    logger.warning(f"Concentrations array shape mismatch. Expected {np.sum(valid_mask)} or {len(valid_mask)}, got {concentrations.shape[0]}.")
                    concentrations_filtered = np.array([])
            else: # Unexpected type
                logger.warning(f"Unexpected type for concentrations: {type(concentrations)}. Expected NumPy array or list of arrays.")
                concentrations_filtered = np.array([])


            if concentrations_filtered.size > 0:
                mean_concentration = np.mean(concentrations_filtered)
                
                # Calculate approximate angular uncertainty in degrees
                # For Von Mises-Fisher distribution, angular deviation ≈ 1/√κ radians
                uncertainty_degrees = (180.0 / np.pi) / np.sqrt(concentrations_filtered + 1e-6)
                mean_uncertainty_degrees = np.mean(uncertainty_degrees)
                
                results['mean_concentration'] = float(mean_concentration)
                results['mean_uncertainty_degrees'] = float(mean_uncertainty_degrees)
                
                # Calculate calibration metrics (correlation between predicted uncertainty and error)
                if uncertainty_degrees.size == angular_diff_deg.size: # Ensure same size for correlation
                    # Ensure both are 1D arrays for corrcoef
                    uncertainty_degrees_1d = uncertainty_degrees.ravel()
                    angular_diff_deg_1d = angular_diff_deg.ravel()
                    correlation = np.corrcoef(uncertainty_degrees_1d, angular_diff_deg_1d)[0, 1]
                    results['uncertainty_error_correlation'] = float(correlation)
                else:
                    logger.warning("Size mismatch between uncertainty_degrees and angular_diff_deg for correlation.")
                
                # Calculate error in predicted uncertainty
                if uncertainty_degrees.size == angular_diff_deg.size:
                    uncertainty_error = np.mean(np.abs(uncertainty_degrees - angular_diff_deg))
                    results['uncertainty_error'] = float(uncertainty_error)
                else:
                    logger.warning("Size mismatch for uncertainty_error calculation.")
                    results['uncertainty_error'] = float('nan')
            else: # Concentrations filtered to empty
                results['mean_concentration'] = float('nan')
                results['mean_uncertainty_degrees'] = float('nan')
                results['uncertainty_error_correlation'] = float('nan')
                results['uncertainty_error'] = float('nan')
        else: # No concentrations provided or empty
            results['mean_concentration'] = float('nan')
            results['mean_uncertainty_degrees'] = float('nan')
            results['uncertainty_error_correlation'] = float('nan')
            results['uncertainty_error'] = float('nan')

    except Exception as e_axis_metric_calc:
        logger.error(f"ERROR during calculate_axis_metrics: {e_axis_metric_calc}", exc_info=True)
        # Return NaNs for all metrics if any calculation error occurs
        results = {
            'mean_angular_error': float('nan'),
            'median_angular_error': float('nan'),
            'success_rate_10deg': 0.0, # Keep as 0.0 for success rates
            'success_rate_30deg': 0.0,
            'mean_concentration': float('nan'),
            'mean_uncertainty_degrees': float('nan'),
            'uncertainty_error_correlation': float('nan'),
            'uncertainty_error': float('nan')
        }

    return results

def save_evaluation_results(results, file_path, logger=None):
    """
    Save evaluation results to a JSON file.
    
    Args:
        results: Evaluation results dictionary
        file_path: Path to save the results
        logger: Logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Convert numpy types to Python types
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.number):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_numpy(results)
        
        # Save to file
        with open(file_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Saved evaluation results to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save evaluation results: {e}") 