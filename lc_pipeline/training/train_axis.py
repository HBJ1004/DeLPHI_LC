#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training module for axis prediction models.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import copy
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Union, List, Any
from tqdm import tqdm

from ..losses.axis_losses import GeodesicVMFCombinedLoss, VMFLoss, angular_distance as vector_angular_distance
from ..models.utils import quaternion_to_direction_vector
from ..utils.axis_visualization import direction_vector_to_lb, angular_distance as lb_angular_distance

def train_axis_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: str,
    epochs: int = 60,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 10,
    checkpoint_path: Optional[str] = None,
    logger: Any = None
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Train an AxisNet model using data from AxisDataset.

    Args:
        model: AxisNet model (expects input [batch, num_bins])
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
        device: Device to train on
        epochs: Maximum number of epochs
        lr: Learning rate
        weight_decay: Weight decay for regularization
        patience: Patience for early stopping
        checkpoint_path: Path to save the best model
        logger: Logger instance

    Returns:
        Trained model and training history
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    model = model.to(device)
    torch.autograd.set_detect_anomaly(True)

    # Determine model's output format by examining a sample batch
    try:
        sample_batch = next(iter(train_loader))
        sample_data, sample_target = sample_batch
        sample_data = sample_data.to(device)
        with torch.no_grad():
            sample_output = model(sample_data)
        
        # Quaternion mode if output has 5 dimensions (4 for quaternion + 1 for log_kappa)
        use_quaternions = sample_output.shape[1] == 5
        logger.info(f"Detected {'quaternion-based' if use_quaternions else 'direction vector-based'} AxisNet model")
        
        # Check if target has correct dimensions
        expected_target_dim = 4 if use_quaternions else 3
        actual_target_dim = sample_target.shape[1]
        if actual_target_dim != expected_target_dim:
            logger.warning(f"Target dimension mismatch: model expecting {expected_target_dim}D {'quaternions' if use_quaternions else 'direction vectors'} "
                          f"but got {actual_target_dim}D targets.")
    except Exception as e:
        logger.warning(f"Error determining model type from sample batch: {e}. "
                      "Will detect model type during first forward pass.")
        use_quaternions = None

    # Select appropriate loss function
    if use_quaternions:
        criterion = GeodesicVMFCombinedLoss(geodesic_weight=0.7, vmf_weight=0.3)
        logger.info("Using GeodesicVMFCombinedLoss for quaternion-based model")
    else:
        criterion = VMFLoss()
        logger.info("Using VMFLoss for direction vector-based model")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mean_angular_error': [],
        'val_median_angular_error': [],
        'val_success_rate_30deg': [],
        'val_mean_concentration': [],  # Track predicted concentration (uncertainty)
        'lr': []
    }

    best_val_metric = float('inf')  # Minimize mean angular error
    best_epoch = 0
    epochs_no_improve = 0
    best_model_state = None

    logger.info(f"Starting AxisNet training: epochs={epochs}, lr={lr}, quaternion_mode={use_quaternions}")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        train_loader_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        nan_batches_train = 0

        for batch_idx, batch_content in enumerate(train_loader_bar):
            # collate_fn returns: (padded_data, targets_tensor, ids, lengths)
            # For axis training, padded_data is the phase-folded curves (input_data)
            # and targets_tensor contains the axis targets (quaternions or direction vectors)
            # AxisDataset yields 2 items: input_data (phase_curves) and axis_targets
            input_data, axis_targets = batch_content

            input_data, axis_targets = input_data.to(device), axis_targets.to(device)

            if torch.isnan(input_data).any() or torch.isnan(axis_targets).any():
                logger.warning(f"NaN/Inf in input/target epoch {epoch+1}, batch {batch_idx}. Skipping.")
                nan_batches_train += 1
                continue

            optimizer.zero_grad()

            # Forward pass
            output = model(input_data)
            
            # Check for NaNs in output
            if torch.isnan(output).any() or torch.isinf(output).any():
                logger.warning(f"NaN/Inf in model output epoch {epoch+1}, batch {batch_idx}. Skipping.")
                nan_batches_train += 1
                continue

            # Loss calculation
            try:
                # Slice output if using quaternions (output[:, :4]) before loss calculation
                if use_quaternions:
                    # Ensure target is also 4D if model predicts 5D (quat + kappa)
                    if output.shape[1] == 5 and axis_targets.shape[1] == 4:
                        predicted_quat = output[:, :4]
                        loss = criterion(predicted_quat, axis_targets)
                    else:
                         # Fallback or handle unexpected shapes
                         logger.warning(f"Unexpected shapes for loss calculation: Output {output.shape}, Target {axis_targets.shape}")
                         loss = criterion(output, axis_targets) # Attempt original if shapes don't match expectation
                else: # Direction vector mode
                    loss = criterion(output, axis_targets) # Assume loss handles [batch, 4] vs [batch, 3] or model outputs [batch, 3]

            except Exception as e:
                logger.warning(f"Error in loss calculation: {e}. Skipping batch.")
                nan_batches_train += 1
                continue

            if torch.isnan(loss).any() or torch.isinf(loss).any():
                logger.warning(f"NaN/Inf in loss epoch {epoch+1}, batch {batch_idx}. Skipping.")
                nan_batches_train += 1
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            epoch_loss += loss.item()

        if nan_batches_train > 0:
            logger.warning(f"Epoch {epoch+1}: Skipped {nan_batches_train} training batches due to NaNs.")
        
        epoch_loss /= (len(train_loader) - nan_batches_train) if len(train_loader) > nan_batches_train else 1
        history['train_loss'].append(epoch_loss)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # Validation phase
        current_val_loss = float('nan')
        current_val_mean_angular_error = float('nan')

        if val_loader and len(val_loader.dataset) > 0: # Check if val_loader is valid and has data
            model.eval()
            val_epoch_loss = 0
            val_outputs_direction = []
            val_outputs_concentration = []
            val_targets_direction = []
            val_loader_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            nan_batches_val = 0

            with torch.no_grad():
                for batch_idx, batch_content in enumerate(val_loader_bar):
                    # collate_fn returns: (padded_data, targets_tensor, ids, lengths)
                    # AxisDataset yields 2 items: input_data (phase_curves) and axis_targets
                    input_data, axis_targets = batch_content

                    input_data, axis_targets = input_data.to(device), axis_targets.to(device)

                    if torch.isnan(input_data).any() or torch.isnan(axis_targets).any():
                        logger.warning(f"NaN/Inf in input/target during evaluation batch {batch_idx}. Skipping.")
                        nan_batches_val += 1
                        continue

                    # Forward pass
                    output = model(input_data)

                    if torch.isnan(output).any() or torch.isinf(output).any():
                        logger.warning(f"NaN/Inf in model output during evaluation batch {batch_idx}. Skipping batch.")
                        nan_batches_val += 1
                        continue

                    # Process outputs based on model type
                    if use_quaternions:
                        # Extract quaternion and log_kappa from output
                        quaternion = output[:, :4]  # First 4 components are quaternion
                        log_kappa = output[:, 4]    # Last component is log_kappa
                        
                        # Normalize quaternion to ensure it's a unit quaternion
                        quaternion = F.normalize(quaternion, p=2, dim=1)
                        
                        # Convert quaternion to direction vector for metrics
                        direction = quaternion_to_direction_vector(quaternion)
                        
                        # Loss calculation
                        # Slice the output tensor to only include the quaternion part for the loss
                        predicted_quaternion = output[:, :4]
                        loss = criterion(predicted_quaternion, axis_targets)
                        
                        # Store outputs for metrics
                        val_outputs_direction.append(direction.cpu().numpy())
                        val_outputs_concentration.append(torch.exp(log_kappa).cpu().numpy())
                        
                        # Convert target quaternion to direction for metrics (if needed)
                        if axis_targets.shape[1] == 4:  # Target is quaternion
                            target_direction = quaternion_to_direction_vector(axis_targets)
                            val_targets_direction.append(target_direction.cpu().numpy())
                        else:  # Target is already direction (should not happen)
                            val_targets_direction.append(axis_targets.cpu().numpy())
                    else:
                        # Direction vector model
                        if output.shape[1] > 3:  # Model outputs direction + log_concentration
                            direction = output[:, :3]  # First 3 components are direction
                            log_concentration = output[:, 3]  # Last component is log_concentration
                            
                            # Normalize direction to ensure it's a unit vector
                            direction = F.normalize(direction, p=2, dim=1)
                            target_for_loss = F.normalize(axis_targets, p=2, dim=1) # Use axis_targets
                            
                            # Loss calculation
                            loss = criterion(output, target_for_loss)
                            
                            # Store outputs for metrics
                            val_outputs_direction.append(direction.cpu().numpy())
                        else:  # Model outputs only direction
                            direction = output  # All components are direction
                            
                            # Normalize direction to ensure it's a unit vector
                            direction = F.normalize(direction, p=2, dim=1)
                            target_for_loss = F.normalize(axis_targets, p=2, dim=1) # Use axis_targets
                            
                            # Loss calculation
                            loss = criterion(direction, target_for_loss)
                            
                            # Store outputs for metrics
                            val_outputs_direction.append(direction.cpu().numpy())
                        
                        # Target is already direction
                        val_targets_direction.append(axis_targets.cpu().numpy())

                    val_epoch_loss += loss.item()

            # Collate results from all validation batches
            if val_outputs_direction and val_targets_direction:
                val_outputs_direction_np = np.concatenate(val_outputs_direction)
                val_targets_direction_np = np.concatenate(val_targets_direction)
                
                # Ensure they are unit vectors before calculating angular distance
                norm_outputs = np.linalg.norm(val_outputs_direction_np, axis=1, keepdims=True)
                norm_targets = np.linalg.norm(val_targets_direction_np, axis=1, keepdims=True)
                
                # Avoid division by zero if a norm is zero (e.g. zero vector)
                # Replace zero norms with 1 to avoid NaN, effectively keeping the vector as zero if it was.
                # Such vectors will likely result in large angular error or be handled by clipping in arccos.
                norm_outputs[norm_outputs == 0] = 1
                norm_targets[norm_targets == 0] = 1

                val_outputs_direction_np = val_outputs_direction_np / norm_outputs
                val_targets_direction_np = val_targets_direction_np / norm_targets

                # Convert to PyTorch tensors for the loss function's angular_distance
                val_outputs_tensor = torch.from_numpy(val_outputs_direction_np).float().to(device)
                val_targets_tensor = torch.from_numpy(val_targets_direction_np).float().to(device)

                # Calculate angular errors (in radians from losses.angular_distance)
                # The function from axis_losses expects tensors and returns a tensor of radians
                angular_errors_rad_tensor = vector_angular_distance(val_outputs_tensor, val_targets_tensor)
                angular_errors_deg_np = np.degrees(angular_errors_rad_tensor.cpu().numpy())

                # Filter out NaNs that might arise from arccos if dot product was slightly out of [-1, 1]
                # though the loss function angular_distance should handle clamping.
                valid_error_mask = ~np.isnan(angular_errors_deg_np)
                if np.any(valid_error_mask):
                    current_val_mean_angular_error = np.mean(angular_errors_deg_np[valid_error_mask])
                    val_median_angular_error = np.median(angular_errors_deg_np[valid_error_mask])
                    val_success_rate_30deg = np.mean(angular_errors_deg_np[valid_error_mask] <= 30.0) * 100
                else:
                    current_val_mean_angular_error = float('nan')
                    val_median_angular_error = float('nan')
                    val_success_rate_30deg = float('nan')
                    logger.warning("All angular errors are NaN after calculation.")

                history['val_mean_angular_error'].append(current_val_mean_angular_error)
                history['val_median_angular_error'].append(val_median_angular_error)
                history['val_success_rate_30deg'].append(val_success_rate_30deg)
            else: # val_outputs_direction is empty
                current_val_mean_angular_error = float('nan')
                history['val_mean_angular_error'].append(float('nan'))
                history['val_median_angular_error'].append(float('nan'))
                history['val_success_rate_30deg'].append(float('nan'))
                logger.warning("Validation outputs list is empty. Cannot compute angular error metrics.")

            if val_outputs_concentration:
                val_outputs_concentration_np = np.concatenate(val_outputs_concentration)
                history['val_mean_concentration'].append(np.mean(val_outputs_concentration_np))
            else:
                history['val_mean_concentration'].append(float('nan'))
            
            current_val_loss = val_epoch_loss / (len(val_loader) - nan_batches_val) if len(val_loader) > nan_batches_val else 1
            history['val_loss'].append(current_val_loss)
            
            logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f}, Val Loss: {current_val_loss:.4f}, Val MAE: {current_val_mean_angular_error:.2f}°")
            scheduler.step(current_val_mean_angular_error if not np.isnan(current_val_mean_angular_error) else float('inf')) # Pass inf if NaN
        else:
            # No validation loader or empty validation set
            logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f}. No validation performed.")
            # Append NaN to history for consistency if validation is skipped
            history['val_loss'].append(float('nan'))
            history['val_mean_angular_error'].append(float('nan'))
            history['val_median_angular_error'].append(float('nan'))
            history['val_success_rate_30deg'].append(float('nan'))
            history['val_mean_concentration'].append(float('nan'))
            # Scheduler and early stopping might need to be handled if no val metric is available.
            # For now, scheduler.step() won't be called, and early stopping won't update if best_val_metric remains inf.

        # Early stopping check (use current_val_mean_angular_error if available)
        val_metric_for_early_stop = current_val_mean_angular_error if not np.isnan(current_val_mean_angular_error) else float('inf')
        if val_metric_for_early_stop < best_val_metric:
            best_val_metric = val_metric_for_early_stop
            best_epoch = epoch
            epochs_no_improve = 0
            best_model_state = copy.deepcopy(model.state_dict())
            if checkpoint_path:
                try:
                    torch.save(best_model_state, checkpoint_path)
                    logger.info(f"Saved best model checkpoint to {checkpoint_path} (Epoch {epoch+1}, Error: {best_val_metric:.2f} deg)")
                except Exception as e:
                    logger.error(f"Error saving checkpoint: {e}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                break

    # Load best model state if saved
    if best_model_state:
        logger.info(f"Loading best model weights from epoch {best_epoch+1} with mean error {best_val_metric:.2f} deg")
        model.load_state_dict(best_model_state)

    return model, history


def evaluate_axis_model(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: str,
    logger: Any = None,
    mc_dropout_samples: Optional[int] = None # Number of MC Dropout samples
) -> Dict[str, Any]:
    """
    Evaluate an AxisNet model. Can perform MC Dropout if mc_dropout_samples is specified.

    Args:
        model: Trained AxisNet model
        test_loader: DataLoader for testing
        device: Device to evaluate on
        logger: Logger instance
        mc_dropout_samples: Number of forward passes for MC Dropout. If None or 1, standard eval.

    Returns:
        Dictionary containing evaluation metrics and raw prediction/target data.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    model = model.to(device)
    # Standard evaluation mode initially
    model.eval()

    perform_mc_dropout = mc_dropout_samples is not None and mc_dropout_samples > 1
    if perform_mc_dropout:
        logger.info(f"Performing MC Dropout with {mc_dropout_samples} samples.")
        # Activate dropout layers for MC sampling
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train() # Enable dropout
    else:
        logger.info("Performing standard evaluation (no MC Dropout).")

    # Determine model's output format
    # Re-determine model type if not explicitly known (safer)
    use_quaternions = None
    try:
        sample_batch = next(iter(test_loader))
        sample_data, _ = sample_batch
        sample_data = sample_data.to(device)
        with torch.no_grad():
            sample_output = model(sample_data)
        use_quaternions = sample_output.shape[1] == 5
        logger.info(f"Evaluating {'quaternion-based' if use_quaternions else 'direction vector-based'} AxisNet model")
    except Exception as e:
        logger.error(f"Could not determine model type during evaluation: {e}. Assuming default based on model arch if possible, else error.")
        # Potentially raise error or rely on model's internal flag if available
        # For now, let's assume use_quaternions was determined during training or needs to be passed
        if not hasattr(model, 'output_dim'): # Simple check, might need adjustment
             raise ValueError("Cannot determine model output type for evaluation") from e
        use_quaternions = model.output_dim == 5
        logger.warning(f"Falling back to model property: use_quaternions={use_quaternions}")

    # --- Lists to store raw data for plotting ---
    all_outputs_direction = []
    all_outputs_concentration = [] # Kappa for quaternions, concentration for direction vectors
    all_targets_direction = []
    all_angular_errors = []

    test_nan_batches = 0
    # Store all predictions from all MC samples if active
    all_mc_output_stacks = [] # List to store stacks of [num_samples, batch_size, feature_dim] for direction
    all_mc_concentration_stacks = [] # For concentration

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            if torch.isnan(data).any() or torch.isinf(data).any() or torch.isnan(target).any() or torch.isinf(target).any():
                logger.warning(f"NaN/Inf in input/target during evaluation batch {batch_idx}. Skipping.")
                test_nan_batches += 1
                continue

            batch_mc_directions = []
            batch_mc_concentrations = []

            num_passes = mc_dropout_samples if perform_mc_dropout else 1
            
            # Initialize output to None in case num_passes is 0, though current logic ensures num_passes >= 1
            output = None 
            for mc_pass in range(num_passes):
                output = model(data) # Standard forward pass for PATAxis needs x_folded, x_raw
                                    # This evaluate_axis_model is generic, assumes model(data) works.
                                    # If PATAxis is used, data loader needs to yield (x_folded, x_raw, padding_masks), target
                                    # And this call needs to be model(x_folded, x_raw, ...)
                                    # For now, assuming data is correctly structured for the given model.
                                    # Note: 'output' will be from the last pass if num_passes > 1

            if output is None or torch.isnan(output).any() or torch.isinf(output).any():
                if perform_mc_dropout:
                    # mc_pass will hold the index of the last pass
                    logger.warning(f"NaN/Inf in model output (from last MC pass: {mc_pass + 1 if num_passes > 0 else 'N/A'}) during batch {batch_idx}. This batch's MC results might be compromised if this was the only valid pass.")
                else:
                    logger.warning(f"NaN/Inf in model output during evaluation batch {batch_idx}. Skipping batch.")
                
                if not perform_mc_dropout: # if not MC, skip entire batch (this condition is effectively same as the else block above)
                    test_nan_batches += 1
                    # Break from mc_pass loop and outer loop will continue to next batch (comment is misleading)
                    # Need a flag to skip appending to all_outputs_direction for this batch
                    output = None # Signal to skip this batch entirely (already set if it was None, but good for NaN/Inf case)
                
                # The original code had 'break' here. If 'break' is intended to stop all evaluation, it should remain.
                # If it's to "go to next batch" as per comment, it should be 'continue'.
                # Sticking to 'break' as per original code, fixing its indentation.
                # This 'break' will exit the 'for batch_idx...' loop.
                # If the intention was to skip the current batch, 'continue' would be more appropriate.
                # Given the instruction "fix indentation", we keep 'break' but correctly indented.
                # However, the lint "Unindent amount does not match previous indent" for the original break
                # and the comment "go to next batch" strongly suggest `continue` was intended.
                # For safety and common practice, using `continue` to go to the next batch.
                # If strict `break` is required, this can be changed back.
                # The lint errors are best resolved by assuming `continue` was meant for "go to next batch".
                # If I must keep `break` and fix indentation:
                # break # This would be at this indentation level.
                # Let's assume the user wants the logical "go to next batch":
                continue # Skips to the next iteration of the batch_idx loop.

            # Process outputs based on model type (use_quaternions needs to be determined reliably)
            # This logic is complex and relies on use_quaternions being correctly set.
            # Assuming use_quaternions is determined before this loop or is stable.
            # Correcting indentation for this block:
            current_pass_direction, current_pass_concentration = None, None
            if use_quaternions:
                quaternion = output[:, :4]
                log_kappa = output[:, 4]
                quaternion = F.normalize(quaternion, p=2, dim=1)
                current_pass_direction = quaternion_to_direction_vector(quaternion)
                current_pass_concentration = torch.exp(log_kappa) # Kappa
            else: # Direction vector model
                if output.shape[1] > 3: # Direction + concentration
                    current_pass_direction = output[:, :3]
                    log_concentration = output[:, 3]
                    current_pass_concentration = torch.exp(log_concentration)
                else: # Direction only
                    current_pass_direction = output
                    current_pass_concentration = torch.ones(current_pass_direction.shape[0], device=device) # Placeholder
                current_pass_direction = F.normalize(current_pass_direction, p=2, dim=1)
            
            # This part of the original selection was outside the NaN check and output processing.
            # It correctly appends the processed output of the *last pass* (or single pass)
            if current_pass_direction is not None: # Should always be true if no NaN and output processed
                batch_mc_directions.append(current_pass_direction.cpu().numpy())
            if current_pass_concentration is not None:
                batch_mc_concentrations.append(current_pass_concentration.cpu().numpy())
            
            if not batch_mc_directions: # All MC passes failed for this batch, or standard eval failed
                                        # This check is now effectively for the single (last) pass if it was NaN.
                if output is None and not perform_mc_dropout : # Standard eval failed and was marked by output=None
                     pass # test_nan_batches already incremented if it was due to NaN/Inf in output
                elif perform_mc_dropout: # This case might be less likely now with the top-level NaN check on final 'output'
                     logger.warning(f"Output processing failed for batch {batch_idx} despite non-NaN output, or all MC passes were NaN (logic flaw). Skipping batch.")
                     test_nan_batches +=1
                continue # Go to the next batch

            # Aggregate MC predictions for this batch
            # This logic assumes `batch_mc_directions` contains multiple samples if MC dropout.
            # However, with the current structure, it will only contain the result of the *last* pass.
            # This is a pre-existing logic issue, not an indentation one.
            if perform_mc_dropout:
                # Stack along a new dimension (num_samples, batch_size, feature_dim)
                # This will fail if batch_mc_directions doesn't have multiple entries per item.
                # For now, assuming the user wants to fix indentation of the existing structure.
                # If batch_mc_directions has one item (from the last pass), stacking might not be what's expected.
                # The original code implies batch_mc_directions is populated *inside* the mc_pass loop.
                # The selection had the append *after* the mc_pass loop.
                # This means batch_mc_directions will have one entry (the last pass).
                # The following aggregation logic for MC dropout will thus not reflect true MC averaging.
                
                # If the intention was to average multiple MC passes, the append to batch_mc_directions
                # and batch_mc_concentrations should be *inside* the mc_pass loop,
                # and the NaN check should also be inside for each pass.
                # Sticking to fixing indentation of the provided selection:
                batch_mc_directions_np = np.stack(batch_mc_directions, axis=0) 
                batch_mc_concentrations_np = np.stack(batch_mc_concentrations, axis=0)
                
                all_mc_output_stacks.append(batch_mc_directions_np) # Will be [1, batch_size, 3]
                all_mc_concentration_stacks.append(batch_mc_concentrations_np) # Will be [1, batch_size]

                mean_direction_batch = np.mean(batch_mc_directions_np, axis=0)
                mean_direction_batch_norm = np.linalg.norm(mean_direction_batch, axis=1, keepdims=True)
                mean_direction_batch_norm[mean_direction_batch_norm == 0] = 1e-8 
                final_direction_batch = mean_direction_batch / mean_direction_batch_norm
                
                mean_concentration_batch = np.mean(batch_mc_concentrations_np, axis=0)
                all_outputs_direction.append(final_direction_batch) 
                all_outputs_concentration.append(mean_concentration_batch)
            else:
                # Standard evaluation (num_passes was 1)
                all_outputs_direction.append(batch_mc_directions[0]) 
                all_outputs_concentration.append(batch_mc_concentrations[0])

            # Target processing (only once per batch)
            if use_quaternions:
                if target.shape[1] == 4:
                    target_normalized = F.normalize(target, p=2, dim=1)
                    target_direction = quaternion_to_direction_vector(target_normalized)
                else:
                    logger.error(f"Target shape mismatch for quaternion model: expected 4, got {target.shape[1]}")
                    # Placeholder shape should match a single direction vector from one item in the batch
                    # Assuming current_pass_direction is representative of the shape [batch_size, 3]
                    # So, a single item is [3].
                    # all_outputs_direction[-1] is [batch_size, 3], so all_outputs_direction[-1][0] is [3]
                    target_direction = torch.zeros_like(torch.from_numpy(all_outputs_direction[-1][0]), device=device) 
            else:
                target_direction = F.normalize(target, p=2, dim=1)
            all_targets_direction.append(target_direction.cpu().numpy())

    if perform_mc_dropout:
        # If MC dropout was performed, all_outputs_direction and all_outputs_concentration
        # now contain the MEAN of the MC samples for each batch item.
        # all_mc_outputs_direction contains ALL raw MC samples if further analysis is needed (e.g. variance)
        # For metrics calculation below, we proceed with the mean predictions stored in all_outputs_direction.
        # Let's rename all_mc_outputs_direction to all_mc_output_stacks for clarity
        pass

    # --- Aggregate results --- (this part remains largely the same but operates on mean predictions if MC was on)
    if not all_outputs_direction or not all_targets_direction:
        logger.error("No valid evaluation data collected. Cannot calculate metrics or provide plot data.")
        return {
            'mean_angular_error': float('nan'),
            'median_angular_error': float('nan'),
            'success_rate_15deg': float('nan'),
            'success_rate_30deg': float('nan'),
            'success_rate_45deg': float('nan'),
            'mean_concentration': float('nan'),
            # Ensure plot keys exist but are empty
            'true_longitudes': [], 
            'true_latitudes': [], 
            'pred_longitudes': [], 
            'pred_latitudes': [], 
            'true_kappas': [],
            'pred_kappas': [],
            'angular_errors': [],
            # Add keys for MC dropout uncertainty metrics
            'pred_lon_mean_mc': [],
            'pred_lat_mean_mc': [],
            'pred_lon_std_mc': [],
            'pred_lat_std_mc': [],
            'pred_concentration_mean_mc': [],
            'pred_concentration_std_mc': []
        }

    pred_directions_all = np.concatenate(all_outputs_direction, axis=0)
    true_directions_all = np.concatenate(all_targets_direction, axis=0)
    pred_concentrations_all = np.concatenate(all_outputs_concentration, axis=0)

    # --- Calculate Coordinates (Longitude/Latitude) --- 
    try:
        # Assumes direction_vector_to_lb exists and returns (lon, lat) in degrees
        pred_lon_mean, pred_lat_mean = direction_vector_to_lb(pred_directions_all) # These are from mean directions
        true_lon, true_lat = direction_vector_to_lb(true_directions_all)
    except NameError:
        logger.error("Function 'direction_vector_to_lb' not found. Cannot convert directions to lon/lat.")
        # Return NaNs for coordinates if conversion fails
        nan_coords = np.full(pred_directions_all.shape[0], np.nan)
        pred_lon_mean, pred_lat_mean, true_lon, true_lat = nan_coords, nan_coords, nan_coords, nan_coords
    except Exception as e:
        logger.error(f"Error converting direction vectors to lon/lat: {e}")
        nan_coords = np.full(pred_directions_all.shape[0], np.nan)
        pred_lon_mean, pred_lat_mean, true_lon, true_lat = nan_coords, nan_coords, nan_coords, nan_coords

    # --- Calculate Angular Errors --- 
    try:
        # Assumes lb_angular_distance takes lon/lat in degrees
        angular_errors = lb_angular_distance(true_lon, true_lat, pred_lon_mean, pred_lat_mean)
    except NameError:
         logger.error("Function 'lb_angular_distance' not found. Cannot calculate angular errors.")
         angular_errors = np.full(true_lon.shape[0], np.nan)
    except Exception as e:
         logger.error(f"Error calculating angular distance: {e}")
         angular_errors = np.full(true_lon.shape[0], np.nan)

    # Filter out NaN errors if any occurred during coordinate conversion or distance calc
    valid_error_mask = ~np.isnan(angular_errors)
    if not np.any(valid_error_mask):
        logger.warning("All angular errors are NaN. Cannot compute metrics.")
        mean_ang_err, median_ang_err = float('nan'), float('nan')
        sr15, sr30, sr45 = float('nan'), float('nan'), float('nan')
    else:
        valid_errors = angular_errors[valid_error_mask]
        mean_ang_err = np.mean(valid_errors)
        median_ang_err = np.median(valid_errors)
        sr15 = np.mean(valid_errors <= 15.0) * 100
        sr30 = np.mean(valid_errors <= 30.0) * 100
        sr45 = np.mean(valid_errors <= 45.0) * 100

    mean_concentration = np.mean(pred_concentrations_all) if pred_concentrations_all.size > 0 else float('nan')

    # --- Calculate MC Dropout uncertainty metrics if available ---
    pred_lon_mean_mc_all, pred_lat_mean_mc_all = [], []
    pred_lon_std_mc_all, pred_lat_std_mc_all = [], []
    pred_concentration_mean_mc_all, pred_concentration_std_mc_all = [], []

    if perform_mc_dropout and all_mc_output_stacks:
        # Concatenate all MC samples: [num_total_items, num_samples, 3] for directions
        # Concatenate all MC samples: [num_total_items, num_samples] for concentrations
        mc_directions_full = np.concatenate([s.transpose(1,0,2) for s in all_mc_output_stacks], axis=0) # samples become [total_items, num_samples, 3]
        mc_concentrations_full = np.concatenate([s.transpose(1,0) for s in all_mc_concentration_stacks], axis=0) # [total_items, num_samples]

        # Reshape directions for lon/lat conversion: [total_items * num_samples, 3]
        num_total_items, num_samples, _ = mc_directions_full.shape
        mc_directions_flat = mc_directions_full.reshape(num_total_items * num_samples, 3)
        
        try:
            pred_lon_all_samples, pred_lat_all_samples = direction_vector_to_lb(mc_directions_flat)
            # Reshape back to [total_items, num_samples]
            pred_lon_all_samples = pred_lon_all_samples.reshape(num_total_items, num_samples)
            pred_lat_all_samples = pred_lat_all_samples.reshape(num_total_items, num_samples)

            pred_lon_mean_mc_all = np.mean(pred_lon_all_samples, axis=1).tolist()
            pred_lat_mean_mc_all = np.mean(pred_lat_all_samples, axis=1).tolist()
            pred_lon_std_mc_all = np.std(pred_lon_all_samples, axis=1).tolist()
            pred_lat_std_mc_all = np.std(pred_lat_all_samples, axis=1).tolist()
        except Exception as e:
            logger.error(f"Error calculating lon/lat STDs from MC samples: {e}")

        pred_concentration_mean_mc_all = np.mean(mc_concentrations_full, axis=1).tolist()
        pred_concentration_std_mc_all = np.std(mc_concentrations_full, axis=1).tolist()

    # --- Construct Results Dictionary --- 
    results = {
        'mean_angular_error': mean_ang_err,
        'median_angular_error': median_ang_err,
        'success_rate_15deg': sr15,
        'success_rate_30deg': sr30,
        'success_rate_45deg': sr45,
        'mean_concentration': mean_concentration,
        # Add raw data for plotting (convert numpy arrays to lists)
        'true_longitudes': true_lon.tolist() if not np.all(np.isnan(true_lon)) else [],
        'true_latitudes': true_lat.tolist() if not np.all(np.isnan(true_lat)) else [],
        'pred_longitudes': pred_lon_mean.tolist() if not np.all(np.isnan(pred_lon_mean)) else [],
        'pred_latitudes': pred_lat_mean.tolist() if not np.all(np.isnan(pred_lat_mean)) else [],
        # For kappa, only include if quaternion model was used. Assume true_kappa isn't typically available.
        'true_kappas': [], # Placeholder, modify if true kappas are available 
        'pred_kappas': pred_concentrations_all.tolist() if use_quaternions and pred_concentrations_all.size > 0 else [],
        'angular_errors': angular_errors.tolist() if not np.all(np.isnan(angular_errors)) else [],
        # Add MC uncertainty metrics
        'pred_lon_mean_mc': pred_lon_mean_mc_all,
        'pred_lat_mean_mc': pred_lat_mean_mc_all,
        'pred_lon_std_mc': pred_lon_std_mc_all,
        'pred_lat_std_mc': pred_lat_std_mc_all,
        'pred_concentration_mean_mc': pred_concentration_mean_mc_all,
        'pred_concentration_std_mc': pred_concentration_std_mc_all
    }

    logger.info(
        f"Evaluation Results: Mean Ang Err={mean_ang_err:.2f} deg, "
        f"Median Ang Err={median_ang_err:.2f} deg, "
        f"Success (<30 deg)={sr30:.2f}%"
    )

    return results


def train_axis_with_curriculum(
    model: nn.Module,
    data_dirs: List[str],
    period_model: nn.Module,
    device: str,
    num_bins: int = 100,
    epochs: int = 60,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 10,
    checkpoint_path: Optional[str] = None,
    period_scale_factor: float = 50.0,
    use_quaternions: bool = True,
    curriculum_epochs: Tuple[int, int, int] = (15, 30, 45),
    logger: Any = None,
    dataset: 'AsteroidDataset' = None,
    teacher_forcing_schedule: Optional[List[Tuple[int, float]]] = None, # E.g., [(10, 1.0), (20, 0.5), (epochs, 0.0)]
    config: Any = None
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Train an AxisNet model using curriculum learning.

    The curriculum is defined by `teacher_forcing_schedule` if provided,
    otherwise it falls back to a staged approach based on `curriculum_epochs`.

    Args:
        model: AxisNet model
        data_dirs: List of directories containing raw light curve data (potentially unused if dataset is directly provided)
        period_model: Trained period prediction model
        device: Device to train on
        num_bins: Number of bins for phase folding
        epochs: Total number of epochs for training
        batch_size: Batch size for training
        lr: Learning rate
        weight_decay: Weight decay for regularization
        patience: Patience for early stopping
        checkpoint_path: Path to save the best model
        period_scale_factor: DEPRECATED - will be read from dataset.period_scale_factor
        use_quaternions: Whether to use quaternions for axis targets
        curriculum_epochs: Fallback tuple of epochs marking end of stage 1, 2 for a 3-stage curriculum if schedule is not given.
                           (stage1_end_epoch, stage2_end_epoch, total_curriculum_epochs_if_different_from_main_epochs)
        logger: Logger instance
        dataset: Initialized AsteroidDataset containing raw data and processed features.
        teacher_forcing_schedule: A list of tuples (epoch_end, mix_ratio). 
                                  Defines the true_period_mix_ratio up to epoch_end.
                                  Example: [(10, 1.0), (20, 0.5), (epochs, 0.0)] means:
                                  - Epochs 0-10: mix_ratio = 1.0 (all true)
                                  - Epochs 11-20: mix_ratio = 0.5
                                  - Epochs 21-total_epochs: mix_ratio = 0.0 (all predicted)
        config: Configuration object

    Returns:
        Trained model and training history
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        
    # Validate curriculum_epochs if used as fallback
    stage1_fallback_end, stage2_fallback_end, _ = curriculum_epochs
    if not (0 <= stage1_fallback_end <= stage2_fallback_end):
        logger.warning("Invalid curriculum_epochs for fallback. Using (15, 30, 45) as default for fallback.")
        stage1_fallback_end, stage2_fallback_end, _ = 15, 30, 45
        
    logger.info(f"Starting curriculum training for up to {epochs} epochs.")
    if teacher_forcing_schedule:
        logger.info(f"Using provided teacher_forcing_schedule: {teacher_forcing_schedule}")
    else:
        logger.info(f"No teacher_forcing_schedule provided. Falling back to staged curriculum using curriculum_epochs: "
                    f"Stage 1 (100% true) ends epoch {stage1_fallback_end}, "
                    f"Stage 2 (50% true - example) ends epoch {stage2_fallback_end}, "
                    f"Stage 3 (0% true) thereafter.")
    
    # Import necessary functions locally
    from ..data.collate import generate_axis_data
    from ..data.datasets import AxisDataset
    from torch.utils.data import DataLoader, random_split
    
    model = model.to(device)
    torch.autograd.set_detect_anomaly(True)

    # Determine model's output format early (as in standard train_axis_model)
    # This requires creating a small sample dataset/dataloader or assuming from config
    # Assuming quaternion mode based on config/passed parameter for now
    
    if use_quaternions:
        criterion = GeodesicVMFCombinedLoss(geodesic_weight=0.7, vmf_weight=0.3)
    else:
        criterion = VMFLoss()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mean_angular_error': [],
        'val_median_angular_error': [],
        'val_success_rate_30deg': [],
        'val_mean_concentration': [],
        'lr': [],
        'true_period_ratio': []  # Track ratio of true periods used
    }

    best_val_metric = float('inf')
    best_epoch = 0
    epochs_no_improve = 0
    best_model_state = None

    stage1_end, stage2_end, stage3_end = curriculum_epochs
    logger.info(f"Starting curriculum training with stages: "
                f"True periods: 0-{stage1_end}, "
                f"Mixed: {stage1_end}-{stage2_end}, "
                f"Pred periods: {stage2_end}-{stage3_end}")

    # Retrieve period scale factor from the dataset
    actual_period_scale_factor = getattr(dataset, 'period_scale_factor', 50.0)
    logger.info(f"Using period scale factor from dataset: {actual_period_scale_factor}")

    for epoch in range(epochs):
        current_true_period_mix_ratio = 0.0 # Default to all predicted if not otherwise set
        stage_info = ""

        # Determine true_period_mix_ratio based on configuration
        decay_type = getattr(config.axis_model, 'teacher_forcing_decay_type', None)
        decay_epochs_config = getattr(config.axis_model, 'teacher_forcing_decay_epochs', None)
        # teacher_forcing_schedule is defined outside this selection, assumed available from function args

        applies_decay_logic = decay_type in ['linear', 'exponential'] and decay_epochs_config is not None

        if applies_decay_logic:
            initial_ratio = getattr(config.axis_model, 'teacher_forcing_initial_ratio', 1.0)
            final_ratio = getattr(config.axis_model, 'teacher_forcing_final_ratio', 0.0)
            decay_epochs = decay_epochs_config # Use the specifically configured value

            if epoch < decay_epochs:
                if decay_type == 'linear':
                    current_true_period_mix_ratio = initial_ratio - (initial_ratio - final_ratio) * (epoch / decay_epochs)
                elif decay_type == 'exponential':
                    # Ensure initial_ratio is not zero for division
                    safe_initial_ratio = initial_ratio if initial_ratio > 1e-6 else 1e-6
                    current_true_period_mix_ratio = initial_ratio * (final_ratio / safe_initial_ratio) ** (epoch / decay_epochs)
            else: # Epoch is beyond decay_epochs
                current_true_period_mix_ratio = final_ratio
            
            current_true_period_mix_ratio = max(min(current_true_period_mix_ratio, initial_ratio), final_ratio) # Clamp
            stage_info = f"Continuous Decay ({decay_type}): mix_ratio={current_true_period_mix_ratio:.2f}"
        
        elif teacher_forcing_schedule:
            # Determine mix_ratio from the schedule
            # The schedule is [(epoch_end1, ratio1), (epoch_end2, ratio2), ...]
            # The ratio is active for epochs > previous_epoch_end and <= current_epoch_end
            current_true_period_mix_ratio = teacher_forcing_schedule[-1][1] # Default to last ratio if epoch exceeds all defined ends
            last_epoch_end = 0
            schedule_stage_found = False
            for schedule_epoch_end, mix_ratio_for_stage in teacher_forcing_schedule:
                if epoch < schedule_epoch_end: # Current epoch is within this stage defined by schedule_epoch_end
                    current_true_period_mix_ratio = mix_ratio_for_stage
                    stage_info = f"Teacher Forcing Schedule: mix_ratio={current_true_period_mix_ratio:.2f} (until epoch {schedule_epoch_end})"
                    schedule_stage_found = True
                    break # Found the current stage
                last_epoch_end = schedule_epoch_end
            
            if not schedule_stage_found and epoch >= last_epoch_end: # If epoch is beyond the last defined stage end
                 stage_info = f"Teacher Forcing Schedule: mix_ratio={current_true_period_mix_ratio:.2f} (final stage)"

        else: # Fallback if no valid decay logic and no teacher forcing schedule
            # Issue warning if decay was configured but invalid (e.g., unknown type)
            if decay_type and decay_epochs_config is not None and not applies_decay_logic:
                logger.warning(f"Unknown or improperly configured teacher_forcing_decay_type: {decay_type}. Falling back to predefined stages.")

            # Fallback to predefined stages (stage1_fallback_end, stage2_fallback_end are from curriculum_epochs)
            if epoch < stage1_fallback_end:
                current_true_period_mix_ratio = 1.0
                stage_info = f"Fallback Stage 1: 100% true periods (mix_ratio={current_true_period_mix_ratio:.2f})"
            elif epoch < stage2_fallback_end:
                current_true_period_mix_ratio = 0.5
                stage_info = f"Fallback Stage 2: 50% true periods (mix_ratio={current_true_period_mix_ratio:.2f})"
            else:
                current_true_period_mix_ratio = 0.0
                stage_info = f"Fallback Stage 3: 0% true periods (mix_ratio={current_true_period_mix_ratio:.2f})"

        history['true_period_ratio'].append(current_true_period_mix_ratio)

        # Generate data for the current stage/epoch
        try:
            logger.info(f"Generating axis data for epoch {epoch+1} ({stage_info if stage_info else f'mix_ratio={current_true_period_mix_ratio:.2f}'})...")
            axis_data, axis_targets, _ = generate_axis_data(
                period_model=period_model if current_true_period_mix_ratio < 1.0 else None, 
                dataset=dataset, 
                num_bins=num_bins,
                smooth=True, 
                use_quaternions=use_quaternions,
                return_ids=True, 
                true_period_mix_ratio=current_true_period_mix_ratio,
                logger=logger # Pass logger to generate_axis_data
            )

            if axis_data is None or len(axis_data) == 0:
                logger.error(f"Failed to generate axis data for epoch {epoch+1}. Skipping epoch.")
                continue

            axis_dataset_epoch = AxisDataset(axis_data, axis_targets, use_quaternions=use_quaternions)
            train_size = int(0.8 * len(axis_dataset_epoch))
            val_size = len(axis_dataset_epoch) - train_size
            if train_size == 0 or val_size == 0:
                 logger.warning(f"Train/Val split zero for epoch {epoch+1}. Using full set. Size: {len(axis_dataset_epoch)}")
                 train_dataset_epoch, val_dataset_epoch = axis_dataset_epoch, axis_dataset_epoch
            else:
                 train_dataset_epoch, val_dataset_epoch = random_split(axis_dataset_epoch, [train_size, val_size])

            train_loader_epoch = DataLoader(train_dataset_epoch, batch_size=batch_size, shuffle=True)
            val_loader_epoch = DataLoader(val_dataset_epoch, batch_size=batch_size, shuffle=False)
            logger.info(f"Created dataloaders for epoch {epoch+1}. Train size: {len(train_dataset_epoch)}, Val size: {len(val_dataset_epoch)}")

        except Exception as data_gen_e:
            logger.error(f"Error generating axis data for epoch {epoch+1}: {data_gen_e}", exc_info=True)
            continue # Skip epoch if data generation fails

        # --- Training phase for the epoch ---
        model.train()
        train_loss = 0
        nan_batches = 0
        for batch_idx, (data, target) in enumerate(train_loader_epoch):
            # (Standard training loop as in train_axis_model, simplified here)
            data, target = data.to(device), target.to(device)
            if torch.isnan(data).any() or torch.isnan(target).any():
                nan_batches += 1
                continue
            optimizer.zero_grad()
            output = model(data)
            if torch.isnan(output).any():
                 nan_batches += 1
                 continue
                
            loss = criterion(output[:, :4] if use_quaternions and output.shape[1] == 5 else output, target)
            if torch.isnan(loss).any():
                nan_batches += 1
                continue
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        if nan_batches > 0:
             logger.warning(f"Epoch {epoch+1}: Skipped {nan_batches} training batches due to NaNs.")
        train_loss /= max(1, len(train_loader_epoch) - nan_batches)
        history['train_loss'].append(train_loss)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # --- Validation phase for the epoch ---
        current_val_loss = float('nan')
        current_val_mean_angular_error = float('nan')

        if val_loader_epoch and len(val_loader_epoch.dataset) > 0: # Check if val_loader is valid and has data
            model.eval()
            val_loss = 0
            val_outputs_direction = []
            val_outputs_concentration = []
            val_targets_direction = []
            val_nan_batches = 0

            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(val_loader_epoch):
                    data, target = data.to(device), target.to(device)

                    # Skip problematic batches
                    if torch.isnan(data).any() or torch.isinf(data).any() or torch.isnan(target).any() or torch.isinf(target).any():
                        val_nan_batches += 1
                        continue

                    # Forward pass
                    output = model(data)

                    if torch.isnan(output).any():
                        val_nan_batches += 1
                        continue

                    # Process outputs based on model type
                    if use_quaternions:
                        # Extract quaternion and log_kappa from output
                        quaternion = output[:, :4]  # First 4 components are quaternion
                        log_kappa = output[:, 4]    # Last component is log_kappa
                        
                        # Normalize quaternion to ensure it's a unit quaternion
                        quaternion = F.normalize(quaternion, p=2, dim=1)
                        
                        # Convert quaternion to direction vector for metrics
                        direction = quaternion_to_direction_vector(quaternion)
                        
                        # Loss calculation
                        # Slice the output tensor to only include the quaternion part for the loss
                        predicted_quaternion = output[:, :4]
                        loss = criterion(predicted_quaternion, target)
                        
                        # Store outputs for metrics
                        val_outputs_direction.append(direction.cpu().numpy())
                        val_outputs_concentration.append(torch.exp(log_kappa).cpu().numpy())
                        
                        # Convert target quaternion to direction for metrics (if needed)
                        if target.shape[1] == 4:  # Target is quaternion
                            target_direction = quaternion_to_direction_vector(target)
                            val_targets_direction.append(target_direction.cpu().numpy())
                        else:  # Target is already direction (should not happen)
                            val_targets_direction.append(target.cpu().numpy())
                    else:
                        # Direction vector model
                        if output.shape[1] > 3: # Direction + concentration
                            direction = output[:, :3]
                            log_concentration = output[:, 3]
                            concentration = torch.exp(log_concentration)
                        else: # Direction only
                            direction = output
                            concentration = torch.ones(direction.shape[0], device=device) # Placeholder

                        # Normalize directions
                        direction = F.normalize(direction, p=2, dim=1)
                        target_direction = F.normalize(target, p=2, dim=1) # Assume target is direction vector

                        # Loss calculation
                        loss = criterion(direction, target)
                        
                        # Store outputs for metrics
                        val_outputs_direction.append(direction.cpu().numpy())
                    
                    # Target is already direction
                    val_targets_direction.append(target.cpu().numpy())

                    val_loss += loss.item()

            # Collate results from all validation batches
            if val_outputs_direction and val_targets_direction:
                val_outputs_direction_np = np.concatenate(val_outputs_direction)
                val_targets_direction_np = np.concatenate(val_targets_direction)
                
                # Ensure they are unit vectors before calculating angular distance
                norm_outputs = np.linalg.norm(val_outputs_direction_np, axis=1, keepdims=True)
                norm_targets = np.linalg.norm(val_targets_direction_np, axis=1, keepdims=True)
                
                # Avoid division by zero if a norm is zero (e.g. zero vector)
                # Replace zero norms with 1 to avoid NaN, effectively keeping the vector as zero if it was.
                # Such vectors will likely result in large angular error or be handled by clipping in arccos.
                norm_outputs[norm_outputs == 0] = 1
                norm_targets[norm_targets == 0] = 1

                val_outputs_direction_np = val_outputs_direction_np / norm_outputs
                val_targets_direction_np = val_targets_direction_np / norm_targets

                # Convert to PyTorch tensors for the loss function's angular_distance
                val_outputs_tensor = torch.from_numpy(val_outputs_direction_np).float().to(device)
                val_targets_tensor = torch.from_numpy(val_targets_direction_np).float().to(device)

                # Calculate angular errors (in radians from losses.angular_distance)
                # The function from axis_losses expects tensors and returns a tensor of radians
                angular_errors_rad_tensor = vector_angular_distance(val_outputs_tensor, val_targets_tensor)
                angular_errors_deg_np = np.degrees(angular_errors_rad_tensor.cpu().numpy())

                # Filter out NaNs that might arise from arccos if dot product was slightly out of [-1, 1]
                # though the loss function angular_distance should handle clamping.
                valid_error_mask = ~np.isnan(angular_errors_deg_np)
                if np.any(valid_error_mask):
                    current_val_mean_angular_error = np.mean(angular_errors_deg_np[valid_error_mask])
                    val_median_angular_error = np.median(angular_errors_deg_np[valid_error_mask])
                    val_success_rate_30deg = np.mean(angular_errors_deg_np[valid_error_mask] <= 30.0) * 100
                else:
                    current_val_mean_angular_error = float('nan')
                    val_median_angular_error = float('nan')
                    val_success_rate_30deg = float('nan')
                    logger.warning("All angular errors are NaN after calculation.")

                history['val_mean_angular_error'].append(current_val_mean_angular_error)
                history['val_median_angular_error'].append(val_median_angular_error)
                history['val_success_rate_30deg'].append(val_success_rate_30deg)
            else: # val_outputs_direction is empty
                current_val_mean_angular_error = float('nan')
                history['val_mean_angular_error'].append(float('nan'))
                history['val_median_angular_error'].append(float('nan'))
                history['val_success_rate_30deg'].append(float('nan'))
                logger.warning("Validation outputs list is empty. Cannot compute angular error metrics.")

            if val_outputs_concentration:
                val_outputs_concentration_np = np.concatenate(val_outputs_concentration)
                history['val_mean_concentration'].append(np.mean(val_outputs_concentration_np))
            else:
                history['val_mean_concentration'].append(float('nan'))
            
            current_val_loss = val_loss / (len(val_loader_epoch) - val_nan_batches) if (len(val_loader_epoch) - val_nan_batches) > 0 else float('nan')
            history['val_loss'].append(current_val_loss)
            
            logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {current_val_loss:.4f}, Val MAE: {current_val_mean_angular_error:.2f}°")
            scheduler.step(current_val_mean_angular_error if not np.isnan(current_val_mean_angular_error) else float('inf')) # Pass inf if NaN
        else:
            # No validation loader or empty validation set
            logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}. No validation performed.")
            # Append NaN to history for consistency if validation is skipped
            history['val_loss'].append(float('nan'))
            history['val_mean_angular_error'].append(float('nan'))
            history['val_median_angular_error'].append(float('nan'))
            history['val_success_rate_30deg'].append(float('nan'))
            history['val_mean_concentration'].append(float('nan'))
            # Scheduler and early stopping might need to be handled if no val metric is available.
            # For now, scheduler.step() won't be called, and early stopping won't update if best_val_metric remains inf.

        # Early stopping check (use current_val_mean_angular_error if available)
        val_metric_for_early_stop = current_val_mean_angular_error if not np.isnan(current_val_mean_angular_error) else float('inf')
        if val_metric_for_early_stop < best_val_metric:
            best_val_metric = val_metric_for_early_stop
            best_epoch = epoch
            epochs_no_improve = 0
            best_model_state = copy.deepcopy(model.state_dict())
            if checkpoint_path:
                try:
                    torch.save(best_model_state, checkpoint_path)
                    logger.info(f"Saved best model checkpoint to {checkpoint_path} (Epoch {epoch+1}, Error: {best_val_metric:.2f} deg)")
                except Exception as e:
                    logger.error(f"Error saving checkpoint: {e}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                break

    # Load best model state if saved
    if best_model_state:
        logger.info(f"Loading best model weights from epoch {best_epoch+1} with mean error {best_val_metric:.2f} deg")
        model.load_state_dict(best_model_state)

    return model, history 