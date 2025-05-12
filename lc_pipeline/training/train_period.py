#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training functions for period prediction models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union
import time
import copy
from tqdm import tqdm
from pathlib import Path
import os

from lc_pipeline.losses.period_losses import ClipPeriodLoss, LogScalePeriodLoss, LogSpacePeriodMAELoss
from lc_pipeline.utils.training import EarlyStopping


def train_period_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: str,
    config: Any,
    logger: Any = None,
    debug: bool = False,
    checkpoint_path: Optional[str] = None,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Train a period prediction model.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        config: Configuration object (e.g., PipelineConfig.period_model or similar containing all period model settings)
        logger: Logger instance
        debug: Enable debug mode with verbose logging
        checkpoint_path: Path to save the best model checkpoint
        
    Returns:
        tuple: (best_trained_model, history_dict)
    """
    # Extract relevant period_model_config (adjust if config structure is different)
    # This is a common pattern, assuming config is the top-level PipelineConfig
    if hasattr(config, 'period_model'):
        period_model_config = config.period_model
    else:
        # If config passed is already period_model_config or a direct dict
        period_model_config = config 

    num_epochs = period_model_config.epochs
    learning_rate = period_model_config.lr
    weight_decay = period_model_config.weight_decay
    patience = period_model_config.patience

    # Ensure model is on the correct device
    model = model.to(device)
    
    # Force all model parameters to be on the specified device
    for param in model.parameters():
        param.data = param.data.to(device)
    
    # If model has lstm attribute, ensure it's properly moved to device (fixes device mismatch issue)
    if hasattr(model, 'lstm') and model.lstm is not None:
        model.lstm = model.lstm.to(device)
        # Store device for dynamic LSTM creation in forward pass
        model.device = device
    
    # Log model and optimizer info
    if logger:
        logger.info(f"Model: {model.__class__.__name__}")
        logger.info(f"Optimizer: Adam(lr={learning_rate}, weight_decay={weight_decay})")
        logger.info(f"Selected Loss Function: {period_model_config.loss_function}")
        logger.info(f"Device: {device}")
        
        # Log model architecture details in debug mode
        if debug:
            logger.debug(f"Model architecture:\n{model}")
            # Count and log number of parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.debug(f"Total parameters: {total_params:,}, Trainable: {trainable_params:,}")
    
    # Set model to training mode
    model.train()
    
    # Define loss function based on config
    if period_model_config.loss_function == "LogSpacePeriodMAELoss":
        criterion = LogSpacePeriodMAELoss(
            min_period_for_scaling=period_model_config.min_period,
            max_period_for_scaling=period_model_config.max_period
        )
        if logger: logger.info(f"Using LogSpacePeriodMAELoss with min_period={period_model_config.min_period}, max_period={period_model_config.max_period}")
    elif period_model_config.loss_function == "ClipPeriodLoss":
        # Assuming ClipPeriodLoss might still need period_scale_factor from config
        # and its own log_scaling parameters (which might differ from PeriodLogScaleNet's use_log_scale)
        # For simplicity, I'm matching the old parameters if they are still in period_model_config
        # ClipPeriodLoss in the file has: log_scaling, min_log_period, l2_lambda
        # These might need to be added to PeriodModelConfig if ClipPeriodLoss is to be fully configurable.
        # For now, using available fields that make sense.
        clip_loss_log_scaling = getattr(period_model_config, 'clip_loss_log_scaling', False) # Default if not in config
        clip_loss_min_log_period = getattr(period_model_config, 'clip_loss_min_log_period', 2.0) # Default
        clip_loss_l2_lambda = getattr(period_model_config, 'clip_loss_l2_lambda', 0.0) # Default
        criterion = ClipPeriodLoss(
            log_scaling=clip_loss_log_scaling,
            min_log_period=clip_loss_min_log_period, 
            l2_lambda=clip_loss_l2_lambda 
            # period_scale_factor might still be needed by ClipPeriodLoss if not in log_scaling mode.
            # If so, it should be part of period_model_config, e.g., period_model_config.period_scale_factor
        )
        if logger: logger.info(f"Using ClipPeriodLoss with log_scaling={clip_loss_log_scaling}, min_log_period={clip_loss_min_log_period}")
    else:
        raise ValueError(f"Unsupported loss function: {period_model_config.loss_function}")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Construct the full path for the checkpoint file if a directory is given
    early_stopping_save_path = None
    # --- BEGIN DEBUG PRINT ---
    print(f"[train_period_model DEBUG] Received checkpoint_path argument: {checkpoint_path} (type: {type(checkpoint_path)})")
    # ---  END DEBUG PRINT  ---
    if checkpoint_path:
        # Ensure checkpoint_path is treated as string path for consistency
        checkpoint_path_str = str(checkpoint_path)
        
        # Check if it's a directory
        if os.path.isdir(checkpoint_path_str):
            # Use a fixed name for the best model checkpoint within this training run
            early_stopping_save_path = os.path.join(checkpoint_path_str, "best_period_model_checkpoint.pt")
        else:
            # If it was already a file path, use it as is
            early_stopping_save_path = checkpoint_path_str
        
        # --- BEGIN DEBUG PRINT ---
        print(f"[train_period_model DEBUG] Constructed early_stopping_save_path: {early_stopping_save_path} (type: {type(early_stopping_save_path)})")
        # ---  END DEBUG PRINT  ---

        # Ensure parent directory for the save path exists
        if early_stopping_save_path:
            os.makedirs(os.path.dirname(early_stopping_save_path), exist_ok=True)
            if logger:
                logger.info(f"Early stopping checkpoint will be saved to: {early_stopping_save_path}")
        elif logger: # Log if original checkpoint_path was a dir but somehow failed to form a file path
             logger.warning(f"Could not form a valid file path from checkpoint_path: {checkpoint_path}")

    # Early stopping setup
    # --- BEGIN DEBUG PRINT ---
    print(f"[train_period_model DEBUG] Path passed to EarlyStopping constructor: {early_stopping_save_path} (type: {type(early_stopping_save_path)})")
    # ---  END DEBUG PRINT  ---
    early_stopping = EarlyStopping(
        patience=patience, 
        min_delta=0.001, 
        mode="min", 
        verbose=True,
        save_path=early_stopping_save_path, # Pass string path
        logger=logger
    )
    
    # Initialize history dictionary
    history = {'train_loss': [], 'val_loss': [], 'val_mae': [], 'val_rmse': []} # Add other metrics if calculated
    
    # Check if model has expected sequence dimension handling method
    has_sequence_dim_handling = hasattr(model, 'use_packed_sequence') or hasattr(model, 'use_attention_mask')
    
    # Inspect a sample batch for debugging purposes
    if debug and logger:
        try:
            sample_batch = next(iter(train_loader))
            data, targets, ids, lengths = sample_batch
            
            logger.debug(f"Sample batch data shape: {data.shape}")
            logger.debug(f"Sample batch targets shape: {targets.shape}")
            logger.debug(f"Sample batch sequence lengths: min={min(lengths)}, max={max(lengths)}, mean={sum(lengths)/len(lengths):.1f}")
            
            # Log memory usage before model forward pass
            if torch.cuda.is_available():
                logger.debug(f"GPU memory before forward pass: {torch.cuda.memory_allocated(device=device)/1024**2:.1f}MB")
            
            # Try a test forward pass
            model.eval()
            with torch.no_grad():
                sample_data = data.to(device)
                sample_ls_features = None
                if len(sample_batch) > 4:
                    sample_ls_features = sample_batch[4]
                    if isinstance(sample_ls_features, torch.Tensor):
                        sample_ls_features = sample_ls_features.to(device)
                
                logger.debug(f"Running test forward pass...")
                _ = model(sample_data, lengths, sample_ls_features)
                logger.debug(f"Test forward pass successful")
            
            # Return to train mode
            model.train()
        except Exception as e:
            logger.error(f"Error in sample batch inspection: {str(e)}")
            if "size mismatch" in str(e) or "invalid for input of size" in str(e):
                logger.error(f"Detected shape mismatch, please check model input tensor shapes")
                # Log model input dimensions expected
                if hasattr(model, 'input_dim'):
                    logger.debug(f"Model expects input_dim={model.input_dim}")
                if hasattr(model, 'hidden_dim'):
                    logger.debug(f"Model uses hidden_dim={model.hidden_dim}")
    
    # Training loop
    try:
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Training phase
            model.train()
            epoch_loss = 0
            train_samples_processed_epoch = 0 # Initialize counter for samples in epoch
            train_loader_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
            nan_batches_train = 0
        
            for batch_idx, batch_content in enumerate(train_loader_bar):
                # Assuming collate_fn provides: (padded_data, targets, ids, lengths)
                # The ids are not typically used in this training loop but are part of the standard collate output.
                # LS features might be added later by a more specialized collate_fn if needed.
                data, period_targets, _ids, lengths = batch_content
                ls_features = None # Placeholder for now, would come from a different collate if used

                data, period_targets = data.to(device), period_targets.to(device)
                lengths = lengths.to(device) # Ensure lengths are on the correct device if model uses them
                if ls_features is not None:
                    ls_features = ls_features.to(device)
                    
                # Check for potential issues with data shapes
                if isinstance(data, torch.Tensor) and data.numel() == 0:
                    if logger:
                        logger.warning(f"Skipping empty batch at index {batch_idx}")
                    continue
                
                # Check for anomalies in lengths
                if hasattr(lengths, 'min') and lengths.min() <= 0: # Check if lengths is a tensor and has min method
                    if logger:
                        logger.warning(f"Batch {batch_idx} has invalid sequence lengths: {lengths}")
                    continue
                
                # Detailed shape verification in debug mode
                if debug and logger and batch_idx == 0:
                    logger.debug(f"Batch data shape: {data.shape}")
                    logger.debug(f"Expected input shape format: [batch_size, seq_len, input_features]")
                    if hasattr(model, 'input_dim'):
                        # Verify correct feature dimension
                        if data.shape[-1] != getattr(model, 'input_dim'):
                            logger.warning(f"Input feature dimension mismatch: got {data.shape[-1]}, expected {getattr(model, 'input_dim')}")
                
                # Forward pass (with error checking)
                # Safe mode that explicitly verifies the shapes before going into the LSTM
                try:
                    # Verify data dimensions for debugging
                    if debug and logger and batch_idx == 0:
                        data_elements = data.numel()
                        data_shape = data.shape
                        if logger:
                            logger.debug(f"Data tensor shape: {data_shape}, elements: {data_elements}")
                            logger.debug(f"Sequence lengths: min={min(lengths)}, max={max(lengths)}")
                            logger.debug(f"Batch size: {data.shape[0]}")
                            if has_sequence_dim_handling:
                                logger.debug(f"Model uses {'packed sequences' if hasattr(model, 'use_packed_sequence') else 'attention mask'}")
            
                    # Forward pass
                    outputs = model(data, lengths, ls_features)
                    
                    # Handle tuple output from models like PeriodLSTMWithLSPrior
                    if isinstance(outputs, tuple):
                        outputs_for_loss = outputs[0] # Use the main combined prediction for loss
                    else:
                        outputs_for_loss = outputs
                    
                    # Calculate loss
                    # Use a distinct variable name for clarity within the batch loop
                    current_batch_loss = criterion(outputs_for_loss, period_targets)
            
                    # Backward pass and optimization
                    optimizer.zero_grad()
                    current_batch_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()

                    # Accumulate loss for the epoch only if batch processing was successful
                    epoch_loss += current_batch_loss.item() * data.size(0)
                    train_samples_processed_epoch += data.size(0) # Accumulate processed samples

                except RuntimeError as e:
                    if "invalid for input of size" in str(e) or "size mismatch" in str(e) or "Sizes of tensors must match" in str(e):
                        # This is a shape error - log detailed diagnostics
                        nan_batches_train += 1
                        if logger:
                            logger.error(f"Shape mismatch in batch {batch_idx}: {str(e)}")
                            logger.error(f"Batch shape: {data.shape}, sequence lengths: min={min(lengths)}, max={max(lengths)}")
                            
                            # Get model-specific shape information
                            if hasattr(model, 'input_dim'):
                                logger.error(f"Model expects input_dim={model.input_dim}")
                            if hasattr(model, 'hidden_dim'):
                                logger.error(f"Model uses hidden_dim={model.hidden_dim}")
                            
                            # LSTM specific debug
                            if "LSTM" in model.__class__.__name__:
                                h_size = getattr(model, 'hidden_dim', 128)
                                expected_size = data.shape[0] * h_size
                                logger.error(f"LSTM expected size: batch_size({data.shape[0]}) * hidden_dim({h_size}) = {expected_size}")
                            
                            # Debug memory for potential OOM
                            if torch.cuda.is_available():
                                logger.error(f"GPU memory: {torch.cuda.memory_allocated(device=device)/1024**2:.1f}MB / "
                                          f"{torch.cuda.max_memory_allocated(device=device)/1024**2:.1f}MB")
                        
                        # Skip this batch and continue
                        continue
                    else:
                        # Other runtime error, re-raise
                        raise
            
                except Exception as forward_err:
                    # Catch any other unexpected error during forward pass
                    if logger:
                        logger.error(f"Unexpected error during forward pass in batch {batch_idx}: {str(forward_err)}", exc_info=True)
                    continue # Skip batch
            
            # Log shape error statistics
            if nan_batches_train > 0 and logger:
                logger.warning(f"Epoch {epoch+1}: Encountered {nan_batches_train} shape mismatch errors")
        
            # Calculate average training loss for the epoch
            if train_samples_processed_epoch > 0:
                avg_train_loss = epoch_loss / train_samples_processed_epoch
            else:
                avg_train_loss = float('nan') # Avoid division by zero if no samples were processed
            history['train_loss'].append(avg_train_loss)
        
            # Validation phase
            model.eval()
            val_epoch_loss = 0
            val_samples_processed_epoch = 0 # Initialize counter for validation samples
            val_predictions = []
            val_true_periods = []
            val_loader_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            nan_batches_val = 0
        
            # Store raw model outputs and true linear targets for metric calculation
            all_raw_model_outputs_val = [] # For LogSpacePeriodMAELoss, these are normalized log-space
            all_true_linear_periods_val = []

            with torch.no_grad():
                for batch_idx, batch_content in enumerate(val_loader_bar):
                    data, period_targets_batch, _ids, lengths = batch_content
                    ls_features = None

                    data = data.to(device)
                    period_targets_batch = period_targets_batch.to(device)
                    lengths = lengths.to(device)
                    if ls_features is not None:
                        ls_features = ls_features.to(device)
                
                    outputs = model(data, lengths, ls_features)
                
                    if isinstance(outputs, tuple):
                        outputs_for_loss = outputs[0]
                        current_preds_for_metrics = outputs[0] 
                    else:
                        outputs_for_loss = outputs
                        current_preds_for_metrics = outputs
            
                    val_loss_value = criterion(outputs_for_loss, period_targets_batch)
                
                    val_epoch_loss += val_loss_value.item() * data.size(0)
                    val_samples_processed_epoch += data.size(0)
                        
                    if current_preds_for_metrics.ndim > 1 and current_preds_for_metrics.shape[-1] == 1:
                        current_preds_for_metrics = current_preds_for_metrics.squeeze(-1)
                    
                    all_raw_model_outputs_val.extend(current_preds_for_metrics.cpu().numpy())
                    all_true_linear_periods_val.extend(period_targets_batch[:, 2].cpu().numpy()) # True periods are at index 2
                    
                    if torch.isnan(val_loss_value) or torch.isinf(val_loss_value):
                        nan_batches_val += 1
            
            val_loss = val_epoch_loss / val_samples_processed_epoch if val_samples_processed_epoch > 0 else float('nan')
            history['val_loss'].append(val_loss)
            
            val_mae = float('nan')
            val_rmse = float('nan')

            if all_raw_model_outputs_val and all_true_linear_periods_val:
                try:
                    from sklearn.metrics import mean_absolute_error, mean_squared_error
                    import numpy as np

                    if isinstance(criterion, LogSpacePeriodMAELoss):
                        # Corrected attribute names accessed from LogSpacePeriodMAELoss instance
                        log_min_period = criterion.actual_log_min_period 
                        log_range = criterion.actual_log_range
                        
                        # Convert PyTorch tensors to NumPy arrays
                        log_min_period_np = log_min_period.cpu().detach().numpy()
                        log_range_np = log_range.cpu().detach().numpy()
                        
                        np_preds_normalized_log = np.array(all_raw_model_outputs_val)
                        linear_predictions = np.exp(np_preds_normalized_log * log_range_np + log_min_period_np)
                        
                        val_mae = mean_absolute_error(all_true_linear_periods_val, linear_predictions)
                        val_rmse = np.sqrt(mean_squared_error(all_true_linear_periods_val, linear_predictions))
                    else:
                        val_mae = mean_absolute_error(all_true_linear_periods_val, all_raw_model_outputs_val)
                        val_rmse = np.sqrt(mean_squared_error(all_true_linear_periods_val, all_raw_model_outputs_val))

                    history['val_mae'].append(val_mae)
                    history['val_rmse'].append(val_rmse)
                    if logger:
                        logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}, Val RMSE: {val_rmse:.4f}")
                except ImportError:
                    if logger:
                        logger.warning("sklearn.metrics not found. Cannot calculate MAE/RMSE.")
                    history['val_mae'].append(float('nan'))
                    history['val_rmse'].append(float('nan'))
                    if logger: # Log without MAE/RMSE
                        logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                # Append NaN if metrics couldn't be calculated
                history['val_mae'].append(float('nan'))
                history['val_rmse'].append(float('nan'))
                if logger:
                    logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Step the scheduler
            scheduler.step(val_loss)
            
            # Early stopping check
            if early_stopping(val_loss, model):
                if logger:
                    logger.info(f"Early stopping after {epoch+1} epochs")
                break
    except Exception as train_err:
        if logger:
            logger.error(f"Fatal error during training loop: {str(train_err)}", exc_info=True)
        raise
    finally:
        if logger:
            logger.info("Loading best model weights")
        best_score = early_stopping.load_best_model(model)
        if best_score is not None:
            logger.info(f"Successfully loaded best model from checkpoint with validation score: {best_score:.4f}")
        else:
            logger.warning("Failed to load best model from checkpoint, returning last model state.")

    model.eval()

    return model, history


def train_enhanced_period_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    ls_features_train: torch.Tensor,
    ls_features_val: torch.Tensor,
    device: str,
    num_epochs: int = 100,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 10,
    ls_weight: float = 0.5,
    logger: Any = None,
    # Adding missing parameters that seem to be used internally,
    # or were intended to be passed, based on context and errors.
    # These are placeholders; the calling code would need to provide them.
    early_stopping: Any = None, # Placeholder for EarlyStopping object
    history: dict = None # Placeholder for history dictionary
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Train an enhanced period prediction model with LS features.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        ls_features_train: LS features for training set
        ls_features_val: LS features for validation set
        device: Device to use (cuda or cpu)
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        patience: Patience for early stopping
        ls_weight: Weight for LS prior influence
        logger: Optional logger
        early_stopping: EarlyStopping instance (placeholder)
        history: Dictionary to store training history (placeholder)
        
    Returns:
        tuple: (trained_model, history_dict)
    """
    model.to(device)
    
    # Define loss function
    if hasattr(model, 'base_model') and hasattr(model, 'log_min'):
        # For log-scale models
        criterion = LogScalePeriodLoss(
            min_period=model.min_period,
            max_period=model.max_period
        )
    else:
        # For regular models
        criterion = ClipPeriodLoss(threshold=0.2, log_scaling=True)
    
    # Define optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=patience // 2
    )
    
    # Set LS weight if model has it
    if hasattr(model, 'prior_weight'):
        model.prior_weight = ls_weight
    
    # Training loop variables
    # These are not used if early_stopping object handles best model logic
    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0
    
    # Initialize history if not provided (basic initialization)
    if history is None:
        history = {'train_loss': [], 'val_loss': [], 'val_mae': [], 'val_rmse': []}

    # Placeholder for val_preds and val_targets_period if they are supposed to be collected
    # This is a structural assumption, actual implementation might differ.
    val_preds_list = []
    val_targets_period_list = []
    
    # Log start of training
    if logger:
        logger.info(f"Starting enhanced period model training for {num_epochs} epochs")
        logger.info(f"Model: {model.__class__.__name__}")
        logger.info(f"Optimizer: Adam(lr={learning_rate}, weight_decay={weight_decay})")
        logger.info(f"Loss: {criterion.__class__.__name__}")
        logger.info(f"LS Weight: {ls_weight}")
        logger.info(f"Device: {device}")
    
    # Training loop
    try:
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Training phase
            model.train()
            train_loss = 0.0
            train_samples = 0
            
            for batch_idx, (data, targets, ids, lengths) in enumerate(train_loader):
                # Move data to device
                data = data.to(device)
                targets = targets.to(device)
                lengths = lengths.to(device) # Assuming lengths is already a tensor or tensor-like
                
                # Get LS features for this batch
                # Ensure batch_size is correctly inferred if train_loader.batch_size is None (e.g. for iterable datasets)
                current_batch_size = data.size(0) # Use actual data size for robustness
                start_idx = batch_idx * train_loader.batch_size if train_loader.batch_size is not None else batch_idx * current_batch_size # Handle iterable dataset case
                end_idx = start_idx + current_batch_size
                
                batch_ls_features = ls_features_train[start_idx:end_idx]
                batch_ls_features = batch_ls_features.to(device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(data, lengths, batch_ls_features)
                
                # If model returns multiple outputs
                if isinstance(outputs, tuple):
                    outputs = outputs[0]  # Use the combined prediction
                
                # Calculate loss
                loss = criterion(outputs, targets)
                
                # Backpropagation
                loss.backward()
                
                # Update parameters
                optimizer.step()
                
                # Update statistics
                train_loss += loss.item() * data.size(0)
                train_samples += data.size(0)
            
            # Calculate average training loss
            avg_train_loss = train_loss / train_samples if train_samples > 0 else 0
            history['train_loss'].append(avg_train_loss)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_samples = 0
            # Reset lists for MAE/RMSE calculation per epoch
            val_preds_list.clear()
            val_targets_period_list.clear()
            
            with torch.no_grad():
                for batch_idx, batch_val in enumerate(val_loader): # Renamed batch to batch_val
                    try:
                        # Extract data and move to device
                        data, targets, ids, lengths = batch_val
                        data = data.to(device)
                        targets = targets.to(device)

                        # Move lengths to device as well
                        if isinstance(lengths, torch.Tensor):
                            lengths = lengths.to(device)
                        else:
                            # If lengths is not a tensor (e.g., list/tuple), convert it
                            try:
                                lengths = torch.tensor(lengths, dtype=torch.long).to(device)
                            except Exception as e:
                                if logger:
                                    logger.error(f"Failed to convert lengths to tensor on device {device}: {e}")
                                continue
                        
                        # Extract period component for loss calculation
                        period_targets = targets[:, 2] # Assuming targets has shape [batch_size, num_target_components]
                        
                        # Get LS features for this validation batch
                        current_batch_size_val = data.size(0) # Use actual data size
                        start_idx_val = batch_idx * val_loader.batch_size if val_loader.batch_size is not None else batch_idx * current_batch_size_val
                        end_idx_val = start_idx_val + current_batch_size_val

                        batch_ls_features_val = ls_features_val[start_idx_val:end_idx_val]
                        batch_ls_features_val = batch_ls_features_val.to(device)
                
                        # Forward pass
                        outputs = model(data, lengths, batch_ls_features_val)
                
                        # Handle tuple output from models like PeriodLSTMWithLSPrior
                        if isinstance(outputs, tuple):
                            outputs_for_loss = outputs[0] # Use the main combined prediction for loss
                            # Assuming the second element might be raw predictions for MAE/RMSE if needed
                            # For now, use outputs_for_loss for predictions as well if not specified otherwise
                            # Example: raw_preds = outputs[1] 
                        else:
                            outputs_for_loss = outputs
                
                        # Calculate loss
                        loss = criterion(outputs_for_loss, period_targets)
                
                        # Record loss
                        val_loss += loss.item() * data.size(0)
                        val_samples += data.size(0)

                        # For MAE/RMSE (assuming outputs_for_loss are the predictions)
                        # This part needs clarification on what `val_preds` and `val_targets_period` should be.
                        # Storing all predictions and targets to calculate metrics once after the loop.
                        val_preds_list.extend(outputs_for_loss.cpu().numpy())
                        val_targets_period_list.extend(period_targets.cpu().numpy())
                            
                    except Exception as val_batch_err:
                        if logger:
                            logger.error(f"Error processing validation batch: {str(val_batch_err)}", exc_info=True)
                        continue # Skip this batch
            
            # Calculate average validation loss
            avg_val_loss = val_loss / val_samples if val_samples > 0 else float('nan')
            history['val_loss'].append(avg_val_loss)
            
            val_mae = float('nan')
            val_rmse = float('nan')

            # Calculate validation MAE/RMSE if predictions were collected
            if val_preds_list and val_targets_period_list:
                try:
                    from sklearn.metrics import mean_absolute_error, mean_squared_error
                    import numpy as np # Ensure numpy is imported
                    val_mae = mean_absolute_error(val_targets_period_list, val_preds_list)
                    val_rmse = np.sqrt(mean_squared_error(val_targets_period_list, val_preds_list))
                    history['val_mae'].append(val_mae)
                    history['val_rmse'].append(val_rmse)
                    if logger:
                        logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val MAE: {val_mae:.4f}, Val RMSE: {val_rmse:.4f}")
                except ImportError:
                    if logger:
                        logger.warning("sklearn.metrics not found. Cannot calculate MAE/RMSE.")
                    history['val_mae'].append(float('nan'))
                    history['val_rmse'].append(float('nan'))
                    if logger: # Log without MAE/RMSE
                        logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            else:
                # Append NaN if metrics couldn't be calculated
                history['val_mae'].append(float('nan'))
                history['val_rmse'].append(float('nan'))
                if logger:
                    logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # Step the scheduler
            scheduler.step(avg_val_loss)
            
            # Early stopping check
            if early_stopping is not None:
                if early_stopping(avg_val_loss, model):
                    if logger:
                        logger.info(f"Early stopping after {epoch+1} epochs")
                    break
            else: # Fallback if no early stopping object is provided (train for num_epochs)
                # Check for simple patience-based improvement if no early_stopping object
                # This part is a simple alternative, real early_stopping object is preferred
                if avg_val_loss < best_val_loss: # best_val_loss needs to be initialized for this path
                    best_val_loss = avg_val_loss
                    best_model_state = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                    if logger:
                        logger.info(f"Validation loss improved to {avg_val_loss:.4f}. Saving model state (fallback).")
                else:
                    epochs_no_improve += 1
                    if logger:
                        logger.info(f"No improvement in validation loss for {epochs_no_improve} epochs (fallback).")
                if epochs_no_improve >= patience:
                    if logger:
                        logger.info(f"Early stopping based on patience after {epoch+1} epochs (fallback).")
                    break
    
    except Exception as train_err:
        if logger:
            logger.error(f"Fatal error during training loop: {str(train_err)}", exc_info=True)
        raise # Re-raise the exception after logging
    
    finally:
        # Load the best model state saved by EarlyStopping (if provided)
        if early_stopping is not None and hasattr(early_stopping, 'load_best_model'):
            if logger:
                logger.info("Loading best model weights via early_stopping object.")
            best_score = early_stopping.load_best_model(model)
            if best_score is not None:
                if logger:
                    logger.info(f"Successfully loaded best model from checkpoint with validation score: {best_score:.4f}")
            else:
                if logger:
                    logger.warning("Failed to load best model from checkpoint via early_stopping, returning last model state.")
        elif best_model_state is not None: # Fallback if using internal best_model_state
            if logger:
                logger.info("Loading best model weights from internal state (fallback).")
            model.load_state_dict(best_model_state)
            if logger:
                logger.info(f"Successfully loaded best model from internal state with validation loss: {best_val_loss:.4f}")
        else:
            if logger:
                logger.info("No best model state to load (no early_stopping or internal best_model_state). Returning current model state.")

    # Ensure model is in eval mode before returning
    model.eval()

    # The function signature implies only model is returned.
    # If history is needed, the signature and return statement should be updated.
    # For now, adhering to the original signature.
    return model, history 