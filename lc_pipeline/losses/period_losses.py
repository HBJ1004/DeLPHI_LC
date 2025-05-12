#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Loss functions for period prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Union, Any
import logging # Keep logging for consistency if other parts of the file use it

# Logger for this module (optional, if you want to use it elsewhere in this file)
logger = logging.getLogger(__name__)

class ClipPeriodLoss(nn.Module):
    """
    Calculates a clipped loss for period prediction.
    Can operate in log-scaled or linearly-scaled mode.
    """
    def __init__(self, threshold: float = 10.0, log_scaling: bool = False, l2_lambda: float = 0.0, min_log_period: float = 2.0):
        super().__init__()
        self.threshold = threshold
        self.log_scaling = log_scaling
        self.l2_lambda = l2_lambda # Retained for constructor compatibility, but forward usage commented out
        self.min_period_for_log = min_log_period
        # print(f"[ClipPeriodLoss Init] log_scaling: {self.log_scaling}, min_log_period: {self.min_period_for_log}, threshold: {self.threshold}")

    def forward(self, pred: torch.Tensor, target_period_full: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        """
        Args:
            pred (Tensor): Model output. Shape: [batch_size, 1] or [batch_size].
                           If self.log_scaling is True, this is expected to be log10(P_pred).
                           Otherwise, it's raw P_pred.
            target_period_full (Tensor): True period values. Shape: [batch_size, 3] or [batch_size, 1] or [batch_size].
                                       If shape is [batch_size, 3], the actual period is at index 2.
                                       If self.log_scaling is True, this is expected to be raw P_true.
                                       Otherwise, it's raw P_true.
            scale (float): Period scale factor from config. 
                           Only used if self.log_scaling is False.
        Returns:
            Tensor: The calculated mean loss for the batch.
        """
        
        pred = pred.squeeze() # Ensure pred is [batch_size]
        target_period_full_on_device = target_period_full.to(pred.device)

        # Extract the actual period value (P_true) if target_period_full has multiple components
        if target_period_full_on_device.ndim > 1 and target_period_full_on_device.shape[-1] == 3:
            target_period_actual = target_period_full_on_device[:, 2]
        elif target_period_full_on_device.ndim > 1 and target_period_full_on_device.shape[-1] == 1:
            target_period_actual = target_period_full_on_device.squeeze(-1)
        else: # Assumed to be [batch_size] already
            target_period_actual = target_period_full_on_device
        
        # --- Begin Debug Prints ---
        print(f"[ClipPeriodLoss DEBUG] Mode: {'Log-Scaled' if self.log_scaling else 'Linearly-Scaled'}")
        print(f"[ClipPeriodLoss DEBUG] Raw pred sample (model output): {pred[:5].detach().cpu().numpy()}")
        # Print the full target initially to see what's coming in
        print(f"[ClipPeriodLoss DEBUG] Raw target_period_full sample (from dataloader): {target_period_full_on_device[:5].detach().cpu().numpy()}") 
        print(f"[ClipPeriodLoss DEBUG] Extracted target_period_actual sample: {target_period_actual[:5].detach().cpu().numpy()}")
        # --- End Debug Prints ---

        if self.log_scaling:
            # pred is already log10(P_pred)
            pred_transformed = pred

            # Convert P_true (target_period_actual) to log10(P_true), clipping to avoid log(<=0)
            clamped_target_period = torch.clamp(target_period_actual, min=self.min_period_for_log)
            target_transformed = torch.log10(clamped_target_period)
            
            print(f"[ClipPeriodLoss DEBUG LOG] pred_transformed (log10(P_pred)): {pred_transformed[:5].detach().cpu().numpy()}")
            print(f"[ClipPeriodLoss DEBUG LOG] clamped_target_period (P_true_clamped): {clamped_target_period[:5].detach().cpu().numpy()}")
            print(f"[ClipPeriodLoss DEBUG LOG] target_transformed (log10(P_true_clamped)): {target_transformed[:5].detach().cpu().numpy()}")

            loss_elements = torch.abs(pred_transformed - target_transformed)
            
            # # Optional L2 regularization on log-scaled predictions (COMMENTED OUT)
            # if self.l2_lambda > 0:
            #     l2_reg = self.l2_lambda * torch.norm(pred_transformed, p=2) 
            #     loss_elements = loss_elements + l2_reg

        else: # Linear scaling logic
            if scale <= 0:
                print(f"[ClipPeriodLoss WARNING LINEAR] Invalid scale factor {scale}. Defaulting to 1.0.")
                scale = 1.0
            
            pred_transformed = pred / scale
            # Use the extracted actual target period for scaling
            target_transformed = target_period_actual / scale

            print(f"[ClipPeriodLoss DEBUG LINEAR] Scale factor used: {scale}")
            print(f"[ClipPeriodLoss DEBUG LINEAR] pred_transformed (P_pred/scale): {pred_transformed[:5].detach().cpu().numpy()}")
            print(f"[ClipPeriodLoss DEBUG LINEAR] target_transformed (P_true/scale): {target_transformed[:5].detach().cpu().numpy()}")
            
            loss_elements = torch.abs(pred_transformed - target_transformed)

            # Apply threshold clipping if specified
            if self.threshold > 0:
                # Scale the threshold because loss_elements are already scaled
                effective_threshold = self.threshold / scale 
                loss_elements = torch.clamp(loss_elements, max=effective_threshold)
                print(f"[ClipPeriodLoss DEBUG LINEAR] Applied clipping. Effective threshold: {effective_threshold:.4f}")
                print(f"[ClipPeriodLoss DEBUG LINEAR] loss_elements after clamp sample: {loss_elements[:5].detach().cpu().numpy()}")
            
            # # Optional L2 regularization on linearly-scaled predictions (COMMENTED OUT)
            # if self.l2_lambda > 0:
            #     l2_reg = self.l2_lambda * torch.norm(pred_transformed, p=2)
            #     loss_elements = loss_elements + l2_reg
        
        final_loss = loss_elements.mean()

        print(f"[ClipPeriodLoss DEBUG FINAL] Abs diff elements sample (before mean): {loss_elements[:5].detach().cpu().numpy()}")
        print(f"[ClipPeriodLoss DEBUG FINAL] Calculated loss (mean): {final_loss.item()}")
        
        return final_loss


class LogScalePeriodLoss(nn.Module):
    """
    Loss function for period prediction in log-period space.
    
    Implements a loss function that operates in logarithmic period space,
    which helps handle the wide range of possible periods more effectively.
    """
    
    def __init__(
        self,
        min_period: float = 2.0,
        max_period: float = 100.0,
        eps: float = 1e-8
    ):
        """
        Initialize the log-scale period loss.
        
        Args:
            min_period: Minimum expected period
            max_period: Maximum expected period
            eps: Small epsilon to avoid log(0)
        """
        super(LogScalePeriodLoss, self).__init__()
        self.min_period = min_period
        self.max_period = max_period
        self.eps = eps
        
        # Precompute log range
        self.log_min = np.log(min_period)
        self.log_max = np.log(max_period)
        self.log_range = self.log_max - self.log_min
    
    def forward(
        self,
        pred: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate the loss in log-period space.
        
        Args:
            pred: Predicted periods [batch_size, 1] or a tuple of tensors where the first element is the prediction
            target: Target periods [batch_size, 1] or [batch_size, 3] with period at index 2
            
        Returns:
            torch.Tensor: Loss value
        """
        # Handle tuple output from models like PeriodLSTMWithLSPrior
        if isinstance(pred, tuple):
            # Use the first element of the tuple (combined prediction)
            pred = pred[0]
            
        # Extract period component if target has multiple dimensions
        if target.size(-1) > 1:
            target_period = target[..., 2:3]
        else:
            target_period = target
        
        # Filter out invalid targets (negative or zero periods)
        mask = target_period > self.eps
        
        if not mask.any():
            # Return zero loss if no valid targets
            return torch.tensor(0.0, device=pred.device)
        
        # Clip predictions and targets to valid range
        pred_clipped = torch.clamp(pred, min=self.min_period, max=self.max_period)
        target_clipped = torch.clamp(target_period, min=self.min_period, max=self.max_period)
        
        # Convert to log space
        log_pred = torch.log(pred_clipped + self.eps)
        log_target = torch.log(target_clipped + self.eps)
        
        # Normalize to [0, 1] range
        norm_log_pred = (log_pred - self.log_min) / self.log_range
        norm_log_target = (log_target - self.log_min) / self.log_range
        
        # Calculate mean squared error in normalized log space
        mse_loss = F.mse_loss(norm_log_pred[mask], norm_log_target[mask])
        
        return mse_loss


class LSMatchingLoss(nn.Module):
    """
    Loss function that incorporates Lomb-Scargle periodogram matching.
    
    Encourages predictions to match the peaks in the Lomb-Scargle periodogram.
    """
    
    def __init__(
        self,
        base_loss_fn: nn.Module,
        ls_weight: float = 0.3,
        min_ls_peaks: int = 3,
        eps: float = 1e-8
    ):
        """
        Initialize the LS matching loss.
        
        Args:
            base_loss_fn: Base loss function for period prediction
            ls_weight: Weight of the LS matching component
            min_ls_peaks: Minimum number of LS peaks to consider
            eps: Small epsilon to avoid division by zero
        """
        super(LSMatchingLoss, self).__init__()
        self.base_loss_fn = base_loss_fn
        self.ls_weight = ls_weight
        self.min_ls_peaks = min_ls_peaks
        self.eps = eps
    
    def forward(
        self,
        pred: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        target: torch.Tensor,
        raw_data: Optional[Tuple[torch.Tensor, ...]] = None,
        ls_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculate the combined loss with LS matching.
        
        Args:
            pred: Predicted periods [batch_size, 1] or a tuple of tensors where the first element is the prediction
            target: Target periods [batch_size, 1] or [batch_size, 3] with period at index 2
            raw_data: Raw light curve data for LS calculation (optional)
            ls_features: Precomputed LS features [batch_size, n_features] (optional)
            
        Returns:
            torch.Tensor: Loss value
        """
        # Handle tuple output from models like PeriodLSTMWithLSPrior
        if isinstance(pred, tuple):
            # Use the first element of the tuple (combined prediction)
            pred = pred[0]
            
        # Calculate base loss
        base_loss = self.base_loss_fn(pred, target)
        
        # If no LS features or raw data provided, return only base loss
        if ls_features is None and raw_data is None:
            return base_loss
        
        # Use provided LS features or compute them
        if ls_features is not None:
            # Extract LS period (first feature) and power (second feature)
            ls_periods = ls_features[:, 0:1]
            ls_powers = ls_features[:, 1:2] if ls_features.size(1) > 1 else None
        else:
            # This would normally compute LS features from raw data
            # For simplicity, we'll skip actual computation here
            return base_loss
        
        # Filter out invalid targets or LS periods
        mask = (ls_periods > self.eps)
        
        if not mask.any():
            # Return only base loss if no valid LS periods
            return base_loss
        
        # Calculate LS matching loss
        ls_match_loss = F.mse_loss(pred[mask], ls_periods[mask])
        
        # Weight the LS loss by power if available
        if ls_powers is not None:
            ls_powers_norm = ls_powers[mask] / (ls_powers[mask].mean() + self.eps)
            ls_match_loss = (ls_match_loss * ls_powers_norm).mean()
        
        # Combine losses
        total_loss = (1.0 - self.ls_weight) * base_loss + self.ls_weight * ls_match_loss
        
        return total_loss


class MultiHarmonicPeriodLoss(nn.Module):
    """
    Loss function that accounts for period harmonics.
    
    Allows for correct predictions that are harmonics of the true period.
    """
    
    def __init__(
        self,
        base_loss_fn: nn.Module,
        harmonics: torch.Tensor = torch.tensor([0.5, 1.0, 2.0]),
        weights: Optional[torch.Tensor] = None,
        eps: float = 1e-8
    ):
        """
        Initialize multi-harmonic period loss.
        
        Args:
            base_loss_fn: Base loss function
            harmonics: Harmonics to consider [n_harmonics]
            weights: Optional weights for each harmonic [n_harmonics]
            eps: Small epsilon to avoid division by zero
        """
        super(MultiHarmonicPeriodLoss, self).__init__()
        self.base_loss_fn = base_loss_fn
        
        # Register harmonics as buffer
        self.register_buffer('harmonics', harmonics)
        
        # Register weights as buffer or use uniform weights
        if weights is None:
            weights = torch.ones_like(harmonics) / len(harmonics)
        self.register_buffer('weights', weights)
        
        self.eps = eps
    
    def forward(
        self,
        pred: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate the loss considering period harmonics.
        
        Args:
            pred: Predicted periods [batch_size, 1] or a tuple of tensors where the first element is the prediction
            target: Target periods [batch_size, 1] or [batch_size, 3] with period at index 2
            
        Returns:
            torch.Tensor: Loss value
        """
        # Handle tuple output from models like PeriodLSTMWithLSPrior
        if isinstance(pred, tuple):
            # Use the first element of the tuple (combined prediction)
            pred = pred[0]
            
        # Extract period component if target has multiple dimensions
        if target.size(-1) > 1:
            target_period = target[..., 2:3]
        else:
            target_period = target
        
        # Filter out invalid targets
        mask = target_period > self.eps
        
        if not mask.any():
            # Return zero loss if no valid targets
            return torch.tensor(0.0, device=pred.device)
        
        # Calculate harmonic targets
        # [batch_size, 1] -> [batch_size, n_harmonics, 1]
        harmonic_targets = target_period.unsqueeze(1) * self.harmonics.view(1, -1, 1)
        
        # Calculate losses for each harmonic
        # [batch_size, 1] -> [batch_size, 1, 1] -> [batch_size, n_harmonics, 1]
        pred_expanded = pred.unsqueeze(1).expand_as(harmonic_targets)
        
        losses = []
        for i in range(self.harmonics.size(0)):
            # Calculate loss for this harmonic
            harmonic_loss = self.base_loss_fn(pred_expanded[:, i], harmonic_targets[:, i])
            losses.append(harmonic_loss)
        
        # Stack losses and find minimum for each sample
        stacked_losses = torch.stack(losses, dim=1)
        min_loss, _ = torch.min(stacked_losses, dim=1)
        
        # Return mean loss
        return min_loss.mean()


class LogSpacePeriodMAELoss(nn.Module):
    """
    Calculates Mean Absolute Error (MAE) for period prediction in a normalized log space.

    The model is assumed to output a value in [0, 1], representing the normalized
    logarithm (natural log) of the period, scaled between a configured min_period
    and max_period.

    The target periods are provided in linear scale and are transformed to this
    normalized log space within the loss function.
    """
    def __init__(self, min_period_for_scaling: float, max_period_for_scaling: float):
        super().__init__()
        if min_period_for_scaling <= 0 or max_period_for_scaling <= 0:
            raise ValueError("min_period and max_period must be positive for log scaling.")
        if min_period_for_scaling >= max_period_for_scaling:
            raise ValueError("min_period must be less than max_period.")

        self.min_period = torch.tensor(min_period_for_scaling, dtype=torch.float32)
        self.max_period = torch.tensor(max_period_for_scaling, dtype=torch.float32)
        
        self.log_min = torch.log(self.min_period)
        self.log_max = torch.log(self.max_period)
        self.log_range = self.log_max - self.log_min
        
        # Add these aliases to match the attribute names used in train_period.py
        self.actual_log_min_period = self.log_min
        self.actual_log_range = self.log_range

        if self.log_range <= 0:
            raise ValueError(
                f"Log range is not positive. min_period: {min_period_for_scaling}, "
                f"max_period: {max_period_for_scaling}, log_min: {self.log_min.item()}, "
                f"log_max: {self.log_max.item()}"
            )
        
        logger.info(
            f"Initialized LogSpacePeriodMAELoss with min_period={min_period_for_scaling:.2f}, "
            f"max_period={max_period_for_scaling:.2f}"
        )
        logger.debug(
            f"Log-space params: log_min={self.log_min.item():.4f}, "
            f"log_max={self.log_max.item():.4f}, log_range={self.log_range.item():.4f}"
        )

    def forward(self, model_normalized_log_output: torch.Tensor, target_linear_period: torch.Tensor) -> torch.Tensor:
        """
        Args:
            model_normalized_log_output (torch.Tensor): Output from the PeriodLogScaleNet,
                                                       expected to be in [0, 1] (normalized natural log space).
            target_linear_period (torch.Tensor): True target periods in linear scale (e.g., hours).
        Returns:
            torch.Tensor: The MAE loss value.
        """
        # Ensure tensors are on the same device
        current_device = model_normalized_log_output.device
        self.log_min = self.log_min.to(current_device)
        self.log_max = self.log_max.to(current_device)
        self.log_range = self.log_range.to(current_device)
        
        # Also move the aliases to the same device
        self.actual_log_min_period = self.log_min
        self.actual_log_range = self.log_range
        
        min_p = self.min_period.to(current_device)
        max_p = self.max_period.to(current_device)

        # 1. Clamp and transform target_linear_period to normalized log space
        # Ensure target_linear_period is float
        target_linear_period = target_linear_period.float()
        
        clipped_target_linear_period = torch.clamp(target_linear_period, min=min_p, max=max_p)
        target_log_period = torch.log(clipped_target_linear_period) # Natural log
        
        # Handle potential division by zero if log_range is somehow zero (checked in init, but defensive)
        if self.log_range.item() == 0:
             # This case should ideally be prevented by __init__ checks
            logger.warning("Log range is zero during forward pass. This might lead to NaN/Inf loss.")
            # Return a high loss or handle as error, for now, let it proceed to see if PyTorch handles it
            # Or, more robustly:
            if torch.all(target_log_period == self.log_min): # All targets are at min_period
                 target_normalized_log_output = torch.zeros_like(target_log_period)
            else: # Should not happen if log_range is 0 and periods are clamped
                 target_normalized_log_output = torch.zeros_like(target_log_period) # Fallback
        else:
            target_normalized_log_output = (target_log_period - self.log_min) / self.log_range
        
        # Ensure model_normalized_log_output is also float (usually is from sigmoid)
        model_normalized_log_output = model_normalized_log_output.float()

        # Debug logging for a few samples
        if logger.isEnabledFor(logging.DEBUG) and model_normalized_log_output.numel() > 0 and target_normalized_log_output.numel() > 0:
            num_samples_to_log = min(5, model_normalized_log_output.shape[0])
            logger.debug(f"[LogSpacePeriodMAELoss] Model normalized output (sample): {model_normalized_log_output.flatten()[:num_samples_to_log].tolist()}")
            logger.debug(f"[LogSpacePeriodMAELoss] Target linear input (sample): {target_linear_period.flatten()[:num_samples_to_log].tolist()}")
            logger.debug(f"[LogSpacePeriodMAELoss] Target normalized output (sample): {target_normalized_log_output.flatten()[:num_samples_to_log].tolist()}")

        # 2. Calculate MAE
        loss = F.l1_loss(model_normalized_log_output, target_normalized_log_output, reduction='mean')
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[LogSpacePeriodMAELoss] Calculated loss: {loss.item()}")
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"NaN or Inf loss detected in LogSpacePeriodMAELoss: {loss.item()}")

        return loss 