#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Loss functions for axis prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Union


def geodesic_distance(q1, q2):
    """
    Calculate geodesic distance between two sets of quaternions.
    
    Args:
        q1: First quaternion [batch_size, 4]
        q2: Second quaternion [batch_size, 4]
        
    Returns:
        torch.Tensor: Geodesic distances [batch_size]
    """
    # Normalize quaternions
    q1_norm = F.normalize(q1, p=2, dim=1)
    q2_norm = F.normalize(q2, p=2, dim=1)
    
    # Compute dot product, clamping for numerical stability
    dot_prod = torch.sum(q1_norm * q2_norm, dim=1).clamp(-1.0, 1.0)
    
    # Taking absolute value because antipodal quaternions represent same rotation
    dot_prod = torch.abs(dot_prod)
    
    # Calculate geodesic distance (angle in radians)
    angle = 2.0 * torch.acos(dot_prod)
    
    return angle


def angular_distance(v1, v2):
    """
    Calculate angular distance between two sets of direction vectors.
    
    Args:
        v1: First direction vector [batch_size, 3]
        v2: Second direction vector [batch_size, 3]
        
    Returns:
        torch.Tensor: Angular distances [batch_size]
    """
    # Normalize vectors
    v1_norm = F.normalize(v1, p=2, dim=1)
    v2_norm = F.normalize(v2, p=2, dim=1)
    
    # Compute dot product, clamping for numerical stability
    dot_prod = torch.sum(v1_norm * v2_norm, dim=1).clamp(-1.0, 1.0)
    
    # Calculate angular distance (angle in radians)
    angle = torch.acos(dot_prod)
    
    return angle


class GeodesicLoss(nn.Module):
    """
    Loss function based on geodesic distance between rotations.
    
    For quaternions, this is the angle between rotations on SO(3).
    For direction vectors, this is the angle between vectors on S^2.
    """
    
    def __init__(self, use_quaternions=True, reduction='mean'):
        """
        Initialize the geodesic loss.
        
        Args:
            use_quaternions: Whether inputs are quaternions or direction vectors
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super(GeodesicLoss, self).__init__()
        self.use_quaternions = use_quaternions
        self.reduction = reduction
    
    def forward(self, pred, target):
        """
        Calculate the geodesic loss.
        
        Args:
            pred: Predicted rotations [batch_size, 4] or [batch_size, 3]
            target: Target rotations [batch_size, 4] or [batch_size, 3]
            
        Returns:
            torch.Tensor: Loss value
        """
        # Choose distance function based on input type
        if self.use_quaternions:
            distances = geodesic_distance(pred, target)
        else:
            distances = angular_distance(pred, target)
        
        # Apply reduction
        if self.reduction == 'mean':
            return distances.mean()
        elif self.reduction == 'sum':
            return distances.sum()
        else:  # 'none'
            return distances


class VMFLoss(nn.Module):
    """
    Von Mises-Fisher loss for directional statistics.
    
    This loss treats rotation prediction as sampling from a
    von Mises-Fisher distribution on the sphere (S^2).
    """
    
    def __init__(self, concentration=20.0, use_antipodal=True, reduction='mean'):
        """
        Initialize the VMF loss.
        
        Args:
            concentration: Concentration parameter of VMF distribution
            use_antipodal: Whether to account for antipodal symmetry
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super(VMFLoss, self).__init__()
        self.concentration = concentration
        self.use_antipodal = use_antipodal
        self.reduction = reduction
    
    def forward(self, pred, target):
        """
        Calculate the VMF loss.
        
        Args:
            pred: Predicted directions [batch_size, 3]
            target: Target directions [batch_size, 3]
            
        Returns:
            torch.Tensor: Loss value
        """
        # Normalize vectors
        pred_norm = F.normalize(pred, p=2, dim=1)
        target_norm = F.normalize(target, p=2, dim=1)
        
        # Compute dot product
        dot_prod = torch.sum(pred_norm * target_norm, dim=1)
        
        # Account for antipodal symmetry if needed
        if self.use_antipodal:
            dot_prod = torch.abs(dot_prod)
        
        # VMF negative log-likelihood (ignoring normalization constant)
        # Higher dot product (closer vectors) should have lower loss
        loss = -self.concentration * dot_prod
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class GeodesicVMFCombinedLoss(nn.Module):
    """
    Combined loss using both Geodesic distance and VMF distribution.
    
    This combines the benefits of both loss functions:
    - Geodesic distance provides a direct measure of angular error
    - VMF loss models the directional uncertainty
    """
    
    def __init__(
        self,
        geodesic_weight=0.7,
        vmf_weight=0.3,
        concentration=20.0,
        use_quaternions=True,
        reduction='mean'
    ):
        """
        Initialize the combined loss.
        
        Args:
            geodesic_weight: Weight for geodesic loss
            vmf_weight: Weight for VMF loss
            concentration: Concentration parameter for VMF loss
            use_quaternions: Whether inputs are quaternions
            reduction: Reduction method
        """
        super(GeodesicVMFCombinedLoss, self).__init__()
        
        self.geodesic_weight = geodesic_weight
        self.vmf_weight = vmf_weight
        self.use_quaternions = use_quaternions
        self.reduction = reduction
        
        # Create component losses
        self.geodesic_loss = GeodesicLoss(
            use_quaternions=use_quaternions,
            reduction=reduction
        )
        
        # VMF loss requires direction vectors
        self.vmf_loss = VMFLoss(
            concentration=concentration,
            reduction=reduction
        )
    
    def forward(self, pred, target):
        """
        Calculate the combined loss.
        
        Args:
            pred: Predicted rotations [batch_size, 4] or [batch_size, 3]
            target: Target rotations [batch_size, 4] or [batch_size, 3]
            
        Returns:
            torch.Tensor: Loss value
        """
        # Calculate geodesic loss
        geo_loss = self.geodesic_loss(pred, target)
        
        # Convert quaternions to direction vectors for VMF loss if needed
        if self.use_quaternions:
            from lc_pipeline.models.utils import quaternion_to_direction_vector
            pred_vec = quaternion_to_direction_vector(pred)
            target_vec = quaternion_to_direction_vector(target)
        else:
            pred_vec = pred
            target_vec = target
        
        # Calculate VMF loss
        vmf_loss = self.vmf_loss(pred_vec, target_vec)
        
        # Combine losses
        combined_loss = self.geodesic_weight * geo_loss + self.vmf_weight * vmf_loss
        
        return combined_loss


class WeightedGeodesicLoss(nn.Module):
    """
    Geodesic loss with sample weights based on data reliability.
    
    Allows for weighting samples differently based on data quality or uncertainty.
    """
    
    def __init__(
        self,
        use_quaternions=True,
        reduction='mean',
        min_weight=0.1
    ):
        """
        Initialize the weighted geodesic loss.
        
        Args:
            use_quaternions: Whether inputs are quaternions
            reduction: Reduction method
            min_weight: Minimum weight for numerical stability
        """
        super(WeightedGeodesicLoss, self).__init__()
        self.geodesic_loss = GeodesicLoss(
            use_quaternions=use_quaternions,
            reduction='none'
        )
        self.reduction = reduction
        self.min_weight = min_weight
    
    def forward(self, pred, target, weights=None):
        """
        Calculate the weighted geodesic loss.
        
        Args:
            pred: Predicted rotations [batch_size, 4] or [batch_size, 3]
            target: Target rotations [batch_size, 4] or [batch_size, 3]
            weights: Optional sample weights [batch_size]
            
        Returns:
            torch.Tensor: Loss value
        """
        # Calculate per-sample geodesic loss
        distances = self.geodesic_loss(pred, target)
        
        # Apply weights if provided
        if weights is not None:
            # Clamp weights for numerical stability
            clamped_weights = torch.clamp(weights, min=self.min_weight)
            
            # Normalize weights to sum to batch size
            normalized_weights = clamped_weights * len(clamped_weights) / clamped_weights.sum()
            
            # Apply weights
            weighted_distances = distances * normalized_weights
        else:
            weighted_distances = distances
        
        # Apply reduction
        if self.reduction == 'mean':
            return weighted_distances.mean()
        elif self.reduction == 'sum':
            return weighted_distances.sum()
        else:  # 'none'
            return weighted_distances


def convert_degrees_to_radians(degrees):
    """Convert angles from degrees to radians."""
    return degrees * np.pi / 180.0


def convert_radians_to_degrees(radians):
    """Convert angles from radians to degrees."""
    return radians * 180.0 / np.pi


if __name__ == "__main__":
    # Example: Calculate angular distance
    v1 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    v2 = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    
    # Both vector pairs should be 90 degrees (π/2 radians) apart
    angles_rad = angular_distance(v1, v2)
    angles_deg = convert_radians_to_degrees(angles_rad)
    
    print(f"Angular distances: {angles_deg.tolist()} degrees") 