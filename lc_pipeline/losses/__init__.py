"""
Loss functions for training models.
"""

from .axis_losses import (
    GeodesicLoss,
    VMFLoss,
    GeodesicVMFCombinedLoss,
    WeightedGeodesicLoss,
    angular_distance,
    geodesic_distance
)
from .period_losses import (
    ClipPeriodLoss,
    LogScalePeriodLoss,
    LSMatchingLoss,
    MultiHarmonicPeriodLoss
)

__all__ = [
    "GeodesicLoss",
    "VMFLoss",
    "GeodesicVMFCombinedLoss",
    "WeightedGeodesicLoss",
    "angular_distance",
    "geodesic_distance",
    "ClipPeriodLoss",
    "LogScalePeriodLoss",
    "LSMatchingLoss",
    "MultiHarmonicPeriodLoss"
] 