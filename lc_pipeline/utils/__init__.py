"""
Utility functions and classes for the pipeline.
"""

# Add SciPy compatibility patching BEFORE any imports
import sys
import numpy as np

# Handle SciPy compatibility issues with NumPy 2.x
try:
    numpy_version = np.__version__
    numpy_major = int(numpy_version.split('.')[0])
    if numpy_major >= 2:
        # Apply direct monkey patch for scipy.special._multiufuncs if possible
        # This needs to happen before any scipy imports
        import importlib
        
        # Define a patch function to apply
        def patch_scipy_special():
            try:
                import scipy.special._multiufuncs as multiufuncs
                
                # Store original init
                original_init = multiufuncs.MultiUFunc.__init__
                
                # Define patched init that handles the ufunc type error
                def patched_init(self, ufunc_or_ufuncs, doc=None, force_complex_output=False, **default_kwargs):
                    try:
                        # Try original init first
                        return original_init(self, ufunc_or_ufuncs, doc, force_complex_output, **default_kwargs)
                    except ValueError as e:
                        # If there's a type error with ufuncs, apply a fallback
                        if "All ufuncs must have type" in str(e):
                            print(f"Utils: Bypassing SciPy ufunc type check")
                            # Set minimal attributes to allow initialization
                            self.ufuncs = [ufunc_or_ufuncs] if isinstance(ufunc_or_ufuncs, np.ufunc) else list(ufunc_or_ufuncs)
                            self.doc = doc
                            self._force_complex_output = force_complex_output
                            self._default_kwargs = default_kwargs
                        else:
                            raise
                
                # Apply the patch
                multiufuncs.MultiUFunc.__init__ = patched_init
                print("Utils: Applied SciPy multiufuncs compatibility patch")
                return True
            except Exception as e:
                print(f"Utils: SciPy patching failed: {e}")
                return False
        
        # Try to import and patch scipy.special directly
        try:
            # Attempt preemptive import of scipy.special to control the import chain
            import scipy.special
            # Then try to patch
            patch_result = patch_scipy_special()
            print(f"Utils: Preemptive SciPy patching result: {patch_result}")
        except ImportError:
            print("Utils: SciPy not found, will continue without patching")
except Exception as e:
    print(f"Utils: Error in SciPy compatibility setup: {e}")

# Ensure that NumPy is imported early in case TensorBoard is imported later
try:
    import numpy as np
except ImportError:
    print("Warning: NumPy could not be imported in utils package")

# Handle TensorBoard import robustly to avoid dependency issues
try:
    import tensorboard
    # If compat attribute exists, make sure notf is available
    if hasattr(tensorboard, 'compat') and not hasattr(tensorboard.compat, 'notf'):
        import types
        tensorboard.compat.notf = types.ModuleType('tensorboard.compat.notf')
except (ImportError, AttributeError):
    # Silently continue if TensorBoard can't be imported
    pass

# Now import the actual utility modules
from .logging import Logger, Timer
from .training import EarlyStopping, MetricTracker
from .phase_folding import phase_fold

__all__ = [
    "Logger",
    "Timer",
    "EarlyStopping",
    "MetricTracker",
    "phase_fold"
] 