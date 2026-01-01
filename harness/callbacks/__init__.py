"""
Callbacks Package
=================

Training callbacks for monitoring and analysis.
"""

from .eigenvalue import (
    EigenvalueMonitorCallback,
    extract_eigenvalues_from_attention,
    compute_spectral_metrics,
)

__all__ = [
    "EigenvalueMonitorCallback",
    "extract_eigenvalues_from_attention", 
    "compute_spectral_metrics",
]


# Placeholder for future basin mapping callback
class BasinMappingCallback:
    """
    TODO: Basin mapping callback for attention pattern analysis.
    
    Will be implemented to run basin probes during or after training.
    """
    pass
