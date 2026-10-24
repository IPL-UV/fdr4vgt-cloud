"""FDR4VGT-CLOUD: cloud detection for Proba-V and SPOT-VGT.

This package contains the model, training code, inference helpers and
release pipeline for the FDR4VGT-CLOUD dataset and model.
"""

__version__ = "1.0.0"

from .model import CombinedNet4, SigmoidWrap, get_model

__all__ = ["CombinedNet4", "SigmoidWrap", "get_model", "__version__"]
