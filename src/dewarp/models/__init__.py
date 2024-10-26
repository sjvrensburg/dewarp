from typing import Dict, Type, Any
import torch

# Base components
from .base import BaseModule

# Core model components
from .scanner import (
    DocumentScanner,
    BasicEncoder,
    BottleneckBlock,
    BasicUpdateBlock,
    FlowHead,
    ConvGRU,
    SepConvGRU,
    BasicMotionEncoder,
    ResidualBlock
)

from .segmentation import (
    U2NET,
    U2NETP,
    AdvancedMotionEncoder,
    MotionFusion,
    REBNCONV,
    RSU4,
    RSU4F,
    RSU5,
    RSU6,
    RSU7
)

# Version information
__version__ = '1.0.0'

# Available model configurations
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    'document_scanner_base': {
        'hidden_dim': 160,
        'context_dim': 160,
        'encoder_output_dim': 320,
        'dropout': 0.0
    },
    'document_scanner_large': {
        'hidden_dim': 192,
        'context_dim': 192,
        'encoder_output_dim': 384,
        'dropout': 0.1
    },
    'u2net_full': {
        'in_ch': 3,
        'out_ch': 1
    },
    'u2net_lite': {
        'in_ch': 3,
        'out_ch': 1,
        'reduced_dims': True
    }
}

def get_model(model_name: str, **kwargs: Any) -> BaseModule:
    """
    Factory function to create model instances.

    Args:
        model_name: Name of the model to create
        **kwargs: Additional arguments to pass to the model constructor

    Returns:
        Instantiated model

    Raises:
        ValueError: If model_name is not recognized
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")

    # Get base configuration
    config = MODEL_CONFIGS.get(model_name, {}).copy()

    # Update with any provided kwargs
    config.update(kwargs)

    # Create and return model instance
    return MODEL_REGISTRY[model_name](**config)

# Registry of available models
MODEL_REGISTRY: Dict[str, Type[BaseModule]] = {
    'document_scanner_base': DocumentScanner,
    'document_scanner_large': DocumentScanner,
    'u2net_full': U2NET,
    'u2net_lite': U2NETP
}

def load_pretrained(model: BaseModule,
                   weights_path: str,
                   strict: bool = True,
                   device: torch.device = None) -> BaseModule:
    """
    Load pretrained weights into a model.

    Args:
        model: Model instance to load weights into
        weights_path: Path to weights file
        strict: Whether to strictly enforce that the keys in state_dict match
        device: Device to load the weights to

    Returns:
        Model with loaded weights

    Raises:
        RuntimeError: If weight loading fails
    """
    model.load_state_dict_from_path(
        weights_path,
        strict=strict,
        device=device
    )
    return model

def list_available_models() -> Dict[str, Dict[str, Any]]:
    """
    Get information about available models.

    Returns:
        Dictionary containing model configurations
    """
    return MODEL_CONFIGS.copy()

class ModelBuilder:
    """Builder class for creating and configuring models."""

    def __init__(self):
        self._model = None
        self._config = {}

    def select_model(self, model_name: str) -> 'ModelBuilder':
        """Select the model to build."""
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_name}")

        self._config = MODEL_CONFIGS.get(model_name, {}).copy()
        self._model_class = MODEL_REGISTRY[model_name]
        return self

    def set_config(self, **kwargs: Any) -> 'ModelBuilder':
        """Set configuration parameters."""
        self._config.update(kwargs)
        return self

    def build(self) -> BaseModule:
        """Build and return the configured model."""
        if self._model_class is None:
            raise ValueError("No model selected")

        return self._model_class(**self._config)

def create_document_scanner(
    pretrained: bool = False,
    weights_path: str = None,
    device: torch.device = None,
    **kwargs: Any
) -> DocumentScanner:
    """
    Convenience function to create a document scanner model.

    Args:
        pretrained: Whether to load pretrained weights
        weights_path: Path to weights file if pretrained=True
        device: Device to create the model on
        **kwargs: Additional arguments to pass to the model constructor

    Returns:
        Configured DocumentScanner model
    """
    model = DocumentScanner(**kwargs)

    if pretrained:
        if weights_path is None:
            raise ValueError("weights_path must be provided when pretrained=True")
        model = load_pretrained(model, weights_path, device=device)

    if device is not None:
        model = model.to(device)

    return model

def create_u2net(
    lite: bool = False,
    pretrained: bool = False,
    weights_path: str = None,
    device: torch.device = None,
    **kwargs: Any
) -> BaseModule:
    """
    Convenience function to create a U2NET model.

    Args:
        lite: Whether to use the lite version
        pretrained: Whether to load pretrained weights
        weights_path: Path to weights file if pretrained=True
        device: Device to create the model on
        **kwargs: Additional arguments to pass to the model constructor

    Returns:
        Configured U2NET model
    """
    model_class = U2NETP if lite else U2NET
    model = model_class(**kwargs)

    if pretrained:
        if weights_path is None:
            raise ValueError("weights_path must be provided when pretrained=True")
        model = load_pretrained(model, weights_path, device=device)

    if device is not None:
        model = model.to(device)

    return model

__all__ = [
    # Main models
    'DocumentScanner',
    'U2NET',
    'U2NETP',

    # Components
    'BaseModule',
    'BasicEncoder',
    'ResidualBlock',
    'BottleneckBlock',
    'BasicUpdateBlock',
    'FlowHead',
    'ConvGRU',
    'SepConvGRU',
    'BasicMotionEncoder',
    'AdvancedMotionEncoder',
    'MotionFusion',

    # U2NET components
    'REBNCONV',
    'RSU4',
    'RSU4F',
    'RSU5',
    'RSU6',
    'RSU7',

    # Factory functions
    'get_model',
    'load_pretrained',
    'create_document_scanner',
    'create_u2net',

    # Utilities
    'list_available_models',
    'ModelBuilder'
]
