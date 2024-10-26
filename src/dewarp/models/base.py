import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List, Type, Dict, Any
from pathlib import Path
import logging
from abc import ABC, abstractmethod

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseModule(nn.Module):
    """
    Base class for all models in the document scanning pipeline.
    Provides common utilities and enforces consistent interface.
    """
    def __init__(self):
        super().__init__()
        self._frozen = False

    def freeze_batch_norm(self) -> None:
        """Freeze all batch normalization layers in the model."""
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False
        self._frozen = True

    def unfreeze_batch_norm(self) -> None:
        """Unfreeze all batch normalization layers in the model."""
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train()
                for param in module.parameters():
                    param.requires_grad = True
        self._frozen = False

    @property
    def is_frozen(self) -> bool:
        """Return whether batch normalization layers are frozen."""
        return self._frozen

    @staticmethod
    def initialize_weights(module: nn.Module) -> None:
        """
        Initialize model weights using appropriate initialization schemes.

        Args:
            module: PyTorch module to initialize
        """
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
            if module.weight is not None:
                nn.init.constant_(module.weight, 1)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def load_state_dict_from_path(self,
                                path: Union[str, Path],
                                strict: bool = True,
                                device: Optional[torch.device] = None) -> None:
        """
        Load state dict from a file path with proper error handling.

        Args:
            path: Path to the state dict file
            strict: Whether to strictly enforce that the keys in state_dict match
            device: Device to load the state dict to
        """
        try:
            if device is None:
                device = next(self.parameters()).device

            state_dict = torch.load(path, map_location=device)

            # Handle state dict format variations
            if isinstance(state_dict, dict):
                if 'state_dict' in state_dict:  # Handle checkpoint format
                    state_dict = state_dict['state_dict']
                elif 'model_state_dict' in state_dict:
                    state_dict = state_dict['model_state_dict']

            self.load_state_dict(state_dict, strict=strict)
            logger.info(f"Successfully loaded weights from {path}")

        except Exception as e:
            logger.error(f"Failed to load weights from {path}: {str(e)}")
            raise

    def save_state_dict(self,
                       path: Union[str, Path],
                       additional_info: Optional[Dict[str, Any]] = None) -> None:
        """
        Save model state dict with optional additional information.

        Args:
            path: Path to save the state dict
            additional_info: Additional information to save with the state dict
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            'model_state_dict': self.state_dict(),
            'model_name': self.__class__.__name__,
        }

        if additional_info:
            save_dict.update(additional_info)

        try:
            torch.save(save_dict, path)
            logger.info(f"Successfully saved model to {path}")
        except Exception as e:
            logger.error(f"Failed to save model to {path}: {str(e)}")
            raise

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_device(self) -> torch.device:
        """Get the device where the model parameters are located."""
        return next(self.parameters()).device

    def to_device(self,
                 device: Union[str, torch.device],
                 non_blocking: bool = False) -> 'BaseModule':
        """
        Move model to specified device with proper error handling.

        Args:
            device: Device to move the model to
            non_blocking: Whether to perform non-blocking transfer

        Returns:
            Self for method chaining
        """
        try:
            return self.to(device=device, non_blocking=non_blocking)
        except Exception as e:
            logger.error(f"Failed to move model to device {device}: {str(e)}")
            raise

    def apply_to_children(self, fn: callable) -> None:
        """
        Apply a function to all child modules recursively.

        Args:
            fn: Function to apply to each module
        """
        for module in self.children():
            if isinstance(module, BaseModule):
                module.apply_to_children(fn)
            fn(module)

    def get_submodule_dict(self) -> Dict[str, nn.Module]:
        """
        Get a dictionary of all named submodules.

        Returns:
            Dictionary mapping names to submodules
        """
        return dict(self.named_modules())

    def freeze(self) -> None:
        """Freeze all model parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze all model parameters."""
        for param in self.parameters():
            param.requires_grad = True

    def freeze_layers(self, layer_names: List[str]) -> None:
        """
        Freeze specific layers by name.

        Args:
            layer_names: List of layer names to freeze
        """
        for name, module in self.named_modules():
            if name in layer_names:
                for param in module.parameters():
                    param.requires_grad = False

    def summary(self) -> str:
        """
        Generate a string summary of the model architecture and parameters.

        Returns:
            String containing model summary
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        summary_str = [
            f"\nModel Summary for {self.__class__.__name__}:",
            f"Total parameters: {total_params:,}",
            f"Trainable parameters: {trainable_params:,}",
            f"Non-trainable parameters: {total_params - trainable_params:,}",
            "\nLayer Information:"
        ]

        for name, module in self.named_children():
            params = sum(p.numel() for p in module.parameters())
            summary_str.append(f"{name}: {module.__class__.__name__} ({params:,} parameters)")

        return "\n".join(summary_str)

    def get_layer_output_sizes(self, input_size: torch.Size) -> Dict[str, torch.Size]:
        """
        Calculate output sizes for each layer given an input size.

        Args:
            input_size: Input tensor size

        Returns:
            Dictionary mapping layer names to their output sizes
        """
        sizes = {}
        hooks = []

        def hook(module, input, output, name):
            sizes[name] = output.shape

        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.MaxPool2d)):
                hooks.append(module.register_forward_hook(
                    lambda mod, inp, out, name=name: hook(mod, inp, out, name)
                ))

        try:
            self.forward(torch.randn(input_size).to(self.get_device()))
        finally:
            for hook in hooks:
                hook.remove()

        return sizes

    @staticmethod
    def get_activation_stats(tensor: torch.Tensor) -> Dict[str, float]:
        """
        Calculate activation statistics for a tensor.

        Args:
            tensor: Input tensor

        Returns:
            Dictionary containing mean, std, min, max values
        """
        with torch.no_grad():
            return {
                'mean': tensor.mean().item(),
                'std': tensor.std().item(),
                'min': tensor.min().item(),
                'max': tensor.max().item()
            }

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing for memory efficiency."""
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self) -> None:
        """Disable gradient checkpointing."""
        self.gradient_checkpointing = False
