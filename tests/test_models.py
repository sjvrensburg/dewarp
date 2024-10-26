"""Unit tests for model components."""

import pytest
import torch

from dewarp.models.base import BaseModule
from dewarp.models.scanner import DocumentScanner
from dewarp.models.segmentation import U2NETP

def test_base_module():
    """Test BaseModule functionality."""
    model = BaseModule()
    
    # Test initialization
    assert not model.is_frozen
    
    # Test freezing/unfreezing
    model.freeze_batch_norm()
    assert model.is_frozen
    model.unfreeze_batch_norm()
    assert not model.is_frozen

def test_document_scanner(test_device):
    """Test DocumentScanner model."""
    model = DocumentScanner().to(test_device)
    
    # Test forward pass with dummy input
    batch_size = 2
    channels = 3
    height = 288
    width = 288
    
    x = torch.randn(batch_size, channels, height, width).to(test_device)
    output = model(x, test_mode=True)
    
    # Check output shape
    assert output.shape == (batch_size, 2, height, width)

def test_u2netp(test_device):
    """Test U2NETP segmentation model."""
    model = U2NETP().to(test_device)
    
    # Test forward pass
    x = torch.randn(2, 3, 288, 288).to(test_device)
    outputs = model(x)
    
    # Check all outputs
    assert len(outputs) == 7  # d0 through d6
    assert all(out.shape == (2, 1, 288, 288) for out in outputs)
