"""Common test fixtures for the dewarp package."""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

@pytest.fixture
def test_device():
    """Provide a torch device for testing."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory for test files."""
    return tmp_path

@pytest.fixture
def model_weights(temp_dir):
    """Create dummy model weights for testing."""
    seg_path = temp_dir / "seg.pth"
    rec_path = temp_dir / "DocScanner-L.pth"
    
    # Create dummy state dictionaries
    dummy_seg_state = {
        "module.stage1.conv_s1.weight": torch.randn(64, 3, 3, 3),
        "module.stage1.bn_s1.weight": torch.randn(64),
    }
    
    dummy_rec_state = {
        "feature_encoder.conv1.weight": torch.randn(80, 3, 7, 7),
        "feature_encoder.norm1.weight": torch.randn(80),
    }
    
    # Save dummy weights
    torch.save(dummy_seg_state, seg_path)
    torch.save(dummy_rec_state, rec_path)
    
    return seg_path, rec_path

@pytest.fixture
def sample_image(temp_dir):
    """Create a simple test image."""
    img = Image.new('RGB', (100, 100), color='white')
    img_array = np.array(img)
    img_array[20:80, 20:80] = [200, 200, 200]  # Gray rectangle
    img_array[40:60, 40:60] = [0, 0, 0]        # Black square
    
    img_path = temp_dir / "test_document.png"
    Image.fromarray(img_array).save(img_path)
    
    return img_path

@pytest.fixture
def sample_distorted_image(temp_dir):
    """Create a sample distorted test image using PIL and numpy."""
    # Create a white background
    img = Image.new('RGB', (100, 100), color='white')
    img_array = np.array(img)

    # Create a simple pattern that can represent a document
    # Create a gradient to simulate perspective distortion
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)
    
    # Create a trapezoidal shape
    mask = ((Y > 0.2 + 0.1*X) & 
           (Y < 0.8 - 0.1*X) & 
           (X > 0.2) & 
           (X < 0.8))
    
    # Apply the mask with a gray color
    img_array[mask] = [200, 200, 200]
    
    # Add some darker lines to simulate text
    for i in range(30, 70, 10):
        y_offset = int(5 * (i-30)/40)  # Create slight perspective effect
        img_array[i:i+2, 35+y_offset:65+y_offset] = [100, 100, 100]
    
    # Save the image
    img_path = temp_dir / "test_distorted.png"
    Image.fromarray(img_array).save(img_path)
    
    return img_path