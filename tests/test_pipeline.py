"""Integration tests for document scanning pipeline."""

import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from dewarp.core.pipeline import DocumentScannerPipeline


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


def test_pipeline_initialization(test_device, model_weights):
    """Test pipeline initialization."""
    seg_path, rec_path = model_weights
    
    pipeline = DocumentScannerPipeline(
        seg_model_path=seg_path,
        rec_model_path=rec_path,
        device=test_device
    )
    
    assert pipeline.device == test_device
    assert pipeline.model is not None


def test_process_single_image(sample_distorted_image, temp_dir, test_device, model_weights):
    """Test processing a single image."""
    seg_path, rec_path = model_weights
    
    pipeline = DocumentScannerPipeline(
        seg_model_path=seg_path,
        rec_model_path=rec_path,
        device=test_device
    )
    
    output_path = temp_dir / "output.png"
    
    # Process image
    pipeline.process_image(sample_distorted_image, output_path)
    
    # Check output exists
    assert output_path.exists()
    
    # Basic sanity check on output
    output_img = Image.open(output_path)
    assert output_img.mode == "RGB"
    assert output_img.size == Image.open(sample_distorted_image).size


def test_process_directory(temp_dir, sample_distorted_image, test_device, model_weights):
    """Test processing a directory of images."""
    seg_path, rec_path = model_weights
    
    # Create input directory with multiple test images
    input_dir = temp_dir / "input"
    input_dir.mkdir()
    output_dir = temp_dir / "output"
    
    # Copy sample image multiple times with different names
    for i in range(3):
        shutil.copy(sample_distorted_image, input_dir / f"test_{i}.png")
    
    # Process directory
    pipeline = DocumentScannerPipeline(
        seg_model_path=seg_path,
        rec_model_path=rec_path,
        device=test_device
    )
    pipeline.process_directory(input_dir, output_dir)
    
    # Check outputs
    assert output_dir.exists()
    assert len(list(output_dir.glob("*.png"))) == 3


def test_invalid_weights_path(temp_dir, test_device):
    """Test handling of invalid weight paths."""
    with pytest.raises(FileNotFoundError):
        DocumentScannerPipeline(
            seg_model_path=temp_dir / "nonexistent_seg.pth",
            rec_model_path=temp_dir / "nonexistent_rec.pth",
            device=test_device
        )


def test_invalid_image_path(temp_dir, test_device, model_weights):
    """Test handling of invalid image path."""
    seg_path, rec_path = model_weights
    
    pipeline = DocumentScannerPipeline(
        seg_model_path=seg_path,
        rec_model_path=rec_path,
        device=test_device
    )
    
    with pytest.raises(FileNotFoundError):
        pipeline.process_image(
            temp_dir / "nonexistent.png",
            temp_dir / "output.png"
        )


def test_empty_directory(temp_dir, test_device, model_weights):
    """Test processing an empty directory."""
    seg_path, rec_path = model_weights
    
    # Create empty input directory
    input_dir = temp_dir / "empty_input"
    input_dir.mkdir()
    output_dir = temp_dir / "empty_output"
    
    pipeline = DocumentScannerPipeline(
        seg_model_path=seg_path,
        rec_model_path=rec_path,
        device=test_device
    )
    
    # Should handle empty directory without errors
    pipeline.process_directory(input_dir, output_dir)
    
    # Check that output directory was created but is empty
    assert output_dir.exists()
    assert len(list(output_dir.glob("*.png"))) == 0
