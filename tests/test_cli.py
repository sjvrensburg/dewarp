"""Tests for command-line interface."""

import shutil
import subprocess
from pathlib import Path

import pytest

def test_cli_help():
    """Test CLI help message."""
    result = subprocess.run(['dewarp', '--help'], capture_output=True, text=True)
    assert result.returncode == 0
    assert 'Document Scanner and Correction' in result.stdout

def test_cli_process_directory(temp_dir, sample_distorted_image):
    """Test CLI directory processing."""
    # Create test directory structure
    input_dir = temp_dir / "input"
    input_dir.mkdir()
    output_dir = temp_dir / "output"
    
    # Copy sample image
    shutil.copy(sample_distorted_image, input_dir / "test.png")
    
    # Run CLI
    result = subprocess.run([
        'dewarp',
        '--input_dir', str(input_dir),
        '--output_dir', str(output_dir)
    ])
    
    assert result.returncode == 0
    assert (output_dir / "test_rec.png").exists()