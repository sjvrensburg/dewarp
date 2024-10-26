#!/usr/bin/env python3
"""Command-line interface for the dewarp package."""

import argparse
import logging
import sys
from pathlib import Path

from dewarp.core.pipeline import DocumentScannerPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_package_weights_dir() -> Path:
    """Get the path to the package's weights directory."""
    return Path(__file__).parent.parent / "resources" / "weights"

def main():
    """Main function for command line interface."""
    parser = argparse.ArgumentParser(description='Document Scanner and Correction')
    parser.add_argument(
        '--seg_model_path',
        type=Path,
        default=get_package_weights_dir() / "seg.pth",
        help='Path to segmentation model weights'
    )
    parser.add_argument(
        '--rec_model_path',
        type=Path,
        default=get_package_weights_dir() / "DocScanner-L.pth",
        help='Path to correction model weights'
    )
    parser.add_argument(
        '--input_dir',
        type=Path,
        required=True,
        help='Input directory with distorted images'
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        required=True,
        help='Output directory for rectified images'
    )
    parser.add_argument(
        '--device',
        choices=['cuda', 'cpu'],
        help='Device to run inference on (default: auto-detect)'
    )
    
    args = parser.parse_args()

    try:
        # Initialize pipeline
        pipeline = DocumentScannerPipeline(
            seg_model_path=args.seg_model_path,
            rec_model_path=args.rec_model_path,
            device=args.device
        )
        
        # Process images
        pipeline.process_directory(
            input_dir=args.input_dir,
            output_dir=args.output_dir
        )
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()