"""
dewarp - A document scanning and dewarping tool that corrects perspective distortion
"""

from dewarp.__version__ import __version__
from dewarp.core.pipeline import DocumentScannerPipeline

__all__ = ["DocumentScannerPipeline", "__version__"]