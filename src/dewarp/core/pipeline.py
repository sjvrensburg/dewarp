"""Core pipeline implementation for document scanning and correction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
from PIL import Image
from pathlib import Path
from typing import Optional, Union, List, Tuple, Dict
import warnings
import logging
from tqdm import tqdm

from dewarp.models.scanner import DocumentScanner
from dewarp.models.segmentation import U2NETP
from dewarp.models.base import BaseModule

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')


def add_margin(pil_img: Image.Image,
              top: int,
              right: int,
              bottom: int,
              left: int,
              color: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    """
    Add margin to image.
    
    Args:
        pil_img: Input PIL image
        top: Top margin size
        right: Right margin size
        bottom: Bottom margin size
        left: Left margin size
        color: Margin color (default: white)
        
    Returns:
        PIL image with margins added
    """
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result


class Net(BaseModule):
    """Combined network for document segmentation and correction."""
    def __init__(self):
        super().__init__()
        self.msk = U2NETP(3, 1)  # Segmentation model
        self.bm = DocumentScanner()  # Correction model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through both segmentation and correction networks.
        
        Args:
            x: Input image tensor (B, C, H, W)
            
        Returns:
            Corrected boundary map tensor
        """
        # Run segmentation model
        msk, *_ = self.msk(x)
        msk = (msk > 0.5).float()
        x = msk * x

        # Run correction model
        bm = self.bm(x, iters=12, test_mode=True)
        
        # Normalize boundary map to [-0.99, 0.99] range
        bm = (2 * (bm / 286.8) - 1) * 0.99

        return bm


class DocumentScannerPipeline:
    """End-to-end pipeline for document scanning and correction."""
    
    def __init__(self,
                 seg_model_path: Union[str, Path],
                 rec_model_path: Union[str, Path],
                 device: Optional[torch.device] = None):
        """
        Initialize the document scanner pipeline.
        
        Args:
            seg_model_path: Path to segmentation model weights
            rec_model_path: Path to correction model weights
            device: Device to run inference on (default: auto-detect)
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.model = self._initialize_model(seg_model_path, rec_model_path)
        self.model.eval()

    def _initialize_model(self,
                         seg_path: Union[str, Path],
                         rec_path: Union[str, Path]) -> Net:
        """Initialize and load model weights."""
        model = Net().to(self.device)
        
        # Load segmentation model weights
        if seg_path:
            self._load_seg_weights(model.msk, seg_path)
            logger.info(f"Loaded segmentation weights from {seg_path}")
            
        # Load correction model weights
        if rec_path:
            self._load_rec_weights(model.bm, rec_path)
            logger.info(f"Loaded correction weights from {rec_path}")
            
        return model

    # @staticmethod
    # def _load_seg_weights(model: nn.Module, path: Union[str, Path]) -> None:
    #     """Load segmentation model weights with module prefix handling."""
    #     try:
    #         model_dict = model.state_dict()
    #         weights = torch.load(path, map_location='cuda:0')
    #         # Handle module prefix in state dict
    #         weights = {k[6:]: v for k, v in weights.items() if k[6:] in model_dict}
    #         model_dict.update(weights)
    #         model.load_state_dict(model_dict)
    #     except Exception as e:
    #         logger.error(f"Failed to load segmentation weights: {str(e)}")
    #         raise

    # @staticmethod
    # def _load_rec_weights(model: nn.Module, path: Union[str, Path]) -> None:
    #     """Load correction model weights."""
    #     try:
    #         model_dict = model.state_dict()
    #         weights = torch.load(path, map_location='cuda:0')
    #         # Filter weights that match the model architecture
    #         weights = {k: v for k, v in weights.items() if k in model_dict}
    #         model_dict.update(weights)
    #         model.load_state_dict(model_dict)
    #     except Exception as e:
    #         logger.error(f"Failed to load correction weights: {str(e)}")
    #         raise
    
    @staticmethod
    def _load_seg_weights(model: nn.Module, path: Union[str, Path]) -> None:
        """Load segmentation model weights with module prefix handling."""
        try:
            model_dict = model.state_dict()
            weights = torch.load(path, map_location='cuda:0', weights_only=True)
            # Handle module prefix in state dict
            weights = {k[6:]: v for k, v in weights.items() if k[6:] in model_dict}
            model_dict.update(weights)
            model.load_state_dict(model_dict)
        except Exception as e:
            logger.error(f"Failed to load segmentation weights: {str(e)}")
            raise

    @staticmethod
    def _load_rec_weights(model: nn.Module, path: Union[str, Path]) -> None:
        """Load correction model weights."""
        try:
            model_dict = model.state_dict()
            weights = torch.load(path, map_location='cuda:0', weights_only=True)
            # Filter weights that match the model architecture
            weights = {k: v for k, v in weights.items() if k in model_dict}
            model_dict.update(weights)
            model.load_state_dict(model_dict)
        except Exception as e:
            logger.error(f"Failed to load correction weights: {str(e)}")
            raise

    def preprocess_image(self,
                        image_path: Union[str, Path, np.ndarray, Image.Image],
                        target_size: Tuple[int, int] = (288, 288)) -> torch.Tensor:
        """
        Preprocess image for model input.
        
        Args:
            image_path: Input image or path to image
            target_size: Size to resize image to
            
        Returns:
            Preprocessed tensor
        """
        # Load image if path is provided
        if isinstance(image_path, (str, Path)):
            image = Image.open(image_path)
        elif isinstance(image_path, np.ndarray):
            image = Image.fromarray(image_path)
        elif isinstance(image_path, Image.Image):
            image = image_path
        else:
            raise ValueError("Unsupported image type")

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Add margins
        margin_size = int(min(image.size) * 0.1)
        image = add_margin(image, margin_size, margin_size, margin_size, margin_size)

        # Resize and convert to numpy
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        image = np.array(image).astype(np.float32) / 255.0

        # Convert to tensor
        image = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)
        return image

    def process_image(self,
                     image_path: Union[str, Path],
                     output_path: Union[str, Path],
                     return_intermediate: bool = False
                     ) -> Optional[Dict[str, np.ndarray]]:
        """
        Process a single image and save the result.
        
        Args:
            image_path: Path to input image
            output_path: Path to save corrected image
            return_intermediate: Whether to return intermediate results
        """
        try:
            # Load and preprocess image
            original_image = Image.open(image_path)
            
            # Store original size
            original_size = original_image.size
            
            # Preprocess
            input_tensor = self.preprocess_image(original_image)
            
            intermediates = {}
            if return_intermediate:
                intermediates['original'] = np.array(original_image)
                intermediates['preprocessed'] = input_tensor.cpu().numpy()

            with torch.no_grad():
                # Run model
                output = self.model(input_tensor.to(self.device))
                if return_intermediate:
                    intermediates['raw_output'] = output.cpu().numpy()

                # Process boundary map
                bm_x = cv2.resize(output[0, 0].cpu().numpy(), original_size)
                bm_y = cv2.resize(output[0, 1].cpu().numpy(), original_size)

                # Apply smoothing
                bm_x = cv2.blur(bm_x, (3, 3))
                bm_y = cv2.blur(bm_y, (3, 3))

                # Create sampling grid
                grid = torch.from_numpy(np.stack([bm_x, bm_y], axis=2)).unsqueeze(0)
                
                # Convert original image to tensor
                orig_tensor = torch.from_numpy(
                    np.array(original_image).transpose(2, 0, 1)
                ).float().unsqueeze(0) / 255.0

                # Sample and get corrected image
                corrected = F.grid_sample(
                    orig_tensor.to(self.device),
                    grid.to(self.device),
                    align_corners=True
                )

                # Convert to numpy
                corrected_image = (corrected[0] * 255).permute(1, 2, 0).cpu().numpy()
                
                if return_intermediate:
                    intermediates['boundary_map'] = np.stack([bm_x, bm_y], axis=2)
                    intermediates['corrected'] = corrected_image

                # Ensure output directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Save result
                cv2.imwrite(
                    str(output_path),
                    cv2.cvtColor(corrected_image.astype(np.uint8), cv2.COLOR_RGB2BGR)
                )

            if return_intermediate:
                return intermediates

        except Exception as e:
            logger.error(f"Failed to process {image_path}: {str(e)}")
            raise

    def process_directory(self,
                         input_dir: Union[str, Path],
                         output_dir: Union[str, Path],
                         extensions: List[str] = ['.jpg', '.jpeg', '.png', '.bmp']) -> None:
        """
        Process all images in a directory.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save corrected images
            extensions: List of valid file extensions
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get all valid image files
        image_files = []
        for ext in extensions:
            image_files.extend(input_dir.glob(f'*{ext}'))
            image_files.extend(input_dir.glob(f'*{ext.upper()}'))

        if not image_files:
            logger.warning(f"No valid images found in {input_dir}")
            return

        # Process each image with progress bar
        for img_path in tqdm(image_files, desc="Processing images"):
            try:
                output_path = output_dir / f"{img_path.stem}_rec{img_path.suffix}"
                self.process_image(img_path, output_path)
            except Exception as e:
                logger.error(f"Failed to process {img_path}: {str(e)}")
                continue


