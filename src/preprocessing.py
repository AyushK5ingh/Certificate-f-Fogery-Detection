"""
Preprocessing module for certificate authenticity detection.
Handles loading and preprocessing of certificates from various formats (PDF, JPG, PNG, etc.)
"""

import os
import cv2
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
from pathlib import Path
from typing import Union, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CertificatePreprocessor:
    """Handles preprocessing of certificate images from various formats."""
    
    def __init__(self, target_size: Tuple[int, int] = (1024, 1024)):
        """
        Initialize preprocessor.
        
        Args:
            target_size: Target image size (width, height) for model input
        """
        self.target_size = target_size
        
    def load_certificate(self, file_path: Union[str, Path]) -> np.ndarray:
        """
        Load certificate from file (supports PDF, JPG, PNG, JPEG).
        
        Args:
            file_path: Path to certificate file
            
        Returns:
            Preprocessed image as numpy array
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_ext = file_path.suffix.lower()
        
        if file_ext == '.pdf':
            return self._load_from_pdf(file_path)
        elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            return self._load_from_image(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    def _load_from_pdf(self, pdf_path: Path) -> np.ndarray:
        """
        Load certificate from PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Image as numpy array
        """
        try:
            # Convert PDF to images (take first page)
            images = convert_from_path(str(pdf_path), dpi=300)
            
            if not images:
                raise ValueError(f"No pages found in PDF: {pdf_path}")
            
            # Convert PIL Image to numpy array
            img_array = np.array(images[0])
            
            # Convert RGB to BGR for OpenCV
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            logger.info(f"Loaded PDF: {pdf_path}")
            return img_array
            
        except Exception as e:
            logger.error(f"Error loading PDF {pdf_path}: {str(e)}")
            raise
    
    def _load_from_image(self, img_path: Path) -> np.ndarray:
        """
        Load certificate from image file.
        
        Args:
            img_path: Path to image file
            
        Returns:
            Image as numpy array
        """
        try:
            # Load image using OpenCV
            img = cv2.imread(str(img_path))
            
            if img is None:
                raise ValueError(f"Failed to load image: {img_path}")
            
            logger.info(f"Loaded image: {img_path}")
            return img
            
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {str(e)}")
            raise
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for model input.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image
        """
        # Resize to target size
        resized = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LANCZOS4)
        
        # Normalize pixel values to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        return normalized
    
    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply image enhancement techniques for better feature extraction.
        
        Args:
            image: Input image
            
        Returns:
            Enhanced image
        """
        # Convert to grayscale for enhancement
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
        
        return denoised
    
    def extract_roi_regions(self, image: np.ndarray) -> dict:
        """
        Extract regions of interest (logo, signature, stamp areas).
        
        Args:
            image: Input certificate image
            
        Returns:
            Dictionary containing ROI regions
        """
        h, w = image.shape[:2]
        
        # Define approximate regions (these can be refined with detection models)
        regions = {
            'top_left': image[0:h//3, 0:w//3],           # Typically logo area
            'top_right': image[0:h//3, 2*w//3:w],        # Alternative logo area
            'bottom_left': image[2*h//3:h, 0:w//3],      # Signature area
            'bottom_right': image[2*h//3:h, 2*w//3:w],   # Stamp area
            'center': image[h//3:2*h//3, w//3:2*w//3],   # Main certificate content
        }
        
        return regions
    
    def load_dataset(self, dataset_path: Path, subset: str = 'train') -> Tuple[List[np.ndarray], List[int]]:
        """
        Load entire dataset from directory structure.
        
        Args:
            dataset_path: Path to dataset root
            subset: One of 'train', 'valid', or 'test'
            
        Returns:
            Tuple of (images, labels) where labels are 0 (fake) or 1 (genuine)
        """
        subset_map = {
            'train': 'train_out',
            'valid': 'valid_out',
            'test': 'test_out'
        }
        
        subset_dir = dataset_path / subset_map.get(subset, subset)
        
        if not subset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {subset_dir}")
        
        images = []
        labels = []
        
        # Load fake certificates (label = 0)
        fake_dir = subset_dir / 'fake'
        if fake_dir.exists():
            for img_file in fake_dir.iterdir():
                if img_file.suffix.lower() in ['.pdf', '.jpg', '.jpeg', '.png']:
                    try:
                        img = self.load_certificate(img_file)
                        preprocessed = self.preprocess(img)
                        images.append(preprocessed)
                        labels.append(0)
                    except Exception as e:
                        logger.warning(f"Failed to load {img_file}: {str(e)}")
        
        # Load genuine certificates (label = 1)
        genuine_dir = subset_dir / 'genuine'
        if genuine_dir.exists():
            for img_file in genuine_dir.iterdir():
                if img_file.suffix.lower() in ['.pdf', '.jpg', '.jpeg', '.png']:
                    try:
                        img = self.load_certificate(img_file)
                        preprocessed = self.preprocess(img)
                        images.append(preprocessed)
                        labels.append(1)
                    except Exception as e:
                        logger.warning(f"Failed to load {img_file}: {str(e)}")
        
        logger.info(f"Loaded {len(images)} images from {subset} set")
        logger.info(f"  Fake: {labels.count(0)}, Genuine: {labels.count(1)}")
        
        return images, labels


if __name__ == "__main__":
    # Example usage
    preprocessor = CertificatePreprocessor()
    
    # Test loading a single certificate
    # img = preprocessor.load_certificate("path/to/certificate.pdf")
    # processed = preprocessor.preprocess(img)
    # print(f"Processed image shape: {processed.shape}")
