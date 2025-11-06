"""
Feature extraction module for certificate authenticity detection.
Extracts and analyzes logos, signatures, stamps, and text from certificates.
"""

import cv2
import numpy as np
import pytesseract
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract features from certificate images for authenticity verification."""
    
    def __init__(self):
        """Initialize feature extractor."""
        # Initialize ORB detector for keypoint detection
        self.orb = cv2.ORB_create(nfeatures=1000)
        
        # Initialize SIFT for more robust feature detection
        try:
            self.sift = cv2.SIFT_create()
        except:
            logger.warning("SIFT not available, using ORB only")
            self.sift = None
    
    def extract_all_features(self, image: np.ndarray) -> Dict:
        """
        Extract all features from certificate image.
        
        Args:
            image: Certificate image
            
        Returns:
            Dictionary containing all extracted features
        """
        features = {
            'logo_features': self.detect_logo_region(image),
            'signature_features': self.detect_signature_region(image),
            'stamp_features': self.detect_stamp_region(image),
            'text_features': self.extract_text_features(image),
            'edge_features': self.extract_edge_features(image),
            'color_features': self.extract_color_features(image),
            'texture_features': self.extract_texture_features(image)
        }
        
        return features
    
    def detect_logo_region(self, image: np.ndarray) -> Dict:
        """
        Detect and extract logo region features.
        
        Args:
            image: Input certificate image
            
        Returns:
            Dictionary with logo features
        """
        # Focus on top region where logos typically appear
        h, w = image.shape[:2]
        top_region = image[0:h//3, :]
        
        # Convert to grayscale
        if len(top_region.shape) == 3:
            gray = cv2.cvtColor(top_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = top_region
        
        # Detect keypoints and descriptors
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        
        # Find contours (potential logo regions)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        logo_contours = [c for c in contours if 500 < cv2.contourArea(c) < 50000]
        
        return {
            'num_keypoints': len(keypoints),
            'num_contours': len(logo_contours),
            'descriptors': descriptors,
            'region': top_region
        }
    
    def detect_signature_region(self, image: np.ndarray) -> Dict:
        """
        Detect and extract signature region features.
        
        Args:
            image: Input certificate image
            
        Returns:
            Dictionary with signature features
        """
        # Focus on bottom-left region where signatures typically appear
        h, w = image.shape[:2]
        sig_region = image[2*h//3:h, 0:w//2]
        
        # Convert to grayscale
        if len(sig_region.shape) == 3:
            gray = cv2.cvtColor(sig_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = sig_region
        
        # Edge detection for signature strokes
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Calculate signature metrics
        total_area = sum([cv2.contourArea(c) for c in contours])
        stroke_density = np.sum(edges > 0) / edges.size
        
        return {
            'num_strokes': len(contours),
            'total_area': total_area,
            'stroke_density': stroke_density,
            'edge_map': edges,
            'region': sig_region
        }
    
    def detect_stamp_region(self, image: np.ndarray) -> Dict:
        """
        Detect and extract stamp region features.
        
        Args:
            image: Input certificate image
            
        Returns:
            Dictionary with stamp features
        """
        # Focus on bottom-right region where stamps typically appear
        h, w = image.shape[:2]
        stamp_region = image[2*h//3:h, w//2:w]
        
        # Convert to grayscale
        if len(stamp_region.shape) == 3:
            gray = cv2.cvtColor(stamp_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = stamp_region
        
        # Detect circular shapes (stamps are often circular)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=50,
            param2=30,
            minRadius=20,
            maxRadius=100
        )
        
        # Color analysis (stamps often have distinct colors)
        if len(stamp_region.shape) == 3:
            mean_color = np.mean(stamp_region, axis=(0, 1))
            color_variance = np.var(stamp_region, axis=(0, 1))
        else:
            mean_color = np.mean(stamp_region)
            color_variance = np.var(stamp_region)
        
        num_circles = 0 if circles is None else len(circles[0])
        
        return {
            'num_circles': num_circles,
            'circles': circles,
            'mean_color': mean_color,
            'color_variance': color_variance,
            'region': stamp_region
        }
    
    def extract_text_features(self, image: np.ndarray) -> Dict:
        """
        Extract text from certificate using OCR.
        
        Args:
            image: Input certificate image
            
        Returns:
            Dictionary with text features
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply thresholding for better OCR
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        try:
            # Extract text using pytesseract
            text = pytesseract.image_to_string(thresh)
            
            # Get detailed data including confidence scores
            data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT)
            
            # Calculate average confidence
            confidences = [int(conf) for conf in data['conf'] if conf != '-1']
            avg_confidence = np.mean(confidences) if confidences else 0
            
            # Count words and lines
            words = [w for w in data['text'] if w.strip()]
            
            return {
                'text': text,
                'num_words': len(words),
                'avg_confidence': avg_confidence,
                'words': words
            }
        except Exception as e:
            logger.warning(f"OCR failed: {str(e)}")
            return {
                'text': '',
                'num_words': 0,
                'avg_confidence': 0,
                'words': []
            }
    
    def extract_edge_features(self, image: np.ndarray) -> Dict:
        """
        Extract edge-based features for tampering detection.
        
        Args:
            image: Input certificate image
            
        Returns:
            Dictionary with edge features
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Multiple edge detection methods
        edges_canny = cv2.Canny(gray, 50, 150)
        edges_sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        edges_sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edges_sobel = np.sqrt(edges_sobel_x**2 + edges_sobel_y**2)
        
        return {
            'edge_density': np.sum(edges_canny > 0) / edges_canny.size,
            'edge_mean_intensity': np.mean(edges_sobel),
            'edge_std': np.std(edges_sobel),
            'edge_map': edges_canny
        }
    
    def extract_color_features(self, image: np.ndarray) -> Dict:
        """
        Extract color-based features.
        
        Args:
            image: Input certificate image
            
        Returns:
            Dictionary with color features
        """
        if len(image.shape) != 3:
            return {'mean': 0, 'std': 0, 'histogram': None}
        
        # Calculate color statistics
        mean_color = np.mean(image, axis=(0, 1))
        std_color = np.std(image, axis=(0, 1))
        
        # Calculate color histogram
        hist_b = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
        hist_r = cv2.calcHist([image], [2], None, [256], [0, 256])
        
        return {
            'mean_color': mean_color,
            'std_color': std_color,
            'histogram': {
                'blue': hist_b.flatten(),
                'green': hist_g.flatten(),
                'red': hist_r.flatten()
            }
        }
    
    def extract_texture_features(self, image: np.ndarray) -> Dict:
        """
        Extract texture features using Local Binary Patterns (LBP) and other methods.
        
        Args:
            image: Input certificate image
            
        Returns:
            Dictionary with texture features
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Calculate texture metrics using statistical methods
        # Variance and entropy as texture measures
        variance = np.var(gray)
        
        # Calculate local variance (texture roughness)
        kernel_size = 9
        mean_kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, mean_kernel)
        local_variance = cv2.filter2D((gray.astype(np.float32) - local_mean)**2, -1, mean_kernel)
        
        return {
            'global_variance': variance,
            'mean_local_variance': np.mean(local_variance),
            'texture_energy': np.sum(gray**2)
        }
    
    def compare_features(self, features1: Dict, features2: Dict) -> float:
        """
        Compare two feature sets for similarity (useful for reference matching).
        
        Args:
            features1: First feature set
            features2: Second feature set
            
        Returns:
            Similarity score (0-1)
        """
        # Implement feature comparison logic
        # This can be used to compare against known genuine certificates
        similarity_scores = []
        
        # Compare keypoint descriptors if available
        if (features1['logo_features']['descriptors'] is not None and 
            features2['logo_features']['descriptors'] is not None):
            
            # Use brute-force matcher
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(
                features1['logo_features']['descriptors'],
                features2['logo_features']['descriptors']
            )
            
            # Normalize match score
            match_score = len(matches) / max(
                len(features1['logo_features']['descriptors']),
                len(features2['logo_features']['descriptors'])
            )
            similarity_scores.append(match_score)
        
        # Return average similarity
        return np.mean(similarity_scores) if similarity_scores else 0.0


if __name__ == "__main__":
    # Example usage
    extractor = FeatureExtractor()
    
    # Test feature extraction
    # img = cv2.imread("path/to/certificate.jpg")
    # features = extractor.extract_all_features(img)
    # print(f"Extracted features: {features.keys()}")
