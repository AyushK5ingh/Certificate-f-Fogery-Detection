"""
Inference script for certificate authenticity detection.
Predicts whether a certificate is genuine or fake.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import argparse
import json
from typing import Dict, Tuple
import logging

from model import create_model
from preprocessing import CertificatePreprocessor
from feature_extraction import FeatureExtractor
import albumentations as A
from albumentations.pytorch import ToTensorV2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CertificateAuthenticityPredictor:
    """Predictor for certificate authenticity."""
    
    def __init__(
        self,
        model_path: str,
        device: str = 'auto',
        img_size: int = 1024
    ):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run on ('cuda', 'cpu', or 'auto')
            img_size: Input image size
        """
        self.img_size = img_size
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint.get('config', {})
        
        # Create model
        model_name = config.get('model_name', 'efficientnet_b3')
        dropout = config.get('dropout', 0.3)
        
        logger.info(f"Loading model: {model_name}")
        self.model = create_model(
            model_name=model_name,
            num_classes=2,
            pretrained=False,
            dropout=dropout
        )
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
        logger.info(f"Model validation F1: {checkpoint.get('val_f1', 'unknown'):.4f}")
        
        # Initialize preprocessor and feature extractor
        self.preprocessor = CertificatePreprocessor(target_size=(img_size, img_size))
        self.feature_extractor = FeatureExtractor()
        
        # Define transforms
        self.transform = A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def predict(
        self,
        certificate_path: str,
        return_probabilities: bool = True,
        return_features: bool = False
    ) -> Dict:
        """
        Predict authenticity of a certificate.
        
        Args:
            certificate_path: Path to certificate file (PDF, JPG, PNG, etc.)
            return_probabilities: Whether to return class probabilities
            return_features: Whether to return extracted features
            
        Returns:
            Dictionary with prediction results
        """
        # Load and preprocess image
        logger.info(f"Loading certificate: {certificate_path}")
        image = self.preprocessor.load_certificate(certificate_path)
        preprocessed = self.preprocessor.preprocess(image)
        
        # Apply transforms
        transformed = self.transform(image=preprocessed)
        img_tensor = transformed['image'].unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]
            predicted_class = torch.argmax(probabilities).item()
        
        # Get probabilities
        prob_fake = probabilities[0].item()
        prob_genuine = probabilities[1].item()
        
        # Determine prediction
        prediction = "GENUINE" if predicted_class == 1 else "FAKE"
        confidence = prob_genuine if predicted_class == 1 else prob_fake
        
        result = {
            'prediction': prediction,
            'confidence': float(confidence),
            'predicted_class': predicted_class
        }
        
        if return_probabilities:
            result['probabilities'] = {
                'fake': float(prob_fake),
                'genuine': float(prob_genuine)
            }
        
        # Extract features if requested
        if return_features:
            logger.info("Extracting features...")
            features = self.feature_extractor.extract_all_features(image)
            
            # Convert features to serializable format
            feature_summary = {
                'logo': {
                    'num_keypoints': features['logo_features']['num_keypoints'],
                    'num_contours': features['logo_features']['num_contours']
                },
                'signature': {
                    'num_strokes': features['signature_features']['num_strokes'],
                    'stroke_density': float(features['signature_features']['stroke_density'])
                },
                'stamp': {
                    'num_circles': features['stamp_features']['num_circles']
                },
                'text': {
                    'num_words': features['text_features']['num_words'],
                    'avg_confidence': float(features['text_features']['avg_confidence'])
                },
                'edge': {
                    'edge_density': float(features['edge_features']['edge_density'])
                }
            }
            
            result['features'] = feature_summary
        
        return result
    
    def predict_batch(
        self,
        certificate_paths: list,
        return_probabilities: bool = True
    ) -> list:
        """
        Predict authenticity for multiple certificates.
        
        Args:
            certificate_paths: List of paths to certificate files
            return_probabilities: Whether to return class probabilities
            
        Returns:
            List of prediction results
        """
        results = []
        
        for cert_path in certificate_paths:
            try:
                result = self.predict(
                    cert_path,
                    return_probabilities=return_probabilities,
                    return_features=False
                )
                result['file'] = cert_path
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {cert_path}: {str(e)}")
                results.append({
                    'file': cert_path,
                    'error': str(e),
                    'prediction': 'ERROR',
                    'confidence': 0.0
                })
        
        return results


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Predict certificate authenticity'
    )
    parser.add_argument(
        'certificate_path',
        type=str,
        help='Path to certificate file (PDF, JPG, PNG, etc.)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='outputs/checkpoints/best_model.pth',
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use for inference'
    )
    parser.add_argument(
        '--img-size',
        type=int,
        default=1024,
        help='Input image size'
    )
    parser.add_argument(
        '--features',
        action='store_true',
        help='Extract and display features'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Save results to JSON file'
    )
    
    args = parser.parse_args()
    
    # Create predictor
    predictor = CertificateAuthenticityPredictor(
        model_path=args.model,
        device=args.device,
        img_size=args.img_size
    )
    
    # Make prediction
    result = predictor.predict(
        args.certificate_path,
        return_probabilities=True,
        return_features=args.features
    )
    
    # Display results
    print("\n" + "="*50)
    print("CERTIFICATE AUTHENTICITY ANALYSIS")
    print("="*50)
    print(f"\nFile: {args.certificate_path}")
    print(f"\nPrediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2%}")
    
    if 'probabilities' in result:
        print(f"\nProbabilities:")
        print(f"  Fake:    {result['probabilities']['fake']:.2%}")
        print(f"  Genuine: {result['probabilities']['genuine']:.2%}")
    
    if 'features' in result:
        print(f"\nExtracted Features:")
        print(f"  Logo Keypoints: {result['features']['logo']['num_keypoints']}")
        print(f"  Signature Strokes: {result['features']['signature']['num_strokes']}")
        print(f"  Stamp Circles: {result['features']['stamp']['num_circles']}")
        print(f"  Text Words: {result['features']['text']['num_words']}")
        print(f"  OCR Confidence: {result['features']['text']['avg_confidence']:.2f}")
    
    print("\n" + "="*50)
    
    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
