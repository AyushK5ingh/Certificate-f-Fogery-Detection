"""
Model architecture for certificate authenticity detection.
Uses EfficientNet backbone with multi-head attention for high accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm
from typing import Tuple, Optional


class MultiHeadAttention(nn.Module):
    """Multi-head attention module for feature importance."""
    
    def __init__(self, embed_dim: int, num_heads: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        # x shape: (batch, seq_len, embed_dim)
        attended, _ = self.attention(x, x, x)
        return self.norm(x + attended)


class CertificateAuthenticityModel(nn.Module):
    """
    Advanced model for certificate authenticity detection.
    Uses EfficientNet-B3 as backbone with attention mechanism.
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.3
    ):
        """
        Initialize model.
        
        Args:
            num_classes: Number of output classes (2 for binary: fake/genuine)
            pretrained: Whether to use pretrained weights
            dropout: Dropout rate for regularization
        """
        super().__init__()
        
        # Load EfficientNet-B3 as backbone (good balance of accuracy and speed)
        self.backbone = timm.create_model(
            'efficientnet_b3',
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool=''  # Remove global pooling
        )
        
        # Get feature dimension from backbone
        self.feature_dim = self.backbone.num_features
        
        # Spatial attention module
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(self.feature_dim, self.feature_dim // 8, 1),
            nn.BatchNorm2d(self.feature_dim // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feature_dim // 8, 1, 1),
            nn.Sigmoid()
        )
        
        # Channel attention (Squeeze-and-Excitation)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.feature_dim, self.feature_dim // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feature_dim // 16, self.feature_dim, 1),
            nn.Sigmoid()
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classification head with multiple FC layers
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes)
        )
        
        # Auxiliary classifier for deep supervision (helps with training)
        self.aux_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.feature_dim, num_classes)
        )
        
    def forward(self, x, return_features=False):
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, channels, height, width)
            return_features: Whether to return intermediate features
            
        Returns:
            Logits or (logits, features) if return_features=True
        """
        # Extract features from backbone
        features = self.backbone(x)
        
        # Apply spatial attention
        spatial_att = self.spatial_attention(features)
        features = features * spatial_att
        
        # Apply channel attention
        channel_att = self.channel_attention(features)
        features = features * channel_att
        
        # Global pooling
        pooled = self.global_pool(features)
        pooled = pooled.flatten(1)
        
        # Main classification
        logits = self.classifier(pooled)
        
        if return_features:
            return logits, pooled
        
        return logits
    
    def forward_with_aux(self, x):
        """
        Forward pass with auxiliary output for training.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (main_logits, aux_logits)
        """
        # Extract features
        features = self.backbone(x)
        
        # Auxiliary prediction (early exit)
        aux_logits = self.aux_classifier(features)
        
        # Apply attentions
        spatial_att = self.spatial_attention(features)
        features = features * spatial_att
        
        channel_att = self.channel_attention(features)
        features = features * channel_att
        
        # Main prediction
        pooled = self.global_pool(features)
        pooled = pooled.flatten(1)
        main_logits = self.classifier(pooled)
        
        return main_logits, aux_logits


class EnsembleModel(nn.Module):
    """
    Ensemble of multiple models for higher accuracy.
    """
    
    def __init__(self, models_list: list):
        """
        Initialize ensemble.
        
        Args:
            models_list: List of models to ensemble
        """
        super().__init__()
        self.models = nn.ModuleList(models_list)
        
    def forward(self, x):
        """
        Forward pass through all models and average predictions.
        
        Args:
            x: Input tensor
            
        Returns:
            Averaged logits
        """
        outputs = []
        for model in self.models:
            output = model(x)
            outputs.append(output)
        
        # Average predictions
        return torch.mean(torch.stack(outputs), dim=0)


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Helps model focus on hard examples.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor
            gamma: Focusing parameter
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        """
        Compute focal loss.
        
        Args:
            inputs: Model predictions (logits)
            targets: Ground truth labels
            
        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def create_model(
    model_name: str = 'efficientnet_b3',
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.3
) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        model_name: Name of the model architecture
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        dropout: Dropout rate
        
    Returns:
        Model instance
    """
    if model_name == 'efficientnet_b3':
        return CertificateAuthenticityModel(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout
        )
    elif model_name.startswith('efficientnet'):
        # Support other EfficientNet variants
        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes
        )
        return model
    else:
        raise ValueError(f"Unknown model: {model_name}")


def count_parameters(model: nn.Module) -> int:
    """
    Count trainable parameters in model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model creation
    model = create_model('efficientnet_b3', num_classes=2)
    print(f"Model created successfully!")
    print(f"Total parameters: {count_parameters(model):,}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 1024, 1024)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
