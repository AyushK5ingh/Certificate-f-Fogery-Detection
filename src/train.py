"""
Training script for certificate authenticity detection model.
Includes data augmentation, validation, and model checkpointing.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import yaml
import argparse
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
import logging

from model import create_model, FocalLoss
from preprocessing import CertificatePreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CertificateDataset(Dataset):
    """Dataset class for certificate images."""
    
    def __init__(self, images, labels, transform=None):
        """
        Initialize dataset.
        
        Args:
            images: List of image arrays
            labels: List of labels (0=fake, 1=genuine)
            transform: Albumentations transform
        """
        self.images = images
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            # Convert to tensor if no transform
            image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        return image, label


def get_train_transforms(img_size=1024):
    """Get training data augmentation transforms."""
    return A.Compose([
        A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0)),
            A.GaussianBlur(blur_limit=(3, 7)),
            A.MedianBlur(blur_limit=5),
        ], p=0.3),
        A.OneOf([
            A.OpticalDistortion(distort_limit=0.1),
            A.GridDistortion(num_steps=5, distort_limit=0.1),
        ], p=0.2),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_val_transforms(img_size=1024):
    """Get validation transforms (no augmentation)."""
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, writer=None):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item()})
        
        # Log to tensorboard
        if writer and batch_idx % 10 == 0:
            step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Train/BatchLoss', loss.item(), step)
    
    # Calculate metrics
    avg_loss = running_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    
    return avg_loss, accuracy, precision, recall, f1


def validate(model, dataloader, criterion, device, epoch, writer=None):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Get probabilities
            probs = torch.softmax(outputs, dim=1)[:, 1]  # Probability of genuine
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
    
    # Calculate metrics
    avg_loss = running_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    
    # ROC AUC
    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
    except:
        roc_auc = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return avg_loss, accuracy, precision, recall, f1, roc_auc, cm


def train(config):
    """Main training function."""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create output directories
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(exist_ok=True)
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Initialize tensorboard
    writer = SummaryWriter(log_dir=output_dir / 'logs')
    
    # Load data
    logger.info("Loading dataset...")
    preprocessor = CertificatePreprocessor(target_size=(config['img_size'], config['img_size']))
    dataset_path = Path(config['dataset_path'])
    
    train_images, train_labels = preprocessor.load_dataset(dataset_path, subset='train')
    val_images, val_labels = preprocessor.load_dataset(dataset_path, subset='valid')
    
    logger.info(f"Train set: {len(train_images)} images")
    logger.info(f"Val set: {len(val_images)} images")
    
    # Create datasets and dataloaders
    train_dataset = CertificateDataset(
        train_images, train_labels,
        transform=get_train_transforms(config['img_size'])
    )
    val_dataset = CertificateDataset(
        val_images, val_labels,
        transform=get_val_transforms(config['img_size'])
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # Create model
    logger.info(f"Creating model: {config['model_name']}")
    model = create_model(
        model_name=config['model_name'],
        num_classes=2,
        pretrained=config['pretrained'],
        dropout=config['dropout']
    )
    model = model.to(device)
    
    # Loss and optimizer
    if config['use_focal_loss']:
        criterion = FocalLoss(alpha=config['focal_alpha'], gamma=config['focal_gamma'])
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config['scheduler_t0'],
        T_mult=config['scheduler_tmult']
    )
    
    # Training loop
    best_f1 = 0.0
    best_accuracy = 0.0
    
    logger.info("Starting training...")
    for epoch in range(1, config['num_epochs'] + 1):
        # Train
        train_loss, train_acc, train_prec, train_rec, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer
        )
        
        # Validate
        val_loss, val_acc, val_prec, val_rec, val_f1, val_auc, cm = validate(
            model, val_loader, criterion, device, epoch, writer
        )
        
        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log metrics
        logger.info(f"\nEpoch {epoch}/{config['num_epochs']}")
        logger.info(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        logger.info(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")
        logger.info(f"LR: {current_lr:.6f}")
        logger.info(f"Confusion Matrix:\n{cm}")
        
        # Tensorboard logging
        writer.add_scalar('Train/Loss', train_loss, epoch)
        writer.add_scalar('Train/Accuracy', train_acc, epoch)
        writer.add_scalar('Train/F1', train_f1, epoch)
        writer.add_scalar('Val/Loss', val_loss, epoch)
        writer.add_scalar('Val/Accuracy', val_acc, epoch)
        writer.add_scalar('Val/F1', val_f1, epoch)
        writer.add_scalar('Val/AUC', val_auc, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_accuracy = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_acc': val_acc,
                'config': config
            }, checkpoint_dir / 'best_model.pth')
            logger.info(f"✓ Saved best model (F1: {val_f1:.4f})")
        
        # Save checkpoint every N epochs
        if epoch % config['save_frequency'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_acc': val_acc,
                'config': config
            }, checkpoint_dir / f'checkpoint_epoch_{epoch}.pth')
    
    writer.close()
    logger.info(f"\nTraining completed!")
    logger.info(f"Best F1: {best_f1:.4f}, Best Accuracy: {best_accuracy:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train certificate authenticity detector')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    train(config)
