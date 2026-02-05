"""
RetinaFace trainer for face detection.
Primary detector with anchor-dense architecture suitable for small faces.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Tuple, Optional
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import (
    create_data_loaders,
    CombinedLoss,
    DetectionMetrics,
    setup_logging,
    save_checkpoint,
    create_optimizer,
    create_scheduler,
    count_parameters
)


class RetinaFaceBackbone(nn.Module):
    """
    Simplified RetinaFace backbone using ResNet-50.
    For full implementation, use pre-trained retinaface library.
    """
    
    def __init__(self, pretrained: bool = True):
        """
        Initialize RetinaFace backbone.
        
        Args:
            pretrained: Whether to use pretrained weights
        """
        super().__init__()
        
        # Load ResNet50 backbone
        from torchvision import models
        resnet = models.resnet50(pretrained=pretrained)
        
        # Extract layers for feature pyramid
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass to extract multi-scale features.
        
        Args:
            x: Input image tensor [batch_size, 3, H, W]
            
        Returns:
            Dictionary with multi-scale features
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        
        return {'c2': c2, 'c3': c3, 'c4': c4}


class FPN(nn.Module):
    """
    Feature Pyramid Network for multi-scale feature extraction.
    """
    
    def __init__(self, in_channels_list: list = [512, 1024, 2048], out_channels: int = 256):
        """
        Initialize FPN.
        
        Args:
            in_channels_list: Input channel sizes from backbone
            out_channels: Output channel size for all pyramid levels
        """
        super().__init__()
        self.out_channels = out_channels
        
        # Lateral convolutions
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            for in_channels in in_channels_list
        ])
        
        # Smooth convolutions
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels_list
        ])
    
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through FPN.
        
        Args:
            features: Multi-scale features from backbone
            
        Returns:
            Dictionary with FPN features at different scales
        """
        # Get feature maps in order
        feature_list = [features['c2'], features['c3'], features['c4']]
        
        # Lateral convolutions
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, feature_list)]
        
        # Top-down path
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] += torch.nn.functional.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], 
                mode='nearest'
            )
        
        # Smooth convolutions
        outputs = [smooth(lat) for smooth, lat in zip(self.smooth_convs, laterals)]
        
        return {'p2': outputs[0], 'p3': outputs[1], 'p4': outputs[2]}


class RetinaFaceHead(nn.Module):
    """
    RetinaFace detection head.
    Predicts face classifications and bounding box regressions.
    """
    
    def __init__(self, in_channels: int = 256, num_anchors: int = 6):
        """
        Initialize detection head.
        
        Args:
            in_channels: Input channel size
            num_anchors: Number of anchors per feature map location
        """
        super().__init__()
        self.num_anchors = num_anchors
        
        # Classification head
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_anchors * 2, kernel_size=3, padding=1)
        )
        
        # Bbox regression head
        self.bbox_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, padding=1)
        )
    
    def forward(self, features: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through detection head.
        
        Args:
            features: FPN features
            
        Returns:
            Tuple of (classification logits, bbox predictions)
        """
        cls_outputs = []
        bbox_outputs = []
        
        for key in ['p2', 'p3', 'p4']:
            feat = features[key]
            cls = self.cls_head(feat)
            bbox = self.bbox_head(feat)
            
            # Reshape: [batch, num_anchors*2, H, W] -> [batch, H, W, num_anchors, 2]
            batch_size = cls.shape[0]
            cls = cls.permute(0, 2, 3, 1).contiguous()
            cls = cls.view(batch_size, -1, 2)
            
            bbox = bbox.permute(0, 2, 3, 1).contiguous()
            bbox = bbox.view(batch_size, -1, 4)
            
            cls_outputs.append(cls)
            bbox_outputs.append(bbox)
        
        # Concatenate outputs from all scales
        cls_preds = torch.cat(cls_outputs, dim=1)
        bbox_preds = torch.cat(bbox_outputs, dim=1)
        
        return cls_preds, bbox_preds


class RetinaFace(nn.Module):
    """
    Complete RetinaFace model for face detection.
    """
    
    def __init__(self, pretrained_backbone: bool = True):
        """
        Initialize RetinaFace model.
        
        Args:
            pretrained_backbone: Whether to use pretrained backbone
        """
        super().__init__()
        
        self.backbone = RetinaFaceBackbone(pretrained=pretrained_backbone)
        self.fpn = FPN()
        self.head = RetinaFaceHead()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            x: Input image tensor [batch_size, 3, H, W]
            
        Returns:
            Tuple of (classification logits, bbox predictions)
        """
        backbone_features = self.backbone(x)
        fpn_features = self.fpn(backbone_features)
        cls_preds, bbox_preds = self.head(fpn_features)
        
        return cls_preds, bbox_preds


class RetinaFaceTrainer:
    """
    Trainer class for RetinaFace model.
    Handles training loop, validation, and checkpointing.
    """
    
    def __init__(
        self,
        config: Dict,
        output_dir: str = './outputs/retinaface',
        device: str = 'cuda'
    ):
        """
        Initialize trainer.
        
        Args:
            config: Configuration dictionary
            output_dir: Output directory for checkpoints and logs
            device: Device to train on ('cuda' or 'cpu')
        """
        self.config = config
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = setup_logging(str(self.output_dir), 'retinaface_training')
        
        # Initialize model
        self.model = RetinaFace(pretrained_backbone=config.get('pretrained_backbone', True))
        self.model = self.model.to(self.device)
        
        # Log model info
        num_params = count_parameters(self.model)
        self.logger.info(f"RetinaFace model created with {num_params:,} parameters")
        
        # Initialize loss function
        self.criterion = CombinedLoss(
            lambda_reg=config.get('lambda_reg', 1.0),
            alpha=config.get('focal_alpha', 0.25),
            gamma=config.get('focal_gamma', 2.0)
        )
        
        # Initialize optimizer and scheduler
        self.optimizer = create_optimizer(
            self.model,
            lr=config.get('learning_rate', 0.001),
            optimizer_type=config.get('optimizer', 'adamw')
        )
        
        self.scheduler = create_scheduler(
            self.optimizer,
            scheduler_type=config.get('scheduler', 'cosine'),
            num_epochs=config.get('num_epochs', 100)
        )
        
        # Metrics
        self.train_metrics = DetectionMetrics()
        self.val_metrics = DetectionMetrics()
        
        # Training state
        self.best_loss = float('inf')
        self.start_epoch = 0
    
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary with loss values
        """
        self.model.train()
        self.train_metrics.reset()
        
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch['images'].to(self.device)
            bboxes = batch['bboxes'].to(self.device)
            masks = batch['masks'].to(self.device)
            
            # Forward pass
            cls_preds, bbox_preds = self.model(images)
            
            # Prepare targets (simplified - in practice, need anchor matching)
            batch_size = images.shape[0]
            num_preds = cls_preds.shape[1]
            
            # Create dummy targets for loss calculation
            class_targets = torch.zeros_like(cls_preds)
            bbox_targets = torch.zeros_like(bbox_preds)
            bbox_masks = torch.zeros(batch_size, num_preds, dtype=torch.bool, device=self.device)
            
            # Calculate loss
            loss, loss_dict = self.criterion(
                cls_preds,
                bbox_preds,
                class_targets,
                bbox_targets,
                bbox_masks
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if (batch_idx + 1) % self.config.get('log_interval', 10) == 0:
                self.logger.info(
                    f"Batch {batch_idx + 1}/{len(train_loader)} - "
                    f"Loss: {loss.item():.4f}"
                )
        
        avg_loss = total_loss / num_batches
        return {'loss': avg_loss}
    
    def validate(self, val_loader) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        self.val_metrics.reset()
        
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['images'].to(self.device)
                bboxes = batch['bboxes'].to(self.device)
                masks = batch['masks'].to(self.device)
                
                # Forward pass
                cls_preds, bbox_preds = self.model(images)
                
                # Dummy targets for loss calculation
                batch_size = images.shape[0]
                num_preds = cls_preds.shape[1]
                
                class_targets = torch.zeros_like(cls_preds)
                bbox_targets = torch.zeros_like(bbox_preds)
                bbox_masks = torch.zeros(batch_size, num_preds, dtype=torch.bool, device=self.device)
                
                # Calculate loss
                loss, loss_dict = self.criterion(
                    cls_preds,
                    bbox_preds,
                    class_targets,
                    bbox_targets,
                    bbox_masks
                )
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return {'loss': avg_loss}
    
    def train(self, train_loader, val_loader):
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        num_epochs = self.config.get('num_epochs', 100)
        
        self.logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(self.start_epoch, num_epochs):
            self.logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            self.logger.info(f"Train Loss: {train_loss['loss']:.4f}")
            
            # Validate
            val_loss = self.validate(val_loader)
            self.logger.info(f"Val Loss: {val_loss['loss']:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Save checkpoint
            if val_loss['loss'] < self.best_loss:
                self.best_loss = val_loss['loss']
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    self.best_loss,
                    str(self.output_dir),
                    'retinaface'
                )
                self.logger.info(f"Saved best model with loss: {self.best_loss:.4f}")
        
        self.logger.info("Training completed!")
        return self.best_loss
