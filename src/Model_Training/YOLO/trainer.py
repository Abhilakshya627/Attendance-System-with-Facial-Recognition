"""
YOLO-based face detector trainer.
Auxiliary detector for speed optimization and ensemble with RetinaFace.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Tuple
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import (
    CombinedLoss,
    DetectionMetrics,
    setup_logging,
    save_checkpoint,
    create_optimizer,
    create_scheduler,
    count_parameters
)


class YOLOBackbone(nn.Module):
    """
    Simplified YOLO backbone using darknet-like architecture.
    For full implementation, use YOLOv8 or similar.
    """
    
    def __init__(self, num_classes: int = 1):
        """
        Initialize YOLO backbone.
        
        Args:
            num_classes: Number of classes (1 for face detection)
        """
        super().__init__()
        self.num_classes = num_classes
        
        # Darknet-like backbone
        self.conv1 = self._make_conv_layer(3, 32, 3, 1)
        self.conv2 = self._make_conv_layer(32, 64, 3, 2)
        self.conv3 = self._make_conv_layer(64, 64, 3, 1)
        self.conv4 = self._make_conv_layer(64, 128, 3, 2)
        self.conv5 = self._make_conv_layer(128, 128, 3, 1)
        self.conv6 = self._make_conv_layer(128, 256, 3, 2)
        self.conv7 = self._make_conv_layer(256, 256, 3, 1)
        self.conv8 = self._make_conv_layer(256, 512, 3, 2)
        self.conv9 = self._make_conv_layer(512, 512, 3, 1)
    
    @staticmethod
    def _make_conv_layer(in_channels: int, out_channels: int, 
                         kernel_size: int, stride: int) -> nn.Sequential:
        """Create a conv layer with batch norm and activation."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
                     padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through backbone."""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        return x


class YOLOHead(nn.Module):
    """
    YOLO detection head.
    Predicts objectness, class, and bounding box coordinates.
    """
    
    def __init__(self, in_channels: int = 512, num_classes: int = 1, 
                 num_anchors: int = 3):
        """
        Initialize YOLO head.
        
        Args:
            in_channels: Input channel size
            num_classes: Number of object classes
            num_anchors: Number of anchor boxes per grid cell
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Output size: (objectness + classes + bbox) * num_anchors
        output_channels = num_anchors * (1 + num_classes + 4)
        
        self.detection_head = nn.Sequential(
            nn.Conv2d(in_channels, 1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, output_channels, kernel_size=1, padding=0)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through detection head.
        
        Args:
            x: Input feature map
            
        Returns:
            Tuple of (objectness, class_probs, bbox_preds)
        """
        output = self.detection_head(x)
        
        batch_size = output.shape[0]
        grid_size = output.shape[2]
        
        # Reshape: [batch, channels, H, W] -> [batch, H, W, num_anchors, (1+classes+4)]
        output = output.permute(0, 2, 3, 1).contiguous()
        output = output.view(batch_size, grid_size, grid_size, 
                            self.num_anchors, 1 + self.num_classes + 4)
        
        objectness = output[..., 0:1]
        class_probs = output[..., 1:1+self.num_classes]
        bbox_preds = output[..., 1+self.num_classes:]
        
        # Flatten spatial dimensions
        objectness = objectness.view(batch_size, -1, 1)
        class_probs = class_probs.view(batch_size, -1, self.num_classes)
        bbox_preds = bbox_preds.view(batch_size, -1, 4)
        
        return objectness, class_probs, bbox_preds


class YOLODetector(nn.Module):
    """
    Complete YOLO-based face detector.
    """
    
    def __init__(self, num_classes: int = 1, num_anchors: int = 3):
        """
        Initialize YOLO detector.
        
        Args:
            num_classes: Number of classes (1 for face detection)
            num_anchors: Number of anchors per grid cell
        """
        super().__init__()
        self.backbone = YOLOBackbone(num_classes)
        self.head = YOLOHead(512, num_classes, num_anchors)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            x: Input image tensor [batch_size, 3, H, W]
            
        Returns:
            Tuple of (objectness, class_probs, bbox_preds)
        """
        features = self.backbone(x)
        objectness, class_probs, bbox_preds = self.head(features)
        
        return objectness, class_probs, bbox_preds


class YOLOTrainer:
    """
    Trainer class for YOLO-based face detector.
    """
    
    def __init__(
        self,
        config: Dict,
        output_dir: str = './outputs/yolo',
        device: str = 'cuda'
    ):
        """
        Initialize YOLO trainer.
        
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
        self.logger = setup_logging(str(self.output_dir), 'yolo_training')
        
        # Initialize model
        num_classes = config.get('num_classes', 1)
        num_anchors = config.get('num_anchors', 3)
        
        self.model = YOLODetector(num_classes, num_anchors)
        self.model = self.model.to(self.device)
        
        # Log model info
        num_params = count_parameters(self.model)
        self.logger.info(f"YOLO detector created with {num_params:,} parameters")
        
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
            objectness, class_probs, bbox_preds = self.model(images)
            
            # Prepare targets (simplified)
            batch_size = images.shape[0]
            num_preds = objectness.shape[1]
            
            objectness_target = torch.zeros_like(objectness)
            class_target = torch.zeros_like(class_probs)
            bbox_target = torch.zeros_like(bbox_preds)
            bbox_mask = torch.zeros(batch_size, num_preds, dtype=torch.bool, device=self.device)
            
            # Combine objectness and class predictions
            class_preds = torch.cat([objectness, class_probs], dim=-1)
            class_targets = torch.cat([objectness_target, class_target], dim=-1)
            
            # Calculate loss
            loss, loss_dict = self.criterion(
                class_preds,
                bbox_preds,
                class_targets,
                bbox_target,
                bbox_mask
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
                objectness, class_probs, bbox_preds = self.model(images)
                
                # Dummy targets
                batch_size = images.shape[0]
                num_preds = objectness.shape[1]
                
                objectness_target = torch.zeros_like(objectness)
                class_target = torch.zeros_like(class_probs)
                bbox_target = torch.zeros_like(bbox_preds)
                bbox_mask = torch.zeros(batch_size, num_preds, dtype=torch.bool, device=self.device)
                
                # Combine predictions
                class_preds = torch.cat([objectness, class_probs], dim=-1)
                class_targets = torch.cat([objectness_target, class_target], dim=-1)
                
                # Calculate loss
                loss, loss_dict = self.criterion(
                    class_preds,
                    bbox_preds,
                    class_targets,
                    bbox_target,
                    bbox_mask
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
        
        self.logger.info(f"Starting YOLO training for {num_epochs} epochs")
        
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
                    'yolo'
                )
                self.logger.info(f"Saved best model with loss: {self.best_loss:.4f}")
        
        self.logger.info("YOLO training completed!")
        return self.best_loss
