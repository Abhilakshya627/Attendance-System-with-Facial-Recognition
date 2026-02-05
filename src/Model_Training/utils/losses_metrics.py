"""
Loss functions and metrics for face detection training.
Optimized for high recall performance.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Dict


class FocalLoss(torch.nn.Module):
    """
    Focal Loss from "Focal Loss for Dense Object Detection"
    (https://arxiv.org/abs/1708.02002)
    
    Used for addressing class imbalance in face detection.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weight for background class
            gamma: Focusing parameter
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Model predictions (logits)
            targets: Ground truth labels (0 or 1)
            
        Returns:
            Scalar loss value
        """
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * focal_weight * ce_loss
        
        return loss.mean()


class IoULoss(torch.nn.Module):
    """
    IoU (Intersection over Union) Loss for bounding box regression.
    Better for object detection than L2 loss.
    """
    
    def forward(self, pred_bboxes: torch.Tensor, target_bboxes: torch.Tensor) -> torch.Tensor:
        """
        Compute IoU loss.
        
        Args:
            pred_bboxes: Predicted bounding boxes [x1, y1, x2, y2]
            target_bboxes: Target bounding boxes [x1, y1, x2, y2]
            
        Returns:
            Scalar loss value
        """
        # Calculate intersection area
        inter_x1 = torch.max(pred_bboxes[:, 0], target_bboxes[:, 0])
        inter_y1 = torch.max(pred_bboxes[:, 1], target_bboxes[:, 1])
        inter_x2 = torch.min(pred_bboxes[:, 2], target_bboxes[:, 2])
        inter_y2 = torch.min(pred_bboxes[:, 3], target_bboxes[:, 3])
        
        inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
        inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
        inter_area = inter_w * inter_h
        
        # Calculate union area
        pred_area = (pred_bboxes[:, 2] - pred_bboxes[:, 0]) * (pred_bboxes[:, 3] - pred_bboxes[:, 1])
        target_area = (target_bboxes[:, 2] - target_bboxes[:, 0]) * (target_bboxes[:, 3] - target_bboxes[:, 1])
        union_area = pred_area + target_area - inter_area
        
        # Calculate IoU
        iou = inter_area / (union_area + 1e-6)
        
        # IoU loss
        loss = 1 - iou
        
        return loss.mean()


class CombinedLoss(torch.nn.Module):
    """
    Combined loss for face detection training.
    Combines classification loss (Focal Loss) and regression loss (IoU Loss).
    """
    
    def __init__(self, lambda_reg: float = 1.0, alpha: float = 0.25, gamma: float = 2.0):
        """
        Initialize combined loss.
        
        Args:
            lambda_reg: Weight for regression loss
            alpha: Alpha parameter for focal loss
            gamma: Gamma parameter for focal loss
        """
        super().__init__()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.iou_loss = IoULoss()
        self.lambda_reg = lambda_reg
    
    def forward(
        self,
        class_pred: torch.Tensor,
        bbox_pred: torch.Tensor,
        class_target: torch.Tensor,
        bbox_target: torch.Tensor,
        bbox_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined loss.
        
        Args:
            class_pred: Classification predictions
            bbox_pred: Bbox predictions
            class_target: Classification targets
            bbox_target: Bbox targets
            bbox_mask: Mask for valid bboxes
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Classification loss
        class_loss = self.focal_loss(class_pred, class_target)
        
        # Regression loss (only for positive samples)
        if bbox_mask.sum() > 0:
            bbox_loss = self.iou_loss(
                bbox_pred[bbox_mask],
                bbox_target[bbox_mask]
            )
        else:
            bbox_loss = torch.tensor(0.0, device=class_pred.device)
        
        # Combined loss
        total_loss = class_loss + self.lambda_reg * bbox_loss
        
        loss_dict = {
            'class_loss': class_loss.detach(),
            'bbox_loss': bbox_loss.detach() if isinstance(bbox_loss, torch.Tensor) else torch.tensor(0.0),
            'total_loss': total_loss.detach()
        }
        
        return total_loss, loss_dict


class DetectionMetrics:
    """
    Metrics for face detection evaluation.
    Calculates Precision, Recall, F1 score, and mAP.
    """
    
    def __init__(self, iou_threshold: float = 0.5):
        """
        Initialize metrics calculator.
        
        Args:
            iou_threshold: IoU threshold for matching predictions to ground truth
        """
        self.iou_threshold = iou_threshold
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.predictions = []
        self.ground_truths = []
    
    def reset(self):
        """Reset metrics."""
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.predictions = []
        self.ground_truths = []
    
    def update(
        self,
        predictions: torch.Tensor,
        confidence: torch.Tensor,
        ground_truth: torch.Tensor,
        masks: torch.Tensor
    ):
        """
        Update metrics with batch predictions.
        
        Args:
            predictions: Predicted bounding boxes [batch_size, num_preds, 4]
            confidence: Confidence scores [batch_size, num_preds]
            ground_truth: Ground truth bboxes [batch_size, num_gt, 4]
            masks: Mask for valid ground truth boxes [batch_size, num_gt]
        """
        batch_size = predictions.shape[0]
        
        for i in range(batch_size):
            pred_boxes = predictions[i]
            pred_conf = confidence[i]
            gt_boxes = ground_truth[i]
            gt_mask = masks[i]
            
            # Filter valid ground truth
            valid_gt = gt_boxes[gt_mask]
            
            if len(valid_gt) == 0:
                # No ground truth
                self.fp += len(pred_boxes)
                continue
            
            # Match predictions to ground truth
            matched = [False] * len(valid_gt)
            
            for pred_box in pred_boxes:
                best_iou = 0
                best_idx = -1
                
                for j, gt_box in enumerate(valid_gt):
                    if matched[j]:
                        continue
                    
                    iou = self._compute_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = j
                
                if best_iou >= self.iou_threshold:
                    self.tp += 1
                    matched[best_idx] = True
                else:
                    self.fp += 1
            
            # False negatives
            self.fn += len(valid_gt) - sum(matched)
    
    @staticmethod
    def _compute_iou(box1: torch.Tensor, box2: torch.Tensor) -> float:
        """Compute IoU between two boxes."""
        x1_inter = max(box1[0].item(), box2[0].item())
        y1_inter = max(box1[1].item(), box2[1].item())
        x2_inter = min(box1[2].item(), box2[2].item())
        y2_inter = min(box1[3].item(), box2[3].item())
        
        inter_w = max(0, x2_inter - x1_inter)
        inter_h = max(0, y2_inter - y1_inter)
        inter_area = inter_w * inter_h
        
        box1_area = (box1[2].item() - box1[0].item()) * (box1[3].item() - box1[1].item())
        box2_area = (box2[2].item() - box2[0].item()) * (box2[3].item() - box2[1].item())
        union_area = box1_area + box2_area - inter_area
        
        iou = inter_area / (union_area + 1e-6)
        return iou
    
    def compute(self) -> Dict[str, float]:
        """
        Compute final metrics.
        
        Returns:
            Dictionary with precision, recall, f1, and support
        """
        precision = self.tp / (self.tp + self.fp + 1e-6)
        recall = self.tp / (self.tp + self.fn + 1e-6)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': self.tp,
            'fp': self.fp,
            'fn': self.fn,
            'support': self.tp + self.fn
        }
