"""
Non-Maximum Suppression (NMS) implementations for face detection.
Includes Hard-NMS and Soft-NMS algorithms for bbox filtering.
"""

import torch
import numpy as np
from typing import Tuple, List, Dict


class NMSProcessor:
    """
    Non-Maximum Suppression processor.
    Implements Hard-NMS, Soft-NMS, and weighted NMS variants.
    """
    
    @staticmethod
    def hard_nms(
        bboxes: torch.Tensor,
        scores: torch.Tensor,
        iou_threshold: float = 0.4
    ) -> torch.Tensor:
        """
        Hard Non-Maximum Suppression.
        Removes boxes with IoU > threshold with higher-scoring boxes.
        
        Args:
            bboxes: Bounding boxes [x1, y1, x2, y2]
            scores: Confidence scores
            iou_threshold: IoU threshold for suppression
            
        Returns:
            Indices of kept boxes
        """
        if len(bboxes) == 0:
            return torch.tensor([], dtype=torch.long)
        
        # Sort by score in descending order
        sorted_idx = torch.argsort(scores, descending=True)
        keep = []
        
        while len(sorted_idx) > 0:
            # Keep the highest score box
            current_idx = sorted_idx[0]
            keep.append(current_idx.item())
            
            if len(sorted_idx) == 1:
                break
            
            # Calculate IoU with remaining boxes
            current_box = bboxes[current_idx]
            remaining_boxes = bboxes[sorted_idx[1:]]
            
            iou = NMSProcessor._compute_iou(current_box, remaining_boxes)
            
            # Keep only boxes with IoU < threshold
            mask = iou < iou_threshold
            sorted_idx = sorted_idx[1:][mask]
        
        return torch.tensor(keep, dtype=torch.long)
    
    @staticmethod
    def soft_nms(
        bboxes: torch.Tensor,
        scores: torch.Tensor,
        iou_threshold: float = 0.4,
        sigma: float = 0.5,
        score_threshold: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Soft Non-Maximum Suppression.
        Reduces scores of nearby boxes instead of removing them.
        
        Args:
            bboxes: Bounding boxes [x1, y1, x2, y2]
            scores: Confidence scores
            iou_threshold: IoU threshold for soft suppression
            sigma: Gaussian parameter for score decay
            score_threshold: Minimum score to keep box
            
        Returns:
            Tuple of (kept_bboxes, kept_scores, kept_indices)
        """
        if len(bboxes) == 0:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])
        
        # Sort by score in descending order
        sorted_idx = torch.argsort(scores, descending=True)
        bboxes_sorted = bboxes[sorted_idx]
        scores_sorted = scores[sorted_idx].clone()
        
        keep_idx = []
        keep_scores = []
        keep_bboxes = []
        
        while len(scores_sorted) > 0:
            # Keep highest score box
            current_idx = 0
            current_bbox = bboxes_sorted[current_idx]
            current_score = scores_sorted[current_idx]
            
            if current_score < score_threshold:
                break
            
            keep_idx.append(sorted_idx[current_idx].item())
            keep_bboxes.append(current_bbox)
            keep_scores.append(current_score)
            
            if len(scores_sorted) == 1:
                break
            
            # Calculate IoU with remaining boxes
            remaining_bboxes = bboxes_sorted[1:]
            remaining_scores = scores_sorted[1:]
            
            iou = NMSProcessor._compute_iou(current_bbox, remaining_bboxes)
            
            # Apply Gaussian penalty to scores
            weight = torch.exp(-(iou ** 2) / sigma)
            scores_sorted = remaining_scores * weight
            bboxes_sorted = remaining_bboxes
            sorted_idx = sorted_idx[1:]
        
        if len(keep_bboxes) == 0:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])
        
        return (
            torch.stack(keep_bboxes),
            torch.stack(keep_scores),
            torch.tensor(keep_idx, dtype=torch.long)
        )
    
    @staticmethod
    def weighted_nms(
        bboxes_list: List[torch.Tensor],
        scores_list: List[torch.Tensor],
        weights: List[float],
        iou_threshold: float = 0.4
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Weighted NMS for ensemble predictions.
        Combines predictions from multiple detectors with weights.
        
        Args:
            bboxes_list: List of bbox tensors from different detectors
            scores_list: List of score tensors from different detectors
            weights: Weights for each detector (should sum to 1)
            iou_threshold: IoU threshold for NMS
            
        Returns:
            Tuple of (final_bboxes, final_scores)
        """
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Combine all detections
        all_bboxes = []
        all_scores = []
        
        for bboxes, scores, weight in zip(bboxes_list, scores_list, weights):
            all_bboxes.append(bboxes)
            all_scores.append(scores * weight)
        
        combined_bboxes = torch.cat(all_bboxes, dim=0)
        combined_scores = torch.cat(all_scores, dim=0)
        
        # Apply Hard-NMS
        keep_idx = NMSProcessor.hard_nms(combined_bboxes, combined_scores, iou_threshold)
        
        final_bboxes = combined_bboxes[keep_idx]
        final_scores = combined_scores[keep_idx]
        
        return final_bboxes, final_scores
    
    @staticmethod
    def _compute_iou(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        """
        Compute IoU between one box and multiple boxes.
        
        Args:
            box: Single box [x1, y1, x2, y2]
            boxes: Multiple boxes [N, 4]
            
        Returns:
            IoU scores [N]
        """
        # Intersection
        inter_x1 = torch.max(box[0], boxes[:, 0])
        inter_y1 = torch.max(box[1], boxes[:, 1])
        inter_x2 = torch.min(box[2], boxes[:, 2])
        inter_y2 = torch.min(box[3], boxes[:, 3])
        
        inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
        inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
        inter_area = inter_w * inter_h
        
        # Union
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union_area = box_area + boxes_area - inter_area
        
        # IoU
        iou = inter_area / (union_area + 1e-6)
        
        return iou
    
    @staticmethod
    def batch_nms(
        bboxes: torch.Tensor,
        scores: torch.Tensor,
        class_ids: torch.Tensor,
        iou_threshold: float = 0.4
    ) -> torch.Tensor:
        """
        NMS per class (for multi-class detection).
        
        Args:
            bboxes: Bounding boxes [N, 4]
            scores: Confidence scores [N]
            class_ids: Class IDs [N]
            iou_threshold: IoU threshold
            
        Returns:
            Indices of kept boxes
        """
        keep_idx = []
        
        # Process each class separately
        for class_id in torch.unique(class_ids):
            class_mask = class_ids == class_id
            class_bboxes = bboxes[class_mask]
            class_scores = scores[class_mask]
            class_indices = torch.where(class_mask)[0]
            
            # Apply NMS to this class
            keep = NMSProcessor.hard_nms(class_bboxes, class_scores, iou_threshold)
            
            # Map back to original indices
            keep_idx.extend(class_indices[keep].tolist())
        
        return torch.tensor(keep_idx, dtype=torch.long)


def apply_nms(
    detections: Dict,
    nms_type: str = 'hard',
    iou_threshold: float = 0.4,
    confidence_threshold: float = 0.3,
    sigma: float = 0.5
) -> Dict:
    """
    Apply NMS to detection results.
    
    Args:
        detections: Dictionary with 'bboxes' and 'scores' keys
        nms_type: Type of NMS ('hard' or 'soft')
        iou_threshold: IoU threshold
        confidence_threshold: Minimum confidence to keep
        sigma: Sigma parameter for Soft-NMS
        
    Returns:
        Filtered detections dictionary
    """
    bboxes = detections['bboxes']
    scores = detections['scores']
    
    # Filter by confidence threshold
    conf_mask = scores >= confidence_threshold
    bboxes = bboxes[conf_mask]
    scores = scores[conf_mask]
    
    if len(bboxes) == 0:
        return {
            'bboxes': torch.tensor([]),
            'scores': torch.tensor([]),
            'count': 0
        }
    
    # Apply NMS
    if nms_type.lower() == 'soft':
        bboxes, scores, _ = NMSProcessor.soft_nms(
            bboxes, scores, iou_threshold, sigma, confidence_threshold
        )
    else:  # Hard NMS
        keep_idx = NMSProcessor.hard_nms(bboxes, scores, iou_threshold)
        bboxes = bboxes[keep_idx]
        scores = scores[keep_idx]
    
    return {
        'bboxes': bboxes,
        'scores': scores,
        'count': len(bboxes)
    }
