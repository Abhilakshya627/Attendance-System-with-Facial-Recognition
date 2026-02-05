"""
Detection Fusion module for combining RetinaFace and YOLO predictions.
Implements weighted ensemble and Soft-NMS for optimal recall.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple, List
import numpy as np
from .nms import NMSProcessor, apply_nms


class DetectionFusion:
    """
    Fuses predictions from multiple face detectors (RetinaFace and YOLO).
    Optimizes for high recall using weighted ensemble and Soft-NMS.
    """
    
    def __init__(
        self,
        retinaface_weight: float = 0.6,
        yolo_weight: float = 0.4,
        iou_threshold: float = 0.5,
        confidence_threshold: float = 0.3,
        fusion_strategy: str = 'soft_nms',
        sigma: float = 0.5
    ):
        """
        Initialize detection fusion.
        
        Args:
            retinaface_weight: Weight for RetinaFace predictions
            yolo_weight: Weight for YOLO predictions
            iou_threshold: IoU threshold for NMS
            confidence_threshold: Minimum confidence to keep detection
            fusion_strategy: Fusion strategy ('soft_nms', 'hard_nms', 'weighted_average')
            sigma: Gaussian parameter for Soft-NMS
        """
        # Normalize weights
        total_weight = retinaface_weight + yolo_weight
        self.retinaface_weight = retinaface_weight / total_weight
        self.yolo_weight = yolo_weight / total_weight
        
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold
        self.fusion_strategy = fusion_strategy.lower()
        self.sigma = sigma
        
        # Validate fusion strategy
        valid_strategies = ['soft_nms', 'hard_nms', 'weighted_average']
        if self.fusion_strategy not in valid_strategies:
            raise ValueError(f"Fusion strategy must be one of {valid_strategies}")
    
    def fuse(
        self,
        retinaface_detections: Dict,
        yolo_detections: Dict,
        apply_nms_filter: bool = True
    ) -> Dict:
        """
        Fuse predictions from RetinaFace and YOLO.
        
        Args:
            retinaface_detections: Dictionary with 'bboxes' and 'scores'
            yolo_detections: Dictionary with 'bboxes' and 'scores'
            apply_nms_filter: Whether to apply NMS after fusion
            
        Returns:
            Fused detections dictionary
        """
        # Get detections
        rf_bboxes = retinaface_detections.get('bboxes', torch.tensor([]))
        rf_scores = retinaface_detections.get('scores', torch.tensor([]))
        
        yolo_bboxes = yolo_detections.get('bboxes', torch.tensor([]))
        yolo_scores = yolo_detections.get('scores', torch.tensor([]))
        
        # Handle empty predictions
        if len(rf_bboxes) == 0 and len(yolo_bboxes) == 0:
            return {
                'bboxes': torch.tensor([]),
                'scores': torch.tensor([]),
                'counts': {'retinaface': 0, 'yolo': 0, 'fused': 0},
                'strategy': self.fusion_strategy
            }
        
        # Apply fusion strategy
        if self.fusion_strategy == 'soft_nms':
            fused = self._fusion_soft_nms(rf_bboxes, rf_scores, yolo_bboxes, yolo_scores)
        elif self.fusion_strategy == 'hard_nms':
            fused = self._fusion_hard_nms(rf_bboxes, rf_scores, yolo_bboxes, yolo_scores)
        else:  # weighted_average
            fused = self._fusion_weighted_average(rf_bboxes, rf_scores, yolo_bboxes, yolo_scores)
        
        # Apply NMS filtering
        if apply_nms_filter:
            filtered = apply_nms(
                fused,
                nms_type='soft' if self.fusion_strategy == 'soft_nms' else 'hard',
                iou_threshold=self.iou_threshold,
                confidence_threshold=self.confidence_threshold,
                sigma=self.sigma
            )
            fused = filtered
        
        # Add metadata
        fused['counts'] = {
            'retinaface': len(rf_bboxes),
            'yolo': len(yolo_bboxes),
            'fused': len(fused['bboxes'])
        }
        fused['strategy'] = self.fusion_strategy
        
        return fused
    
    def _fusion_soft_nms(
        self,
        rf_bboxes: torch.Tensor,
        rf_scores: torch.Tensor,
        yolo_bboxes: torch.Tensor,
        yolo_scores: torch.Tensor
    ) -> Dict:
        """
        Fusion using Soft-NMS on weighted ensemble.
        Combines predictions and applies Soft-NMS.
        
        Args:
            rf_bboxes, rf_scores: RetinaFace detections
            yolo_bboxes, yolo_scores: YOLO detections
            
        Returns:
            Fused detections
        """
        # Weight scores
        if len(rf_scores) > 0:
            rf_scores = rf_scores * self.retinaface_weight
        if len(yolo_scores) > 0:
            yolo_scores = yolo_scores * self.yolo_weight
        
        # Combine detections
        if len(rf_bboxes) > 0 and len(yolo_bboxes) > 0:
            all_bboxes = torch.cat([rf_bboxes, yolo_bboxes], dim=0)
            all_scores = torch.cat([rf_scores, yolo_scores], dim=0)
        elif len(rf_bboxes) > 0:
            all_bboxes = rf_bboxes
            all_scores = rf_scores
        else:
            all_bboxes = yolo_bboxes
            all_scores = yolo_scores
        
        # Apply Soft-NMS
        bboxes, scores, _ = NMSProcessor.soft_nms(
            all_bboxes,
            all_scores,
            iou_threshold=self.iou_threshold,
            sigma=self.sigma,
            score_threshold=0.0  # Don't filter yet
        )
        
        return {
            'bboxes': bboxes,
            'scores': scores
        }
    
    def _fusion_hard_nms(
        self,
        rf_bboxes: torch.Tensor,
        rf_scores: torch.Tensor,
        yolo_bboxes: torch.Tensor,
        yolo_scores: torch.Tensor
    ) -> Dict:
        """
        Fusion using Hard-NMS on weighted ensemble.
        
        Args:
            rf_bboxes, rf_scores: RetinaFace detections
            yolo_bboxes, yolo_scores: YOLO detections
            
        Returns:
            Fused detections
        """
        # Weight scores
        if len(rf_scores) > 0:
            rf_scores = rf_scores * self.retinaface_weight
        if len(yolo_scores) > 0:
            yolo_scores = yolo_scores * self.yolo_weight
        
        # Combine detections
        if len(rf_bboxes) > 0 and len(yolo_bboxes) > 0:
            all_bboxes = torch.cat([rf_bboxes, yolo_bboxes], dim=0)
            all_scores = torch.cat([rf_scores, yolo_scores], dim=0)
        elif len(rf_bboxes) > 0:
            all_bboxes = rf_bboxes
            all_scores = rf_scores
        else:
            all_bboxes = yolo_bboxes
            all_scores = yolo_scores
        
        # Apply Hard-NMS
        keep_idx = NMSProcessor.hard_nms(
            all_bboxes,
            all_scores,
            iou_threshold=self.iou_threshold
        )
        
        return {
            'bboxes': all_bboxes[keep_idx],
            'scores': all_scores[keep_idx]
        }
    
    def _fusion_weighted_average(
        self,
        rf_bboxes: torch.Tensor,
        rf_scores: torch.Tensor,
        yolo_bboxes: torch.Tensor,
        yolo_scores: torch.Tensor
    ) -> Dict:
        """
        Fusion using weighted average of overlapping detections.
        Matches detections between models and averages overlapping boxes.
        
        Args:
            rf_bboxes, rf_scores: RetinaFace detections
            yolo_bboxes, yolo_scores: YOLO detections
            
        Returns:
            Fused detections
        """
        fused_bboxes = []
        fused_scores = []
        used_yolo = set()
        
        # Try to match RetinaFace detections with YOLO detections
        for i, rf_bbox in enumerate(rf_bboxes):
            best_match = -1
            best_iou = 0.3  # Minimum IoU threshold for matching
            
            for j, yolo_bbox in enumerate(yolo_bboxes):
                if j in used_yolo:
                    continue
                
                iou = self._compute_iou(rf_bbox, yolo_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_match = j
            
            if best_match >= 0:
                # Matched detection - average
                yolo_bbox = yolo_bboxes[best_match]
                avg_bbox = (rf_bbox + yolo_bbox) / 2
                avg_score = (rf_scores[i] * self.retinaface_weight + 
                            yolo_scores[best_match] * self.yolo_weight)
                
                fused_bboxes.append(avg_bbox)
                fused_scores.append(avg_score)
                used_yolo.add(best_match)
            else:
                # Unmatched RetinaFace detection
                fused_bboxes.append(rf_bbox)
                fused_scores.append(rf_scores[i] * self.retinaface_weight)
        
        # Add unmatched YOLO detections
        for j, yolo_bbox in enumerate(yolo_bboxes):
            if j not in used_yolo:
                fused_bboxes.append(yolo_bbox)
                fused_scores.append(yolo_scores[j] * self.yolo_weight)
        
        if len(fused_bboxes) == 0:
            return {
                'bboxes': torch.tensor([]),
                'scores': torch.tensor([])
            }
        
        return {
            'bboxes': torch.stack(fused_bboxes),
            'scores': torch.stack(fused_scores)
        }
    
    @staticmethod
    def _compute_iou(box1: torch.Tensor, box2: torch.Tensor) -> float:
        """Compute IoU between two boxes."""
        inter_x1 = max(box1[0].item(), box2[0].item())
        inter_y1 = max(box1[1].item(), box2[1].item())
        inter_x2 = min(box1[2].item(), box2[2].item())
        inter_y2 = min(box1[3].item(), box2[3].item())
        
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        
        box1_area = (box1[2].item() - box1[0].item()) * (box1[3].item() - box1[1].item())
        box2_area = (box2[2].item() - box2[0].item()) * (box2[3].item() - box2[1].item())
        union_area = box1_area + box2_area - inter_area
        
        iou = inter_area / (union_area + 1e-6)
        return iou
