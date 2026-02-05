"""
Main inference pipeline for face detection and extraction.
Orchestrates end-to-end processing of classroom images.
"""

import torch
import cv2
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import logging
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.fusion import DetectionFusion
from utils.face_extractor import FaceExtractor
from utils.nms import apply_nms


class InferencePipeline:
    """
    End-to-end inference pipeline for face detection.
    Processes classroom images, detects faces, and extracts them.
    """
    
    def __init__(
        self,
        config_path: str,
        device: str = 'cuda'
    ):
        """
        Initialize inference pipeline.
        
        Args:
            config_path: Path to inference configuration
            device: Device to use ('cuda' or 'cpu')
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        
        # Setup logging
        self.logger = self._setup_logging()
        self.logger.info(f"Initialized inference pipeline on {self.device}")
        
        # Initialize fusion
        fusion_config = self.config.get('fusion', {})
        self.fusion = DetectionFusion(
            retinaface_weight=fusion_config.get('retinaface_weight', 0.6),
            yolo_weight=fusion_config.get('yolo_weight', 0.4),
            iou_threshold=fusion_config.get('iou_threshold', 0.5),
            confidence_threshold=fusion_config.get('confidence_threshold', 0.3),
            fusion_strategy=fusion_config.get('strategy', 'soft_nms'),
            sigma=fusion_config.get('sigma', 0.5)
        )
        
        # Initialize face extractor
        extractor_config = self.config.get('face_extraction', {})
        self.face_extractor = FaceExtractor(
            face_size=extractor_config.get('face_size', 224),
            min_face_size=extractor_config.get('min_face_size', 20),
            padding_ratio=extractor_config.get('padding_ratio', 0.1),
            quality_threshold=extractor_config.get('quality_threshold', 0.3)
        )
        
        # Models will be loaded separately
        self.retinaface_model = None
        self.yolo_model = None
    
    def load_models(
        self,
        retinaface_checkpoint: Optional[str] = None,
        yolo_checkpoint: Optional[str] = None
    ):
        """
        Load trained models.
        
        Args:
            retinaface_checkpoint: Path to RetinaFace checkpoint
            yolo_checkpoint: Path to YOLO checkpoint
        """
        # Import models dynamically to avoid hard dependencies
        if retinaface_checkpoint:
            try:
                from Model_Training.RetinaFace.trainer import RetinaFace
                self.retinaface_model = RetinaFace()
                checkpoint = torch.load(retinaface_checkpoint, map_location=self.device)
                self.retinaface_model.load_state_dict(checkpoint['model_state_dict'])
                self.retinaface_model = self.retinaface_model.to(self.device).eval()
                self.logger.info(f"Loaded RetinaFace from {retinaface_checkpoint}")
            except Exception as e:
                self.logger.error(f"Failed to load RetinaFace: {str(e)}")
        
        if yolo_checkpoint:
            try:
                from Model_Training.YOLO.trainer import YOLODetector
                self.yolo_model = YOLODetector()
                checkpoint = torch.load(yolo_checkpoint, map_location=self.device)
                self.yolo_model.load_state_dict(checkpoint['model_state_dict'])
                self.yolo_model = self.yolo_model.to(self.device).eval()
                self.logger.info(f"Loaded YOLO from {yolo_checkpoint}")
            except Exception as e:
                self.logger.error(f"Failed to load YOLO: {str(e)}")
    
    def detect_faces(
        self,
        image: torch.Tensor
    ) -> Dict:
        """
        Run face detection on image.
        
        Args:
            image: Input image tensor [3, H, W]
            
        Returns:
            Dictionary with fused detections
        """
        retinaface_detections = {'bboxes': torch.tensor([]), 'scores': torch.tensor([])}
        yolo_detections = {'bboxes': torch.tensor([]), 'scores': torch.tensor([])}
        
        # Run RetinaFace if available
        if self.retinaface_model is not None:
            with torch.no_grad():
                cls_preds, bbox_preds = self.retinaface_model(image.unsqueeze(0))
                
                # Process predictions (simplified)
                # In practice, need to decode anchors and apply post-processing
                scores = torch.sigmoid(cls_preds[0, :, 1])  # Face class score
                conf_mask = scores > self.config.get('detection', {}).get('confidence_threshold', 0.3)
                
                if conf_mask.sum() > 0:
                    retinaface_detections = {
                        'bboxes': bbox_preds[0, conf_mask],
                        'scores': scores[conf_mask]
                    }
        
        # Run YOLO if available
        if self.yolo_model is not None:
            with torch.no_grad():
                objectness, class_probs, bbox_preds = self.yolo_model(image.unsqueeze(0))
                
                # Process predictions
                objectness_scores = torch.sigmoid(objectness[0, :, 0])
                conf_mask = objectness_scores > self.config.get('detection', {}).get('confidence_threshold', 0.3)
                
                if conf_mask.sum() > 0:
                    yolo_detections = {
                        'bboxes': bbox_preds[0, conf_mask],
                        'scores': objectness_scores[conf_mask]
                    }
        
        # Fuse detections
        fused = self.fusion.fuse(retinaface_detections, yolo_detections)
        
        return fused
    
    def process_image(
        self,
        image_path: str,
        save_faces: bool = True,
        output_dir: Optional[str] = None,
        visualize: bool = True
    ) -> Dict:
        """
        Process single image for face detection and extraction.
        
        Args:
            image_path: Path to input image
            save_faces: Whether to save extracted faces
            output_dir: Directory to save outputs
            visualize: Whether to create visualization
            
        Returns:
            Processing result dictionary
        """
        try:
            # Read image
            image_cv = cv2.imread(image_path)
            if image_cv is None:
                return {'error': f'Failed to read image: {image_path}'}
            
            # Convert to tensor
            image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
            image_tensor = torch.from_numpy(image_rgb).float() / 255.0
            image_tensor = image_tensor.permute(2, 0, 1).to(self.device)
            
            # Detect faces
            detections = self.detect_faces(image_tensor)
            
            # Extract faces
            image_name = Path(image_path).stem
            extract_result = self.face_extractor.extract_faces(
                image_cv, detections, image_name
            )
            
            result = {
                'image_path': image_path,
                'image_name': image_name,
                'total_detections': len(detections.get('bboxes', [])),
                'total_faces_extracted': extract_result['total_faces'],
                'detections': {
                    'bboxes': detections.get('bboxes'),
                    'scores': detections.get('scores'),
                    'counts': detections.get('counts', {})
                }
            }
            
            # Save faces if requested
            if save_faces and output_dir:
                output_path = Path(output_dir) / 'faces'
                save_result = self.face_extractor.save_faces(
                    extract_result['faces'],
                    extract_result['metadata'],
                    str(output_path),
                    image_name
                )
                result['save_result'] = save_result
            
            # Create visualization if requested
            if visualize and output_dir:
                viz_path = Path(output_dir) / 'visualizations'
                viz_path.mkdir(parents=True, exist_ok=True)
                viz_file = viz_path / f"{image_name}_detections.jpg"
                FaceExtractor.visualize_extraction(image_cv, detections, str(viz_file))
                result['visualization_path'] = str(viz_file)
            
            result['status'] = 'success'
            
        except Exception as e:
            result = {
                'image_path': image_path,
                'error': str(e),
                'status': 'failed'
            }
        
        return result
    
    def process_batch(
        self,
        image_paths: List[str],
        output_dir: str,
        save_faces: bool = True,
        visualize: bool = True
    ) -> Dict:
        """
        Process multiple images.
        
        Args:
            image_paths: List of image paths
            output_dir: Output directory
            save_faces: Whether to save faces
            visualize: Whether to create visualizations
            
        Returns:
            Batch processing summary
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        summary = {
            'total_images': len(image_paths),
            'successful': 0,
            'failed': 0,
            'total_faces': 0,
            'results': []
        }
        
        for idx, img_path in enumerate(image_paths):
            self.logger.info(f"Processing {idx + 1}/{len(image_paths)}: {img_path}")
            
            result = self.process_image(
                img_path,
                save_faces=save_faces,
                output_dir=str(output_path),
                visualize=visualize
            )
            
            summary['results'].append(result)
            
            if result['status'] == 'success':
                summary['successful'] += 1
                summary['total_faces'] += result.get('total_faces_extracted', 0)
            else:
                summary['failed'] += 1
        
        # Save summary
        summary_path = output_path / 'processing_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"\nBatch processing completed:")
        self.logger.info(f"  Total: {summary['total_images']}")
        self.logger.info(f"  Successful: {summary['successful']}")
        self.logger.info(f"  Failed: {summary['failed']}")
        self.logger.info(f"  Total faces extracted: {summary['total_faces']}")
        
        return summary
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        # Console handler
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
