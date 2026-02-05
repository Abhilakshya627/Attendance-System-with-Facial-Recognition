"""
Face extraction and preprocessing module.
Crops detected face regions from images and saves them.
"""

import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json


class FaceExtractor:
    """
    Extracts face regions from images based on detection bboxes.
    Handles face cropping, resizing, and quality checks.
    """
    
    def __init__(
        self,
        face_size: int = 224,
        min_face_size: int = 20,
        padding_ratio: float = 0.1,
        quality_threshold: float = 0.3
    ):
        """
        Initialize face extractor.
        
        Args:
            face_size: Target size for extracted face images
            min_face_size: Minimum face width/height to keep
            padding_ratio: Padding around face region (ratio of face size)
            quality_threshold: Minimum face size ratio threshold
        """
        self.face_size = face_size
        self.min_face_size = min_face_size
        self.padding_ratio = padding_ratio
        self.quality_threshold = quality_threshold
    
    def extract_faces(
        self,
        image: np.ndarray,
        detections: Dict,
        image_id: str = None
    ) -> Dict:
        """
        Extract face regions from image.
        
        Args:
            image: Input image (BGR format)
            detections: Detection results with 'bboxes' and 'scores'
            image_id: Identifier for the image
            
        Returns:
            Dictionary with extracted faces and metadata
        """
        bboxes = detections.get('bboxes', torch.tensor([]))
        scores = detections.get('scores', torch.tensor([]))
        
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.cpu().numpy()
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        
        img_height, img_width = image.shape[:2]
        
        faces = []
        face_metadata = []
        
        for bbox_idx, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox.astype(int)
            score = scores[bbox_idx] if len(scores) > bbox_idx else 1.0
            
            # Validate and adjust bbox
            x1, y1, x2, y2 = self._validate_bbox(x1, y1, x2, y2, img_width, img_height)
            
            face_width = x2 - x1
            face_height = y2 - y1
            
            # Check minimum size
            if min(face_width, face_height) < self.min_face_size:
                continue
            
            # Apply padding
            x1_padded, y1_padded, x2_padded, y2_padded = self._apply_padding(
                x1, y1, x2, y2, img_width, img_height
            )
            
            # Crop face region
            face_region = image[y1_padded:y2_padded, x1_padded:x2_padded]
            
            # Resize to target size
            face_resized = cv2.resize(face_region, (self.face_size, self.face_size))
            
            # Calculate quality score
            quality_score = self._calculate_quality(
                face_width, face_height, score
            )
            
            faces.append(face_resized)
            
            # Store metadata
            face_metadata.append({
                'bbox': bbox.tolist(),
                'bbox_padded': [x1_padded, y1_padded, x2_padded, y2_padded],
                'score': float(score),
                'quality_score': float(quality_score),
                'face_width': face_width,
                'face_height': face_height,
                'index': bbox_idx
            })
        
        return {
            'faces': faces,
            'metadata': face_metadata,
            'image_id': image_id,
            'total_faces': len(faces)
        }
    
    def save_faces(
        self,
        faces: List[np.ndarray],
        metadata: List[Dict],
        output_dir: str,
        image_name: str,
        save_format: str = 'jpg'
    ) -> Dict:
        """
        Save extracted face images to disk.
        
        Args:
            faces: List of face images
            metadata: List of face metadata
            output_dir: Output directory for faces
            image_name: Name of the source image (without extension)
            save_format: Image format ('jpg', 'png')
            
        Returns:
            Dictionary with save paths and statistics
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_faces = []
        failed_faces = []
        
        for idx, (face, meta) in enumerate(zip(faces, metadata)):
            try:
                # Create face filename
                face_filename = f"{image_name}_face_{idx:04d}.{save_format}"
                face_path = output_path / face_filename
                
                # Save face image
                cv2.imwrite(str(face_path), face)
                
                # Update metadata with save path
                meta['save_path'] = str(face_path)
                meta['filename'] = face_filename
                
                saved_faces.append({
                    'path': str(face_path),
                    'index': idx,
                    'quality_score': meta['quality_score']
                })
                
            except Exception as e:
                failed_faces.append({
                    'index': idx,
                    'error': str(e)
                })
        
        # Save metadata JSON
        metadata_path = output_path / f"{image_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return {
            'total_saved': len(saved_faces),
            'total_failed': len(failed_faces),
            'saved_faces': saved_faces,
            'failed_faces': failed_faces,
            'metadata_path': str(metadata_path)
        }
    
    def batch_extract_and_save(
        self,
        image_paths: List[str],
        detections_list: List[Dict],
        output_dir: str,
        save_format: str = 'jpg'
    ) -> Dict:
        """
        Extract and save faces from multiple images.
        
        Args:
            image_paths: List of input image paths
            detections_list: List of detection results
            output_dir: Output directory for faces
            save_format: Image format
            
        Returns:
            Batch processing summary
        """
        summary = {
            'total_images': len(image_paths),
            'total_faces': 0,
            'total_saved': 0,
            'total_failed': 0,
            'images': []
        }
        
        for img_path, detections in zip(image_paths, detections_list):
            try:
                # Read image
                image = cv2.imread(img_path)
                if image is None:
                    summary['images'].append({
                        'path': img_path,
                        'error': 'Failed to read image'
                    })
                    continue
                
                # Extract faces
                image_name = Path(img_path).stem
                extract_result = self.extract_faces(image, detections, image_name)
                
                # Save faces
                save_result = self.save_faces(
                    extract_result['faces'],
                    extract_result['metadata'],
                    output_dir,
                    image_name,
                    save_format
                )
                
                # Update summary
                summary['total_faces'] += extract_result['total_faces']
                summary['total_saved'] += save_result['total_saved']
                summary['total_failed'] += save_result['total_failed']
                
                summary['images'].append({
                    'path': img_path,
                    'faces_extracted': extract_result['total_faces'],
                    'faces_saved': save_result['total_saved'],
                    'save_result': save_result
                })
                
            except Exception as e:
                summary['images'].append({
                    'path': img_path,
                    'error': str(e)
                })
        
        return summary
    
    def _validate_bbox(
        self,
        x1: int, y1: int, x2: int, y2: int,
        img_width: int, img_height: int
    ) -> Tuple[int, int, int, int]:
        """
        Validate and clip bbox to image boundaries.
        
        Args:
            x1, y1, x2, y2: Bounding box coordinates
            img_width, img_height: Image dimensions
            
        Returns:
            Clipped bbox coordinates
        """
        x1 = max(0, min(x1, img_width - 1))
        y1 = max(0, min(y1, img_height - 1))
        x2 = max(x1 + 1, min(x2, img_width))
        y2 = max(y1 + 1, min(y2, img_height))
        
        return x1, y1, x2, y2
    
    def _apply_padding(
        self,
        x1: int, y1: int, x2: int, y2: int,
        img_width: int, img_height: int
    ) -> Tuple[int, int, int, int]:
        """
        Apply padding around face bbox.
        
        Args:
            x1, y1, x2, y2: Bounding box coordinates
            img_width, img_height: Image dimensions
            
        Returns:
            Padded bbox coordinates
        """
        face_width = x2 - x1
        face_height = y2 - y1
        
        pad_x = int(face_width * self.padding_ratio)
        pad_y = int(face_height * self.padding_ratio)
        
        x1_padded = max(0, x1 - pad_x)
        y1_padded = max(0, y1 - pad_y)
        x2_padded = min(img_width, x2 + pad_x)
        y2_padded = min(img_height, y2 + pad_y)
        
        return x1_padded, y1_padded, x2_padded, y2_padded
    
    def _calculate_quality(
        self,
        face_width: int,
        face_height: int,
        detection_score: float
    ) -> float:
        """
        Calculate quality score for extracted face.
        Considers size and detection confidence.
        
        Args:
            face_width: Face width in pixels
            face_height: Face height in pixels
            detection_score: Detection confidence score
            
        Returns:
            Quality score [0, 1]
        """
        # Size score (normalize by face_size)
        size_score = min(1.0, max(face_width, face_height) / self.face_size)
        
        # Combined quality
        quality = size_score * 0.5 + detection_score * 0.5
        
        return quality
    
    @staticmethod
    def visualize_extraction(
        image: np.ndarray,
        detections: Dict,
        output_path: str,
        draw_metadata: bool = True
    ):
        """
        Visualize extracted faces with bboxes.
        
        Args:
            image: Input image
            detections: Detection results
            output_path: Path to save visualization
            draw_metadata: Whether to draw scores
        """
        img_viz = image.copy()
        
        bboxes = detections.get('bboxes', torch.tensor([]))
        scores = detections.get('scores', torch.tensor([]))
        
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.cpu().numpy()
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        
        for bbox, score in zip(bboxes, scores):
            x1, y1, x2, y2 = bbox.astype(int)
            
            # Draw bbox
            cv2.rectangle(img_viz, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw score
            if draw_metadata:
                text = f"{score:.2f}"
                cv2.putText(
                    img_viz, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )
        
        # Add statistics
        cv2.putText(
            img_viz, f"Detections: {len(bboxes)}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
        
        cv2.imwrite(output_path, img_viz)
