"""
WiderFace Dataset Preparation Script
This script prepares the WiderFace dataset for training, validation, and testing.
It processes the annotations and organizes the data for the face detection model.
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict
import json
import shutil


class WiderFacePreparation:
    """Class to prepare WiderFace dataset for training."""
    
    def __init__(self, data_root: str = None):
        """
        Initialize the WiderFace data preparation class.
        
        Args:
            data_root: Root directory of the data folder
        """
        if data_root is None:
            # Get the project root (2 levels up from src/Data_Prep)
            current_dir = Path(__file__).resolve().parent
            project_root = current_dir.parent.parent
            self.data_root = project_root / "data" / "WiderFace"
        else:
            self.data_root = Path(data_root)
        
        self.train_img_dir = self.data_root / "WIDER_train" / "WIDER_train" / "images"
        self.val_img_dir = self.data_root / "WIDER_val" / "WIDER_val" / "images"
        self.test_img_dir = self.data_root / "WIDER_test" / "WIDER_test" / "images"
        
        self.annotations_dir = self.data_root / "wider_face_annotations" / "wider_face_split"
        
        self.train_annot_file = self.annotations_dir / "wider_face_train_bbx_gt.txt"
        self.val_annot_file = self.annotations_dir / "wider_face_val_bbx_gt.txt"
        self.test_filelist = self.annotations_dir / "wider_face_test_filelist.txt"
        
        # Output directories
        self.output_dir = self.data_root / "processed"
        self.output_dir.mkdir(exist_ok=True)
        
    def parse_annotation_file(self, annot_file: Path, img_dir: Path) -> List[Dict]:
        """
        Parse the WiderFace annotation file.
        
        Format:
        - Line 1: Image file path
        - Line 2: Number of faces
        - Lines 3+: Face bounding boxes (x, y, w, h, blur, expression, illumination, invalid, occlusion, pose)
        
        Args:
            annot_file: Path to annotation file
            img_dir: Path to images directory
            
        Returns:
            List of dictionaries containing image paths and annotations
        """
        print(f"Parsing annotation file: {annot_file}")
        
        annotations = []
        
        with open(annot_file, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            # Get image path
            img_path = lines[i].strip()
            full_img_path = img_dir / img_path
            
            # Get number of faces
            i += 1
            if i >= len(lines):
                break
                
            # Try to parse as number of faces
            try:
                num_faces = int(lines[i].strip())
            except ValueError:
                # If it's not a number, it might be bbox data, skip this image
                print(f"Warning: Could not parse face count for {img_path}, skipping")
                i += 1
                continue
            
            # Get bounding boxes
            boxes = []
            i += 1
            for j in range(num_faces):
                if i >= len(lines):
                    break
                    
                line_data = lines[i].strip()
                
                # Skip empty lines
                if not line_data:
                    i += 1
                    continue
                
                # Check if this line is the next image path (shouldn't start with a digit for bbox)
                if not line_data[0].isdigit() and '/' in line_data:
                    # This is the next image, don't increment i
                    break
                
                try:
                    bbox_info = list(map(int, line_data.split()))
                except ValueError:
                    # Not a bbox line, probably next image
                    break
                
                if len(bbox_info) >= 4:
                    x, y, w, h = bbox_info[:4]
                    
                    # Filter out invalid boxes
                    if w > 0 and h > 0:
                        box_dict = {
                            'x': x,
                            'y': y,
                            'w': w,
                            'h': h,
                            'x2': x + w,
                            'y2': y + h
                        }
                        
                        # Add additional attributes if available
                        if len(bbox_info) >= 10:
                            box_dict.update({
                                'blur': bbox_info[4],
                                'expression': bbox_info[5],
                                'illumination': bbox_info[6],
                                'invalid': bbox_info[7],
                                'occlusion': bbox_info[8],
                                'pose': bbox_info[9]
                            })
                        
                        boxes.append(box_dict)
                
                i += 1
            
            # Only add if we found the image and have valid boxes
            if full_img_path.exists() and len(boxes) > 0:
                annotations.append({
                    'image_path': str(full_img_path),
                    'relative_path': img_path,
                    'num_faces': len(boxes),
                    'boxes': boxes
                })
            elif not full_img_path.exists():
                print(f"Warning: Image not found: {full_img_path}")
            
            # Move to next image (if we didn't already break out to next image)
            if i < len(lines) and not ('/' in lines[i].strip() and not lines[i].strip()[0].isdigit()):
                i += 1
        
        print(f"Parsed {len(annotations)} images with annotations")
        return annotations
    
    def parse_test_filelist(self, filelist: Path, img_dir: Path) -> List[Dict]:
        """
        Parse the test file list.
        
        Args:
            filelist: Path to test file list
            img_dir: Path to test images directory
            
        Returns:
            List of dictionaries containing test image paths
        """
        print(f"Parsing test file list: {filelist}")
        
        test_images = []
        
        with open(filelist, 'r') as f:
            for line in f:
                img_path = line.strip()
                full_img_path = img_dir / img_path
                
                if full_img_path.exists():
                    test_images.append({
                        'image_path': str(full_img_path),
                        'relative_path': img_path
                    })
                else:
                    print(f"Warning: Test image not found: {full_img_path}")
        
        print(f"Found {len(test_images)} test images")
        return test_images
    
    def visualize_annotations(self, annotations: List[Dict], num_samples: int = 5, 
                            output_dir: Path = None):
        """
        Visualize sample annotations.
        
        Args:
            annotations: List of annotation dictionaries
            num_samples: Number of samples to visualize
            output_dir: Directory to save visualized images
        """
        if output_dir is None:
            output_dir = self.output_dir / "visualizations"
        output_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"Visualizing {num_samples} samples...")
        
        for idx, annot in enumerate(annotations[:num_samples]):
            img_path = annot['image_path']
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"Failed to load image: {img_path}")
                continue
            
            # Draw bounding boxes
            for box in annot['boxes']:
                x, y, w, h = box['x'], box['y'], box['w'], box['h']
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Add labels if available
                if 'invalid' in box and box['invalid'] == 1:
                    cv2.putText(img, 'Invalid', (x, y - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
            # Add face count
            cv2.putText(img, f"Faces: {annot['num_faces']}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            # Save visualization
            output_path = output_dir / f"sample_{idx + 1}.jpg"
            cv2.imwrite(str(output_path), img)
            print(f"Saved visualization: {output_path}")
    
    def convert_to_csv(self, annotations: List[Dict], output_file: Path):
        """
        Convert annotations to CSV format.
        
        Args:
            annotations: List of annotation dictionaries
            output_file: Output CSV file path
        """
        print(f"Converting annotations to CSV: {output_file}")
        
        rows = []
        for annot in annotations:
            img_path = annot['image_path']
            
            for box in annot['boxes']:
                row = {
                    'image_path': img_path,
                    'relative_path': annot['relative_path'],
                    'x': box['x'],
                    'y': box['y'],
                    'w': box['w'],
                    'h': box['h'],
                    'x2': box['x2'],
                    'y2': box['y2']
                }
                
                # Add optional attributes
                for key in ['blur', 'expression', 'illumination', 'invalid', 'occlusion', 'pose']:
                    row[key] = box.get(key, -1)
                
                rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)
        print(f"Saved CSV with {len(rows)} annotations from {len(annotations)} images")
        
        return df
    
    def convert_to_json(self, annotations: List[Dict], output_file: Path):
        """
        Convert annotations to JSON format (COCO-like).
        
        Args:
            annotations: List of annotation dictionaries
            output_file: Output JSON file path
        """
        print(f"Converting annotations to JSON: {output_file}")
        
        coco_format = {
            'images': [],
            'annotations': [],
            'categories': [{'id': 1, 'name': 'face'}]
        }
        
        annotation_id = 1
        
        for img_id, annot in enumerate(annotations, 1):
            # Add image info
            img_path = Path(annot['image_path'])
            img = cv2.imread(str(img_path))
            
            if img is None:
                continue
            
            height, width = img.shape[:2]
            
            coco_format['images'].append({
                'id': img_id,
                'file_name': annot['relative_path'],
                'width': width,
                'height': height,
                'path': annot['image_path']
            })
            
            # Add annotations
            for box in annot['boxes']:
                annotation = {
                    'id': annotation_id,
                    'image_id': img_id,
                    'category_id': 1,
                    'bbox': [box['x'], box['y'], box['w'], box['h']],
                    'area': box['w'] * box['h'],
                    'iscrowd': 0
                }
                
                # Add optional attributes
                for key in ['blur', 'expression', 'illumination', 'invalid', 'occlusion', 'pose']:
                    if key in box:
                        annotation[key] = box[key]
                
                coco_format['annotations'].append(annotation)
                annotation_id += 1
        
        with open(output_file, 'w') as f:
            json.dump(coco_format, f, indent=2)
        
        print(f"Saved JSON with {len(coco_format['images'])} images and {len(coco_format['annotations'])} annotations")
    
    def get_dataset_statistics(self, annotations: List[Dict]) -> Dict:
        """
        Calculate statistics about the dataset.
        
        Args:
            annotations: List of annotation dictionaries
            
        Returns:
            Dictionary containing dataset statistics
        """
        total_images = len(annotations)
        total_faces = sum(annot['num_faces'] for annot in annotations)
        
        # Face count distribution
        face_counts = [annot['num_faces'] for annot in annotations]
        
        # Bounding box sizes
        box_areas = []
        box_widths = []
        box_heights = []
        
        for annot in annotations:
            for box in annot['boxes']:
                box_areas.append(box['w'] * box['h'])
                box_widths.append(box['w'])
                box_heights.append(box['h'])
        
        stats = {
            'total_images': total_images,
            'total_faces': total_faces,
            'avg_faces_per_image': total_faces / total_images if total_images > 0 else 0,
            'max_faces_per_image': max(face_counts) if face_counts else 0,
            'min_faces_per_image': min(face_counts) if face_counts else 0,
            'avg_box_area': np.mean(box_areas) if box_areas else 0,
            'avg_box_width': np.mean(box_widths) if box_widths else 0,
            'avg_box_height': np.mean(box_heights) if box_heights else 0,
            'min_box_width': min(box_widths) if box_widths else 0,
            'max_box_width': max(box_widths) if box_widths else 0,
            'min_box_height': min(box_heights) if box_heights else 0,
            'max_box_height': max(box_heights) if box_heights else 0
        }
        
        return stats
    
    def prepare_all(self, visualize: bool = True, num_vis_samples: int = 5):
        """
        Prepare all datasets (train, val, test).
        
        Args:
            visualize: Whether to visualize sample annotations
            num_vis_samples: Number of samples to visualize for each split
        """
        print("=" * 80)
        print("Starting WiderFace Dataset Preparation")
        print("=" * 80)
        
        # Prepare training set
        print("\n" + "=" * 80)
        print("Processing Training Set")
        print("=" * 80)
        train_annotations = self.parse_annotation_file(self.train_annot_file, self.train_img_dir)
        
        # Save train annotations
        train_csv = self.output_dir / "train_annotations.csv"
        train_json = self.output_dir / "train_annotations.json"
        self.convert_to_csv(train_annotations, train_csv)
        self.convert_to_json(train_annotations, train_json)
        
        # Get train statistics
        train_stats = self.get_dataset_statistics(train_annotations)
        print("\nTraining Set Statistics:")
        for key, value in train_stats.items():
            print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
        
        # Visualize train samples
        if visualize:
            self.visualize_annotations(train_annotations, num_vis_samples, 
                                      self.output_dir / "train_visualizations")
        
        # Prepare validation set
        print("\n" + "=" * 80)
        print("Processing Validation Set")
        print("=" * 80)
        val_annotations = self.parse_annotation_file(self.val_annot_file, self.val_img_dir)
        
        # Save val annotations
        val_csv = self.output_dir / "val_annotations.csv"
        val_json = self.output_dir / "val_annotations.json"
        self.convert_to_csv(val_annotations, val_csv)
        self.convert_to_json(val_annotations, val_json)
        
        # Get val statistics
        val_stats = self.get_dataset_statistics(val_annotations)
        print("\nValidation Set Statistics:")
        for key, value in val_stats.items():
            print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
        
        # Visualize val samples
        if visualize:
            self.visualize_annotations(val_annotations, num_vis_samples, 
                                      self.output_dir / "val_visualizations")
        
        # Prepare test set
        print("\n" + "=" * 80)
        print("Processing Test Set")
        print("=" * 80)
        test_images = self.parse_test_filelist(self.test_filelist, self.test_img_dir)
        
        # Save test file list
        test_json = self.output_dir / "test_images.json"
        with open(test_json, 'w') as f:
            json.dump(test_images, f, indent=2)
        print(f"Saved test images list: {test_json}")
        
        # Save overall statistics
        print("\n" + "=" * 80)
        print("Saving Overall Statistics")
        print("=" * 80)
        
        overall_stats = {
            'train': train_stats,
            'val': val_stats,
            'test': {'total_images': len(test_images)}
        }
        
        stats_file = self.output_dir / "dataset_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(overall_stats, f, indent=2)
        print(f"Saved statistics: {stats_file}")
        
        print("\n" + "=" * 80)
        print("WiderFace Dataset Preparation Complete!")
        print("=" * 80)
        print(f"\nOutput directory: {self.output_dir}")
        print("\nGenerated files:")
        print(f"  - train_annotations.csv")
        print(f"  - train_annotations.json")
        print(f"  - val_annotations.csv")
        print(f"  - val_annotations.json")
        print(f"  - test_images.json")
        print(f"  - dataset_statistics.json")
        if visualize:
            print(f"  - train_visualizations/ ({num_vis_samples} samples)")
            print(f"  - val_visualizations/ ({num_vis_samples} samples)")


def main():
    """Main function to run the data preparation."""
    # Initialize the data preparation class
    prep = WiderFacePreparation()
    
    # Prepare all datasets
    prep.prepare_all(visualize=True, num_vis_samples=5)


if __name__ == "__main__":
    main()
