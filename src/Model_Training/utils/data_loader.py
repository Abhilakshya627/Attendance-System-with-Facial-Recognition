"""
Data loader utilities for face detection training.
Handles loading and preprocessing of WIDER FACE and CrowdHuman datasets.
"""

import torch
import cv2
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


class FaceDetectionDataset(Dataset):
    """
    Dataset class for face detection training.
    Loads images and annotations from WIDER FACE dataset.
    """
    
    def __init__(
        self,
        annotation_json: str,
        img_dir: str,
        transforms=None,
        img_size: int = 640
    ):
        """
        Initialize the face detection dataset.
        
        Args:
            annotation_json: Path to JSON file with COCO-format annotations
            img_dir: Base directory for images
            transforms: Albumentations transforms to apply
            img_size: Target image size for resizing
        """
        self.img_size = img_size
        self.img_dir = Path(img_dir)
        
        # Load annotations
        with open(annotation_json, 'r') as f:
            self.coco_data = json.load(f)
        
        # Create image id to annotations mapping
        self.image_to_annotations = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.image_to_annotations:
                self.image_to_annotations[img_id] = []
            self.image_to_annotations[img_id].append(ann)
        
        self.images = self.coco_data['images']
        
        # Default transforms if none provided
        if transforms is None:
            self.transforms = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
        else:
            self.transforms = transforms
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary with image tensor, bboxes, and metadata
        """
        image_info = self.images[idx]
        image_id = image_info['id']
        
        # Load image
        img_path = Path(image_info['path'])
        image = cv2.imread(str(img_path))
        if image is None:
            # Return empty sample if image not found
            return self._get_empty_sample()
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        
        # Get annotations for this image
        annotations = self.image_to_annotations.get(image_id, [])
        
        # Extract bboxes
        bboxes = []
        for ann in annotations:
            bbox = ann['bbox']  # [x, y, w, h]
            # Convert to pascal_voc format [x_min, y_min, x_max, y_max]
            x, y, w, h = bbox
            x_max = min(x + w, width)
            y_max = min(y + h, height)
            x = max(0, x)
            y = max(0, y)
            
            if x_max > x and y_max > y and w > 0 and h > 0:
                bboxes.append([x, y, x_max, y_max])
        
        # Apply transforms
        class_labels = [1] * len(bboxes)  # All are faces (class 1)
        
        if len(bboxes) > 0:
            transformed = self.transforms(
                image=image,
                bboxes=bboxes,
                class_labels=class_labels
            )
            image = transformed['image']
            bboxes = torch.tensor(transformed['bboxes'], dtype=torch.float32)
        else:
            transformed = self.transforms(image=image, bboxes=[], class_labels=[])
            image = transformed['image']
            bboxes = torch.zeros((0, 4), dtype=torch.float32)
        
        return {
            'image': image,
            'bboxes': bboxes,
            'num_faces': len(bboxes),
            'image_id': image_id,
            'img_path': str(img_path),
            'orig_size': torch.tensor([width, height], dtype=torch.float32)
        }
    
    def _get_empty_sample(self) -> Dict:
        """Return an empty sample when image cannot be loaded."""
        return {
            'image': torch.zeros((3, self.img_size, self.img_size)),
            'bboxes': torch.zeros((0, 4), dtype=torch.float32),
            'num_faces': 0,
            'image_id': -1,
            'img_path': '',
            'orig_size': torch.tensor([self.img_size, self.img_size])
        }


def create_augmentations(img_size: int = 640, augment: bool = True) -> A.Compose:
    """
    Create albumentations augmentation pipeline.
    
    Args:
        img_size: Target image size
        augment: Whether to apply augmentations
        
    Returns:
        Albumentations Compose object
    """
    if augment:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.5),
            A.GaussNoise(p=0.3),
            A.GaussianBlur(blur_limit=3, p=0.3),
            A.JPEG(quality_lower=85, quality_upper=100, p=0.2),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))


def create_data_loaders(
    train_annotation: str,
    val_annotation: str,
    img_dir: str,
    batch_size: int = 16,
    num_workers: int = 4,
    img_size: int = 640,
    augment: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.
    
    Args:
        train_annotation: Path to training annotation JSON
        val_annotation: Path to validation annotation JSON
        img_dir: Base directory for images
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        img_size: Target image size
        augment: Whether to apply augmentations to training data
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create augmentations
    train_transforms = create_augmentations(img_size, augment=augment)
    val_transforms = create_augmentations(img_size, augment=False)
    
    # Create datasets
    train_dataset = FaceDetectionDataset(
        annotation_json=train_annotation,
        img_dir=img_dir,
        transforms=train_transforms,
        img_size=img_size
    )
    
    val_dataset = FaceDetectionDataset(
        annotation_json=val_annotation,
        img_dir=img_dir,
        transforms=val_transforms,
        img_size=img_size
    )
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for batching samples with variable number of bboxes.
    
    Args:
        batch: List of samples from the dataset
        
    Returns:
        Batched samples
    """
    images = []
    all_bboxes = []
    all_labels = []
    max_boxes = 0
    
    for sample in batch:
        images.append(sample['image'])
        num_boxes = sample['bboxes'].shape[0]
        max_boxes = max(max_boxes, num_boxes)
    
    images = torch.stack(images)
    
    # Pad bboxes to max length in batch
    bboxes_padded = []
    for sample in batch:
        bboxes = sample['bboxes']
        num_boxes = bboxes.shape[0]
        if num_boxes < max_boxes:
            padding = torch.zeros((max_boxes - num_boxes, 4))
            bboxes = torch.cat([bboxes, padding], dim=0)
        bboxes_padded.append(bboxes)
    
    bboxes_batch = torch.stack(bboxes_padded)
    
    # Create masks for valid bboxes
    masks = []
    for sample in batch:
        mask = torch.zeros(max_boxes, dtype=torch.bool)
        mask[:sample['bboxes'].shape[0]] = True
        masks.append(mask)
    masks_batch = torch.stack(masks)
    
    return {
        'images': images,
        'bboxes': bboxes_batch,
        'masks': masks_batch,
        'num_faces': torch.tensor([s['num_faces'] for s in batch])
    }
