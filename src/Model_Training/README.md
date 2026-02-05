# Model Training Module

This module contains the training infrastructure for face detection models using RetinaFace and YOLO.

## Overview

The training pipeline consists of:

1. **RetinaFace Trainer** - Primary detector with anchor-dense architecture
2. **YOLO Trainer** - Auxiliary detector for speed optimization
3. **Training Orchestrator** - Coordinates training of both models
4. **Utilities** - Data loaders, loss functions, and metrics

## Directory Structure

```
Model_Training/
├── RetinaFace/
│   ├── trainer.py          # RetinaFace model and trainer
│   └── __init__.py
├── YOLO/
│   ├── trainer.py          # YOLO model and trainer
│   └── __init__.py
├── utils/
│   ├── data_loader.py      # Dataset class and data loading utilities
│   ├── losses_metrics.py   # Loss functions and evaluation metrics
│   ├── helpers.py          # Training utilities and helpers
│   └── __init__.py
├── configs/
│   └── training_config.yaml  # Training configuration
├── train.py                # Main training orchestrator
└── __init__.py
```

## Quick Start

### 1. Run Training

```bash
python src/Model_Training/train.py \
    --config src/Model_Training/configs/training_config.yaml \
    --output-dir ./outputs
```

### 2. Configuration

Edit `configs/training_config.yaml` to customize:
- Data paths and batch sizes
- Model parameters
- Training hyperparameters
- Device settings
- Loss function weights

## Key Components

### RetinaFace Trainer (`RetinaFace/trainer.py`)

- **RetinaFaceBackbone**: ResNet-50 based feature extractor
- **FPN**: Feature Pyramid Network for multi-scale features
- **RetinaFaceHead**: Detection head with classification and bbox regression
- **RetinaFaceTrainer**: Training loop with validation and checkpointing

Features:
- Anchor-dense architecture for small face detection
- Multi-scale inference support
- Focal loss for handling class imbalance
- IoU loss for bbox regression

### YOLO Trainer (`YOLO/trainer.py`)

- **YOLOBackbone**: Darknet-like backbone
- **YOLOHead**: YOLO detection head
- **YOLODetector**: Complete YOLO model
- **YOLOTrainer**: Training loop

Features:
- Faster inference compared to RetinaFace
- Grid-based detection
- Objectness + class + bbox predictions

### Data Loading (`utils/data_loader.py`)

- **FaceDetectionDataset**: Loads WIDER FACE annotations in COCO format
- **Augmentations**: Multi-scale augmentation pipeline using Albumentations
- **Data Loaders**: Batched data loading with variable-length bbox support

Features:
- Support for variable number of faces per image
- Multiple data augmentation techniques
- COCO format annotation support

### Loss Functions (`utils/losses_metrics.py`)

- **FocalLoss**: Addresses class imbalance in dense object detection
- **IoULoss**: IoU-based bbox regression loss
- **CombinedLoss**: Combines classification and regression losses
- **DetectionMetrics**: Precision, recall, F1 score calculation

### Training Utilities (`utils/helpers.py`)

- Checkpoint saving/loading
- Learning rate scheduling with warmup
- Model parameter counting
- Configuration management
- Logging setup

## Training Configuration

### Key Parameters

```yaml
# Data
batch_size: 16
img_size: 640
augmentation: true

# Training
num_epochs: 100
learning_rate: 0.001
optimizer: "adamw"
scheduler: "cosine"

# RetinaFace
focal_alpha: 0.25  # Focal loss alpha
focal_gamma: 2.0   # Focal loss gamma
lambda_reg: 1.0    # Weight for regression loss

# YOLO
num_anchors: 3
lambda_reg: 1.0

# Detection Fusion
iou_threshold: 0.5
confidence_threshold: 0.3
```

## Loss Functions

### Focal Loss
For addressing class imbalance in dense object detection:
- Used for classification
- Focuses on hard examples
- Alpha controls background/foreground balance
- Gamma controls focusing parameter

### IoU Loss
For bounding box regression:
- Better than L2 loss for object detection
- Considers IoU between predicted and target boxes
- Scale-invariant loss

## Training Features

1. **Multi-scale Detection**
   - RetinaFace uses FPN for multi-scale features
   - YOLO uses single-scale output
   - Both support detection of small and large faces

2. **Augmentation Pipeline**
   - Geometric: rotation, flipping
   - Photometric: brightness, contrast, blur
   - Compression artifacts simulation

3. **High Recall Strategy**
   - Low confidence threshold
   - Loose NMS parameters
   - Focal loss with appropriate alpha/gamma

4. **Model Checkpointing**
   - Automatic checkpoint saving
   - Best model selection based on validation loss
   - Resume training from checkpoint

## Expected Results

After training on WIDER FACE dataset:

**RetinaFace:**
- ~11.36 avg faces per training image
- ~55,645 total training faces
- Designed for high recall on small faces

**YOLO:**
- Faster inference
- Simpler architecture
- Good for real-time applications

## Next Steps

After training:
1. **Detection Fusion** - Combine RetinaFace and YOLO predictions
2. **Inference Pipeline** - Real-time face detection on classroom images
3. **Face Extraction** - Crop and save detected faces
4. **Face Recognition** - (Future phase) Match faces to student database

## Dependencies

- torch >= 1.9.0
- torchvision >= 0.10.0
- albumentations >= 1.0.0
- numpy >= 1.19.0
- PyYAML >= 5.4.0
- opencv-python >= 4.5.0

## Troubleshooting

### Out of Memory
- Reduce `batch_size` in config
- Reduce `img_size` (e.g., 416 or 512 instead of 640)
- Use gradient accumulation

### Slow Data Loading
- Increase `num_workers` in config
- Ensure images are on fast disk (SSD)
- Consider pre-processing images

### Poor Validation Metrics
- Check data loading (verify annotations)
- Increase `num_epochs`
- Adjust learning rate
- Verify augmentation parameters

## References

- RetinaFace: [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
- YOLO: [You Only Look Once](https://arxiv.org/abs/1612.08242)
- WIDER FACE Dataset: [A Face Detection Benchmark](http://shuoyang1213.me/WIDERFACE/)
