# Inference Module - Face Detection & Extraction

This module handles end-to-end inference for face detection and extraction from classroom images.

## Overview

The inference module processes classroom group photos to:
1. Detect all visible faces using RetinaFace and YOLO
2. Fuse predictions from both models using weighted ensemble
3. Apply Soft-NMS for final detection refinement
4. Extract and save individual face crops
5. Generate detection visualizations and metadata

## Directory Structure

```
Inference/
├── utils/
│   ├── nms.py              # NMS algorithms (Hard-NMS, Soft-NMS)
│   ├── fusion.py           # Detection fusion and ensemble
│   ├── face_extractor.py   # Face cropping and extraction
│   └── __init__.py
├── configs/
│   └── inference_config.yaml  # Inference configuration
├── inference.py            # Main inference pipeline
├── README.md               # This file
└── __init__.py
```

## Quick Start

### 1. Basic Inference

```python
from src.Inference import InferencePipeline

# Initialize pipeline
pipeline = InferencePipeline(
    config_path='src/Inference/configs/inference_config.yaml',
    device='cuda'
)

# Load trained models
pipeline.load_models(
    retinaface_checkpoint='outputs/retinaface/retinaface_best.pt',
    yolo_checkpoint='outputs/yolo/yolo_best.pt'
)

# Process single image
result = pipeline.process_image(
    'path/to/classroom/image.jpg',
    save_faces=True,
    output_dir='./detection_output'
)

print(f"Detected {result['total_detections']} faces")
print(f"Extracted {result['total_faces_extracted']} face crops")
```

### 2. Batch Processing

```python
image_paths = [
    'path/to/image1.jpg',
    'path/to/image2.jpg',
    'path/to/image3.jpg'
]

summary = pipeline.process_batch(
    image_paths=image_paths,
    output_dir='./batch_output',
    save_faces=True,
    visualize=True
)

print(f"Processed {summary['successful']} images successfully")
print(f"Extracted {summary['total_faces']} total faces")
```

## Key Components

### NMS Algorithms (`utils/nms.py`)

**Hard-NMS:**
- Removes boxes with IoU > threshold
- Standard NMS approach
- Fast but may remove valid detections

**Soft-NMS:**
- Reduces scores of nearby boxes instead of removing
- Better for crowded scenes
- Preserves more detections (higher recall)
- Formula: `score = score * exp(-(IoU²) / σ)`

**Weighted NMS:**
- Combines predictions from multiple detectors
- Applies weights to detector outputs
- Then applies Hard-NMS

**Usage:**
```python
from src.Inference.utils import NMSProcessor

# Hard-NMS
keep_idx = NMSProcessor.hard_nms(bboxes, scores, iou_threshold=0.4)

# Soft-NMS
final_bboxes, final_scores, keep_idx = NMSProcessor.soft_nms(
    bboxes, scores, 
    iou_threshold=0.4,
    sigma=0.5
)

# Weighted NMS (ensemble)
final_bboxes, final_scores = NMSProcessor.weighted_nms(
    bboxes_list=[rf_bboxes, yolo_bboxes],
    scores_list=[rf_scores, yolo_scores],
    weights=[0.6, 0.4],
    iou_threshold=0.4
)
```

### Detection Fusion (`utils/fusion.py`)

**Fusion Strategies:**
1. **Soft-NMS Strategy:** Best for high recall
   - Combines weighted predictions
   - Applies Soft-NMS for smooth score decay
   - Recommended for crowded classrooms

2. **Hard-NMS Strategy:** Balanced approach
   - Combines weighted predictions
   - Applies Hard-NMS for cleaner output
   - Good trade-off between recall and precision

3. **Weighted Average Strategy:** Finest control
   - Matches overlapping detections
   - Averages overlapping boxes
   - Keeps non-overlapping detections

**Usage:**
```python
from src.Inference.utils import DetectionFusion

fusion = DetectionFusion(
    retinaface_weight=0.6,
    yolo_weight=0.4,
    iou_threshold=0.5,
    confidence_threshold=0.3,
    fusion_strategy='soft_nms',
    sigma=0.5
)

fused = fusion.fuse(
    retinaface_detections={'bboxes': rf_boxes, 'scores': rf_scores},
    yolo_detections={'bboxes': yolo_boxes, 'scores': yolo_scores}
)

print(f"RetinaFace: {fused['counts']['retinaface']} faces")
print(f"YOLO: {fused['counts']['yolo']} faces")
print(f"Fused: {fused['counts']['fused']} faces")
```

### Face Extraction (`utils/face_extractor.py`)

**Features:**
- Crops face regions from detected bboxes
- Applies padding for context
- Resizes to consistent size (224x224)
- Calculates quality scores
- Saves metadata

**Usage:**
```python
from src.Inference.utils import FaceExtractor

extractor = FaceExtractor(
    face_size=224,
    min_face_size=20,
    padding_ratio=0.1,
    quality_threshold=0.3
)

# Extract faces
result = extractor.extract_faces(image_cv, detections)
print(f"Extracted {result['total_faces']} faces")

# Save faces
save_result = extractor.save_faces(
    faces=result['faces'],
    metadata=result['metadata'],
    output_dir='./faces',
    image_name='classroom_01'
)

print(f"Saved {save_result['total_saved']} faces")
print(f"Failed: {save_result['total_failed']}")
```

### Main Inference Pipeline (`inference.py`)

**Workflow:**
```
Input Image
    ↓
Load to Tensor
    ↓
RetinaFace Detection
    ↓
YOLO Detection
    ↓
Detection Fusion
    ↓
Face Extraction
    ↓
Save Faces & Metadata
    ↓
Generate Visualization
    ↓
Output Results
```

## Configuration

Edit `configs/inference_config.yaml` to customize:

```yaml
detection:
  confidence_threshold: 0.3

fusion:
  strategy: "soft_nms"
  retinaface_weight: 0.6
  yolo_weight: 0.4
  iou_threshold: 0.5
  sigma: 0.5

face_extraction:
  face_size: 224
  min_face_size: 20
  padding_ratio: 0.1

output:
  save_faces: true
  save_visualizations: true
```

## Output Structure

```
output_dir/
├── faces/
│   ├── image_01_face_0000.jpg
│   ├── image_01_face_0001.jpg
│   ├── image_01_metadata.json
│   └── ...
├── visualizations/
│   ├── image_01_detections.jpg
│   └── ...
└── processing_summary.json
```

**Metadata JSON Example:**
```json
{
  "bbox": [100, 150, 250, 400],
  "bbox_padded": [90, 140, 260, 410],
  "score": 0.95,
  "quality_score": 0.88,
  "face_width": 150,
  "face_height": 250,
  "index": 0,
  "filename": "image_01_face_0000.jpg"
}
```

## Advanced Features

### High Recall Strategy
For classroom attendance, high recall is critical (missing a student is worse than a false positive):

```yaml
detection:
  confidence_threshold: 0.3    # Low threshold

fusion:
  strategy: "soft_nms"          # Preserve detections
  retinaface_weight: 0.6
  yolo_weight: 0.4
  iou_threshold: 0.5
  sigma: 0.5                    # Higher sigma = more preservation
```

### Batch Processing with Progress

```python
image_paths = [f'images/class_{i}.jpg' for i in range(100)]

summary = pipeline.process_batch(
    image_paths=image_paths,
    output_dir='./attendance_output',
    save_faces=True,
    visualize=False  # Disable for speed
)

# Results
print(f"Total images: {summary['total_images']}")
print(f"Successfully processed: {summary['successful']}")
print(f"Total faces extracted: {summary['total_faces']}")
```

### Quality Filtering

Extracted faces have quality scores based on size and detection confidence:

```python
# Filter by quality
high_quality_faces = [
    face for face, meta in zip(faces, metadata)
    if meta['quality_score'] > 0.7
]
```

## Performance Considerations

### Memory Usage
- Soft-NMS uses more memory than Hard-NMS
- Reduce batch size if out of memory
- Use CPU for very large images

### Speed
- YOLO is faster than RetinaFace
- Use only YOLO for real-time requirements
- Disable visualizations for batch processing

### Quality
- Soft-NMS gives higher recall (better for attendance)
- Hard-NMS gives higher precision
- Weighted average for fine-grained control

## Troubleshooting

### No faces detected
- Lower `confidence_threshold` (e.g., 0.2)
- Check model checkpoints are loaded
- Verify image format and size

### Too many false positives
- Increase `confidence_threshold` (e.g., 0.5)
- Use Hard-NMS instead of Soft-NMS
- Reduce YOLO weight in fusion

### Memory errors
- Reduce batch size
- Process images one at a time
- Use CPU instead of GPU

### Poor face quality
- Increase `min_face_size`
- Increase `padding_ratio` for more context
- Check `face_size` parameter

## Next Steps

After inference:
1. **Face Recognition** - Match extracted faces to student database
2. **Attendance Marking** - Record attendance based on matches
3. **Quality Assessment** - Filter low-quality detections
4. **Database Integration** - Store results for attendance records

## References

- Hard-NMS: Classic NMS algorithm
- Soft-NMS: [Improving Object Detection With One Line of Code](https://arxiv.org/abs/1704.04503)
- Ensemble Detection: [Fusion methods for object detection](https://arxiv.org/abs/2009.12243)
