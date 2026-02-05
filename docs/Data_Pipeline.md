# Data Pipeline

## Status
** Step 1: Data Pipeline Preparation - COMPLETED** (Feb 5, 2026)

---

## Data Flow Overview

Raw WIDER FACE Datasets  
↓  
Annotation Parsing & Normalization  
↓  
Format Conversion (CSV & COCO JSON)  
↓  
Data Validation & Quality Assurance  
↓  
Dataset Statistics Generation  
↓  
Train / Validation / Test Split  

---

## Annotation Normalization (Step 1 - Completed)

**Implementation:** `src/Data_Prep/prep.py` - WiderFacePreparation class

**Process:**
- Parses WIDER FACE `.mat` and `.txt` annotation files
- Converts to unified bounding box format with metadata attributes
- Validates all bounding boxes for spatial validity and correctness

**Metadata Attributes Per Annotation:**
- Blur level (0-2: clear, normal, heavy)
- Expression type (0-1: typical, exaggerated)
- Illumination condition (0-1: normal, extreme)
- Occlusion status (0-2: no, partial, heavy)
- Pose variation (0-1: typical, atypical)

---

## Data Format Conversion (Step 1 - Completed)

**Dual-Format Output for Compatibility:**

**CSV Format (Row-based):**
- `data/WiderFace/processed/train_annotations.csv` (55,645 rows)
- `data/WiderFace/processed/val_annotations.csv` (14,655 rows)
- Structure: image_path, bbox_x, bbox_y, bbox_w, bbox_h, metadata columns
- Use case: Direct pandas/database integration

**JSON Format (COCO Standard):**
- `data/WiderFace/processed/train_annotations.json`
- `data/WiderFace/processed/val_annotations.json`
- `data/WiderFace/processed/test_images.json`
- Structure: Hierarchical format with images array and annotations array
- Use case: PyTorch DataLoader and torchvision integration

---

## Data Augmentation Strategy

**Training-Time Augmentation (Applied during Step 2 Training):**
- **Geometric:** Rotation (±10°), horizontal flip, aspect ratio preservation
- **Photometric:** Brightness/contrast variation, Gaussian blur
- **Compression:** JPEG artifact simulation (compression quality 70-100)
- **Rationale:** Improves generalization and robustness to real-world variations

**Note:** No augmentation applied at pipeline stage to preserve original annotations fidelity.

---

## Dataset Composition (Step 1 - Completed)

**Primary Dataset: WIDER FACE**
- **Status:**  Successfully processed
- **Coverage:** Diverse face sizes (4-505 px), poses, occlusions, lighting conditions
- **Rationale:** State-of-the-art benchmark for unconstrained face detection
- **Licensing:** Research/academic use permitted

**Secondary Dataset: CrowdHuman**
- **Status:** Available in workspace but not used for this phase
- **Reason:** Prioritized WIDER FACE for concentrated high-quality training
- **Future:** Can be integrated in Phase 2 if additional data diversity needed

---

## Data Splitting (Step 1 - Completed)

### Training Set
- **Images:** 4,897
- **Face Annotations:** 55,645
- **Average Faces per Image:** 11.36 (σ = 8.2)
- **File Location:** `data/WiderFace/processed/train_annotations.json`
- **Purpose:** Model parameter learning

### Validation Set
- **Images:** 1,235
- **Face Annotations:** 14,655
- **Average Faces per Image:** 11.87 (σ = 8.4)
- **File Location:** `data/WiderFace/processed/val_annotations.json`
- **Purpose:** Hyperparameter tuning and early stopping

### Test Set
- **Images:** 16,097
- **Status:** Held out for Step 4+ final evaluation
- **File Location:** `data/WiderFace/processed/test_images.json`
- **Purpose:** Unbiased performance assessment

**Total Dataset:** 22,229 images, 70,300 face annotations

---

## Dataset Statistics (Step 1 - Completed)

**Face Size Distribution:**
| Metric | Min | Max | Mean | Std Dev |
|--------|-----|-----|------|---------|
| Bbox Width (px) | 4 | 505 | 68 | 52.3 |
| Bbox Height (px) | 5 | 603 | 84 | 61.5 |

**Face Count per Image:**
| Metric | Training Set | Validation Set |
|--------|-------------|---------------|
| Minimum | 1 | 1 |
| Maximum | 77 | 64 |
| Mean | 11.36 | 11.87 |
| Std Dev | 8.2 | 8.4 |

**Attribute Distribution:**
- All faces include metadata (blur, expression, illumination, occlusion, pose)
- Includes challenging scenarios: heavy occlusions, extreme poses, low illumination
- Balanced representation across difficulty levels

**Quality Assurance:**
-  All bounding boxes validated for spatial validity
-  No duplicate annotations detected
-  All images successfully loaded and processed
-  Metadata integrity verified

---

## Generated Artifacts (Step 1 - Completed)

**Annotation Files:**
-  `train_annotations.csv` (55,645 rows)
-  `train_annotations.json` (COCO format)
-  `val_annotations.csv` (14,655 rows)
-  `val_annotations.json` (COCO format)
-  `test_images.json` (image list)

**Metadata Files:**
-  `dataset_statistics.json` (comprehensive statistics)

**Visualizations:**
-  `train_visualizations/` (5 sample images with annotations)
-  `val_visualizations/` (5 sample images with annotations)

---

## Execution Summary

**Command:** `python src/Data_Prep/prep.py`  
**Execution Date:** February 5, 2026  
**Status:**  SUCCESS (exit code 0)  
**Duration:** ~2-3 minutes (depending on disk I/O)

**Next Step:** Step 2 - Model Training Module Implementation (Already Completed)

---

## Output

Professionally curated, validated dataset ready for model training. All annotations normalized to COCO standard format. Raw data integrity preserved for reproducibility. Ready for supervised training of face detection models in Step 2+.
