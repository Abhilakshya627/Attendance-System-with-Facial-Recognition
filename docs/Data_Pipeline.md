# Data Pipeline

## Data Flow Overview

Raw Datasets  
↓  
Annotation Normalization  
↓  
Data Augmentation  
↓  
Dataset Mixing  
↓  
Train / Validation / Test Split  

## Annotation Normalization
- Convert all annotations into a unified bounding box format
- CrowdHuman head annotations are converted into approximate face regions

## Data Augmentation
- Gaussian blur
- JPEG compression artifacts
- Brightness and contrast variations
- Random downscaling and upscaling

## Dataset Mixing Strategy
- 70% WIDER FACE (Hard subset)
- 30% CrowdHuman

## Data Splitting
- Training set: 70%
- Validation set: 20%
- Test set: 10%

## Output
- Cleaned, augmented, and standardized dataset ready for model training
