# System Architecture

## Architecture Overview
The system follows a modular, pipeline-based architecture where each component is responsible for a specific task. This design ensures scalability, maintainability, and fault tolerance.

## Core Components

### 1. Image Ingestion Module
- Accepts classroom group images uploaded by teachers

### 2. Preprocessing Module
- Image normalization and resizing
- Noise and blur handling

### 3. Face Detection Engine
- Primary detector: RetinaFace
- Auxiliary detector: YOLO-based face detector for speed optimization

### 4. Detection Fusion Module
- Merges outputs from multiple detectors
- Applies soft filtering and validation

### 5. Face Cropping Module
- Extracts and normalizes face regions
- Stores cropped face images for downstream processing

## Design Principles
- Recall-first detection strategy
- Loose coupling between modules
- Extensible architecture for future face recognition and attendance logic
