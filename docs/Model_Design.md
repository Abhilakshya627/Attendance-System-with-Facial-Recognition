# Model Design

## Model Selection Rationale

### RetinaFace (Primary Detector)
- Anchor-dense architecture suitable for small faces
- High recall in crowded scenes
- Landmark-assisted supervision improves detection quality

### YOLO-Based Face Detector (Auxiliary)
- Fast region proposal generation
- Used to reduce computational overhead
- Not used as a standalone detector

## Design Philosophy
The system prioritizes recall over precision to avoid missing any student. False positives are acceptable and can be handled during later recognition stages.

## Detection Strategy
- Multi-scale inference
- Low confidence thresholds
- Soft Non-Maximum Suppression (Soft-NMS)

## Output Format
- Bounding box coordinates
- Confidence scores
- Cropped face images
