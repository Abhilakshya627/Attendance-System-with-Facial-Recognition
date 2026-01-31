# Requirements Specification

## Functional Requirements (FR)
- **FR1:** The system shall accept a single group photograph as input.
- **FR2:** The system shall detect all visible human faces in the image.
- **FR3:** The system shall extract and store individual face images.
- **FR4:** The system shall support images containing more than 200 faces.
- **FR5:** The system shall handle low-resolution and blurred images.

## Non-Functional Requirements (NFR)
- **NFR1:** Face detection recall must be ≥ 99%.
- **NFR2:** False negatives are less acceptable than false positives.
- **NFR3:** The system should process an image within acceptable latency.
- **NFR4:** The system must be hardware-agnostic and scalable.
- **NFR5:** The system must be modular to allow future upgrades.

## Performance Requirements
- High recall under crowded and occluded conditions
- Stable performance under variable lighting and camera quality

## Reliability Requirements
- The system should include fallback mechanisms when detection confidence is low
- Re-processing must be possible without data loss
