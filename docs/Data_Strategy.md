# Data Strategy

## Dataset Sources

### Primary Face Detection Dataset
- **WIDER FACE (Hard subset)**
  - Contains diverse face sizes, poses, occlusions, and lighting conditions
  - Includes challenging scenarios such as small faces, partial occlusions, and crowded scenes
  - Serves as the primary supervised dataset for face detection in Phase 1

### Supplementary Crowd Images (Images Only)
- **Crowd and Non-Crowd Image Collections**
  - Contain real-world crowded and non-crowded scenes
  - Do not include bounding box annotations

> These images are used only for qualitative evaluation and inference testing.
> They are **not used for supervised training** due to the absence of reliable annotations.

---

## Data Usage Strategy
- **WIDER FACE** is used for:
  - Supervised training of face detection models
  - Learning facial geometry, scale variation, and pose diversity
- **Crowd images without annotations** are used for:
  - Visual validation of detection robustness
  - Stress-testing inference on dense classroom-like scenes
  - Demonstration and sanity-check purposes

The training pipeline is designed to remain extensible, allowing additional
annotated crowd datasets to be integrated in the future if available.

---

## Data Governance
- All datasets used are publicly available and widely adopted in academic research
- Dataset usage follows respective licensing and redistribution policies
- No personal, institutional, or student-specific data is included in training
- Raw datasets are preserved without modification for reproducibility

---

## Ethical and Privacy Considerations
- The current phase performs **face detection only**, without identity recognition
- No biometric embeddings or personal identifiers are stored
- Data usage is limited to research and educational purposes
- Privacy-aware and responsible AI practices are followed throughout development

---

## Update in Approach
The CrowdHuman dataset was initially considered to improve robustness in dense crowd scenarios. However, due to the unavailability of reliable and accessible annotated versions at the time of development, it is not included in supervised training for Phase 1.

WIDER FACE sufficiently covers challenging detection scenarios, and the system architecture allows future integration of additional annotated datasets when available.
