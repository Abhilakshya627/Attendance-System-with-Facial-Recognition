# Data Strategy

## Dataset Sources

### Primary Face Detection Dataset
- **WIDER FACE (Hard subset)**
  - Contains diverse face sizes, poses, occlusions, and lighting conditions

### Crowd Robustness Dataset
- **CrowdHuman**
  - Used for dense crowd scenarios via head annotations
  - Improves robustness in overlapping and occluded environments

## Data Usage Strategy
- WIDER FACE is used to learn facial geometry and scale variation
- CrowdHuman is used to improve detection robustness in high-density scenes

## Data Governance
- All datasets are publicly available and used according to license terms
- No personal or institution-specific data is used in training

## Ethical and Privacy Considerations
- Face detection does not perform identity recognition
- No biometric information is stored during the detection phase
- Privacy-aware and responsible AI practices are followed
