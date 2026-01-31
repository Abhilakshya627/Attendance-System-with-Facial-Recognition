# Smart Attendance System (Phase 1: Face Detection)

## Overview
This project aims to build a **Smart Attendance System** that automatically marks student attendance using a **single group photograph** taken in a classroom. The primary motivation is to reduce time and errors associated with manual attendance, especially in large or combined classes where student strength can exceed 100–200.

The project is being developed in **phases**.  
The current phase focuses **exclusively on face detection and face extraction** from group images.

---

## Phase 1: Face Detection & Extraction (Current Scope)

### Objective
- Detect **all visible faces** from a single classroom group image
- Handle **crowded scenes**, occlusions, and small faces
- Prioritize **high recall** (missing a face is unacceptable)
- Extract and store individual face crops for later processing

> ⚠️ Face recognition and attendance marking are **not included** in this phase.

---

## Why This Project?
Traditional attendance systems:
- Consume classroom time
- Are error-prone
- Do not scale well for large classes

This system aims to:
- Save time for teachers
- Scale to large classrooms
- Work with images captured from **personal mobile phones**
- Provide a foundation for automated attendance

---

## Technical Approach (Phase 1)

- **Primary Face Detector:** RetinaFace  
- **Auxiliary Detector (Speed Optimization):** YOLO-based face detector  
- **Strategy:** Recall-first detection with multi-scale inference  
- **Datasets Used:**
  - WIDER FACE (Hard subset)
  - CrowdHuman (head annotations adapted for faces)

---

## Repository Structure

```text
docs/        → Project documentation (Phase 1)
src/         → Source code (to be added)
experiments/ → Experiment logs and evaluation (future)
