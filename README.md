# Group Face Attendance System (Starter)

This is a FastAPI starter backend for your workflow:

1. Admin creates faculty and students.
2. Admin assigns students to a faculty.
3. Student logs in first time and uploads face photos from different angles.
4. Faculty starts class attendance session and uploads one or more group photos.
5. System auto-marks present students using face recognition.
6. Faculty manually corrects attendance if someone was missed.
7. Faculty finalizes attendance.
8. Student dashboard shows attended/absent counts.

## Tech Stack

- FastAPI
- SQLite (SQLAlchemy)
- JWT auth
- RetinaFace + DeepFace embeddings with OpenCV fallback for matching

## Project Structure

```text
app/
  auth.py
  database.py
  main.py
  models.py
  schemas.py
  routers/
    auth.py
    admin.py
    faculty.py
    student.py
  services/
    face_service.py
data/
requirements.txt
README.md
```

## Setup

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open Swagger UI at: `http://127.0.0.1:8000/docs`

## Docker

Build and run locally:

```bash
docker build -t attendance-system .
docker run --rm -p 7860:7860 -v attendance-data:/data -e APP_DATA_DIR=/data attendance-system
```

The app is exposed on port `7860` because that is the default Hugging Face Spaces port.

For persistence, mount a volume at `/data` and set `APP_DATA_DIR=/data` in the container environment. That keeps SQLite, uploaded photos, and face review files across restarts.

## Hugging Face Spaces

1. Create a new Space and choose the `Docker` SDK.
2. Push this repository to the Space, making sure the root `Dockerfile` is included.
3. Wait for the image build to complete. The first boot can take longer because DeepFace downloads its models.
4. Open the Space URL and sign in with the default admin account:
  - Username: `admin`
  - Password: `admin123`

If you want persistent storage on Hugging Face, add a Space volume and mount it at `/data`. The app will use `APP_DATA_DIR=/data` from the Dockerfile and keep `attendance.db`, student uploads, group photos, and review artifacts there.

## Default Admin

- Username: `admin`
- Password: `admin123`

Change this in `app/main.py` before production.

## Main APIs

### Auth

- `POST /auth/login`

### Admin

- `POST /admin/faculty`
- `POST /admin/student`
- `POST /admin/assign`
- `GET /admin/dashboard`

### Student

- `POST /student/register-photos` (upload 3+ files)
- `GET /student/dashboard`
- `GET /student/my-faculty`

### Faculty

- `POST /faculty/attendance/start`
- `POST /faculty/attendance/{session_id}/scan` (upload one or more group photos)
- `PATCH /faculty/attendance/{session_id}/manual`
- `GET /faculty/attendance/{session_id}`
- `POST /faculty/attendance/{session_id}/finalize`

## Notes

- For best accuracy, student registration photos should be clear and from multiple angles.
- Group photos should have visible faces with decent lighting.
- Matching confidence threshold is configured in `app/services/face_service.py`.
- Detection is handled by RetinaFace, and recognition is handled by DeepFace embeddings plus cosine similarity.
- If scan results look stale in the admin or faculty dashboard, refresh the page to reload the latest table state.
- Missing student image files are skipped automatically during scan instead of breaking the request.

### Accuracy Tuning

You can tune recognition behavior with environment variables before starting the app:

- `FACE_MODEL_NAME` (default: `Facenet512`)
- `FACE_SIMILARITY_THRESHOLD` (default depends on model)
  - `Facenet`: `0.60`
  - `Facenet512`: `0.68`
  - `ArcFace`: `0.68`
- `FACE_MIN_SIZE_PX` (default: `60`) minimum detected face size.
- `FACE_MIN_SHARPNESS` (default: `40.0`) blur filter using Laplacian variance.

Example (PowerShell):

```powershell
$env:FACE_MODEL_NAME="Facenet512"
$env:FACE_SIMILARITY_THRESHOLD="0.7"
$env:FACE_MIN_SIZE_PX="90"
$env:FACE_MIN_SHARPNESS="110"
uvicorn app.main:app --reload
```

## Production Improvements

- Add subject/class timetable entities.
- Add per-class section and semester mapping.
- Add image quality checks and anti-spoofing.
- Add frontend dashboards (React or Flutter).
- Move secrets to environment variables.
- Add migrations with Alembic.
