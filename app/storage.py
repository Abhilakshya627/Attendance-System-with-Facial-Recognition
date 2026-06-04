from pathlib import Path
import os


BASE_DIR = Path(__file__).resolve().parent.parent
_default_data_dir = BASE_DIR / "data"
_configured_data_dir = os.getenv("APP_DATA_DIR")

if _configured_data_dir:
    data_dir = Path(_configured_data_dir)
    if not data_dir.is_absolute():
        data_dir = (BASE_DIR / data_dir).resolve()
else:
    data_dir = _default_data_dir

DATA_DIR = data_dir
DATABASE_PATH = DATA_DIR / "attendance.db"
STUDENT_IMAGES_DIR = DATA_DIR / "student_images"
GROUP_IMAGES_DIR = DATA_DIR / "group_images"
GROUP_FACES_DIR = DATA_DIR / "group_faces"
FACE_REVIEW_DIR = DATA_DIR / "session_face_reviews"

for directory in (DATA_DIR, STUDENT_IMAGES_DIR, GROUP_IMAGES_DIR, GROUP_FACES_DIR, FACE_REVIEW_DIR):
    directory.mkdir(parents=True, exist_ok=True)