"""The API. Start it with: uvicorn src.api:app --port 8000"""

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import UnidentifiedImageError

from src import database, retrain
from src.insights import dataset_insights
from src.model import MODEL_PATH, evaluate_model, load_saved_model
from src.prediction import predict_image
from src.preprocessing import (CLASS_NAMES, DATA_DIR, IMAGE_TYPES, ROOT_DIR, UPLOAD_DIR,
                               load_test_data)

UI_DIR = ROOT_DIR / "ui"
INSIGHTS_FILE = DATA_DIR / "insights.json"
START_TIME = time.time()

app = FastAPI(title="Vehicle Damage Claim Triage")
app.mount("/static", StaticFiles(directory=UI_DIR), name="static")
database.create_tables()
database.fail_unfinished_runs()


def to_iso(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat(timespec="seconds")


def save_upload(file, label):
    """Save one photo and write it in the database."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in IMAGE_TYPES:
        return False
    folder = UPLOAD_DIR / label
    folder.mkdir(parents=True, exist_ok=True)
    file_path = folder / f"{uuid.uuid4().hex}{suffix}"
    file_path.write_bytes(file.file.read())
    database.add_upload(file_path, label)
    return True


@app.get("/")
def home():
    return FileResponse(UI_DIR / "index.html")


@app.get("/health")
def health():
    """Show if the model is up."""
    return {
        "status": "up",
        "uptime_seconds": int(time.time() - START_TIME),
        "started_at": to_iso(START_TIME),
        "model_last_trained": to_iso(MODEL_PATH.stat().st_mtime),
        "new_uploads_waiting": len(database.get_new_uploads()),
    }


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    """Predict the damage level of one photo."""
    try:
        return predict_image(file.file.read())
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="The file is not a valid image.")


@app.get("/insights")
def insights():
    """Give the numbers for the charts."""
    if not INSIGHTS_FILE.exists():
        INSIGHTS_FILE.write_text(json.dumps(dataset_insights()))
    return json.loads(INSIGHTS_FILE.read_text())


@app.post("/upload")
def upload(background_tasks: BackgroundTasks, label: str = Form(...), files: list[UploadFile] = File(...)):
    """Save many photos of one label for retraining."""
    if label not in CLASS_NAMES:
        raise HTTPException(status_code=400, detail=f"Label must be one of {CLASS_NAMES}.")
    saved = sum(save_upload(file, label) for file in files)

    auto_started = retrain.should_auto_retrain()
    if auto_started:
        background_tasks.add_task(retrain.run_retraining)
    return {
        "saved": saved,
        "skipped": len(files) - saved,
        "new_uploads_waiting": len(database.get_new_uploads()),
        "auto_retrain_started": auto_started,
    }


@app.post("/retrain")
def trigger_retrain(background_tasks: BackgroundTasks):
    """Start retraining."""
    if retrain.is_running():
        raise HTTPException(status_code=409, detail="Retraining is already running.")
    if not database.get_new_uploads():
        raise HTTPException(status_code=400, detail="No new images. Upload some first.")
    background_tasks.add_task(retrain.run_retraining)
    return {"message": "Retraining started."}


@app.get("/retrain/status")
def retrain_status():
    return database.get_last_run() or {"status": "never run"}


@app.get("/evaluate")
def evaluate():
    """Test the model on the test photos."""
    return evaluate_model(load_saved_model(), load_test_data())