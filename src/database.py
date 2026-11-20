"""A small database to save uploads and training runs."""

import sqlite3
from datetime import datetime, timezone

from src.preprocessing import DATA_DIR

DB_PATH = DATA_DIR / "app.db"

CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS uploads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT NOT NULL,
    label TEXT NOT NULL,
    uploaded_at TEXT NOT NULL,
    used_for_training INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS training_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL,
    image_count INTEGER DEFAULT 0,
    message TEXT
);
"""


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def create_tables():
    with connect() as db:
        db.executescript(CREATE_TABLES)


def add_upload(file_path, label):
    with connect() as db:
        db.execute(
            "INSERT INTO uploads (file_path, label, uploaded_at) VALUES (?, ?, ?)",
            (str(file_path), label, now()),
        )


def get_new_uploads():
    """Photos not used for training yet."""
    with connect() as db:
        rows = db.execute("SELECT * FROM uploads WHERE used_for_training = 0").fetchall()
    return [dict(row) for row in rows]


def mark_uploads_used(upload_ids):
    with connect() as db:
        db.executemany(
            "UPDATE uploads SET used_for_training = 1 WHERE id = ?",
            [(upload_id,) for upload_id in upload_ids],
        )


def start_run():
    with connect() as db:
        cursor = db.execute(
            "INSERT INTO training_runs (started_at, status) VALUES (?, 'running')", (now(),)
        )
        return cursor.lastrowid


def finish_run(run_id, status, image_count, message):
    with connect() as db:
        db.execute(
            "UPDATE training_runs SET finished_at = ?, status = ?, image_count = ?, message = ? WHERE id = ?",
            (now(), status, image_count, message, run_id),
        )


def fail_unfinished_runs():
    """If the app stopped during training, mark that run as failed."""
    with connect() as db:
        db.execute(
            "UPDATE training_runs SET status = 'failed', message = 'App restarted during the run.' "
            "WHERE status = 'running'"
        )


def get_last_run():
    with connect() as db:
        row = db.execute("SELECT * FROM training_runs ORDER BY id DESC LIMIT 1").fetchone()
    return dict(row) if row else None