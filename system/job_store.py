import json
import shutil
from datetime import datetime, UTC
from pathlib import Path
from typing import List
from uuid import uuid4

from fastapi import UploadFile


def create_job(file: UploadFile, emails: List[str] | None):
    job_id: str = str(uuid4())
    job_path: Path = Path(f'output/{job_id}')
    job_path.mkdir(parents=True, exist_ok=True)
    file_ext: str = Path(str(file.filename)).suffix.lower()
    source_path = job_path / f"source{file_ext}"
    job = {
        "job_id": job_id,
        "status": "queued",
        "emails": emails,
        "created_at": datetime.now(UTC).isoformat(),
        "updated_at": datetime.now(UTC).isoformat(),
        "error": None,
        "artifacts": [str(source_path)]
    }
    if emails is None:
        job["warning"] = "Emails not found. Get you result via job id."
    try:
        with open(source_path, mode='wb') as f:
            shutil.copyfileobj(file.file, f)
        with open(job_path / "job.json", mode='w', encoding='UTF-8') as f:
            json.dump(job, f)
    except PermissionError as err:
        return {"status": "failed", "error": err.strerror}
    except OSError as err:
        return {"status": "failed", "error": err.strerror}
    return {"status": "queued", "job_id": job_id}

def get_status(job_id: str):
    job_path: Path = Path(f"output/{job_id}")
    try:
        with open(job_path / "job.json", mode='r', encoding='UTF-8') as f:
            status = json.load(f)["status"]
        return {"job_id": job_id ,"status": status}
    except FileNotFoundError:
        return {"job_id": job_id, "status": "failed", "error": "Job not found"}

def get_result(job_id: str):
    job_path: Path = Path(f"output/{job_id}")
    try:
        with open(job_path / "job.json", mode='r', encoding='UTF-8') as f:
            job = json.load(f)
    except FileNotFoundError:
        return {'job_id': job_id, "status": "failed", "error": "Job not found"}
    if job["status"] != "done":
        return {"job_id": job_id, "status": job["status"]}
    transcript_path: Path = job_path / "transcript.txt"
    summary_path: Path = job_path / "summary.txt"
    with transcript_path.open(mode='r', encoding='UTF-8') as f:
        transcript = f.read()
    with summary_path.open(mode='r', encoding='UTF-8') as f:
        summary = f.read()
    return {"job_id": job_id, "status": job["status"], "transcript": transcript, "summary": summary}


def process_job(job_id: str):
    job_path: Path = Path(f"output/{job_id}")
    with open(job_path / "job.json", mode='r', encoding='UTF-8') as f:
        job = json.load(f)
    job["status"] = "done"
    with open(job_path / "job.json", mode='w', encoding='UTF-8') as f:
        json.dump(job, f)