import shutil
import tempfile
from datetime import datetime, UTC
from enum import StrEnum
from pathlib import Path
from typing import List
from uuid import uuid4

from fastapi import UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from storage.storage import upload_file, download_file
from .models import Job


class JobStatus(StrEnum):
    QUEUED = "queued"
    ASR_RUNNING = "asr_running"
    SUMMARY_RUNNING = "summary_running"
    NON_EXISTING = "non_existing"
    DONE = "done"
    FAILED = "failed"
    SMTP_FAILED = 'email_failed'

async def create_job(file: UploadFile, emails: List[str] | None, db_session: AsyncSession):
    job_id: str = str(uuid4())
    job_path: Path = Path(f'output/{job_id}')
    job_path.mkdir(parents=True, exist_ok=True)
    file_ext: str = Path(str(file.filename)).suffix.lower()
    source_path = job_path / f"source{file_ext}"
    job = Job(
        id = job_id,
        status = JobStatus.QUEUED,
        emails = emails or [],
        created_at = datetime.now(UTC),
        updated_at = datetime.now(UTC),
        artifacts = {
            "source": str(source_path),
        },
        error = None,
        warning = "Emails not found. Get you result via job id." if emails is None else None,
    )
    try:
        async with db_session.begin():
            db_session.add(job)
            with tempfile.NamedTemporaryFile(suffix=file_ext) as tmp:
                shutil.copyfileobj(file.file, tmp)
                tmp.flush()
                upload_file(tmp.name, job_path)
    except Exception:
        return {"status": JobStatus.FAILED, "error": "Internal server error. Job not create"}
    return {"status": JobStatus.QUEUED, "job_id": job_id}

async def get_status(job_id: str, db_session: AsyncSession):
    async with db_session.begin():
        stmt = select(Job).where(Job.id == job_id)
        result = await db_session.execute(stmt)
        job = result.scalar_one_or_none()
        if job is None:
            return {"job_id": job_id, "status": JobStatus.NON_EXISTING, "error": "Job not found"}
        else:
            return {"job_id": job_id, "status": job.status}

async def get_result(job_id: str, db_session: AsyncSession):
    async with db_session.begin():
        stmt = select(Job).where(Job.id == job_id)
        result = await db_session.execute(stmt)
        job = result.scalar_one_or_none()
        if job is None:
            return {"job_id": job_id, "status": JobStatus.NON_EXISTING, "error": "Job not found"}

    if job.status != "done":
        return {"job_id": job_id, "status": job.status}
    summary_path: Path = Path(job.artifacts['summary'])

    with tempfile.NamedTemporaryFile() as tmp:
        download_file(str(summary_path), tmp.name)
        summary = tmp.read()
    return {"job_id": job_id, "status": job.status, "summary": summary}


async def update_job(job_id: str, session: AsyncSession, **fields):
    async with session.begin():
        stmt = select(Job).where(Job.id == job_id)
        result = await session.execute(stmt)
        job = result.scalar_one_or_none()
        if job is None:
            raise ValueError("Job not found")
        for field, value in fields.items():
            setattr(job, field, value)
