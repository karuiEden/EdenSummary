import asyncio
import json
import logging
import shutil
import tempfile
from datetime import datetime, UTC
from enum import StrEnum
from pathlib import Path
from typing import List
from uuid import uuid4

from fastapi import UploadFile
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from eden_summary.storage import upload_file, download_file
from .models import Job, JobFieldEdit

logger = logging.getLogger(__name__)

# Summary fields graded by the Tier-2 judge — mirrors faithfulness._FIELD_ORDER.
# Kept local to avoid a core -> quality import cycle (quality imports from core).
# Only these get edit rows: each must have a quality_eval.field_scores entry to
# join at calibration time; the title is a label, not a judged claim.
_EDIT_FIELDS = ('risks', 'decisions', 'action_items', 'tldr')


class JobStatus(StrEnum):
    QUEUED = "queued"
    ASR_RUNNING = "asr_running"
    SUMMARY_RUNNING = "summary_running"
    NON_EXISTING = "non_existing"
    DONE = "done"
    FAILED = "failed"
    EMAIL_FAILED = 'email_failed'

async def create_job(file: UploadFile, emails: List[str] | None, db_session: AsyncSession):
    job_id: str = str(uuid4())
    file_ext: str = Path(str(file.filename)).suffix.lower()
    source_path = f"{job_id}/source{file_ext}"
    job = Job(
        id = job_id,
        status = JobStatus.QUEUED,
        emails = emails or [],
        created_at = datetime.now(UTC),
        updated_at = datetime.now(UTC),
        artifacts = {
            "source": source_path,
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
            await asyncio.to_thread(upload_file, tmp.name, source_path)
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
    summary_path: str = job.artifacts['summary']
    with tempfile.NamedTemporaryFile() as tmp:
        try:
            await asyncio.to_thread(download_file, summary_path, tmp.name)
            with open(tmp.name, mode='r', encoding='UTF-8') as file:
                summary = file.read()
        except Exception:
            return {"job_id": job_id, "status": JobStatus.FAILED, "error": "Internal server error"}
    # Also surface the structured summary so a client can edit it field-by-field
    # and submit corrections via PATCH /result (Q3). Best-effort: older jobs have
    # no summary_json artifact, and a broken one must not break the text result.
    structured = await _load_structured_summary(job.artifacts)
    return {
        "job_id": job_id,
        "status": job.status,
        "summary": summary,
        "structured": structured,
        # Quality signals (advisory). quality_flags (Tier-1 inline) is present as
        # soon as the job is done. quality_eval (Tier-2 faithfulness) and summq_eval
        # (SummQ consistency) are produced by post-terminal async tasks, so they read
        # null on the first fetch(es) and populate seconds-to-minutes later — re-poll.
        "quality_flags": job.quality_flags,
        "quality_eval": job.quality_eval,
        "summq_eval": job.summq_eval,
    }


async def _load_structured_summary(artifacts: dict) -> dict | None:
    """Download and parse the stored summary_json.json artifact (asdict(Summary)),
    or None when it is absent/unreadable."""
    key = artifacts.get('summary_json')
    if not key:
        return None
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = f'{tmp_dir}/summary.json'
            await asyncio.to_thread(download_file, key, path)
            with open(path, encoding='UTF-8') as file:
                data = json.load(file)
        return data if isinstance(data, dict) else None
    except Exception:
        logger.warning("Could not load structured summary for editing", exc_info=True)
        return None


async def update_job(job_id: str, session: AsyncSession, **fields):
    async with session.begin():
        stmt = select(Job).where(Job.id == job_id)
        result = await session.execute(stmt)
        job = result.scalar_one_or_none()
        if job is None:
            raise ValueError("Job not found")
        for field, value in fields.items():
            setattr(job, field, value)


def _canon(value: object) -> str:
    """Canonical JSON for stable equality: dict key order doesn't matter, list
    order does (a reorder counts as an edit — acceptable for a v1 edit signal),
    and surface whitespace inside strings still matters."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _diff_fields(stored: dict, submitted: dict) -> list[dict]:
    """Per-field structural diff over the judged summary fields. Returns one row
    per field with `edited` = (submitted differs from stored). Pure — no IO — so
    the calibration label logic is unit-testable without a DB or S3."""
    rows: list[dict] = []
    for field in _EDIT_FIELDS:
        original = stored.get(field, [])
        corrected = submitted.get(field, [])
        rows.append({
            'field': field,
            'edited': _canon(original) != _canon(corrected),
            'original': original,
            'corrected': corrected,
        })
    return rows


async def record_result_edits(job_id: str, submitted: dict, db_session: AsyncSession):
    """Record a user's correction of a DONE job's summary as Q3 calibration
    labels. A PATCH is one review event: every judged field is written (changed →
    edited=True, untouched → genuine negative). Upserts on (job_id, field) so a
    repeat PATCH overwrites — last edit wins, no double-counting. Does NOT touch
    the stored summary_json (it is the immutable diff baseline the judge scored)."""
    async with db_session.begin():
        result = await db_session.execute(select(Job).where(Job.id == job_id))
        job = result.scalar_one_or_none()
        if job is None:
            return {"job_id": job_id, "status": JobStatus.NON_EXISTING, "error": "Job not found"}
        status = job.status
        artifacts = dict(job.artifacts)
    if status != JobStatus.DONE:
        return {"job_id": job_id, "status": status, "error": "Result is not available for editing"}
    if 'summary_json' not in artifacts:
        return {"job_id": job_id, "status": status, "error": "Structured summary not available for editing"}

    stored = await _load_structured_summary(artifacts)
    if stored is None:
        return {"job_id": job_id, "status": status, "error": "Structured summary not available for editing"}

    rows = _diff_fields(stored, submitted)
    now = datetime.now(UTC)
    async with db_session.begin():
        for row in rows:
            stmt = pg_insert(JobFieldEdit).values(
                job_id=job_id,
                field=row['field'],
                edited=row['edited'],
                original=row['original'],
                corrected=row['corrected'],
                created_at=now,
            ).on_conflict_do_update(
                index_elements=['job_id', 'field'],
                set_={
                    'edited': row['edited'],
                    'original': row['original'],
                    'corrected': row['corrected'],
                    'created_at': now,
                },
            )
            await db_session.execute(stmt)
    edited_fields = [row['field'] for row in rows if row['edited']]
    return {"job_id": job_id, "status": status, "recorded": len(rows), "edited_fields": edited_fields}
