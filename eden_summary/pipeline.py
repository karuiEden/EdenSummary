import logging
import tempfile
from pathlib import Path
from typing import List

from faster_whisper.transcribe import Segment
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from eden_summary.core.models import Job
from eden_summary.core import JobStatus, update_job
from eden_summary.core import get_app_cfg
from eden_summary.email_service import send_email
from eden_summary.summarize import build_summary, Summary
from eden_summary.transcribe import chunk_segments, transcribe
from eden_summary.transcribe import convert_to_wav
from eden_summary.storage.storage import upload_file
from eden_summary.storage.storage import download_file

logger = logging.getLogger(__name__)

async def process_job(job_id: str, db_session: AsyncSession):
    logger.info('Job started', extra={"job_id": job_id})
    async with db_session.begin():
        stmt = select(Job).where(Job.id == job_id)
        result = await db_session.execute(stmt)
        job = result.scalar_one_or_none()
        if job is None:
            raise ValueError("Job not found")
    try:
        logger.info('Audio Preprocessing started', extra={"job_id": job_id})
        with tempfile.NamedTemporaryFile() as tmp_source:
            with tempfile.NamedTemporaryFile() as tmp_prep:
                download_file(job.artifacts['source'], tmp_source.name)
                convert_to_wav(Path(tmp_source.name), Path(tmp_prep.name))
                upload_file(tmp_prep.name, f'{job_id}/preprocessed.wav')
        logger.info('Audio Preprocessing done', extra={"job_id": job_id})
    except Exception:
        logger.exception("Audio conversion failed")
        await update_job(job_id, db_session,status=JobStatus.FAILED, error="Audio conversion failed")
        return
    job.artifacts['preprocessed'] = f'{job_id}/preprocessed.wav'
    await update_job(job_id, db_session, status=JobStatus.ASR_RUNNING, artifacts=job.artifacts)
    try:
        logger.info('Transcription started', extra={"job_id": job_id})
        with tempfile.NamedTemporaryFile() as tmp:
            download_file(job.artifacts['preprocessed'], tmp.name)
            segments: List[Segment] = transcribe(audio_path=Path(tmp.name))
        logger.info('Transcription done', extra={"job_id": job_id})
    except Exception:
        logger.exception("Transcription failed")
        await update_job(job_id, db_session, status=JobStatus.FAILED, error="Transcription failed")
        return
    chunks: List[str] = chunk_segments(segments)
    await update_job(job_id, db_session, status=JobStatus.SUMMARY_RUNNING)
    try:
        logger.info('LLM summarization started', extra={"job_id": job_id})
        summary: Summary = build_summary(chunks)
        logger.info('LLM summarization done', extra={"job_id": job_id})
    except Exception:
        logger.exception("LLM summarization failed")
        await update_job(job_id, db_session, status=JobStatus.FAILED, error="LLM summarization failed")
        return
    with tempfile.NamedTemporaryFile(mode='w', encoding='UTF-8') as tmp:
        tmp.write(summary.to_text())
        upload_file(tmp.name, f'{job_id}/summary.txt')
    job.artifacts['summary'] = f'{job_id}/summary.txt'
    await update_job(job_id, db_session, artifacts=job.artifacts)
    try:
        logger.info('Email sending started', extra={"job_id": job_id})
        send_email(recipients=job.emails,
               subject=summary.title,
               body= summary.to_text()
        )
        logger.info('Email sending done', extra={"job_id": job_id})
    except Exception:
        logger.exception("Email sending failed")
        await update_job(job_id, db_session, status=JobStatus.SMTP_FAILED, error=f"Email sending failed\nTo view the summary, go to "
                                                               f"/{job_id}/result")
        return
    await update_job(job_id, db_session, status=JobStatus.DONE)
    logger.info('Job done', extra={"job_id": job_id})
