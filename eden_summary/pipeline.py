import asyncio
import json
import logging
import time
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import List

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from eden_summary.core import Job, JobStatus, update_job
from eden_summary.email_service import send_email
from eden_summary.metrics import job_stage_seconds, asr_processing_ratio, job_terminal_total, quality_guard_seconds
from eden_summary.quality import run_inline_guards
from eden_summary.storage import upload_file, download_file
from eden_summary.summarize import summarize_transcript, Summary
from eden_summary.transcribe import transcribe, convert_to_wav, get_duration, Transcription

logger = logging.getLogger(__name__)


async def process_job(job_id: str, language: str | None, db_session: AsyncSession):
    logger.info('Job started', extra={"job_id": job_id})
    async with db_session.begin():
        stmt = select(Job).where(Job.id == job_id)
        result = await db_session.execute(stmt)
        job = result.scalar_one_or_none()
        if job is None:
            raise ValueError("Job not found")
        if job.status != JobStatus.QUEUED:
            logger.info('Job already processed', extra={"job_id": job_id})
            return
    job_start = time.monotonic()
    audio_duration: float | None = None
    try:
        logger.info('Audio Preprocessing started', extra={"job_id": job_id})
        source_ext = Path(job.artifacts['source']).suffix
        with tempfile.NamedTemporaryFile(suffix=source_ext) as tmp_source:
            with tempfile.NamedTemporaryFile(suffix='.wav') as tmp_prep:
                download_file(job.artifacts['source'], tmp_source.name)
                logger.info("source size: %d", Path(tmp_source.name).stat().st_size)
                convert_to_wav(Path(tmp_source.name), Path(tmp_prep.name))
                audio_duration = get_duration(Path(tmp_prep.name))
                upload_file(tmp_prep.name, f'{job_id}/preprocessed.wav')
        logger.info('Audio Preprocessing done', extra={"job_id": job_id})
    except Exception as e:
        logger.exception(f"Audio conversion failed\n{getattr(e, 'stderr', b'').decode()}", extra={"job_id": job_id})
        await update_job(job_id, db_session, status=JobStatus.FAILED, error="Audio conversion failed")
        job_terminal_total.labels(status='failed').inc()
        return
    artifacts = {**job.artifacts, 'preprocessed': f'{job_id}/preprocessed.wav'}
    await update_job(job_id, db_session, status=JobStatus.ASR_RUNNING, artifacts=artifacts)
    try:
        logger.info('Transcription started', extra={"job_id": job_id})
        asr_start = time.monotonic()
        with tempfile.NamedTemporaryFile() as tmp:
            download_file(job.artifacts['preprocessed'], tmp.name)
            transcription: Transcription = transcribe(audio_path=Path(tmp.name), language=language)
        asr_elapsed = time.monotonic() - asr_start
        job_stage_seconds.labels(stage='asr').observe(asr_elapsed)
        if audio_duration:
            asr_processing_ratio.observe(asr_elapsed / audio_duration)
        segments: List[str] = transcription.segments
        detected_lang: str | None = transcription.language
        logger.info('Transcription done', extra={"job_id": job_id})
    except Exception:
        logger.exception("Transcription failed")
        await update_job(job_id, db_session, status=JobStatus.FAILED, error="Transcription failed")
        job_terminal_total.labels(status='failed').inc()
        return
    if not segments:
        logger.warning("Empty transcription", extra={"job_id": job_id})
        await update_job(job_id, db_session, status=JobStatus.FAILED, error="Transcription returned empty result")
        job_terminal_total.labels(status='failed').inc()
        return
    await update_job(job_id, db_session, status=JobStatus.SUMMARY_RUNNING)
    try:
        logger.info('LLM summarization started', extra={"job_id": job_id})
        summarize_start = time.monotonic()
        summary: Summary = await asyncio.to_thread(summarize_transcript, segments)
        job_stage_seconds.labels(stage='summarize').observe(time.monotonic() - summarize_start)
        logger.info('LLM summarization done', extra={"job_id": job_id})
    except Exception:
        logger.exception("LLM summarization failed")
        await update_job(job_id, db_session, status=JobStatus.FAILED, error="LLM summarization failed")
        job_terminal_total.labels(status='failed').inc()
        return
    # Tier 1 inline quality guards: advisory, never fail the job.
    quality_flags: dict | None = None
    try:
        quality_start = time.monotonic()
        guard_result = await asyncio.to_thread(run_inline_guards, summary, '\n'.join(segments), detected_lang)
        quality_guard_seconds.observe(time.monotonic() - quality_start)
        quality_flags = guard_result.to_metadata()
        if not guard_result.passed:
            logger.info('Quality guards flagged %d item(s)', len(guard_result.flags), extra={"job_id": job_id})
    except Exception:
        logger.exception("Quality guards step failed; continuing", extra={"job_id": job_id})
    artifacts = {**job.artifacts}
    with tempfile.NamedTemporaryFile(mode='w', encoding='UTF-8') as tmp:
        tmp.write(summary.to_text(detected_lang))
        tmp.flush()
        upload_file(tmp.name, f'{job_id}/summary.txt')
    artifacts['summary'] = f'{job_id}/summary.txt'
    # Persist the raw transcript and structured summary so the async Tier-2
    # faithfulness eval (a separate process, after this job ends) can reload them.
    for name, payload in (
        ('transcript', {'language': detected_lang, 'segments': segments}),
        ('summary_json', asdict(summary)),
    ):
        with tempfile.NamedTemporaryFile(mode='w', encoding='UTF-8', suffix='.json') as tmp:
            json.dump(payload, tmp, ensure_ascii=False)
            tmp.flush()
            key = f'{job_id}/{name}.json'
            upload_file(tmp.name, key)
            artifacts[name] = key
    await update_job(job_id, db_session, artifacts=artifacts, quality_flags=quality_flags)
    try:
        logger.info('Email sending started', extra={"job_id": job_id})
        send_email(recipients=job.emails, subject=summary.title, body=summary.to_text(detected_lang))
        logger.info('Email sending done', extra={"job_id": job_id})
    except Exception:
        logger.exception("Email sending failed")
        await update_job(job_id, db_session, status=JobStatus.EMAIL_FAILED, error=f"Email sending failed\nTo view the summary, go to /{job_id}/result")
        job_terminal_total.labels(status='email_failed').inc()
        return
    job_stage_seconds.labels(stage='total').observe(time.monotonic() - job_start)
    job_terminal_total.labels(status='done').inc()
    await update_job(job_id, db_session, status=JobStatus.DONE)
    logger.info('Job done', extra={"job_id": job_id})