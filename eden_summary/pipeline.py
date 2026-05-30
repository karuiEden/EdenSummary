import json
import logging
from pathlib import Path
from typing import List

from faster_whisper.transcribe import Segment

from eden_summary.core import JobStatus, update_job
from eden_summary.core import get_app_cfg
from eden_summary.email_service import send_email
from eden_summary.summarize import build_summary, Summary
from eden_summary.transcribe import chunk_segments, transcribe
from eden_summary.transcribe import convert_to_wav

logger = logging.getLogger(__name__)

def process_job(job_id: str):
    config = get_app_cfg()
    logger.info('Job started', extra={"job_id": job_id})
    job_path: Path = Path(f"{config.output_dir}/{job_id}")
    with open(job_path / "job.json", mode='r', encoding='UTF-8') as f:
        job = json.load(f)
    try:
        logger.info('Audio Preprocessing started', extra={"job_id": job_id})
        convert_to_wav(job['artifacts']['source'], job_path/'preprocessed.wav')
        logger.info('Audio Preprocessing done', extra={"job_id": job_id})
    except Exception:
        logger.exception("Audio conversion failed")
        update_job(job_id, status=JobStatus.FAILED, error="Audio conversion failed")
        return
    job['artifacts']['preprocessed'] = str(job_path/'preprocessed.wav')
    job['status'] = JobStatus.ASR_RUNNING
    update_job(job_id, status=job['status'], artifacts=job['artifacts'])
    try:
        logger.info('Transcription started', extra={"job_id": job_id})
        segments: List[Segment] = transcribe(audio_path=Path(job['artifacts']['preprocessed']))
        logger.info('Transcription done', extra={"job_id": job_id})
    except Exception:
        logger.exception("Transcription failed")
        update_job(job_id, status=JobStatus.FAILED, error="Transcription failed")
        return
    chunks: List[str] = chunk_segments(segments)
    job['status'] = JobStatus.SUMMARY_RUNNING
    update_job(job_id, status=job['status'])
    try:
        logger.info('LLM summarization started', extra={"job_id": job_id})
        summary: Summary = build_summary(chunks)
        logger.info('LLM summarization done', extra={"job_id": job_id})
    except Exception:
        logger.exception("LLM summarization failed")
        update_job(job_id, status=JobStatus.FAILED, error="LLM summarization failed")
        return
    with open(job_path/'summary.txt', mode='w', encoding='UTF-8') as f:
        f.write(summary.to_text())
    try:
        logger.info('Email sending started', extra={"job_id": job_id})
        send_email(recipients=job['emails'],
               subject=summary.title,
               body= summary.to_text()
        )
        logger.info('Email sending done', extra={"job_id": job_id})
    except Exception:
        logger.exception("Email sending failed")
        update_job(job_id, status=JobStatus.SMTP_FAILED, error=f"Email sending failed\nTo view the summary, go to "
                                                               f"/{job_id}/result")
        return
    update_job(job_id, status=JobStatus.DONE)
    logger.info('Job done', extra={"job_id": job_id})
