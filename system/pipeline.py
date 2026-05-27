import json
import smtplib
from pathlib import Path
from subprocess import CalledProcessError
from typing import List

from faster_whisper.transcribe import Segment
from litellm import AuthenticationError, RateLimitError, APIError

from system.audio import convert_to_wav
from system.config import get_app_cfg
from system.emailer import send_email
from system.job_store import JobStatus, update_job
from system.summarize import build_summary, Summary
from system.transcribe import transcribe, chunk_segments


def process_job(job_id: str):
    config = get_app_cfg()
    job_path: Path = Path(f"{config.output_dir}/{job_id}")
    with open(job_path / "job.json", mode='r', encoding='UTF-8') as f:
        job = json.load(f)
    try:
        convert_to_wav(job['artifacts']['source'], job_path/'preprocessed.wav')
    except CalledProcessError as e:
        update_job(job_id, status=JobStatus.FAILED, error=str(e))
        return
    except Exception as e:
        update_job(job_id, status=JobStatus.FAILED, error=str(e))
        return
    job['artifacts']['preprocessed'] = str(job_path/'preprocessed.wav')
    job['status'] = JobStatus.ASR_RUNNING
    update_job(job_id, status=job['status'], artifacts=job['artifacts'])
    try:
        segments:List[Segment] = transcribe(audio_path=Path(job['artifacts']['preprocessed']))
    except Exception as e:
        update_job(job_id, status=JobStatus.FAILED, error=str(e))
        return
    chunks: List[str] = chunk_segments(segments)
    job['status'] = JobStatus.SUMMARY_RUNNING
    update_job(job_id, status=job['status'])
    try:
        summary: Summary = build_summary(chunks)
    except (AuthenticationError, RateLimitError, APIError) as e:
        update_job(job_id, status=JobStatus.FAILED, error=str(e))
        return
    with open(job_path/'summary.txt', mode='w', encoding='UTF-8') as f:
        f.write(summary.to_text())
    try:
        send_email(recipients=job['emails'],
               subject=summary.title,
               body= summary.to_text()
        )
    except (smtplib.SMTPException, ConnectionRefusedError) as e:
        update_job(job_id, status=JobStatus.SMTP_FAILED, error=str(e))
        return
    update_job(job_id, status=JobStatus.DONE)
