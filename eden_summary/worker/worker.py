import asyncio
import logging
from datetime import datetime, timedelta, UTC

from celery import Celery
from sqlalchemy import select

from eden_summary import pipeline
from eden_summary.core import get_celery_cfg, AsyncLocalSession, Job, JobStatus

logger = logging.getLogger(__name__)

cfg = get_celery_cfg()
redis_url = f'redis://:{cfg.redis_password}@redis:6379/0'
celery_app = Celery('worker', broker=redis_url, backend=redis_url)

celery_app.conf.beat_schedule = {
    'reap-stale-jobs': {
        'task': 'eden_summary.worker.worker.reap_stale_jobs',
        'schedule': 300.0,
    },
}

_ACTIVE_STATUSES = [
    JobStatus.QUEUED,
    JobStatus.ASR_RUNNING,
    JobStatus.SUMMARY_RUNNING,
]


@celery_app.task(acks_late=True, reject_on_worker_lost=True, soft_time_limit=cfg.soft_timeout, time_limit=cfg.hard_timeout)
def process_job(job_id: str, language: str | None = None):
    async def _run():
        async with AsyncLocalSession() as db_session:
            await pipeline.process_job(job_id, language, db_session)
    asyncio.run(_run())


@celery_app.task
def reap_stale_jobs():
    asyncio.run(_reap())


async def _reap() -> None:
    threshold = datetime.now(UTC) - timedelta(seconds=cfg.reaper_stale_seconds)
    async with AsyncLocalSession() as db:
        async with db.begin():
            stmt = select(Job).where(
                Job.status.in_(_ACTIVE_STATUSES),
                Job.updated_at < threshold,
            )
            result = await db.execute(stmt)
            stale = result.scalars().all()
            for job in stale:
                logger.warning(
                    "Reaper: failing stale job %s (status=%s, last update=%s)",
                    job.id, job.status, job.updated_at,
                )
                job.status = JobStatus.FAILED
                job.error = "Job timed out (reaper)"
                job.updated_at = datetime.now(UTC)