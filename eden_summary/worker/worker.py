import asyncio

from celery import Celery

from eden_summary import pipeline
from eden_summary.core import get_celery_cfg, AsyncLocalSession


cfg = get_celery_cfg()
redis_url = f'redis://:{cfg.redis_password}@redis:6379/0'
celery_app = Celery('worker', broker=redis_url, backend=redis_url)

@celery_app.task
def process_job(job_id: str):
    async def _run():
        async with AsyncLocalSession() as db_session:
            await pipeline.process_job(job_id, db_session)
    asyncio.run(_run())