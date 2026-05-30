from celery import Celery

from system import pipeline
from system.config import get_celery_cfg

cfg = get_celery_cfg()

celery_app = Celery('worker', broker=cfg.redis_url, backend=cfg.redis_url)

@celery_app.task
def process_job(job_id: str):
    pipeline.process_job(job_id)