from celery import Celery

from system import pipeline
from system.config import get_celery_cfg

cfg = get_celery_cfg()
redis_url = f'redis://:{cfg.redis_password}@redis:6380/0'
celery_app = Celery('worker', broker=redis_url, backend=redis_url)

@celery_app.task
def process_job(job_id: str):
    pipeline.process_job(job_id)