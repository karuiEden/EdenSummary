import asyncio
import json
import logging
import tempfile
from datetime import datetime, timedelta, UTC

from celery import Celery
from sqlalchemy import select

from eden_summary import pipeline
from eden_summary.core import get_celery_cfg, get_llm_cfg, AsyncLocalSession, Job, JobFieldEdit, JobStatus, update_job
from eden_summary.quality import fit_calibration, judge_faithfulness, verify_summq_consistency
from eden_summary.storage import download_file
from eden_summary.summarize import Summary

logger = logging.getLogger(__name__)

cfg = get_celery_cfg()
redis_url = f'redis://:{cfg.redis_password}@redis:6379/0'
celery_app = Celery('worker', broker=redis_url, backend=redis_url)

celery_app.conf.beat_schedule = {
    'reap-stale-jobs': {
        'task': 'eden_summary.worker.worker.reap_stale_jobs',
        'schedule': 300.0,
    },
    # Q3: refit the judge_score -> P(edit) calibration from accumulated user
    # edits. Monthly batch — compute-and-log only (no live apply-path yet).
    'calibrate-edit-scores': {
        'task': 'eden_summary.worker.worker.calibrate_edit_scores',
        'schedule': 2592000.0,  # ~30 days
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
    asyncio.run(_dispatch_post_terminal(job_id))


async def _dispatch_post_terminal(job_id: str) -> None:
    """Dispatch the post-terminal quality evals — each as a separate task so they
    stay out of the summarization hot path; each self-guards on status==DONE and is
    independent.

    §6.6 single source of truth: regeneration computes and persists its metric
    in-pipeline, so re-running it post-terminal would overwrite the stored score with
    a non-deterministic re-eval. We therefore dispatch only the eval that is NOT
    already persisted. Keying off the persisted state (not the regen config flag) also
    self-heals the case where regen is enabled but its in-pipeline computation failed
    (e.g. a rate limit) and wrote nothing — that eval must still run."""
    llm_cfg = get_llm_cfg()
    async with AsyncLocalSession() as db:
        async with db.begin():
            job = (await db.execute(select(Job).where(Job.id == job_id))).scalar_one_or_none()
            quality_eval = job.quality_eval if job else None
            summq_eval = job.summq_eval if job else None
    if llm_cfg.judge_enabled and quality_eval is None:
        evaluate_summary.delay(job_id)
    if llm_cfg.summq_enabled and summq_eval is None:
        verify_summq.delay(job_id)


@celery_app.task
def evaluate_summary(job_id: str):
    """Async Tier-2 faithfulness eval. Strictly post-terminal and advisory: it
    never changes job status and never fails the job — any error is logged and
    swallowed. Runs only when the job is DONE and the eval artifacts exist."""
    try:
        asyncio.run(_evaluate(job_id))
    except Exception:
        logger.exception("Tier-2 eval failed for job %s; leaving job untouched", job_id)


async def _evaluate(job_id: str) -> None:
    async with AsyncLocalSession() as db:
        async with db.begin():
            job = (await db.execute(select(Job).where(Job.id == job_id))).scalar_one_or_none()
            if job is None or job.status != JobStatus.DONE:
                logger.info("Tier-2 eval skipped for job %s (not DONE)", job_id)
                return
            artifacts = dict(job.artifacts)
        if 'transcript' not in artifacts or 'summary_json' not in artifacts:
            logger.info("Tier-2 eval skipped for job %s (missing artifacts)", job_id)
            return
        with tempfile.TemporaryDirectory() as tmp_dir:
            t_path = f'{tmp_dir}/transcript.json'
            s_path = f'{tmp_dir}/summary.json'
            await asyncio.to_thread(download_file, artifacts['transcript'], t_path)
            await asyncio.to_thread(download_file, artifacts['summary_json'], s_path)
            with open(t_path, encoding='UTF-8') as f:
                transcript_data = json.load(f)
            with open(s_path, encoding='UTF-8') as f:
                summary = Summary(**json.load(f))
        transcript = '\n'.join(transcript_data.get('segments', []))
        result = await asyncio.to_thread(judge_faithfulness, summary, transcript)
        await update_job(job_id, db, quality_eval=result.to_metadata())
        logger.info("Tier-2 eval done for job %s (evaluated=%s, score=%s)",
                    job_id, result.evaluated, result.overall_score)


@celery_app.task
def verify_summq(job_id: str):
    """Q4 SummQ QA-consistency check. Post-terminal, advisory, never-fail — mirrors
    evaluate_summary and is independent of it. Quizzes the summary, answers blind
    from the transcript, compares deterministically. Compute-and-log only: stores
    the consistency score in jobs.summq_eval; no live regeneration yet."""
    try:
        asyncio.run(_verify_summq(job_id))
    except Exception:
        logger.exception("Q4 SummQ failed for job %s; leaving job untouched", job_id)


async def _verify_summq(job_id: str) -> None:
    async with AsyncLocalSession() as db:
        async with db.begin():
            job = (await db.execute(select(Job).where(Job.id == job_id))).scalar_one_or_none()
            if job is None or job.status != JobStatus.DONE:
                logger.info("Q4 SummQ skipped for job %s (not DONE)", job_id)
                return
            artifacts = dict(job.artifacts)
        if 'transcript' not in artifacts or 'summary_json' not in artifacts:
            logger.info("Q4 SummQ skipped for job %s (missing artifacts)", job_id)
            return
        with tempfile.TemporaryDirectory() as tmp_dir:
            t_path = f'{tmp_dir}/transcript.json'
            s_path = f'{tmp_dir}/summary.json'
            await asyncio.to_thread(download_file, artifacts['transcript'], t_path)
            await asyncio.to_thread(download_file, artifacts['summary_json'], s_path)
            with open(t_path, encoding='UTF-8') as f:
                transcript_data = json.load(f)
            with open(s_path, encoding='UTF-8') as f:
                summary = Summary(**json.load(f))
        transcript = '\n'.join(transcript_data.get('segments', []))
        result = await asyncio.to_thread(verify_summq_consistency, summary, transcript)
        meta = result.to_metadata()
        await update_job(job_id, db, summq_eval=meta)
        logger.info("Q4 SummQ done for job %s (evaluated=%s, consistency=%s, below_threshold=%s)",
                    job_id, result.evaluated, result.consistency_score, meta['below_threshold'])


@celery_app.task
def calibrate_edit_scores():
    """Q3 monthly batch: refit P(edit) from accumulated user edits and log it.
    Advisory and self-contained — never touches jobs. The judge score is joined
    from jobs.quality_eval at fit time (not snapshotted into the edit row), so
    edits made before their async Tier-2 eval finished are still usable."""
    asyncio.run(_calibrate())


def _labels_from_edits(edits, scores_by_job) -> list[tuple[float, bool]]:
    """Join each edit row to its judged field score (jobs.quality_eval.field_scores)
    and drop rows whose field has no score — the judge was skipped or its async
    eval had not finished. Pure, so the join/skip logic is unit-testable."""
    labels: list[tuple[float, bool]] = []
    for e in edits:
        score = scores_by_job.get(e.job_id, {}).get(e.field)
        if score is None:
            continue
        labels.append((float(score), bool(e.edited)))
    return labels


async def _calibrate() -> None:
    async with AsyncLocalSession() as db:
        async with db.begin():
            edits = list((await db.execute(select(JobFieldEdit))).scalars().all())
            job_ids = {e.job_id for e in edits}
            jobs = []
            if job_ids:
                jobs = list((await db.execute(select(Job).where(Job.id.in_(job_ids)))).scalars().all())
        scores_by_job = {
            j.id: (j.quality_eval or {}).get('field_scores', {}) for j in jobs
        }
        labels = _labels_from_edits(edits, scores_by_job)
        calibration = fit_calibration(labels)
        logger.info(
            "Edit calibration refit: method=%s n=%d (edits=%d) params=%s",
            calibration.method, calibration.n, len(edits), calibration.to_metadata(),
        )


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