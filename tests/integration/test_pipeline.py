import contextlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from eden_summary.core import JobStatus
from eden_summary.pipeline import process_job
from eden_summary.summarize import Summary
from eden_summary.transcribe import Transcription


class _AsyncCM:
    """Stub for `async with db_session.begin():`."""
    async def __aenter__(self):
        return None
    async def __aexit__(self, *exc):
        return False


def _make_db_session(job):
    db = MagicMock()
    db.begin = MagicMock(return_value=_AsyncCM())
    result = MagicMock()
    result.scalar_one_or_none = MagicMock(return_value=job)
    db.execute = AsyncMock(return_value=result)  # execute is awaited
    return db


def _make_job():
    return SimpleNamespace(
        id="job-1",
        status=JobStatus.QUEUED,
        artifacts={"source": "job-1/source.mp3"},  # the suffix matters for the tmp file
        emails=["a@b.com"],
    )


@pytest.fixture
def mocks():
    job = _make_job()

    # update_job is mocked, but the pipeline reads job.artifacts['preprocessed'] and
    # updated fields AFTER it — so the mock reflects changes back into the in-memory job.
    async def _fake_update(job_id, session, **fields):
        for k, v in fields.items():
            setattr(job, k, v)

    with contextlib.ExitStack() as stack:
        # IMPORTANT: patch names where they are USED (in eden_summary.pipeline),
        # not where they are defined. pipeline did `from ... import name`, so its
        # namespace holds its own reference to the function.
        def p(name, **kw):
            return stack.enter_context(patch(f"eden_summary.pipeline.{name}", **kw))

        yield SimpleNamespace(
            job=job,
            db=_make_db_session(job),
            download_file=p("download_file"),
            upload_file=p("upload_file"),
            convert_to_wav=p("convert_to_wav"),
            get_duration=p("get_duration", return_value=100.0),
            transcribe=p("transcribe", return_value=Transcription(segments=["seg"], language="en-US")),
            summarize_transcript=p("summarize_transcript", return_value=Summary(
                title="Title", tldr=["t"], decisions=[], action_items=[], risks=[])),
            send_email=p("send_email"),
            update_job=p("update_job", new=AsyncMock(side_effect=_fake_update)),
            # regen is read in-pipeline; default it OFF so existing tests stay
            # hermetic (no dependency on .env completeness) and behaviour is unchanged.
            get_llm_cfg=p("get_llm_cfg", return_value=SimpleNamespace(summq_regen_enabled=False)),
            repair_if_inconsistent=p("repair_if_inconsistent", return_value=(Summary(
                title="Title", tldr=["t"], decisions=[], action_items=[], risks=[]), {})),
        )


def _statuses(update_job_mock):
    """Sequence of statuses passed to update_job (skipping calls without status)."""
    return [c.kwargs["status"] for c in update_job_mock.await_args_list if "status" in c.kwargs]


def _error_for(update_job_mock, status):
    for c in update_job_mock.await_args_list:
        if c.kwargs.get("status") == status:
            return c.kwargs.get("error")
    return None


async def test_happy_path_status_sequence(mocks):
    await process_job("job-1", "en-US", mocks.db)
    assert _statuses(mocks.update_job) == [
        JobStatus.ASR_RUNNING,
        JobStatus.SUMMARY_RUNNING,
        JobStatus.DONE,
    ]


async def test_happy_path_email_uses_title_as_subject(mocks):
    await process_job("job-1", "en-US", mocks.db)
    mocks.send_email.assert_called_once()
    assert mocks.send_email.call_args.kwargs["subject"] == "Title"
    assert mocks.send_email.call_args.kwargs["recipients"] == ["a@b.com"]


async def test_happy_path_persists_eval_artifacts_and_quality_flags(mocks):
    # quality wiring: summary.txt + transcript.json + summary_json.json
    # uploaded, and quality_flags written alongside artifacts (without failing the job).
    await process_job("job-1", "en-US", mocks.db)
    uploaded = " ".join(str(c.args) for c in mocks.upload_file.call_args_list)
    assert "summary.txt" in uploaded
    assert "transcript.json" in uploaded
    assert "summary_json.json" in uploaded
    # the update_job that wrote artifacts also carried quality_flags (inline guards, advisory)
    artifact_calls = [c for c in mocks.update_job.await_args_list if "artifacts" in c.kwargs]
    assert artifact_calls and "quality_flags" in artifact_calls[-1].kwargs


async def test_job_not_found_raises(mocks):
    mocks.db.execute.return_value.scalar_one_or_none.return_value = None
    with pytest.raises(ValueError):
        await process_job("job-1", "en-US", mocks.db)


async def test_audio_conversion_failure_stops_pipeline(mocks):
    mocks.convert_to_wav.side_effect = RuntimeError("ffmpeg boom")
    await process_job("job-1", "en-US", mocks.db)

    assert _statuses(mocks.update_job) == [JobStatus.FAILED]
    assert _error_for(mocks.update_job, JobStatus.FAILED) == "Audio conversion failed"
    mocks.transcribe.assert_not_called()  # went no further


async def test_transcription_failure_stops_before_llm(mocks):
    mocks.transcribe.side_effect = RuntimeError("asr boom")
    await process_job("job-1", "en-US", mocks.db)

    assert _statuses(mocks.update_job) == [JobStatus.ASR_RUNNING, JobStatus.FAILED]
    assert _error_for(mocks.update_job, JobStatus.FAILED) == "Transcription failed"
    mocks.summarize_transcript.assert_not_called()


async def test_summarization_failure_stops_before_email(mocks):
    mocks.summarize_transcript.side_effect = RuntimeError("llm boom")
    await process_job("job-1", "en-US", mocks.db)

    assert _statuses(mocks.update_job) == [
        JobStatus.ASR_RUNNING, JobStatus.SUMMARY_RUNNING, JobStatus.FAILED]
    assert _error_for(mocks.update_job, JobStatus.FAILED) == "LLM summarization failed"
    mocks.send_email.assert_not_called()


async def test_regeneration_runs_when_enabled(mocks):
    # with regen on, repair_if_inconsistent runs before the email, the chosen
    # summary is the one served, and its in-pipeline metrics are persisted.
    repaired = Summary(title="Repaired", tldr=["r"], decisions=["d2"], action_items=[], risks=[])
    mocks.get_llm_cfg.return_value = SimpleNamespace(summq_regen_enabled=True)
    mocks.repair_if_inconsistent.return_value = (repaired, {"summq_eval": {"consistency_score": 0.9}})

    await process_job("job-1", "en-US", mocks.db)

    mocks.repair_if_inconsistent.assert_called_once()
    assert mocks.send_email.call_args.kwargs["subject"] == "Repaired"  # served = repaired
    artifact_calls = [c for c in mocks.update_job.await_args_list if "artifacts" in c.kwargs]
    assert artifact_calls[-1].kwargs.get("summq_eval") == {"consistency_score": 0.9}


async def test_smtp_failure_keeps_result_available(mocks):
    mocks.send_email.side_effect = RuntimeError("smtp boom")
    await process_job("job-1", "en-US", mocks.db)

    statuses = _statuses(mocks.update_job)
    assert JobStatus.EMAIL_FAILED in statuses
    assert JobStatus.DONE not in statuses  # key point: an SMTP failure != a failed job
    # summary is uploaded to S3 before the email is sent anyway
    assert any("summary.txt" in str(c.args) for c in mocks.upload_file.call_args_list)
