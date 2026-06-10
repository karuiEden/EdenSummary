import contextlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from eden_summary.core import JobStatus
from eden_summary.pipeline import process_job
from eden_summary.summarize import Summary


class _AsyncCM:
    """Заглушка под `async with db_session.begin():`."""
    async def __aenter__(self):
        return None
    async def __aexit__(self, *exc):
        return False


def _make_db_session(job):
    db = MagicMock()
    db.begin = MagicMock(return_value=_AsyncCM())
    result = MagicMock()
    result.scalar_one_or_none = MagicMock(return_value=job)
    db.execute = AsyncMock(return_value=result)  # execute ожидается через await
    return db


def _make_job():
    return SimpleNamespace(
        id="job-1",
        artifacts={"source": "job-1/source.mp3"},  # суффикс важен для tmp-файла
        emails=["a@b.com"],
    )


@pytest.fixture
def mocks():
    job = _make_job()
    with contextlib.ExitStack() as stack:
        # ВАЖНО: патчим имена там, где они ИСПОЛЬЗУЮТСЯ (в eden_summary.pipeline),
        # а не там, где определены. pipeline сделал `from ... import name`,
        # поэтому в его namespace лежит своя ссылка на функцию.
        def p(name, **kw):
            return stack.enter_context(patch(f"eden_summary.pipeline.{name}", **kw))

        yield SimpleNamespace(
            job=job,
            db=_make_db_session(job),
            download_file=p("download_file"),
            upload_file=p("upload_file"),
            convert_to_wav=p("convert_to_wav"),
            transcribe=p("transcribe", return_value=["seg"]),
            chunk_segments=p("chunk_segments", return_value=["chunk text"]),
            build_summary=p("build_summary", return_value=Summary(
                title="Title", tldr=["t"], decisions=[], action_items=[], risks=[])),
            send_email=p("send_email"),
            update_job=p("update_job", new_callable=AsyncMock),
        )


def _statuses(update_job_mock):
    """Последовательность статусов, переданных в update_job (пропуская вызовы без status)."""
    return [c.kwargs["status"] for c in update_job_mock.await_args_list if "status" in c.kwargs]


def _error_for(update_job_mock, status):
    for c in update_job_mock.await_args_list:
        if c.kwargs.get("status") == status:
            return c.kwargs.get("error")
    return None


async def test_happy_path_status_sequence(mocks):
    await process_job("job-1", mocks.db)
    assert _statuses(mocks.update_job) == [
        JobStatus.ASR_RUNNING,
        JobStatus.SUMMARY_RUNNING,
        JobStatus.DONE,
    ]


async def test_happy_path_email_uses_title_as_subject(mocks):
    await process_job("job-1", mocks.db)
    mocks.send_email.assert_called_once()
    assert mocks.send_email.call_args.kwargs["subject"] == "Title"
    assert mocks.send_email.call_args.kwargs["recipients"] == ["a@b.com"]


async def test_job_not_found_raises(mocks):
    mocks.db.execute.return_value.scalar_one_or_none.return_value = None
    with pytest.raises(ValueError):
        await process_job("job-1", mocks.db)


async def test_audio_conversion_failure_stops_pipeline(mocks):
    mocks.convert_to_wav.side_effect = RuntimeError("ffmpeg boom")
    await process_job("job-1", mocks.db)

    assert _statuses(mocks.update_job) == [JobStatus.FAILED]
    assert _error_for(mocks.update_job, JobStatus.FAILED) == "Audio conversion failed"
    mocks.transcribe.assert_not_called()  # дальше не пошли


async def test_transcription_failure_stops_before_llm(mocks):
    mocks.transcribe.side_effect = RuntimeError("asr boom")
    await process_job("job-1", mocks.db)

    assert _statuses(mocks.update_job) == [JobStatus.ASR_RUNNING, JobStatus.FAILED]
    assert _error_for(mocks.update_job, JobStatus.FAILED) == "Transcription failed"
    mocks.build_summary.assert_not_called()


async def test_summarization_failure_stops_before_email(mocks):
    mocks.build_summary.side_effect = RuntimeError("llm boom")
    await process_job("job-1", mocks.db)

    assert _statuses(mocks.update_job) == [
        JobStatus.ASR_RUNNING, JobStatus.SUMMARY_RUNNING, JobStatus.FAILED]
    assert _error_for(mocks.update_job, JobStatus.FAILED) == "LLM summarization failed"
    mocks.send_email.assert_not_called()


async def test_smtp_failure_keeps_result_available(mocks):
    mocks.send_email.side_effect = RuntimeError("smtp boom")
    await process_job("job-1", mocks.db)

    statuses = _statuses(mocks.update_job)
    assert JobStatus.EMAIL_FAILED in statuses
    assert JobStatus.DONE not in statuses  # ключевое: SMTP-сбой != провал джоба
    # summary всё равно загружен в S3 до отправки письма
    assert any("summary.txt" in str(c.args) for c in mocks.upload_file.call_args_list)