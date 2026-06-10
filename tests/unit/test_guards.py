import io
import sys
from unittest.mock import MagicMock

import pytest
from starlette.datastructures import UploadFile

from eden_summary.core.guards import (
    check_and_parse_emails,
    check_id,
    check_file,
)


class TestCheckAndParseEmails:
    def test_none_returns_none(self):
        assert check_and_parse_emails(None) is None

    def test_empty_string_returns_none(self):
        assert check_and_parse_emails("") is None

    def test_single_valid(self):
        assert check_and_parse_emails("a@b.com") == ["a@b.com"]

    def test_multiple_valid_and_trimmed(self):
        assert check_and_parse_emails(" a@b.com , c@d.com ") == ["a@b.com", "c@d.com"]

    def test_invalid_dropped(self):
        assert check_and_parse_emails("garbage, a@b.com") == ["a@b.com"]

    def test_all_invalid_returns_none(self):
        assert check_and_parse_emails("garbage, nope") is None


class TestCheckId:
    def test_valid_uuid4(self):
        assert check_id("3a95402f-42ec-49ce-89ea-49bbc9f51462") is True

    def test_invalid_string(self):
        assert check_id("not-a-uuid") is False


class TestCheckFile:
    """Двухслойная валидация: magic-категория + ffprobe (is_audio)."""

    @pytest.mark.asyncio
    async def test_accepts_audio(self, monkeypatch):
        mock_magic = MagicMock()
        mock_magic.from_file.return_value = "audio/x-wav"
        monkeypatch.setitem(sys.modules, 'magic', mock_magic)
        monkeypatch.setattr("eden_summary.core.guards.is_audio", lambda p: True)
        f = UploadFile(filename="meeting.wav", file=io.BytesIO(b"RIFF...."))
        assert await check_file(f) is True

    @pytest.mark.asyncio
    async def test_rejects_non_media_mime(self, monkeypatch):
        mock_magic = MagicMock()
        mock_magic.from_file.return_value = "image/png"
        monkeypatch.setitem(sys.modules, 'magic', mock_magic)
        monkeypatch.setattr("eden_summary.core.guards.is_audio", lambda p: True)
        f = UploadFile(filename="evil.png", file=io.BytesIO(b"\x89PNG"))
        assert await check_file(f) is False

    @pytest.mark.asyncio
    async def test_rejects_when_no_audio_stream(self, monkeypatch):
        mock_magic = MagicMock()
        mock_magic.from_file.return_value = "audio/x-wav"
        monkeypatch.setitem(sys.modules, 'magic', mock_magic)
        monkeypatch.setattr("eden_summary.core.guards.is_audio", lambda p: False)
        f = UploadFile(filename="fake.wav", file=io.BytesIO(b"not audio"))
        assert await check_file(f) is False