from dataclasses import dataclass

import pytest

import eden_summary.transcribe.transcribe_audio as ta
from eden_summary.transcribe.transcribe_audio import chunk_segments


@dataclass
class FakeSegment:
    # chunk_segments читает только .text
    text: str


@pytest.fixture
def max_chars_10(monkeypatch):
    class Cfg:
        chunk_max_chars = 10
    monkeypatch.setattr(ta, "get_whisper_cfg", lambda: Cfg())


class TestChunkSegments:
    def test_empty_input(self, max_chars_10):
        assert chunk_segments([]) == []

    def test_single_short_segment_one_chunk(self, max_chars_10):
        assert chunk_segments([FakeSegment("hi")]) == ["hi"]

    def test_accumulates_until_threshold(self, max_chars_10):
        # 5 + 5 == 10 >= 10 -> один чанк, остаток пуст
        segs = [FakeSegment("abcde"), FakeSegment("fghij")]
        assert chunk_segments(segs) == ["abcdefghij"]

    def test_splits_when_threshold_crossed(self, max_chars_10):
        # первый сегмент уже >= 10 -> сброс; "xyz" уходит в финальный чанк
        segs = [FakeSegment("abcdefghijk"), FakeSegment("xyz")]
        assert chunk_segments(segs) == ["abcdefghijk", "xyz"]

    def test_trailing_remainder_flushed(self, max_chars_10):
        # ничего не достигает порога -> всё уходит одним финальным чанком
        segs = [FakeSegment("ab"), FakeSegment("cd")]
        assert chunk_segments(segs) == ["abcd"]