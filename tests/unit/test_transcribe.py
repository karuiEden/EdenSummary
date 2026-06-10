from dataclasses import dataclass

import pytest

import eden_summary.transcribe.transcribe_audio as ta
from eden_summary.transcribe.transcribe_audio import chunk_segments, _to_bcp47


@pytest.fixture
def max_chars_10(monkeypatch):
    class Cfg:
        chunk_max_chars = 10
    monkeypatch.setattr(ta, "get_whisper_cfg", lambda: Cfg())


class TestChunkSegments:
    def test_empty_input(self, max_chars_10):
        assert chunk_segments([]) == []

    def test_single_short_segment_one_chunk(self, max_chars_10):
        assert chunk_segments(["hi"]) == ["hi"]

    def test_accumulates_until_threshold(self, max_chars_10):
        # 5 + 5 == 10 >= 10 -> один чанк, остаток пуст
        segs = ["abcde", "fghij"]
        assert chunk_segments(segs) == ["abcdefghij"]

    def test_splits_when_threshold_crossed(self, max_chars_10):
        # первый сегмент уже >= 10 -> сброс; "xyz" уходит в финальный чанк
        segs = ["abcdefghijk", "xyz"]
        assert chunk_segments(segs) == ["abcdefghijk", "xyz"]

    def test_trailing_remainder_flushed(self, max_chars_10):
        # ничего не достигает порога -> всё уходит одним финальным чанком
        segs = ["ab", "cd"]
        assert chunk_segments(segs) == ["abcd"]


class TestToBcp47:
    def test_full_name_russian(self):
        assert _to_bcp47('russian') == 'ru'

    def test_iso_code(self):
        assert _to_bcp47('en') == 'en'

    def test_case_insensitive(self):
        assert _to_bcp47('RUSSIAN') == 'ru'

    def test_unknown_returns_none(self):
        assert _to_bcp47('klingon') is None

    def test_none_returns_none(self):
        assert _to_bcp47(None) is None