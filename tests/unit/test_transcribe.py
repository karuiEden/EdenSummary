from dataclasses import dataclass

import pytest

import eden_summary.transcribe.transcribe_audio as ta
from eden_summary.transcribe.transcribe_audio import chunk_segments, _to_bcp47


@pytest.fixture
def max_chars_10(monkeypatch):
    class Cfg:
        chunk_max_chars = 10
        chunk_overlap_chars = 0
    monkeypatch.setattr(ta, "get_whisper_cfg", lambda: Cfg())


@pytest.fixture
def max10_overlap4(monkeypatch):
    class Cfg:
        chunk_max_chars = 10
        chunk_overlap_chars = 4
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

    def test_overlap_carries_boundary_into_next_chunk(self, max10_overlap4):
        # max=10, overlap=4: на флаше тащим хвост ~4 симв. сегментами в след. чанк
        segs = ["aaa", "bbb", "ccc", "ddd", "eee"]
        chunks = chunk_segments(segs)
        assert chunks == ["aaabbbcccddd", "cccdddeee"]
        # граничные сегменты присутствуют в обоих соседних чанках
        assert "ccc" in chunks[0] and "ccc" in chunks[1]
        assert "ddd" in chunks[0] and "ddd" in chunks[1]

    def test_no_duplicate_trailing_overlap_chunk(self, max10_overlap4):
        # флаш происходит ровно на последнем сегменте -> хвост-overlap не дублируется
        segs = ["aaa", "bbb", "ccc", "ddd"]
        assert chunk_segments(segs) == ["aaabbbcccddd"]


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