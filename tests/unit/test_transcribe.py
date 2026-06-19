from pathlib import Path
from unittest import mock

import litellm
import pytest

import eden_summary.transcribe.audio as audio
import eden_summary.transcribe.transcribe_audio as ta
from eden_summary.transcribe.transcribe_audio import chunk_segments, _to_bcp47, Transcription


class _FakeResp:
    """Stand-in for litellm.transcription's verbose_json response."""
    def __init__(self, texts, language):
        self.segments = [{"text": t} for t in texts]
        self.language = language


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


@pytest.fixture
def transcribe_cfg(monkeypatch):
    class Cfg:
        model = "whisper-large-v3"
        api_key = "k"
        api_base = "http://asr"
        lang = None
        asr_chunk_seconds = 600
    monkeypatch.setattr(ta, "get_whisper_cfg", lambda: Cfg())
    return Cfg


class TestTranscribeFile:
    """The single-request path: segment extraction, language mapping, request args."""
    def test_extracts_nonblank_segments_and_maps_language(self, transcribe_cfg, monkeypatch, tmp_path):
        f = tmp_path / "a.wav"
        f.write_bytes(b"x")
        captured = {}
        monkeypatch.setattr(litellm, "transcription",
                            lambda **kw: captured.update(kw) or _FakeResp(["  ", "hello", "", "world"], "english"))
        result = ta._transcribe_file(f, "ru")
        assert result.segments == ["hello", "world"]   # blank/whitespace dropped
        assert result.language == "en"                  # provider name -> BCP-47
        assert captured["language"] == "ru"             # explicit hint wins over cfg.lang
        assert captured["model"] == "whisper-large-v3"
        assert captured["api_base"] == "http://asr"


class TestTranscribe:
    """Orchestration: single-vs-split branch, concat order, language pick. Patches
    _transcribe_file so no real file IO or network is touched."""
    def test_short_file_single_call_no_split(self, transcribe_cfg, monkeypatch):
        monkeypatch.setattr(ta, "get_duration", lambda p: 300.0)  # <= 600 -> no split
        split_spy = mock.Mock()
        monkeypatch.setattr(ta, "split_audio", split_spy)
        calls = []
        monkeypatch.setattr(ta, "_transcribe_file",
                            lambda path, lang: calls.append(path) or Transcription(["hello"], "en"))
        result = ta.transcribe(Path("a.wav"), None)
        split_spy.assert_not_called()
        assert len(calls) == 1
        assert result.segments == ["hello"]
        assert result.language == "en"

    def test_long_file_splits_concats_in_order_first_language(self, transcribe_cfg, monkeypatch):
        monkeypatch.setattr(ta, "get_duration", lambda p: 1500.0)  # > 600 -> split
        chunks = [Path("chunk_000.wav"), Path("chunk_001.wav"), Path("chunk_002.wav")]
        monkeypatch.setattr(ta, "split_audio", lambda path, secs, out: chunks)
        parts = {
            chunks[0]: Transcription(["a1", "a2"], "en"),
            chunks[1]: Transcription(["b1"], None),
            chunks[2]: Transcription(["c1", "c2"], "ru"),
        }
        seen = []
        monkeypatch.setattr(ta, "_transcribe_file", lambda path, lang: seen.append(path) or parts[path])
        result = ta.transcribe(Path("long.wav"), None)
        assert seen == chunks                                       # sequential, in order
        assert result.segments == ["a1", "a2", "b1", "c1", "c2"]    # order preserved
        assert result.language == "en"                              # first non-None detected

    def test_long_file_passes_chunk_seconds_to_split(self, transcribe_cfg, monkeypatch):
        monkeypatch.setattr(ta, "get_duration", lambda p: 1500.0)
        seen = {}
        def fake_split(path, secs, out):
            seen["secs"] = secs
            return [Path("chunk_000.wav")]
        monkeypatch.setattr(ta, "split_audio", fake_split)
        monkeypatch.setattr(ta, "_transcribe_file", lambda path, lang: Transcription(["x"], "en"))
        ta.transcribe(Path("long.wav"), None)
        assert seen["secs"] == 600


class TestSplitAudio:
    def test_ffmpeg_segment_args_timeout_and_sorted_output(self, monkeypatch, tmp_path):
        captured = {}
        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs
            (tmp_path / "chunk_001.wav").write_bytes(b"x")
            (tmp_path / "chunk_000.wav").write_bytes(b"x")
            return mock.Mock(returncode=0)
        monkeypatch.setattr(audio.subprocess, "run", fake_run)
        out = audio.split_audio(Path("in.wav"), 600, tmp_path)
        cmd, kwargs = captured["cmd"], captured["kwargs"]
        assert "-f" in cmd and "segment" in cmd
        assert "-segment_time" in cmd and "600" in cmd
        assert "timeout" in kwargs and kwargs.get("check") is True
        assert [p.name for p in out] == ["chunk_000.wav", "chunk_001.wav"]  # sorted


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