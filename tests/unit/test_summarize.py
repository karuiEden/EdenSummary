from unittest.mock import patch, MagicMock

import pytest

from eden_summary.summarize.summarize import _parse_json, _ensure_list, Summary, summarize_chunk, _render_item


def _fake_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.choices[0].message.content = content
    return resp

class TestParseJson:
    def test_plain_json(self):
        assert _parse_json('{"a": 1}') == {"a": 1}

    def test_strips_surrounding_text(self):
        assert _parse_json('Here you go: {"a": 1} done') == {"a": 1}

    def test_strips_markdown_fences(self):
        assert _parse_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_handles_nested_objects(self):
        assert _parse_json('{"a": {"b": 2}}') == {"a": {"b": 2}}

    def test_raises_when_no_json_object(self):
        with pytest.raises(ValueError):
            _parse_json("no json here at all")


class TestEnsureList:
    def test_wraps_bare_string(self):
        # LLM иногда отдаёт строку вместо списка — её надо обернуть
        assert _ensure_list("single") == ["single"]

    def test_passes_list_through(self):
        assert _ensure_list(["a", "b"]) == ["a", "b"]

    def test_none_becomes_empty(self):
        assert _ensure_list(None) == []

    def test_unexpected_type_becomes_empty(self):
        assert _ensure_list(123) == []


class TestSummaryToText:
    def _full(self):
        return Summary(
            title="Title",
            tldr=["point one"],
            decisions=["decided X"],
            action_items=["do Y"],
            risks=["risk Z"],
        )

    def test_renders_all_sections(self):
        text = self._full().to_text()
        for header in ("TL;DR", "Решения", "Задачи", "Риски"):
            assert header in text
        for item in ("- point one", "- decided X", "- do Y", "- risk Z"):
            assert item in text

    def test_skips_empty_sections(self):
        s = Summary(title="T", tldr=["p"], decisions=[], action_items=[], risks=[])
        text = s.to_text()
        assert "TL;DR" in text
        assert "Решения" not in text
        assert "Задачи" not in text
        assert "Риски" not in text

    def test_title_is_not_rendered(self):
        # Документируем текущее поведение: title в to_text() не попадает
        s = Summary(title="UNIQUE_TITLE", tldr=["p"], decisions=[], action_items=[], risks=[])
        assert "UNIQUE_TITLE" not in s.to_text()

    def test_all_empty_is_empty_string(self):
        s = Summary(title="T", tldr=[], decisions=[], action_items=[], risks=[])
        assert s.to_text() == ""

    def test_to_text_russian_headers(self):
        s = self._full()
        text = s.to_text('ru-RU')
        assert 'Решения' in text
        assert 'Задачи' in text

    def test_to_text_english_headers(self):
        s = self._full()
        text = s.to_text('en-US')
        assert 'Decisions' in text
        assert 'Action Items' in text

    def test_to_text_unknown_lang_falls_back_to_default(self):
        s = self._full()
        text = s.to_text('zh')
        # неизвестный язык → DEFAULT_LOCALE ('ru')
        assert 'Решения' in text

    def test_to_text_none_lang_falls_back_to_default(self):
        s = self._full()
        assert s.to_text(None) == s.to_text()


class TestStructuredActionItems:
    # LLM возвращает action_items словарями {task, who, deadline} — to_text должен
    # рендерить их в читаемую строку, отбрасывая плейсхолдеры ('unspecified' и т.п.)
    def _with_actions(self, action_items):
        return Summary(title="T", tldr=[], decisions=[],
                       action_items=action_items, risks=[])

    def test_dict_with_who_and_deadline(self):
        s = self._with_actions([{'task': 'work on designs', 'who': 'industrial designer', 'deadline': 'next meeting'}])
        assert "- work on designs (industrial designer; by next meeting)" in s.to_text('en')

    def test_placeholder_who_is_dropped(self):
        s = self._with_actions([{'task': 'work on designs', 'who': 'unspecified', 'deadline': 'next meeting'}])
        text = s.to_text('en')
        assert "- work on designs (by next meeting)" in text
        assert "unspecified" not in text

    def test_task_only_no_meta(self):
        text = self._with_actions([{'task': 'do the thing', 'who': 'unspecified', 'deadline': 'n/a'}]).to_text('en')
        assert "- do the thing" in text
        assert "(" not in text

    def test_mixed_string_and_dict(self):
        text = self._with_actions(["plain task", {'task': 'structured', 'who': 'bob'}]).to_text('en')
        assert "- plain task" in text
        assert "- structured (bob)" in text

    def test_all_placeholder_dict_renders_empty_and_section_skipped(self):
        text = self._with_actions([{'who': 'unspecified', 'deadline': 'n/a'}]).to_text('en')
        assert "Action Items" not in text

    def test_render_item_plain_string_passthrough(self):
        assert _render_item("just text") == "just text"


class TestSummarizeChunkRetry:

    @patch(
        "eden_summary.summarize.summarize.time.sleep")  # не ждать в тесте
    @patch("eden_summary.summarize.summarize.get_llm_cfg")
    @patch("eden_summary.summarize.summarize.completion")
    def test_summarize_chunk_retries_then_succeeds(self, mock_completion, mock_cfg, _sleep):
        mock_completion.side_effect = [
            _fake_response("garbage no json"),
            # attempt 0 -> fail
            _fake_response('{"decisions": ["ok"]}'),
            # attempt 1 -> success
        ]
        mock_cfg.return_value.max_parse_attempts = 3
        result = summarize_chunk("transcript")
        assert result == {"decisions": ["ok"]}
        assert mock_completion.call_count == 2


    @patch("eden_summary.summarize.summarize.time.sleep")
    @patch("eden_summary.summarize.summarize.get_llm_cfg")
    @patch("eden_summary.summarize.summarize.completion")
    def test_summarize_chunk_raises_after_three_failures(self, mock_completion, mock_cfg, _sleep):
        mock_completion.return_value = _fake_response("never valid json")
        mock_cfg.return_value.max_parse_attempts = 3
        with pytest.raises(ValueError):
            summarize_chunk("transcript")
        assert mock_completion.call_count == 3  # 0,1,2 -> stop