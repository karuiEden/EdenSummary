from unittest.mock import patch, MagicMock

import pytest

from eden_summary.summarize.summarize import _parse_json, _ensure_list, Summary, summarize_chunk, _render_item, _extract_numbers, summarize_transcript, _estimate_tokens


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
        # the LLM sometimes returns a bare string instead of a list — wrap it
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
        # documents current behaviour: the title is not rendered by to_text()
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
        # unknown language → DEFAULT_LOCALE ('ru')
        assert 'Решения' in text

    def test_to_text_none_lang_falls_back_to_default(self):
        s = self._full()
        assert s.to_text(None) == s.to_text()


class TestStructuredActionItems:
    # the LLM returns action_items as dicts {task, who, deadline} — to_text must
    # render them into a readable line, dropping placeholders ('unspecified', etc.)
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


class TestExtractNumbers:
    # real cases from the AMI IS1000a transcript
    def test_currency_word(self):
        assert _extract_numbers("selling price will be about 25 euros each") == ["25 euros"]

    def test_decimal_currency(self):
        assert _extract_numbers("production cost is 12.50 euros") == ["12.50 euros"]

    def test_magnitude_with_currency(self):
        assert _extract_numbers("profit aim, about 15 million euro") == ["15 million euro"]

    def test_magnitude_bare(self):
        assert _extract_numbers("we have to sell at least 4 million") == ["4 million"]

    def test_currency_symbol(self):
        assert _extract_numbers("it costs €25 total") == ["€25"]

    def test_percentage(self):
        assert _extract_numbers("margin of 50% and 20 percent growth") == ["50%", "20 percent"]

    def test_comma_grouped_and_time(self):
        assert _extract_numbers("1,000 units by 10:30") == ["1,000", "10:30"]

    def test_decimal_not_split_from_currency(self):
        # "12.50 euros" is one fact, not "12.50" + "12.50 euros"
        assert _extract_numbers("12.50 euros") == ["12.50 euros"]

    def test_dedup_preserves_first(self):
        assert _extract_numbers("25 euros now, still 25 euros later") == ["25 euros"]

    def test_order_preserved(self):
        assert _extract_numbers("first 25 euros then 15 million") == ["25 euros", "15 million"]

    def test_no_numbers(self):
        assert _extract_numbers("no figures were mentioned here") == []


class TestExtractNumbersRussian:
    # Russian formats for the inline guard — additive to the English patterns
    def test_currency_euro(self):
        assert _extract_numbers("цена будет 25 евро за штуку") == ["25 евро"]

    def test_currency_rubles(self):
        assert _extract_numbers("стоит 1000 рублей") == ["1000 рублей"]

    def test_magnitude_with_currency(self):
        assert _extract_numbers("прибыль около 15 миллионов евро") == ["15 миллионов евро"]

    def test_magnitude_bare(self):
        assert _extract_numbers("бюджет 50 миллионов") == ["50 миллионов"]

    def test_percent_word(self):
        assert _extract_numbers("маржа 34 процента и рост 20 процентов") == ["34 процента", "20 процентов"]

    def test_no_numbers(self):
        assert _extract_numbers("цифры не назывались на встрече") == []


class TestExtractNumbersParked:
    # _extract_numbers is read-only here and must NOT be injected into prompts.
    @patch("eden_summary.summarize.summarize.get_llm_cfg")
    @patch("eden_summary.summarize.summarize.completion")
    def test_chunk_prompt_has_no_numeric_anchoring(self, mock_completion, mock_cfg):
        mock_completion.return_value = _fake_response('{"decisions": []}')
        mock_cfg.return_value.max_parse_attempts = 3
        summarize_chunk("the price is 25 euros and profit 15 million")
        sent = mock_completion.call_args.kwargs["messages"][1]["content"]
        assert "Numeric facts" not in sent
        assert "use ONLY" not in sent.lower() and "copy a value" not in sent


class TestSummarizeTranscriptRouting:
    # summarize_transcript: single pass when the transcript fits, map-reduce otherwise.
    @patch("eden_summary.summarize.summarize.get_llm_cfg")
    @patch("eden_summary.summarize.summarize.completion")
    def test_short_transcript_uses_single_pass(self, mock_completion, mock_cfg):
        mock_completion.return_value = _fake_response(
            '{"title":"t","tldr":[],"decisions":["d"],"action_items":[],"risks":[]}')
        cfg = mock_cfg.return_value
        cfg.max_parse_attempts = 3
        cfg.single_pass_token_limit = 32000
        result = summarize_transcript(["a short meeting transcript"])
        assert mock_completion.call_count == 1  # one call, no map-reduce
        sent = mock_completion.call_args.kwargs["messages"][1]["content"]
        assert "complete meeting transcript" in sent  # SINGLE_PASS_PROMPT used
        assert result.decisions == ["d"]

    @patch("eden_summary.summarize.summarize.chunk_segments")
    @patch("eden_summary.summarize.summarize.get_llm_cfg")
    @patch("eden_summary.summarize.summarize.completion")
    def test_long_transcript_falls_back_to_map_reduce(self, mock_completion, mock_cfg, mock_chunk):
        mock_completion.return_value = _fake_response(
            '{"title":"t","tldr":[],"decisions":[],"action_items":[],"risks":[]}')
        cfg = mock_cfg.return_value
        cfg.max_parse_attempts = 3
        cfg.max_workers = 1
        cfg.single_pass_token_limit = 1  # force the map-reduce fallback
        mock_chunk.return_value = ["chunk one", "chunk two"]
        summarize_transcript(["some long transcript text well over the limit"])
        assert mock_completion.call_count == 3  # 2 map + 1 reduce
        mock_chunk.assert_called_once()
        all_sent = " ".join(c.kwargs["messages"][1]["content"] for c in mock_completion.call_args_list)
        assert "Pay equal attention" not in all_sent  # single-pass prompt not used

    def test_estimate_tokens_roughly_quarter_of_chars(self):
        assert _estimate_tokens("a" * 400) == 100


class TestJsonMode:
    # native JSON output: response_format is sent when json_mode is on, omitted when off.
    @patch("eden_summary.summarize.summarize.get_llm_cfg")
    @patch("eden_summary.summarize.summarize.completion")
    def test_json_mode_on_passes_response_format(self, mock_completion, mock_cfg):
        mock_completion.return_value = _fake_response('{"decisions": []}')
        cfg = mock_cfg.return_value
        cfg.max_parse_attempts = 3
        cfg.json_mode = True
        summarize_chunk("some transcript")
        assert mock_completion.call_args.kwargs["response_format"] == {"type": "json_object"}

    @patch("eden_summary.summarize.summarize.get_llm_cfg")
    @patch("eden_summary.summarize.summarize.completion")
    def test_json_mode_off_omits_response_format(self, mock_completion, mock_cfg):
        mock_completion.return_value = _fake_response('{"decisions": []}')
        cfg = mock_cfg.return_value
        cfg.max_parse_attempts = 3
        cfg.json_mode = False
        summarize_chunk("some transcript")
        assert "response_format" not in mock_completion.call_args.kwargs


class TestSummarizeChunkRetry:

    @patch(
        "eden_summary.summarize.summarize.time.sleep")  # don't sleep in the test
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