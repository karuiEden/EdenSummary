import json

import pytest

from eden_summary.summarize.summarize import (
    Summary,
    _ensure_list,
    _parse_json,
)


class TestParseJson:
    def test_clean_json(self):
        assert _parse_json('{"title": "x"}') == {"title": "x"}

    def test_wrapped_in_markdown_fence(self):
        # модель часто оборачивает ответ в ```json ... ```
        raw = '```json\n{"title": "x", "tldr": ["a"]}\n```'
        assert _parse_json(raw) == {"title": "x", "tldr": ["a"]}

    def test_prose_around_json(self):
        # "Here is the summary: {...}. Hope it helps!"
        raw = 'Sure! Here is the result:\n{"decisions": ["d1"]}\nLet me know.'
        assert _parse_json(raw) == {"decisions": ["d1"]}

    def test_nested_objects(self):
        # rfind('}') должен брать ВНЕШНЮЮ закрывающую скобку, не вложенную
        raw = '{"a": {"b": 1}, "c": 2}'
        assert _parse_json(raw) == {"a": {"b": 1}, "c": 2}

    def test_no_json_raises(self):
        # документируем текущее поведение: пустой/мусорный ответ -> ошибка парсинга
        with pytest.raises(json.JSONDecodeError):
            _parse_json("no json here at all")


class TestEnsureList:
    def test_string_wrapped_to_list(self):
        # модель вернула строку вместо списка
        assert _ensure_list("single item") == ["single item"]

    def test_list_passthrough(self):
        assert _ensure_list(["a", "b"]) == ["a", "b"]

    def test_none_to_empty(self):
        assert _ensure_list(None) == []

    def test_unexpected_type_to_empty(self):
        # число/словарь -> пустой список, а не падение
        assert _ensure_list(42) == []  # type: ignore[arg-type]
        assert _ensure_list({"a": 1}) == []  # type: ignore[arg-type]


class TestSummaryToText:
    def test_empty_sections_skipped(self):
        s = Summary(title="t", tldr=["point"], decisions=[], action_items=[], risks=[])
        text = s.to_text()
        assert "TL;DR" in text
        assert "Решения" not in text  # пустая секция не выводится

    def test_all_empty_yields_empty_string(self):
        s = Summary(title="t", tldr=[], decisions=[], action_items=[], risks=[])
        assert s.to_text() == ""

    def test_title_not_rendered(self):
        # ВАЖНО: to_text() НЕ выводит title — только tldr/decisions/action_items/risks
        s = Summary(title="SECRET_TITLE", tldr=["x"], decisions=[], action_items=[], risks=[])
        assert "SECRET_TITLE" not in s.to_text()