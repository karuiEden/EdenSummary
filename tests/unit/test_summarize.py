from eden_summary.summarize.summarize import _parse_json, _ensure_list, Summary


class TestParseJson:
    def test_plain_json(self):
        assert _parse_json('{"a": 1}') == {"a": 1}

    def test_strips_surrounding_text(self):
        assert _parse_json('Here you go: {"a": 1} done') == {"a": 1}

    def test_strips_markdown_fences(self):
        assert _parse_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_handles_nested_objects(self):
        assert _parse_json('{"a": {"b": 2}}') == {"a": {"b": 2}}


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