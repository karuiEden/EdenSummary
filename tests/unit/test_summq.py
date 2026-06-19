"""Q4 SummQ QA-consistency. No real meeting-edit data exists yet — these are
synthetic, but the checks are NOT by-construction: the deterministic compare is
exercised on the exact failure mode the metric exists to catch (a fabricated
number must read as inconsistent), and the two-call scorer is driven with mocked
LLM output so the blind-answer + compare wiring is verified end to end."""
from unittest.mock import patch

from eden_summary.quality.summq import (
    SummQItem, SummQResult, _answer_match, _has_claims, verify_summq_consistency,
)
from eden_summary.summarize.summarize import Summary


def _summary(**fields) -> Summary:
    base = dict(title="T", tldr=[], decisions=[], action_items=[], risks=[])
    base.update(fields)
    return Summary(**base)


class TestAnswerMatch:
    def test_exact(self):
        assert _answer_match("25 euros", "25 euros") is True

    def test_case_and_punctuation_insensitive(self):
        assert _answer_match("Q3 Budget.", "q3 budget") is True

    def test_summary_answer_subset_of_transcript(self):
        # transcript gives a fuller answer than the summary's short form
        assert _answer_match("John", "John Smith said yes") is True

    def test_transcript_answer_subset_of_summary(self):
        assert _answer_match("25 euros", "25") is True

    def test_not_stated_is_mismatch(self):
        # the summary asserted something the transcript does not support
        assert _answer_match("25 euros", "NOT_STATED") is False

    def test_empty_transcript_answer_is_mismatch(self):
        assert _answer_match("anything", "") is False

    def test_fabricated_number_is_mismatch_despite_scale_word_overlap(self):
        # the whole point of the metric: 80M vs 50M must NOT pass on shared 'million'
        assert _answer_match("80 million", "50 million") is False

    def test_fabricated_number_vs_not_stated(self):
        assert _answer_match("80 million", "NOT_STATED") is False

    def test_spoken_number_folds_to_digits(self):
        assert _answer_match("15 million", "fifteen million") is True

    def test_grouping_comma_normalized(self):
        assert _answer_match("1,000", "1000") is True

    def test_disjoint_names_mismatch(self):
        assert _answer_match("Alice", "Bob") is False


class TestHasClaims:
    def test_empty_summary_has_no_claims(self):
        assert _has_claims(_summary()) is False

    def test_title_only_has_no_claims(self):
        assert _has_claims(_summary(title="A long title")) is False

    def test_empty_and_whitespace_items_do_not_count(self):
        assert _has_claims(_summary(decisions=["", "   "])) is False

    def test_placeholder_only_dict_does_not_count(self):
        # _render_item collapses a placeholder-only dict to '' (same as the Tier-2
        # judge's _collect_claims), so it is not a checkable claim
        assert _has_claims(_summary(action_items=[{"who": "unspecified"}])) is False

    def test_real_item_counts(self):
        assert _has_claims(_summary(risks=["budget overrun"])) is True


def _fake_cfg(mock_cfg, *, threshold=0.7, max_q=8, token_limit=32000):
    cfg = mock_cfg.return_value
    cfg.summq_consistency_threshold = threshold
    cfg.summq_max_questions = max_q
    cfg.judge_token_limit = token_limit
    return cfg


class TestGuards:
    @patch("eden_summary.quality.summq.get_llm_cfg")
    def test_no_claims_skips_without_call(self, mock_cfg):
        _fake_cfg(mock_cfg)
        with patch("eden_summary.quality.summq._summq_complete") as mock_complete:
            result = verify_summq_consistency(_summary(), "transcript")
        assert result.evaluated is False
        assert "no claims" in result.reason
        mock_complete.assert_not_called()

    @patch("eden_summary.quality.summq.get_llm_cfg")
    def test_too_long_skips_without_call(self, mock_cfg):
        _fake_cfg(mock_cfg, token_limit=1)
        with patch("eden_summary.quality.summq._summq_complete") as mock_complete:
            result = verify_summq_consistency(_summary(decisions=["d"]), "x" * 4000)
        assert result.evaluated is False
        assert "too long" in result.reason
        mock_complete.assert_not_called()

    @patch("eden_summary.quality.summq._summq_complete")
    @patch("eden_summary.quality.summq.get_llm_cfg")
    def test_no_questions_generated_skips_answer_call(self, mock_cfg, mock_complete):
        _fake_cfg(mock_cfg)
        mock_complete.return_value = '{"questions": []}'
        result = verify_summq_consistency(_summary(decisions=["d"]), "transcript")
        assert result.evaluated is False
        assert "no questions" in result.reason
        assert mock_complete.call_count == 1  # only the generation call, no answering


class TestScoring:
    GEN = ('{"questions": ['
           '{"question": "How much was the budget?", "answer": "25 euros"},'
           '{"question": "What is the projected profit?", "answer": "80 million"}'
           ']}')
    # answered blind from the transcript: first agrees, second is unsupported
    ANSWER = '{"answers": [{"id": 0, "answer": "25 euros"}, {"id": 1, "answer": "NOT_STATED"}]}'

    @patch("eden_summary.quality.summq._summq_complete")
    @patch("eden_summary.quality.summq.get_llm_cfg")
    def test_two_calls_and_consistency(self, mock_cfg, mock_complete):
        _fake_cfg(mock_cfg)
        mock_complete.side_effect = [self.GEN, self.ANSWER]
        s = _summary(decisions=["budget 25 euros"], risks=["projected profit 80 million"])
        result = verify_summq_consistency(s, "we agreed 25 euros for the venue")
        assert mock_complete.call_count == 2  # generate, then answer
        assert result.evaluated is True
        assert result.n_questions == 2
        assert result.n_consistent == 1
        assert result.consistency_score == 0.5
        assert result.items[0].consistent is True
        assert result.items[1].consistent is False

    @patch("eden_summary.quality.summq._summq_complete")
    @patch("eden_summary.quality.summq.get_llm_cfg")
    def test_answer_call_never_sees_reference_answer(self, mock_cfg, mock_complete):
        # blind answering is the design — the summary's answer must not leak into
        # the transcript-answering prompt, or the metric just confirms itself.
        _fake_cfg(mock_cfg)
        mock_complete.side_effect = [self.GEN, self.ANSWER]
        verify_summq_consistency(_summary(decisions=["d"]), "transcript")
        answer_prompt = mock_complete.call_args_list[1].args[0][1]["content"]
        assert "How much was the budget?" in answer_prompt
        assert "25 euros" not in answer_prompt
        assert "80 million" not in answer_prompt

    @patch("eden_summary.quality.summq._summq_complete")
    @patch("eden_summary.quality.summq.get_llm_cfg")
    def test_max_questions_caps_generation(self, mock_cfg, mock_complete):
        _fake_cfg(mock_cfg, max_q=1)
        gen = ('{"questions": ['
               '{"question": "q1", "answer": "a1"}, {"question": "q2", "answer": "a2"}]}')
        mock_complete.side_effect = [gen, '{"answers": [{"id": 0, "answer": "a1"}]}']
        result = verify_summq_consistency(_summary(decisions=["d"]), "transcript")
        assert result.n_questions == 1


class TestMetadata:
    def test_below_threshold_flag(self):
        result = SummQResult(
            evaluated=True,
            items=[SummQItem("q", "a", "b", False)],
            consistency_score=0.5,
            n_questions=2,
            n_consistent=1,
            threshold=0.7,
        )
        meta = result.to_metadata()
        assert meta["consistency_score"] == 0.5
        assert meta["threshold"] == 0.7
        assert meta["below_threshold"] is True
        assert meta["n_questions"] == 2
        assert meta["items"][0]["consistent"] is False

    def test_above_threshold_not_flagged(self):
        meta = SummQResult(evaluated=True, consistency_score=0.9, threshold=0.7).to_metadata()
        assert meta["below_threshold"] is False

    def test_skipped_has_no_false_flag(self):
        # evaluated=False (no score) must not read as below_threshold
        meta = SummQResult(evaluated=False, reason="no claims to quiz").to_metadata()
        assert meta["evaluated"] is False
        assert meta["below_threshold"] is False
        assert meta["items"] == []
