from unittest.mock import patch, MagicMock

from eden_summary.quality.faithfulness import (
    judge_faithfulness, _collect_claims, _scores, ClaimVerdict, JudgeResult,
)
from eden_summary.summarize.summarize import Summary


def _summary(**fields) -> Summary:
    base = dict(title="T", tldr=[], decisions=[], action_items=[], risks=[])
    base.update(fields)
    return Summary(**base)


def _fake_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.choices[0].message.content = content
    return resp


class TestCollectClaims:
    def test_order_risks_first(self):
        s = _summary(tldr=["t"], decisions=["d"], action_items=["a"], risks=["r"])
        assert _collect_claims(s) == [("risks", "r"), ("decisions", "d"),
                                      ("action_items", "a"), ("tldr", "t")]

    def test_skips_empty_and_placeholder(self):
        s = _summary(decisions=["real", ""], action_items=[{"who": "unspecified"}])
        assert _collect_claims(s) == [("decisions", "real")]

    def test_title_not_graded(self):
        s = _summary(title="A long descriptive title", decisions=["d"])
        assert _collect_claims(s) == [("decisions", "d")]


class TestScores:
    def test_empty(self):
        assert _scores([]) == (None, {})

    def test_overall_and_per_field(self):
        verdicts = [
            ClaimVerdict("risks", "r1", "supported", 1.0, "q"),
            ClaimVerdict("risks", "r2", "unsupported", 0.0, ""),
            ClaimVerdict("decisions", "d1", "partial", 0.5, "q"),
        ]
        overall, fields = _scores(verdicts)
        assert overall == 0.5
        assert fields == {"risks": 0.5, "decisions": 0.5}


class TestJudgeFaithfulness:
    @patch("eden_summary.quality.faithfulness.get_llm_cfg")
    def test_no_claims_skips_without_call(self, mock_cfg):
        with patch("eden_summary.quality.faithfulness._judge_complete") as mock_complete:
            result = judge_faithfulness(_summary(), "some transcript")
        assert result.evaluated is False
        assert "no claims" in result.reason
        mock_complete.assert_not_called()

    @patch("eden_summary.quality.faithfulness.get_llm_cfg")
    def test_too_long_skips_without_call(self, mock_cfg):
        mock_cfg.return_value.judge_token_limit = 1
        with patch("eden_summary.quality.faithfulness._judge_complete") as mock_complete:
            result = judge_faithfulness(_summary(decisions=["d"]), "x" * 4000)
        assert result.evaluated is False
        assert "too long" in result.reason
        mock_complete.assert_not_called()

    @patch("eden_summary.quality.faithfulness._judge_complete")
    @patch("eden_summary.quality.faithfulness.get_llm_cfg")
    def test_one_call_with_full_transcript_and_all_claims(self, mock_cfg, mock_complete):
        mock_cfg.return_value.judge_token_limit = 32000
        mock_complete.return_value = (
            '{"claims": [{"id": 0, "verdict": "supported", "evidence": "q"},'
            ' {"id": 1, "verdict": "unsupported", "evidence": ""}]}'
        )
        s = _summary(decisions=["budget 25 euros"], risks=["fabricated 80M"])
        result = judge_faithfulness(s, "we agreed 25 euros")
        assert mock_complete.call_count == 1  # ONE call, transcript sent once
        sent = mock_complete.call_args.args[0][1]["content"]
        assert "we agreed 25 euros" in sent
        assert "budget 25 euros" in sent and "fabricated 80M" in sent
        # risks first (id 0), decisions second (id 1)
        assert result.claims[0].field == "risks"
        assert result.claims[0].verdict == "supported"
        assert result.claims[1].verdict == "unsupported"
        assert result.overall_score == 0.5

    @patch("eden_summary.quality.faithfulness._judge_complete")
    @patch("eden_summary.quality.faithfulness.get_llm_cfg")
    def test_missing_verdict_defaults_unsupported(self, mock_cfg, mock_complete):
        # judge omitted claim id 1 → it must default to unsupported, not vanish
        mock_cfg.return_value.judge_token_limit = 32000
        mock_complete.return_value = '{"claims": [{"id": 0, "verdict": "supported", "evidence": "q"}]}'
        s = _summary(decisions=["a", "b"])
        result = judge_faithfulness(s, "transcript")
        assert len(result.claims) == 2
        assert result.claims[1].verdict == "unsupported"

    @patch("eden_summary.quality.faithfulness._judge_complete")
    @patch("eden_summary.quality.faithfulness.get_llm_cfg")
    def test_unknown_verdict_string_coerced(self, mock_cfg, mock_complete):
        mock_cfg.return_value.judge_token_limit = 32000
        mock_complete.return_value = '{"claims": [{"id": 0, "verdict": "maybe", "evidence": ""}]}'
        result = judge_faithfulness(_summary(decisions=["d"]), "transcript")
        assert result.claims[0].verdict == "unsupported"


class TestToMetadata:
    def test_roundtrip_shape(self):
        result = JudgeResult(
            evaluated=True,
            claims=[ClaimVerdict("risks", "r", "supported", 1.0, "q")],
            overall_score=1.0,
            field_scores={"risks": 1.0},
        )
        meta = result.to_metadata()
        assert meta["evaluated"] is True
        assert meta["overall_score"] == 1.0
        assert meta["field_scores"] == {"risks": 1.0}
        assert meta["claims"][0]["verdict"] == "supported"

    def test_skipped_shape(self):
        meta = JudgeResult(evaluated=False, reason="too long").to_metadata()
        assert meta["evaluated"] is False
        assert meta["reason"] == "too long"
        assert meta["claims"] == []
