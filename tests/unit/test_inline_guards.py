from unittest.mock import patch

from eden_summary.quality.inline_guards import (
    run_inline_guards, _check_numbers, _digit_core, _is_grounded,
    QualityFlag, InlineGuardResult,
)
from eden_summary.summarize.summarize import Summary


def _summary(**fields) -> Summary:
    base = dict(title="T", tldr=[], decisions=[], action_items=[], risks=[])
    base.update(fields)
    return Summary(**base)


class TestDigitCore:
    def test_currency_symbol_stripped(self):
        assert _digit_core("€25") == "25"

    def test_grouping_comma_removed(self):
        assert _digit_core("1,000") == "1000"

    def test_magnitude_word_dropped(self):
        assert _digit_core("15 million") == "15"

    def test_decimal_kept(self):
        assert _digit_core("12.50 euros") == "12.50"

    def test_time_kept(self):
        assert _digit_core("10:30") == "10:30"

    def test_percent_dropped(self):
        assert _digit_core("50%") == "50"

    def test_russian_decimal_comma_grounds_on_integer_part(self):
        assert _digit_core("12,50 евро") == "12"

    def test_english_grouping_comma_preserved(self):
        assert _digit_core("1,000") == "1000"

    def test_dot_decimal_preserved(self):
        assert _digit_core("12.50 euros") == "12.50"


class TestIsGrounded:
    def test_standalone_match(self):
        assert _is_grounded("25", "the budget is 25 euros") is True

    def test_not_embedded_in_longer_number(self):
        # '50' must not be considered grounded by '509'
        assert _is_grounded("50", "there were 509 attendees") is False

    def test_empty_core_is_grounded(self):
        # nothing to check → don't flag
        assert _is_grounded("", "anything at all") is True


class TestCheckNumbers:
    def test_grounded_number_no_flag(self):
        flags = _check_numbers("- budget set at 25 euros", "we agreed the budget is 25 euros total")
        assert flags == []

    def test_fabricated_number_flagged(self):
        flags = _check_numbers("- profit target is 80 million", "no financials were discussed")
        assert len(flags) == 1
        assert flags[0].kind == "ungrounded_number"
        assert flags[0].value == "80 million"

    def test_currency_surface_matches_bare_in_transcript(self):
        # summary writes '€25', transcript says '25 euros' — same value, no flag
        flags = _check_numbers("- price €25 each", "it was priced at 25 euros")
        assert flags == []

    def test_comma_grouping_matches_plain_digits(self):
        flags = _check_numbers("- order of 1,000 units", "we ordered 1000 units")
        assert flags == []

    def test_salient_number_not_grounded_by_longer_run(self):
        # '50 million' (core '50') must not be treated as grounded by '509'
        flags = _check_numbers("- we need 50 million in funding", "there were 509 attendees, no funding")
        assert len(flags) == 1
        assert flags[0].value == "50 million"

    def test_bare_integer_is_not_checked(self):
        # _extract_numbers only captures salient numbers (money/magnitude/%/time/
        # decimal); a plain '50 units' is intentionally out of scope, so no flag.
        assert _check_numbers("- we need 50 units", "no quantities mentioned") == []

    def test_spelled_out_transcript_grounds_digit_summary(self):
        # AMI-style: transcript spells the number out, summary uses digits.
        # ITN normalization must ground it so it is NOT flagged.
        flags = _check_numbers(
            "- priced at 25 euros",
            "Twenty five Euros makes a nice little present",
        )
        assert flags == []

    def test_spelled_out_percent_grounds_digit_summary(self):
        flags = _check_numbers(
            "- 34% found it hard to learn",
            "and that was thirty four percent but even more important",
        )
        assert flags == []

    def test_number_absent_from_spoken_transcript_is_flagged(self):
        flags = _check_numbers(
            "- profit target is 80 million",
            "Twenty five Euros makes a nice little present",
        )
        assert len(flags) == 1
        assert flags[0].value == "80 million"


class TestRunInlineGuards:
    def test_clean_summary_passes(self):
        result = run_inline_guards(_summary(decisions=["ship the product"]), "we will ship the product", "en-US")
        assert result.passed is True
        assert result.flags == []
        assert result.checked == ("numbers",)

    def test_fabricated_number_in_decisions_flagged(self):
        result = run_inline_guards(
            _summary(decisions=["budget approved at 80 million"]),
            "the team met but no budget figure was named",
            "en",
        )
        assert result.passed is False
        assert result.flags[0].value == "80 million"

    def test_russian_faithful_summary_passes(self):
        # numbers spoken in the transcript (двадцать пять / тридцать четыре)
        # ground the digit summary after ITN → no flags.
        result = run_inline_guards(
            _summary(decisions=["цена 25 евро", "одобрили 34%"]),
            "продаём за двадцать пять евро, тридцать четыре процента нашли это сложным",
            "ru-RU",
        )
        assert result.passed is True
        assert result.checked == ("numbers",)

    def test_russian_fabricated_number_flagged(self):
        result = run_inline_guards(
            _summary(decisions=["бюджет 80 миллионов евро"]),
            "на встрече никаких сумм не называли",
            "ru",
        )
        assert result.passed is False
        assert result.flags[0].value == "80 миллионов евро"

    def test_russian_declined_number_grounds(self):
        # was a false positive before declension support: faithful '25 евро'
        # spoken as the genitive 'двадцати пяти' must NOT be flagged.
        result = run_inline_guards(
            _summary(decisions=["цена 25 евро"]),
            "договорились около двадцати пяти евро",
            "ru",
        )
        assert result.passed is True

    def test_russian_decimal_comma_grounds(self):
        # was a false positive: '12,50 евро' grounds on the integer part '12'.
        result = run_inline_guards(
            _summary(decisions=["цена 12,50 евро"]),
            "двенадцать евро пятьдесят центов",
            "ru",
        )
        assert result.passed is True

    def test_russian_decimal_comma_grounds_against_digit_transcript(self):
        # the case the spelled-out test missed: transcript renders the decimal in
        # digits (as ASR does). Both sides must collapse '12,50'->'12' symmetrically,
        # otherwise an identical faithful number is flagged.
        result = run_inline_guards(
            _summary(decisions=["цена 12,50 евро"]),
            "договорились на 12,50 евро",
            "ru",
        )
        assert result.passed is True

    def test_english_grouping_comma_grounds_against_digit_transcript(self):
        # symmetry must not break English grouping: '1,000' vs '1,000'.
        result = run_inline_guards(
            _summary(decisions=["order of 1,000 units"]),
            "we agreed on 1,000 units",
            "en",
        )
        assert result.passed is True

    def test_russian_fabrication_still_caught_amid_declined_text(self):
        result = run_inline_guards(
            _summary(decisions=["бюджет 80 миллионов"]),
            "обсудили около двадцати пяти евро, сумм больше не называли",
            "ru",
        )
        assert result.passed is False
        assert result.flags[0].value == "80 миллионов"

    def test_unsupported_language_skips_number_check(self):
        # A language we can't normalize (no word->digit) skips rather than flag
        # every number (~100% FP). 'numbers' absent from `checked`: not verified,
        # not "verified clean".
        result = run_inline_guards(
            _summary(decisions=["budget 80 million"]),
            "no figures were named",
            "de-DE",
        )
        assert result.flags == []
        assert result.checked == ()

    def test_unknown_language_skips_number_check(self):
        # language=None (ASR didn't detect) → skip, don't guess.
        result = run_inline_guards(_summary(decisions=["budget at 80 million"]), "no figures", None)
        assert result.flags == []
        assert result.checked == ()

    def test_never_raises_on_internal_error(self):
        with patch("eden_summary.quality.inline_guards._check_numbers", side_effect=RuntimeError("boom")):
            result = run_inline_guards(_summary(decisions=["x"]), "transcript", "en")
        assert result.flags == []
        assert result.checked == ()  # records that no check ran
        assert result.passed is True


class TestToMetadata:
    def test_clean_metadata(self):
        meta = InlineGuardResult(flags=[], checked=("numbers",)).to_metadata()
        assert meta == {"passed": True, "checked": ["numbers"], "flags": []}

    def test_metadata_with_flag(self):
        flag = QualityFlag(kind="ungrounded_number", severity="warning", detail="d", value="80 million")
        meta = InlineGuardResult(flags=[flag], checked=("numbers",)).to_metadata()
        assert meta["passed"] is False
        assert meta["checked"] == ["numbers"]
        assert meta["flags"][0]["value"] == "80 million"
        assert meta["flags"][0]["kind"] == "ungrounded_number"

    def test_failed_run_metadata_distinguishable(self):
        # checked == [] means 'guards did not run', vs [] flags with checked set
        meta = InlineGuardResult(flags=[], checked=()).to_metadata()
        assert meta["checked"] == []