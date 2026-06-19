import re

# Lightweight inverse text normalization (ITN) for spelled-out numbers.
# Meeting transcripts (and inconsistent ASR) often spell numbers out
# ("twenty five euros", "двадцать пять евро") while LLM summaries emit digits
# ("25 euros", "25 евро"). The inline number guard grounds summary digits
# against the transcript, so we rewrite spoken numbers to digits first —
# otherwise every number reads as ungrounded.
#
# Deliberately narrow, not a full ITN. Scale words (hundred/thousand/million,
# тысяча/миллион/миллиард) are left as text, which is exactly what _digit_core
# expects ('fifteen million' -> '15 million' -> core '15').
#
# Precision direction (important, and easy to get backwards): converting a spoken
# number REDUCES false positives versus no normalization. But a coverage GAP — a
# salient number whose spoken form we fail to convert — leaves its digits absent
# from the transcript, so a faithful summary that writes it in digits gets flagged.
# Gaps are therefore FALSE-POSITIVE sources, not false negatives; residual FP is
# bounded by coverage. (Over-conversion, conversely, only adds spurious digits and
# can at most miss a flag — a false negative.) Known gaps that can produce FP:
# English 'a hundred'/year forms ('nineteen eighty four'); Russian numeral
# declension ('двадцати пяти') and the decimal comma ('12,50').

# ---- English ----------------------------------------------------------------

_EN_UNITS = {
    'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
    'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
}
_EN_TEENS = {
    'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
    'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19,
}
_EN_TENS = {
    'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50,
    'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90,
}
_EN_ALL = {**_EN_UNITS, **_EN_TEENS, **_EN_TENS}

_EN_TENS_RE = '|'.join(_EN_TENS)
_EN_UNITS_RE = '|'.join(_EN_UNITS)
_EN_ALL_RE = '|'.join(_EN_ALL)

# 'twenty five' / 'twenty-five' -> 25; matched before standalone words so the
# tens word isn't replaced on its own first.
_EN_COMPOUND = re.compile(rf'\b({_EN_TENS_RE})[-\s]+({_EN_UNITS_RE})\b', re.IGNORECASE)
_EN_SINGLE = re.compile(rf'\b({_EN_ALL_RE})\b', re.IGNORECASE)


def _en_to_digits(text: str) -> str:
    def _compound(match: re.Match) -> str:
        return str(_EN_TENS[match.group(1).lower()] + _EN_UNITS[match.group(2).lower()])

    def _single(match: re.Match) -> str:
        return str(_EN_ALL[match.group(1).lower()])

    return _EN_SINGLE.sub(_single, _EN_COMPOUND.sub(_compound, text))


# ---- Russian ----------------------------------------------------------------

# Each numeral includes its oblique forms (genitive=dative=prepositional for
# most; instrumental where common), because speech declines numbers constantly
# after quantity prepositions ('около двадцати пяти', 'до пятидесяти'). A gap
# would leave the number unconverted and flag a faithful summary (false positive).
# 'один' obliques (одного/одной/одним) are intentionally omitted — they are far
# more common as non-numeric words than as a salient number, and converting them
# would add noise (FN risk only, but pointless); '1' as a salient figure is rare.
_RU_UNITS = {
    'ноль': 0, 'нуля': 0,
    'один': 1, 'одна': 1, 'одно': 1,
    'два': 2, 'две': 2, 'двух': 2, 'двум': 2, 'двумя': 2,
    'три': 3, 'трёх': 3, 'трех': 3, 'трём': 3, 'трем': 3, 'тремя': 3,
    'четыре': 4, 'четырёх': 4, 'четырех': 4, 'четырём': 4, 'четырем': 4, 'четырьмя': 4,
    'пять': 5, 'пяти': 5, 'пятью': 5,
    'шесть': 6, 'шести': 6, 'шестью': 6,
    'семь': 7, 'семи': 7, 'семью': 7,
    'восемь': 8, 'восьми': 8, 'восемью': 8, 'восьмью': 8,
    'девять': 9, 'девяти': 9, 'девятью': 9,
}
_RU_TEENS = {
    'десять': 10, 'десяти': 10, 'десятью': 10,
    'одиннадцать': 11, 'одиннадцати': 11, 'одиннадцатью': 11,
    'двенадцать': 12, 'двенадцати': 12, 'двенадцатью': 12,
    'тринадцать': 13, 'тринадцати': 13, 'тринадцатью': 13,
    'четырнадцать': 14, 'четырнадцати': 14, 'четырнадцатью': 14,
    'пятнадцать': 15, 'пятнадцати': 15, 'пятнадцатью': 15,
    'шестнадцать': 16, 'шестнадцати': 16, 'шестнадцатью': 16,
    'семнадцать': 17, 'семнадцати': 17, 'семнадцатью': 17,
    'восемнадцать': 18, 'восемнадцати': 18, 'восемнадцатью': 18,
    'девятнадцать': 19, 'девятнадцати': 19, 'девятнадцатью': 19,
}
_RU_TENS = {
    'двадцать': 20, 'двадцати': 20, 'двадцатью': 20,
    'тридцать': 30, 'тридцати': 30, 'тридцатью': 30,
    'сорок': 40, 'сорока': 40,
    'пятьдесят': 50, 'пятидесяти': 50, 'пятьюдесятью': 50,
    'шестьдесят': 60, 'шестидесяти': 60, 'шестьюдесятью': 60,
    'семьдесят': 70, 'семидесяти': 70, 'семьюдесятью': 70,
    'восемьдесят': 80, 'восьмидесяти': 80, 'восемьюдесятью': 80, 'восьмьюдесятью': 80,
    'девяносто': 90, 'девяноста': 90,
}
_RU_HUNDREDS = {
    'сто': 100, 'ста': 100,
    'двести': 200, 'двухсот': 200, 'двумстам': 200, 'двумястами': 200, 'двухстах': 200,
    'триста': 300, 'трёхсот': 300, 'трехсот': 300, 'тремястами': 300,
    'четыреста': 400, 'четырёхсот': 400, 'четырехсот': 400, 'четырьмястами': 400,
    'пятьсот': 500, 'пятисот': 500, 'пятьюстами': 500,
    'шестьсот': 600, 'шестисот': 600,
    'семьсот': 700, 'семисот': 700,
    'восемьсот': 800, 'восьмисот': 800,
    'девятьсот': 900, 'девятисот': 900,
}
_RU_VALUES = {**_RU_UNITS, **_RU_TEENS, **_RU_TENS, **_RU_HUNDREDS}


def _alt(words: dict[str, int]) -> str:
    # longest-first so e.g. 'восемьсот' wins over 'восемь', 'пятьдесят' over 'пять'
    return '|'.join(sorted(words, key=len, reverse=True))


_RU_H = _alt(_RU_HUNDREDS)
_RU_TE = _alt(_RU_TEENS)
_RU_TN = _alt(_RU_TENS)
_RU_U = _alt(_RU_UNITS)

# A single well-formed number group, in strictly descending magnitude:
# optional hundreds, then either a teen, or optional-tens + unit, or tens alone;
# plus hundreds-alone. Crucially this does NOT merge same-magnitude neighbours
# ('пять шесть' stays two numbers -> '5 6'), so it can't destroy a standalone
# number and turn grounding into a false positive — only valid compounds like
# 'сто двадцать пять' -> 125 are summed.
_RU_GROUP = re.compile(
    rf'\b(?:'
    rf'(?:(?:{_RU_H})[\s-]+)?(?:(?:{_RU_TE})|(?:(?:{_RU_TN})[\s-]+)?(?:{_RU_U})|(?:{_RU_TN}))'
    rf'|(?:{_RU_H})'
    rf')\b',
    re.IGNORECASE,
)


def _ru_to_digits(text: str) -> str:
    def _group(match: re.Match) -> str:
        words = re.split(r'[\s-]+', match.group(0))
        return str(sum(_RU_VALUES[w.lower()] for w in words if w))

    return _RU_GROUP.sub(_group, text)


# ---- public ------------------------------------------------------------------

def spoken_to_digits(text: str) -> str:
    """Rewrite English and Russian number words into digits ('twenty five' ->
    '25', 'сто двадцать пять' -> '125'), leaving scale words and everything else
    untouched. Case-insensitive, word-boundary safe ('someone'/'однажды' are not
    touched). Both languages are applied unconditionally — a transcript is one
    language, so the other's rules are a no-op."""
    return _ru_to_digits(_en_to_digits(text))
