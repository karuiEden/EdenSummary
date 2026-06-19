from eden_summary.quality.numbers import spoken_to_digits


class TestSpokenToDigits:
    def test_unit(self):
        assert spoken_to_digits("we need five units") == "we need 5 units"

    def test_teen(self):
        assert spoken_to_digits("fifteen people") == "15 people"

    def test_round_ten(self):
        assert spoken_to_digits("forty percent") == "40 percent"

    def test_compound_with_space(self):
        assert spoken_to_digits("twenty five euros") == "25 euros"

    def test_compound_with_hyphen(self):
        assert spoken_to_digits("twenty-five euros") == "25 euros"

    def test_compound_percent(self):
        assert spoken_to_digits("thirty four percent") == "34 percent"

    def test_scale_word_left_as_text(self):
        # 'million' stays so _digit_core still yields '15'
        assert spoken_to_digits("fifteen million euro") == "15 million euro"

    def test_case_insensitive(self):
        assert spoken_to_digits("Twenty Five Euros") == "25 Euros"

    def test_word_boundary_safe(self):
        # number words embedded in larger words must not be touched
        assert spoken_to_digits("someone often listens") == "someone often listens"

    def test_no_number_words_unchanged(self):
        assert spoken_to_digits("the remote control design") == "the remote control design"

    def test_multiple_numbers_in_sentence(self):
        assert spoken_to_digits("from ten to forty people") == "from 10 to 40 people"

    def test_ami_style_sentence(self):
        # mirrors the real IS1009b transcript phrasing
        out = spoken_to_digits("Twenty five Euros makes a nice little present")
        assert out == "25 Euros makes a nice little present"


class TestSpokenToDigitsRussian:
    def test_unit_and_gender_forms(self):
        assert spoken_to_digits("один два две одна") == "1 2 2 1"

    def test_teen(self):
        assert spoken_to_digits("пятнадцать человек") == "15 человек"

    def test_round_ten(self):
        assert spoken_to_digits("пятьдесят процентов") == "50 процентов"

    def test_compound_tens_unit(self):
        assert spoken_to_digits("двадцать пять евро") == "25 евро"

    def test_compound_percent(self):
        assert spoken_to_digits("тридцать четыре процента") == "34 процента"

    def test_hundred(self):
        assert spoken_to_digits("сто") == "100"

    def test_hundred_tens_unit_run(self):
        assert spoken_to_digits("сто двадцать пять") == "125"

    def test_two_hundreds_compound(self):
        assert spoken_to_digits("двести пятьдесят") == "250"

    def test_scale_word_left_as_text(self):
        # 'миллионов' stays so _digit_core still yields '50'
        assert spoken_to_digits("пятьдесят миллионов евро") == "50 миллионов евро"

    def test_keeps_surrounding_text(self):
        assert spoken_to_digits("бюджет двадцать пять евро") == "бюджет 25 евро"

    def test_word_boundary_safe(self):
        # value words embedded in larger words must not be touched
        assert spoken_to_digits("однажды просто на столе") == "однажды просто на столе"

    def test_case_insensitive(self):
        assert spoken_to_digits("Двадцать Пять") == "25"


class TestSpokenToDigitsRussianDeclined:
    # natural declined speech (genitive/instrumental), NOT nominative — the forms
    # that actually appear after quantity prepositions in real meetings.
    def test_genitive_compound_after_preposition(self):
        assert spoken_to_digits("около двадцати пяти евро") == "около 25 евро"

    def test_genitive_tens(self):
        assert spoken_to_digits("до пятидесяти процентов") == "до 50 процентов"

    def test_genitive_tens_with_scale(self):
        assert spoken_to_digits("порядка пятидесяти миллионов") == "порядка 50 миллионов"

    def test_genitive_hundred_tens_unit_run(self):
        assert spoken_to_digits("свыше ста двадцати пяти тысяч") == "свыше 125 тысяч"

    def test_sorok_genitive(self):
        assert spoken_to_digits("сорока евро") == "40 евро"

    def test_sto_genitive(self):
        assert spoken_to_digits("ста рублей") == "100 рублей"

    def test_hundreds_genitive(self):
        assert spoken_to_digits("двухсот евро") == "200 евро"

    def test_instrumental_compound(self):
        assert spoken_to_digits("с двадцатью пятью процентами") == "с 25 процентами"
