"""Q3 calibration math. No real edit data exists yet (same as Tier-2 / Russian
ITN) — these are synthetic. The checks are deliberately NOT by-construction:
monotonicity over a grid, hand-computed PAV pooling, and degenerate fallbacks."""
from eden_summary.quality.calibration import Calibration, fit_calibration


def _grid(step: float = 0.05) -> list[float]:
    return [i * step for i in range(int(1 / step) + 1)]


class TestRegimeSelection:
    def test_few_labels_use_logistic(self):
        labels = [(0.9, False), (0.8, False), (0.2, True), (0.1, True)]
        cal = fit_calibration(labels)  # default threshold 200
        assert cal.method == 'logistic'
        assert cal.n == 4

    def test_many_labels_use_isotonic(self):
        labels = [(0.0, False), (1.0, True)]
        cal = fit_calibration(labels, regime_threshold=0)  # force isotonic
        assert cal.method == 'isotonic'


class TestDegenerate:
    def test_empty_is_base_rate_none(self):
        cal = fit_calibration([])
        assert cal.method == 'base_rate'
        assert cal.n == 0
        assert cal.base_rate is None
        assert cal.predict(0.5) == 0.0  # safe default, never crashes

    def test_all_edited_is_base_rate_one(self):
        cal = fit_calibration([(0.9, True), (0.2, True), (0.5, True)])
        assert cal.method == 'base_rate'
        assert cal.base_rate == 1.0
        assert cal.predict(0.99) == 1.0

    def test_none_edited_is_base_rate_zero(self):
        cal = fit_calibration([(0.9, False), (0.2, False)])
        assert cal.method == 'base_rate'
        assert cal.base_rate == 0.0
        assert cal.predict(0.01) == 0.0

    def test_single_point_is_base_rate(self):
        cal = fit_calibration([(0.7, True)])
        assert cal.method == 'base_rate'
        assert cal.base_rate == 1.0


class TestIsotonicPooling:
    # scores 0.9/0.8/0.7 -> t = 0.1/0.2/0.3; labels 1/0/1 over ascending t violate
    # monotonicity. PAV pools the first two (1,0)->0.5 and leaves the last at 1.0.
    LABELS = [(0.9, True), (0.8, False), (0.7, True)]

    def test_knots_match_hand_computed_pav(self):
        cal = fit_calibration(self.LABELS, regime_threshold=0)
        assert cal.method == 'isotonic'
        assert [round(t, 6) for t in cal.knots_t] == [0.1, 0.2, 0.3]  # t = 1 - score
        # pooled: (0.1,0.2)->0.5, (0.3)->1.0
        assert [round(p, 6) for p in cal.knots_p] == [0.5, 0.5, 1.0]

    def test_predict_at_knots(self):
        cal = fit_calibration(self.LABELS, regime_threshold=0)
        assert round(cal.predict(0.9), 6) == 0.5   # t=0.1, clamped to first knot
        assert round(cal.predict(0.8), 6) == 0.5   # t=0.2
        assert round(cal.predict(0.7), 6) == 1.0   # t=0.3, clamped to last knot

    def test_predict_non_increasing_in_score(self):
        cal = fit_calibration(self.LABELS, regime_threshold=0)
        preds = [cal.predict(s) for s in _grid()]
        # higher faithfulness score -> never higher P(edit)
        assert all(a >= b - 1e-9 for a, b in zip(preds, preds[1:]))


class TestLogisticMonotone:
    # clearly downward-trending: high score rarely edited, low score usually edited
    LABELS = (
        [(0.95, False)] * 8 + [(0.9, False)] * 6 + [(0.85, True)] * 2
        + [(0.2, True)] * 7 + [(0.1, True)] * 8 + [(0.15, False)] * 1
    )

    def test_method_and_bounds(self):
        cal = fit_calibration(self.LABELS)
        assert cal.method == 'logistic'
        for s in _grid():
            p = cal.predict(s)
            assert 0.0 < p < 1.0

    def test_non_increasing_in_score(self):
        cal = fit_calibration(self.LABELS)
        preds = [cal.predict(s) for s in _grid()]
        assert all(a >= b - 1e-9 for a, b in zip(preds, preds[1:]))

    def test_high_score_less_likely_edited_than_low(self):
        cal = fit_calibration(self.LABELS)
        assert cal.predict(0.95) < cal.predict(0.1)


class TestMetadata:
    def test_to_metadata_roundtrips_fields(self):
        cal = fit_calibration([(0.9, True), (0.8, False), (0.7, True)], regime_threshold=0)
        meta = cal.to_metadata()
        assert meta['method'] == 'isotonic'
        assert [round(t, 6) for t in meta['knots_t']] == [0.1, 0.2, 0.3]
        assert isinstance(cal, Calibration)
