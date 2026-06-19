"""Q3 edit-calibration: map a Tier-2 judge field score to P(edit) — the
probability that a human will correct that field.

Pure functions, no sklearn/numpy. The judge score is a faithfulness signal in
[0, 1]; we want an actionable probability of human editing. We expect P(edit) to
be NON-INCREASING in the judge score (more faithful → less editing), so we fit on
the risk feature t = 1 - score, where the relationship is non-decreasing, and map
back in `predict`.

Finite-Calibration Regime Map: with few labels a full isotonic fit overfits the
noise, so below `_REGIME_THRESHOLD` we use a 2-parameter logistic (Platt-style,
monotone) and only switch to isotonic (PAV) once there is enough data. Degenerate
inputs (no labels, a single class) fall back to a base rate.

v1 is compute-and-log only: there is NO live apply-path. Until real user edits
exist, every fit here is on synthetic data — see docs/ml-experiments.md.
"""
import logging
import math
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Below this many labels, prefer the low-parameter logistic over isotonic.
_REGIME_THRESHOLD = 200


@dataclass(frozen=True)
class Calibration:
    method: str                 # 'logistic' | 'isotonic' | 'base_rate'
    n: int                      # number of labels the fit was built from
    a: float | None = None      # logistic slope on t = 1 - score
    b: float | None = None      # logistic intercept
    knots_t: tuple[float, ...] = field(default_factory=tuple)   # isotonic breakpoints (t asc)
    knots_p: tuple[float, ...] = field(default_factory=tuple)   # isotonic fitted P(edit)
    base_rate: float | None = None  # constant fallback

    def predict(self, score: float) -> float:
        """P(edit) for a judge field score. Non-increasing in `score` by
        construction (we model the risk feature t = 1 - score)."""
        t = 1.0 - score
        if self.method == 'logistic' and self.a is not None and self.b is not None:
            return _sigmoid(self.a * t + self.b)
        if self.method == 'isotonic' and self.knots_t:
            return _interp(t, self.knots_t, self.knots_p)
        return self.base_rate if self.base_rate is not None else 0.0

    def to_metadata(self) -> dict:
        return {
            'method': self.method,
            'n': self.n,
            'a': self.a,
            'b': self.b,
            'knots_t': list(self.knots_t),
            'knots_p': list(self.knots_p),
            'base_rate': self.base_rate,
        }


def fit_calibration(
    labels: list[tuple[float, bool]],
    regime_threshold: int = _REGIME_THRESHOLD,
) -> Calibration:
    """Fit P(edit) from (judge_score, edited) pairs. Chooses logistic vs isotonic
    by label count; falls back to a base rate when the data is degenerate."""
    pairs = [(float(s), bool(e)) for s, e in labels]
    n = len(pairs)
    if n == 0:
        return Calibration(method='base_rate', n=0, base_rate=None)
    ys = [1.0 if e else 0.0 for _, e in pairs]
    if all(y == ys[0] for y in ys):
        # one class only — a slope is meaningless; report the constant rate.
        return Calibration(method='base_rate', n=n, base_rate=ys[0])
    ts = [1.0 - s for s, _ in pairs]
    if n < regime_threshold:
        a, b = _fit_logistic(ts, ys)
        return Calibration(method='logistic', n=n, a=a, b=b)
    knots_t, knots_p = _isotonic(ts, ys)
    return Calibration(method='isotonic', n=n, knots_t=tuple(knots_t), knots_p=tuple(knots_p))


def _sigmoid(x: float) -> float:
    # split form avoids exp overflow for large |x|
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def _fit_logistic(ts: list[float], ys: list[float],
                  max_iter: int = 100, tol: float = 1e-8) -> tuple[float, float]:
    """2-parameter logistic regression P = sigmoid(a*t + b) via Newton/IRLS.
    Uses Platt's smoothed targets so perfectly separable data does not send the
    coefficients to infinity. A tiny ridge keeps the 2x2 Hessian invertible."""
    n_pos = sum(ys)
    n_neg = len(ys) - n_pos
    hi = (n_pos + 1.0) / (n_pos + 2.0)
    lo = 1.0 / (n_neg + 2.0)
    targets = [hi if y == 1.0 else lo for y in ys]
    a, b = 0.0, 0.0
    for _ in range(max_iter):
        g0 = g1 = 0.0
        h00 = h01 = h11 = 0.0
        for t, tau in zip(ts, targets):
            p = _sigmoid(a * t + b)
            d = p - tau
            g0 += d * t
            g1 += d
            w = p * (1.0 - p)
            h00 += w * t * t
            h01 += w * t
            h11 += w
        h00 += 1e-10
        h11 += 1e-10
        det = h00 * h11 - h01 * h01
        if abs(det) < 1e-12:
            break
        da = (h11 * g0 - h01 * g1) / det
        db = (-h01 * g0 + h00 * g1) / det
        a -= da
        b -= db
        if abs(da) + abs(db) < tol:
            break
    return a, b


def _isotonic(ts: list[float], ys: list[float]) -> tuple[list[float], list[float]]:
    """Pool-Adjacent-Violators isotonic regression: a non-decreasing step fit of
    P(edit) over the risk feature t. Returns (unique t breakpoints, fitted P)."""
    agg: dict[float, list[float]] = {}
    for t, y in zip(ts, ys):
        bucket = agg.setdefault(t, [0.0, 0.0])  # [sum_y, count]
        bucket[0] += y
        bucket[1] += 1.0
    uniq_t = sorted(agg)
    means = [agg[t][0] / agg[t][1] for t in uniq_t]
    weights = [agg[t][1] for t in uniq_t]

    blocks: list[list[float]] = []  # [weight, value, n_breakpoints]
    for v, w in zip(means, weights):
        block = [w, v, 1.0]
        while blocks and blocks[-1][1] >= block[1]:
            pw, pv, pn = blocks.pop()
            new_w = pw + block[0]
            new_v = (pv * pw + block[1] * block[0]) / new_w
            block = [new_w, new_v, pn + block[2]]
        blocks.append(block)

    fitted: list[float] = []
    for _w, v, nbp in blocks:
        fitted.extend([v] * int(nbp))
    return uniq_t, fitted


def _interp(t: float, xs: tuple[float, ...], ys: tuple[float, ...]) -> float:
    """Monotone piecewise-linear interpolation over isotonic knots, clamped at
    the ends. Preserves non-decreasing-ness, so predict stays non-increasing in
    the judge score."""
    if not xs:
        return 0.0
    if t <= xs[0]:
        return ys[0]
    if t >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if t <= xs[i]:
            x0, x1 = xs[i - 1], xs[i]
            y0, y1 = ys[i - 1], ys[i]
            if x1 == x0:
                return y1
            return y0 + (y1 - y0) * (t - x0) / (x1 - x0)
    return ys[-1]
