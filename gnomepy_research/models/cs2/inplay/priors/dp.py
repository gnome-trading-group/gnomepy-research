"""The prior carried through the score by a constant-strength race: if team A wins each round with probability p,
W(score; p) is its chance of taking the map under MR12 with MR3 overtime. p is set per map so that W(0-0; p) equals the
prior, so the feature says how much of the prior survives at any score, smoothly, including scores the trees rarely
see (late overtime)."""
from __future__ import annotations

import numpy as np

from ..features import need

P_GRID = np.linspace(0.02, 0.98, 481)
MAX_SCORE = 40


def _table() -> np.ndarray:
    """W[p, sa, sb] for scores up to MAX_SCORE; overtime blocks repeat, so the value at 12+3k+i, 12+3k+j is the value
    at 12+i, 12+j, and the 15-15 block start equals the 12-12 start (solved by fixed-point iteration)."""
    p = P_GRID[:, None, None]
    n = MAX_SCORE + 1
    W = np.full((len(P_GRID), n, n), 0.5)
    term = {}
    for sa in range(n):
        for sb in range(n):
            na, nb = need(sa, sb)
            if na <= 0:
                term[(sa, sb)] = 1.0
            elif nb <= 0:
                term[(sa, sb)] = 0.0
    order = sorted(((sa, sb) for sa in range(n) for sb in range(n) if (sa, sb) not in term), key=lambda s: -(s[0] + s[1]))
    for _ in range(60):
        for sa, sb in order:
            if sa >= 15 and sb >= 15 and min(sa, sb) >= 15:
                k = (min(sa, sb) - 12) // 3
                ra, rb = sa - 3 * k, sb - 3 * k
                if (ra, rb) != (sa, sb):
                    W[:, sa, sb] = W[:, ra, rb]
                    continue
            wa = term.get((sa + 1, sb)) if sa + 1 < n else None
            wb = term.get((sa, sb + 1)) if sb + 1 < n else None
            va = wa if wa is not None else (W[:, sa + 1, sb] if sa + 1 < n else W[:, sa, sb])
            vb = wb if wb is not None else (W[:, sa, sb + 1] if sb + 1 < n else W[:, sa, sb])
            W[:, sa, sb] = P_GRID * va + (1 - P_GRID) * vb
    return W


_W = None


def table() -> np.ndarray:
    global _W
    if _W is None:
        _W = _table()
    return _W


def round_p(prior_logit) -> np.ndarray:
    """Per-round win probability whose 0-0 map value matches the prior."""
    w0 = table()[:, 0, 0]
    target = 1 / (1 + np.exp(-np.asarray(prior_logit, float)))
    return np.interp(target, w0, P_GRID)


def value(score_a, score_b, prior_logit) -> np.ndarray:
    W = table()
    p = round_p(prior_logit)
    i = np.clip(np.searchsorted(P_GRID, p), 1, len(P_GRID) - 1)
    lo, hi = P_GRID[i - 1], P_GRID[i]
    t = (p - lo) / (hi - lo)
    sa, sb = np.asarray(score_a, int), np.asarray(score_b, int)
    m = np.minimum(sa, sb)
    k = np.where(m >= 15, (m - 12) // 3, 0)
    sa, sb = np.clip(sa - 3 * k, 0, MAX_SCORE), np.clip(sb - 3 * k, 0, MAX_SCORE)
    return (1 - t) * W[i - 1, sa, sb] + t * W[i, sa, sb]
