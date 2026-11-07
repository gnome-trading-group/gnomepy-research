"""Map win probability from the score if every round were an independent coin with the same probability q for team A,
including MR12 regulation and MR3 overtime blocks (first to 4 of 6, 3-3 replays the block). q is chosen so that the
0-0 value equals the pre-map prior. A smooth, exactly-shaped summary of how a prior and a score combine, given to the
map trees as one feature."""
from __future__ import annotations

import numpy as np

Q = np.linspace(0.02, 0.98, 961)


def _tables(q):
    G = len(q)
    win_block = q ** 4 * (1 + 4 * (1 - q) + 10 * (1 - q) ** 2)
    tie = 20 * q ** 3 * (1 - q) ** 3
    v_ot = win_block / (1 - tie)
    B = np.zeros((G, 5, 5))
    B[:, 4, :4] = 1.0
    for x in range(3, -1, -1):
        for y in range(3, -1, -1):
            if x == 3 and y == 3:
                B[:, 3, 3] = v_ot
                continue
            B[:, x, y] = q * B[:, x + 1, y] + (1 - q) * B[:, x, y + 1]
    V = np.zeros((G, 14, 14))
    V[:, 13, :13] = 1.0
    for a in range(12, -1, -1):
        for b in range(12, -1, -1):
            if a == 12 and b == 12:
                V[:, 12, 12] = v_ot
                continue
            V[:, a, b] = q * V[:, a + 1, b] + (1 - q) * V[:, a, b + 1]
    return V, B


V, B = _tables(Q)
START = V[:, 0, 0]


def dp_value(need_a, need_b, ot, prior_logit):
    """Value at the given rounds-needed state for the q whose 0-0 value matches sigmoid(prior_logit)."""
    p0 = 1.0 / (1.0 + np.exp(-np.asarray(prior_logit, float)))
    qi = np.clip(np.searchsorted(START, p0), 0, len(Q) - 1)
    na, nb = np.asarray(need_a, int), np.asarray(need_b, int)
    ot = np.asarray(ot, int) == 1
    out = np.empty(len(p0))
    ra = np.clip(13 - na, 0, 13); rb = np.clip(13 - nb, 0, 13)
    out[~ot] = V[qi[~ot], ra[~ot], rb[~ot]]
    xa = np.clip(4 - na, 0, 4); xb = np.clip(4 - nb, 0, 4)
    out[ot] = B[qi[ot], xa[ot], xb[ot]]
    return out
