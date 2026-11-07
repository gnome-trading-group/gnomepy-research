"""Map value by dynamic programming over the score: a round-start model q(score state) = P(A wins the next round),
learnt from every round's outcome, rolled forward to P(A wins the map) from any score.

Map outcomes are one label per map; round outcomes are ~20 per map, so a chain of round probabilities is a much less
noisy estimate of the map value than regressing the map outcome on the score directly. The score enters q itself, so
in-map form (a team out-performing its prior) carries into the rounds that remain. The economy enters through the
current streak k (signed: >0 means A won the last k rounds), which drives the loss bonus."""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from ..features import need, resets, side_swaps_at

QCOLS = ["need_a", "need_b", "ct_a", "ot", "round_in_half", "pistol", "map_code", "prior_logit", "has_prior", "streak"]
MAX_SCORE = 28
KMAX = 4


def streaks(rt: pd.DataFrame) -> np.ndarray:
    """Signed win streak of the last winner at the start of each round, reset at money resets."""
    orig = rt.index
    rt = rt.sort_values(["game_id", "round"])
    prev = rt.groupby("game_id").score_a.shift(1)
    a_won_prev = (rt.score_a - prev).to_numpy()
    rounds, games = rt["round"].to_numpy(), rt.game_id.to_numpy()
    out = np.zeros(len(rt), int)
    for i in range(1, len(rt)):
        if games[i] != games[i - 1] or resets(rounds[i]) or np.isnan(a_won_prev[i]):
            continue
        out[i] = step_streak(out[i - 1], a_won_prev[i] == 1)
    return pd.Series(out, index=rt.index).reindex(orig).to_numpy()


def step_streak(k, a_won):
    k = np.asarray(k)
    return np.where(a_won, np.where(k > 0, np.minimum(k + 1, KMAX), 1), np.where(k < 0, np.maximum(k - 1, -KMAX), -1))


def ct_a_at(ct_a_now, round_now, n):
    """A's side at round n given its side at round_now (sides swap at side_swaps_at rounds)."""
    swaps = lambda r: np.asarray(r >= 13, int) + np.where(r >= 28, (r - 28) // 6 + 1, 0)
    flips = (swaps(n) - swaps(round_now)) % 2
    return np.where(flips == 1, 1 - ct_a_now, ct_a_now)


def legal_scores():
    """Non-terminal scores (sa, sb), up to MAX_SCORE."""
    out = []
    for sa in range(MAX_SCORE):
        for sb in range(MAX_SCORE):
            na, nb = need(sa, sb)
            if na > 0 and nb > 0 and na <= 13 and nb <= 13:
                if min(sa, sb) >= 12 and max(sa, sb) - min(sa, sb) > 3:
                    continue
                out.append((sa, sb))
    return out


SCORES = legal_scores()
SCORE_IX = {s: i for i, s in enumerate(SCORES)}


def q_rows(statics: pd.DataFrame, sa: int, sb: int, ks: np.ndarray) -> pd.DataFrame:
    n = sa + sb + 1
    na, nb = need(sa, sb)
    ot = int(min(sa, sb) >= 12)
    m = len(statics)
    rih = (n - 25) % 3 if ot else (n - 1) % 12
    return pd.DataFrame(dict(
        need_a=np.full(m * len(ks), na), need_b=np.full(m * len(ks), nb),
        ct_a=np.tile(ct_a_at(statics.ct_a.to_numpy(), statics["round"].to_numpy(), n), len(ks)),
        ot=ot, round_in_half=rih, pistol=int(n in (1, 13)), map_code=np.tile(statics.map_code.to_numpy(), len(ks)),
        prior_logit=np.tile(statics.prior_logit.to_numpy(), len(ks)), has_prior=np.tile(statics.has_prior.to_numpy(), len(ks)),
        streak=np.repeat(ks, m)))


def values(q_model, statics: pd.DataFrame) -> np.ndarray:
    """V[map, score index, k + KMAX]: P(A wins the map) at the start of a round with that score and streak.
    statics: one row per map with ct_a/round (A's side at some known round), map_code, prior_logit, has_prior."""
    m = len(statics)
    V = np.full((m, len(SCORES), 2 * KMAX + 1), np.nan)
    ks_all = np.arange(-KMAX, KMAX + 1)

    def val(sa, sb, k):
        na, nb = need(sa, sb)
        if na <= 0:
            return np.ones(m)
        if nb <= 0:
            return np.zeros(m)
        if (sa, sb) not in SCORE_IX:
            return np.full(m, 0.5)
        return V[:, SCORE_IX[(sa, sb)], k + KMAX]

    for total in range(2 * MAX_SCORE, -1, -1):
        pairs = [(sa, sb) for sa, sb in SCORES if sa + sb == total]
        if not pairs:
            continue
        n = total + 1
        X, meta = [], []
        for sa, sb in pairs:
            X.append(q_rows(statics, sa, sb, ks_all))
            meta.append((sa, sb))
        q = q_model.predict_proba(pd.concat(X, ignore_index=True)[QCOLS])[:, 1].reshape(len(pairs), len(ks_all), m)
        reset_next = bool(resets(np.array([n + 1]))[0])
        for j, (sa, sb) in enumerate(meta):
            for ki, k in enumerate(ks_all):
                kw = 0 if reset_next else int(step_streak(k, True))
                kl = 0 if reset_next else int(step_streak(k, False))
                V[:, SCORE_IX[(sa, sb)], ki] = q[j, ki] * val(sa + 1, sb, kw) + (1 - q[j, ki]) * val(sa, sb + 1, kl)
    return V


class Avg:
    """Mean probability of several models: the DP compounds q's errors over many rounds, so q's variance matters."""

    def __init__(self, models):
        self.models = models

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.models], axis=0)


def fit_q(rt: pd.DataFrame, seed: int, n_seeds: int = 1, weight=None, cols=None, **kw):
    params = dict(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=100,
                  reg_lambda=5, n_jobs=8, random_state=seed, tree_method="hist")
    params.update(kw)
    cols = cols or QCOLS
    models = []
    for i in range(n_seeds):
        params["random_state"] = seed * 1000 + i
        models.append(xgb.XGBClassifier(**params).fit(rt[cols], rt.a_wins, sample_weight=weight))
    return Avg(models)


def next_values(V: np.ndarray, map_ix: np.ndarray, sa, sb, k, n) -> tuple[np.ndarray, np.ndarray]:
    """M(score if A wins this round), M(score if A loses) for rows (map index, score, streak) at round n."""
    sa, sb, k, n = (np.asarray(v, int) for v in (sa, sb, k, n))
    reset_next = resets(n + 1)
    out = []
    for a_won in (1, 0):
        sa2, sb2 = sa + a_won, sb + 1 - a_won
        k2 = np.where(reset_next, 0, step_streak(k, a_won == 1))
        nd = np.array([need(x, y) for x, y in zip(sa2, sb2)]).reshape(-1, 2)
        si = np.array([SCORE_IX.get((x, y), -1) for x, y in zip(sa2, sb2)], int)
        v = V[map_ix, np.maximum(si, 0), k2 + KMAX]
        v = np.where(si < 0, 0.5, v)
        out.append(np.where(nd[:, 0] <= 0, 1.0, np.where(nd[:, 1] <= 0, 0.0, v)))
    return out[0], out[1]
