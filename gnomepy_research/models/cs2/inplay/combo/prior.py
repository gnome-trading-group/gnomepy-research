"""The players entry's pre-map prior as a drop-in replacement for the benchmark prior: the roster-strength logistic
regression where a map has no kickoff market, the market stacked with e12 and map-specific roster terms where it
does. Strength is point in time: every count comes from maps that ended before the map began."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ..data import Bundle
from ..features import round_table
from ..players import entry as P


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


class RosterPrior:
    def __init__(self, train: Bundle, asof: pd.Timestamp, map_price: bool = False):
        hist = P.History(train, P.CFG["half_life"], asof)
        m = dict(hist=hist, pri=P.priors_from(hist))
        r = train.rounds[train.rounds.a_wins.notna()]
        ct_won = np.where(r.side_a == "CT", r.a_wins, 1 - r.a_wins)
        m["map_ct"] = pd.Series(ct_won).groupby(r["map"].to_numpy()).mean().to_dict()
        done = train.games[train.games.map_a_wins.notna()]
        rates, team = P.strength(m, train, done.game_id)
        pm = P.premap(m, train, done.game_id, rates, team)
        m["premap"] = P.fit_premap(pm, done.set_index("game_id").map_a_wins)
        m["stack"] = P.fit_stack(train, P.premap_z(pm, m["premap"]), pm)
        self.m = m
        # Fitting prices the training maps several times over; their strength table is already at hand.
        self._train_pm = (id(train), pm)
        self.map_stack = self._fit_map_stack(train) if map_price else None

    def _fit_map_stack(self, train: Bundle):
        """The map's own pre-map market price stacked with the roster prior. The series price behind the stacked
        market prior cannot know which map is played or the score of the series so far; the map market can."""
        if "p_map" not in train.priors:
            return None
        g = train.games
        ids = set(train.priors.game_id[train.priors.p_map.notna()]) & set(g.game_id[g.map_a_wins.notna()])
        if len(ids) < 300:
            return None
        rt = round_table(train.subset(ids))
        rt = rt[rt["round"] == rt.groupby("game_id")["round"].transform("min")]
        rt = self._base(train, rt)
        return LogisticRegression(C=1.0).fit(np.c_[logit(rt.p_map), rt.prior_logit], rt.map_a_wins)

    def _base(self, b: Bundle, rt: pd.DataFrame) -> pd.DataFrame:
        return P.with_prior2(rt, self.premap(b, rt.game_id.unique()), self.m["premap"], self.m["stack"])

    def premap(self, b: Bundle, game_ids) -> pd.DataFrame:
        ids = pd.unique(np.asarray(list(game_ids)))
        if self._train_pm is not None and self._train_pm[0] == id(b) and set(ids) <= set(self._train_pm[1].index):
            return self._train_pm[1]
        rates, team = P.strength(self.m, b, ids)
        return P.premap(self.m, b, ids, rates, team)

    def __call__(self, b: Bundle, rt: pd.DataFrame) -> pd.DataFrame:
        rt = self._base(b, rt)
        if getattr(self, "map_stack", None) is None or "p_map" not in rt:
            return rt
        ok = rt.p_map.notna().to_numpy()
        z = rt.prior_logit.to_numpy(float).copy()
        if ok.any():
            z[ok] = self.map_stack.decision_function(np.c_[logit(rt.p_map[ok]), z[ok]])
        return rt.assign(prior_logit=z, has_prior=np.where(ok, 1, rt.has_prior))

    def release_training(self):
        self._train_pm = None
