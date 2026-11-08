"""The players entry's pre-map prior as a drop-in replacement for the benchmark prior: the roster-strength logistic
regression where a map has no kickoff market, the market stacked with e12 and map-specific roster terms where it
does. Strength is point in time: every count comes from maps that ended before the map began."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import Bundle
from ..players import entry as P


class RosterPrior:
    def __init__(self, train: Bundle, asof: pd.Timestamp):
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

    def premap(self, b: Bundle, game_ids) -> pd.DataFrame:
        ids = pd.unique(np.asarray(list(game_ids)))
        if self._train_pm is not None and self._train_pm[0] == id(b) and set(ids) <= set(self._train_pm[1].index):
            return self._train_pm[1]
        rates, team = P.strength(self.m, b, ids)
        return P.premap(self.m, b, ids, rates, team)

    def __call__(self, b: Bundle, rt: pd.DataFrame) -> pd.DataFrame:
        return P.with_prior2(rt, self.premap(b, rt.game_id.unique()), self.m["premap"], self.m["stack"])

    def release_training(self):
        self._train_pm = None
