"""
FACEIT matchmaking skill: a player-skill fit on the FACEIT matchmaking, hub and championship games of linked players and
everyone they played with or against. Pros play thousands of these games, which no competitive-map fit sees, and a fit
(unlike a raw win rate) credits a win by the strength of the opponents.

    logit P(faction1 wins) = mean(s over faction1) - mean(s over faction2),   s_p ~ N(0, 1 / RIDGE)

each game weighted by 0.5^(age / HALF_LIFE_DAYS) over the WINDOW_DAYS before the cutoff.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import scipy.sparse as sp
from scipy.optimize import minimize
from scipy.special import expit

from gnomepy_research.models.cs2.e12.tables import Tables

HALF_LIFE_DAYS = 90.0
WINDOW_DAYS = 365
RIDGE = 0.5
TEAM = 5
COMPETITION_TYPES = ("matchmaking", "hub", "championship")


class MatchmakingGames:
    """Five-a-side FACEIT games known before `before`, players as integer codes."""

    def __init__(self, tables: Tables, before: pd.Timestamp):
        link = tables.load("hltv_faceit_map")
        m = tables.load("faceit_history_matches", before, ["match_id", "competition_type", "finished_at", "winner"])
        m = m[m.competition_type.isin(COMPETITION_TYPES) & m.winner.isin(["faction1", "faction2"])]
        t = tables.history_players()
        mid = t["match_id"].unify_dictionaries().combine_chunks()
        code_in_m = pc.index_in(mid.dictionary, value_set=pa.array(m.match_id.to_numpy())).to_numpy(zero_copy_only=False)
        code_in_m = np.where(np.isnan(code_in_m.astype(float)), -1, code_in_m).astype(np.int64)
        match_codes = code_in_m[mid.indices.to_numpy()]
        ids = t["faceit_id"].unify_dictionaries().combine_chunks()
        player = ids.indices.to_numpy().astype(np.int64)
        names = ids.dictionary.to_pylist()
        fac = t["faction"].unify_dictionaries().combine_chunks()
        side = (np.asarray(fac.dictionary.to_pylist()) == "faction1")[fac.indices.to_numpy()]
        known = match_codes >= 0
        match_codes, player, side = match_codes[known], player[known], side[known]
        del t, mid, ids, fac
        counts = np.bincount(match_codes * 2 + (~side), minlength=2 * len(m)).reshape(-1, 2)
        full = (counts == TEAM).all(axis=1)
        keep = full[match_codes]
        self.match = match_codes[keep]
        self.player = player[keep]
        self.side = np.where(side[keep], 1.0, -1.0)
        self.time = pd.to_datetime(m.finished_at.to_numpy(), unit="s").to_numpy()
        self.y = (m.winner.to_numpy() == "faction1").astype(float)
        code_of = {f: i for i, f in enumerate(names)}
        self.linked = {int(h): code_of[f] for f, h in zip(link.faceit_id, link.hltv_id) if f in code_of}

    def fit(self, cutoff: pd.Timestamp) -> dict[int, float]:
        """HLTV player id -> matchmaking skill from games before the cutoff (linked players who played any)."""
        c = np.datetime64(cutoff)
        in_window = (self.time < c) & (self.time >= c - np.timedelta64(WINDOW_DAYS, "D"))
        rows = in_window[self.match]
        games = np.flatnonzero(in_window)
        g_index = np.full(len(self.time), -1)
        g_index[games] = np.arange(len(games))
        r = g_index[self.match[rows]]
        players, col = np.unique(self.player[rows], return_inverse=True)
        A = sp.csr_matrix((self.side[rows] / TEAM, (r, col)), shape=(len(games), len(players)))
        At = A.T.tocsr()
        y = self.y[games]
        w = 0.5 ** (((c - self.time[games]) / np.timedelta64(1, "D")) / HALF_LIFE_DAYS)

        def objective(s: np.ndarray) -> tuple[float, np.ndarray]:
            d = A @ s
            nll = -(w * (y * d - np.logaddexp(0.0, d))).sum() + 0.5 * RIDGE * (s @ s)
            return nll, -(At @ (w * (y - expit(d)))) + RIDGE * s

        s = minimize(objective, np.zeros(len(players)), jac=True, method="L-BFGS-B", options={"maxiter": 500}).x
        out = {}
        for h, code in self.linked.items():
            i = np.searchsorted(players, code)
            if i < len(players) and players[i] == code:
                out[h] = float(s[i])
        return out
