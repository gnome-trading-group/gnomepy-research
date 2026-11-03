"""
FACEIT matchmaking form of a five: how its HLTV-linked players have been doing in their own FACEIT matchmaking games.

Pros play matchmaking against a pool of near-uniform strength (almost everyone is FACEIT level 10), so a player's
recent matchmaking win rate is a large-sample measure of individual form that none of the competitive-map fits sees.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from gnomepy_research.sessions.cs2_comp.bundle import load

HALF_LIFE_DAYS = 60.0
SHRINK_GAMES = 100.0
WINDOW_DAYS = 365
COMPETITION_TYPES = ("matchmaking", "hub")


class MatchmakingForm:
    """Every finished matchmaking game of each HLTV-linked player, known before `before`."""

    def __init__(self, data_dir: str | Path, before: pd.Timestamp):
        link = load(data_dir, "faceit/hltv_faceit_map")
        hltv_of = dict(zip(link.faceit_id, link.hltv_id.astype(np.int64)))
        matches = load(data_dir, "faceit/history_matches", before, ["match_id", "competition_type", "finished_at", "winner"])
        matches = matches[matches.competition_type.isin(COMPETITION_TYPES)]
        # history_players is cut through its matches; reading only the linked players' rows keeps memory small, and
        # the inner join below with the cut matches applies the same point-in-time rule as bundle.load.
        players = pq.read_table(Path(data_dir) / "faceit/history_players.parquet", columns=["match_id", "faction", "faceit_id"])
        players = players.filter(pc.is_in(players["faceit_id"], value_set=pa.array(list(hltv_of)))).to_pandas()
        d = players.merge(matches, on="match_id")
        d = d.assign(hltv_id=d.faceit_id.map(hltv_of), t=pd.to_datetime(d.finished_at, unit="s"),
                     won=(d.winner == d.faction).astype(float))
        d = d.sort_values(["hltv_id", "t"], kind="stable")
        self.games = {int(k): (g.t.to_numpy(), g.won.to_numpy()) for k, g in d.groupby("hltv_id")}

    def player(self, player_id: int, cutoff: pd.Timestamp) -> float:
        """Shrunk, recency-weighted matchmaking win rate minus one half over the window before the cutoff."""
        g = self.games.get(int(player_id))
        if g is None:
            return 0.0
        t, won = g
        c = np.datetime64(cutoff)
        i = np.searchsorted(t, c)
        j = np.searchsorted(t, c - np.timedelta64(WINDOW_DAYS, "D"))
        w = 0.5 ** (((c - t[j:i]) / np.timedelta64(1, "D")) / HALF_LIFE_DAYS)
        return float(((won[j:i] - 0.5) * w).sum() / (w.sum() + SHRINK_GAMES))

    def five(self, ids, cutoff: pd.Timestamp) -> float:
        ids = [] if ids is None else list(ids)
        return float(np.mean([self.player(p, cutoff) for p in ids])) if ids else 0.0

    def diff(self, lineups_a, lineups_b, cutoffs) -> np.ndarray:
        """Five A's form minus five B's, in tenths of a win rate."""
        return np.array([self.five(a, c) - self.five(b, c) for a, b, c in zip(lineups_a, lineups_b, cutoffs)]) * 10.0
