"""Matchmaking skill of linked pros: FACEIT result regressed on both teams' mean skill level plus one ridge column
per linked player, so a player's coefficient is skill beyond what their level says."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression

from gnomepy_research.sessions.cs2_comp import bundle

HALF_LIFE_DAYS = 300.0
WINDOW_DAYS = 730
C_RIDGE = 0.3
ROUNDS_MIN = 16
BINARY_ROUNDS = 19.0
ROUND_INFLATION = 8.0


class MatchmakingTable:
    """Per match: mean skill level of each faction and the linked players on it. Built once, cut by finished_at on use."""

    def __init__(self, data_dir: str):
        link = pd.read_parquet(f"{data_dir}/faceit/hltv_faceit_map.parquet")
        self.to_hltv = dict(zip(link.faceit_id, link.hltv_id.astype(int)))
        linked_ids = set(self.to_hltv)
        sums, linked = [], []
        for batch in pq.ParquetFile(f"{data_dir}/faceit/history_players.parquet").iter_batches(
                batch_size=2_000_000, columns=["match_id", "faction", "faceit_id", "skill_level"]):
            b = batch.to_pandas()
            sums.append(b.groupby(["match_id", "faction"]).skill_level.agg(["sum", "count"]))
            linked.append(b[b.faceit_id.isin(linked_ids)][["match_id", "faction", "faceit_id"]])
        total = pd.concat(sums).groupby(level=[0, 1]).sum()
        mean = (total["sum"] / total["count"]).unstack("faction")
        linked = pd.concat(linked, ignore_index=True)
        linked["hltv_id"] = linked.faceit_id.map(self.to_hltv)
        self.levels = mean
        self.linked = linked[["match_id", "faction", "hltv_id"]]
        matches = pd.read_parquet(f"{data_dir}/faceit/history_matches.parquet",
                                  columns=["match_id", "finished_at", "winner", "teams_size", "game_mode", "score_faction1", "score_faction2", "competition_type"])
        self.matches = matches[(matches.teams_size == 5) & matches.winner.isin(["faction1", "faction2"])
                               & matches.game_mode.isin(["5v5", "5v5 League"])]

    def fit(self, cutoff: pd.Timestamp, c: float = C_RIDGE, half_life: float = HALF_LIFE_DAYS,
            window_days: int = WINDOW_DAYS, rounds: bool = False,
            competition_types: tuple[str, ...] | None = None, min_level: float = 0.0) -> tuple[dict[int, float], float]:
        end = int(cutoff.tz_localize("UTC").timestamp()) if cutoff.tzinfo is None else int(cutoff.timestamp())
        start = end - window_days * 86400
        m = self.matches[(self.matches.finished_at < end) & (self.matches.finished_at >= start)]
        if competition_types:
            m = m[m.competition_type.isin(competition_types)]
        if min_level:
            lv0 = self.levels.reindex(m.match_id)
            m = m[(lv0.mean(axis=1) >= min_level).to_numpy()]
        lk = self.linked[self.linked.match_id.isin(m.match_id)]
        m = m[m.match_id.isin(lk.match_id)].reset_index(drop=True)
        lv = self.levels.reindex(m.match_id)
        diff = (lv["faction1"] - lv["faction2"]).fillna(0.0).to_numpy()
        players = sorted(lk.hltv_id.unique())
        pidx = {p: i for i, p in enumerate(players)}
        midx = pd.Series(np.arange(len(m)), index=m.match_id)
        sign = np.where(lk.faction == "faction1", 1.0, -1.0)
        X = sp.csr_matrix((sign, (midx[lk.match_id].to_numpy(), lk.hltv_id.map(pidx).to_numpy())), shape=(len(m), len(players)))
        X = sp.hstack([X, sp.csr_matrix(diff[:, None])]).tocsr()
        age = (end - m.finished_at.to_numpy()) / 86400
        decay = 0.5 ** (age / half_life)
        clf = LogisticRegression(C=c, fit_intercept=True, solver="lbfgs", max_iter=200)
        if rounds:
            s1, s2 = m.score_faction1.to_numpy(), m.score_faction2.to_numpy()
            by_rounds = np.isfinite(s1) & np.isfinite(s2) & (s1 + s2 >= ROUNDS_MIN)
            win1 = (m.winner == "faction1").to_numpy()
            w1 = np.where(by_rounds, s1, np.where(win1, BINARY_ROUNDS, 0.0)) / ROUND_INFLATION
            w2 = np.where(by_rounds, s2, np.where(win1, 0.0, BINARY_ROUNDS)) / ROUND_INFLATION
            clf.fit(sp.vstack([X, X]), np.r_[np.ones(len(m)), np.zeros(len(m))], sample_weight=np.r_[w1 * decay, w2 * decay])
        else:
            clf.fit(X, (m.winner == "faction1").to_numpy(), sample_weight=decay)
        return dict(zip(players, clf.coef_[0][:-1])), float(clf.coef_[0][-1])

    def activity(self, cutoff: pd.Timestamp, days: int = 30) -> dict[int, int]:
        end = int(cutoff.tz_localize("UTC").timestamp()) if cutoff.tzinfo is None else int(cutoff.timestamp())
        m = self.matches[(self.matches.finished_at < end) & (self.matches.finished_at >= end - days * 86400)]
        return self.linked[self.linked.match_id.isin(m.match_id)].hltv_id.value_counts().to_dict()
