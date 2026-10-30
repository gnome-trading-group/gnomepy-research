"""
Two filters, one per source, and the features a series gets from them.

The HLTV filter sees only HLTV maps. The FACEIT filter sees only FACEIT maps, so its skills sit on the FACEIT scale
(an ESEA division is not the HLTV top 30) and a linked player's HLTV and FACEIT skills are separate numbers. The stage
that combines them (stack.py) learns how far a FACEIT skill moves an HLTV match, and a lineup with no FACEIT history
contributes nothing instead of a level-shifted guess.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gnomepy_research.sessions.cs2_comp.e07.data import LINEUP, RankPoints, Streams
from gnomepy_research.sessions.cs2_comp.e07.filter import Filter, FilterConfig
from gnomepy_research.sessions.cs2_comp.e07.lineup import RosterMoves

FIVE_DAYS = 180.0
ACTIVITY_DAYS = 60.0


@dataclass(frozen=True)
class ModelConfig:
    hltv: FilterConfig = field(default_factory=lambda: FilterConfig(use_faceit=False))
    faceit: FilterConfig = field(default_factory=lambda: FilterConfig(use_hltv=False, rank_prior_faceit=False,
                                                                       faceit_prior=0.0))
    use_faceit: bool = True
    roster_moves: bool = True


class Model:
    def __init__(self, cfg: ModelConfig, streams: Streams, rank_points: RankPoints, moves: RosterMoves | None = None):
        self.cfg = cfg
        self.keys = list(streams.index)
        self.index = streams.index
        self.moves = moves if cfg.roster_moves else None
        self.h = Filter(cfg.hltv, streams, rank_points)
        self.f = Filter(cfg.faceit, streams, rank_points) if cfg.use_faceit else None

    def advance(self, day: float) -> None:
        self.h.advance(day)
        if self.f is not None:
            self.f.advance(day)

    def _uncertainty_weighted(self, ids: np.ndarray, t: float) -> float:
        """FACEIT skill of each covered player, weighted by how unsure the HLTV filter still is about him."""
        covered = self.f.faceit_games[ids] > 0
        if not covered.any():
            return 0.0
        c = self.cfg.hltv
        seen = self.h.seen[ids]
        var = np.where(np.isnan(seen), c.sigma0 ** 2, self.h.var[ids] + c.tau ** 2 * np.maximum(t - np.nan_to_num(seen), 0.0))
        weight = np.minimum(var / c.sigma0 ** 2, 1.0)
        return float(((self.f.mu[ids] - self.cfg.faceit.faceit_prior) * weight)[covered].sum() / LINEUP)

    def lineup(self, team: int, cutoff_day: float) -> tuple[list, float | None]:
        """Slots of the team's expected five (player index, or None when unknown) and the day of its last HLTV map."""
        known = self.h.last_five.get(team)
        if known is None:
            return [], None
        ids, last_t = known
        if self.moves is None:
            return ids.tolist(), last_t
        roster = self.moves.roster(team, [self.keys[i] for i in ids], last_t, cutoff_day)
        return [self.index.get(k) if isinstance(k, int) else None for k in roster], last_t

    def soft_lineup(self, team: int, cutoff_day: float) -> list[tuple[int | None, float]]:
        """Probability that each recent player of the team plays: recency-weighted appearances over its last maps,
        edited by the roster moves since the last map."""
        recent = self.h.recent_fives(team)
        if not recent:
            return [(None, float(LINEUP))]
        decay = self.cfg.hltv.recency
        probs: dict[int, float] = {}
        total = 0.0
        for r, ids in enumerate(recent):
            for i in ids.tolist():
                probs[i] = probs.get(i, 0.0) + decay ** r
            total += decay ** r
        probs = {i: v / total for i, v in probs.items()}
        unknown_joiners: set[int] = set()
        if self.moves is not None:
            last_t = self.h.last_five[team][1]
            for person, sign in self.moves.changes(team, last_t, cutoff_day):
                idx = self.index.get(person)
                if sign < 0:
                    probs.pop(idx, None)
                    unknown_joiners.discard(person)
                elif idx is None:
                    unknown_joiners.add(person)
                else:
                    probs[idx] = 1.0
        joiners = len(unknown_joiners)
        return self._normalise(probs, joiners)

    @staticmethod
    def _normalise(probs: dict, joiners: int) -> list[tuple[int | None, float]]:
        slots = list(probs.items())
        pinned = [(i, w) for i, w in slots if w >= 1.0]
        free = [(i, w) for i, w in slots if w < 1.0]
        budget = LINEUP - len(pinned) - joiners
        mass = sum(w for _, w in free)
        scale = 1.0 if mass <= budget or mass == 0 else budget / mass
        out = pinned + [(i, w * scale) for i, w in free]
        if joiners:
            out += [(None, 1.0)] * joiners
        gap = LINEUP - sum(w for _, w in out)
        if gap > 1e-9:
            out.append((None, gap))
        return out

    def features(self, team_a: int, team_b: int, t: float, cutoff_day: float) -> dict:
        slots = {side: self.lineup(team, cutoff_day) for side, team in (("a", team_a), ("b", team_b))}
        q_last, mean_last, var_last = self.h.price(team_a, team_b, t, cutoff_day)
        if self.moves is None:
            q, mean, var = q_last, mean_last, var_last
        else:
            q, mean, var = self.h.price(team_a, team_b, t, cutoff_day, lineups=(slots["a"][0], slots["b"][0]))
        soft_a, soft_b = self.soft_lineup(team_a, cutoff_day), self.soft_lineup(team_b, cutoff_day)
        q_soft, mean_soft, var_soft = self.h.price(team_a, team_b, t, cutoff_day, weighted=(soft_a, soft_b))
        row = {"q_h": q, "mean_h": mean, "var_h": var, "q_last": q_last, "q_soft": q_soft, "mean_soft": mean_soft,
               "var_soft": var_soft}
        for side, team in (("a", team_a), ("b", team_b)):
            slot, last_t = slots[side]
            ids = np.array([i for i in slot if i is not None], dtype=np.int64)
            row[f"five_h_{side}"] = 0 if len(ids) < 5 else self.h.five_maps(ids, t, FIVE_DAYS)
            row[f"act_h_{side}"] = 0.0 if not len(ids) else self.h.activity(ids, t, ACTIVITY_DAYS)
            row[f"days_h_{side}"] = 999.0 if last_t is None else t - last_t
            row[f"tm90_{side}"] = self.h.team_maps(team, t, 90.0)
            row[f"swap_{side}"] = 0 if last_t is None else int(len(ids) < 5 or set(ids.tolist()) != set(self.h.last_five[team][0].tolist()))
            if self.f is None:
                continue
            s, v, n = (0.0, 0.0, 0) if not len(ids) else self.f.profile(ids, t)
            row[f"f_s_{side}"], row[f"f_v_{side}"], row[f"f_n_{side}"] = s, v, n
            row[f"f_su_{side}"] = 0.0 if not len(ids) else self._uncertainty_weighted(ids, t)
            row[f"five_f_{side}"] = 0 if len(ids) < 5 else self.f.five_maps(ids, t, FIVE_DAYS)
            row[f"act_f_{side}"] = 0.0 if not len(ids) else self.f.activity(ids, t, ACTIVITY_DAYS)
        return row
