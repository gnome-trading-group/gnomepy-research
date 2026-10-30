"""Walk-forward over weekly cutoffs: one chronological pass of the filter, pricing each week's series from the state
the filter held at that week's cutoff. Maps become visible in order of availability (`avail_day`), so the pass at a
cutoff has seen exactly the maps a bundle cut at that cutoff holds."""
from __future__ import annotations

import collections
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.bundle import load
from gnomepy_research.sessions.cs2_comp.e09.lineups import HISTORY, Entry, Moves, candidate_features, play_probabilities, sample_fives
from gnomepy_research.sessions.cs2_comp.e09.maps import EPOCH, build_maps
from gnomepy_research.sessions.cs2_comp.e09.skill import Filter, Params, RankPoints

LINEUP_SAMPLES = 32
UNKNOWN_BASE = 10 ** 9


def _day(ts: pd.Timestamp) -> float:
    return (pd.Timestamp(ts) - EPOCH).total_seconds() / 86400.0


def unknown_five(team: int) -> tuple[int, ...]:
    return tuple(UNKNOWN_BASE + team * 5 + i for i in range(5))


class TeamHistory:
    """The last few series each team fielded, with the stand-ins HLTV flagged in them."""

    def __init__(self, standins: dict[tuple[str, int], frozenset]) -> None:
        self.recent: dict[int, collections.deque] = collections.defaultdict(lambda: collections.deque(maxlen=HISTORY))
        self.standins = standins

    def add(self, team: int, match_id: str, day: float, five: tuple[int, ...]) -> None:
        q = self.recent[team]
        if not q or q[-1].match_id != match_id:
            q.append(Entry(day, match_id, five, self.standins.get((match_id, team), frozenset())))


def standin_table(lineups: pd.DataFrame) -> dict[tuple[str, int], frozenset]:
    flagged = lineups[lineups.is_standin == True]
    table: dict[tuple[str, int], set] = collections.defaultdict(set)
    for mid, team, player in zip(flagged.match_id.astype(str), flagged.team_id, flagged.player_id):
        table[(mid, int(team))].add(int(player))
    return {k: frozenset(v) for k, v in table.items()}


def _divergence(samples: list[tuple[int, ...]], top: tuple[int, ...]) -> float:
    return float(np.mean([5 - len(set(s) & set(top)) for s in samples])) / 5.0


def priced_lineups(team: int, cutoff_day: float, history: TeamHistory, moves: Moves, rng: np.random.Generator,
                   n: int) -> tuple[list[tuple[int, ...]], tuple[int, ...]]:
    cands, x = candidate_features(team, cutoff_day, list(history.recent.get(team, ())), moves)
    if not cands:
        five = unknown_five(team)
        return [five] * n, five
    probs = play_probabilities(x)
    top = tuple(cands[i] for i in np.argsort(-probs, kind="stable")[:5])
    top = top + unknown_five(team)[: 5 - len(top)]
    return sample_fives(team, cands, probs, rng, n), top


def walk(data_dir: str | Path, targets: pd.DataFrame, params: Params, maps: pd.DataFrame | None = None,
         ranks: RankPoints | None = None) -> tuple[pd.DataFrame, Filter]:
    cutoffs = sorted(pd.to_datetime(targets.week_cutoff).unique())
    last = pd.Timestamp(cutoffs[-1])
    if maps is None:
        maps = build_maps(data_dir, last, params.use_faceit)
    elif not params.use_faceit:
        maps = maps[maps.src == "hltv"]
    if ranks is None:
        ranks = RankPoints(load(data_dir, "hltv/cs2_team_rankings", last))
    f = Filter(params, ranks)
    history = TeamHistory(standin_table(load(data_dir, "hltv/cs2_match_lineups", last, ["match_id", "match_date", "team_id", "player_id", "is_standin"])))
    moves = Moves(load(data_dir, "hltv/cs2_roster_moves", last))
    avail = maps.avail_day.to_numpy()
    cols = {c: maps[c].to_numpy() for c in ("day", "team_a", "team_b", "pa", "pb", "ra", "rb", "src", "comp", "match_id", "sa", "sb")}
    i, n = 0, len(maps)
    out = []
    tg = targets.assign(week_cutoff=pd.to_datetime(targets.week_cutoff))
    params_samples = LINEUP_SAMPLES
    for cutoff in cutoffs:
        cut_day = _day(pd.Timestamp(cutoff))
        while i < n and avail[i] < cut_day:
            f.observe(cols["day"][i], int(cols["team_a"][i]), int(cols["team_b"][i]), cols["pa"][i], cols["pb"][i],
                      int(cols["ra"][i]), int(cols["rb"][i]), cols["src"][i], cols["comp"][i], cols["sa"][i], cols["sb"][i])
            if cols["src"][i] == "hltv":
                history.add(int(cols["team_a"][i]), cols["match_id"][i], cols["day"][i], cols["pa"][i])
                history.add(int(cols["team_b"][i]), cols["match_id"][i], cols["day"][i], cols["pb"][i])
            i += 1
        f.rank_cap = cut_day
        for r in tg[tg.week_cutoff == cutoff].itertuples():
            ta, tb = int(r.team_a_id), int(r.team_b_id)
            kick = _day(r.match_time.tz_convert(None) if getattr(r.match_time, "tzinfo", None) else r.match_time)
            known_a, known_b = getattr(r, "lineup_a", None), getattr(r, "lineup_b", None)
            rng = np.random.default_rng(int(r.match_id))
            if known_a is not None and known_b is not None:
                sa_, top_a, sb_, top_b = [known_a], known_a, [known_b], known_b
            else:
                sa_, top_a = priced_lineups(ta, cut_day, history, moves, rng, params_samples)
                sb_, top_b = priced_lineups(tb, cut_day, history, moves, rng, params_samples)
                for team, top in ((ta, top_a), (tb, top_b)):
                    core = [f.mu[p] for p in top if p in f.mu]
                    if core:
                        f.anchor[team] = float(np.mean(core))
            qs, gaps, vars_ = [], [], []
            for la, lb in zip(sa_, sb_):
                q, g = f.series_price(ta, tb, la, lb, kick)
                qs.append(q)
                gaps.append(g.mean)
                vars_.append(g.var)
            out.append({"match_id": r.match_id, "q": float(np.mean(qs)), "fit_cutoff": pd.Timestamp(cutoff), "gap": float(np.mean(gaps)),
                        "gap_var": float(np.mean(vars_)), "div_a": _divergence(sa_, top_a), "div_b": _divergence(sb_, top_b), "cov_a": sum(f.faceit_games[p] > 0 for p in top_a),
                        "cov_b": sum(f.faceit_games[p] > 0 for p in top_b)})
        f.rank_cap = float("inf")
    return pd.DataFrame(out), f
