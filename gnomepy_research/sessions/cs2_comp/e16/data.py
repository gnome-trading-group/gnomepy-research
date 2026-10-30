"""
Point-in-time inputs for one fit: HLTV and FACEIT maps as one table of paired comparisons, per-map player impact
rows from both sources, and each player's prior features.

Player keys are strings: "h<hltv id>" for every HLTV player and for FACEIT players linked to one, "f<faceit id>"
for unlinked FACEIT players, so FACEIT opponents still connect the rating graph. Team keys follow the same scheme.

Tables are loaded once with `bundle.load` at the latest cutoff and cut again per fit with the same rules, so a fit
sees exactly the rows the bundle would give it at its own cutoff.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.hltv.config import showmatch_mask
from gnomepy_research.sessions.cs2_comp.bundle import load

LINEUP = 5
DIVISIONS = ("entry", "open", "intermediate", "main", "advanced")
_DIVISION_RE = re.compile(r"\b(entry|open|intermediate|main|advanced)\b", re.I)


def _naive(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    return s.dt.tz_convert(None) if s.dt.tz is not None else s


def division(name: str | None) -> int:
    """ESEA-style division of a competition (1 entry .. 5 advanced), or 0 when the name has none."""
    m = _DIVISION_RE.search(name or "")
    return DIVISIONS.index(m.group(1).lower()) + 1 if m else 0


@dataclass
class Bundle:
    """Every table a fit reads, loaded once up to the latest cutoff."""

    hltv_maps: pd.DataFrame
    hltv_ratings: pd.DataFrame
    rankings: pd.DataFrame
    faceit_maps: pd.DataFrame | None
    faceit_ratings: pd.DataFrame | None
    faceit_levels: pd.DataFrame | None


def read_bundle(data_dir: str | Path, before: pd.Timestamp, faceit: bool) -> Bundle:
    hist = load(data_dir, "hltv/cs2_match_history", before)
    stats = load(data_dir, "hltv/cs2_player_map_stats", before,
                 ["match_id", "match_date", "map_name", "player_id", "team_id", "side", "rating"])
    rankings = load(data_dir, "hltv/cs2_team_rankings", before, ["date", "team_id", "rank"])
    rankings = rankings.assign(date=_naive(rankings.date))
    hmaps = _hltv_maps(hist)
    hrat = _hltv_ratings(stats, hmaps)
    if not faceit:
        return Bundle(hmaps, hrat, rankings, None, None, None)
    fmaps, frat, levels = _faceit(data_dir, before)
    return Bundle(hmaps, hrat, rankings, fmaps, frat, levels)


def _regulation(score_a: np.ndarray, score_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Regulation rounds won by A and played; an overtime map counts as 12-12 of 24, as in the Bayes model."""
    ot = score_a + score_b > 24
    return np.where(ot, 12, score_a), np.where(ot, 24, score_a + score_b)


def _hltv_maps(hist: pd.DataFrame) -> pd.DataFrame:
    h = hist[(hist.team_a_score != hist.team_b_score) & ~showmatch_mask(hist)].copy()
    h["time"] = _naive(h.match_time.fillna(h.match_date))
    h["cut"] = np.maximum(h.time, _naive(h.match_date))

    def five(played, announced):
        for ids in (played, announced):
            if ids is not None and len(ids) == LINEUP and not any(pd.isna(x) for x in ids):
                return tuple(f"h{int(x)}" for x in ids)
        return None

    h["lineup_a"] = [five(p, a) for p, a in zip(h.team_a_player_ids, h.team_a_lineup_ids)]
    h["lineup_b"] = [five(p, a) for p, a in zip(h.team_b_player_ids, h.team_b_lineup_ids)]
    h = h[h.lineup_a.notna() & h.lineup_b.notna()].copy()
    h["rounds_a"], h["rounds"] = _regulation(h.team_a_score.to_numpy(), h.team_b_score.to_numpy())
    h["team_a"] = "h" + h.team_a_id.astype(int).astype(str)
    h["team_b"] = "h" + h.team_b_id.astype(int).astype(str)
    h["source"] = "hltv"
    h["division"] = 0
    return h[["match_id", "map_name", "time", "cut", "source", "team_a", "team_b", "lineup_a", "lineup_b", "rounds_a",
              "rounds", "division", "bo_type", "team_a_won"]].reset_index(drop=True)


def _hltv_ratings(stats: pd.DataFrame, maps: pd.DataFrame) -> pd.DataFrame:
    """One row per player per map: his rating minus his team's mean on that map."""
    s = stats[(stats.side == "all") & stats.rating.notna() & stats.player_id.notna()].copy()
    s["dev"] = s.rating - s.groupby(["match_id", "map_name", "team_id"]).rating.transform("mean")
    s["player"] = "h" + s.player_id.astype(int).astype(str)
    s["team"] = "h" + s.team_id.astype(int).astype(str)
    s = s.merge(maps[["match_id", "map_name", "time", "cut", "team_a", "team_b", "lineup_a", "lineup_b"]],
                on=["match_id", "map_name"])
    s["cut"] = np.maximum(s.cut, _naive(s.match_date))
    s["own"] = np.where(s.team == s.team_a, s.lineup_a, np.where(s.team == s.team_b, s.lineup_b, None))
    s = s[[o is not None and p in o for p, o in zip(s.player, s.own)]]
    return s[["time", "cut", "player", "own", "dev"]].reset_index(drop=True)


def _faceit(data_dir: str | Path, before: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matches = load(data_dir, "faceit/faceit_matches", before,
                   ["match_id", "competition_name", "finished_at", "has_stats"])
    excl = load(data_dir, "faceit/faceit_exclusions", before)
    link = load(data_dir, "faceit/faceit_hltv_series_link", before)
    hist = load(data_dir, "faceit/faceit_match_history", before)
    drop_series = set(excl[excl.map_name.isna()].match_id) | set(link.faceit_match_id) \
        | set(matches[~matches.has_stats.astype(bool)].match_id)
    drop_maps = set(zip(excl[excl.map_name.notna()].match_id, excl[excl.map_name.notna()].map_name))
    h = hist[hist.match_id.isin(set(matches.match_id)) & ~hist.match_id.isin(drop_series)].copy()
    h = h[[k not in drop_maps for k in zip(h.match_id, h.map_name)]]
    h = h[(h.team_a_players == LINEUP) & (h.team_b_players == LINEUP) & (h.team_a_score != h.team_b_score)]
    h["time"] = _naive(h.match_time.fillna(h.match_date))
    finished = pd.to_datetime(matches.set_index("match_id").finished_at, unit="s")
    h["cut"] = h.match_id.map(finished)

    def five(faceit_ids, hltv_ids):
        if faceit_ids is None or len(faceit_ids) != LINEUP:
            return None
        hl = list(hltv_ids) if hltv_ids is not None and len(hltv_ids) == LINEUP else [np.nan] * LINEUP
        return tuple(f"f{f}" if pd.isna(x) else f"h{int(x)}" for f, x in zip(faceit_ids, hl))

    h["lineup_a"] = [five(f, x) for f, x in zip(h.team_a_faceit_ids, h.team_a_player_ids)]
    h["lineup_b"] = [five(f, x) for f, x in zip(h.team_b_faceit_ids, h.team_b_player_ids)]
    h = h[h.lineup_a.notna() & h.lineup_b.notna()]
    h = h[[len(set(a) | set(b)) == 2 * LINEUP for a, b in zip(h.lineup_a, h.lineup_b)]].copy()
    h["rounds_a"], h["rounds"] = _regulation(h.team_a_score.to_numpy(), h.team_b_score.to_numpy())
    h["team_a"] = np.where(h.team_a_id.notna(), "h" + h.team_a_id.fillna(0).astype(int).astype(str), "f" + h.team_a_faceit_id.astype(str))
    h["team_b"] = np.where(h.team_b_id.notna(), "h" + h.team_b_id.fillna(0).astype(int).astype(str), "f" + h.team_b_faceit_id.astype(str))
    h["source"] = "faceit"
    comp = matches.set_index("match_id").competition_name
    h["division"] = h.match_id.map(comp).map(division).astype(int)
    h["total_rounds"] = h.team_a_score + h.team_b_score
    fmaps = h[["match_id", "map_name", "time", "cut", "source", "team_a", "team_b", "lineup_a", "lineup_b", "rounds_a",
               "rounds", "division", "bo_type", "team_a_won", "total_rounds", "team_a_faceit_id"]].reset_index(drop=True)

    cols = ["match_id", "map_name", "faceit_id", "hltv_id", "faction", "kills", "deaths", "assists", "adr"]
    ps = load(data_dir, "faceit/faceit_player_map_stats", before, cols)
    ps = ps.merge(fmaps[["match_id", "map_name", "time", "cut", "total_rounds", "lineup_a", "lineup_b", "team_a_faceit_id"]],
                  on=["match_id", "map_name"])
    ps = ps[ps.kills.notna() & ps.deaths.notna() & (ps.total_rounds > 0)].copy()
    ps["player"] = np.where(ps.hltv_id.notna(), "h" + ps.hltv_id.fillna(0).astype(int).astype(str), "f" + ps.faceit_id.astype(str))
    n = ps.total_rounds.astype(float)
    kpr, dpr, apr = ps.kills / n, ps.deaths / n, ps.assists.fillna(0) / n
    adr = ps.adr.fillna(105.0 * kpr)
    # HLTV 2.0's published weights with KAST left out; constants cancel in the within-team deviation.
    ps["impact"] = 0.864 * kpr - 0.533 * dpr + 0.0996 * apr + 0.0032 * adr
    ps["side_key"] = ps.match_id + "|" + ps.map_name + "|" + ps.faction.astype(str)
    ps["dev"] = ps.impact - ps.groupby("side_key").impact.transform("mean")
    in_a = np.array([p in a for p, a in zip(ps.player, ps.lineup_a)])
    in_b = np.array([p in b for p, b in zip(ps.player, ps.lineup_b)])
    ps["own"] = np.where(in_a, ps.lineup_a, np.where(in_b, ps.lineup_b, None))
    ps = ps[in_a | in_b]
    frat = ps[["time", "cut", "player", "own", "dev"]].reset_index(drop=True)

    ros = load(data_dir, "faceit/faceit_rosters", before, ["match_id", "faceit_id", "hltv_id", "skill_level", "played"])
    ros = ros[ros.skill_level.notna() & (ros.skill_level > 0)]
    ros = ros.merge(matches[["match_id", "finished_at"]], on="match_id")
    ros["time"] = pd.to_datetime(ros.finished_at, unit="s")
    ros["cut"] = ros.time
    ros["player"] = np.where(ros.hltv_id.notna(), "h" + ros.hltv_id.fillna(0).astype(int).astype(str), "f" + ros.faceit_id.astype(str))
    ros = ros.sort_values(["player", "cut"], kind="stable")
    changed = (ros.player != ros.player.shift()) | (ros.skill_level != ros.skill_level.shift())
    levels = ros.loc[changed, ["time", "cut", "player", "skill_level"]].reset_index(drop=True)
    return fmaps.drop(columns=["total_rounds", "team_a_faceit_id"]), frat, levels


def team_lineups(maps: pd.DataFrame) -> dict[str, tuple[tuple[str, ...], pd.Timestamp]]:
    """Each team's five in its latest map of `maps`."""
    rows = pd.concat([maps[["time", "team_a", "lineup_a"]].set_axis(["time", "team", "five"], axis=1),
                      maps[["time", "team_b", "lineup_b"]].set_axis(["time", "team", "five"], axis=1)])
    last = rows.sort_values("time", kind="stable").groupby("team").tail(1)
    return {t: (f, d) for t, f, d in zip(last.team, last.five, last.time)}
