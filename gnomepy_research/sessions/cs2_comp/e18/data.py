"""Point-in-time observations for the e18 model, built from the competition bundle.

Every map is one observation: the two fives that played, the regulation rounds and how many team A won, the day and
the source. Players are keyed "h<hltv id>" when HLTV knows them (FACEIT players go through hltv_faceit_map) and
"f<faceit id>" otherwise, so FACEIT-only opponents still connect the rating graph. Teams are keyed the same way, with
a FACEIT faction taking its linked HLTV team id where the bundle has one.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.hltv.bayes_series import RankStrength, UNRANKED
from gnomepy_research.models.cs2.hltv.config import showmatch_mask
from gnomepy_research.sessions.cs2_comp.bundle import load

LINEUP = 5
REGULATION = 24
DIVISIONS = {"entry": 0.0, "open": 1.0, "intermediate": 2.0, "main": 3.0, "advanced": 4.0}
_DIVISION_RE = re.compile(r"\b(entry|open|intermediate|main|advanced)\b", re.I)
COVARIATES = ["rank_z", "unranked", "level", "has_level", "division", "has_division"]


@dataclass
class Observations:
    maps: pd.DataFrame
    ranks: RankStrength
    levels: dict[str, float]
    divisions: dict[str, float]
    latest_lineup: dict[str, list[str]]


def _five(ids) -> bool:
    if ids is None or isinstance(ids, float):
        return False
    arr = np.asarray(ids)
    return len(arr) == LINEUP and not pd.isna(arr).any()


def _rounds(score_a: pd.Series, score_b: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    overtime = (score_a + score_b).to_numpy() > REGULATION
    rounds = np.where(overtime, REGULATION, (score_a + score_b).to_numpy())
    rounds_a = np.where(overtime, REGULATION // 2, score_a.to_numpy())
    return rounds.astype(int), rounds_a.astype(int)


def hltv_maps(data_dir, cutoff: pd.Timestamp) -> pd.DataFrame:
    h = load(data_dir, "hltv/cs2_match_history", cutoff)
    h = h[~showmatch_mask(h) & (h.team_a_score != h.team_b_score)]
    h = h[h.team_a_player_ids.map(_five) & h.team_b_player_ids.map(_five)]
    rounds, rounds_a = _rounds(h.team_a_score, h.team_b_score)
    return pd.DataFrame({
        "day": pd.to_datetime(h.match_date).dt.normalize().to_numpy(),
        "match_id": h.match_id.astype(str).to_numpy(),
        "position": h.map_position_in_series.fillna(0).astype(int).to_numpy(),
        "source": "hltv",
        "team_a": ("h" + h.team_a_id.astype(int).astype(str)).to_numpy(),
        "team_b": ("h" + h.team_b_id.astype(int).astype(str)).to_numpy(),
        "lineup_a": [[f"h{int(p)}" for p in ids] for ids in h.team_a_player_ids],
        "lineup_b": [[f"h{int(p)}" for p in ids] for ids in h.team_b_player_ids],
        "rounds": rounds,
        "rounds_a": rounds_a,
        "team_a_won": h.team_a_won.astype(int).to_numpy(),
        "bo_type": h.bo_type.fillna(0).astype(int).to_numpy(),
        "division": np.nan,
    })


def faceit_key(link: pd.DataFrame):
    fmap = dict(zip(link.faceit_id, "h" + link.hltv_id.astype(int).astype(str)))

    def key(faceit_id: str) -> str:
        return fmap.get(faceit_id, "f" + faceit_id)

    return key


def faceit_maps(data_dir, cutoff: pd.Timestamp) -> pd.DataFrame:
    f = load(data_dir, "faceit/faceit_match_history", cutoff)
    fm = load(data_dir, "faceit/faceit_matches", cutoff, ["match_id", "competition_name"])
    # Both the date and the finish time must precede the cutoff so a truncated bundle yields the same rows.
    f = f[f.match_id.isin(fm.match_id)]
    ex = load(data_dir, "faceit/faceit_exclusions")
    whole = set(ex.match_id[ex.map_name.isna()])
    partial = set(zip(ex.match_id[ex.map_name.notna()], ex.map_name[ex.map_name.notna()]))
    f = f[~f.match_id.isin(whole)]
    f = f[[(m, n) not in partial for m, n in zip(f.match_id, f.map_name)]]
    f = f[f.team_a_score != f.team_b_score]
    f = f[f.team_a_faceit_ids.map(_five) & f.team_b_faceit_ids.map(_five)]
    key = faceit_key(load(data_dir, "faceit/hltv_faceit_map"))
    division = fm.set_index("match_id").competition_name.map(_division)
    rounds, rounds_a = _rounds(f.team_a_score, f.team_b_score)
    return pd.DataFrame({
        "day": pd.to_datetime(f.match_date).dt.normalize().to_numpy(),
        "match_id": f.match_id.astype(str).to_numpy(),
        "position": f.map_position_in_series.fillna(0).astype(int).to_numpy(),
        "source": "faceit",
        "team_a": [f"h{int(t)}" if pd.notna(t) else f"f{fid}" for t, fid in zip(f.team_a_id, f.team_a_faceit_id)],
        "team_b": [f"h{int(t)}" if pd.notna(t) else f"f{fid}" for t, fid in zip(f.team_b_id, f.team_b_faceit_id)],
        "lineup_a": [[key(p) for p in ids] for ids in f.team_a_faceit_ids],
        "lineup_b": [[key(p) for p in ids] for ids in f.team_b_faceit_ids],
        "rounds": rounds,
        "rounds_a": rounds_a,
        "team_a_won": f.team_a_won.astype(int).to_numpy(),
        "bo_type": f.bo_type.fillna(0).astype(int).to_numpy(),
        "division": f.match_id.map(division).to_numpy(float),
    })


def _division(name) -> float:
    m = _DIVISION_RE.search(name or "")
    return DIVISIONS[m.group(1).lower()] if m else np.nan


def faceit_levels(data_dir, cutoff: pd.Timestamp) -> dict[str, float]:
    """Each player's latest FACEIT skill level before the cutoff."""
    fm = load(data_dir, "faceit/faceit_matches", cutoff, ["match_id", "finished_at"])
    ro = load(data_dir, "faceit/faceit_rosters", cutoff, ["match_id", "faceit_id", "skill_level"])
    ro = ro[ro.skill_level.notna() & ro.match_id.isin(fm.match_id)]
    ro = ro.merge(fm, on="match_id").sort_values(["finished_at", "match_id", "faceit_id"], kind="stable")
    last = ro.drop_duplicates("faceit_id", keep="last")
    key = faceit_key(load(data_dir, "faceit/hltv_faceit_map"))
    return {key(f): float(l) for f, l in zip(last.faceit_id, last.skill_level)}


def build(data_dir, cutoff: pd.Timestamp, faceit: bool, window_days: int) -> Observations:
    cutoff = pd.Timestamp(cutoff)
    parts = [hltv_maps(data_dir, cutoff)]
    if faceit:
        parts.append(faceit_maps(data_dir, cutoff))
    maps = pd.concat(parts, ignore_index=True)
    maps = maps[maps.day >= cutoff - pd.Timedelta(days=window_days)]
    maps = maps.sort_values(["day", "source", "match_id", "position"], kind="stable").reset_index(drop=True)
    ranks = RankStrength(load(data_dir, "hltv/cs2_team_rankings", cutoff))
    levels = faceit_levels(data_dir, cutoff) if faceit else {}
    divisions: dict[str, float] = {}
    if faceit:
        fd = maps[maps.division.notna()]
        rows = pd.DataFrame({"player": [p for five in fd.lineup_a for p in five] + [p for five in fd.lineup_b for p in five],
                             "division": np.repeat(fd.division.to_numpy(), LINEUP).tolist() * 2})
        divisions = rows.groupby("player").division.mean().to_dict()
    latest: dict[str, list[str]] = {}
    for team, five in zip(maps.team_a, maps.lineup_a):
        latest[team] = five
    for team, five in zip(maps.team_b, maps.lineup_b):
        latest[team] = five
    return Observations(maps=maps, ranks=ranks, levels=levels, divisions=divisions, latest_lineup=latest)


def player_prior_inputs(obs: Observations) -> pd.DataFrame:
    """One row per player in the maps: the team and day of the player's first map in the window."""
    rows = []
    seen: set[str] = set()
    for r in obs.maps.itertuples():
        for five, team in ((r.lineup_a, r.team_a), (r.lineup_b, r.team_b)):
            for p in five:
                if p not in seen:
                    seen.add(p)
                    rows.append((p, team, r.day))
    return pd.DataFrame(rows, columns=["player", "team", "day"])


def rank_strength(obs: Observations, team: str, day: pd.Timestamp) -> float:
    if team.startswith("h"):
        return obs.ranks(int(team[1:]), day)
    return -np.log(UNRANKED)


def covariates(obs: Observations, players: list[str], teams: list[str], days: list[pd.Timestamp],
               mu: float | None = None, sd: float | None = None) -> tuple[np.ndarray, float, float]:
    """Prior covariates per player: standardized rank strength of the given team on the given day, the FACEIT skill
    level and ESEA division where known, with indicators for where they are known."""
    raw = np.array([rank_strength(obs, t, d) for t, d in zip(teams, days)])
    if mu is None:
        mu, sd = float(raw.mean()), float(raw.std()) or 1.0
    unranked = (raw <= -np.log(UNRANKED) + 1e-9).astype(float)
    level = np.array([obs.levels.get(p, np.nan) for p in players])
    division = np.array([obs.divisions.get(p, np.nan) for p in players])
    x = np.column_stack([
        (raw - mu) / sd,
        unranked,
        np.nan_to_num((level - 10.0) / 3.0),
        np.isfinite(level).astype(float),
        np.nan_to_num((division - 3.0) / 2.0),
        np.isfinite(division).astype(float),
    ])
    return x, mu, sd
