"""Pregame quarterback adjustment: announced starter vs the QB play a team's ratings were learned from."""
from __future__ import annotations

import pandas as pd

from gnomepy_research.models.cfb.features.cutoffs import week_cutoffs
from gnomepy_research.models.cfb.features.qb import dropbacks, game_starters, preseason_qb
from gnomepy_research.models.cfb.features.source import SourceData

REPLACEMENT_DROPBACKS = 150


def _ordered(db: pd.DataFrame) -> pd.DataFrame:
    return db.sort_values(["kickoff", "play_seq"], kind="stable")


def replacement_ppa(db: pd.DataFrame) -> float:
    d = _ordered(db)
    early = d[d.groupby("qb_id").cumcount() < REPLACEMENT_DROPBACKS]
    return float(early["ppa"].mean())


def qb_quality(db: pd.DataFrame) -> pd.Series:
    rep = replacement_ppa(db)
    g = db.groupby("qb_id")["ppa"].agg(["sum", "size"])
    return ((g["sum"] + REPLACEMENT_DROPBACKS * rep) / (g["size"] + REPLACEMENT_DROPBACKS)).rename("qb_quality")


def team_qb_baseline(db_season: pd.DataFrame, quality: pd.Series) -> pd.Series:
    n = db_season.groupby(["team_id", "qb_id"]).size().rename("n").reset_index()
    n["q"] = n["qb_id"].map(quality)
    n["w"] = n["n"] * n["q"]
    g = n.groupby("team_id")[["w", "n"]].sum()
    return (g["w"] / g["n"]).rename("qb_baseline")


def historical_starters(src_full: SourceData) -> pd.DataFrame:
    """Actual first passer of each team-game; a documented look-ahead exception (starter treated as known pregame)."""
    return game_starters(dropbacks(src_full))[["game_id", "team_id", "qb_id"]]


def qb_deltas(src_full: SourceData, games: pd.DataFrame, starters: pd.DataFrame) -> pd.DataFrame:
    cuts = week_cutoffs(games)
    g = games.merge(cuts, on=["season", "season_type", "week"])
    start = starters.set_index(["game_id", "team_id"])["qb_id"]
    rows = []
    for (season, cutoff), wk in g.groupby(["season", "cutoff"]):
        view = src_full.as_of(cutoff)
        db = dropbacks(view)
        db = db.assign(kickoff=db["game_id"].map(view.games.set_index("game_id")["kickoff"]))
        if db.empty:
            rows.extend({"game_id": r.game_id, "home_qb_delta": 0.0, "away_qb_delta": 0.0} for r in wk.itertuples())
            continue
        quality, rep = qb_quality(db), replacement_ppa(db)
        base = team_qb_baseline(db[db["season"] == season], quality)
        pre = preseason_qb(view, season).set_index("team_id")["qb_expected_starter"]

        def delta(game_id: int, team_id: int) -> float:
            qb = start.get((game_id, team_id))
            if qb is None:
                return 0.0
            baseline = base.get(team_id)
            if baseline is None:
                baseline = quality.get(pre.get(team_id), rep)
            return float(quality.get(qb, rep) - baseline)

        for r in wk.itertuples():
            rows.append({"game_id": r.game_id, "home_qb_delta": delta(r.game_id, r.home_id),
                         "away_qb_delta": delta(r.game_id, r.away_id)})
    return pd.DataFrame(rows, columns=["game_id", "home_qb_delta", "away_qb_delta"])
