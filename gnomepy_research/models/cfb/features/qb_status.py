"""Pregame quarterback adjustment: announced starter vs the QB play a team's ratings were learned from."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from gnomepy_research.models.cfb.features.cutoffs import week_cutoffs
from gnomepy_research.models.cfb.features.qb import dropbacks, game_starters, preseason_qb, roster_keys, team_name_ids
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


OVERRIDE_COLUMNS = ["game_id", "team_id", "qb_id", "recorded_at"]


def _blank(v) -> bool:
    return pd.isna(v) or not str(v).strip()


def _truthy(v) -> bool:
    return not pd.isna(v) and str(v).strip().lower() in {"1", "true", "yes"}


def load_qb_status(path: Path, src_full: SourceData) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype={"season_type": str, "team": str, "passer_key": str, "allow_unknown": str})
    if raw.empty:
        return pd.DataFrame({"game_id": pd.Series(dtype="int64"), "team_id": pd.Series(dtype="int64"),
                             "qb_id": pd.Series(dtype="string"),
                             "recorded_at": pd.Series(dtype="datetime64[ns, UTC]")})
    ids = team_name_ids(src_full.games)
    keys = roster_keys(src_full.rosters)
    games = src_full.games
    rows = []
    for n, r in enumerate(raw.itertuples(index=False), start=1):
        team = str(r.team).strip()
        if team not in ids:
            raise ValueError(f"qb_status row {n}: unknown team {team!r}")
        team_id = int(ids[team])
        if _blank(r.passer_key) or _blank(r.season_type):
            raise ValueError(f"qb_status row {n}: passer_key and season_type must be non-blank (team {team!r})")
        hit = games[(games["season"] == r.season) & (games["season_type"] == r.season_type.strip())
                    & (games["week"] == r.week) & ((games["home_id"] == team_id) | (games["away_id"] == team_id))]
        if len(hit) != 1:
            raise ValueError(f"qb_status row {n}: {len(hit)} games match {team!r} "
                             f"{r.season} {r.season_type} week {r.week}, expected 1")
        key = r.passer_key.strip().lower()
        match = keys[(keys["season"] == r.season) & (keys["team_id"] == team_id) & (keys["passer_key"] == key)]
        if len(match):
            qb_id = str(match["player_id"].iloc[0])
        elif _truthy(getattr(r, "allow_unknown", None)):
            qb_id = f"t{team_id}:{key}"
        else:
            raise ValueError(f"qb_status row {n}: passer_key {key!r} is not on the {r.season} roster of {team!r}; "
                             f"set allow_unknown=1 to use a synthetic id")
        rows.append({"game_id": int(hit["game_id"].iloc[0]), "team_id": team_id, "qb_id": qb_id,
                     "recorded_at": pd.Timestamp(r.recorded_at, tz="UTC") if pd.Timestamp(r.recorded_at).tzinfo is None
                     else pd.Timestamp(r.recorded_at).tz_convert("UTC")})
    out = pd.DataFrame(rows, columns=OVERRIDE_COLUMNS)
    out = out.sort_values("recorded_at", kind="stable").drop_duplicates(["game_id", "team_id"], keep="last")
    return out.sort_index().reset_index(drop=True)


def apply_overrides(starters: pd.DataFrame, overrides: pd.DataFrame) -> pd.DataFrame:
    keep = starters.merge(overrides[["game_id", "team_id"]], on=["game_id", "team_id"], how="left", indicator=True)
    keep = keep[keep["_merge"] == "left_only"].drop(columns="_merge")
    return pd.concat([keep, overrides[["game_id", "team_id", "qb_id"]]], ignore_index=True)
