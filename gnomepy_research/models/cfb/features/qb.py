"""Quarterback identity and quality from play-by-play.

A team-game's starter is the first passer of the game. Passers are tied to CFBD player ids via
that season's roster (first initial + surname within the team), which lets a QB's history follow
him through transfers. Career PPA per dropback is shrunk toward the league mean.
"""
from __future__ import annotations

import pandas as pd

from gnomepy_research.models.cfb.features.source import SourceData
from gnomepy_research.models.cfb.features.team_talent import UNRATED, roster_ratings

# Dropbacks of evidence at which a QB's own PPA counts as much as the league mean.
QB_SHRINK_DROPBACKS = 150
_SUFFIX_RE = r"\s+(jr\.?|sr\.?|ii|iii|iv|v)$"


def team_name_ids(games: pd.DataFrame) -> dict[str, int]:
    names = pd.concat([
        games[["home_team", "home_id"]].set_axis(["team", "team_id"], axis=1),
        games[["away_team", "away_id"]].set_axis(["team", "team_id"], axis=1),
    ])
    return names.drop_duplicates("team").set_index("team")["team_id"].to_dict()


def roster_keys(rosters: pd.DataFrame) -> pd.DataFrame:
    last = rosters["last_name"].fillna("").str.lower().str.replace(_SUFFIX_RE, "", regex=True).str.split().str[-1]
    key = rosters["first_name"].fillna("").str[:1].str.lower() + "." + last.fillna("")
    out = rosters.assign(passer_key=key).dropna(subset=["team_id"])
    # Two same-initial same-surname players on one roster can't be told apart; leave them unlinked.
    out = out.drop_duplicates(["season", "team_id", "passer_key"], keep=False)
    return out[["season", "team_id", "passer_key", "player_id"]].assign(team_id=lambda d: d["team_id"].astype("int64"))


def dropbacks(src: SourceData) -> pd.DataFrame:
    """Visible dropbacks with team id and (where linkable) player id. Unlinked passers keep a
    team-scoped synthetic id so they still count within that team."""
    p = src.passers.assign(team_id=src.passers["offense"].map(team_name_ids(src.games)))
    p = p.dropna(subset=["team_id"]).assign(team_id=lambda d: d["team_id"].astype("int64"))
    p = p.merge(roster_keys(src.rosters), on=["season", "team_id", "passer_key"], how="left")
    synthetic = "t" + p["team_id"].astype(str) + ":" + p["passer_key"]
    p["qb_id"] = p["player_id"].astype("string").fillna(synthetic)
    return p


def game_starters(db: pd.DataFrame) -> pd.DataFrame:
    """First passer of each team-game."""
    first = db.sort_values("play_seq").drop_duplicates(["game_id", "team_id"])
    return first[["game_id", "season", "team_id", "qb_id", "passer"]]


def qb_career(db: pd.DataFrame) -> pd.DataFrame:
    league = db["ppa"].mean()
    c = db.groupby("qb_id").agg(qb_dropbacks=("ppa", "size"), qb_ppa_sum=("ppa", "sum"))
    c["qb_ppa_shrunk"] = (c["qb_ppa_sum"] + QB_SHRINK_DROPBACKS * league) / (c["qb_dropbacks"] + QB_SHRINK_DROPBACKS)
    return c.drop(columns="qb_ppa_sum")


def qb_recruit_rating(src: SourceData, season: int) -> pd.Series:
    ro = roster_ratings(src, season).drop_duplicates("player_id")
    return ro.set_index(ro["player_id"].astype(str))["rating"]


def preseason_qb(src: SourceData, season: int) -> pd.DataFrame:
    """Expected starter for the coming season: last season's primary passer if he's on this
    season's roster, else the rostered QB with the most career dropbacks, else the top-rated QB."""
    db = dropbacks(src)
    career = qb_career(db)
    ratings = qb_recruit_rating(src, season)
    roster = src.rosters[(src.rosters["season"] == season) & (src.rosters["position"] == "QB")].dropna(subset=["team_id"])
    roster = roster.assign(team_id=roster["team_id"].astype("int64"), qb_id=roster["player_id"].astype(str))

    prev = db[db["season"] == season - 1].groupby(["team_id", "qb_id"]).size().rename("n").reset_index()
    primary = (prev.sort_values(["n", "qb_id"], ascending=[False, True])
               .drop_duplicates("team_id").set_index("team_id")["qb_id"])

    rows = []
    for team_id, qbs in roster.groupby("team_id"):
        ids = set(qbs["qb_id"])
        prior = primary.get(team_id)
        if prior is not None and prior in ids:
            starter, returning = prior, True
        else:
            returning = False
            exp = career.reindex(list(ids))["qb_dropbacks"].fillna(0)
            if exp.max() > 0:
                starter = exp.idxmax()
            else:
                starter = ratings.reindex(list(ids)).fillna(UNRATED).idxmax()
        stats = career.reindex([starter]).iloc[0]
        rows.append({
            "team_id": team_id,
            "qb_expected_starter": starter,
            "qb_returning_starter": returning,
            "qb_prior_dropbacks": float(stats["qb_dropbacks"]) if pd.notna(stats["qb_dropbacks"]) else 0.0,
            "qb_prior_ppa": float(stats["qb_ppa_shrunk"]) if pd.notna(stats["qb_ppa_shrunk"]) else float(db["ppa"].mean()),
            "qb_recruit_rating": float(ratings.get(starter, UNRATED)),
        })
    return pd.DataFrame(rows)


def current_qb(src: SourceData, season: int) -> pd.DataFrame:
    """In-season: the starter of each team's most recent visible game, and his career to date."""
    db = dropbacks(src)
    career = qb_career(db)
    starters = game_starters(db[db["season"] == season])
    if starters.empty:
        return pd.DataFrame(columns=["team_id", "qb_current_starter", "qb_current_dropbacks", "qb_current_ppa",
                                     "qb_current_starts"])
    kickoff = src.games.set_index("game_id")["kickoff"]
    starters = starters.assign(kickoff=starters["game_id"].map(kickoff)).sort_values("kickoff")
    last = starters.drop_duplicates("team_id", keep="last").set_index("team_id")["qb_id"]
    starts = starters.groupby(["team_id", "qb_id"]).size()
    out = pd.DataFrame({"qb_current_starter": last})
    out["qb_current_dropbacks"] = last.map(career["qb_dropbacks"]).astype(float)
    out["qb_current_ppa"] = last.map(career["qb_ppa_shrunk"]).astype(float)
    out["qb_current_starts"] = [starts.get((t, q), 0) for t, q in last.items()]
    return out.rename_axis("team_id").reset_index()
