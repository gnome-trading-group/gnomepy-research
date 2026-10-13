"""Per team-game form stats beyond PPA: early-down success, luck on late downs and drive finishing,
field position (special teams + turnovers, which PPA ignores) and pace. Luck terms are signed so
positive means running hot; they should regress."""
from __future__ import annotations

import numpy as np
import pandas as pd

from gnomepy_research.models.cfb.features.qb import team_name_ids

FORM_FEATURES = ["early_success", "late_down_luck", "finishing_luck", "field_position", "pace"]
QUALITY_DRIVE_YTG = 40
GARBAGE_MARGIN = {2: 38, 3: 28, 4: 22}
SUCCESS_SHARE = {1: 0.5, 2: 0.7, 3: 1.0, 4: 1.0}
FORM_SHRINK_GAMES = 3.0


def is_success(down, distance, yards_gained) -> np.ndarray:
    share = pd.Series(np.asarray(down)).map(SUCCESS_SHARE).fillna(1.0).to_numpy()
    return np.asarray(yards_gained) >= share * np.asarray(distance)


def is_garbage(period, offense_score, defense_score) -> np.ndarray:
    limit = pd.Series(np.asarray(period)).map(GARBAGE_MARGIN).fillna(np.inf).to_numpy()
    return np.abs(np.asarray(offense_score) - np.asarray(defense_score)) > limit


def team_game_form(plays: pd.DataFrame, drives: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    ids = team_name_ids(games)
    p = plays[plays["ppa"].notna() & plays["down"].between(1, 4)].copy()
    p = p[~is_garbage(p["period"].to_numpy(), p["offense_score"].to_numpy(), p["defense_score"].to_numpy())]
    p["team_id"] = p["offense"].map(ids)
    p["success"] = is_success(p["down"].to_numpy(), p["distance"].to_numpy(), p["yards_gained"].to_numpy())
    keys = ["game_id", "team_id"]
    early = p[p["down"] <= 2].groupby(keys)["success"].mean().rename("early_success")
    late = p[p["down"] >= 3].groupby(keys)["success"].mean().rename("late_success")
    pace = p.groupby(keys).size().rename("pace").astype(float)

    d = drives.dropna(subset=["offense_id", "defense_id"]).copy()
    quality = (d["end_yards_to_goal"] <= QUALITY_DRIVE_YTG) | (d["off_points"] > 0)
    q = d[quality].groupby(["game_id", "offense_id"])["off_points"].mean().rename_axis(keys).rename("pts_per_qd")
    own = d.groupby(["game_id", "offense_id"])["start_yards_to_goal"].mean().rename_axis(keys)
    opp = d.groupby(["game_id", "defense_id"])["start_yards_to_goal"].mean().rename_axis(keys)
    fp = (opp - own).rename("field_position")

    parts = [s.astype(float) for s in (early, late, pace, q, fp)]
    out = pd.concat([s.reset_index().set_index(keys) for s in parts], axis=1).reset_index()
    out["team_id"] = out["team_id"].astype("int64")
    g = games.set_index("game_id")
    out["season"] = out["game_id"].map(g["season"])
    out["available_at"] = out["game_id"].map(g["kickoff"]) + pd.Timedelta(days=1)
    return out


def with_luck(rows: pd.DataFrame) -> pd.DataFrame:
    """Luck residuals against league relationships fit on `rows` only (pass point-in-time rows)."""
    r = rows.copy()
    ok = r[["early_success", "late_success"]].notna().all(axis=1)
    if ok.sum() > 2 and r.loc[ok, "early_success"].nunique() > 1:
        b, a = np.polyfit(r.loc[ok, "early_success"], r.loc[ok, "late_success"], 1)
    else:
        b, a = 0.0, r["late_success"].mean()
    r["late_down_luck"] = r["late_success"] - (a + b * r["early_success"])
    r["finishing_luck"] = r["pts_per_qd"] - r["pts_per_qd"].mean()
    return r


def team_form(form_rows: pd.DataFrame, season: int, shrink_games: float = FORM_SHRINK_GAMES) -> pd.DataFrame:
    r = form_rows[form_rows["season"] == season]
    if r.empty:
        return pd.DataFrame(columns=FORM_FEATURES)
    league = r[FORM_FEATURES].mean()
    g = r.groupby("team_id")[FORM_FEATURES]
    n = g.count()
    shrunk = (g.sum() + shrink_games * league) / (n + shrink_games)
    sd = shrunk.std().replace(0, np.nan)
    return ((shrunk - shrunk.mean()) / sd).fillna(0.0)
