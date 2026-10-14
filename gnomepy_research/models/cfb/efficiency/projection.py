"""Preseason projection of each team's per-unit efficiency ratings.

Season S's rating for a unit (pass/rush x offense/defense) is projected from season S-1's rating,
interacted with how much of the producing personnel returns, plus QB, portal, coaching, talent and
conference context. One ridge regression per unit, fitted rolling-origin on (s-1 -> s) pairs whose
target season s is < S. Targets are full-season opponent-adjusted ridge ratings of that unit.

    AWS_PROFILE=... poetry run python -m gnomepy_research.models.cfb.efficiency.projection 2020 2024
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from gnomepy_research.models.cfb.efficiency.data import _with_conference_means
from gnomepy_research.models.cfb.efficiency.inference import Inputs
from gnomepy_research.models.cfb.features.physical import SEASON_START_GRACE, blue_chip_ratio
from gnomepy_research.models.cfb.features.ratings import FCS_TEAM, ridge_ratings
from gnomepy_research.models.cfb.features.source import SourceData, season_starts
from gnomepy_research.models.cfb.features.team_talent import roster_ratings

FIRST_DATA_SEASON = 2018
RIDGE_ALPHA = 1.0
LOW_CONTINUITY = 0.4
UNIT_PLAYS = {"pass": ("pass_plays", "off_pass_ppa"), "rush": ("rush_plays", "off_rush_ppa")}
SIDES = ("off", "def")
DEF_POSITIONS = {"DL", "DT", "DE", "NT", "LB", "ILB", "OLB", "DB", "CB", "S", "FS", "SS"}
# Returning-share columns per (unit, side); each enters alone and interacted with last season's rating.
SHARES = {
    ("pass", "off"): ["returning_pct_passing_ppa", "returning_pct_receiving_ppa"],
    ("rush", "off"): ["returning_pct_rushing_ppa"],
    ("pass", "def"): ["def_continuity"],
    ("rush", "def"): ["def_continuity"],
}
COMMON = ["coach_new", "portal_in_stars", "portal_out_stars", "talent", "blue_chip_ratio", "conf_recruiting"]
SIDE_EXTRA = {"off": ["qb_prior_ppa", "conf_prior_off", "conf_talent_offense"],
              "def": ["conf_prior_def", "conf_talent_defense"]}
RESULTS_DIR = Path(__file__).resolve().parents[3] / "sessions" / "cfb_win_probability" / "results" / "backtest"


def last_col(unit: str, side: str) -> str:
    return f"last_{unit}_{side}"


def target_col(unit: str, side: str) -> str:
    return f"target_{unit}_{side}"


def unit_ridge_ratings(eff: pd.DataFrame, unit: str) -> pd.DataFrame:
    """ratings.ridge_ratings on one unit's plays and PPA (higher def_rating = better defense)."""
    plays, ppa = UNIT_PLAYS[unit]
    e = eff[eff["usable"] & (eff[plays].astype(float) > 0) & np.isfinite(eff[ppa].astype(float))]
    e = e.drop(columns=["plays", "off_ppa"], errors="ignore").rename(columns={plays: "plays", ppa: "off_ppa"})
    out, _ = ridge_ratings(e)
    return out


def season_unit_ratings(efficiency: pd.DataFrame, seasons: list[int]) -> pd.DataFrame:
    """One row per (season, FBS team) with columns {unit}_{side} of full-season unit ratings."""
    frames = []
    for s in seasons:
        e = efficiency[efficiency["season"] == s]
        if e.empty:
            continue
        wide = None
        for unit in UNIT_PLAYS:
            r = unit_ridge_ratings(e, unit)
            r = r[r["team_id"] != FCS_TEAM].set_index("team_id")[["off_rating", "def_rating"]]
            r.columns = [f"{unit}_off", f"{unit}_def"]
            wide = r if wide is None else wide.join(r, how="outer")
        frames.append(wide.rename_axis("team_id").reset_index().assign(season=s))
    cols = ["season", "team_id", *[f"{u}_{sd}" for u in UNIT_PLAYS for sd in SIDES]]
    return pd.concat(frames, ignore_index=True)[cols] if frames else pd.DataFrame(columns=cols)


def def_continuity(src: SourceData, season: int) -> pd.Series:
    """Recruiting-rating-weighted share of last season's defensive roster still on the team's roster."""
    prev = roster_ratings(src, season - 1)
    prev = prev[prev["position"].isin(DEF_POSITIONS)]
    cur = src.rosters[src.rosters["season"] == season].dropna(subset=["team_id"])
    on_roster = set(zip(cur["team_id"].astype("int64"), cur["player_id"].astype(str)))
    back = np.array([(t, p) in on_roster for t, p in zip(prev["team_id"], prev["player_id"].astype(str))])
    w = prev.assign(back_w=prev["rating"] * back)
    g = w.groupby("team_id")
    return (g["back_w"].sum() / g["rating"].sum()).rename("def_continuity")


def season_extras(src: SourceData, season: int) -> pd.DataFrame:
    pit = src.as_of(season_starts(src.games)[season] + SEASON_START_GRACE)
    out = pd.DataFrame(def_continuity(pit, season)).join(blue_chip_ratio(pit.recruits, season), how="outer")
    return out.rename_axis("team_id").reset_index().assign(season=season)


def projection_table(preseason: pd.DataFrame, src: SourceData, games: pd.DataFrame, seasons: list[int]) -> pd.DataFrame:
    """Raw projection inputs (and targets where the season has been played) per (season, FBS team)."""
    pre = preseason[preseason["season"].isin(seasons)].copy()
    pre = _with_conference_means(pre, games, seasons)
    extras = pd.concat([season_extras(src, s) for s in seasons], ignore_index=True)
    pre = pre.merge(extras, on=["season", "team_id"], how="left")
    ratings = season_unit_ratings(src.efficiency, sorted({s - 1 for s in seasons} | set(seasons)))
    unit_cols = [c for c in ratings.columns if c not in ("season", "team_id")]
    last = ratings.assign(season=ratings["season"] + 1).rename(columns={c: f"last_{c}" for c in unit_cols})
    target = ratings.rename(columns={c: f"target_{c}" for c in unit_cols})
    pre = pre.merge(last, on=["season", "team_id"], how="left").merge(target, on=["season", "team_id"], how="left")
    return pre.sort_values(["season", "team_id"]).reset_index(drop=True)


def feature_names(unit: str, side: str) -> list[str]:
    last = last_col(unit, side)
    shares = SHARES[(unit, side)]
    names = [last, *shares, *[f"{last}*{s}" for s in shares]]
    if (unit, side) == ("pass", "off"):
        names += ["qb_returning_starter", f"{last}*qb_returning_starter"]
    return names + COMMON + SIDE_EXTRA[side]


def design(table: pd.DataFrame, unit: str, side: str) -> pd.DataFrame:
    """Features z-scored within season (missing -> 0, the season mean)."""
    t = table.copy()
    for c in ("qb_returning_starter", "coach_new"):
        t[c] = t[c].astype(float)
    out = pd.DataFrame(index=t.index)
    for name in feature_names(unit, side):
        if "*" in name:
            a, b = name.split("*")
            v = t[a].astype(float) * t[b].astype(float)
        else:
            v = t[name].astype(float)
        mean = v.groupby(t["season"]).transform("mean")
        sd = v.groupby(t["season"]).transform("std").replace(0, np.nan)
        out[name] = ((v - mean) / sd).fillna(0.0)
    return out


def returning_share(table: pd.DataFrame, unit: str, side: str) -> pd.Series:
    s = table[SHARES[(unit, side)]].astype(float).mean(axis=1)
    return s.fillna(s.groupby(table["season"]).transform("median")).fillna(0.5).clip(0.0, 1.0)


def fit_unit(table: pd.DataFrame, season: int, unit: str, side: str) -> tuple[Ridge, float, int] | None:
    """Ridge on pairs with target season < `season`; None when there are none."""
    tgt = target_col(unit, side)
    train = table[(table["season"] < season) & table[tgt].notna() & table[last_col(unit, side)].notna()]
    if train.empty:
        return None
    x = design(table, unit, side).loc[train.index]
    m = Ridge(alpha=RIDGE_ALPHA).fit(x.to_numpy(), train[tgt].to_numpy(float))
    resid = train[tgt].to_numpy(float) - m.predict(x.to_numpy())
    return m, float(np.sqrt(np.mean(resid ** 2))), len(train)


def project(table: pd.DataFrame, season: int) -> pd.DataFrame:
    """[season, team_id, unit, side, proj, proj_sd, returning_share]; proj is NaN when no training pairs exist."""
    rows = table[table["season"] == season]
    frames = []
    for unit in UNIT_PLAYS:
        for side in SIDES:
            share = returning_share(rows, unit, side)
            fit = fit_unit(table, season, unit, side)
            if fit is None:
                proj, sd = np.full(len(rows), np.nan), np.full(len(rows), np.nan)
            else:
                m, resid_sd, _ = fit
                proj = m.predict(design(table, unit, side).loc[rows.index].to_numpy())
                sd = resid_sd * (1.0 + (1.0 - share.to_numpy()))
            frames.append(pd.DataFrame({"season": season, "team_id": rows["team_id"].to_numpy(), "unit": unit,
                                        "side": side, "proj": proj, "proj_sd": sd,
                                        "returning_share": share.to_numpy()}))
    return pd.concat(frames, ignore_index=True)


def project_seasons(table: pd.DataFrame, seasons: list[int]) -> pd.DataFrame:
    return pd.concat([project(table, s) for s in seasons], ignore_index=True)


def r2(y: np.ndarray, p: np.ndarray) -> float:
    return float(1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def evaluate(table: pd.DataFrame, seasons: list[int]) -> tuple[pd.DataFrame, dict]:
    """Out-of-sample R^2 per (unit, side, season), full model vs last-rating only; latest fit's coefficients."""
    rows, coefs = [], {}
    for unit in UNIT_PLAYS:
        for side in SIDES:
            tgt, last = target_col(unit, side), last_col(unit, side)
            x_full = design(table, unit, side)
            for s in seasons:
                train = table[(table["season"] < s) & table[tgt].notna() & table[last].notna()].index
                test = table[(table["season"] == s) & table[tgt].notna() & table[last].notna()].index
                if train.empty or test.empty:
                    continue
                y_tr, y_te = table.loc[train, tgt].to_numpy(float), table.loc[test, tgt].to_numpy(float)
                full = Ridge(alpha=RIDGE_ALPHA).fit(x_full.loc[train].to_numpy(), y_tr)
                base = Ridge(alpha=RIDGE_ALPHA).fit(x_full.loc[train, [last]].to_numpy(), y_tr)
                rows.append({"unit": unit, "side": side, "season": s, "n_train": len(train), "n_test": len(test),
                             "r2_full": r2(y_te, full.predict(x_full.loc[test].to_numpy())),
                             "r2_last_only": r2(y_te, base.predict(x_full.loc[test, [last]].to_numpy()))})
                if s == max(seasons):
                    coefs[f"{unit}_{side}"] = dict(zip(x_full.columns, full.coef_.round(4)))
    return pd.DataFrame(rows), coefs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("first", type=int)
    ap.add_argument("last", type=int)
    args = ap.parse_args()
    inputs = Inputs.load()
    src = SourceData.load()
    table = projection_table(inputs.preseason, src, inputs.games, list(range(FIRST_DATA_SEASON + 1, args.last + 1)))
    res, coefs = evaluate(table, list(range(args.first, args.last + 1)))
    lines = ["Preseason projection: out-of-sample R^2 (rolling origin, Ridge alpha=1 on within-season z-scored features)",
             "", res.to_string(index=False, float_format=lambda v: f"{v:.3f}"), "",
             "Mean R^2 by unit:", res.groupby(["unit", "side"])[["r2_full", "r2_last_only"]].mean().round(3).to_string(),
             "", f"Coefficients of the latest fit (target season {args.last}; per 1 sd of feature):"]
    for k, c in coefs.items():
        lines += ["", k, *[f"  {n:45s} {v:+.4f}" for n, v in c.items()]]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "projection_eval.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
