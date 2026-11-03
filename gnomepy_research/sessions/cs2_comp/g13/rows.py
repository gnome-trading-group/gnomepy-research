"""
One feature row per bo3 series, from the e12 skill fits at the series' weekly cutoff and its Polymarket path.

Rows for past series (the layer's training data) use the fives HLTV recorded for them and the series' own price
path; rows for target series use their announced fives and their pre-kickoff path. Both go through the same fits, so
a target's row is what its training row will be once it is history.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.bundle import load, target_paths
from gnomepy_research.sessions.cs2_comp.g13.e12.data import Inputs
from gnomepy_research.sessions.cs2_comp.g13.e12.features import feature_rows
from gnomepy_research.sessions.cs2_comp.g13.e12.predict import CONFIG_FACEIT, CONFIG_HLTV, LAYER_FACEIT, LAYER_HLTV, design
from gnomepy_research.sessions.cs2_comp.g13.market import PATH_COLUMNS, path_features

WEEK = pd.Timedelta(days=7)


def monday(t: pd.Timestamp) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return (t - pd.Timedelta(days=t.weekday())).normalize()


def _naive(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    return s.dt.tz_convert(None) if s.dt.tz is not None else s


def market_cutoffs(data_dir, until: pd.Timestamp) -> list[pd.Timestamp]:
    """Weekly cutoffs from the week of the first priced series through `until` (itself a Monday)."""
    kick = _naive(load(data_dir, "market/series_paths", until, ["kickoff"]).kickoff)
    first = monday(kick.min())
    return list(pd.date_range(first, until, freq=WEEK))


def history_frame(inputs: Inputs, cutoffs: list[pd.Timestamp]) -> pd.DataFrame:
    """Past bo3 series with an outcome, each assigned to the cutoff of its week, with HLTV's fives for it."""
    s = inputs.series[inputs.series.bo_type == 3]
    parts = [s[(s.time >= c) & (s.time < c + WEEK)].assign(week_cutoff=c) for c in cutoffs]
    cols = ["match_id", "team_a", "team_b", "week_cutoff", "y", "known", "time", "is_lan", "lineup_a", "lineup_b", "sub_a", "sub_b"]
    return pd.concat(parts, ignore_index=True)[cols]


def target_frame(targets: pd.DataFrame) -> pd.DataFrame:
    t = targets.rename(columns={"team_a_id": "team_a", "team_b_id": "team_b"})[["match_id", "team_a", "team_b", "week_cutoff", "is_lan"]].copy()
    t["week_cutoff"] = _naive(t.week_cutoff)
    t["time"] = _naive(targets.match_time).to_numpy()
    for side in ("a", "b"):
        t[f"lineup_{side}"] = targets[f"team_{side}_lineup_ids"].to_numpy()
        t[f"sub_{side}"] = [0 if v is None else len(v) for v in targets[f"team_{side}_standin_ids"]]
    return t


def skill_rows(inputs: Inputs, frame: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """e12's two fits per cutoff, summarised to the layer's inputs plus e12's own price."""
    h = feature_rows(inputs, frame, CONFIG_HLTV, False, verbose=verbose)
    f = feature_rows(inputs, frame, CONFIG_FACEIT, True, verbose=verbose)
    X = design(h, f)
    out = h[[c for c in frame.columns if c not in ("lineup_a", "lineup_b")]].copy()
    out["z_e12"] = X[list(LAYER_FACEIT)].to_numpy() @ np.array(list(LAYER_FACEIT.values()))
    out["z_e12h"] = X[list(LAYER_HLTV)].to_numpy() @ np.array(list(LAYER_HLTV.values()))
    out["zh"], out["zf"] = X.zh.to_numpy(), X.zf.to_numpy()
    out["thin"] = (h.thin_a + h.thin_b).to_numpy(float)
    out["thin_f"] = (f.thin_a + f.thin_b).to_numpy(float)
    out["new"] = (h.new_a + h.new_b).to_numpy(float)
    out["standin"] = (h.standin_a - h.standin_b).to_numpy(float)
    out["idle"] = np.log1p(h.idle_a.to_numpy(float)) - np.log1p(h.idle_b.to_numpy(float))
    out["linked"] = np.minimum(f.linked_a, f.linked_b).to_numpy(float)
    out["log_rank"] = np.log(h.rank_a.to_numpy(float)) - np.log(h.rank_b.to_numpy(float))
    out["unranked"] = (h.rank_a >= 300).astype(float).to_numpy() + (h.rank_b >= 300).astype(float).to_numpy()
    return out


def with_paths(rows: pd.DataFrame, paths: pd.DataFrame) -> pd.DataFrame:
    """Attach path features; a series without prints keeps NaN in every path column."""
    return rows.merge(path_features(paths), on="match_id", how="left")


def history_rows(data_dir, inputs: Inputs, cutoffs: list[pd.Timestamp], verbose: bool = False) -> pd.DataFrame:
    frame = history_frame(inputs, cutoffs)
    rows = skill_rows(inputs, frame, verbose)
    paths = load(data_dir, "market/series_paths", cutoffs[-1] + WEEK)
    return with_paths(rows, paths[paths.match_id.isin(rows.match_id)])


def target_rows(data_dir, inputs: Inputs, targets: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    rows = skill_rows(inputs, target_frame(targets), verbose)
    return with_paths(rows, target_paths(data_dir, rows.match_id))
