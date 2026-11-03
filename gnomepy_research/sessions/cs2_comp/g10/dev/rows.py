"""Feature rows for g10: every past market-joined bo3 series (the layer's training set) and the targets, each priced
by the e12 skill fits at its own week's Monday cutoff, plus a summary of its Polymarket path up to kickoff.

A past series enters the training set of week W only when it started at least `SETTLE` before W, so its result and
its whole pre-kickoff path were known by W whichever later weeks the run also covers."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit

from gnomepy_research.sessions.cs2_comp.bundle import load, target_paths
from gnomepy_research.sessions.cs2_comp.g10.e12_data import Inputs
from gnomepy_research.sessions.cs2_comp.g10.e12_predict import LAYER_FACEIT, LAYER_HLTV, design, rows
from gnomepy_research.sessions.cs2_comp.g10.dev.paths import path_features

SETTLE = pd.Timedelta(hours=12)


def monday(ts: pd.Series) -> pd.Series:
    d = pd.to_datetime(ts).dt.normalize()
    return d - pd.to_timedelta(d.dt.weekday, unit="D")


def _naive(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    return s.dt.tz_convert(None) if s.dt.tz is not None else s


def _history(inputs: Inputs, until: pd.Timestamp, data_dir) -> pd.DataFrame:
    paths = load(data_dir, "market/series_paths", until)
    paths = paths.assign(kickoff=_naive(paths.kickoff), t=_naive(paths.t))
    paths = paths[paths.kickoff < until - SETTLE]
    s = inputs.series
    s = s[(s.bo_type == 3) & (s.time < until - SETTLE) & s.match_id.isin(paths.match_id)].copy()
    pf = path_features(paths[paths.match_id.isin(s.match_id)])
    s = s.merge(pf, on="match_id")
    s["week_cutoff"] = monday(s.time)
    s["sub_a"] = 0
    s["sub_b"] = 0
    return s[["match_id", "time", "team_a", "team_b", "week_cutoff", "lineup_a", "lineup_b", "sub_a", "sub_b", "y"]
             + [c for c in pf.columns if c != "match_id"]]


def _targets(targets: pd.DataFrame, data_dir) -> pd.DataFrame:
    t = targets.rename(columns={"team_a_id": "team_a", "team_b_id": "team_b"})
    t = t.assign(time=_naive(t.match_time), lineup_a=t.team_a_lineup_ids.to_numpy(), lineup_b=t.team_b_lineup_ids.to_numpy(),
                 sub_a=[0 if v is None else len(v) for v in t.team_a_standin_ids],
                 sub_b=[0 if v is None else len(v) for v in t.team_b_standin_ids], y=np.nan)
    paths = target_paths(data_dir, t.match_id)
    paths = paths.assign(kickoff=_naive(paths.kickoff), t=_naive(paths.t))
    pf = path_features(paths)
    t = t.merge(pf, on="match_id", how="left")
    return t[["match_id", "time", "team_a", "team_b", "week_cutoff", "lineup_a", "lineup_b", "sub_a", "sub_b", "y"]
             + [c for c in pf.columns if c != "match_id"]]


def build(data_dir, targets: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """One row per past series (is_target False) and per target, with the e12 rows, q_e12 and path features."""
    until = pd.Timestamp(targets.week_cutoff.max())
    inputs = Inputs(data_dir, until, True)
    hist = _history(inputs, until, data_dir).assign(is_target=False)
    tg = _targets(targets, data_dir).assign(is_target=True)
    frame = pd.concat([hist, tg], ignore_index=True)
    frame["week_cutoff"] = pd.to_datetime(frame.week_cutoff)
    frame["key"] = np.arange(len(frame))
    t = frame[["key", "team_a", "team_b", "week_cutoff", "lineup_a", "lineup_b", "sub_a", "sub_b"]].rename(columns={"key": "match_id"})
    h, f = rows(inputs, t, True, verbose)
    q_h = expit(design(h, None)[list(LAYER_HLTV)].to_numpy() @ np.array(list(LAYER_HLTV.values())))
    q_f = expit(design(h, f)[list(LAYER_FACEIT)].to_numpy() @ np.array(list(LAYER_FACEIT.values())))
    keep_h = ["gap", "gap_last", "thin_a", "thin_b", "new_a", "new_b", "idle_a", "idle_b", "rank_a", "rank_b",
              "standin_a", "standin_b"]
    out = pd.DataFrame({"key": h.match_id.to_numpy(), "q_e12": q_f, "q_e12h": q_h})
    for c in keep_h:
        out[f"h_{c}"] = h[c].to_numpy()
    for c in ["gap", "thin_a", "thin_b", "linked_a", "linked_b"]:
        out[f"f_{c}"] = f.set_index("match_id").loc[h.match_id, c].to_numpy()
    frame = frame.drop(columns=["lineup_a", "lineup_b"]).merge(out, on="key").drop(columns="key")
    return frame
