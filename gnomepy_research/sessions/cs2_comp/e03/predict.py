"""Pre-match series probability with FACEIT player form (competitor e03).

    python -m gnomepy_research.sessions.cs2_comp.e03.predict --data <bundle> --targets <parquet> --out <parquet> [--no-faceit]

Each week's targets are fit on series that finished before that week's Monday, and every feature is read from rows
dated before the same cutoff. A run on a bundle truncated at one week's cutoff therefore reproduces that week's rows.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.hltv.config import showmatch_mask
from gnomepy_research.sessions.cs2_comp.bundle import load
from gnomepy_research.sessions.cs2_comp.e03 import elo_path, faceit_form, model

TRAIN_WINDOW_DAYS = 365
LABEL_LAG_DAYS = 2
RIDGE = 5.0
PROB_CLIP = 0.01
BEST_OF_THREE = 3
COVERAGE_PLAYERS = 5
MIN_LINKED_PER_LINEUP = 1
HLTV_FEATURES = ["elo_diff", "rank_diff"]
FACEIT_METRICS = ["vol30", "vol90", "rusty", "newish", "form", "kd", "adr", "skill"]
FACEIT_DIFFS = [f"{m}_d" for m in FACEIT_METRICS]
FACEIT_FEATURES = FACEIT_DIFFS + ["link_d", "faceit_missing"]
HIST_COLUMNS = [
    "match_id", "match_date", "match_time", "event_name", "format", "bo_type", "map_position_in_series",
    "map_name", "team_a_id", "team_b_id", "team_a_score", "team_b_score", "team_a_h1_score", "team_b_h1_score",
    "team_a_started_ct", "team_a_won", "team_a_player_ids", "team_b_player_ids", "team_a_lineup_ids",
    "team_b_lineup_ids",
]


def _naive(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    return s.dt.tz_convert(None) if s.dt.tz is not None else s


def _maps(data_dir: Path, max_cut: pd.Timestamp) -> pd.DataFrame:
    hist = load(data_dir, "hltv/cs2_match_history", max_cut, HIST_COLUMNS)
    hist = hist[~showmatch_mask(hist)].dropna(
        subset=["team_a_id", "team_b_id", "team_a_score", "team_b_score", "team_a_won"]).copy()
    hist["match_date"] = _naive(hist["match_date"])
    hist["match_time"] = _naive(hist["match_time"].fillna(hist["match_date"]))
    return hist


def _rankings(data_dir: Path, max_cut: pd.Timestamp) -> pd.DataFrame:
    r = load(data_dir, "hltv/cs2_team_rankings", max_cut, ["date", "team_id", "points"])
    r = r.assign(date=_naive(r["date"]).astype("datetime64[ns]"), team_id=r["team_id"].astype(np.int64))
    return r.dropna(subset=["points"]).sort_values("date")


def _hltv_to_faceit(data_dir: Path) -> dict[int, str]:
    m = load(data_dir, "faceit/hltv_faceit_map", None, ["hltv_id", "faceit_id"])
    m = m.drop_duplicates("hltv_id")
    return dict(zip(m["hltv_id"].astype(np.int64), m["faceit_id"].astype(str)))


def _latest_points(rankings: pd.DataFrame, asof: pd.Series, team_id: pd.Series) -> np.ndarray:
    left = pd.DataFrame({
        "row": np.arange(len(asof)),
        "asof": pd.to_datetime(asof.to_numpy()).astype("datetime64[ns]"),
        "team_id": team_id.to_numpy(np.int64),
    }).sort_values("asof")
    right = rankings.rename(columns={"date": "rdate"})[["rdate", "team_id", "points"]].sort_values("rdate")
    m = pd.merge_asof(left, right, left_on="asof", right_on="rdate", by="team_id",
                      direction="backward", allow_exact_matches=False)
    out = np.full(len(asof), np.nan)
    out[m["row"].to_numpy()] = m["points"].to_numpy(float)
    return out


def _faceit_block(data_dir: Path, max_cut: pd.Timestamp, x: pd.DataFrame) -> pd.DataFrame:
    tl = faceit_form.Timelines.build(data_dir, max_cut, _hltv_to_faceit(data_dir))
    sides = []
    for side, key_offset in (("a", 0), ("b", 1)):
        s = x[["row", "asof_s", f"lineup_{side}"]].rename(columns={f"lineup_{side}": "lineup"})
        s = s.assign(key=s["row"] * 2 + key_offset)
        sides.append(s)
    lineups = pd.concat(sides).explode("lineup").dropna(subset=["lineup"])
    lineups = lineups.rename(columns={"lineup": "hltv_id"})
    lineups["hltv_id"] = lineups["hltv_id"].astype(np.int64)
    keys = np.arange(2 * len(x))
    tf = faceit_form.team_features(tl, lineups[["key", "hltv_id", "asof_s"]], keys)

    a = tf.loc[keys[0::2]].reset_index(drop=True)
    b = tf.loc[keys[1::2]].reset_index(drop=True)
    n_a = a["n_link"].fillna(0).to_numpy(float)
    n_b = b["n_link"].fillna(0).to_numpy(float)
    out = pd.DataFrame(index=x["row"].to_numpy())
    for m in FACEIT_METRICS:
        out[f"{m}_d"] = a[m].to_numpy() - b[m].to_numpy()
    out[FACEIT_DIFFS] = out[FACEIT_DIFFS].fillna(0.0)
    out["link_d"] = (n_a - n_b) / COVERAGE_PLAYERS
    out["faceit_missing"] = ((n_a == 0) | (n_b == 0)).astype(float)
    out["fallback"] = np.minimum(n_a, n_b) < MIN_LINKED_PER_LINEUP
    out["n_link_a"] = n_a
    out["n_link_b"] = n_b
    return out


def _hltv_frame(data_dir: Path, targets: pd.DataFrame) -> pd.DataFrame:
    max_cut = pd.Timestamp(targets["week_cutoff"].max())
    maps = _maps(data_dir, max_cut)
    rankings = _rankings(data_dir, max_cut)
    target_cuts = {pd.Timestamp(c): g[["match_id", "team_a_id", "team_b_id"]]
                   for c, g in targets.groupby("week_cutoff")}
    series, tgt = elo_path.run(maps, rankings, target_cuts)

    x = pd.concat([series.assign(is_target=False), tgt.assign(is_target=True)], ignore_index=True)
    x["row"] = np.arange(len(x))
    x["asof_s"] = x["match_time"].astype("datetime64[ns]").astype(np.int64) // 1_000_000_000
    x["elo_diff"] = x["elo_a"] - x["elo_b"]
    played = (x["maps_a"] + x["maps_b"]).fillna(0)
    x["train_ok"] = (~x["is_target"]) & (x["bo_type"] == BEST_OF_THREE) & (played >= 2) \
        & (np.maximum(x["maps_a"], x["maps_b"]).fillna(0) >= 2)
    x["y"] = (x["maps_a"] > x["maps_b"]).astype(float).where(~x["is_target"])

    pts_a = _latest_points(rankings, x["match_time"], x["team_a_id"])
    pts_b = _latest_points(rankings, x["match_time"], x["team_b_id"])
    x["rank_diff"] = np.log1p(np.nan_to_num(pts_a)) - np.log1p(np.nan_to_num(pts_b))
    return x


def _fit_predict(x: pd.DataFrame, features: list[str], cut: pd.Timestamp) -> pd.DataFrame:
    window_start = cut - pd.Timedelta(days=TRAIN_WINDOW_DAYS)
    label_end = cut - pd.Timedelta(days=LABEL_LAG_DAYS)
    train = x[x["train_ok"] & (x["match_date"] >= window_start) & (x["match_date"] <= label_end)]
    test = x[x["is_target"] & (x["match_date"] == cut)]
    fitted = model.fit(train[features].to_numpy(float), train["y"].to_numpy(float), RIDGE)
    q = np.clip(fitted.predict(test[features].to_numpy(float)), PROB_CLIP, 1.0 - PROB_CLIP)
    out = pd.DataFrame({"match_id": test["match_id"].astype(np.int64).to_numpy(), "q": q,
                        "fit_cutoff": pd.Timestamp(cut)})
    if "fallback" in test:
        out["fallback"] = test["fallback"].fillna(False).astype(bool).to_numpy()
    return out


def predict(data_dir: Path, targets: pd.DataFrame, faceit: bool) -> pd.DataFrame:
    targets = targets.assign(week_cutoff=_naive(targets["week_cutoff"]))
    max_cut = pd.Timestamp(targets["week_cutoff"].max())
    x = _hltv_frame(data_dir, targets)
    x_f = x.join(_faceit_block(data_dir, max_cut, x), on="row") if faceit else None

    frames = []
    for cut in sorted(targets["week_cutoff"].unique()):
        cut = pd.Timestamp(cut)
        hltv = _fit_predict(x, HLTV_FEATURES, cut)
        if not faceit:
            frames.append(hltv)
            continue
        full = _fit_predict(x_f, HLTV_FEATURES + FACEIT_FEATURES, cut)
        hltv_q = hltv.set_index("match_id")["q"]
        q = full["q"].where(~full["fallback"], full["match_id"].map(hltv_q).to_numpy())
        frames.append(full.assign(q=q).drop(columns="fallback"))
    out = pd.concat(frames, ignore_index=True)
    missing = set(targets["match_id"].astype(np.int64)) - set(out["match_id"])
    if missing:
        raise ValueError(f"no prediction for {len(missing)} targets")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-faceit", action="store_true")
    a = ap.parse_args(argv)
    targets = pd.read_parquet(a.targets)
    out = predict(Path(a.data), targets, faceit=not a.no_faceit)
    out.to_parquet(a.out, index=False)
    print(f"wrote {len(out)} predictions to {a.out} (faceit={not a.no_faceit})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
