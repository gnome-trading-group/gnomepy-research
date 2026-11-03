"""
Candidate feature rows for every bo3 series of the walk-forward weeks, keyed (match_id, week_cutoff).

    python -m gnomepy_research.sessions.cs2_comp.g13.experiments.features --data <bundle> --out <parquet>
        --first 2026-02-23 --last 2026-07-27

EKF skill estimates, map-pool and head-to-head features, short-memory HLTV skill fits (form, as distinct from
e12's year-long class), the brief's thin-series flag, event tier, and whether an announced five lacks the IGL or
AWPer its team last fielded. Join them to the rows of `g13.dev_rows` for `oracle.py` and `layer_walk.py`.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.bundle import load
from gnomepy_research.sessions.cs2_comp.g13.e12.data import Inputs
from gnomepy_research.sessions.cs2_comp.g13.e12.features import feature_rows
from gnomepy_research.sessions.cs2_comp.g13.e12.model import Config, series_price
from gnomepy_research.sessions.cs2_comp.g13.experiments.ekf import ekf_rows
from gnomepy_research.sessions.cs2_comp.g13.experiments.maps import map_rows
from gnomepy_research.sessions.cs2_comp.g13.rows import WEEK, history_frame

SHORT_FITS = {
    "s20": Config(sigma=0.5, half_life_days=20.0, window_days=120, rating_gamma=0.6, rating_sd=0.31),
    "s45": Config(sigma=0.5, half_life_days=45.0, window_days=180, rating_gamma=0.6, rating_sd=0.31),
    "s90": Config(sigma=0.5, half_life_days=90.0, window_days=270, rating_gamma=0.6, rating_sd=0.31),
    "nr300": Config(sigma=0.5, half_life_days=300.0, window_days=365, rating_gamma=0.0),
}


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def context_rows(data_dir, frame: pd.DataFrame, until: pd.Timestamp) -> pd.DataFrame:
    """The brief's thin flag (online, and a team with <= 15 HLTV maps in 90 days or neither team top-100), event tier,
    and roster-role absence: the announced five lacks the IGL / AWPer flagged in the team's last series (A minus B)."""
    h = load(data_dir, "hltv/cs2_match_history", until)
    h = h[h.team_a_id.notna() & h.team_b_id.notna()]
    h = h.assign(match_date=pd.to_datetime(h.match_date))
    first = h.drop_duplicates("match_id").set_index("match_id")
    team_days = pd.concat([h[["match_date", "team_a_id"]].set_axis(["d", "t"], axis=1),
                           h[["match_date", "team_b_id"]].set_axis(["d", "t"], axis=1)])
    by_team = {k: np.sort(g.d.to_numpy()) for k, g in team_days.groupby("t")}
    lu = load(data_dir, "hltv/cs2_match_lineups", until, ["match_id", "match_date", "team_id", "player_id", "is_igl", "is_awp"])
    lu = lu.assign(match_date=pd.to_datetime(lu.match_date)).sort_values("match_date")
    roles: dict[int, list] = {}
    for (team, _), g in lu.groupby(["team_id", "match_id"], sort=False):
        roles.setdefault(int(team), []).append((g.match_date.iloc[0], set(g.player_id[g.is_igl == 1]), set(g.player_id[g.is_awp == 1])))

    def maps90(team, day) -> int:
        a = by_team.get(team, np.array([], "datetime64[ns]"))
        return int(np.searchsorted(a, np.datetime64(day)) - np.searchsorted(a, np.datetime64(day - pd.Timedelta(days=90))))

    def lacks(team, five, cutoff, k) -> float:
        past = [r for r in roles.get(int(team), []) if r[0] < cutoff]
        if not past or five is None or len(five) == 0:
            return 0.0
        usual = max(past, key=lambda r: r[0])[k]
        return float(bool(usual) and not (usual & set(int(p) for p in five)))

    out = []
    for r in frame.itertuples(index=False):
        f = first.loc[r.match_id]
        day, c = pd.Timestamp(r.time).normalize(), pd.Timestamp(r.week_cutoff)
        ranks = [x for x in (f.team_a_rank, f.team_b_rank) if pd.notna(x)]
        thin = float(f.is_lan == 0 and (min(maps90(r.team_a, day), maps90(r.team_b, day)) <= 15 or (min(ranks) if ranks else 999) > 100))
        out.append({"thin_brief": thin, "tier": float(f.event_tier) if pd.notna(f.event_tier) else np.nan,
                    "igl_missing": lacks(r.team_a, r.lineup_a, c, 1) - lacks(r.team_b, r.lineup_b, c, 1),
                    "awp_missing": lacks(r.team_a, r.lineup_a, c, 2) - lacks(r.team_b, r.lineup_b, c, 2)})
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--first", required=True)
    ap.add_argument("--last", required=True)
    a = ap.parse_args()
    cutoffs = list(pd.date_range(a.first, a.last, freq=WEEK))
    until = cutoffs[-1] + WEEK
    inputs = Inputs(a.data, until, faceit=False)
    frame = history_frame(inputs, cutoffs)
    out = frame[["match_id", "week_cutoff"]].copy()
    t0 = time.time()
    out = out.join(ekf_rows(a.data, frame, until).drop(columns="match_id"))
    out = out.join(map_rows(a.data, frame, until).drop(columns="match_id"))
    out = out.join(context_rows(a.data, frame, until))
    print(f"ekf + maps + context {time.time() - t0:.0f}s", flush=True)
    for name, cfg in SHORT_FITS.items():
        t0 = time.time()
        h = feature_rows(inputs, frame, cfg, False)
        out["z_" + name] = _logit(series_price(h.gap.to_numpy()))
        print(f"{name} {time.time() - t0:.0f}s", flush=True)
    out.to_parquet(a.out, index=False)
    print(f"wrote {len(out)} rows to {a.out}")


if __name__ == "__main__":
    main()
