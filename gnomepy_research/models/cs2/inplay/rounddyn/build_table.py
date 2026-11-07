"""Research cache: every scored-type state of 2026 (legal maps) with the benchmark's round features, the timeline
dynamics, round and map labels, and the benchmark map model's continuations (walk-forward by month, Apr-Aug), so round
model variants can be compared on the map log loss without rerunning the harness."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.inplay import baseline
from gnomepy_research.models.cs2.inplay.data import Bundle
from gnomepy_research.models.cs2.inplay.features import legal_maps, map_states
from gnomepy_research.models.cs2.inplay.rounddyn.feats import dyn_features

DATA = Path.home() / "gnome-research-work/cs2/comp3/data"
OUT = Path(__file__).resolve().parents[5] / ".cache"
MONTHS = ["2026-04", "2026-05", "2026-06", "2026-07", "2026-08"]


def main():
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    b = Bundle.load(DATA)
    g = b.games
    ids = set(g.game_id[g.begin >= pd.Timestamp("2026-01-01", tz="UTC")]) & legal_maps(b)
    st = map_states(b, ids)
    print("states", len(st), time.time() - t0, file=sys.stderr, flush=True)
    x, rt = baseline.round_rows(b, st)
    print("base", time.time() - t0, file=sys.stderr, flush=True)
    d = dyn_features(b, st)
    print("dyn", time.time() - t0, file=sys.stderr, flush=True)
    x = x.join(d.drop(columns=[c for c in d.columns if c in x.columns]), rsuffix="_d")
    x = x.join(rt[["a_wins", "map_a_wins", "prior_logit", "has_prior", "a_money", "b_money", "a_money_pre", "b_money_pre",
                   "score_a", "score_b", "a_loss_ctr", "b_loss_ctr", "ot", "round_in_half", "pistol"]], on=["game_id", "round"])
    x = x.merge(g[["game_id", "begin", "end", "match_begin", "tier", "month"]], on="game_id", how="left")
    x["y_ct"] = np.where(x.side_a == "CT", x.a_wins, 1 - x.a_wins)
    x["m1"] = np.nan; x["m0"] = np.nan
    for m in MONTHS:
        start = pd.Timestamp(m + "-01", tz="UTC")
        train = b.ended_before(start)
        legal = legal_maps(train)
        r = baseline.with_prior(baseline.round_table(train.subset(legal)))
        r = r[r.map_a_wins.notna()]
        mm = baseline.trees(0).fit(r[baseline.MAPCOLS], r.map_a_wins)
        sel = (x.match_begin >= start) & (x.match_begin < start + pd.offsets.MonthBegin(1))
        keys = x.loc[sel, ["game_id", "round"]].drop_duplicates()
        rr = rt.loc[pd.MultiIndex.from_frame(keys)].reset_index()
        for a_won, col in ((1, "m1"), (0, "m0")):
            X, done = baseline.next_round(rr, a_won)
            p = mm.predict_proba(X)[:, 1]
            s = pd.Series(np.where(np.isnan(done), p, done), index=pd.MultiIndex.from_frame(rr[["game_id", "round"]]))
            idx = pd.MultiIndex.from_frame(x.loc[sel, ["game_id", "round"]])
            x.loc[sel, col] = s.reindex(idx).to_numpy()
        print(m, time.time() - t0, file=sys.stderr, flush=True)
    x.to_parquet(OUT / "states.parquet")
    print("done", time.time() - t0, file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
