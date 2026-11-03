"""Where the model gains on the market: pool coefficients and log-loss gains by segment of the dev targets."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.g12.data import load_hltv
from gnomepy_research.sessions.cs2_comp.g12.dev_eval import ll, pool
from gnomepy_research.sessions.cs2_comp.g12.stack import fit_logistic, logit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/work/in/comp")
    ap.add_argument("--panel", default="/work/in/eval/panel.parquet")
    ap.add_argument("--res", default="/work/in/eval/res_gauss.parquet")
    a = ap.parse_args()
    p = pd.read_parquet(a.panel)
    ev = p[p.target].merge(pd.read_parquet(a.res), on=["match_id", "week_cutoff"])
    full = load_hltv(Path(a.data), pd.Timestamp("2026-08-01"))[1]
    ev["y"] = ev.match_id.map(full.set_index("match_id").y)
    ev = ev[ev.y.notna() & ev.m0.notna()].reset_index(drop=True)
    y = ev.y.to_numpy(float)
    months = pd.to_datetime(ev.time).dt.to_period("M").astype(str).to_numpy()
    ev["gain"] = ll(pool(ev.m0, ev.q, y, months), y) - ll(ev.m0, y)
    unk = ev.player_unknown_a + ev.player_unknown_b
    segs = {
        "unknown=0": unk == 0, "unknown=1-2": unk.between(1, 2), "unknown>=3": unk >= 3,
        "lan": ev.is_lan == 1, "online": ev.is_lan == 0,
        "ranked both": (ev.rank_a <= 100) & (ev.rank_b <= 100), "unranked side": (ev.rank_a > 100) | (ev.rank_b > 100),
        "young book": ev.age_h < 12, "still book": ev.changes < 30, "active book": ev.changes >= 100,
        "new>=2": (ev.new_players_a + ev.new_players_b) >= 2, "sub": (ev.sub_a + ev.sub_b) > 0,
        "var high": (ev.gauss_var_a + ev.gauss_var_b) > (ev.gauss_var_a + ev.gauss_var_b).median(),
        "var low": (ev.gauss_var_a + ev.gauss_var_b) <= (ev.gauss_var_a + ev.gauss_var_b).median(),
    }
    for name, m in segs.items():
        m = m.to_numpy()
        X = np.column_stack([logit(ev.m0[m]), logit(ev.q[m]), np.ones(m.sum())])
        b = fit_logistic(X, y[m], 1e-6)
        print(f"{name:16s} n={m.sum():5d} gain {ev.gain[m].mean():+.4f}  in-seg pool weights m {b[0]:+.2f} q {b[1]:+.2f} c {b[2]:+.2f}")


if __name__ == "__main__":
    main()
