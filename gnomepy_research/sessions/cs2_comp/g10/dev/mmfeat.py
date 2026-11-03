"""Matchmaking form for every HLTV bo3 series from Jan 2025 and every target, at each series' Monday cutoff.
python -m ...g10.dev.mmfeat --data D"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.hltv.config import showmatch_mask
from gnomepy_research.sessions.cs2_comp.g10.mm import MMConfig, five_mean, mm_rows, player_form
from gnomepy_research.sessions.cs2_comp.g10.rows import monday

CONFIGS = {"mm": MMConfig(), "mm60": MMConfig(half_life_days=60.0, window_days=180)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    a = ap.parse_args()
    t0 = time.time()
    h = pd.read_parquet(f"{a.data}/hltv/cs2_match_history.parquet")
    h = h[~showmatch_mask(h) & (h.bo_type == 3) & (h.match_date >= "2025-01-06")]
    s = h.groupby("match_id").agg(time=("match_time", "first"), date=("match_date", "first"),
                                  lineup_a=("team_a_lineup_ids", "first"), lineup_b=("team_b_lineup_ids", "first")).reset_index()
    s["time"] = pd.to_datetime(s.time.fillna(s.date))
    s["time"] = s.time.dt.tz_convert(None) if s.time.dt.tz is not None else s.time
    t = pd.read_parquet(f"{a.data}/targets_tune.parquet")
    tg = pd.DataFrame({"match_id": t.match_id, "lineup_a": t.team_a_lineup_ids, "lineup_b": t.team_b_lineup_ids,
                       "week_cutoff": pd.to_datetime(t.week_cutoff)})
    s = s[~s.match_id.isin(tg.match_id)].assign(week_cutoff=lambda x: monday(x.time))
    allr = pd.concat([s[["match_id", "lineup_a", "lineup_b", "week_cutoff"]], tg], ignore_index=True)
    rows = mm_rows(a.data, allr.week_cutoff.max())
    print(f"mm rows {len(rows)} players {rows.hltv_id.nunique()} {time.time() - t0:.0f}s", flush=True)
    out = []
    for c in sorted(allr.week_cutoff.unique()):
        c = pd.Timestamp(c)
        wk = allr[allr.week_cutoff == c]
        rec = {"match_id": wk.match_id.to_numpy()}
        for name, cfg in CONFIGS.items():
            form = player_form(rows, c, cfg)
            for col, short in (("form", name), ("activity", f"{name}act"), ("round_form", f"{name}r")):
                rec[f"{short}_a"] = [five_mean(form, v, col) for v in wk.lineup_a]
                rec[f"{short}_b"] = [five_mean(form, v, col) for v in wk.lineup_b]
        out.append(pd.DataFrame(rec))
    res = pd.concat(out, ignore_index=True)
    print(res.describe().T.to_string(), flush=True)
    res.to_parquet(Path(os.environ.get("OUTPUT_DIR", ".")) / "mm.parquet", index=False)


if __name__ == "__main__":
    main()
