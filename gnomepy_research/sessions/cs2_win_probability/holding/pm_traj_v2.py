"""
Polymarket price trajectories for the full-corpus test split, anchored on real kickoff.

v1 anchored horizons on the market's `end`, which is kickoff + 6.0h, so its
"T-12h" was really 6h before kickoff and its "T-6h" was the opening whistle.
Here every horizon is measured back from HLTV's match_time.
"""
import datetime as dt
import re
import time

import numpy as np
import pandas as pd
import requests

from gnomepy_research.sessions.cs2_win_probability.series_model import load_series_frame
from gnomepy_research.sessions.cs2_win_probability.splits import temporal_split_by_series
from gnomepy_research.sessions.cs2_win_probability.holding._paths import DATA_DIR

D = str(DATA_DIR)
HORIZONS_H = [24, 12, 6, 3, 1, 0]

history = pd.read_parquet(f"{D}/cs2_match_history.parquet")
history["match_date"] = pd.to_datetime(history["match_date"])
priors = pd.read_parquet(f"{D}/priors_full.parquet")
sf = load_series_frame(priors=priors, history=history)
sf = sf[sf.bo_type == 3].reset_index(drop=True)
_, _, te = temporal_split_by_series(sf, group_col="match_id")
test_ids = set(sf.loc[te, "match_id"])
print(f"BO3 test series: {len(test_ids)}  "
      f"({sf.loc[te,'match_date'].min().date()} -> {sf.loc[te,'match_date'].max().date()})", flush=True)


def norm(n):
    n = (n or "").lower().strip()
    n = re.sub(r"\b(esports|esport|team|gaming|club|academy|nxt|fe)\b", "", n)
    return re.sub(r"[^a-z0-9]", "", n)


# An h2h "event" holds every market between two teams across several meetings:
# the series moneyline plus per-map winners (-game1/2/3), handicaps and totals.
# v1 took the highest-volume market in a +/-1 day window, which was a map or
# handicap market 16.6% of the time - and sometimes from a different meeting.
# Only a slug ending in the match date is the series moneyline.
MONEYLINE = re.compile(r"-(\d{4}-\d{2}-\d{2})$")

pm = pd.read_parquet(f"{D}/pm_cs2_h2h.parquet").copy()
pm["slug_date"] = pm.market_slug.str.extract(MONEYLINE)[0]
pm = pm[pm.slug_date.notna()].copy()
pm["slug_date"] = pd.to_datetime(pm.slug_date).dt.date
pm["o0"] = [norm(list(o)[0]) for o in pm.outcomes]
pm["o1"] = [norm(list(o)[1]) for o in pm.outcomes]
pm["pair"] = [tuple(sorted(p)) for p in zip(pm.o0, pm.o1)]
idx = {}
for r in pm.itertuples():
    idx.setdefault((r.slug_date, r.pair), []).append(r)
print(f"moneyline markets: {len(pm)}", flush=True)

series = history[history.match_id.isin(test_ids)].drop_duplicates("match_id")
kickoff = history.groupby("match_id").match_time.min()
a_won = history.groupby("match_id").team_a_won.agg(lambda w: int(w.sum() > len(w) - w.sum()))
matched, dropped = [], {"none": 0, "ambiguous": 0, "settle_mismatch": 0}
for r in series.itertuples():
    a, b = norm(r.team_a_name), norm(r.team_b_name)
    pair = tuple(sorted([a, b]))
    cands = idx.get((r.match_date.date(), pair), [])
    if not cands:
        for off in (-1, 1):
            cands = idx.get((r.match_date.date() + dt.timedelta(days=off), pair), [])
            if cands:
                break
    if not cands:
        dropped["none"] += 1
        continue
    if len(cands) > 1:
        dropped["ambiguous"] += 1
        continue
    mk = cands[0]
    a_is_o0 = mk.o0 == a
    fin = [float(x) for x in list(mk.final)] if mk.final is not None else []
    if len(fin) == 2 and max(fin) > 0.9:
        if int((fin[0] > fin[1]) == a_is_o0) != a_won[r.match_id]:
            dropped["settle_mismatch"] += 1
            continue
    matched.append({"match_id": r.match_id, "tokens": mk.tokens, "team_a_is_outcome0": a_is_o0,
                    "volume": float(mk.volume), "kickoff": kickoff.get(r.match_id)})
m = pd.DataFrame(matched).dropna(subset=["kickoff"])
print(f"matched to a moneyline: {len(m)} of {len(test_ids)}  dropped: {dropped}", flush=True)

s = requests.Session()
s.headers["User-Agent"] = "research/1.0"
rows = []
for i, r in enumerate(m.itertuples()):
    tok = list(r.tokens)[0 if r.team_a_is_outcome0 else 1]
    k = pd.Timestamp(r.kickoff)
    params = {"market": tok, "startTs": int((k - pd.Timedelta(days=4)).timestamp()),
              "endTs": int(k.timestamp()) + 60, "fidelity": 1}
    try:
        resp = s.get("https://clob.polymarket.com/prices-history", params=params, timeout=30)
        hist = resp.json().get("history", []) if resp.status_code == 200 else []
    except Exception:
        hist = []
    if not hist:
        continue
    t = np.array([x["t"] for x in hist])
    v = np.array([float(x["p"]) for x in hist])
    rec = {"match_id": r.match_id, "volume": r.volume, "pm_open": float(v[0])}
    for h in HORIZONS_H:
        prior = v[t <= k.timestamp() - h * 3600]
        rec[f"k_{h}h"] = float(prior[-1]) if len(prior) else np.nan
    rows.append(rec)
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(m)} fetched, kept {len(rows)}", flush=True)
    time.sleep(0.12)

out = pd.DataFrame(rows)
out.to_parquet(f"{D}/pm_traj_v2.parquet", index=False)
print(f"\ntrajectories: {len(out)}")
print("coverage:", {c: int(out[c].notna().sum()) for c in out.columns if c.startswith("k_")})
