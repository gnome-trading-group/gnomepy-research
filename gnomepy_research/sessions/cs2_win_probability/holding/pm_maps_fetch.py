"""
Fetch Polymarket per-map winner markets (-game1/2/3) for the L1 test split.

Stores raw minute-level price paths in long form so map boundaries and
pricing moments can be re-derived later without refetching.
"""
import datetime as dt
import re
import time

import numpy as np
import pandas as pd
import requests

from gnomepy_research.sessions.cs2_win_probability.splits import temporal_split_by_series
from gnomepy_research.sessions.cs2_win_probability.holding._paths import DATA_DIR

D = str(DATA_DIR)
GAME = re.compile(r"-(\d{4}-\d{2}-\d{2})-game(\d)$")

history = pd.read_parquet(f"{D}/cs2_match_history.parquet")
history["match_date"] = pd.to_datetime(history["match_date"])
priors = pd.read_parquet(f"{D}/priors_full.parquet")
cols = ["match_id", "map_name", "team_a_won", "team_a_score", "team_b_score"]
df = priors.merge(history[cols], on=["match_id", "map_name"], how="inner")
df = df[df.team_a_score != df.team_b_score].sort_values(["match_date", "match_id"]).reset_index(drop=True)
_, _, te = temporal_split_by_series(df)
test = df.iloc[te]
test = test[test.bo_type == 3]
test_ids = set(test.match_id)
print(f"L1 test split: {len(df.iloc[te])} maps; BO3: {len(test)} maps in {len(test_ids)} series "
      f"({test.match_date.min().date()} -> {test.match_date.max().date()})", flush=True)


def norm(n):
    n = (n or "").lower().strip()
    n = re.sub(r"\b(esports|esport|team|gaming|club|academy|nxt|fe)\b", "", n)
    return re.sub(r"[^a-z0-9]", "", n)


pm = pd.read_parquet(f"{D}/pm_cs2_h2h.parquet").copy()
ext = pm.market_slug.str.extract(GAME)
pm["slug_date"], pm["game"] = ext[0], ext[1]
pm = pm[pm.slug_date.notna()].copy()
pm["slug_date"] = pd.to_datetime(pm.slug_date).dt.date
pm["game"] = pm.game.astype(int)
pm["o0"] = [norm(list(o)[0]) for o in pm.outcomes]
pm["o1"] = [norm(list(o)[1]) for o in pm.outcomes]
pm["pair"] = [tuple(sorted(p)) for p in zip(pm.o0, pm.o1)]
idx = {}
for r in pm.itertuples():
    idx.setdefault((r.slug_date, r.pair, r.game), []).append(r)
print(f"map markets available: {len(pm)}", flush=True)

kickoff = history.groupby("match_id").match_time.min()
maps = history[history.match_id.isin(test_ids)][
    ["match_id", "match_date", "map_position_in_series", "map_name", "team_a_name", "team_b_name", "team_a_won"]]
matched, dropped = [], {"none": 0, "ambiguous": 0, "settle_mismatch": 0, "unsettled": 0}
for r in maps.itertuples():
    a, b = norm(r.team_a_name), norm(r.team_b_name)
    pair = tuple(sorted([a, b]))
    key_date = r.match_date.date()
    cands = idx.get((key_date, pair, r.map_position_in_series), [])
    if not cands:
        for off in (-1, 1):
            cands = idx.get((key_date + dt.timedelta(days=off), pair, r.map_position_in_series), [])
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
    if len(fin) != 2 or max(fin) < 0.9:
        dropped["unsettled"] += 1
        continue
    if int((fin[0] > fin[1]) == a_is_o0) != int(r.team_a_won):
        dropped["settle_mismatch"] += 1
        continue
    matched.append({"match_id": r.match_id, "map_pos": r.map_position_in_series, "map_name": r.map_name,
                    "token_a": list(mk.tokens)[0 if a_is_o0 else 1], "volume": float(mk.volume),
                    "kickoff": kickoff.get(r.match_id)})
m = pd.DataFrame(matched).dropna(subset=["kickoff"])
print(f"maps matched: {len(m)}  by position: {m.map_pos.value_counts().sort_index().to_dict()}  "
      f"dropped: {dropped}", flush=True)
m.drop(columns=["token_a"]).to_parquet(f"{D}/pm_maps_matched.parquet", index=False)

s = requests.Session()
s.headers["User-Agent"] = "research/1.0"
paths = []
for i, r in enumerate(m.itertuples()):
    k = pd.Timestamp(r.kickoff)
    params = {"market": r.token_a, "startTs": int((k - pd.Timedelta(days=2)).timestamp()),
              "endTs": int((k + pd.Timedelta(hours=8)).timestamp()), "fidelity": 1}
    try:
        resp = s.get("https://clob.polymarket.com/prices-history", params=params, timeout=30)
        hist = resp.json().get("history", []) if resp.status_code == 200 else []
    except Exception:
        hist = []
    for x in hist:
        paths.append((r.match_id, r.map_pos, int(x["t"]), float(x["p"])))
    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{len(m)} fetched", flush=True)
    time.sleep(0.12)

out = pd.DataFrame(paths, columns=["match_id", "map_pos", "t", "p_a"])
out.to_parquet(f"{D}/pm_maps_paths.parquet", index=False)
print(f"\nprice points: {len(out)} across {out.groupby(['match_id','map_pos']).ngroups} map markets")
