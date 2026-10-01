"""
Polymarket price paths for every BO3 series from March onward: series moneyline, map 1 and map 2.

Feeds the six-month backtest. Matching follows the rules learned the hard way:
moneyline slugs end in the match date, map winners end -gameN, never pick by
volume, read orientation from `outcomes`, and drop anything whose settlement
disagrees with HLTV. Resumable - already-fetched markets are skipped - because
a few thousand requests is long enough to be interrupted.
"""
import datetime as dt
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests

from gnomepy_research.artifacts import DatasetStore
from gnomepy_research.sessions.cs2_win_probability.holding._paths import DATA_DIR

START = pd.Timestamp("2026-03-01")
WORKERS = 4
SLUG = re.compile(r"-(\d{4}-\d{2}-\d{2})(?:-game(\d))?$")
PATHS = DATA_DIR / "pm_all_paths.parquet"
MATCHED = DATA_DIR / "pm_all_matched.parquet"


def norm(n):
    n = (n or "").lower().strip()
    n = re.sub(r"\b(esports|esport|team|gaming|club|academy|nxt|fe)\b", "", n)
    return re.sub(r"[^a-z0-9]", "", n)


history = DatasetStore().load("cs2_match_history")
history["match_date"] = pd.to_datetime(history["match_date"])
history = history[(history.match_date >= START) & (history.bo_type == 3)]
kickoff = history.groupby("match_id").match_time.min()
series_won = history.groupby("match_id").team_a_won.agg(lambda w: int(w.sum() > len(w) - w.sum()))
print(f"BO3 series since {START.date()}: {history.match_id.nunique()}", flush=True)

pm = pd.read_parquet(DATA_DIR / "pm_cs2_h2h.parquet").copy()
ext = pm.market_slug.str.extract(SLUG)
pm["slug_date"], pm["game"] = ext[0], ext[1].fillna("0")
pm = pm[pm.slug_date.notna()].copy()
pm["slug_date"] = pd.to_datetime(pm.slug_date).dt.date
pm["market"] = np.where(pm.game == "0", "series", "game" + pm.game)
pm = pm[pm.market.isin(["series", "game1", "game2"])]
pm["o0"] = [norm(list(o)[0]) for o in pm.outcomes]
pm["o1"] = [norm(list(o)[1]) for o in pm.outcomes]
pm["pair"] = [tuple(sorted(p)) for p in zip(pm.o0, pm.o1)]
idx = {}
for r in pm.itertuples():
    idx.setdefault((r.slug_date, r.pair, r.market), []).append(r)

maps = history.set_index(["match_id", "map_position_in_series"]).team_a_won
first = history.drop_duplicates("match_id")
matched, dropped = [], {}
for r in first.itertuples():
    a, b = norm(r.team_a_name), norm(r.team_b_name)
    pair = tuple(sorted([a, b]))
    for market in ("series", "game1", "game2"):
        cands = idx.get((r.match_date.date(), pair, market), [])
        if not cands:
            for off in (-1, 1):
                cands = idx.get((r.match_date.date() + dt.timedelta(days=off), pair, market), [])
                if cands:
                    break
        if len(cands) != 1:
            dropped[f"{market}:{'none' if not cands else 'ambiguous'}"] = dropped.get(
                f"{market}:{'none' if not cands else 'ambiguous'}", 0) + 1
            continue
        mk = cands[0]
        a_is_o0 = mk.o0 == a
        truth = series_won.get(r.match_id) if market == "series" else maps.get((r.match_id, int(market[-1])))
        fin = [float(x) for x in list(mk.final)] if mk.final is not None else []
        if truth is None or len(fin) != 2 or max(fin) < 0.9 or int((fin[0] > fin[1]) == a_is_o0) != int(truth):
            dropped[f"{market}:unsettled_or_mismatch"] = dropped.get(f"{market}:unsettled_or_mismatch", 0) + 1
            continue
        matched.append({"match_id": r.match_id, "market": market, "volume": float(mk.volume),
                        "token_a": list(mk.tokens)[0 if a_is_o0 else 1], "kickoff": kickoff.get(r.match_id)})
m = pd.DataFrame(matched).dropna(subset=["kickoff"])
m.drop(columns=["token_a"]).to_parquet(MATCHED, index=False)
print(f"matched: {m.market.value_counts().to_dict()}  dropped: {dropped}", flush=True)

done = set()
existing = pd.DataFrame(columns=["match_id", "market", "t", "p_a"])
if PATHS.exists():
    existing = pd.read_parquet(PATHS)
    done = set(zip(existing.match_id, existing.market))
todo = [r for r in m.itertuples() if (r.match_id, r.market) not in done]
print(f"to fetch: {len(todo)}  (already have {len(done)})", flush=True)


def fetch(r):
    s = requests.Session()
    s.headers["User-Agent"] = "research/1.0"
    k = pd.Timestamp(r.kickoff)
    params = {"market": r.token_a, "startTs": int((k - pd.Timedelta(days=4)).timestamp()),
              "endTs": int((k + pd.Timedelta(hours=8)).timestamp()), "fidelity": 1}
    for attempt in range(4):
        try:
            resp = s.get("https://clob.polymarket.com/prices-history", params=params, timeout=30)
            if resp.status_code == 429:
                time.sleep(2 ** attempt * 2)
                continue
            hist = resp.json().get("history", []) if resp.status_code == 200 else []
            return [(r.match_id, r.market, int(x["t"]), float(x["p"])) for x in hist]
        except Exception:
            time.sleep(1 + attempt)
    return []


buf, n = [], 0
with ThreadPoolExecutor(WORKERS) as pool:
    futures = [pool.submit(fetch, r) for r in todo]
    for f in as_completed(futures):
        buf.extend(f.result())
        n += 1
        if n % 250 == 0 or n == len(todo):
            existing = pd.concat([existing, pd.DataFrame(buf, columns=existing.columns)], ignore_index=True)
            existing.to_parquet(PATHS, index=False)
            buf = []
            print(f"  {n}/{len(todo)} fetched", flush=True)

print(f"\npaths: {existing.groupby(['match_id','market']).ngroups} markets, {len(existing)} points -> {PATHS}")
