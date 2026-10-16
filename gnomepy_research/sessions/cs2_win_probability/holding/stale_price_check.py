"""
Is the edge an artifact of stale Polymarket prices?

prices-history gives one value per minute. In a quiet market that value can sit
unchanged for hours, and trading against a price nobody would still honour is a
fake edge. Split every comparison by how long the price had gone unchanged and
see whether the P&L survives in markets whose price had just moved.

Scores the leak-free predictions from leak_checks.py: map-3-symmetric at
kickoff, strict (no veto / lineup features) at 12h and 6h.
"""
import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_win_probability.holding._fees import TakerFees
from gnomepy_research.sessions.cs2_win_probability.holding._paths import DATA_DIR

FRESH_MIN = 30

Q = pd.read_parquet(DATA_DIR / "leak_checks_preds.parquet")
matched = pd.read_parquet(DATA_DIR / "pm_all_matched.parquet")
kick = dict(zip(matched.match_id, pd.to_datetime(matched.kickoff)))
series = matched[matched.market == "series"]
tokens = dict(zip(series.match_id, zip(series.token_a, series.token_b)))
paths = pd.read_parquet(DATA_DIR / "pm_all_paths.parquet")
paths = paths[paths.market == "series"].sort_values(["match_id", "t"])
by_mid = {k: (g.t.to_numpy(), g.p_a.to_numpy()) for k, g in paths.groupby("match_id")}


def price_and_age(mid, ts):
    if mid not in by_mid:
        return np.nan, np.nan
    t, p = by_mid[mid]
    i = np.searchsorted(t, ts, side="right") - 1
    if i < 0:
        return np.nan, np.nan
    j = i
    while j > 0 and p[j - 1] == p[i]:
        j -= 1
    changed_at = t[j] if j > 0 else t[0]
    return float(p[i]), (ts - changed_at) / 60.0


def simulate(p, q, y, fee_a, fee_b, hs=0.01, th=0.05):
    ea, eb = q - (p + hs) - fee_a, (1 - q) - ((1 - p) + hs) - fee_b
    ba = ea > th
    bb = (eb > th) & ~ba
    pnl = np.where(ba, y - (p + hs) - fee_a, 0.0) + np.where(bb, (1 - y) - ((1 - p) + hs) - fee_b, 0.0)
    cost = np.where(ba, p + hs, 0.0) + np.where(bb, 1 - p + hs, 0.0)
    return pnl[ba | bb], cost[ba | bb]


def boot(x, n=1000):
    if len(x) < 10:
        return np.nan, np.nan
    rng = np.random.default_rng(0)
    return np.percentile([rng.choice(x, len(x)).sum() for _ in range(n)], [2.5, 97.5])


fees = TakerFees(t for mid in set(Q.match_id) & tokens.keys() for t in tokens[mid])
print(f"{'horizon':>9} {'prediction':>16} {'price age':>12} {'n':>5} {'trades':>7} {'P&L':>8} {'95% CI':>17} {'per $':>7}")
for h, qcol in ((12, "q_strict"), (6, "q_strict"), (0, "q_map3_sym")):
    rows = []
    for r in Q.itertuples():
        if r.match_id not in kick:
            continue
        ts = kick[r.match_id].timestamp() - h * 3600
        p, age = price_and_age(r.match_id, ts)
        if p == p:
            token_a, token_b = tokens[r.match_id]
            rows.append({"p": p, "t_p": ts, "token_a": token_a, "token_b": token_b, "age": age,
                         "q": getattr(r, qcol), "y": r.y})
    d = pd.DataFrame(rows)
    d["fee_a"], d["fee_b"] = fees.for_column(d, "p")
    for label, m in ((f"<= {FRESH_MIN} min", d.age <= FRESH_MIN), (f"> {FRESH_MIN} min", d.age > FRESH_MIN), ("all", d.age >= 0)):
        g = d[m]
        pnl, cost = simulate(g.p.to_numpy(), g.q.to_numpy(), g.y.to_numpy(), g.fee_a.to_numpy(), g.fee_b.to_numpy())
        lo, hi = boot(pnl)
        print(f"{('T-'+str(h)+'h') if h else 'kickoff':>9} {qcol:>16} {label:>12} {len(g):5d} {len(pnl):7d} "
              f"{pnl.sum():+8.2f} [{lo:+6.1f},{hi:+6.1f}] {pnl.sum()/max(cost.sum(),1e-9):+7.1%}")
    print(f"{'':>9} median price age: {d.age.median():.0f} min")
