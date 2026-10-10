"""
Strategy design backtests on the walk-forward predictions.

1. Blended fair value (model + market, weights fit on earlier months) versus
   the raw model as the trading signal.
2. Fractional-Kelly sizing with per-match and per-order caps, run as a
   bankroll path to see drawdowns, not just averages.
3. Closing line value by entry time: does the market move toward us before
   kickoff, and how much of the edge is captured by when we enter?
4. P&L by segment, including edge size and favourite/underdog side.

Everything is walk-forward: blend weights for a month are fit only on earlier
months, so March (no earlier month) is excluded from every comparison.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from gnomepy_research.sessions.cs2_win_probability.holding._paths import DATA_DIR

TAKER = 0.07
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
sig = lambda z: 1 / (1 + np.exp(-z))

E = pd.read_parquet(DATA_DIR / "backtest_6mo_eval.parquet")
kick = pd.read_parquet(DATA_DIR / "pm_all_matched.parquet").drop_duplicates("match_id").set_index("match_id").kickoff
E["kickoff"] = pd.to_datetime(E.match_id.map(kick))
priors = pd.read_parquet(DATA_DIR / "priors_full.parquet")
ctx = priors[priors.map_position_in_series == 1].drop_duplicates("match_id").set_index("match_id")[
    ["rank_known_both", "event_tier", "is_lan", "elo_games_min"]]
E = E.join(ctx, on="match_id")

CASES = {  # name: (market, price column, half-spread)
    "series @ 6h": ("series", "k6", 0.010),
    "series @ kickoff": ("series", "k0", 0.0125),
    "map 1 @ kickoff": ("game1", "k0", 0.0125),
}


def fit_weights(m, q, y):
    X = np.column_stack([lg(m), lg(q)])
    nll = lambda w: -np.mean(y * (X @ w) - np.log1p(np.exp(X @ w)))
    return minimize(nll, [0.5, 0.5], method="BFGS").x


def with_fair(market, col):
    d = E[(E.market == market) & E[col].notna()].copy()
    out = []
    for mo in sorted(d.month.unique())[1:]:
        prior, cur = d[d.month < mo], d[d.month == mo].copy()
        w = fit_weights(prior[col].to_numpy(), prior.q.to_numpy(), prior.y.to_numpy())
        cur["fair_blend"] = sig(w[0] * lg(cur[col].to_numpy()) + w[1] * lg(cur.q.to_numpy()))
        cur["w_market"], cur["w_model"] = w
        out.append(cur)
    return pd.concat(out)


def select(p, fair, hs, th):
    """Side and per-share cost/edge for the better side, if it clears the threshold."""
    cost_a, cost_b = p + hs, (1 - p) + hs
    fee_a, fee_b = TAKER * p * (1 - p), TAKER * p * (1 - p)
    edge_a, edge_b = fair - cost_a - fee_a, (1 - fair) - cost_b - fee_b
    side = np.where(edge_a >= edge_b, 1, -1)
    edge = np.maximum(edge_a, edge_b)
    cost = np.where(side == 1, cost_a + fee_a, cost_b + fee_b)
    prob = np.where(side == 1, fair, 1 - fair)
    return side, edge, cost, prob, edge >= th


def realized(side, y, cost):
    win = np.where(side == 1, y == 1, y == 0)
    return win.astype(float) - cost


def boot_sum(x, n=1000):
    if len(x) < 10:
        return np.nan, np.nan
    rng = np.random.default_rng(0)
    return np.percentile([rng.choice(x, len(x)).sum() for _ in range(n)], [2.5, 97.5])


# ===================================================================================
print("=" * 100, "\n1. SIGNAL: blended fair value vs raw model   (Apr-Sep, 1 share per trade)\n", "=" * 100)
frames = {name: with_fair(mk, col) for name, (mk, col, _) in CASES.items()}
for name, d in frames.items():
    print(f"\n{name}: n={len(d)}  fitted weights (market, model) mean {d.w_market.mean():.2f}, {d.w_model.mean():.2f}")
    print(f"  {'signal':>8} {'thresh':>6} {'trades':>6} {'P&L':>8} {'95% CI':>17} {'per $':>7} {'months +':>8} {'avg edge':>8}")
    p, y, hs = d[CASES[name][1]].to_numpy(), d.y.to_numpy(), CASES[name][2]
    for sig_name, fair in (("raw", d.q.to_numpy()), ("blend", d.fair_blend.to_numpy())):
        for th in (0.02, 0.03, 0.05):
            side, edge, cost, _, take = select(p, fair, hs, th)
            pnl = realized(side, y, cost)[take]
            lo, hi = boot_sum(pnl)
            months = pd.Series(pnl, index=d.month.to_numpy()[take]).groupby(level=0).sum()
            print(f"  {sig_name:>8} {th:6.2f} {take.sum():6d} {pnl.sum():+8.2f} [{lo:+6.1f},{hi:+6.1f}] "
                  f"{pnl.sum() / cost[take].sum():+7.1%} {int((months > 0).sum())}/{len(months):<6} {edge[take].mean():8.3f}")

# ===================================================================================
print("\n" + "=" * 100, "\n2. SIZING: fractional Kelly on blended fair, series @ 6h + map 1 @ kickoff\n", "=" * 100)
s = frames["series @ 6h"].assign(leg="series", entry=lambda d: d.kickoff - pd.Timedelta(hours=6), price=lambda d: d.k6)
m1 = frames["map 1 @ kickoff"].assign(leg="map1", entry=lambda d: d.kickoff, price=lambda d: d.k0)
legs = pd.concat([s, m1]).sort_values("entry").reset_index(drop=True)
hs_of = {"series": 0.010, "map1": 0.0125}
side, edge, cost, prob, take = select(legs.price.to_numpy(), legs.fair_blend.to_numpy(),
                                      legs.leg.map(hs_of).to_numpy(), 0.03)
legs = legs.assign(side=side, edge=edge, cost=cost, prob=prob, go=take)
legs = legs[legs.go].reset_index(drop=True)


def run_bankroll(kelly_frac, start=10_000.0, match_cap=0.02, order_cap=300.0, flat=None):
    bank, peak, max_dd = start, start, 0.0
    used = {}
    path = []
    for r in legs.itertuples():
        f_star = (r.prob - r.cost) / (1 - r.cost)
        stake = flat if flat is not None else kelly_frac * max(f_star, 0) * bank
        room = match_cap * bank - used.get(r.match_id, 0.0)
        stake = max(0.0, min(stake, order_cap, room))
        if stake <= 0:
            continue
        used[r.match_id] = used.get(r.match_id, 0.0) + stake
        shares = stake / r.cost
        win = (r.y == 1) if r.side == 1 else (r.y == 0)
        bank += shares * (1.0 if win else 0.0) - stake
        peak = max(peak, bank)
        max_dd = max(max_dd, (peak - bank) / peak)
        path.append((r.month, bank, stake))
    p = pd.DataFrame(path, columns=["month", "bank", "stake"])
    monthly = p.groupby("month").bank.last()
    rets = monthly.pct_change().fillna(monthly.iloc[0] / start - 1)
    return bank, max_dd, len(p), p.stake.mean(), rets


print(f"  bets placed on: {len(legs)} legs ({legs.leg.value_counts().to_dict()}), start $10,000, "
      f"caps: 2% of bankroll per match (both legs combined), $300 per order")
print(f"  {'sizing':>16} {'final $':>10} {'total':>8} {'max drawdown':>12} {'bets':>5} {'avg stake':>9}   monthly returns")
for label, kw in (("flat $50", {"kelly_frac": 0, "flat": 50.0}), ("1/4 Kelly", {"kelly_frac": 0.25}),
                  ("1/2 Kelly", {"kelly_frac": 0.5}), ("full Kelly", {"kelly_frac": 1.0})):
    bank, dd, n, avg, rets = run_bankroll(**kw)
    print(f"  {label:>16} {bank:10,.0f} {bank / 10_000 - 1:+8.1%} {dd:12.1%} {n:5d} {avg:9.0f}   "
          + " ".join(f"{r:+.0%}" for r in rets))

# ===================================================================================
print("\n" + "=" * 100, "\n3. CLOSING LINE VALUE: entry price vs the kickoff price, for trades the raw rule takes\n", "=" * 100)
print("  CLV > 0 means the market moved toward our side between entry and kickoff.")
print(f"  {'market':>8} {'entry':>6} {'trades':>6} {'mean CLV':>9} {'95% CI':>17} {'beat close':>10} {'P&L/share':>9} {'P&L if bought at close':>22}")
for market, entries in (("series", ("k12", "k6", "k1")), ("game1", ("k6", "k1"))):
    for col in entries:
        d = E[(E.market == market) & E[col].notna() & E.k0.notna()]
        hs = 0.010 if market == "series" else 0.0125
        side, edge, cost, _, take = select(d[col].to_numpy(), d.q.to_numpy(), hs, 0.05)
        p_in, p_close = d[col].to_numpy()[take], d.k0.to_numpy()[take]
        clv = np.where(side[take] == 1, p_close - p_in, p_in - p_close)
        rng = np.random.default_rng(0)
        bs = [rng.choice(clv, len(clv)).mean() for _ in range(1000)]
        pnl = realized(side, d.y.to_numpy(), cost)[take]
        close_cost = np.where(side[take] == 1, p_close, 1 - p_close) + hs + TAKER * p_close * (1 - p_close)
        pnl_close = np.where(side[take] == 1, d.y.to_numpy()[take] == 1, d.y.to_numpy()[take] == 0) - close_cost
        print(f"  {('map 1' if market == 'game1' else market):>8} {col:>6} {take.sum():6d} {100 * clv.mean():+8.2f}c "
              f"[{100 * np.percentile(bs, 2.5):+5.2f},{100 * np.percentile(bs, 97.5):+5.2f}]c {np.mean(clv > 0):10.1%} "
              f"{pnl.mean():+9.3f} {pnl_close.mean():+22.3f}")

# ===================================================================================
print("\n" + "=" * 100, "\n4. SEGMENTS: where does the edge live?  (raw rule, 5c threshold)\n", "=" * 100)
for name in ("series @ 6h", "map 1 @ kickoff"):
    market, col, hs = CASES[name]
    d = E[(E.market == market) & E[col].notna()].copy()
    side, edge, cost, _, take = select(d[col].to_numpy(), d.q.to_numpy(), hs, 0.05)
    d = d.assign(side=side, edge=edge, cost=cost, pnl=realized(side, d.y.to_numpy(), cost))[take]
    p = d[col]
    d["bought"] = np.where((d.side == 1) == (p > 0.5), "favourite", "underdog")
    d["edge_bucket"] = pd.cut(d.edge, [0.05, 0.10, 0.15, 0.25, 1], labels=["5-10c", "10-15c", "15-25c", "25c+"])
    d["ranked"] = np.where(d.rank_known_both == 1, "both ranked", "a team unranked")
    d["tier"] = np.where(d.event_tier > 0, "rated event", "unrated event")
    d["lan"] = np.where(d.is_lan == 1, "LAN", "online")
    d["history"] = np.where(d.elo_games_min >= 20, "both 20+ games", "thin history")
    d["volume"] = pd.qcut(d.volume, 3, labels=["low vol", "mid vol", "high vol"])
    print(f"\n{name}: {len(d)} trades, P&L/share {d.pnl.mean():+.3f}")
    print(f"  {'segment':>16} {'value':>16} {'trades':>6} {'P&L/share':>9} {'95% CI (total)':>18} {'per $':>7}")
    for seg in ("bought", "edge_bucket", "ranked", "tier", "lan", "history", "volume"):
        for val, g in d.groupby(seg, observed=True):
            lo, hi = boot_sum(g.pnl.to_numpy())
            print(f"  {seg:>16} {str(val):>16} {len(g):6d} {g.pnl.mean():+9.3f} [{lo:+6.1f},{hi:+6.1f}] "
                  f"{g.pnl.sum() / g.cost.sum():+7.1%}")
