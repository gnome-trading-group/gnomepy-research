"""
#1 final: walk-forward series predictions vs the Polymarket moneyline, log-loss and fee-aware P&L.

Each fold's predictions come from an L1 trained only on earlier data, and the
series temperature for fold k is fit only on folds < k - what a monthly
retrain would actually have produced.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from gnomepy_research.sessions.cs2_win_probability.pre_map_model import _calibrated, _fit_temperature
from gnomepy_research.sessions.cs2_win_probability.splits import calibration_slope, clustered_bootstrap_delta
from gnomepy_research.sessions.cs2_win_probability.holding._paths import DATA_DIR

D = str(DATA_DIR)

TAKER = 0.07
oos = pd.read_parquet(f"{D}/series_wf_oos.parquet")
parts = []
for k in sorted(oos.fold.unique()):
    cur = oos[oos.fold == k].copy()
    prior = oos[oos.fold < k]
    cur["q"] = _calibrated(_fit_temperature(prior.raw.to_numpy(), prior.y.to_numpy()), cur.raw.to_numpy()) \
        if len(prior) >= 1500 else np.clip(cur.raw.to_numpy(), 0.01, 0.99)
    parts.append(cur)
oos = pd.concat(parts)

tj = pd.read_parquet(f"{D}/pm_traj_v2.parquet")
j = oos.merge(tj, on="match_id", how="inner")
print(f"series with walk-forward prediction AND a moneyline path: {len(j)}  (folds {sorted(j.fold.unique())})")
y = j.y.to_numpy()
print(f"model: ll {log_loss(y, j.q):.4f}  slope {calibration_slope(y, j.q.to_numpy()):.3f}\n")

H = [("pm_open", "open"), ("k_12h", "12h before"), ("k_6h", "6h before"), ("k_3h", "3h before"),
     ("k_1h", "1h before"), ("k_0h", "kickoff")]
print(f"{'horizon':>11} {'n':>4} {'market':>7} {'model':>7} {'gap [CI]':>28}")
for col, lab in H:
    s = j[col].notna().to_numpy()
    if s.sum() < 60:
        continue
    mk = np.clip(j[col].to_numpy()[s], 1e-4, 1 - 1e-4)
    q = j.q.to_numpy()[s]
    d, lo, hi = clustered_bootstrap_delta(y[s], q, mk, j.match_id.to_numpy()[s], log_loss, n_boot=400)
    print(f"{lab:>11} {int(s.sum()):4d} {log_loss(y[s], mk):7.4f} {log_loss(y[s], q):7.4f} "
          f"{d:+.4f} [{lo:+.4f},{hi:+.4f}] {'SIG' if (lo > 0) == (hi > 0) else 'ns'}")


def simulate(p, q, y, half_spread, threshold):
    """One share per series on whichever side clears fee + spread + threshold."""
    fee = TAKER * p * (1 - p)
    edge_a = q - (p + half_spread) - fee
    edge_b = (1 - q) - ((1 - p) + half_spread) - fee
    buy_a, buy_b = edge_a > threshold, (edge_b > threshold) & ~(edge_a > threshold)
    pnl = np.where(buy_a, y - (p + half_spread) - fee, 0.0) + \
          np.where(buy_b, (1 - y) - ((1 - p) + half_spread) - fee, 0.0)
    cost = np.where(buy_a, p + half_spread, 0.0) + np.where(buy_b, 1 - p + half_spread, 0.0)
    traded = buy_a | buy_b
    return pnl[traded], cost[traded]


def ci(x, n=2000, seed=0):
    if len(x) < 5:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    m = [rng.choice(x, len(x)).sum() for _ in range(n)]
    return np.percentile(m, [2.5, 97.5])


print("\nfee-aware P&L, 1 share per series per horizon (taker fee 0.07*p*(1-p); price = Polymarket history mid)")
print(f"{'horizon':>11} {'spread':>7} {'thresh':>7} {'trades':>7} {'P&L $':>8} {'95% CI':>18} {'per $ risked':>13}")
for col, lab in H:
    s = j[col].notna().to_numpy()
    if s.sum() < 60:
        continue
    p, q, yy = j[col].to_numpy()[s], j.q.to_numpy()[s], y[s]
    for hs in (0.0, 0.01, 0.02):
        for th in (0.02, 0.05):
            pnl, cost = simulate(p, q, yy, hs, th)
            lo, hi = ci(pnl)
            roi = pnl.sum() / cost.sum() if cost.sum() else np.nan
            print(f"{lab:>11} {hs:7.2f} {th:7.2f} {len(pnl):7d} {pnl.sum():8.2f} [{lo:+7.2f},{hi:+7.2f}] {roi:+12.1%}")


print("\n--- independent information? blend of model and market vs market alone ---")
for col, lab in H[1:]:
    s = j[col].notna().to_numpy()
    mk = np.clip(j[col].to_numpy()[s], 1e-4, 1 - 1e-4); q = j.q.to_numpy()[s]; yy = y[s]
    lg = lambda x: np.log(x / (1 - x))
    blend = 1 / (1 + np.exp(-(lg(mk) + lg(q)) / 2))
    d, lo, hi = clustered_bootstrap_delta(yy, blend, mk, j.match_id.to_numpy()[s], log_loss, n_boot=400)
    print(f"{lab:>11}  err corr {np.corrcoef(yy - q, yy - mk)[0,1]:.2f}   blend - market {d:+.4f} [{lo:+.4f},{hi:+.4f}] "
          f"{'SIG' if (lo > 0) == (hi > 0) else 'ns'}")

print("\n--- does profit survive in liquid markets? (6h before, 1c half-spread, 5c threshold) ---")
s = j["k_6h"].notna().to_numpy()
jj = j[s].copy()
jj["vt"] = pd.qcut(jj.volume, 3, labels=["low vol", "mid vol", "high vol"])
for t, g in jj.groupby("vt", observed=True):
    pnl, cost = simulate(g.k_6h.to_numpy(), g.q.to_numpy(), g.y.to_numpy(), 0.01, 0.05)
    lo, hi = ci(pnl)
    print(f"  {t:>8}: volume ${g.volume.min():>9,.0f}-${g.volume.max():>10,.0f}  trades {len(pnl):3d}  "
          f"P&L {pnl.sum():+6.2f} [{lo:+.2f},{hi:+.2f}]  per $ {pnl.sum()/max(cost.sum(),1e-9):+.1%}")
