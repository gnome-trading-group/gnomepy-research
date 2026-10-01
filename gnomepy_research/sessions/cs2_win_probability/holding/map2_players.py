"""
Does per-player map-1 performance improve map-2 prediction beyond who won and by how much?

Features are surprises against each player's pre-series form, so they carry what
map 1 revealed rather than restating that the better team has better ratings.
Pre-series form is an EWMA over strictly earlier dates, so nothing from the
series itself leaks into the baseline it is compared against.
"""
import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import log_loss

from gnomepy_research.sessions.cs2_win_probability.pre_map_features import PRE_MAP_FEATURE_NAMES
from gnomepy_research.sessions.cs2_win_probability.pre_map_model import CS2PreMapModel, _matrix
from gnomepy_research.sessions.cs2_win_probability.splits import clustered_bootstrap_delta, temporal_split_by_series
from gnomepy_research.sessions.cs2_win_probability.holding._paths import DATA_DIR

logging.basicConfig(level=logging.ERROR)
D = str(DATA_DIR)
HALFLIFE = 20

h = pd.read_parquet(f"{D}/cs2_match_history.parquet")
h["match_date"] = pd.to_datetime(h.match_date)
pr = pd.read_parquet(f"{D}/priors_full.parquet")
ps = pd.read_parquet(f"{D}/cs2_player_map_stats.parquet")
ps = ps[ps.side == "all"].copy()
ps["match_date"] = pd.to_datetime(ps.match_date).dt.tz_localize(None)
for c in ("rating", "adr", "eadr", "kast", "ekast", "round_swing_pct"):
    ps[c] = pd.to_numeric(ps[c], errors="coerce")

# Pre-series form: per-player EWMA of rating, read as of the last date strictly before the series.
ps = ps.sort_values(["player_id", "match_date", "match_id"])
ps["form_ewm"] = ps.groupby("player_id").rating.transform(lambda r: r.ewm(halflife=HALFLIFE, ignore_na=True).mean())
daily = ps.groupby(["player_id", "match_date"]).form_ewm.last().reset_index().sort_values("match_date")

m1_name = h[h.map_position_in_series == 1].set_index("match_id").map_name
p1 = ps[ps.match_id.isin(m1_name.index)]
p1 = p1[p1.map_name == p1.match_id.map(m1_name)].copy()
p1 = pd.merge_asof(p1.sort_values("match_date"), daily.rename(columns={"form_ewm": "form"}),
                   on="match_date", by="player_id", allow_exact_matches=False)
p1["surprise"] = p1.rating - p1.form
p1["adr_resid"] = p1.adr - p1.eadr


def team_side(df, is_a):
    g = df[df.is_team_a == is_a].groupby("match_id")
    return pd.DataFrame({"rating": g.rating.mean(), "surprise": g.surprise.mean(),
                         "worst_surprise": g.surprise.min(), "adr_resid": g.adr_resid.mean(),
                         "swing": g.round_swing_pct.mean()})


A, B = team_side(p1, True), team_side(p1, False)
feat = pd.DataFrame({
    "m1_rating_diff": A.rating - B.rating,
    "m1_surprise_diff": A.surprise - B.surprise,
    "m1_worst_surprise_diff": A.worst_surprise - B.worst_surprise,
    "m1_adr_resid_diff": (A.adr_resid - B.adr_resid) / 10.0,
    "m1_swing_diff": A.swing - B.swing,
})
awp_ids = h.drop_duplicates("match_id").set_index("match_id")[["team_a_awp_id", "team_b_awp_id"]]
p1i = p1.set_index(["match_id", "player_id"]).surprise
def awp_surprise(col):
    out = {}
    for mid, pid in awp_ids[col].dropna().items():
        out[mid] = p1i.get((mid, int(pid)), np.nan)
    return pd.Series(out)
feat["m1_awp_surprise_diff"] = awp_surprise("team_a_awp_id") - awp_surprise("team_b_awp_id")

m1 = h[h.map_position_in_series == 1].set_index("match_id")
feat["dom"] = np.sign(m1.team_a_score - m1.team_b_score) * np.minimum((m1.team_a_score - m1.team_b_score).abs(), 13) / 13.0
print("map-1 feature coverage:", feat.notna().mean().round(3).to_dict())

cols = ["match_id", "map_name", "team_a_won", "team_a_score", "team_b_score"]
mf = pr.merge(h[cols], on=["match_id", "map_name"])
mf = mf[mf.team_a_score != mf.team_b_score].sort_values(["match_date", "match_id"]).reset_index(drop=True)
_, val, te = temporal_split_by_series(mf)
model = CS2PreMapModel(f"{D}/cs2_l1_harvested.xgb")


def block(idx):
    b = mf.iloc[idx]
    b = b[(b.bo_type == 3) & (b.map_position_in_series == 2)].copy()
    b["q"] = model.predict_batch(_matrix(b, list(PRE_MAP_FEATURE_NAMES)))
    return b.join(feat, on="match_id")


V, T = block(val), block(te)
FEATS = ["m1_rating_diff", "m1_surprise_diff", "m1_worst_surprise_diff", "m1_adr_resid_diff", "m1_swing_diff",
         "m1_awp_surprise_diff"]
for f in FEATS:
    mu = V[f].mean()
    V[f], T[f] = V[f].fillna(mu), T[f].fillna(mu)
print(f"map-2 rows: fit {len(V)}  test {len(T)}")

lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))


def design(b, fs):
    return np.column_stack([lg(b.q.to_numpy())] + [b[f].to_numpy() for f in fs])


def fit(b, fs, l2=1.0):
    X, y = design(b, fs), b.team_a_won.to_numpy()
    sd = np.r_[1.0, X[:, 1:].std(axis=0) + 1e-9]
    Xs = X / sd
    nll = lambda w: -np.mean(y * (Xs @ w) - np.log1p(np.exp(Xs @ w))) + l2 / len(y) * np.sum(w[1:] ** 2)
    w = minimize(nll, np.r_[1.0, np.zeros(len(fs))], method="BFGS").x
    return w / sd


def predict(b, w, fs):
    return 1 / (1 + np.exp(-(design(b, fs) @ w)))


y, g = T.team_a_won.to_numpy(), T.match_id.to_numpy()
ref = np.clip(T.q.to_numpy(), 1e-4, 1 - 1e-4)
print(f"\ntest log-loss, current map-2 model: {log_loss(y, ref):.4f}\n")
SETS = [["dom"], ["dom", "m1_rating_diff"], ["dom", "m1_surprise_diff"], ["dom", "m1_awp_surprise_diff"],
        ["dom", "m1_worst_surprise_diff"], ["dom", "m1_adr_resid_diff"], ["dom", "m1_swing_diff"], ["dom"] + FEATS]
results = {}
for fs in SETS:
    w = fit(V, fs)
    p = predict(T, w, fs)
    d, lo, hi = clustered_bootstrap_delta(y, p, ref, g, log_loss, n_boot=500)
    results[tuple(fs)] = p
    name = "dom + all player feats" if len(fs) > 3 else " + ".join(fs)
    print(f"  {name:38s} ll {log_loss(y, p):.4f}  vs current {d:+.4f} [{lo:+.4f},{hi:+.4f}] "
          f"{'SIG' if (lo > 0) == (hi > 0) else 'ns '}  coefs {np.round(w[1:], 3)}")

ev = pd.read_parquet(f"{D}/pm_maps_eval.parquet")
ev = ev[(ev.map_pos == 2) & ev.mk_pre5.notna()].join(feat, on="match_id")
for f in FEATS:
    ev[f] = ev[f].fillna(V[f].mean())
print(f"\nvs Polymarket, map 2 priced 5 min after map 1 is decided (n={len(ev)}):")
X = np.column_stack([np.ones(len(ev)), lg(ev.q_h.to_numpy()), ev.dom] + [ev[f] for f in FEATS])
beta = np.linalg.lstsq(X, lg(ev.mk_pre5.to_numpy()), rcond=None)[0]
print("  market's own weights, logit(market) ~ logit(model) + dom + player feats:")
for n, b_ in zip(["dom"] + FEATS, beta[2:]):
    print(f"     {n:24s} {b_:+.3f}")
pd.to_pickle({"feat": feat, "fit": V, "test": T}, f"{D}/map2_players.pkl")
