"""Walk-forward diagnostics for state configurations, run on Batch: one task per JSON list of RatingConfig overrides.

    python -m gnomepy_research.sessions.cs2_comp.g08.tune '<json list of overrides>' [layer columns]

For each configuration: log loss of the weekly-refitted layer on Sep 2025 - Mar 2026 bo3 series (before every dev
week), and on the dev targets q - market, pool - market and, with the e12 reference prices at /work/in/e12, the stack
over market and e12 (pools refitted with each month held out, as the harness does). Dev outcomes come from the bundle's
HLTV history; they are only ever used to score."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from gnomepy_research.sessions.cs2_comp.bundle import load
from gnomepy_research.sessions.cs2_comp.g08.data import Inputs
from gnomepy_research.sessions.cs2_comp.g08.market import path_features
from gnomepy_research.sessions.cs2_comp.g08.model import _queries, layer_features, logit
from gnomepy_research.sessions.cs2_comp.g08.ratings import RatingConfig, run

DATA = Path("/work/in/comp")
E12 = Path("/work/in/e12/e12_dev.parquet")
PRE_START = pd.Timestamp("2025-09-01")


def ll(p, y):
    p = np.clip(np.asarray(p, float), 0.01, 0.99)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def pool(T, cols):
    X = np.column_stack([logit(T[c]) for c in cols])
    y = T.y.to_numpy()
    out = np.zeros(len(T))
    for m in T.month.unique():
        tr = (T.month != m).to_numpy()
        out[~tr] = LogisticRegression(C=1e4).fit(X[tr], y[tr]).predict_proba(X[~tr])[:, 1]
    return ll(out, y)


def ci(d):
    rng = np.random.default_rng(0)
    b = d[rng.integers(0, len(d), (1000, len(d)))].mean(1)
    return [float(d.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def _residual_features(F: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Antisymmetric (flip sign with the teams) and symmetric descriptors for the boosted residual layer."""
    anti = np.column_stack([logit(F.series_veto), (F.form_a - F.form_b), (F.cohesion_a - F.cohesion_b), (F.unknown_a - F.unknown_b),
                            (F.var_a - F.var_b), np.log1p(F.maps90_a) - np.log1p(F.maps90_b),
                            np.log1p(F.idle_a) - np.log1p(F.idle_b), (F.lowmaps_a - F.lowmaps_b)]).astype(float)
    sym = np.column_stack([F.var_a + F.var_b, F.lan, F.tier, np.log1p(np.minimum(F.maps90_a, F.maps90_b)),
                           F.unknown_a + F.unknown_b, F.cohesion_a + F.cohesion_b]).astype(float)
    return anti, sym


def evaluate(inputs, targets, market, cfg, cols):
    Q = _queries(inputs, targets)
    F = Q.merge(run(inputs.maps, Q, cfg), on="qid")
    F["y"] = F.match_id.map(inputs.series.set_index("match_id").y)
    boost = cols[-1] == "boost"
    cols = [c for c in cols if c != "boost"]
    X = layer_features(F, cols)
    A, S = _residual_features(F)
    weeks = sorted(set(F.cutoff[~F.target & (F.cutoff >= PRE_START) & (F.cutoff < targets.week_cutoff.min())]) | set(targets.week_cutoff))
    p = np.full(len(F), np.nan)
    for w in weeks:
        w = pd.Timestamp(w)
        tr = (~F.target & (F.known < w) & (F.cutoff >= w - pd.Timedelta(days=540))).to_numpy()
        te = (F.cutoff == w).to_numpy() & (F.target | (w < targets.week_cutoff.min())).to_numpy()
        yt = np.r_[F.y.values[tr], 1 - F.y.values[tr]]
        m = LogisticRegression(fit_intercept=False).fit(np.vstack([X[tr], -X[tr]]), yt)
        p[te] = m.predict_proba(X[te])[:, 1]
        if boost:
            z_tr = np.r_[X[tr] @ m.coef_[0], -(X[tr] @ m.coef_[0])]
            D = np.vstack([np.hstack([A[tr], S[tr]]), np.hstack([-A[tr], S[tr]])])
            b = XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                              min_child_weight=20, random_state=0, n_jobs=2)
            b.fit(D, yt, base_margin=z_tr)
            z = X[te] @ m.coef_[0]
            pa = b.predict_proba(np.hstack([A[te], S[te]]), base_margin=z)[:, 1]
            pb = b.predict_proba(np.hstack([-A[te], S[te]]), base_margin=-z)[:, 1]
            p[te] = 0.5 * (pa + 1 - pb)
    F["q"] = p
    pre = F[~F.target & F.q.notna() & F.y.notna()]
    T = F[F.target][["match_id", "q", "y"]].merge(market, on="match_id")
    if E12.exists():
        T = T.merge(pd.read_parquet(E12)[["match_id", "q"]].rename(columns={"q": "e12"}), on="match_id")
    y = T.y.to_numpy()
    lm = ll(T.m0, y)
    res = {"pre_ll": float(ll(pre.q, pre.y).mean()), "n_pre": len(pre), "q_m": ci(ll(T.q, y) - lm), "pool_m": ci(pool(T, ["m0", "q"]) - lm)}
    if "e12" in T:
        lb = pool(T, ["m0", "e12"])
        res["stack"] = ci(pool(T, ["m0", "q", "e12"]) - lb)
        T["ens"] = 1 / (1 + np.exp(-(0.5 * logit(T.q) + 0.5 * logit(T.e12))))
        res["ens_pool_m"] = ci(pool(T, ["m0", "ens"]) - lm)
    return res, F


def main() -> None:
    overrides = json.loads(sys.argv[1])
    cols = sys.argv[2].split(",") if len(sys.argv) > 2 else ["sv"]
    out = Path(os.environ.get("OUTPUT_DIR", "."))
    targets = pd.read_parquet(DATA / "targets_tune.parquet")
    inputs = Inputs(DATA, pd.Timestamp("2026-08-01"), faceit=True, market=True)
    paths = load(DATA, "market/series_paths", pd.Timestamp("2026-08-01"))
    market = path_features(paths)[["match_id", "m0", "p_last", "kickoff"]]
    market["m0"] = market.m0.fillna(market.p_last)
    market["month"] = market.kickoff.dt.month
    market = market[["match_id", "m0", "month"]]
    results = []
    for ov in overrides:
        res, _ = evaluate(inputs, targets, market, RatingConfig(**ov), cols)
        res["config"] = ov
        res["cols"] = cols
        print(json.dumps(res), flush=True)
        results.append(res)
    (out / f"tune_{abs(hash(sys.argv[1])) % 10**8}.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
