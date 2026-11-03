"""Weekly refit of the symmetric player-skill model: ridge logistic and boosted trees on lineup-difference features."""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

TRAIN_START = pd.Timestamp("2024-08-01")
LABEL_LAG = pd.Timedelta(days=1)
LR_C = 0.3
XGB_PARAMS = dict(n_estimators=250, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7,
                  min_child_weight=20, reg_lambda=5, n_jobs=2, random_state=0)
LR_WEIGHT = 0.5


def diff_cols(f: pd.DataFrame) -> list[str]:
    return [c for c in f.columns if c.startswith("d_")]


def sum_cols(f: pd.DataFrame) -> list[str]:
    return [c for c in f.columns if c.startswith("s_")] + ["tier", "is_lan", "pts_miss", "bo"]


def _flip(x: pd.DataFrame, dcols: list[str]) -> pd.DataFrame:
    return x.assign(**{c: -x[c] for c in dcols})


def fit_predict(train: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    dcols, scols = diff_cols(train), sum_cols(train)
    X, y = train[dcols].fillna(0.0).to_numpy(), train.y.to_numpy()
    sc = StandardScaler(with_mean=False).fit(X)
    lr = LogisticRegression(C=LR_C, fit_intercept=False, max_iter=500)
    lr.fit(sc.transform(np.vstack([X, -X])), np.r_[y, 1 - y])
    p_lr = lr.predict_proba(sc.transform(target[dcols].fillna(0.0).to_numpy()))[:, 1]

    cols = dcols + scols
    xa = pd.concat([train[cols], _flip(train[cols], dcols)])
    gb = xgb.XGBClassifier(**XGB_PARAMS).fit(xa, np.r_[y, 1 - y])
    tg = target[cols]
    p_gb = (gb.predict_proba(tg)[:, 1] + 1 - gb.predict_proba(_flip(tg, dcols))[:, 1]) / 2
    return LR_WEIGHT * p_lr + (1 - LR_WEIGHT) * p_gb


def predict_targets(feats: pd.DataFrame) -> pd.DataFrame:
    hist = feats[~feats.is_target & feats.y.notna() & (feats.bo >= 3) & (feats.t >= TRAIN_START)]
    out = []
    for week, tg in feats[feats.is_target].groupby("week"):
        tr = hist[hist.t < week - LABEL_LAG]
        out.append(pd.DataFrame({"match_id": tg.match_id.to_numpy(), "q": fit_predict(tr, tg), "fit_cutoff": week}))
    return pd.concat(out, ignore_index=True)
