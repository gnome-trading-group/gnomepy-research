"""rounddyn: the benchmark's decomposition P(A wins round) x M(score if A wins) + P(A loses round) x M(score if A
loses), with a richer round model and an economy-aware continuation.

Round model: P(CT wins the round) from the benchmark's state (alive, equipment alive, AWPs, kits, plant, timing,
round-start equipment, rounds needed, map) plus what the timeline says about the round so far (feats.dyn_features), the
pre-map prior turned to the CT team, and the CT team's form earlier in the map (feats.form_features). A round already
decided on the timeline (round_end or bomb_exploded at or before the state) is priced from its known winner.

Continuation: M is learned from round n's rows with their actual outcome (the next score state as the benchmark builds
it, plus round n's start-of-round economy and team A's form), so it is E[map win | round n's start, who won round n]
rather than a score-only round-start model: a team that wins a round with $9k left over is better placed than one
that won it on a force buy."""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from gnomepy_research.models.cs2.inplay import baseline
from gnomepy_research.models.cs2.inplay.data import Bundle
from gnomepy_research.models.cs2.inplay.features import legal_maps, map_states
from gnomepy_research.models.cs2.inplay.rounddyn.feats import FORM, dyn_features, form_features

ROUND = baseline.ROUND + ["ct_rifles", "t_rifles", "ct_smgs", "t_smgs", "ct_helm", "t_helm", "ct_armor", "t_armor",
                          "site", "bomb_left", "clock_left", "defuse_kit_ok", "defuse_nokit_ok", "since_kill",
                          "last_dead", "ct_dead_5s", "t_dead_5s", "last_trade", "open_dead", "ct_max_cls", "t_max_cls",
                          "ct_prior", "has_prior", "ct_kd_prev", "ct_upsets_prev", "ct_gun_wr_prev", "rounds_prev",
                          "gun_prev", "ct_resid_prev"]
ROUND_PARAMS = dict(n_estimators=600, max_depth=6, learning_rate=0.05)
ECON = ["a_money", "b_money", "a_equip", "b_equip", "a_money_pre", "b_money_pre", "a_awp", "b_awp", "a_rifles", "b_rifles"]
CONT = baseline.MAPCOLS + ECON + ["a_cash_next", "b_cash_next", "a_resid_prev"]
EXPECT = ["a_equip", "b_equip", "ct_a", "map_code", "pistol", "a_money_pre", "b_money_pre"]


def round_info(b: Bundle) -> pd.DataFrame:
    rt = baseline.with_prior(baseline.round_table(b))
    return rt.join(form_features(b, rt[["game_id", "round"]]))


def round_won(rt: pd.DataFrame) -> np.ndarray:
    # Known from the next round's score; NaN for a map's latest visible round.
    nxt = rt.groupby("game_id").score_a.shift(-1)
    nr = rt.groupby("game_id")["round"].shift(-1)
    return np.where(nr == rt["round"] + 1, nxt - rt.score_a, np.nan)


def add_resid(rt: pd.DataFrame, expect) -> pd.DataFrame:
    """Team A's round wins earlier in this map beyond what both teams' buys predicted: in-map strength."""
    won = round_won(rt)
    p = expect.predict_proba(rt[EXPECT])[:, 1]
    res = pd.Series(np.where(np.isnan(won), 0.0, won - p), index=rt.index)
    return rt.assign(a_resid_prev=res.groupby(rt.game_id).cumsum() - res)


def rows(b: Bundle, states: pd.DataFrame, expect):
    sub = b.subset(states.game_id)
    x, _ = baseline.round_rows(sub, states)
    rt = add_resid(round_info(sub), expect).set_index(["game_id", "round"])
    d = dyn_features(sub, states)
    x = x.join(d.drop(columns=[c for c in d.columns if c in x.columns]))
    x = x.join(rt[["prior_logit", "has_prior", "a_resid_prev"] + FORM], on=["game_id", "round"])
    sg = np.where(x.side_a == "CT", 1.0, -1.0)
    x["ct_prior"] = sg * x.prior_logit
    x["ct_kd_prev"] = sg * x.a_kd_prev
    x["ct_upsets_prev"] = sg * x.a_upsets_prev
    x["ct_gun_wr_prev"] = np.where(sg > 0, x.a_gun_wr_prev, 1 - x.a_gun_wr_prev)
    x["ct_resid_prev"] = sg * x.a_resid_prev
    return x, rt


def income(ctr, won):
    # Team income for the next round: the win reward, or the loss bonus at the counter after this loss.
    return np.where(won, 3250.0, 1400.0 + 500.0 * np.clip(ctr - 1, 0, 4)) * 5


def cont_rows(rt: pd.DataFrame, a_won: int):
    X, done = baseline.next_round(rt, a_won)
    hp = rt.has_players.to_numpy(bool)
    # Rounds without player data carry zeros for the economy: missing, not broke.
    X = X.assign(**{c: np.where(hp, rt[c].to_numpy(float), np.nan) for c in ECON}, a_resid_prev=rt.a_resid_prev.to_numpy())
    X["a_cash_next"] = X.a_money + income(X.a_loss_ctr.to_numpy(), a_won == 1)
    X["b_cash_next"] = X.b_money + income(X.b_loss_ctr.to_numpy(), a_won == 0)
    return X, done


def fit(train: Bundle, asof: pd.Timestamp, seed: int):
    legal = legal_maps(train)
    rt = round_info(train.subset(legal))
    rt = rt[rt.map_a_wins.notna() & rt.a_wins.notna()].reset_index(drop=True)
    expect = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, n_jobs=8, random_state=seed)
    expect.fit(rt[EXPECT], rt.a_wins)
    rt = add_resid(rt, expect)
    Xs, ys = [], []
    for w in (1, 0):
        X, done = cont_rows(rt, w)
        keep = ((rt.a_wins == w) & np.isnan(done)).to_numpy()
        Xs.append(X[keep][CONT]); ys.append(rt.map_a_wins[keep])
    cont_model = baseline.trees(seed).fit(pd.concat(Xs), pd.concat(ys))
    st = map_states(train, legal)
    x, rti = rows(train, st, expect)
    x = x.join(rti[["a_wins"]], on=["game_id", "round"])
    y = np.where(x.side_a == "CT", x.a_wins, 1 - x.a_wins)
    round_model = baseline.trees(seed, **ROUND_PARAMS).fit(x[ROUND], y)
    return dict(cont_model=cont_model, round_model=round_model, expect=expect)


def predict(model, ctx: Bundle, states: pd.DataFrame) -> np.ndarray:
    x, rt = rows(ctx, states, model["expect"])
    p_ct = model["round_model"].predict_proba(x[ROUND])[:, 1]
    known = ((x.over == 1) & x.over_ct_won.notna()).to_numpy()
    p_ct = np.where(known, x.over_ct_won.to_numpy(), p_ct)
    keys = pd.MultiIndex.from_frame(states[["game_id", "round"]])
    rr = rt.loc[keys.unique()].reset_index()
    m = {}
    for a_won in (1, 0):
        X, done = cont_rows(rr, a_won)
        p = model["cont_model"].predict_proba(X[CONT])[:, 1]
        m[a_won] = pd.Series(np.where(np.isnan(done), p, done), index=pd.MultiIndex.from_frame(rr[["game_id", "round"]]))
    p_a = np.where(x.side_a == "CT", p_ct, 1 - p_ct)
    return p_a * m[1].reindex(keys).to_numpy() + (1 - p_a) * m[0].reindex(keys).to_numpy()
