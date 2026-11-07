"""In-map form: the benchmark's round model and score-state map model, with the map model also told how the two teams
have actually played on this map so far, beyond what the score says.

The score counts rounds won, but not how they were won. A team that wins rounds it was economically expected to lose,
or out-kills its opponent, is playing better than its pre-map prior said. The map model gets, from team A's view:

- res_mean: rounds won minus rounds expected from both teams' round-start equipment (a logistic model fitted on the
  training maps), over the previous rounds, shrunk towards 0 by four pseudo-rounds. Who won a previous round is read
  off the next round's score, so this exists for every map since 2025, and each branch of the combination formula
  updates it exactly (the current round's expectation is known at its start).
- kill_mean: kills minus deaths over the rounds before the previous one, from the event timeline (2026 maps only;
  NaN otherwise). It lags a round so that the next round's value needs only completed rounds.
- res_ct_mean, res_t_mean, kill_ct_mean, kill_t_mean: the same over the rounds A played on each side. They separate
  "A is stronger today" from "this matchup plays CT-sided today", which flips at every side swap.
- res_dec, kill_dec: recency-weighted versions (weight decay^age), for momentum the long averages smooth away.

Everything comes from the priced map's rounds up to the priced one (scores, round-start equipment) and events of
earlier rounds, so a state's price is unchanged when later rounds and events are removed."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from gnomepy_research.models.cs2.inplay.baseline import MAPCOLS, ROUND, next_round, round_rows, trees, with_prior
from gnomepy_research.models.cs2.inplay.data import Bundle
from gnomepy_research.models.cs2.inplay.features import legal_maps, map_states, round_table

FORM = ["res_mean", "kill_mean", "res_ct_mean", "res_t_mean", "kill_ct_mean", "kill_t_mean", "res_dec", "kill_dec"]
SHRINK = 4.0
DECAY = 0.2
MAP_KW = dict(min_child_weight=200)


def exp_design(rt: pd.DataFrame) -> np.ndarray:
    a_ct = rt.ct_a.to_numpy() == 1
    lc = np.log1p(np.where(a_ct, rt.a_equip, rt.b_equip) / 1000)
    lt = np.log1p(np.where(a_ct, rt.b_equip, rt.a_equip) / 1000)
    cols = [lc, lt, lc * lt, np.eye(11)[rt.map_code.to_numpy() + 1]]
    for k in ("awp", "rifles", "helmets"):
        cols += [np.where(a_ct, rt[f"a_{k}"], rt[f"b_{k}"]), np.where(a_ct, rt[f"b_{k}"], rt[f"a_{k}"])]
    cols += [rt.pistol.to_numpy(), lc ** 2, lt ** 2]
    return np.column_stack(cols)


def fit_expect(rt: pd.DataFrame) -> LogisticRegression:
    y_ct = np.where(rt.ct_a == 1, rt.a_wins, 1 - rt.a_wins)
    return LogisticRegression(C=10, max_iter=2000).fit(exp_design(rt), y_ct)


def round_kills(b: Bundle) -> pd.Series:
    """Kills minus deaths of team A per (game, round), from kill victims (kill credits miss team kills)."""
    ev = b.events[b.events.type == "kill"][["game_id", "round", "victim_team"]]
    ev = ev.merge(b.games[["game_id", "team_a", "team_b"]], on="game_id")
    kd = (ev.victim_team == ev.team_b).astype(int) - (ev.victim_team == ev.team_a).astype(int)
    return kd.groupby([ev.game_id, ev["round"]]).sum().rename("kd")


def decayed(games: np.ndarray, res: np.ndarray, kd: np.ndarray):
    """Recency-weighted sums over earlier rounds of the same map: residuals through the previous round (S, W), and
    kills through the round before it (K, KW) and through the previous round (Kn, KWn)."""
    n = len(games)
    S, W, K, KW, Kn, KWn = (np.zeros(n) for _ in range(6))
    for i in range(1, n):
        if games[i] == games[i - 1]:
            S[i], W[i] = DECAY * S[i - 1] + res[i - 1], DECAY * W[i - 1] + 1
            Kn[i], KWn[i] = DECAY * Kn[i - 1] + kd[i - 1], DECAY * KWn[i - 1] + 1
            K[i], KW[i] = Kn[i - 1], KWn[i - 1]
    return S, W, K, KW, Kn, KWn


def history(b: Bundle, rt: pd.DataFrame, expect: LogisticRegression) -> pd.DataFrame:
    """Per round, the form features as known at its start, plus what the next round's start needs."""
    rt = rt.sort_values(["game_id", "round"]).copy()
    g = rt.game_id
    p_ct = expect.predict_proba(exp_design(rt))[:, 1]
    rt["p_exp_a"] = np.where(rt.ct_a == 1, p_ct, 1 - p_ct)
    a_won = rt.groupby("game_id").score_a.shift(-1) - rt.score_a
    res = a_won - rt.p_exp_a
    rt["res_sum"] = res.groupby(g).cumsum().groupby(g).shift(1).fillna(0.0)
    rt["n_prev"] = rt["round"] - 1
    rt["res_mean"] = rt.res_sum / (rt.n_prev + SHRINK)
    has_ev = g.isin(set(b.events.game_id)).to_numpy()
    kd = rt.join(round_kills(b), on=["game_id", "round"]).kd.fillna(0.0)
    cs = kd.groupby(g).cumsum()
    rt["kill_mean"] = np.where(has_ev, cs.groupby(g).shift(2).fillna(0.0) / ((rt["round"] - 2).clip(lower=0) + SHRINK), np.nan)
    rt["kill_next"] = np.where(has_ev, cs.groupby(g).shift(1).fillna(0.0) / (rt["round"] - 1 + SHRINK), np.nan)
    for side, flag in (("ct", 1), ("t", 0)):
        on = (rt.ct_a == flag).astype(float)
        rt[f"res_{side}_sum"] = (res * on).fillna(0.0).groupby(g).cumsum().groupby(g).shift(1).fillna(0.0)
        rt[f"n_{side}"] = on.groupby(g).cumsum().groupby(g).shift(1).fillna(0.0)
        rt[f"res_{side}_mean"] = rt[f"res_{side}_sum"] / (rt[f"n_{side}"] + SHRINK)
        kv, kn = (kd * on).groupby(g).cumsum(), on.groupby(g).cumsum()
        rt[f"kill_{side}_mean"] = np.where(has_ev, kv.groupby(g).shift(2).fillna(0.0) / (kn.groupby(g).shift(2).fillna(0.0) + SHRINK), np.nan)
        rt[f"kill_{side}_next"] = np.where(has_ev, kv.groupby(g).shift(1).fillna(0.0) / (kn.groupby(g).shift(1).fillna(0.0) + SHRINK), np.nan)
    S, W, K, KW, Kn, KWn = decayed(g.to_numpy(), res.fillna(0.0).to_numpy(), np.asarray(kd, float))
    rt["res_dec_S"], rt["res_dec_W"] = S, W
    rt["res_dec"] = S / (W + SHRINK)
    rt["kill_dec"] = np.where(has_ev, K / (KW + SHRINK), np.nan)
    rt["kill_dec_next"] = np.where(has_ev, Kn / (KWn + SHRINK), np.nan)
    return rt


def table(b: Bundle, expect: LogisticRegression) -> pd.DataFrame:
    return history(b, with_prior(round_table(b)), expect)


def next_ext(rr: pd.DataFrame, a_won: int):
    """Map-model features at the next round's start, given whether A wins the current one; done is 1/0 where the
    current round ends the map."""
    X, done = next_round(rr, a_won)
    res_now = a_won - rr.p_exp_a
    X["res_mean"] = ((rr.res_sum + res_now) / (rr.n_prev + 1 + SHRINK)).to_numpy()
    X["kill_mean"] = rr.kill_next.to_numpy()
    for side, flag in (("ct", 1), ("t", 0)):
        on = (rr.ct_a == flag).astype(float)
        X[f"res_{side}_mean"] = ((rr[f"res_{side}_sum"] + on * res_now) / (rr[f"n_{side}"] + on + SHRINK)).to_numpy()
        X[f"kill_{side}_mean"] = rr[f"kill_{side}_next"].to_numpy()
    X["res_dec"] = ((DECAY * rr.res_dec_S + res_now) / (DECAY * rr.res_dec_W + 1 + SHRINK)).to_numpy()
    X["kill_dec"] = rr.kill_dec_next.to_numpy()
    return X[MAPCOLS + FORM], done


def fit(train: Bundle, asof: pd.Timestamp, seed: int):
    legal = legal_maps(train)
    tb = train.subset(legal)
    expect = fit_expect(with_prior(round_table(tb)))
    rt = table(tb, expect)
    rt = rt[rt.map_a_wins.notna()]
    map_model = trees(seed, **MAP_KW).fit(rt[MAPCOLS + FORM], rt.map_a_wins)
    st = map_states(train, legal)
    x, rti = round_rows(train, st)
    x = x.join(rti[["a_wins"]], on=["game_id", "round"])
    y = np.where(x.side_a == "CT", x.a_wins, 1 - x.a_wins)
    round_model = trees(seed, max_depth=6, learning_rate=0.05).fit(x[ROUND], y)
    return dict(map_model=map_model, round_model=round_model, expect=expect)


def predict(model, ctx: Bundle, states: pd.DataFrame) -> np.ndarray:
    x, _ = round_rows(ctx, states)
    p_ct = model["round_model"].predict_proba(x[ROUND])[:, 1]
    keys = pd.MultiIndex.from_frame(states[["game_id", "round"]])
    rt = table(ctx.subset(states.game_id), model["expect"]).set_index(["game_id", "round"])
    rr = rt.loc[keys.unique()].reset_index()
    m = {}
    for a_won in (1, 0):
        X, done = next_ext(rr, a_won)
        p = model["map_model"].predict_proba(X)[:, 1]
        m[a_won] = pd.Series(np.where(np.isnan(done), p, done), index=pd.MultiIndex.from_frame(rr[["game_id", "round"]]))
    p_a = np.where(x.side_a == "CT", p_ct, 1 - p_ct)
    return p_a * m[1].reindex(keys).to_numpy() + (1 - p_a) * m[0].reindex(keys).to_numpy()
