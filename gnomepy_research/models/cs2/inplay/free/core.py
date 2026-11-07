"""In-map form: the benchmark's round model and score-state map model, with the map model also told how the two teams
have actually played on this map so far, beyond what the score says.

The score already counts rounds won, but not how they were won. A team that wins rounds it was economically expected
to lose, or that out-kills its opponent in the rounds it loses, is stronger than its pre-map prior said. Two summaries
of the map so far go into the map model:

- res_mean: rounds won minus rounds expected from both teams' round-start equipment (a small logistic model), over the
  previous rounds, shrunk towards 0 by four pseudo-rounds. Known at a round's start from the score progression and the
  rounds table, so it exists for every map since 2025 and is updated exactly in each branch of the combine formula
  (the current round's expectation is known at its start).
- kill_mean: kills minus deaths over the rounds before the previous one (from the event timeline, so 2026 only; NaN
  otherwise), shrunk the same way. It lags one round so that the branch for the next round needs only completed
  rounds: the current round's kills are not complete until it ends.

Everything comes from rounds up to the priced round (score, round-start equipment) and events of earlier rounds, so a
state's price is unchanged when later rounds/events are removed."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from gnomepy_research.models.cs2.inplay.baseline import MAPCOLS, ROUND, next_round, round_rows, trees, with_prior
from gnomepy_research.models.cs2.inplay.data import Bundle
from gnomepy_research.models.cs2.inplay.features import legal_maps, map_states, round_table

TIER = {"s": 0, "a": 1, "b": 2, "c": 3, "d": 4}
SHRINK = 4.0

CFG = dict(map_extra=["res_mean", "kill_mean"], map_kw={}, round_extra=[], round_kw={})


def exp_design(rt: pd.DataFrame) -> np.ndarray:
    ct_eq = np.where(rt.ct_a == 1, rt.a_equip, rt.b_equip) / 1000
    t_eq = np.where(rt.ct_a == 1, rt.b_equip, rt.a_equip) / 1000
    lc, lt = np.log1p(ct_eq), np.log1p(t_eq)
    return np.c_[lc, lt, lc * lt, np.eye(11)[rt.map_code.to_numpy() + 1]]


def fit_expect(rt: pd.DataFrame) -> LogisticRegression:
    y_ct = np.where(rt.ct_a == 1, rt.a_wins, 1 - rt.a_wins)
    return LogisticRegression(C=10, max_iter=2000).fit(exp_design(rt), y_ct)


def round_kills(b: Bundle) -> pd.DataFrame:
    """Kills minus deaths of team A per (game, round), from kill victims (kill credits miss team kills)."""
    ev = b.events[b.events.type == "kill"][["game_id", "round", "victim_team"]]
    ev = ev.merge(b.games[["game_id", "team_a", "team_b"]], on="game_id")
    kd = (ev.victim_team == ev.team_b).astype(int) - (ev.victim_team == ev.team_a).astype(int)
    return kd.groupby([ev.game_id, ev["round"]]).sum().rename("kd")


def history(b: Bundle, rt: pd.DataFrame, expect: LogisticRegression) -> pd.DataFrame:
    """Per round, the in-map form features as known at its start, plus what the next round's start needs."""
    rt = rt.sort_values(["game_id", "round"]).copy()
    g = rt.game_id
    p_ct = expect.predict_proba(exp_design(rt))[:, 1]
    rt["p_exp_a"] = np.where(rt.ct_a == 1, p_ct, 1 - p_ct)
    # Who won a round is known once the next round starts, from the score.
    a_won = rt.groupby("game_id").score_a.shift(-1) - rt.score_a
    res = a_won - rt.p_exp_a
    rt["res_sum"] = res.groupby(g).cumsum().groupby(g).shift(1).fillna(0.0)
    rt["n_prev"] = rt["round"] - 1
    rt["res_mean"] = rt.res_sum / (rt.n_prev + SHRINK)
    has_ev = g.isin(set(b.events.game_id)).to_numpy()
    kd = rt.join(round_kills(b), on=["game_id", "round"]).kd.fillna(0.0)
    cs = kd.groupby(g).cumsum()
    lag2 = cs.groupby(g).shift(2).fillna(0.0)
    lag1 = cs.groupby(g).shift(1).fillna(0.0)
    rt["kill_mean"] = np.where(has_ev, lag2 / ((rt["round"] - 2).clip(lower=0) + SHRINK), np.nan)
    rt["kill_next"] = np.where(has_ev, lag1 / (rt["round"] - 1 + SHRINK), np.nan)
    rt["tier_code"] = rt.tier.map(TIER).fillna(4).astype(int)
    return rt


def table(b: Bundle, expect) -> pd.DataFrame:
    rt = with_prior(round_table(b)).merge(b.games[["game_id", "tier"]], on="game_id", how="left")
    return history(b, rt, expect)


def next_ext(rr: pd.DataFrame, a_won: int, extra):
    X, done = next_round(rr, a_won)
    for c in extra:
        if c == "res_mean":
            X[c] = ((rr.res_sum + a_won - rr.p_exp_a) / (rr.n_prev + 1 + SHRINK)).to_numpy()
        elif c == "kill_mean":
            X[c] = rr.kill_next.to_numpy()
        else:
            X[c] = rr[c].to_numpy()
    return X, done


def fit(train: Bundle, asof: pd.Timestamp, seed: int):
    legal = legal_maps(train)
    tb = train.subset(legal)
    rt0 = with_prior(round_table(tb))
    expect = fit_expect(rt0)
    rt = table(tb, expect)
    rt = rt[rt.map_a_wins.notna()]
    mapcols = MAPCOLS + CFG["map_extra"]
    map_model = trees(seed, **CFG["map_kw"]).fit(rt[mapcols], rt.map_a_wins)
    st = map_states(train, legal)
    x, rti = round_rows(train, st)
    x = x.join(rti[["a_wins"]], on=["game_id", "round"])
    y = np.where(x.side_a == "CT", x.a_wins, 1 - x.a_wins)
    rkw = dict(max_depth=6, learning_rate=0.05)
    rkw.update(CFG["round_kw"])
    round_model = trees(seed, **rkw).fit(x[ROUND], y)
    return dict(map_model=map_model, round_model=round_model, expect=expect)


def predict(model, ctx: Bundle, states: pd.DataFrame) -> np.ndarray:
    x, _ = round_rows(ctx, states)
    p_ct = model["round_model"].predict_proba(x[ROUND])[:, 1]
    keys = pd.MultiIndex.from_frame(states[["game_id", "round"]])
    rt = table(ctx.subset(states.game_id), model["expect"]).set_index(["game_id", "round"])
    rr = rt.loc[keys.unique()].reset_index()
    extra = CFG["map_extra"]
    m = {}
    for a_won in (1, 0):
        X, done = next_ext(rr, a_won, extra)
        p = model["map_model"].predict_proba(X[MAPCOLS + extra])[:, 1]
        m[a_won] = pd.Series(np.where(np.isnan(done), p, done), index=pd.MultiIndex.from_frame(rr[["game_id", "round"]]))
    p_a = np.where(x.side_a == "CT", p_ct, 1 - p_ct)
    return p_a * m[1].reindex(keys).to_numpy() + (1 - p_a) * m[0].reindex(keys).to_numpy()
