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
CFG = dict(map_extra=["res_mean", "kill_mean"], map_kw={}, round_kw={}, shrink=4.0, expect="basic", alive_end_kills=False, nop_fix=False, decay=0.0)


def exp_design(rt: pd.DataFrame) -> np.ndarray:
    ct_eq = np.where(rt.ct_a == 1, rt.a_equip, rt.b_equip) / 1000
    t_eq = np.where(rt.ct_a == 1, rt.b_equip, rt.a_equip) / 1000
    lc, lt = np.log1p(ct_eq), np.log1p(t_eq)
    cols = [lc, lt, lc * lt, np.eye(11)[rt.map_code.to_numpy() + 1]]
    if CFG["expect"] == "rich":
        a_ct = rt.ct_a.to_numpy() == 1
        for k in ("awp", "rifles", "helmets"):
            cols += [np.where(a_ct, rt[f"a_{k}"], rt[f"b_{k}"]), np.where(a_ct, rt[f"b_{k}"], rt[f"a_{k}"])]
        cols += [rt.pistol.to_numpy(), lc ** 2, lt ** 2]
    return np.column_stack(cols)


def map_design(rt: pd.DataFrame) -> np.ndarray:
    return np.c_[np.eye(11)[rt.map_code.to_numpy() + 1], rt.pistol.to_numpy()]


def fit_expect(rt: pd.DataFrame):
    y_ct = np.where(rt.ct_a == 1, rt.a_wins, 1 - rt.a_wins)
    if not CFG["nop_fix"]:
        return LogisticRegression(C=10, max_iter=2000).fit(exp_design(rt), y_ct)
    # Rounds without player data have no equipment (zeros), so they get a side-and-map-only expectation instead.
    hp = rt.has_players.to_numpy(bool)
    return (LogisticRegression(C=10, max_iter=2000).fit(exp_design(rt[hp]), y_ct[hp]),
            LogisticRegression(C=10, max_iter=2000).fit(map_design(rt[hp]), y_ct[hp]))


def expect_ct(expect, rt: pd.DataFrame) -> np.ndarray:
    if not isinstance(expect, tuple):
        return expect.predict_proba(exp_design(rt))[:, 1]
    full, bare = expect
    return np.where(rt.has_players.to_numpy(bool), full.predict_proba(exp_design(rt))[:, 1], bare.predict_proba(map_design(rt))[:, 1])


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
    shrink = CFG["shrink"]
    p_ct = expect_ct(expect, rt)
    rt["p_exp_a"] = np.where(rt.ct_a == 1, p_ct, 1 - p_ct)
    # Who won a round is known once the next round starts, from the score.
    a_won = rt.groupby("game_id").score_a.shift(-1) - rt.score_a
    res = a_won - rt.p_exp_a
    rt["res_sum"] = res.groupby(g).cumsum().groupby(g).shift(1).fillna(0.0)
    rt["n_prev"] = rt["round"] - 1
    rt["res_mean"] = rt.res_sum / (rt.n_prev + shrink)
    has_ev = g.isin(set(b.events.game_id)).to_numpy()
    kd = rt.join(round_kills(b), on=["game_id", "round"]).kd.fillna(0.0)
    if CFG["alive_end_kills"] and "a_alive_end" in rt:
        # Maps without a timeline (2025) still record survivors per round in training data, and with equal rosters
        # kills minus deaths equals survivors minus the opponent's survivors. Scored maps always use the timeline.
        surv = (rt.a_alive_end - rt.b_alive_end).astype(float)
        use = ~has_ev & rt.groupby("game_id").a_alive_end.transform(lambda v: v.notna().all()).to_numpy()
        kd = pd.Series(np.where(use, surv.fillna(0.0), kd), index=rt.index)
        has_ev = has_ev | use
    cs = kd.groupby(g).cumsum()
    lag2 = cs.groupby(g).shift(2).fillna(0.0)
    lag1 = cs.groupby(g).shift(1).fillna(0.0)
    rt["kill_mean"] = np.where(has_ev, lag2 / ((rt["round"] - 2).clip(lower=0) + shrink), np.nan)
    rt["kill_next"] = np.where(has_ev, lag1 / (rt["round"] - 1 + shrink), np.nan)
    # kill1_*: kills through the previous round, for the variant that adds the current round's kills so far
    rt["kill1_mean"], rt["kill_cs1"] = rt.kill_next, np.where(has_ev, lag1, np.nan)
    for side, flag in (("ct", 1), ("t", 0)):
        on = (rt.ct_a == flag).astype(float)
        v = (res * on).fillna(0.0)
        rt[f"res_{side}_sum"] = v.groupby(g).cumsum().groupby(g).shift(1).fillna(0.0)
        rt[f"n_{side}"] = on.groupby(g).cumsum().groupby(g).shift(1).fillna(0.0)
        rt[f"res_{side}_mean"] = rt[f"res_{side}_sum"] / (rt[f"n_{side}"] + shrink)
        kv, kn = (kd * on).groupby(g).cumsum(), on.groupby(g).cumsum()
        rt[f"kill_{side}_mean"] = np.where(has_ev, kv.groupby(g).shift(2).fillna(0.0) / (kn.groupby(g).shift(2).fillna(0.0) + shrink), np.nan)
        rt[f"kill_{side}_next"] = np.where(has_ev, kv.groupby(g).shift(1).fillna(0.0) / (kn.groupby(g).shift(1).fillna(0.0) + shrink), np.nan)
        rt[f"kill1_{side}_mean"] = rt[f"kill_{side}_next"]
        rt[f"kill_{side}_cs1"] = np.where(has_ev, kv.groupby(g).shift(1).fillna(0.0), np.nan)
        rt[f"kill_{side}_n1"] = kn.groupby(g).shift(1).fillna(0.0)
    if CFG["decay"]:
        d = CFG["decay"]
        games, r_ = g.to_numpy(), res.fillna(0.0).to_numpy()
        k_ = np.asarray(kd, float)
        n = len(rt)
        S, W, K, KW, Kn, KWn = (np.zeros(n) for _ in range(6))
        for i in range(1, n):
            if games[i] == games[i - 1]:
                S[i], W[i] = d * S[i - 1] + r_[i - 1], d * W[i - 1] + 1
                Kn[i], KWn[i] = d * Kn[i - 1] + k_[i - 1], d * KWn[i - 1] + 1
                K[i], KW[i] = Kn[i - 1], KWn[i - 1]
        rt["res_dec_S"], rt["res_dec_W"] = S, W
        rt["res_dec"] = S / (W + shrink)
        rt["kill_dec"] = np.where(has_ev, K / (KW + shrink), np.nan)
        rt["kill_dec_next"] = np.where(has_ev, Kn / (KWn + shrink), np.nan)
    rt["last_pexp"] = rt.groupby("game_id").p_exp_a.shift(1).fillna(0.5)
    rt["last_won"] = a_won.groupby(g).shift(1).fillna(0.5)
    rt["tier_code"] = rt.tier.map(TIER).fillna(4).astype(int)
    return rt


def table(b: Bundle, expect) -> pd.DataFrame:
    rt = with_prior(round_table(b)).merge(b.games[["game_id", "tier"]], on="game_id", how="left")
    return history(b, rt, expect)


def next_ext(rr: pd.DataFrame, a_won: int, extra, kd_now=None):
    """Map-model features of the next round's start given who wins this one. kd_now (per row, team A's kills minus
    deaths so far in this round) feeds the kill1_* features."""
    shrink = CFG["shrink"]
    X, done = next_round(rr, a_won)
    for c in extra:
        if c.startswith("kill1_"):
            side = c.split("_")[1] if c != "kill1_mean" else None
            if side is None:
                X[c] = ((rr.kill_cs1 + kd_now) / (rr.n_prev + 1 + shrink)).to_numpy()
            else:
                on = (rr.ct_a == (1 if side == "ct" else 0)).astype(float)
                X[c] = ((rr[f"kill_{side}_cs1"] + on * kd_now) / (rr[f"kill_{side}_n1"] + on + shrink)).to_numpy()
            continue
        if c == "res_mean":
            X[c] = ((rr.res_sum + a_won - rr.p_exp_a) / (rr.n_prev + 1 + shrink)).to_numpy()
        elif c == "res_dec":
            d = CFG["decay"]
            X[c] = ((d * rr.res_dec_S + a_won - rr.p_exp_a) / (d * rr.res_dec_W + 1 + shrink)).to_numpy()
        elif c == "last_pexp":
            X[c] = rr.p_exp_a.to_numpy()
        elif c == "last_won":
            X[c] = float(a_won)
        elif c == "kill_dec":
            X[c] = rr.kill_dec_next.to_numpy()
        elif c in ("res_ct_mean", "res_t_mean"):
            side = c.split("_")[1]
            on = (rr.ct_a == (1 if side == "ct" else 0)).astype(float)
            X[c] = ((rr[f"res_{side}_sum"] + on * (a_won - rr.p_exp_a)) / (rr[f"n_{side}"] + on + shrink)).to_numpy()
        elif c.startswith("kill_"):
            X[c] = rr[c.replace("_mean", "_next")].to_numpy()
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
    rt = table(ctx.subset(states.game_id), model["expect"])
    rr = states[["game_id", "round"]].join(rt.set_index(["game_id", "round"]), on=["game_id", "round"]).reset_index(drop=True)
    a_ct = (x.side_a == "CT").to_numpy()
    # Equal starting rosters, so A's kills minus deaths so far is A's alive count minus B's.
    kd_now = np.where(a_ct, x.ct_alive - x.t_alive, x.t_alive - x.ct_alive)
    return combine(model["map_model"], rr, p_ct, a_ct, kd_now)


def combine(map_model, rr, p_ct, a_ct, kd_now):
    extra = CFG["map_extra"]
    m = {}
    for a_won in (1, 0):
        X, done = next_ext(rr, a_won, extra, kd_now)
        m[a_won] = np.where(np.isnan(done), map_model.predict_proba(X[MAPCOLS + extra])[:, 1], done)
    p_a = np.where(a_ct, p_ct, 1 - p_ct)
    return p_a * m[1] + (1 - p_a) * m[0]
