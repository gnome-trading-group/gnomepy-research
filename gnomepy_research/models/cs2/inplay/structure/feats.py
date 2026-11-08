"""Vectorised state features for the structure entry.

Every per-state feature is a cumulative quantity over the state's own round up to its event_seq (or round-start
information), so it is identical whether the bundle holds the rest of the map or is truncated at the state."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..baseline import with_prior
from ..data import Bundle
from ..features import loss_step, need, round_table, side_swaps_at

RIFLES = {"ak47", "m4a1_s", "m4a4", "galilar", "famas", "aug", "sg553"}
SIDE = {"counter_terrorists": "CT", "terrorists": "T"}


def rounds_frame(b: Bundle, game_ids, prior=None) -> pd.DataFrame:
    """round_table plus the prior oriented to team A, economy left after the buy, and per-round derived columns.
    prior(b, rt) replaces the benchmark prior (kickoff market, else Elo) when given."""
    rt = round_table(b.subset(game_ids))
    rt = prior(b, rt) if prior is not None else with_prior(rt)
    rt["score_diff"] = rt.score_a - rt.score_b
    rt["round_no"] = rt["round"]
    rt["d_need"] = rt.need_a - rt.need_b
    return rt


def next_round_features(rt: pd.DataFrame, a_won: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Score-state features of the round after each row given who won it, plus this round's economy carried over;
    done is 1/0 where this round ends the map."""
    sa, sb = rt.score_a + a_won, rt.score_b + 1 - a_won
    n = rt["round"] + 1
    nd = np.array([need(a, c) for a, c in zip(sa, sb)]).reshape(-1, 2)
    ot = (np.minimum(sa, sb) >= 12).astype(int)
    la, lb = loss_step(rt.a_loss_ctr.to_numpy(), rt.b_loss_ctr.to_numpy(), np.full(len(rt), a_won), n.to_numpy())
    reset = np.asarray((n == 13) | ((n >= 25) & ((n - 25) % 3 == 0)))
    x = pd.DataFrame(dict(
        need_a=nd[:, 0], need_b=nd[:, 1], ct_a=np.where(side_swaps_at(n), 1 - rt.ct_a, rt.ct_a), ot=ot,
        round_in_half=np.where(ot == 1, (n - 25) % 3, (n - 1) % 12), pistol=n.isin([1, 13]).astype(int),
        map_code=rt.map_code, a_loss_ctr=la, b_loss_ctr=lb, prior_logit=rt.prior_logit, has_prior=rt.has_prior,
        # Money left after this round's buy carries into the next round unless the half resets it.
        a_carry=np.where(reset, np.nan, rt.a_money), b_carry=np.where(reset, np.nan, rt.b_money),
        a_eq_prev=np.where(reset, np.nan, rt.a_equip), b_eq_prev=np.where(reset, np.nan, rt.b_equip),
        a_won_last=np.full(len(rt), a_won),
    ), index=rt.index)
    done = np.where(sa >= rt.score_a + rt.need_a, 1.0, np.where(sb >= rt.score_b + rt.need_b, 0.0, np.nan))
    return x, done


def current_round_map_features(rt: pd.DataFrame) -> pd.DataFrame:
    """The same map features for each round as it stands at its own start (for training M on all rounds)."""
    g = rt.groupby("game_id")
    prev = lambda c: g[c].shift(1)
    reset = np.asarray(rt["round"].isin([1, 13]) | ((rt["round"] >= 25) & ((rt["round"] - 25) % 3 == 0)))
    a_won_last = (rt.score_a - prev("score_a")).to_numpy()
    return pd.DataFrame(dict(
        need_a=rt.need_a, need_b=rt.need_b, ct_a=rt.ct_a, ot=rt.ot, round_in_half=rt.round_in_half, pistol=rt.pistol,
        map_code=rt.map_code, a_loss_ctr=rt.a_loss_ctr, b_loss_ctr=rt.b_loss_ctr, prior_logit=rt.prior_logit,
        has_prior=rt.has_prior,
        a_carry=np.where(reset, np.nan, prev("a_money")), b_carry=np.where(reset, np.nan, prev("b_money")),
        a_eq_prev=np.where(reset, np.nan, prev("a_equip")), b_eq_prev=np.where(reset, np.nan, prev("b_equip")),
        a_won_last=a_won_last,
    ), index=rt.index)


def event_features(b: Bundle, game_ids) -> pd.DataFrame:
    """One row per round_start / kill / bomb_planted event with the CT-vs-T state just after it."""
    ids = set(game_ids)
    ev = b.events[b.events.game_id.isin(ids) & b.events.type.isin(["round_start", "kill", "bomb_planted"])]
    ev = ev.sort_values(["game_id", "seq"]).reset_index(drop=True)
    P = b.players[b.players.game_id.isin(ids)]
    ros = pd.DataFrame(dict(game_id=P.game_id.to_numpy(), round=P["round"].to_numpy(), victim_id=P.player_id.to_numpy(),
                            v_side=P.side.to_numpy(), v_eq=P.equip.to_numpy(float), v_awp=(P.prim == "awp").to_numpy(float),
                            v_kit=P.kit.fillna(False).astype(bool).to_numpy(float),
                            v_rifle=P.prim.isin(RIFLES).to_numpy(float)))
    ros = ros.drop_duplicates(["game_id", "round", "victim_id"])
    start = ros.groupby(["game_id", "round", "v_side"])[["v_eq", "v_awp", "v_kit", "v_rifle"]].agg("sum")
    start["n"] = ros.groupby(["game_id", "round", "v_side"]).size()
    start = start.unstack("v_side")
    start.columns = [f"{s.lower()}_{c[2:] if c.startswith('v_') else 'alive'}0" for c, s in start.columns]
    start = start.fillna(0.0)

    is_kill = (ev.type == "kill").to_numpy()
    k = ev[["game_id", "round", "victim_id"]].copy()
    k["victim_id"] = k.victim_id.astype("float64")
    ros_k = ros.assign(victim_id=ros.victim_id.astype("float64"))
    k = k.merge(ros_k, on=["game_id", "round", "victim_id"], how="left")
    first = ~k.duplicated(["game_id", "round", "victim_id"]) & is_kill & k.v_side.notna().to_numpy()
    out = ev[["game_id", "round", "seq", "type", "elapsed"]].copy()
    for s in ("CT", "T"):
        m = (first & (k.v_side == s)).to_numpy()
        for c in ("eq", "awp", "kit", "rifle"):
            out[f"_{s}_{c}"] = np.where(m, k[f"v_{c}"].to_numpy(), 0.0)
        out[f"_{s}_n"] = m.astype(float)
    grp = out.groupby(["game_id", "round"], sort=False)
    cum = grp[[c for c in out.columns if c.startswith("_")]].cumsum()
    out = out.join(start, on=["game_id", "round"])
    for s in ("ct", "t"):
        S = s.upper()
        for c in ("alive", "eq", "awp", "kit", "rifle"):
            if f"{s}_{c}0" not in out:
                out[f"{s}_{c}0"] = 0.0
        out[f"{s}_alive0"] = out[f"{s}_alive0"].fillna(0.0)
        out[f"{s}_alive"] = out[f"{s}_alive0"] - cum[f"_{S}_n"]
        out[f"{s}_eq"] = out[f"{s}_eq0"].fillna(0) - cum[f"_{S}_eq"]
        out[f"{s}_awp"] = out[f"{s}_awp0"].fillna(0) - cum[f"_{S}_awp"]
        out[f"{s}_kits"] = out[f"{s}_kit0"].fillna(0) - cum[f"_{S}_kit"]
        out[f"{s}_rifles"] = out[f"{s}_rifle0"].fillna(0) - cum[f"_{S}_rifle"]
    plant = (out.type == "bomb_planted")
    out["planted"] = plant.astype(float).groupby([out.game_id, out["round"]], sort=False).cummax()
    plant_t = pd.Series(np.where(plant, out.elapsed, np.nan), index=out.index).groupby([out.game_id, out["round"]], sort=False).ffill()
    out["since_plant"] = np.where(out.planted > 0, out.elapsed - plant_t, np.nan)
    site = pd.Series(np.where(plant, (ev.bomb_site == "A").astype(float), np.nan), index=out.index)
    out["site_a"] = site.groupby([out.game_id, out["round"]], sort=False).ffill()
    out["t"] = out.elapsed.astype(float)

    # Sequence features: momentum and trades over the round's kills so far.
    kt = pd.Series(np.where(first, out.elapsed, np.nan), index=out.index)
    last_kill_t = kt.groupby([out.game_id, out["round"]], sort=False).ffill()
    out["since_kill"] = out.t - last_kill_t
    dead_side = np.where(first & (k.v_side == "CT").to_numpy(), 1.0, np.where(first & (k.v_side == "T").to_numpy(), -1.0, np.nan))
    ds = pd.Series(dead_side, index=out.index)
    out["last_dead_ct"] = ds.groupby([out.game_id, out["round"]], sort=False).ffill()
    g = ds.groupby([out.game_id, out["round"]], sort=False)
    seen = ds.notna().astype(int).groupby([out.game_id, out["round"]], sort=False).cumsum()
    out["first_dead_ct"] = np.where(seen > 0, g.transform("first"), np.nan)
    return out.drop(columns=[c for c in out.columns if c.startswith("_")])

