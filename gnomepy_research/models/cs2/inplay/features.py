"""Shared feature code: the score state of each round (rounds still needed, side, overtime, loss-bonus counters,
cleaned economy), the state after each event in a round (players alive, equipment alive, plant, timing), and the map
states the harness scores. Nothing here reads a round's own outcome: previous rounds' results come from the score
progression, which is known before the next round starts."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Bundle

PRICE_KEYS = ("equip", "money", "money_pre", "awp", "rifles", "helmets")


def need(sa: int, sb: int) -> tuple[int, int]:
    """Rounds each side still needs: 13 in regulation (MR12); in overtime, first to 4 of each six-round block (MR3)."""
    if min(sa, sb) < 12:
        return 13 - sa, 13 - sb
    base = 12 + 3 * ((min(sa, sb) - 12) // 3)
    return base + 4 - sa, base + 4 - sb


def resets(n):
    # Money (and so the loss bonus) resets at each half and at the start of each overtime half.
    n = np.asarray(n)
    return (n == 1) | (n == 13) | ((n >= 25) & ((n - 25) % 3 == 0))


def side_swaps_at(n):
    # Sides swap at round 13; overtime keeps the second-half sides for its first three rounds, then swaps every three.
    n = np.asarray(n)
    return (n == 13) | ((n >= 28) & ((n - 28) % 6 == 0))


def loss_step(la, lb, a_won, next_round):
    """Loss-bonus counters after a round as they stand at the start of next_round: +1 per loss (max 4), -1 per win.
    CS2's exact rule on a win is uncertain (reset vs step down); this is an approximation."""
    la2 = np.where(a_won == 1, np.maximum(la - 1, 0), np.minimum(la + 1, 4))
    lb2 = np.where(a_won == 1, np.minimum(lb + 1, 4), np.maximum(lb - 1, 0))
    z = resets(next_round)
    return np.where(z, 0, la2), np.where(z, 0, lb2)


def legal_maps(b: Bundle) -> set:
    """Maps with a possible CS2 final score and contiguous rounds (needs outcomes: training/scoring only)."""
    r = b.rounds.sort_values(["game_id", "round"])
    last = r.groupby("game_id").tail(1)
    fa, fb = last.score_a + last.a_wins, last.score_b + 1 - last.a_wins
    hi, lo = np.maximum(fa, fb), np.minimum(fa, fb)
    ok = ((hi == 13) & (lo <= 11)) | ((hi >= 16) & ((hi - 13) % 3 == 0) & (hi - lo >= 2) & (hi - lo <= 4) & (lo >= 12))
    contig = r.groupby("game_id")["round"].agg(lambda x: (x.to_numpy() == np.arange(1, len(x) + 1)).all())
    nd = np.array([need(a, c) for a, c in zip(r.score_a, r.score_b)])
    bad = set(r.game_id[(nd <= 0).any(axis=1)])
    return set(last.game_id[ok.to_numpy()]) & set(contig.index[contig]) - bad


def round_table(b: Bundle) -> pd.DataFrame:
    """One row per round with the score-state and economy features known at its start, plus elo_logit and the
    market/e12 priors where present. Previous results come from the score, so this works without outcomes."""
    r = b.rounds.sort_values(["game_id", "round"]).copy()
    r = r.merge(b.elo[["game_id", "elo_logit"]], on="game_id", how="left")
    r = r.merge(b.priors, on="game_id", how="left")
    r["elo_logit"] = r.elo_logit.fillna(0.0)
    for t in ("a", "b"):
        # Pistol rounds are always $800 a player; PandaScore sometimes snapshots warmup money (16000) there,
        # and uses 99999 as a placeholder elsewhere.
        r.loc[r["round"].isin([1, 13]), f"{t}_money_pre"] = 4000
        r.loc[r[f"{t}_money_pre"] > 80000, f"{t}_money_pre"] = np.nan
    nxt = r.groupby("game_id").score_a.shift(-1)
    a_won_prev = (nxt - r.score_a).groupby(r.game_id).shift(1)
    la = np.zeros(len(r), int); lb = np.zeros(len(r), int)
    rounds, games, aw = r["round"].to_numpy(), r.game_id.to_numpy(), a_won_prev.to_numpy()
    for i in range(len(r)):
        if i == 0 or games[i] != games[i - 1] or resets(rounds[i]) or np.isnan(aw[i]):
            la[i] = lb[i] = 0
        else:
            x, y = loss_step(np.array(la[i - 1]), np.array(lb[i - 1]), np.array(aw[i]), np.array(rounds[i]))
            la[i], lb[i] = int(x), int(y)
    r["a_loss_ctr"], r["b_loss_ctr"] = la, lb
    r["ot"] = (r[["score_a", "score_b"]].min(axis=1) >= 12).astype(int)
    nd = np.array([need(a, c) for a, c in zip(r.score_a, r.score_b)])
    r["need_a"], r["need_b"] = nd[:, 0], nd[:, 1]
    r["ct_a"] = (r.side_a == "CT").astype(int)
    r["round_in_half"] = np.where(r.ot == 1, (r["round"] - 25) % 3, (r["round"] - 1) % 12)
    r["pistol"] = r["round"].isin([1, 13]).astype(int)
    for c in PRICE_KEYS:
        r[f"d_{c}"] = r[f"a_{c}"] - r[f"b_{c}"]
    r["map_code"] = r["map"].map(MAP_CODES).fillna(-1).astype(int)
    return r


MAP_CODES = {m: i for i, m in enumerate(["Ancient", "Anubis", "Cache", "Dust2", "Inferno", "Mirage", "Nuke", "Overpass", "Train", "Vertigo"])}


def map_states(b: Bundle, game_ids=None) -> pd.DataFrame:
    """The states the harness prices: for every round with a timeline and player data, its start (the round_start
    event) and the moment after each kill and bomb plant. event_seq is the last event a state may use."""
    ev = b.events if game_ids is None else b.events[b.events.game_id.isin(set(game_ids))]
    ok = b.rounds[b.rounds.has_players][["game_id", "round"]]
    ev = ev.merge(ok, on=["game_id", "round"])
    s = ev[ev.type.isin(["round_start", "kill", "bomb_planted"])].sort_values(["game_id", "seq"])
    s = s[s.groupby(["game_id", "round"]).type.transform(lambda t: (t == "round_start").any())]
    s = s[["game_id", "round", "seq", "type", "ingame_ms", "elapsed"]].rename(columns={"seq": "event_seq", "type": "kind"})
    s["kind"] = s.kind.replace({"round_start": "start"})
    s["step"] = s.groupby(["game_id", "round"]).cumcount()
    return s.reset_index(drop=True)


def state_features(b: Bundle, states: pd.DataFrame) -> pd.DataFrame:
    """CT-vs-T state after each event (round-start loadouts minus the dead): alive, equipment alive, AWPs alive, kits
    alive, planted, seconds elapsed and since the plant. Uses only events up to each state's event_seq."""
    P = b.players[b.players.game_id.isin(set(states.game_id))]
    roster = {k: list(zip(x.player_id, x.side, x.equip, x.prim == "awp", x.kit.fillna(False).astype(bool)))
              for k, x in P.groupby(["game_id", "round"])}
    ev = b.events[b.events.game_id.isin(set(states.game_id)) & b.events.type.isin(["kill", "bomb_planted"])]
    by_round = {k: list(zip(x.seq, x.type, x.victim_id, x.elapsed)) for k, x in ev.sort_values("seq").groupby(["game_id", "round"])}
    out = np.full((len(states), 9), np.nan)
    for i, (gid, rnd, seq) in enumerate(zip(states.game_id, states["round"], states.event_seq)):
        alive = {pid: v for pid, *v in roster.get((gid, rnd), [])}
        planted, plant_t, t = 0, np.nan, 0.0
        for s, typ, victim, el in by_round.get((gid, rnd), []):
            if s > seq:
                break
            t = el
            if typ == "kill":
                alive.pop(victim, None)
            else:
                planted, plant_t = 1, el
        v = {"CT": [0, 0, 0, 0], "T": [0, 0, 0, 0]}
        for side, eq, awp, kit in alive.values():
            x = v[side]; x[0] += 1; x[1] += eq; x[2] += awp; x[3] += kit
        out[i] = [*v["CT"], *v["T"][:3], planted, (t - plant_t) if planted else np.nan]
    f = pd.DataFrame(out, columns=["ct_alive", "ct_eq", "ct_awp", "ct_kits", "t_alive", "t_eq", "t_awp", "planted", "since_plant"], index=states.index)
    f["t"] = states.elapsed.to_numpy()
    return f
