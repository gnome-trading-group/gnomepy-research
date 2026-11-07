"""Point-in-time player and team strength.

Every map contributes per-player and per-team counts (rounds, kills, deaths, opening duels, clutches, AWP kills; rounds
won by side) stamped with the map's end time. A target map sees the cumulative counts of maps that ended strictly
before it began (merge_asof with allow_exact_matches=False), so nothing from the map itself or anything later leaks in.

Counts come from the event timeline where a map has one (it is never blanked, so maps of the scored month that already
ended still count), otherwise from the end-of-round player columns (blanked for the scored month, which then simply
contribute nothing)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import Bundle

PCOUNTS = ["n", "k", "d", "n_ev", "ok", "od", "ca", "cw", "awk", "awr", "mk"]
SIDE = {"counter_terrorists": "CT", "terrorists": "T"}


def _kills(b: Bundle, gids) -> pd.DataFrame:
    ev = b.events
    k = ev[ev.game_id.isin(gids) & (ev.type == "kill")][["game_id", "round", "seq", "killer_id", "victim_id", "killer_side", "victim_side", "weapon"]]
    k = k.sort_values(["game_id", "seq"])
    k["vside"] = k.victim_side.map(SIDE)
    k["kside"] = k.killer_side.map(SIDE)
    return k


def player_map_counts(b: Bundle, gids) -> pd.DataFrame:
    """One row per (game_id, player_id) with counts over that map's rounds that have player rows."""
    gids = set(gids)
    P = b.players[b.players.game_id.isin(gids)][["game_id", "round", "player_id", "team_id", "side", "prim", "alive_end", "kills"]]
    ev_games = set(b.events.game_id[b.events.game_id.isin(gids)].unique())
    base = P.assign(awr=(P.prim == "awp").astype(float))
    has_ev = base.game_id.isin(ev_games)

    k = _kills(b, ev_games)
    k = k.merge(P[["game_id", "round"]].drop_duplicates(), on=["game_id", "round"])
    enemy = k.kside.notna() & (k.kside != k.vside)
    kc = k[enemy].assign(k=1.0, awk=(k.weapon == "awp").astype(float)).groupby(["game_id", "round", "killer_id"])[["k", "awk"]].sum()
    kc.index.names = ["game_id", "round", "player_id"]
    dc = k.groupby(["game_id", "round", "victim_id"]).size().rename("d")
    dc.index.names = ["game_id", "round", "player_id"]
    first = k[enemy].groupby(["game_id", "round"]).head(1)
    okc = first.groupby(["game_id", "round", "killer_id"]).size().rename("ok"); okc.index.names = ["game_id", "round", "player_id"]
    odc = first.groupby(["game_id", "round", "victim_id"]).size().rename("od"); odc.index.names = ["game_id", "round", "player_id"]

    cl = clutches(b, ev_games, P, k)

    x = base.set_index(["game_id", "round", "player_id"])
    x = x.join(kc).join(dc).join(okc).join(odc).join(cl)
    x = x.reset_index()
    ev = has_ev.to_numpy()
    for c in ["k", "awk", "d", "ok", "od", "ca", "cw"]:
        x[c] = x[c].fillna(0.0)
    known_pt = x.alive_end.notna().to_numpy()
    x["k"] = np.where(ev, x.k, x.kills.astype(float))
    x["d"] = np.where(ev, x.d, np.where(known_pt, 1.0 - x.alive_end.astype(float), np.nan))
    x["mk"] = (x.k >= 3).astype(float)
    x["n"] = (ev | known_pt).astype(float)
    x["n_ev"] = ev.astype(float)
    for c in ["k", "d", "mk", "awk", "awr"]:
        x[c] = np.where(x.n > 0, x[c], 0.0)
    for c in ["ok", "od", "ca", "cw"]:
        x[c] = np.where(ev, x[c], 0.0)
    out = x.groupby(["game_id", "player_id"])[PCOUNTS].sum().reset_index()
    team = x.groupby(["game_id", "player_id"]).team_id.first().reset_index()
    return out.merge(team, on=["game_id", "player_id"])


def clutches(b: Bundle, ev_games, P: pd.DataFrame, k: pd.DataFrame) -> pd.DataFrame:
    """1vN situations: the first kill after which a side has one player left and the other side at least one. The
    clutcher is that side's only roster player not yet killed; cw=1 if their side won the round."""
    roster_n = P.groupby(["game_id", "round", "side"]).size().unstack(fill_value=0)
    kk = k.join(roster_n, on=["game_id", "round"])
    dead = pd.DataFrame({"CT": (kk.vside == "CT").astype(int), "T": (kk.vside == "T").astype(int)}).groupby([kk.game_id, kk["round"]]).cumsum()
    kk["ct_alive"] = kk.get("CT", 5) - dead.CT
    kk["t_alive"] = kk.get("T", 5) - dead["T"]
    ev = b.events
    win = ev[ev.game_id.isin(ev_games) & (ev.type == "round_end")][["game_id", "round", "winner_side"]]
    win = win.assign(wside=win.winner_side.map(SIDE)).drop_duplicates(["game_id", "round"]).set_index(["game_id", "round"]).wside
    rows = []
    for side, mine, other in (("CT", "ct_alive", "t_alive"), ("T", "t_alive", "ct_alive")):
        s = kk[(kk[mine] == 1) & (kk[other] >= 1)].groupby(["game_id", "round"]).head(1)[["game_id", "round", "seq"]]
        if s.empty:
            continue
        cand = P[P.side == side][["game_id", "round", "player_id"]].merge(s, on=["game_id", "round"])
        died = k[["game_id", "round", "victim_id", "seq"]].rename(columns={"victim_id": "player_id", "seq": "dseq"})
        cand = cand.merge(died.drop_duplicates(["game_id", "round", "player_id"]), on=["game_id", "round", "player_id"], how="left")
        cand = cand[cand.dseq.isna() | (cand.dseq > cand.seq)]
        w = win.reindex(pd.MultiIndex.from_frame(cand[["game_id", "round"]])).to_numpy()
        rows.append(cand.assign(ca=1.0, cw=(w == side).astype(float))[["game_id", "round", "player_id", "ca", "cw"]])
    if not rows:
        return pd.DataFrame(columns=["ca", "cw"], index=pd.MultiIndex.from_tuples([], names=["game_id", "round", "player_id"]))
    c = pd.concat(rows).groupby(["game_id", "round", "player_id"])[["ca", "cw"]].sum()
    return c


def team_map_counts(b: Bundle, gids) -> pd.DataFrame:
    """One row per (game_id, team_id): rounds played and won on CT and on T in that map."""
    gids = set(gids)
    r = b.rounds[b.rounds.game_id.isin(gids)][["game_id", "round", "map", "side_a", "a_wins"]]
    r = r.merge(b.games[["game_id", "team_a", "team_b"]], on="game_id")
    ev = b.events
    end = ev[ev.game_id.isin(gids) & (ev.type == "round_end")][["game_id", "round", "winner_team"]].drop_duplicates(["game_id", "round"])
    r = r.merge(end, on=["game_id", "round"], how="left")
    a_won = np.where(r.a_wins.notna(), r.a_wins, np.where(r.winner_team.isna(), np.nan, (r.winner_team == r.team_a).astype(float)))
    r = r.assign(a_won=a_won)[~np.isnan(a_won)]
    a_ct = (r.side_a == "CT").to_numpy()
    rows = []
    for team, won, ct in ((r.team_a, r.a_won, a_ct), (r.team_b, 1 - r.a_won, ~a_ct)):
        rows.append(pd.DataFrame(dict(game_id=r.game_id.to_numpy(), team_id=team.to_numpy(), map=r["map"].to_numpy(),
                                      ct_n=ct.astype(float), ct_w=np.where(ct, won, 0.0), t_n=(~ct).astype(float), t_w=np.where(~ct, won, 0.0))))
    x = pd.concat(rows)
    return x.groupby(["game_id", "team_id", "map"])[["ct_n", "ct_w", "t_n", "t_w"]].sum().reset_index()


def stamp(rows: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    return rows.merge(games[["game_id"]].assign(time=known_at(games)), on="game_id")


def known_at(games: pd.DataFrame) -> pd.Series:
    # A handful of maps have an end stamped a few seconds before their begin; never let a map count before it began.
    return games.end.where(games.end >= games.begin, games.begin)


def cumulative(rows: pd.DataFrame, key, cols, half_life_days=None) -> pd.DataFrame:
    """Running totals per key in time order. With a half life, totals are exponentially decayed: stored as weights
    relative to a fixed epoch so the as-of lookup can rescale them to the target time."""
    rows = rows.dropna(subset=["time"]).sort_values("time")
    v = rows[cols].to_numpy(float)
    if half_life_days:
        age = (rows.time - EPOCH).dt.total_seconds().to_numpy() / 86400.0
        v = v * np.power(2.0, age / half_life_days)[:, None]
    c = pd.DataFrame(v, columns=cols, index=rows.index).groupby([rows[k] for k in key]).cumsum()
    return pd.concat([rows[key + ["time"]], c], axis=1)


EPOCH = pd.Timestamp("2026-01-01", tz="UTC")


def asof(targets: pd.DataFrame, cum: pd.DataFrame, key, cols, half_life_days=None) -> pd.DataFrame:
    """Totals from rows strictly before each target's `time` (targets need key columns and `time`); a target without
    a time gets zeros."""
    t = targets.reset_index(drop=True).assign(_i=lambda d: np.arange(len(d)))
    t = t[t.time.notna()].sort_values("time")
    t["time"] = t.time.astype(cum.time.dtype) if len(cum) else t.time
    m = pd.merge_asof(t, cum.sort_values("time").rename(columns={"time": "time_c"}), left_on="time", right_on="time_c",
                      by=key, allow_exact_matches=False, direction="backward")
    m = m.set_index("_i").reindex(np.arange(len(targets)))
    v = m[cols].fillna(0.0).to_numpy(float)
    if half_life_days:
        age = (m.time - EPOCH).dt.total_seconds().to_numpy() / 86400.0
        v = np.nan_to_num(v * np.power(2.0, -age / half_life_days)[:, None])
    return pd.DataFrame(v, columns=cols, index=targets.index)


def map_results(b: Bundle, gids) -> pd.DataFrame:
    """Per finished map: first-round rosters of both teams and rounds won by each, from the rounds table where outcomes
    are present, else from the event timeline's round winners."""
    gids = set(gids)
    tc = team_map_counts(b, gids)
    tc = tc.assign(w=tc.ct_w + tc.t_w, n=tc.ct_n + tc.t_n)
    g = b.games[b.games.game_id.isin(gids)][["game_id", "begin", "end", "team_a", "team_b"]]
    g = g.assign(end=known_at(g))
    wa = tc.rename(columns={"team_id": "team_a", "w": "wa", "n": "na"})[["game_id", "team_a", "wa", "na"]]
    wb = tc.rename(columns={"team_id": "team_b", "w": "wb"})[["game_id", "team_b", "wb"]]
    g = g.merge(wa, on=["game_id", "team_a"], how="left").merge(wb, on=["game_id", "team_b"], how="left")
    P = b.players[b.players.game_id.isin(gids)][["game_id", "round", "player_id", "team_id"]]
    P = P[P["round"] == P.groupby("game_id")["round"].transform("min")]
    ros = P.groupby(["game_id", "team_id"]).player_id.agg(tuple)
    g["ra"] = ros.reindex(pd.MultiIndex.from_frame(g[["game_id", "team_a"]])).to_numpy()
    g["rb"] = ros.reindex(pd.MultiIndex.from_frame(g[["game_id", "team_b"]])).to_numpy()
    g["map"] = g.game_id.map(b.rounds[b.rounds.game_id.isin(gids)].groupby("game_id")["map"].first())
    return g


class PlayerElo:
    """Online player ratings on the logit scale: a team's strength is the mean of its five players' ratings (plus, with
    k_map, the mean of their ratings on this map), and after each map every player moves by k times the team's surprise
    in round share (rounds won minus expected). Updates happen at map end and lookups at map begin, so a map only sees
    maps that ended before it began."""

    def __init__(self, k=0.3, k_map=0.0, mode="rounds"):
        self.k, self.k_map, self.mode = k, k_map, mode
        self.r: dict = {}
        self.rm: dict = {}
        self.n: dict = {}
        self.nm: dict = {}

    def _team(self, ps, mp):
        r, rm = self.r, self.rm
        return np.mean([r.get(p, 0.0) for p in ps]), np.mean([rm.get((p, mp), 0.0) for p in ps])

    def run(self, results: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
        """results: map_results rows to apply; targets: map_results rows to rate. Returns per target each roster's mean
        rating, mean map rating and mean count of rated maps, as of the target's begin."""
        ups = results.dropna(subset=["end", "wa", "wb"])
        ups = ups[ups.ra.notna() & ups.rb.notna() & (ups.wa + ups.wb > 0)]
        # Integer nanoseconds, not Timestamps: a NaT among Timestamps breaks the sort's ordering silently. Targets
        # without a begin time are not rated.
        tu = ups.end.to_numpy("datetime64[ns]").astype(np.int64)
        tb = targets.begin.to_numpy("datetime64[ns]")
        has_t = ~np.isnat(tb)
        tq = tb.astype(np.int64)
        times = np.concatenate([tu, tq[has_t]])
        kinds = np.concatenate([np.ones(len(tu), int), np.zeros(int(has_t.sum()), int)])
        idx = np.concatenate([np.arange(len(tu)), np.flatnonzero(has_t)])
        order = np.lexsort((kinds, times))
        ev = list(zip(times[order], kinds[order], idx[order]))
        out = np.full((len(targets), 8), np.nan)
        ura, urb, uwa, uwb, ump = ups.ra.to_numpy(), ups.rb.to_numpy(), ups.wa.to_numpy(), ups.wb.to_numpy(), ups["map"].to_numpy()
        tra, trb, tmp = targets.ra.to_numpy(), targets.rb.to_numpy(), targets["map"].to_numpy()
        r, rm, n, nm = self.r, self.rm, self.n, self.nm
        for _, kind, i in ev:
            if kind == 0:
                a, bb = tra[i], trb[i]
                if isinstance(a, tuple) and isinstance(bb, tuple):
                    (ra, ma), (rb, mb) = self._team(a, tmp[i]), self._team(bb, tmp[i])
                    mp = tmp[i]
                    out[i] = [ra, rb, ma, mb, np.mean([n.get(p, 0) for p in a]), np.mean([n.get(p, 0) for p in bb]),
                              np.mean([(nm.get((p, mp), 0) + 1) / (n.get(p, 0) + 7) for p in a]),
                              np.mean([(nm.get((p, mp), 0) + 1) / (n.get(p, 0) + 7) for p in bb])]
                continue
            a, bb, mp = ura[i], urb[i], ump[i]
            (ra, ma), (rb, mb) = self._team(a, mp), self._team(bb, mp)
            e = 1.0 / (1.0 + np.exp(-(ra + ma - rb - mb)))
            tot = uwa[i] + uwb[i]
            if self.mode == "rounds":
                surprise = (uwa[i] / tot - e) * np.sqrt(tot / 22.0)
            else:
                surprise = float(uwa[i] > uwb[i]) - e
            s, sm = self.k * surprise, self.k_map * surprise
            for p in a + bb:
                nm[(p, mp)] = nm.get((p, mp), 0) + 1
            for p in a:
                r[p] = r.get(p, 0.0) + s; n[p] = n.get(p, 0) + 1
                if sm:
                    rm[(p, mp)] = rm.get((p, mp), 0.0) + sm
            for p in bb:
                r[p] = r.get(p, 0.0) - s; n[p] = n.get(p, 0) + 1
                if sm:
                    rm[(p, mp)] = rm.get((p, mp), 0.0) - sm
        return pd.DataFrame(out, columns=["pelo_a", "pelo_b", "pmap_a", "pmap_b", "pelo_na", "pelo_nb", "pshare_a", "pshare_b"], index=targets.index)

    def copy(self) -> "PlayerElo":
        c = PlayerElo(self.k, self.k_map, self.mode)
        c.r, c.rm, c.n, c.nm = dict(self.r), dict(self.rm), dict(self.n), dict(self.nm)
        return c
