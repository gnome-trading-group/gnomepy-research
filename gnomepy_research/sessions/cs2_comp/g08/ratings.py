"""A recurrent player-skill state run over every decisive map in time order, read out at weekly cutoffs.

Each player carries a Gaussian belief (mean, variance) about their skill. A map is one observation of the gap between
the two fives' mean skills through its regulation rounds:

    rounds_A ~ Binomial(n, sigmoid(mean(s_A) - mean(s_B)))

and the beliefs of the ten players are updated by one extended-Kalman step, with the round likelihood tempered by
`round_weight` because rounds within a map are not independent. Between appearances a player's variance grows with
time (`drift_per_day`), so a team that changed or went quiet is believed less firmly. This is the sequence model: the
state after a team's latest maps summarises its whole history, and the update weights are set by the beliefs' own
uncertainty rather than a fixed K.

At every Monday 00:00 UTC cutoff the state is read out for the series of that week, using only maps known before the
cutoff, so every row (training series and target series alike) is point-in-time for its own week.

Alongside the skill state, each team keeps a short exponentially weighted record of how far its recent maps beat the
state's own expectation (form), the dates it played, and which fives it fielded, from which cohesion and activity are
read."""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.hltv.player_skill import regulation_map_win

FIVE = 5


@dataclass(frozen=True)
class RatingConfig:
    prior_var: float = 1.0
    drift_per_day: float = 0.0001
    round_weight: float = 0.35
    faceit_weight: float = 1.0
    new_offset: float = -0.5
    form_half_life_maps: float = 6.0
    # A team effect on top of its five (what stays with the organisation when players change) and a per-player LAN
    # offset (some players are better or worse on stage than online). Zero prior variance turns either off.
    team_var: float = 0.0
    team_drift_per_day: float = 0.0001
    lan_var: float = 0.0
    # A past series' kickoff price, mapped to the round-logit gap scale (logit(price) / market_scale), is a noisy
    # measurement of the two fives' gap with variance market_var. Zero market_var leaves prices out.
    market_var: float = 0.0
    market_scale: float = 4.0
    # Per (team, map) offsets: how much better a team is on one map than its five's skill says. A bo3 is then priced
    # through a simulated veto (each side removes its worst maps and picks its best) instead of one gap for all maps.
    map_var: float = 0.0
    map_drift_per_day: float = 0.0002
    # HLTV map ratings relative to teammates measure who in a five carries it:
    # rating_p - mean(rating_five) ~ N(rating_gamma * (s_p - mean(s_five)), rating_sd^2). Zero rating_gamma turns it off.
    rating_gamma: float = 0.0
    rating_sd: float = 0.3
    # FACEIT maps carry a rating proxy fitted on HLTV maps (kills, deaths and damage per round); zero leaves it out.
    faceit_rating_sd: float = 0.0


def regulation(score_a: np.ndarray, score_b: np.ndarray, overtime: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """An overtime map counts as 12-12 of 24 regulation rounds."""
    ot = overtime | (score_a + score_b > 24)
    return np.where(ot, 12, score_a), np.where(ot, 24, score_a + score_b)


class SkillState:
    """Gaussian beliefs keyed by (kind, id): "p" player skill, "t" team effect, "l" player LAN offset."""

    def __init__(self, cfg: RatingConfig):
        self.cfg = cfg
        self.mu: dict[tuple, float] = {}
        self.var: dict[tuple, float] = {}
        self.seen: dict[tuple, float] = {}
        self.maps: dict[int, int] = defaultdict(int)
        self.team_form: dict[int, float] = defaultdict(float)
        self.team_days: dict[int, deque] = defaultdict(lambda: deque(maxlen=400))
        self.team_fives: dict[int, deque] = defaultdict(lambda: deque(maxlen=60))

    def _prior(self, key: tuple, mates: list[int]) -> tuple[float, float]:
        cfg = self.cfg
        if key[0] == "t":
            return 0.0, cfg.team_var
        if key[0] == "l":
            return 0.0, cfg.lan_var
        if key[0] == "m":
            return 0.0, cfg.map_var
        known = [self.mu[("p", m)] for m in mates if ("p", m) in self.mu]
        if len(known) >= 2:
            return float(np.mean(known)) + cfg.new_offset, cfg.prior_var
        # Ratings are zero-sum, so a wholly new five entering at zero sits at the population's mean.
        return 0.0, cfg.prior_var

    def belief(self, key: tuple, day: float, mates: list[int]) -> tuple[float, float, bool]:
        cfg = self.cfg
        if key in self.mu:
            if key[0] == "t":
                drift, cap = cfg.team_drift_per_day, cfg.team_var
            elif key[0] == "l":
                drift, cap = 0.0, cfg.lan_var
            elif key[0] == "m":
                drift, cap = cfg.map_drift_per_day, cfg.map_var
            else:
                drift, cap = cfg.drift_per_day, cfg.prior_var * 2
            v = self.var[key] + drift * max(day - self.seen[key], 0.0)
            return self.mu[key], min(v, cap), True
        m, v = self._prior(key, mates)
        return m, v, False

    def _entries(self, five: list[int], team: int, lan: int, map_name: str | None = None) -> list[tuple[tuple, float]]:
        e = [(("p", p), 1.0 / FIVE) for p in five]
        if self.cfg.map_var > 0 and team >= 0 and map_name:
            e.append((("m", team, map_name), 1.0))
        if self.cfg.team_var > 0 and team >= 0:
            e.append((("t", team), 1.0))
        if self.cfg.lan_var > 0 and lan:
            e += [(("l", p), 1.0 / FIVE) for p in five]
        return e

    def update(self, five_a: list[int], five_b: list[int], rounds_a: float, rounds: float, day: float, weight: float,
               team_a: int, team_b: int, lan: int, map_name: str | None = None) -> None:
        cfg = self.cfg
        ea = self._entries(five_a, team_a, lan, map_name)
        eb = [(k, -c) for k, c in self._entries(five_b, team_b, lan, map_name)]
        entries = ea + eb
        bel = [self.belief(k, day, five_a if c > 0 else five_b) for k, c in entries]
        d = sum(c * b[0] for (_, c), b in zip(entries, bel))
        s = sum(c * c * b[1] for (_, c), b in zip(entries, bel))
        pr = 1.0 / (1.0 + np.exp(-d))
        w = cfg.round_weight * weight
        g = w * (rounds_a - rounds * pr)
        info = w * rounds * pr * (1 - pr)
        denom = 1.0 + info * s
        for (k, c), (m, v, _) in zip(entries, bel):
            self.mu[k] = m + v * c * g / denom
            self.var[k] = max(v - (v * c) ** 2 * info / denom, 1e-5)
            self.seen[k] = day
        for p in five_a + five_b:
            self.maps[p] += 1
        if rounds > 0:
            share = (rounds_a + 0.5) / (rounds + 1.0)
            resid = np.log(share / (1 - share)) - d
            a = 1.0 - 0.5 ** (1.0 / cfg.form_half_life_maps)
            for team, r in ((team_a, resid), (team_b, -resid)):
                if team >= 0:
                    self.team_form[team] = (1 - a) * self.team_form[team] + a * r * weight
        for team, five in ((team_a, five_a), (team_b, five_b)):
            if team >= 0:
                self.team_days[team].append(day)
                self.team_fives[team].append((day, frozenset(five)))

    def observe_ratings(self, five: list[int], ratings, day: float, sd: float) -> None:
        """Sequential scalar Kalman updates, one per player, of the rating-deviation measurement."""
        cfg = self.cfg
        dev = np.asarray(ratings, float) - float(np.mean(ratings))
        keys = [("p", p) for p in five]
        for j in range(FIVE):
            c = np.full(FIVE, -1.0 / FIVE)
            c[j] += 1.0
            c *= cfg.rating_gamma
            bel = [self.belief(k, day, five) for k in keys]
            pred = sum(cj * b[0] for cj, b in zip(c, bel))
            S = sum(cj * cj * b[1] for cj, b in zip(c, bel)) + sd ** 2
            # Without cross-covariances the update would also move the five's mean; ratings only say who in the
            # five carries it, so the shift is taken back out.
            delta = np.array([v * cj * (dev[j] - pred) / S for cj, (_, v, _) in zip(c, bel)])
            delta -= delta.mean()
            # A diagonal state cannot hold the within-five contrast these measurements pin down; shrinking each
            # player's variance would freeze the five's mean too, so only the means move.
            for k, dl, (m, v, _) in zip(keys, delta, bel):
                self.mu[k] = m + dl
                self.var[k] = v
                self.seen[k] = day

    def observe_gap(self, five_a: list[int], five_b: list[int], z: float, day: float, team_a: int, team_b: int,
                    lan: int) -> None:
        """A linear-Gaussian measurement z ~ N(gap, market_var) of the two fives' gap."""
        entries = self._entries(five_a, team_a, lan) + [(k, -c) for k, c in self._entries(five_b, team_b, lan)]
        bel = [self.belief(k, day, five_a if c > 0 else five_b) for k, c in entries]
        d = sum(c * b[0] for (_, c), b in zip(entries, bel))
        s = sum(c * c * b[1] for (_, c), b in zip(entries, bel)) + self.cfg.market_var
        for (k, c), (m, v, _) in zip(entries, bel):
            self.mu[k] = m + v * c * (z - d) / s
            self.var[k] = max(v - (v * c) ** 2 / s, 1e-5)
            self.seen[k] = day

    def map_offsets(self, team: int, pool: list[str], day: float) -> np.ndarray:
        return np.array([self.belief(("m", team, m), day, [])[0] for m in pool])

    def five_features(self, five: list[int] | None, team: int, day: float) -> dict:
        if not five:
            five = []
            last = self.team_fives.get(team)
            if last:
                five = list(last[-1][1])
        bel = [self.belief(("p", p), day, five) for p in five]
        team_mu = self.belief(("t", team), day, five)[0] if self.cfg.team_var > 0 else 0.0
        if not bel:
            return {"mu": 0.0, "var": self.cfg.prior_var, "unknown": FIVE, "lowmaps": FIVE, "cohesion": 0.0,
                    "maps90": 0, "idle": 365.0, "form": 0.0, "sd_mu": 0.0, "teff": team_mu, "lanoff": 0.0}
        mus = np.array([b[0] for b in bel])
        lan_mu = float(np.mean([self.belief(("l", p), day, five)[0] for p in five])) if self.cfg.lan_var > 0 else 0.0
        days = np.array(self.team_days.get(team, []))
        fives = self.team_fives.get(team, [])
        fset = set(five)
        recent = [f for dd, f in fives if day - dd <= 90]
        coh = float(np.mean([len(f & fset) >= 4 for f in recent])) if recent else 0.0
        return {"mu": float(mus.mean()), "var": float(np.mean([b[1] for b in bel])),
                "unknown": int(sum(not b[2] for b in bel)), "lowmaps": int(sum(self.maps.get(p, 0) < 10 for p in five)),
                "cohesion": coh, "maps90": int(((day - days) <= 90).sum()) if len(days) else 0,
                "idle": float(day - days[-1]) if len(days) else 365.0, "form": float(self.team_form.get(team, 0.0)),
                "sd_mu": float(mus.std()), "teff": float(team_mu), "lanoff": lan_mu}


GH_X, GH_W = np.polynomial.hermite_e.hermegauss(7)
GH_W = GH_W / GH_W.sum()
POOL_DAYS = 90
POOL_MIN_SHARE = 0.05


def active_pool(recent: list[str]) -> list[str]:
    """The maps that made up at least POOL_MIN_SHARE of HLTV maps over the last POOL_DAYS."""
    if not recent:
        return []
    names, counts = np.unique(np.array(recent, dtype=object).astype(str), return_counts=True)
    return sorted(names[counts >= POOL_MIN_SHARE * len(recent)].tolist())


def bo3_price(p: np.ndarray) -> float:
    return float(p[0] * p[0] * (3 - 2 * p[0]))


def veto_price(p: np.ndarray) -> float:
    """P(A wins a bo3) when A bans its worst map, B bans A's best, A picks its best, B picks A's worst, the two
    remaining bans follow the same rule and the last map left is the decider."""
    left = list(range(len(p)))
    if len(left) < 3:
        return bo3_price(np.full(1, float(np.mean(p))))
    order = []
    for step in ("ban_a", "ban_b", "pick_a", "pick_b", "ban_a", "ban_b"):
        if len(left) <= 1:
            break
        i = min(left, key=lambda k: p[k]) if step in ("ban_a", "pick_b") else max(left, key=lambda k: p[k])
        left.remove(i)
        if step.startswith("pick"):
            order.append(i)
    p1, p2 = p[order[0]], p[order[1]]
    p3 = float(np.mean(p[left])) if left else 0.5 * (p1 + p2)
    return float(p1 * p2 + p1 * (1 - p2) * p3 + (1 - p1) * p2 * p3)


def _days(ts: pd.Series | np.ndarray) -> np.ndarray:
    return (pd.to_datetime(ts).to_numpy().astype("datetime64[s]").astype(np.int64)) / 86400.0


def run(maps: pd.DataFrame, queries: pd.DataFrame, cfg: RatingConfig) -> pd.DataFrame:
    """maps: match_id, known, source, five_a, five_b, rounds_a, rounds, team_a, team_b, is_lan (any order); rows with
    source "market" carry a past series' kickoff price p_a instead of rounds and are measurements of the gap.
    queries: qid, cutoff, team_a, team_b, five_a, five_b (fives may be None).
    Returns one feature row per query, read from the state holding every map known before the query's cutoff."""
    if not cfg.market_var:
        maps = maps[maps.source != "market"]
    # A series' price was known before its own maps were played.
    order = (maps.source != "market").astype(int)
    maps = maps.assign(_o=order).sort_values(["known", "_o", "match_id"], kind="stable").reset_index(drop=True)
    is_mkt = (maps.source == "market").to_numpy()
    rt_a = maps.ratings_a.to_numpy() if "ratings_a" in maps and cfg.rating_gamma else np.full(len(maps), None)
    rt_b = maps.ratings_b.to_numpy() if "ratings_b" in maps and cfg.rating_gamma else np.full(len(maps), None)
    z = (np.log(np.clip(maps.p_a.to_numpy(float), 0.02, 0.98)) - np.log(1 - np.clip(maps.p_a.to_numpy(float), 0.02, 0.98))) / cfg.market_scale \
        if "p_a" in maps else np.zeros(len(maps))
    mk = maps.known.to_numpy()
    mday = _days(maps.known)
    fa, fb = maps.five_a.to_numpy(), maps.five_b.to_numpy()
    ra, n = maps.rounds_a.to_numpy(float), maps.rounds.to_numpy(float)
    wt = np.where(maps.source.to_numpy() == "faceit", cfg.faceit_weight, 1.0)
    ta, tb = maps.team_a.to_numpy(np.int64), maps.team_b.to_numpy(np.int64)
    lan = maps.is_lan.to_numpy(int)
    mname = maps.map_name.to_numpy() if "map_name" in maps else np.full(len(maps), None)
    hltv = (maps.source == "hltv").to_numpy()
    state = SkillState(cfg)
    recent_maps: deque = deque()
    out = []
    i = 0
    for cutoff, qs in queries.groupby("cutoff", sort=True):
        c64 = np.datetime64(pd.Timestamp(cutoff))
        stop = np.searchsorted(mk, c64, side="left")
        for j in range(i, stop):
            if is_mkt[j]:
                state.observe_gap(list(fa[j]), list(fb[j]), z[j], mday[j], ta[j], tb[j], lan[j])
                continue
            state.update(list(fa[j]), list(fb[j]), ra[j], n[j], mday[j], wt[j], ta[j], tb[j], lan[j], mname[j])
            sd = cfg.rating_sd if hltv[j] else cfg.faceit_rating_sd
            if sd > 0:
                for five, rt in ((fa[j], rt_a[j]), (fb[j], rt_b[j])):
                    if rt is not None and not np.isnan(rt).any():
                        state.observe_ratings(list(five), rt, mday[j], sd)
            if hltv[j]:
                recent_maps.append((mday[j], mname[j]))
        i = stop
        cday = _days(np.array([c64]))[0]
        while recent_maps and cday - recent_maps[0][0] > POOL_DAYS:
            recent_maps.popleft()
        pool = active_pool([m for _, m in recent_maps])
        for q in qs.itertuples(index=False):
            r = {"qid": q.qid}
            for side, team, five in (("a", q.team_a, q.five_a), ("b", q.team_b, q.five_b)):
                f = state.five_features(None if five is None or len(five) == 0 else [int(x) for x in five], int(team), cday)
                r.update({f"{k}_{side}": v for k, v in f.items()})
            gap = r["mu_a"] - r["mu_b"]
            r["series_flat"] = bo3_price(np.full(1, regulation_map_win(np.array([gap]))[0]))
            if cfg.map_var > 0 and pool:
                off = state.map_offsets(int(q.team_a), pool, cday) - state.map_offsets(int(q.team_b), pool, cday)
            else:
                off = np.zeros(1)
            r["series_veto"] = veto_price(regulation_map_win(gap + off)) if len(off) >= 3 else r["series_flat"]
            # The same price averaged over the state's own uncertainty about the gap: unsure gaps pull toward 0.5.
            sd_gap = np.sqrt((r["var_a"] + r["var_b"]) / FIVE)
            r["series_unc"] = float(sum(w * (veto_price(regulation_map_win(gap + sd_gap * x + off)) if len(off) >= 3
                                             else bo3_price(regulation_map_win(np.array([gap + sd_gap * x]))))
                                        for x, w in zip(GH_X, GH_W)))
            out.append(r)
    return pd.DataFrame(out)
