"""
Pre-match bo3 series prices from player power ratings (HLTV + FACEIT), one snapshot per Monday cutoff.

    python -m gnomepy_research.sessions.cs2_comp.e13.predict --data <bundle> --targets <targets> --out <parquet> [--no-faceit]

One pass replays every map before the last cutoff in the order its result became known (data.py). At each Monday W it
prices every series of the coming week from the ratings and rosters known at W, before any map of that week is seen:
the targets whose week_cutoff is W, and (for calibration) every HLTV bo3 series of that week. A target's q is then a
logistic recalibration of its raw price, fitted on the HLTV series of the CAL_WEEKS weeks before W. Nothing depends on
other weeks' targets or on data at or after W, so a run on a bundle truncated at W reproduces the same q.

The five a team will field is the last five it played with before W (on HLTV, or on FACEIT when the FACEIT match is
linked to that HLTV team and is more recent), adjusted by HLTV roster moves dated after that map: departures are
removed and arrivals fill the gaps; a slot still empty is an unknown newcomer at the team's prior.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.hltv.player_skill import regulation_map_win
from gnomepy_research.sessions.cs2_comp.e13.data import FACEIT, Bundle, build
from gnomepy_research.sessions.cs2_comp.e13.rating import RatingConfig, RatingFilter

CAL_WEEKS = 26
CORE_MAPS = 1
RICH_RECAL = False
SIGMA_MAP = 0.15
SIGMA_SERIES = 0.10
EPOCH = pd.Timestamp("2020-01-01")
_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(15)
_GH_W = _GH_W / _GH_W.sum()
_GRID = np.linspace(-5.0, 5.0, 2001)


def _series_curve(sigma_map: float) -> np.ndarray:
    p = regulation_map_win(_GRID[:, None] + sigma_map * _GH_X[None, :]) @ _GH_W
    return p * p * (3 - 2 * p)


def series_price(d: float, s2: float, curve: np.ndarray, sigma_series: float) -> float:
    sd = math.sqrt(max(s2, 0.0) + sigma_series ** 2)
    return float(np.interp(d + sd * _GH_X, _GRID, curve) @ _GH_W)


class Rosters:
    """The five each team is expected to field at a cutoff.

    With core_maps > 1 the five is the team's regulars over its last core_maps maps (most appearances, then the most
    recent), not just its last five: a stand-in who filled one series does not displace the player he stood in for."""

    def __init__(self, moves: pd.DataFrame, core_maps: int = 1):
        self.core_maps = max(1, core_maps)
        self.hist: dict[int, collections.deque] = {}
        self.moves: dict[int, list[tuple[float, int, bool]]] = collections.defaultdict(list)
        for r in moves.itertuples():
            now = _days(r.event_date)
            if r.from_team_id and r.from_team_id == r.from_team_id:
                self.moves[int(r.from_team_id)].append((now, int(r.person_id), False))
            if r.to_team_id and r.to_team_id == r.to_team_id and r.to_status == "active":
                self.moves[int(r.to_team_id)].append((now, int(r.person_id), True))

    def played(self, team: int, now: float, five: tuple[int, ...]) -> None:
        h = self.hist.get(team)
        if h is None:
            h = self.hist[team] = collections.deque(maxlen=self.core_maps)
        h.append((now, five))

    def five(self, team: int, now: float, use_moves: bool) -> tuple[list[int], int]:
        h = self.hist.get(team)
        if not h:
            return [], 5
        since = h[-1][0]
        last_seen: dict[int, float] = {}
        count: collections.Counter = collections.Counter()
        for t, five in h:
            for p in five:
                count[p] += 1
                last_seen[p] = t
        ranked = sorted(count, key=lambda p: (-count[p], -last_seen[p], p))
        departed: set[int] = set()
        joined: list[int] = []
        if use_moves:
            for when, person, arrives in self.moves.get(team, ()):
                if when >= now:
                    continue
                if arrives:
                    if when >= since and person not in ranked[:5] and person not in joined:
                        joined.insert(0, person)
                    departed.discard(person)
                elif when >= last_seen.get(person, since):
                    departed.add(person)
                    if person in joined:
                        joined.remove(person)
        players = [p for p in ranked[:5] if p not in departed]
        for p in joined + ranked[5:]:
            if len(players) >= 5:
                break
            if p not in departed and p not in players:
                players.append(p)
        return players, 5 - len(players)


def _days(t) -> float:
    return (pd.Timestamp(t) - EPOCH).total_seconds() / 86400.0


def _fit_logistic(X: np.ndarray, y: np.ndarray, ridge: float = 1.0) -> np.ndarray:
    """Ridge logistic regression shrunk toward the identity map (slope 1 on the first column, 0 elsewhere)."""
    prior = np.zeros(X.shape[1])
    prior[0] = 1.0
    w = prior.copy()
    for _ in range(30):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (y - p) - ridge * (w - prior)
        h = X.T @ (X * (p * (1 - p))[:, None]) + ridge * np.eye(len(w))
        w = w + np.linalg.solve(h, g)
    return w


def recal_features(d: pd.DataFrame, rich: bool) -> np.ndarray:
    """The recalibration design: logit of the raw price, and with rich, its interactions with how little is known
    about the two fives (a new or incomplete five, thin history), so uncertain matchups are shrunk harder."""
    x = _logit(d.p_raw.to_numpy(float))
    cols = [x]
    if rich:
        new_five = ((np.minimum(d.five_a, d.five_b) < 3) | (d.unk_a + d.unk_b > 0)).to_numpy(float)
        thin = (d.games_min < 40).to_numpy(float)
        cols += [x * new_five, x * thin]
    return np.column_stack(cols + [np.ones(len(x))])


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def data_cutoff(targets: pd.DataFrame) -> pd.Timestamp:
    return pd.Timestamp(pd.to_datetime(targets.week_cutoff).max())


def run(b: Bundle, targets: pd.DataFrame, cfg: RatingConfig, use_moves: bool = True,
        sigma_map: float = SIGMA_MAP, sigma_series: float = SIGMA_SERIES, history_from: pd.Timestamp | None = None,
        actual_fives: bool = False, core_maps: int = CORE_MAPS, rich_recal: bool = RICH_RECAL) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (target prices, HLTV bo3 series priced at each cutoff from history_from on); b is built at
    data_cutoff(targets). actual_fives prices the history series with the fives that went on to play them, which
    only a diagnostic may do: it measures what predicting the fives costs."""
    cutoffs = sorted(pd.to_datetime(targets.week_cutoff).unique())
    last = pd.Timestamp(cutoffs[-1])
    first = pd.Timestamp(cutoffs[0]) - pd.Timedelta(weeks=CAL_WEEKS)
    if history_from is not None:
        first = min(first, pd.Timestamp(history_from))
    weeks = list(pd.date_range(first, last, freq="7D"))
    f = RatingFilter(cfg, b.rankings, b.faceit_level)
    rosters = Rosters(b.moves, core_maps)
    curve = _series_curve(sigma_map)
    series = b.series[(b.series.bo_type == 3) & b.series.y.notna()].copy()
    series["week"] = series.t.dt.normalize() - pd.to_timedelta(series.t.dt.dayofweek, unit="D")
    by_week = {w: g for w, g in series.groupby("week")}
    tg = targets.assign(week_cutoff=pd.to_datetime(targets.week_cutoff))
    t_by_week = {w: g for w, g in tg.groupby("week_cutoff")}

    played = b.maps[b.maps.src != FACEIT].drop_duplicates("match_id").set_index("match_id")[["players_a", "players_b"]] \
        if actual_fives else None

    def price(team_a: int, team_b: int, w: pd.Timestamp, now: float, match_id=None) -> dict:
        fa, ua = rosters.five(team_a, now, use_moves)
        fb, ub = rosters.five(team_b, now, use_moves)
        same = {}
        if played is not None and str(match_id) in played.index:
            pa, pb = played.loc[str(match_id)]
            same = {"same_a": len(set(pa) & set(fa)), "same_b": len(set(pb) & set(fb))}
            fa, fb, ua, ub = list(pa), list(pb), 0, 0
        la = f.lineup(tuple(fa), team_a, w, now, 0, -1, ua)
        lb = f.lineup(tuple(fb), team_b, w, now, 0, -1, ub)
        d, s2, _, _ = f.gap(la, lb)
        return {"p_raw": series_price(d, s2, curve, sigma_series), "d": d, "gap_sd": math.sqrt(max(s2, 0)),
                "unk_a": ua, "unk_b": ub, "fonly_a": la.faceit_only, "fonly_b": lb.faceit_only,
                "five_a": la.five_maps, "five_b": lb.five_maps,
                "games_min": min(la.games + lb.games) if la.games and lb.games else 0, **same}

    hist_rows, out_rows = [], []
    maps = b.maps
    t_arr = maps.t.to_numpy()
    i, n = 0, len(maps)
    cols = [maps[c].to_numpy() for c in ("src", "division", "team_a", "team_b", "players_a", "players_b", "rounds_a", "rounds_b")]
    for w in weeks:
        wn = np.datetime64(w)
        while i < n and t_arr[i] < wn:
            src, div, ta, tb, pa, pb, ra, rb = (c[i] for c in cols)
            now = _days(t_arr[i])
            day = pd.Timestamp(t_arr[i]).normalize()
            f.observe(day, now, int(src), int(div), int(ta), int(tb), pa, pb, int(ra), int(rb))
            for side, five in ((0, pa), (1, pb)):
                devs = [b.devs.get((i, side, int(p))) for p in five]
                if None not in devs:
                    f.observe_dev(five, devs, int(src))
            if src != FACEIT or ta > 0:
                rosters.played(int(ta), now, pa)
            if src != FACEIT or tb > 0:
                rosters.played(int(tb), now, pb)
            i += 1
        now = _days(w)
        for r in by_week.get(w, pd.DataFrame()).itertuples():
            hist_rows.append({"match_id": r.match_id, "week": w, "y": r.y, "is_lan": r.is_lan,
                              **price(int(r.team_a), int(r.team_b), w, now, r.match_id)})
        for r in t_by_week.get(w, pd.DataFrame()).itertuples():
            out_rows.append({"match_id": r.match_id, "fit_cutoff": w, **price(int(r.team_a_id), int(r.team_b_id), w, now)})
    hist = pd.DataFrame(hist_rows)
    out = pd.DataFrame(out_rows)
    hist["q"] = np.nan
    for w in weeks:
        cal = hist[(hist.week < w) & (hist.week >= w - pd.Timedelta(weeks=CAL_WEEKS))]
        if len(cal) < 50:
            continue
        coef = _fit_logistic(recal_features(cal, rich_recal), cal.y.to_numpy(float))
        sel = hist.week == w
        hist.loc[sel, "q"] = 1 / (1 + np.exp(-recal_features(hist[sel], rich_recal) @ coef))
        if len(out):
            sel = out.fit_cutoff == w
            out.loc[sel, "q"] = 1 / (1 + np.exp(-recal_features(out[sel], rich_recal) @ coef))
    return out, hist


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--targets", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-faceit", dest="faceit", action="store_false")
    ap.add_argument("--no-moves", dest="moves", action="store_false")
    ap.add_argument("--cfg", default="{}", help="JSON overrides of RatingConfig")
    ap.add_argument("--history-from", default=None, help="also price HLTV bo3 series from this Monday on (diagnostics)")
    ap.add_argument("--history-out", type=Path, default=None)
    a = ap.parse_args()
    cfg = replace(RatingConfig(), **json.loads(a.cfg))
    targets = pd.read_parquet(a.targets)
    out, hist = run(build(a.data, data_cutoff(targets), a.faceit), targets, cfg, a.moves,
                    history_from=pd.Timestamp(a.history_from) if a.history_from else None)
    missing = set(targets.match_id) - set(out.match_id)
    if missing:
        raise RuntimeError(f"{len(missing)} targets unpriced")
    out["q"] = out.q.fillna(out.p_raw).clip(0.01, 0.99)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out[["match_id", "q", "fit_cutoff"]].to_parquet(a.out, index=False)
    if a.history_out is not None:
        hist.to_parquet(a.history_out, index=False)
        out.to_parquet(a.history_out.with_name(a.history_out.stem + "_targets.parquet"), index=False)


if __name__ == "__main__":
    main()
