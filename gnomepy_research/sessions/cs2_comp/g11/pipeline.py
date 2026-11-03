"""
Per-week feature rows: every series with a Polymarket path is priced by the graph fit at its own week cutoff, so the
market layer can be refitted each week on series that were priced exactly as a target is.

A series of a past week is always priced from the HLTV history (its announced fives there), never from a targets file,
so the training rows of a week do not depend on which other target weeks are in the run.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.bundle import load, target_paths
from gnomepy_research.sessions.cs2_comp.g11.e12.data import FIVE, Inputs
from gnomepy_research.sessions.cs2_comp.g11.e12.model import Lineups, team_state
from gnomepy_research.sessions.cs2_comp.g11.graph import Config, GraphFit

WEEK = pd.Timedelta(days=7)
KICKOFF_WINDOW = pd.Timedelta(minutes=30)


def _naive(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    return s.dt.tz_convert(None) if s.dt.tz is not None else s


def monday(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts).normalize()
    return ts - pd.Timedelta(days=ts.weekday())


def market_summary(paths: pd.DataFrame) -> pd.DataFrame:
    """Per series: kickoff, m0 (last print within 30 minutes before kickoff), the last print at all, and its age."""
    p = paths.assign(kickoff=_naive(paths.kickoff), t=_naive(paths.t))
    p = p[p.t <= p.kickoff].sort_values(["match_id", "t"], kind="stable")
    last = p.groupby("match_id").tail(1).set_index("match_id")
    out = pd.DataFrame({"kickoff": last.kickoff, "p_last": last.p_a,
                        "last_age_min": (last.kickoff - last.t).dt.total_seconds() / 60.0})
    out["m0"] = np.where(out.last_age_min <= KICKOFF_WINDOW.total_seconds() / 60.0, out.p_last, np.nan)
    out["n_prints"] = p.groupby("match_id").size()
    return out.reset_index()


def _announced(five) -> tuple[int, ...] | None:
    if five is None or len(five) != FIVE or len(set(five)) != FIVE:
        return None
    return tuple(sorted(int(p) for p in five))


class Standins:
    """Share of a team's last five HLTV series played with a stand-in, as known before a cutoff."""

    def __init__(self, inputs: Inputs):
        s = inputs.standins.sort_values("known", kind="stable")
        self.by_team = {int(t): (g.known.to_numpy(), (g.standins > 0).to_numpy(float)) for t, g in s.groupby("team_id")}

    def share(self, team: int, cutoff: pd.Timestamp, last_n: int = 5) -> float:
        h = self.by_team.get(int(team))
        if h is None:
            return 0.0
        i = np.searchsorted(h[0], np.datetime64(cutoff), side="left")
        return float(h[1][max(0, i - last_n):i].mean()) if i else 0.0


def price_rows(inputs: Inputs, fit: GraphFit, lineups: Lineups, standins: Standins, series: pd.DataFrame,
               cutoff: pd.Timestamp) -> pd.DataFrame:
    """series: match_id, team_a, team_b, lineup_a, lineup_b, sub_a, sub_b (+ anything to carry along)."""
    rows = lineups.at(cutoff)
    by_team = {int(k): g for k, g in rows.groupby("team")}
    empty = rows.iloc[:0]
    out = []
    for r in series.itertuples(index=False):
        rec = r._asdict()
        fives, ranks = {}, {}
        for side, team in (("a", int(r.team_a)), ("b", int(r.team_b))):
            st = team_state(by_team.get(team, empty), team, cutoff)
            five = _announced(rec.pop(f"lineup_{side}"))
            if five is None:
                five, new = st["five"], 0
            else:
                new = FIVE - len(set(five) & set(st["five"] or ()))
            fives[side], ranks[side] = five, inputs.rank(team, cutoff)
            rec.update({f"new_{side}": new, f"idle_{side}": st["days_idle"], f"rank_{side}": ranks[side],
                        f"standin_{side}": standins.share(team, cutoff)})
        rec.update(fit.gap(fives["a"], ranks["a"], fives["b"], ranks["b"], int(r.team_a), int(r.team_b)))
        rec["week_cutoff"] = cutoff
        out.append(rec)
    return pd.DataFrame(out)


def build(data_dir: str, targets: pd.DataFrame, cfg: Config, faceit: bool, history_weeks: int,
          verbose: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (train rows: past market series priced at their own week, target rows)."""
    until = pd.Timestamp(targets.week_cutoff.max())
    inputs = Inputs(data_dir, until, faceit)
    lineups = Lineups(inputs, faceit)
    standins = Standins(inputs)
    paths = load(data_dir, "market/series_paths", until)
    mk = market_summary(paths)
    mk["week"] = mk.kickoff.map(monday)
    ser = inputs.series.rename(columns={"time": "series_time"})
    ser["match_id"] = ser.match_id.astype(np.int64)
    hist = mk.merge(ser[["match_id", "team_a", "team_b", "lineup_a", "lineup_b", "sub_a", "sub_b", "y", "known", "is_lan"]],
                    on="match_id")
    hist = hist[hist.kickoff < until]
    tm = market_summary(target_paths(data_dir, targets.match_id))
    t = targets.rename(columns={"team_a_id": "team_a", "team_b_id": "team_b", "team_a_lineup_ids": "lineup_a",
                                "team_b_lineup_ids": "lineup_b"})
    t["sub_a"] = [0 if v is None else len(v) for v in targets.team_a_standin_ids]
    t["sub_b"] = [0 if v is None else len(v) for v in targets.team_b_standin_ids]
    t = t[["match_id", "team_a", "team_b", "lineup_a", "lineup_b", "sub_a", "sub_b", "is_lan", "week_cutoff"]].merge(
        tm.drop(columns=["kickoff"]), on="match_id", how="left")
    target_weeks = sorted(pd.Timestamp(c) for c in t.week_cutoff.unique())
    weeks = set(target_weeks)
    for c in target_weeks:
        w = c - history_weeks * WEEK
        while w < c:
            weeks.add(w)
            w += WEEK
    train, tgt = [], []
    for w in sorted(weeks):
        hw = hist[hist.week == w]
        tw = t[t.week_cutoff == w]
        if hw.empty and tw.empty:
            continue
        t0 = time.time()
        fit = GraphFit.fit(inputs, w, cfg, faceit)
        if not hw.empty:
            train.append(price_rows(inputs, fit, lineups, standins, hw.drop(columns=["week"]), w))
        if not tw.empty:
            tgt.append(price_rows(inputs, fit, lineups, standins, tw.drop(columns=["week_cutoff"]), w))
        if verbose:
            print(f"{w:%Y-%m-%d} fit {time.time() - t0:.1f}s maps {fit.info['maps']} players {fit.info['players']} "
                  f"gnorm {fit.info['gnorm']:.1e} hist {len(hw)} targets {len(tw)}", flush=True)
    return (pd.concat(train, ignore_index=True) if train else pd.DataFrame(),
            pd.concat(tgt, ignore_index=True) if tgt else pd.DataFrame())
