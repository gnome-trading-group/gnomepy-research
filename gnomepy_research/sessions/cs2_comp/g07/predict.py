"""g07 predictions: P(team A wins the bo3) for every target series, from data before its week_cutoff only.

    python -m gnomepy_research.sessions.cs2_comp.g07.predict --data <bundle> --targets <targets parquet> --out <parquet>
        [--weeks 2026-04-06,2026-04-13] [--rows <parquet>]

One pass of the skill filter (`model.py`) runs over every map in the order the maps became known. At each Monday from
the first priced series on, the filter stops at that Monday and prices (a) every market series kicking off that week,
with the fives HLTV lists for it, and (b) the targets whose week_cutoff is that Monday, with their announced fives.

The final price is a Bayesian logistic layer (`layer.py`) over the market's kickoff price, the shape of its path, and
the filter's price with its uncertainty. For a target week it is fitted on the market series that kicked off before the
cutoff and whose result was known by then, each priced at its own Monday, so the layer is point-in-time too."""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.bundle import load, target_paths
from gnomepy_research.sessions.cs2_comp.g07.data import FIVE, Inputs, naive
from gnomepy_research.sessions.cs2_comp.g07.layer import fit_and_price
from gnomepy_research.sessions.cs2_comp.g07.market import path_features
from gnomepy_research.sessions.cs2_comp.g07.model import Config, Filter, bo3_price

EPOCH = np.datetime64("2023-01-01")
CONFIG = Config()
WEEK = pd.Timedelta(days=7)


def _five(v) -> tuple[int, ...] | None:
    if v is None or len(v) != FIVE:
        return None
    ids = tuple(int(x) for x in v)
    return ids if len(set(ids)) == FIVE else None


def _days(t) -> float:
    return float((np.datetime64(pd.Timestamp(t)) - EPOCH) / np.timedelta64(1, "D"))


def _monday(t: pd.Timestamp) -> pd.Timestamp:
    t = pd.Timestamp(t).normalize()
    return t - pd.Timedelta(days=t.weekday())


def price_rows(f: Filter, rows: pd.DataFrame, last_five: dict) -> pd.DataFrame:
    """rows: match_id, time, team_a, team_b, five_a, five_b. Adds the filter's view of each side."""
    out = []
    for r in rows.itertuples(index=False):
        now = _days(r.time)
        rec = {"match_id": r.match_id}
        for side, team, five in (("a", int(r.team_a), r.five_a), ("b", int(r.team_b), r.five_b)):
            if five is None:
                five = last_five.get(team)
            if five is None:
                rec.update({f"mu_{side}": np.nan, f"var_{side}": np.nan, f"unk_{side}": FIVE, f"n_{side}": 0.0,
                            f"new_{side}": FIVE})
                continue
            m, v, unk, n = f.side(five, team, now)
            prev = last_five.get(team)
            new = FIVE - len(set(five) & set(prev)) if prev is not None else FIVE
            rec.update({f"mu_{side}": m, f"var_{side}": v, f"unk_{side}": unk, f"n_{side}": n, f"new_{side}": new})
        out.append(rec)
    out = pd.DataFrame(out)
    reg = [f.region_gap(int(a), int(b), _days(t)) for a, b, t in zip(rows.team_a, rows.team_b, rows.time)]
    out["rgap"] = [r[0] for r in reg]
    out["rvar"] = [r[1] for r in reg]
    ok = out.mu_a.notna() & out.mu_b.notna()
    out["gap"] = (out.mu_a - out.mu_b + out.rgap).where(ok, 0.0)
    out["gvar"] = (out.var_a + out.var_b + out.rvar).where(ok, 1.0)
    out["q_skill"] = bo3_price(out.gap.to_numpy(), out.gvar.to_numpy())
    out["has_skill"] = ok.astype(float)
    return out


class LastFive:
    """The five each HLTV team fielded on its latest map known to the filter."""

    def __init__(self, maps: pd.DataFrame):
        h = maps[maps.src == 0]
        self.known = h.known.to_numpy()
        self.team_a, self.team_b = h.team_a.to_numpy(), h.team_b.to_numpy()
        self.five_a, self.five_b = h.five_a.to_numpy(), h.five_b.to_numpy()
        self.cursor = 0
        self.last: dict[int, tuple] = {}

    def advance(self, cutoff: np.datetime64) -> dict:
        stop = int(np.searchsorted(self.known, cutoff, side="left"))
        for i in range(self.cursor, stop):
            self.last[int(self.team_a[i])] = self.five_a[i]
            self.last[int(self.team_b[i])] = self.five_b[i]
        self.cursor = max(self.cursor, stop)
        return self.last


def weekly_rows(data_dir: str | Path, targets: pd.DataFrame, cfg: Config = CONFIG) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter prices for every market series (training rows, each at its own Monday) and every target."""
    until = targets.week_cutoff.max()
    inputs = Inputs(data_dir, until)
    paths = load(data_dir, "market/series_paths", until)
    paths = paths.assign(kickoff=naive(paths.kickoff))
    kick = paths.groupby("match_id").kickoff.first()
    kick = kick[kick < until]
    s = inputs.series.assign(match_id=inputs.series.match_id.astype(np.int64))
    s = s[s.match_id.isin(kick.index)].copy()
    s["kickoff"] = s.match_id.map(kick)
    s["week"] = [_monday(k) for k in s.kickoff]
    s["five_a"] = [_five(v) for v in s.lineup_a]
    s["five_b"] = [_five(v) for v in s.lineup_b]
    s["sub_a"] = [inputs.standins.get((int(m), int(tm)), 0) for m, tm in zip(s.match_id, s.team_a)]
    s["sub_b"] = [inputs.standins.get((int(m), int(tm)), 0) for m, tm in zip(s.match_id, s.team_b)]
    s = s.rename(columns={"kickoff": "time_k"})
    t = targets.copy()
    t["five_a"] = [_five(v) for v in t.team_a_lineup_ids]
    t["five_b"] = [_five(v) for v in t.team_b_lineup_ids]
    for side in ("a", "b"):
        col = f"team_{side}_standin_ids"
        t[f"sub_{side}"] = [0 if v is None else len(v) for v in t[col]] if col in t else 0
    t = t.rename(columns={"team_a_id": "team_a", "team_b_id": "team_b"})
    t["time_k"] = naive(t.match_time)
    first = min(s.week.min() if len(s) else until, t.week_cutoff.min())
    weeks = pd.date_range(first, until, freq="7D")
    f = Filter(cfg)
    lf = LastFive(inputs.maps)
    train, pred = [], []
    for w in weeks:
        cut = np.datetime64(w)
        f.advance(inputs.maps, cut, EPOCH)
        last = lf.advance(cut)
        sw = s[s.week == w]
        if len(sw):
            r = price_rows(f, sw.rename(columns={"time_k": "time"})[["match_id", "time", "team_a", "team_b", "five_a", "five_b"]], last)
            r = r.merge(sw[["match_id", "time_k", "known", "y", "is_lan", "tier", "sub_a", "sub_b"]], on="match_id")
            r["week"] = w
            train.append(r)
        tw = t[t.week_cutoff == w]
        if len(tw):
            r = price_rows(f, tw.rename(columns={"time_k": "time"})[["match_id", "time", "team_a", "team_b", "five_a", "five_b"]], last)
            r = r.merge(tw[["match_id", "time_k", "week_cutoff", "is_lan", "event_tier", "sub_a", "sub_b"]].rename(columns={"event_tier": "tier"}),
                        on="match_id")
            pred.append(r)
    train = pd.concat(train, ignore_index=True) if train else pd.DataFrame()
    pred = pd.concat(pred, ignore_index=True)
    if len(train):
        train = train.merge(path_features(paths[paths.match_id.isin(train.match_id)]), on="match_id", how="left")
    pred = pred.merge(path_features(target_paths(data_dir, pred.match_id)), on="match_id", how="left")
    return train, pred


def predict(data_dir: str | Path, targets: pd.DataFrame, cfg: Config = CONFIG) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train, pred = weekly_rows(data_dir, targets, cfg)
    out = []
    for w, g in pred.groupby("week_cutoff"):
        tr = train[(train.time_k < w) & (train.known < w)] if len(train) else train
        out.append(g.assign(q=fit_and_price(tr, g)))
    out = pd.concat(out, ignore_index=True)
    res = pd.DataFrame({"match_id": out.match_id.to_numpy(), "q": out.q.to_numpy(),
                        "fit_cutoff": pd.to_datetime(out.week_cutoff).to_numpy()})
    return res, train, out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weeks", help="comma-separated week_cutoff dates to predict (default: all)")
    ap.add_argument("--rows", help="also save the training and target feature rows with this path prefix")
    ap.add_argument("--config", help="filter overrides for experiments, k=v,...")
    a = ap.parse_args()
    targets = pd.read_parquet(a.targets)
    targets["week_cutoff"] = naive(targets.week_cutoff)
    if a.weeks:
        targets = targets[targets.week_cutoff.isin(pd.to_datetime(a.weeks.split(",")))]
    cfg = dataclasses.replace(CONFIG, **{k: float(v) for k, v in (p.split("=") for p in a.config.split(","))}) if a.config else CONFIG
    res, train, rows = predict(a.data, targets, cfg)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_parquet(a.out, index=False)
    if a.rows:
        train.to_parquet(f"{a.rows}_train.parquet", index=False)
        rows.to_parquet(f"{a.rows}_targets.parquet", index=False)
    print(f"wrote {len(res)} predictions to {a.out}")


if __name__ == "__main__":
    main()
