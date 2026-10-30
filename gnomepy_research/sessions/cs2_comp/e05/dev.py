"""Walk-forward on HLTV bo3 series before the tune period: weekly cutoffs, lineups inferred from history, no market."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.hltv.config import showmatch_mask
from gnomepy_research.sessions.cs2_comp.e05.pricing import PriceConfig, SeriesPricer, series_probability
from gnomepy_research.sessions.cs2_comp.e05.stream import Config
from gnomepy_research.sessions.cs2_comp.e05.world import World


def dev_series(world: World, start: str, end: str) -> pd.DataFrame:
    h = world.history(world.last_cutoff)
    h = h[~showmatch_mask(h) & (h.bo_type == 3) & (h.team_a_score != h.team_b_score)]
    h = h.assign(time=pd.to_datetime(h.match_time.fillna(h.match_date), utc=True).dt.tz_convert(None))
    s = h.groupby("match_id").agg(time=("time", "min"), team_a=("team_a_id", "first"), team_b=("team_b_id", "first"),
                                  wins=("team_a_won", "sum"), maps=("team_a_won", "size"), lan=("is_lan", "first"),
                                  event=("event_name", "first")).reset_index()
    s = s[(s.maps >= 2) & (s.wins * 2 != s.maps) & (s.time >= start) & (s.time < end)]
    s["y"] = (s.wins * 2 > s.maps).astype(float)
    s["week"] = s.time.dt.normalize() - pd.to_timedelta(s.time.dt.dayofweek, unit="D")
    return s.reset_index(drop=True)


def actual_lineups(world: World, series: pd.DataFrame) -> dict[tuple[int, int], list[int]]:
    h = world.history(world.last_cutoff)
    first = h[h.match_id.isin(series.match_id)].drop_duplicates("match_id").set_index("match_id")
    return {(m, side): [int(p) for p in first.loc[m, f"team_{side}_player_ids"]] for m in first.index for side in "ab"}


@dataclass
class Parts:
    """Per-series pieces of the price that do not depend on the price scales, so those can be tuned without refitting."""
    frame: pd.DataFrame        # match_id, y, cover
    parts: np.ndarray          # (n, 6) see pricing.series_probability
    table: np.ndarray
    lam: float

    def q(self, price: PriceConfig) -> np.ndarray:
        return series_probability(self.table, self.lam, self.parts, price)

    def result(self, price: PriceConfig) -> pd.DataFrame:
        return self.frame.assign(q=self.q(price))


def run_parts(world: World, series: pd.DataFrame, cfg: Config, price: PriceConfig, oracle: bool = False) -> Parts:
    flt = world.new_filter(cfg)
    actual = actual_lineups(world, series) if oracle else {}
    rows, parts = [], []
    for week, wk in series.groupby("week"):
        cutoff = pd.Timestamp(week)
        flt.advance(cutoff)
        lineups = world.lineups(cutoff, set(wk.team_a.astype(int)) | set(wk.team_b.astype(int)), price)
        pricer = SeriesPricer(flt, world.rankings, cutoff, price)
        for r in wk.itertuples():
            la, lb = (([(p, 1.0) for p in actual[(r.match_id, "a")]], [(p, 1.0) for p in actual[(r.match_id, "b")]]) if oracle
                      else (lineups.get(int(r.team_a), []), lineups.get(int(r.team_b), [])))
            parts.append(pricer.components(int(r.team_a), la, int(r.team_b), lb))
            top = lambda l: [p for p, _ in sorted(l, key=lambda t: -t[1])[:5]]
            cover = min(sum(p in world.faceit_players for p in top(la)), sum(p in world.faceit_players for p in top(lb)))
            rows.append((r.match_id, r.y, cover))
    pricer = SeriesPricer(flt, world.rankings, pd.Timestamp(series.week.max()), price)
    return Parts(pd.DataFrame(rows, columns=["match_id", "y", "cover"]), np.array(parts), pricer.table, pricer.hltv_lam)


def run_dev(world: World, series: pd.DataFrame, cfg: Config, price: PriceConfig, oracle: bool = False) -> pd.DataFrame:
    return run_parts(world, series, cfg, price, oracle).result(price)


def logloss(q, y) -> np.ndarray:
    q = np.clip(np.asarray(q, float), 0.01, 0.99)
    return -(y * np.log(q) + (1 - y) * np.log(1 - q))


def summarize(res: pd.DataFrame, name: str = "") -> str:
    ok = res.dropna(subset=["q"])
    parts = [f"{name} n={len(ok)}/{len(res)} ll={logloss(ok.q, ok.y).mean():.4f}"]
    for lab, m in (("c0", ok.cover == 0), ("c1-3", ok.cover.between(1, 3)), ("c4-5", ok.cover >= 4)):
        if m.sum():
            parts.append(f"{lab}[{int(m.sum())}]={logloss(ok.q[m], ok.y[m]).mean():.4f}")
    return " ".join(parts)
