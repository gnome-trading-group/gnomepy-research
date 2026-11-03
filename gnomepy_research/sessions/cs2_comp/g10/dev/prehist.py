"""Pre-dev validation of skill increments on HLTV outcomes alone: for each weekly cutoff in [start, end), the HLTV-only
e12 skill fit prices every bo3 series of that week (announced fives), with and without the map-pool veto and the
LAN/online team offsets. python -m ...g10.dev.prehist --data D --start 2025-01-06 --end 2025-04-07"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from gnomepy_research.sessions.cs2_comp.g10.e12_data import Inputs
from gnomepy_research.sessions.cs2_comp.g10.e12_model import Config, Lineups, SkillFit, series_price, team_state
from gnomepy_research.sessions.cs2_comp.g10.mapveto import MapConfig, MapModel, load_maps

BASE = Config(sigma=0.5, half_life_days=300.0, rating_gamma=0.6, rating_sd=0.31)
MAPS = {"m120": MapConfig(), "m240": MapConfig(half_life_days=240.0, prior_rounds=120.0),
        "m240s": MapConfig(half_life_days=240.0, prior_rounds=240.0), "m365": MapConfig(half_life_days=365.0, prior_rounds=120.0)}


def _shrunk(team: np.ndarray, g: np.ndarray, h: np.ndarray, prior: float) -> dict:
    agg = pd.DataFrame({"team": team, "g": g, "h": h}).groupby("team")[["g", "h"]].sum()
    return (agg.g / (agg.h + prior)).to_dict()


def team_offsets(mm: MapModel, form_half_life_days: float = 21.0) -> dict[str, dict]:
    """Team residuals over the whole window, over the last few weeks, and per pair of teams (head to head)."""
    x = mm.residuals
    a, b = x.team_a.to_numpy(), x.team_b.to_numpy()
    wr, wi = (x.w * x.r).to_numpy(), (x.w * x.info).to_numpy()
    out = {"team": _shrunk(np.r_[a, b], np.r_[wr, -wr], np.r_[wi, wi], mm.prior)}
    recent = (x.age <= 3 * form_half_life_days).to_numpy()
    wf = 0.5 ** (x.age.to_numpy()[recent] / form_half_life_days)
    fr, fi = wf * x.r.to_numpy()[recent], wf * x.info.to_numpy()[recent]
    out["form"] = _shrunk(np.r_[a[recent], b[recent]], np.r_[fr, -fr], np.r_[fi, fi], mm.prior)
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    sign = np.where(a == lo, 1.0, -1.0)
    pair = pd.DataFrame({"lo": lo, "hi": hi, "g": sign * wr, "h": wi}).groupby(["lo", "hi"])[["g", "h"]].sum()
    out["pair"] = (pair.g / (pair.h + mm.prior)).to_dict()
    return out


def _gap(offsets: dict, which: str, gap: float, team_a: int, team_b: int) -> float:
    o = offsets[which]
    if which == "pair":
        v = o.get((min(team_a, team_b), max(team_a, team_b)), 0.0)
        return gap + (v if team_a < team_b else -v)
    return gap + o.get(team_a, 0.0) - o.get(team_b, 0.0)


def _five(v):
    if v is None or len(v) != 5 or len(set(v)) != 5:
        return None
    return tuple(int(p) for p in v)


def main() -> None:
    threadpool_limits(1)
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    a = ap.parse_args()
    weeks = pd.date_range(a.start, a.end, freq="7D", inclusive="left")
    until = weeks[-1] + pd.Timedelta(days=7)
    inputs = Inputs(a.data, until, False)
    lineups = Lineups(inputs, False)
    maps = load_maps(a.data, until)
    s = inputs.series
    s = s[s.bo_type == 3]
    out = []
    for c in weeks:
        t0 = time.time()
        week = s[(s.time >= c) & (s.time < c + pd.Timedelta(days=7))]
        if week.empty:
            continue
        rows = lineups.at(c)
        by_team = {int(k): g for k, g in rows.groupby("team")}
        fit = SkillFit.fit(inputs, c, BASE, False)
        gaps = []
        for r in week.itertuples(index=False):
            g = []
            for team, lu in ((int(r.team_a), r.lineup_a), (int(r.team_b), r.lineup_b)):
                five = _five(lu) or team_state(by_team.get(team, rows.iloc[:0]), team, c)["five"]
                g.append(fit.lineup(five, inputs.rank(team, c))[0])
            gaps.append(g[0] - g[1])
        gaps = np.array(gaps)
        rec = {"match_id": week.match_id.to_numpy(), "time": week.time.to_numpy(), "y": week.y.to_numpy(),
               "is_lan": week.is_lan.to_numpy(), "q_base": series_price(gaps)}
        for name, cfg in MAPS.items():
            mm = MapModel(maps, fit, c, cfg)
            rec[f"q_{name}"] = [mm.series_price(g, int(x), int(z)) for g, x, z in zip(gaps, week.team_a, week.team_b)]
            if name == "m240":
                lg = np.array([mm.lan_gap(g, int(x), int(z), int(l)) for g, x, z, l in
                               zip(gaps, week.team_a, week.team_b, week.is_lan)])
                rec["q_lan"] = series_price(lg)
                rec["q_lanmap"] = [mm.series_price(g2, int(x), int(z)) for g2, x, z in zip(lg, week.team_a, week.team_b)]
                offs = team_offsets(mm)
                for which in ("team", "form", "pair"):
                    rec[f"q_{which}"] = series_price(np.array([_gap(offs, which, g, int(x), int(z))
                                                               for g, x, z in zip(gaps, week.team_a, week.team_b)]))
        out.append(pd.DataFrame(rec))
        print(f"{c:%Y-%m-%d} {len(week)} {time.time() - t0:.1f}s", flush=True)
    res = pd.concat(out, ignore_index=True)
    res.to_parquet(Path(os.environ.get("OUTPUT_DIR", ".")) / f"prehist_{a.start}.parquet", index=False)


if __name__ == "__main__":
    main()
