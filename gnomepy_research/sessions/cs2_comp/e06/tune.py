"""Pre-tune diagnostics: weekly point-in-time prices of HLTV bo3 series against their outcomes, no market involved.

    python -m gnomepy_research.sessions.cs2_comp.e06.tune --data <bundle> --start 2025-10-01 --end 2026-04-01 [--no-faceit]
        [--oracle] [--set name=value ...]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.hltv.config import showmatch_mask
from gnomepy_research.sessions.cs2_comp.e06.run import load_inputs, run_pass
from gnomepy_research.sessions.cs2_comp.e06.skill import NAMES, Params


def monday(ts: pd.Series) -> pd.Series:
    ts = ts.dt.normalize()
    return ts - pd.to_timedelta(ts.dt.dayofweek, unit="D")


def build_queries(data_dir: Path, start: str, end: str) -> tuple[pd.DataFrame, dict]:
    h = pd.read_parquet(Path(data_dir) / "hltv/cs2_match_history.parquet")
    h = h[~showmatch_mask(h) & (h.bo_type == 3)].copy()
    h["t"] = pd.to_datetime(h.match_time.fillna(h.match_date)).dt.tz_convert(None)
    h = h[(h.t >= start) & (h.t < end)].sort_values(["t", "match_id", "map_position_in_series"])
    won = h.groupby("match_id").team_a_won.agg(lambda w: int(w.sum() * 2 > len(w)))
    nmaps = h.groupby("match_id").size()
    first = h.drop_duplicates("match_id").copy()
    first = first[first.match_id.map(nmaps) >= 2]
    first["y"] = first.match_id.map(won)
    first["kickoff"] = first.t.astype("datetime64[ns]").astype("int64") / 1e9
    first["week_cutoff"] = monday(first.t)
    lineups = {r.match_id: (tuple(int(p) for p in r.team_a_lineup_ids), tuple(int(p) for p in r.team_b_lineup_ids))
               for r in first.itertuples() if r.team_a_lineup_ids is not None and r.team_b_lineup_ids is not None
               and len(r.team_a_lineup_ids) == 5 and len(r.team_b_lineup_ids) == 5}
    first = first[first.match_id.isin(lineups)]
    full = pd.read_parquet(Path(data_dir) / "hltv/cs2_match_history.parquet", columns=["match_id", "match_date", "team_a_id", "team_b_id"])
    days = pd.concat([full[["match_date", "team_a_id"]].set_axis(["d", "t"], axis=1), full[["match_date", "team_b_id"]].set_axis(["d", "t"], axis=1)])
    by_team = {k: np.sort(pd.to_datetime(g.d).to_numpy()) for k, g in days.groupby("t")}

    def maps90(team, day):
        a = by_team.get(team, np.array([], "datetime64[ns]"))
        return np.searchsorted(a, np.datetime64(day)) - np.searchsorted(a, np.datetime64(day - pd.Timedelta(days=90)))

    cut = first.week_cutoff
    first["maps90"] = [min(maps90(a, c), maps90(b, c)) for a, b, c in zip(first.team_a_id, first.team_b_id, cut)]
    best = np.fmin(first.team_a_rank.fillna(999), first.team_b_rank.fillna(999))
    first["thin"] = (first.is_lan == 0) & ((first.maps90 <= 15) | (best > 100))
    return first[["match_id", "week_cutoff", "kickoff", "team_a_id", "team_b_id", "y", "thin", "is_lan"]], lineups


def log_loss(p, y):
    p = np.clip(np.asarray(p, float), 0.01, 0.99)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def summarize(res: pd.DataFrame, q: pd.DataFrame) -> dict:
    d = q.merge(res, on="match_id")
    d["ll"] = log_loss(d.q, d.y)
    out = {"n": len(d), "ll": float(d.ll.mean())}
    for name, g in [("thin", d[d.thin]), ("rest", d[~d.thin])] + [(f"cov{c}", d[d.faceit_depth.between(lo, hi)])
                                                                   for c, (lo, hi) in {"0": (0, 0), "1-3": (1, 3), "4-5": (4, 5)}.items()]:
        out[name] = {"n": len(g), "ll": float(g.ll.mean()) if len(g) else None}
    return out


def evaluate(data_dir: Path, params: Params, start: str, end: str, use_faceit: bool = True, oracle: bool = False,
             preloaded=None, joint: bool = True) -> tuple[dict, pd.DataFrame]:
    events, rankings, moves = preloaded or load_inputs(data_dir, use_faceit)
    q, lineups = build_queries(data_dir, start, end)
    res = run_pass(events, rankings, moves, params, q, known_lineups=lineups if oracle else None, use_faceit=use_faceit,
                   joint=joint)
    return summarize(res, q), res.merge(q, on="match_id")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--start", default="2025-10-01")
    ap.add_argument("--end", default="2026-04-01")
    ap.add_argument("--no-faceit", action="store_true")
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    kv = {}
    for item in args.set:
        k, v = item.split("=")
        assert k in NAMES, k
        kv[k] = json.loads(v)
    params = Params().with_values(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in kv.items()})
    t0 = time.time()
    summary, _ = evaluate(args.data, params, args.start, args.end, not args.no_faceit, args.oracle)
    print(json.dumps(summary), f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
