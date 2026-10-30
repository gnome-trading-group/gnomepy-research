"""
Development harness: walk-forward on HLTV bo3 series before April 2026, scored on their own outcomes (no market).

    python -m gnomepy_research.sessions.cs2_comp.e08.dev --data <bundle> [--no-faceit] [--set key=value ...]

Parameters are chosen here, on the Oct 2025 - Mar 2026 series, and fixed in the code before the tune period.
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.hltv.config import showmatch_mask
from gnomepy_research.sessions.cs2_comp.bundle import load
from gnomepy_research.sessions.cs2_comp.e08.data import RankPoints, build_maps
from gnomepy_research.sessions.cs2_comp.e08.filter import Params
from gnomepy_research.sessions.cs2_comp.e08.series import Moves, SeriesParams
from gnomepy_research.sessions.cs2_comp.e08.walk import walk

DEV_START, DEV_END = pd.Timestamp("2025-10-06"), pd.Timestamp("2026-03-30")
TRACK_FROM = int(pd.Timestamp("2025-01-01").value)
THIN_MAPS, TOP = 15, 100


def dev_targets(data_dir: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    h = load(data_dir, "hltv/cs2_match_history", end)
    h = h[~showmatch_mask(h) & (h.team_a_score != h.team_b_score)].copy()
    h["time"] = pd.to_datetime(h.match_time.fillna(h.match_date)).dt.tz_convert(None)
    history = collections.defaultdict(list)
    for t, a, b in zip(h.time.to_numpy("datetime64[ns]"), h.team_a_id, h.team_b_id):
        history[a].append(t)
        history[b].append(t)
    history = {k: np.sort(np.array(v)) for k, v in history.items()}
    h = h[h.bo_type == 3]
    wins = h.groupby("match_id").team_a_won.agg(["sum", "count"])
    first = h.sort_values("time").drop_duplicates("match_id").set_index("match_id")
    first["y"] = (wins["sum"] * 2 > wins["count"]).astype(int)
    first["maps_played"] = wins["count"]
    first = first[(first.time >= start) & (first.time < end) & (first.maps_played >= 2)].reset_index()
    window = np.timedelta64(90, "D")

    def recent_maps(team: int, t) -> int:
        times = history[team]
        t = np.datetime64(t)
        return int(np.searchsorted(times, t, side="left") - np.searchsorted(times, t - window, side="left"))

    thin = [
        (not lan) and (min(recent_maps(a, t), recent_maps(b, t)) <= THIN_MAPS or not (ra <= TOP or rb <= TOP))
        for a, b, t, lan, ra, rb in zip(first.team_a_id, first.team_b_id, first.time, first.is_lan.astype(bool),
                                        first.team_a_rank.fillna(999), first.team_b_rank.fillna(999))
    ]
    first["thin"] = thin
    first["match_time"] = first.time
    first["week_cutoff"] = first.time.dt.normalize() - pd.to_timedelta(first.time.dt.weekday, unit="D")
    return first[["match_id", "match_time", "team_a_id", "team_b_id", "week_cutoff", "y", "thin"]]


def ll(p, y) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 0.01, 0.99)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def summarize(out: pd.DataFrame, label: str) -> dict:
    d = ll(out.q, out.y)
    cov = np.minimum(out.n_face_a, out.n_face_b)
    seg = {"all": d.mean()}
    for name, mask in (("thin", out.thin.astype(bool)), ("rest", ~out.thin.astype(bool)), ("cov0", cov == 0),
                       ("cov1-3", (cov >= 1) & (cov <= 3)), ("cov4-5", cov >= 4)):
        seg[name] = d[mask.to_numpy()].mean() if mask.any() else np.nan
        seg["n_" + name] = int(mask.sum())
    print(label, " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in seg.items()), flush=True)
    return seg


def apply_overrides(items: list[str]) -> tuple[Params, SeriesParams]:
    params, sp = Params(), SeriesParams()
    for item in items:
        k, v = item.split("=")
        for obj_name, obj in (("p", params), ("s", sp)):
            if k in {f.name for f in dataclasses.fields(obj)}:
                cur = getattr(obj, k)
                val = tuple(float(x) for x in v.split(",")) if isinstance(cur, tuple) else type(cur)(float(v) if not isinstance(cur, bool) else v == "1")
                obj = dataclasses.replace(obj, **{k: val})
                params, sp = (obj, sp) if obj_name == "p" else (params, obj)
                break
        else:
            raise KeyError(k)
    return params, sp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--run", nargs="*", default=[""], help="each a comma-separated list of key=value overrides")
    ap.add_argument("--end", default=None)
    ap.add_argument("--save", type=Path, default=None)
    args = ap.parse_args()
    end = pd.Timestamp(args.end) if args.end else DEV_END
    targets = dev_targets(args.data, DEV_START, end)
    if args.cache and args.cache.exists():
        maps = pickle.load(open(args.cache, "rb"))
    else:
        maps = build_maps(args.data, end, use_faceit=True)
        if args.cache:
            pickle.dump(maps, open(args.cache, "wb"))
    ranks = RankPoints(load(args.data, "hltv/cs2_team_rankings", end))
    moves = Moves(load(args.data, "hltv/cs2_roster_moves", end))
    for k, spec in enumerate(args.run):
        params, sp = apply_overrides([x for x in spec.split(",") if x])
        t0 = time.time()
        out, f, ff = walk(maps, targets, moves, ranks, params, sp, track_from=TRACK_FROM)
        out = out.merge(targets[["match_id", "y", "thin"]], on="match_id")
        face = f"{ff.nll[1] / max(ff.nmaps[1], 1):.5f}" if ff is not None else "-"
        print(f"[{spec or 'default'}] {len(out)} series {time.time() - t0:.0f}s prequential round LL hltv "
              f"{f.nll[0] / max(f.nmaps[0], 1):.5f} linked {f.linked_nll / max(f.nmaps[0], 1):.5f} faceit {face}", flush=True)
        summarize(out, "  series LL")
        if args.save:
            out.to_parquet(args.save.with_name(f"{args.save.stem}_{k}.parquet"))


if __name__ == "__main__":
    main()
