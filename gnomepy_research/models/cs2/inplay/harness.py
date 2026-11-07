"""Competition 3 scorer: an in-play CS2 map-win model against the baseline, walk-forward by month.

    python -m gnomepy_research.models.cs2.inplay.harness ENTRY.py --data DIR [--months 2026-07 2026-08] [--quick]
        [--seeds 1] [--audit 100] [--cache DIR] [--out FILE.json] [--log FILE.jsonl] [--name NAME] [--holdout DIR]

An entry is a Python file defining
    fit(train: Bundle, asof: pd.Timestamp, seed: int) -> model
    predict(model, ctx: Bundle, states: pd.DataFrame) -> array of P(team A wins the map), one per states row
For each scored month: `train` holds every map that ended before the month starts (outcomes included). `ctx` holds the
same history plus the month's maps with their outcome columns blanked (data.OUTCOME); `states` lists the month's
states to price (round start and after each kill / bomb plant; game_id, round, step, kind, event_seq, ingame_ms,
elapsed). A state's price may use only: that map's rounds up to its own (round-start info only for its own round), that
map's events with seq <= event_seq, and other maps that ended before this map began. The audit re-prices sampled
states on a bundle truncated to exactly that and must reproduce them (|diff| <= 1e-6), so predictions must not depend on
which other states share the batch.

Score: mean log loss over all states, entry minus benchmark; 95% interval from a map-level bootstrap; months better.
Development months default to 2026-04..2026-08 (events start 2026-01, so the round model needs Jan–Mar to learn from).
--quick scores 2026-08 only. --holdout DIR (scorer only) adds the vault and scores 2026-09..2026-11."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import baseline
from .data import Bundle, concat
from .features import legal_maps, map_states

DEV_MONTHS = ["2026-04", "2026-05", "2026-06", "2026-07", "2026-08"]
HOLDOUT_MONTHS = ["2026-09", "2026-10", "2026-11"]
EPS = 1e-4


def ll(p, y):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def load_entry(path):
    spec = importlib.util.spec_from_file_location("entry", path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(Path(path).parent))
    spec.loader.exec_module(mod)
    return mod


def month_split(b: Bundle, month: str):
    start = pd.Timestamp(month + "-01", tz="UTC")
    end = start + pd.offsets.MonthBegin(1)
    g = b.games
    ids = g.game_id[(g.match_begin >= start) & (g.match_begin < end) & g.map_a_wins.notna()]
    test = b.subset(set(ids) & legal_maps(b.subset(ids)))
    train = b.ended_before(start)
    states = map_states(test)
    labels = states[["game_id"]].merge(test.games[["game_id", "map_a_wins"]], on="game_id").map_a_wins.to_numpy(float)
    ctx = concat([train, test.without_outcomes()])
    return start, train, ctx, states, labels


def run(entry, b: Bundle, months, seed, log=print):
    out = []
    for m in months:
        t0 = time.time()
        asof, train, ctx, states, y = month_split(b, m)
        model = entry.fit(train, asof, seed)
        p = np.asarray(entry.predict(model, ctx, states), float)
        assert p.shape == (len(states),) and np.isfinite(p).all(), f"bad predictions for {m}"
        out.append(states.assign(month=m, p=p, y=y))
        log(f"  {m}: {len(states)} states, {states.game_id.nunique()} maps, log loss {ll(p, y).mean():.4f} ({time.time() - t0:.0f}s)")
    return pd.concat(out, ignore_index=True), model, (asof, ctx, states)


def audit(entry, model, last, n, seed):
    """Truncation invariance on n sampled states of the last scored month."""
    asof, ctx, states = last
    full = np.asarray(entry.predict(model, ctx, states), float)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(states), min(n, len(states)), replace=False)
    g = ctx.games.set_index("game_id")
    worst = 0.0
    for i in pick:
        s = states.iloc[[i]]
        gid, rnd, seq = int(s.game_id.iloc[0]), int(s["round"].iloc[0]), int(s.event_seq.iloc[0])
        began = g.begin.get(gid)
        visible = set(g.index[(g.end < began) | g.index.isin(ctx.games.game_id[ctx.games.end < asof])]) | {gid}
        sub = ctx.subset(visible).truncate(gid, rnd, seq)
        p = float(np.asarray(entry.predict(model, sub, s.reset_index(drop=True)), float)[0])
        worst = max(worst, abs(p - full[i]))
    return dict(states=len(pick), max_abs_diff=worst, passed=bool(worst <= 1e-6))


def compare(e: pd.DataFrame, bm: pd.DataFrame, rng) -> dict:
    d = e.merge(bm[["game_id", "event_seq", "p"]], on=["game_id", "event_seq"], suffixes=("", "_b"))
    y = d.y.to_numpy()
    diff = ll(d.p, y) - ll(d.p_b, y)
    s = pd.DataFrame(dict(d=diff, g=d.game_id.to_numpy())).groupby("g").d.agg(["sum", "size"])
    a, n = s["sum"].to_numpy(), s["size"].to_numpy()
    boot = [a[i].sum() / n[i].sum() for i in (rng.integers(0, len(a), len(a)) for _ in range(2000))]
    per = d.assign(x=diff).groupby("month").x.mean()
    lo, hi = np.quantile(boot, [0.025, 0.975])
    dead = (d.step.clip(upper=8)).astype(int)
    return {
        "states": len(d), "maps": int(d.game_id.nunique()),
        "entry_ll": float(ll(d.p, y).mean()), "bench_ll": float(ll(d.p_b, y).mean()),
        "diff": float(diff.mean()), "lo": float(lo), "hi": float(hi),
        "months_better": int((per < 0).sum()), "months": len(per), "per_month": {k: round(float(v), 5) for k, v in per.items()},
        "round_start_diff": float(diff[d.kind.to_numpy() == "start"].mean()),
        "by_events_so_far": {int(k): round(float(v), 5) for k, v in pd.Series(diff).groupby(dead.to_numpy()).mean().items()},
        "win": bool(hi < 0 and (per < 0).sum() * 2 > len(per)),
    }


def bench_preds(b: Bundle, months, data_key, cache: Path, log):
    f = cache / f"bench_{data_key}_{'_'.join(months)}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    log("benchmark (cached after the first run):")
    p, _, _ = run(baseline, b, months, 0, log)
    cache.mkdir(parents=True, exist_ok=True)
    p.to_parquet(f)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("entry")
    ap.add_argument("--data", required=True)
    ap.add_argument("--holdout", default=None)
    ap.add_argument("--months", nargs="*", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--audit", type=int, default=100)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--log", default=None)
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    dirs = [a.data] + ([a.holdout] if a.holdout else [])
    b = Bundle.load(*dirs)
    months = a.months or (HOLDOUT_MONTHS if a.holdout else ["2026-08"] if a.quick else DEV_MONTHS)
    key = hashlib.sha1("|".join(f"{d}:{(Path(d) / 'events.parquet').stat().st_size}" for d in dirs).encode()).hexdigest()[:10]
    cache = Path(a.cache) if a.cache else Path(a.data).parent / "bench_cache"
    log = lambda m: print(m, file=sys.stderr, flush=True)
    bm = bench_preds(b, months, key, cache, log)
    entry = load_entry(a.entry)
    t0 = time.time()
    log(f"entry {a.entry}:")
    runs = []
    for s in range(a.seeds):
        p, model, last = run(entry, b, months, s, log)
        runs.append(p)
    rng = np.random.default_rng(0)
    res = compare(runs[0], bm, rng)
    if a.seeds > 1:
        res["diff_by_seed"] = [round(compare(r, bm, rng)["diff"], 5) for r in runs]
    if a.audit:
        log(f"audit: re-pricing {a.audit} states on truncated bundles")
        res["audit"] = audit(entry, model, last, a.audit, 0)
    res.update(name=a.name or Path(a.entry).stem, entry=str(a.entry), months_scored=months, set="holdout" if a.holdout else "development",
               minutes=round((time.time() - t0) / 60, 1), finished=pd.Timestamp.now(tz="UTC").isoformat())
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
        runs[0].to_parquet(Path(a.out).with_suffix(".preds.parquet"))
    if a.log:
        with open(a.log, "a") as fh:
            fh.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
