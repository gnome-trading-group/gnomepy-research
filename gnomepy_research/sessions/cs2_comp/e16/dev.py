"""
Dev-period walk-forward diagnostics (before the tune period, no market): HLTV bo3 series of Jan-Mar 2026 as
targets, scored against their outcomes.

    python -m gnomepy_research.sessions.cs2_comp.e16.dev build <bundle> <dev targets parquet>
    python -m gnomepy_research.sessions.cs2_comp.e16.dev score <dev targets parquet> <predictions parquet> [...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.hltv.config import showmatch_mask

FIRST, END = pd.Timestamp("2026-01-05"), pd.Timestamp("2026-03-30")


def build(data: Path, out: Path) -> None:
    h = pd.read_parquet(data / "hltv/cs2_match_history.parquet")
    h = h[~showmatch_mask(h) & (h.bo_type == 3)]
    h["match_date"] = pd.to_datetime(h.match_date)
    h = h[(h.match_date >= FIRST) & (h.match_date < END)]
    s = h.sort_values("map_position_in_series").groupby("match_id").agg(
        match_date=("match_date", "first"), team_a_id=("team_a_id", "first"), team_b_id=("team_b_id", "first"),
        a=("team_a_won", "sum"), n=("team_a_won", "size"), rank_a=("team_a_rank", "first"),
        rank_b=("team_b_rank", "first"), is_lan=("is_lan", "first"), pa=("team_a_player_ids", "first"),
        pb=("team_b_player_ids", "first")).reset_index()
    s = s[(s.n >= 2) & (s.a * 2 != s.n)]
    s["y"] = (s.a * 2 > s.n).astype(int)
    s["week_cutoff"] = s.match_date - pd.to_timedelta(s.match_date.dt.weekday, unit="D")
    lu = pd.read_parquet(data / "faceit/faceit_lineups.parquet", columns=["player_id", "match_date"]).dropna()
    first = lu.groupby(lu.player_id.astype(int)).match_date.min()
    cov = []
    for r in s.itertuples():
        c = []
        for ids in (r.pa, r.pb):
            ids = [int(x) for x in ids] if ids is not None else []
            c.append(sum(1 for p in ids if p in first.index and first[p] < r.week_cutoff))
        cov.append(min(c))
    s["faceit_cov"] = pd.cut(cov, [-1, 0, 3, 5], labels=["0", "1-3", "4-5"]).astype(str)
    s["ranked"] = s[["rank_a", "rank_b"]].min(axis=1).fillna(999) <= 100
    s = s.drop(columns=["pa", "pb", "a", "n"])
    out.parent.mkdir(parents=True, exist_ok=True)
    s.to_parquet(out, index=False)
    print(f"{len(s)} dev series", s.faceit_cov.value_counts().to_dict(), "ranked", int(s.ranked.sum()))


def _ll(p, y):
    p = np.clip(np.asarray(p, float), 0.01, 0.99)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def score(targets: Path, preds: list[Path]) -> None:
    t = pd.read_parquet(targets)
    base = None
    for path in preds:
        d = t.merge(pd.read_parquet(path)[["match_id", "q"]], on="match_id")
        z = np.log(np.clip(d.q, 1e-4, 1 - 1e-4) / (1 - np.clip(d.q, 1e-4, 1 - 1e-4)))
        # A one-parameter recalibration, fitted in-sample: the pool rescales q too, so the slope is what matters.
        best = min(((s, _ll(1 / (1 + np.exp(-s * z)), d.y).mean()) for s in np.linspace(0.3, 2.0, 69)), key=lambda x: x[1])
        line = f"{path.name}: n={len(d)} LL {_ll(d.q, d.y).mean():.4f} slope {best[0]:.2f} LLcal {best[1]:.4f}"
        ll = _ll(1 / (1 + np.exp(-best[0] * z)), d.y)
        for name, g in [("ranked", d.ranked), ("unranked", ~d.ranked)] + [(f"cov{c}", d.faceit_cov == c) for c in ("0", "1-3", "4-5")]:
            line += f" | {name} {ll[g.to_numpy()].mean():.4f} ({int(g.sum())})"
        print(line)
        cur = pd.Series(np.asarray(ll), index=d.match_id.to_numpy())
        if base is None:
            base = cur
        else:
            diff = (cur - base).dropna().to_numpy()
            rng = np.random.default_rng(0)
            bs = diff[rng.integers(0, len(diff), (2000, len(diff)))].mean(1)
            print(f"   vs first: {diff.mean():+.4f} [{np.percentile(bs, 2.5):+.4f},{np.percentile(bs, 97.5):+.4f}]")


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        score(Path(sys.argv[2]), [Path(p) for p in sys.argv[3:]])
