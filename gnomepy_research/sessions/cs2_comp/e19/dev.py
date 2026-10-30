"""
Score the weekly one-step-ahead HLTV bo3 channel prices written by predict.py --dev-out against their outcomes on
weeks before the tune period, with the same trailing stack the predict module fits.

    python -m gnomepy_research.sessions.cs2_comp.e19.dev <dev parquet> [<dev parquet> ...] --from 2026-01-05 --to 2026-03-30
        [--channels hltv,both]

Each file is scored with every channel subset given by --channels (default: the first channel alone, and all of
the file's channels); the first line printed is the reference every other line is compared with.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.e19.predict import CHANNELS, apply_stack, calibrate, features


def log_loss(p, y):
    p = np.clip(np.asarray(p, float), 0.01, 0.99)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def stacked(w: pd.DataFrame, channels: tuple[str, ...]) -> pd.Series:
    out = pd.Series(np.nan, index=w.index)
    for m, g in w.groupby("week"):
        out[g.index] = apply_stack(features(g, channels), calibrate(w, m, channels))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--from", dest="start", default="2026-01-05")
    ap.add_argument("--to", dest="end", default="2026-03-30")
    ap.add_argument("--channels", action="append")
    a = ap.parse_args()
    base = None
    rng = np.random.default_rng(0)
    for path in a.files:
        w = pd.read_parquet(path)
        present = tuple(c for c in CHANNELS if f"q_{c}" in w)
        subsets = [tuple(s.split(",")) for s in a.channels] if a.channels else [present[:1], present]
        for channels in dict.fromkeys(subsets):
            if any(c not in present for c in channels):
                continue
            q = stacked(w, channels)
            s = (w.week >= pd.Timestamp(a.start)) & (w.week < pd.Timestamp(a.end))
            l = pd.Series(log_loss(q[s], w.y[s].to_numpy(float)), index=w.match_id[s].to_numpy())
            raw = log_loss(w[f"q_{channels[0]}"][s], w.y[s].to_numpy(float)).mean()
            line = f"{path.split('/')[-1]:28s} {'+'.join(channels):18s} n={int(s.sum())} raw({channels[0]}) {raw:.4f} stacked {l.mean():.4f}"
            if base is None:
                base = l
            else:
                d = l.sub(base).dropna().to_numpy()
                lo, hi = np.percentile(d[rng.integers(0, len(d), (2000, len(d)))].mean(1), [2.5, 97.5])
                line += f"  vs first {d.mean():+.4f} [{lo:+.4f},{hi:+.4f}]{'*' if hi < 0 or lo > 0 else ''}"
            print(line)


if __name__ == "__main__":
    main()
