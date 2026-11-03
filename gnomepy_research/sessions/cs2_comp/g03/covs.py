"""Per-player covariates as z-scores: recency-weighted, prior-shrunk means of stat columns before each week."""
from __future__ import annotations

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.g03 import faceit


def _naive(s):
    s = pd.to_datetime(s)
    return s.dt.tz_convert(None) if s.dt.tz is not None else s


def rolling_z(df: pd.DataFrame, weeks, pcol: str, dcol: str, vcol: str, window: int, half_life: float, prior_n: float, center: float, spread: float, count: bool = False):
    d = df[[pcol, dcol, vcol]].dropna()
    dates = _naive(d[dcol])
    out = {}
    for w in weeks:
        m = (dates < w) & (dates >= w - pd.Timedelta(days=window))
        win = d[m.to_numpy()]
        age = (w - dates[m]).dt.total_seconds().to_numpy() / 86400.0
        wt = 0.5 ** (age / half_life)
        g = pd.DataFrame({"p": win[pcol].to_numpy(), "w": wt, "wv": wt * (win[vcol].to_numpy() - center) / spread}).groupby("p").sum()
        out[w] = (np.log1p(g.w) / 3.0 if count else g.wv / (g.w + prior_n)).to_dict()
    return out


def build(data_dir, cutoff, weeks, hltv_stats: pd.DataFrame, specs: list[str]):
    """specs: names from SPECS; returns {week: [dict, ...]} aligned to specs."""
    per = {}
    s = hltv_stats[hltv_stats.side == "all"]
    fst = None
    for name in specs:
        kind, col, window, hl, prior, center, spread = SPECS[name]
        count = name.startswith("cnt")
        if kind == "hltv":
            per[name] = rolling_z(s, weeks, "player_id", "match_date", col, window, hl, prior, center, spread, count)
        else:
            if fst is None:
                fst = faceit.faceit_frames(data_dir, cutoff)[0]
            per[name] = rolling_z(fst, weeks, "hltv_id", "d", col, window, hl, prior, center, spread, count)
    return {w: [per[n][w] for n in specs] for w in weeks}


RATING = (1.09, 0.385)
SPECS = {
    "hr": ("hltv", "rating", 120, 60.0, 10.0, *RATING),
    "hr_long": ("hltv", "rating", 365, 250.0, 15.0, *RATING),
    "adr": ("hltv", "adr", 180, 90.0, 10.0, 73.3, 19.5),
    "kast": ("hltv", "kast", 180, 90.0, 10.0, 71.3, 12.4),
    "swing": ("hltv", "round_swing_pct", 180, 90.0, 10.0, 0.0, 4.5),
    "cnt_h": ("hltv", "rating", 365, 250.0, 1.0, 0.0, 1.0),
    "cnt_f": ("faceit", "score", 180, 90.0, 1.0, 0.0, 1.0),
    "fs": ("faceit", "score", 180, 90.0, 10.0, 0.526, 0.125),
}
