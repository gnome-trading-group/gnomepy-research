"""
The price the strategy trades against, and the games it stays out of.

The market's price is the twin-combined mid of the game's two listings (each team's own market on Kalshi, the long
and short sides on Polymarket US), which uses both books and cancels a one-sided skew. The fair price moves from the
market toward the model by w in log-odds: w = 1 is the model's own price, w = 0 the market's. B0's information weight
against Kalshi was 0.16 ± 0.09 pooled 2021-24 and negative in 2026, so the anchored variant (w 0.25) only trades when
the model disagrees a lot, and the raw variant (w 1) is the control.

Games are skipped on rules from the 2026 diagnosis: a stale prediction, an uncertain QB (the model prices one starter),
a bottom-35 team in weeks 5-9 (where the model trails the line most, 2022-24) and FBS newcomers (24% of 2026 games but
57% of the model's excess loss).
"""
from __future__ import annotations

import math

from gnomepy_research.sessions.cfb_pregame.prediction_feed import Prediction

CLIP = 0.01


def twin_mid(bid_home: float, ask_home: float, bid_away: float, ask_away: float) -> float:
    """P(home) from both legs' mids; NaN unless both books are two-sided."""
    if any(math.isnan(x) for x in (bid_home, ask_home, bid_away, ask_away)):
        return math.nan
    return ((bid_home + ask_home) / 2 + 1 - (bid_away + ask_away) / 2) / 2


def _logit(p: float) -> float:
    p = min(max(p, CLIP), 1 - CLIP)
    return math.log(p / (1 - p))


def anchored_fair(p_model: float, m: float, w: float) -> float:
    x = _logit(m) + w * (_logit(p_model) - _logit(m))
    return 1 / (1 + math.exp(-x))


def skip_reason(pred: Prediction, now_ns: int, *, max_age_ns: int, min_qb_p: float, skip_bottom: int,
                skip_weeks: tuple[int, int], skip_newcomers: bool) -> str | None:
    """Why the game is not traded now, or None."""
    if now_ns - pred.at_ns > max_age_ns:
        return "stale prediction"
    if any(not math.isnan(q) and q < min_qb_p for q in (pred.home_qb_p, pred.away_qb_p)):
        return "uncertain QB"
    floor = pred.fbs_teams - skip_bottom
    if skip_weeks[0] <= pred.week <= skip_weeks[1] and any(
            r is not None and r > floor for r in (pred.home_rank, pred.away_rank)):
        return "bottom team"
    if skip_newcomers and (pred.home_newcomer or pred.away_newcomer):
        return "newcomer"
    return None
