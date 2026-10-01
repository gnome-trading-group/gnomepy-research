"""
Per-player form, aggregated to the announced lineup.

The scraper always computed per-player kills/deaths/ADR/KAST/rating and threw
all but the team mean away. This keeps it. That matters because the team mean
is dominated by the roster's median player, while a lineup's weakest slot and
its stand-in status are what actually move a price.

Two choices worth stating.

Form is read off the *announced* lineup (team_a_lineup_ids), not the players who
turn up in the post-match stats table. An earlier attempt used the latter and it
was unusable: who played is only known afterwards, so the feature could not have
existed at prediction time. When the announced lineup is missing, every column
here is NaN rather than silently falling back.

Skill is tracked as a residual against HLTV's own opponent-strength expectation
(adr - eadr, kast - ekast, kills - ek) as well as raw rating. A 1.15 rating
against tier-3 opposition and a 1.15 against top-10 are different observations,
and the raw number cannot tell them apart.

State updates run through pit.pit_scan, so ratings freeze per series by
construction rather than by remembering to.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from gnomepy_research.pipelines.hltv_cs2.pit import SeriesBatch, pit_scan

logger = logging.getLogger(__name__)

PLAYERS_AUX = "players"

_SIDES = ("all", "ct", "t")


@dataclass(frozen=True)
class FormConfig:
    halflife_maps: float = 20.0
    min_maps: int = 3


@dataclass
class _PlayerForm:
    rating: dict[str, float] = field(default_factory=dict)
    swing: float | None = None
    adr_resid: float | None = None
    kast_resid: float | None = None
    kd_resid: float | None = None
    maps: int = 0


def _ewma(prev: float | None, value: float, alpha: float) -> float:
    return value if prev is None else prev + alpha * (value - prev)


def _num(value) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _lineup(row, side: str) -> list[int]:
    ids = getattr(row, f"team_{side}_lineup_ids", None)
    if ids is None or (isinstance(ids, float) and math.isnan(ids)):
        return []
    return [int(p) for p in ids if p is not None]


class PlayerFormAccumulator:
    """Tracks per-player EWMA form; emits lineup aggregates per series."""

    def __init__(self, cfg: FormConfig | None = None):
        self.cfg = cfg or FormConfig()
        self.alpha = 1.0 - 0.5 ** (1.0 / self.cfg.halflife_maps)
        self.players: dict[int, _PlayerForm] = {}

    def _aggregate(self, ids: list[int]) -> dict:
        known = [self.players[p] for p in ids if p in self.players
                 and self.players[p].maps >= self.cfg.min_maps]
        if not known:
            return {
                "form_rating": np.nan, "form_worst": np.nan, "form_spread": np.nan,
                "form_swing": np.nan, "form_adr_resid": np.nan, "form_kast_resid": np.nan,
                "form_ct": np.nan, "form_t": np.nan,
                "form_maps_min": 0.0, "form_known": 0.0,
            }
        ratings = [f.rating.get("all") for f in known if f.rating.get("all") is not None]
        ct = [f.rating.get("ct") for f in known if f.rating.get("ct") is not None]
        t = [f.rating.get("t") for f in known if f.rating.get("t") is not None]
        swing = [f.swing for f in known if f.swing is not None]
        adr = [f.adr_resid for f in known if f.adr_resid is not None]
        kast = [f.kast_resid for f in known if f.kast_resid is not None]
        return {
            "form_rating": float(np.mean(ratings)) if ratings else np.nan,
            "form_worst": float(np.min(ratings)) if ratings else np.nan,
            "form_spread": float(np.std(ratings)) if len(ratings) > 1 else np.nan,
            "form_swing": float(np.mean(swing)) if swing else np.nan,
            "form_adr_resid": float(np.mean(adr)) if adr else np.nan,
            "form_kast_resid": float(np.mean(kast)) if kast else np.nan,
            "form_ct": float(np.mean(ct)) if ct else np.nan,
            "form_t": float(np.mean(t)) if t else np.nan,
            "form_maps_min": float(min(f.maps for f in known)),
            "form_known": len(known) / max(len(ids), 1),
        }

    def snapshot(self, batch: SeriesBatch) -> dict:
        row = next(batch.rows.itertuples())
        a = self._aggregate(_lineup(row, "a"))
        b = self._aggregate(_lineup(row, "b"))
        out = {}
        for name, value in a.items():
            out[f"team_a_{name}"] = value
        for name, value in b.items():
            out[f"team_b_{name}"] = value
        for name in ("form_rating", "form_worst", "form_swing", "form_adr_resid",
                     "form_kast_resid", "form_ct", "form_t"):
            out[f"{name}_diff"] = a[name] - b[name]
        out["form_maps_min"] = min(a["form_maps_min"], b["form_maps_min"])
        out["form_known_min"] = min(a["form_known"], b["form_known"])
        return out

    def update(self, batch: SeriesBatch) -> None:
        stats = batch.aux_frame(PLAYERS_AUX)
        if stats.empty:
            return
        for rec in stats.itertuples():
            pid = _num(getattr(rec, "player_id", None))
            if pid is None:
                continue
            form = self.players.setdefault(int(pid), _PlayerForm())
            side = str(getattr(rec, "side", "all"))
            if side not in _SIDES:
                continue

            rating = _num(getattr(rec, "rating", None))
            if rating is not None:
                form.rating[side] = _ewma(form.rating.get(side), rating, self.alpha)

            if side != "all":
                continue
            form.maps += 1
            swing = _num(getattr(rec, "round_swing_pct", None))
            if swing is not None:
                form.swing = _ewma(form.swing, swing, self.alpha)
            adr, eadr = _num(getattr(rec, "adr", None)), _num(getattr(rec, "eadr", None))
            if adr is not None and eadr is not None:
                form.adr_resid = _ewma(form.adr_resid, adr - eadr, self.alpha)
            kast, ekast = _num(getattr(rec, "kast", None)), _num(getattr(rec, "ekast", None))
            if kast is not None and ekast is not None:
                form.kast_resid = _ewma(form.kast_resid, kast - ekast, self.alpha)
            kills, ek = _num(getattr(rec, "kills", None)), _num(getattr(rec, "ek", None))
            deaths, ed = _num(getattr(rec, "deaths", None)), _num(getattr(rec, "ed", None))
            if None not in (kills, ek, deaths, ed):
                form.kd_resid = _ewma(form.kd_resid, (kills - ek) - (deaths - ed), self.alpha)


PLAYER_FORM_FEATURES = [
    f"team_{s}_{n}" for s in ("a", "b") for n in
    ("form_rating", "form_worst", "form_spread", "form_swing",
     "form_adr_resid", "form_kast_resid", "form_ct", "form_t",
     "form_maps_min", "form_known")
] + [
    f"{n}_diff" for n in
    ("form_rating", "form_worst", "form_swing", "form_adr_resid",
     "form_kast_resid", "form_ct", "form_t")
] + ["form_maps_min", "form_known_min"]


def compute_player_form_features(
    history: pd.DataFrame,
    player_stats: pd.DataFrame,
    cfg: FormConfig | None = None,
) -> pd.DataFrame:
    """Pre-series lineup form for every map row, keyed (match_id, map_name)."""
    if "team_a_lineup_ids" not in history.columns:
        logger.warning("team_a_lineup_ids absent — player form unavailable, emitting no rows")
        return pd.DataFrame(columns=["match_id", "map_name"])
    return pit_scan(history, PlayerFormAccumulator(cfg), aux={PLAYERS_AUX: player_stats})
