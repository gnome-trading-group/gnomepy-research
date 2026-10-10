"""
The predictions table: the only thing the strategy knows about the model.

Predictions are computed outside the strategy and handed to it as a table, so
the strategy stays a thin execution layer and every number it trades on can be
audited and replayed. The live pipeline appends a row per market every run, so a
market has a history of rows:

    match_id         HLTV match id
    market           "series" or "game1"
    p_team_a         model probability that HLTV's team A wins that market
    rank_known_both  1.0 if both teams had a rank when predicted (for the filter)
    kickoff          scheduled start (UTC) as known when the row was written
    model_version    identifies the frozen model that produced the row
    generated_at     UTC time the row was written

The strategy uses the latest row with generated_at <= now: point-in-time in a
backtest, newest in a live session. Kickoff is read the same way, so a match
HLTV reschedules moves its entry window without relaunching the session.

Team A is HLTV's first-listed team; the config maps it to a Polymarket token, so
the orientation is fixed at config time and never inferred from prices.
"""
from __future__ import annotations

import bisect

import pandas as pd

from gnomepy_research.artifacts import DatasetStore

COLUMNS = ["match_id", "market", "p_team_a", "rank_known_both", "kickoff", "model_version", "generated_at"]
MARKETS = ("series", "game1")
WALK_FORWARD_LEAD = pd.Timedelta(hours=24)


def _utc(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    return s.dt.tz_convert("UTC") if s.dt.tz is not None else s.dt.tz_localize("UTC")


def validate(df: pd.DataFrame) -> pd.DataFrame:
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"predictions table missing columns: {sorted(missing)}")
    bad_market = set(df.market) - set(MARKETS)
    if bad_market:
        raise ValueError(f"unknown market values: {sorted(bad_market)}")
    p = pd.to_numeric(df.p_team_a, errors="coerce")
    if p.isna().any() or (p <= 0).any() or (p >= 1).any():
        raise ValueError("p_team_a must be a probability strictly inside (0, 1)")
    if df.kickoff.isna().any() or df.generated_at.isna().any():
        raise ValueError("kickoff and generated_at are required on every row")
    if df.duplicated(["match_id", "market", "generated_at"]).any():
        raise ValueError("duplicate (match_id, market, generated_at) rows")
    return df[COLUMNS].assign(match_id=df.match_id.astype(int), p_team_a=p.astype(float),
                              kickoff=_utc(df.kickoff), generated_at=_utc(df.generated_at))


def load(path: str) -> pd.DataFrame:
    """A parquet path, or a DatasetStore name."""
    df = pd.read_parquet(path) if path.endswith(".parquet") or path.startswith("/") else DatasetStore().load(path)
    return validate(df)


class PredictionBook:
    """Per-market prediction history with as-of lookup."""

    def __init__(self, df: pd.DataFrame):
        self._rows: dict[tuple[int, str], tuple[list[int], list[dict]]] = {}
        for (match_id, market), g in df.sort_values("generated_at").groupby(["match_id", "market"]):
            self._rows[(int(match_id), market)] = (
                [t.value for t in g.generated_at],
                [{"p_team_a": float(r.p_team_a), "rank_known_both": r.rank_known_both,
                  "kickoff_ns": r.kickoff.value} for r in g.itertuples()],
            )

    def __contains__(self, key: tuple[int, str]) -> bool:
        return key in self._rows

    def as_of(self, match_id: int, market: str, now_ns: int) -> dict | None:
        """The latest prediction written at or before now_ns, or None if there is none yet."""
        hit = self._rows.get((match_id, market))
        if hit is None:
            return None
        i = bisect.bisect_right(hit[0], now_ns)
        return hit[1][i - 1] if i else None


def from_walk_forward(preds: pd.DataFrame, priors: pd.DataFrame, kickoffs: pd.Series,
                      model_version: str) -> pd.DataFrame:
    """
    Build the table from the six-month walk-forward predictions.

    Each was made by a model fitted only on earlier months, so a backtest on this
    table is out of sample in the same sense as the research backtest. They are
    stamped WALK_FORWARD_LEAD before kickoff - before any entry window opens -
    so as-of lookup sees them for the whole window. kickoffs: match_id -> UTC.
    """
    p = preds[preds.market.isin(MARKETS)].copy()
    rank = priors[priors.map_position_in_series == 1].drop_duplicates("match_id").set_index("match_id").rank_known_both
    p["rank_known_both"] = p.match_id.map(rank)
    p["p_team_a"] = p.q.clip(0.01, 0.99)
    p["kickoff"] = _utc(p.match_id.map(kickoffs))
    p = p[p.kickoff.notna()]
    p["model_version"] = model_version
    p["generated_at"] = p.kickoff - WALK_FORWARD_LEAD
    return validate(p)
