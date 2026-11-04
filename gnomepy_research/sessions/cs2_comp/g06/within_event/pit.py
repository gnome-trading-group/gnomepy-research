"""Point-in-time rule for HLTV series: a series counts as known only once it has surely finished.

HLTV rows carry the scheduled start (match_time), not the end. The bundle loader and e12's Inputs treat a map as known
from its start, which is harmless for a Monday 00:00 fit but not for a daily one. Measured on 5,591 HLTV series linked
to FACEIT (finished_at): bo1 ends within 3.0 h of HLTV's start (max), bo3 within 4.9 h at the 99.9th percentile (one
27 h reschedule), bo5 within 6.6 h. So a series is known from start + BUFFER_H[bo_type]."""
import pandas as pd

from gnomepy_research.sessions.cs2_comp import bundle
from gnomepy_research.sessions.cs2_comp.g06.e12 import data as e12_data
from gnomepy_research.sessions.cs2_comp.g06 import history

BUFFER_H = {1: 3.0, 3: 6.0, 5: 9.0}
_original = bundle.load


def known_at(match_time: pd.Series, bo_type: pd.Series) -> pd.Series:
    t = pd.to_datetime(match_time)
    t = t.dt.tz_convert(None) if t.dt.tz is not None else t
    return t + pd.to_timedelta(bo_type.map(BUFFER_H).fillna(max(BUFFER_H.values())), unit="h")


def load(data_dir, table, before=None, columns=None):
    if table != "hltv/cs2_match_history":
        return _original(data_dir, table, before, columns)
    cols = None if columns is None else list(dict.fromkeys(columns + ["match_time", "match_date", "bo_type"]))
    d = _original(data_dir, table, before, cols)
    end = known_at(d.match_time, d.bo_type)
    if before is not None:
        d = d[end < pd.Timestamp(before)]
        end = end[d.index]
    # e12 derives both `time` and `known` from match_time; moving it to the end makes `known` the end. The few hours'
    # shift in `time` (map-weight ageing, which five a team fielded last) is the same for every variant using the rule.
    d = d.assign(match_time=end.dt.tz_localize("UTC"))
    return d if columns is None else d[columns]


def install() -> None:
    """Applies the rule to every reader of the bundle's HLTV series."""
    e12_data.load = load
    history.load = load


# ---- the prod pipelines' schedule ---------------------------------------------------------------------------------
# hltv_cs2 scrapes results daily at 06:00 UTC and faceit_daily loads FACEIT at 07:30, so a fit run after both (08:30)
# sees an HLTV series only from the first 06:00 after it surely ended, and a FACEIT match from the first 07:30 after
# FACEIT finished it.
HLTV_SCRAPE_H = 6.0
FACEIT_LOAD_H = 7.5
FACEIT_TIMED = ("faceit/faceit_matches", "faceit/history_matches")


def next_daily(t: pd.Series, hour: float) -> pd.Series:
    """The first time of day `hour` at or after each t."""
    at = t.dt.normalize() + pd.Timedelta(hours=hour)
    return at.where(at >= t, at + pd.Timedelta(days=1))


def _scheduled_load(data_dir, table, before=None, columns=None):
    if table == "hltv/cs2_match_history":
        cols = None if columns is None else list(dict.fromkeys(columns + ["match_time", "match_date", "bo_type"]))
        d = _original(data_dir, table, None, cols)
        seen = next_daily(known_at(d.match_time, d.bo_type), HLTV_SCRAPE_H)
        if before is not None:
            d, seen = d[seen < pd.Timestamp(before)], seen[seen < pd.Timestamp(before)]
        d = d.assign(match_time=seen.dt.tz_localize("UTC"))
        return d if columns is None else d[columns]
    if table in FACEIT_TIMED:
        cols = None if columns is None else list(dict.fromkeys(columns + ["finished_at"]))
        d = _original(data_dir, table, None, cols)
        seen = next_daily(pd.to_datetime(d.finished_at, unit="s"), FACEIT_LOAD_H)
        d = d.assign(finished_at=(seen - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1))
        if before is not None:
            d = d[seen.to_numpy() < pd.Timestamp(before).to_datetime64()]
        return d if columns is None else d[columns]
    return _original(data_dir, table, before, columns)


def install_schedule() -> None:
    """Applies the pipelines' schedule to every reader: the bundle's own recursive cuts (FACEIT tables cut through
    faceit_matches) included."""
    from gnomepy_research.sessions.cs2_comp.g06 import mm_skill
    bundle.load = _scheduled_load
    e12_data.load = _scheduled_load
    history.load = _scheduled_load
    mm_skill.load = _scheduled_load
