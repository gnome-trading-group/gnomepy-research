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
