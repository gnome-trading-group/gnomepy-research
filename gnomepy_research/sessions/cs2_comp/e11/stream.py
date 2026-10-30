"""
One time-ordered stream of maps from HLTV and (optionally) FACEIT, each tagged with the time it became available.

A map is available at a cutoff c when the bundle's own point-in-time rules would return it for c: an HLTV map when
both its match_date and match_time are before c, a FACEIT map when its series finished and its match_date is before
c. The stream is sorted by that availability time, so advancing a filter through "every map available before c"
visits the same maps in the same order whether the bundle was cut at c or later.

Players are keyed by HLTV id. A FACEIT player with an HLTV link takes that id; an unlinked one gets a negative id
derived from the FACEIT id, so FACEIT-only opponents still connect the rating graph.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.hltv.config import showmatch_mask
from gnomepy_research.sessions.cs2_comp import bundle

HLTV, FACEIT = 0, 1
EPOCH = np.datetime64("1970-01-01", "D")


def day_number(ts) -> int:
    return int((np.datetime64(pd.Timestamp(ts).to_datetime64(), "D") - EPOCH).astype(np.int64))


def faceit_key(faceit_id: str) -> int:
    return -int(faceit_id.replace("-", "")[:15], 16)


@dataclass
class MapStream:
    avail: np.ndarray        # (N,) datetime64[ns] availability time
    day: np.ndarray          # (N,) day number of the map
    source: np.ndarray       # (N,) HLTV or FACEIT
    team_a: np.ndarray       # (N,) HLTV team id, or -1 for a FACEIT team with no HLTV link
    team_b: np.ndarray
    players_a: list          # tuples of player keys that played
    players_b: list
    rounds_a: np.ndarray
    rounds_b: np.ndarray
    series_key: np.ndarray   # (N,) str, the series the map belongs to
    skill_a: np.ndarray      # (N,) mean FACEIT skill level of the five (nan for HLTV maps)
    skill_b: np.ndarray

    def __len__(self) -> int:
        return len(self.avail)

    def subset(self, keep: np.ndarray) -> MapStream:
        idx = np.flatnonzero(keep)
        return MapStream(
            avail=self.avail[idx], day=self.day[idx], source=self.source[idx], team_a=self.team_a[idx],
            team_b=self.team_b[idx], players_a=[self.players_a[i] for i in idx],
            players_b=[self.players_b[i] for i in idx],
            rounds_a=self.rounds_a[idx], rounds_b=self.rounds_b[idx], series_key=self.series_key[idx],
            skill_a=self.skill_a[idx], skill_b=self.skill_b[idx])


def _naive(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    return s.dt.tz_convert(None) if s.dt.tz is not None else s


def _ids(v) -> tuple[int, ...]:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ()
    return tuple(int(p) for p in v)


def hltv_maps(data: Path, before: pd.Timestamp) -> pd.DataFrame:
    h = bundle.load(data, "hltv/cs2_match_history", before)
    h = h[~showmatch_mask(h) & h.team_a_score.notna() & h.team_b_score.notna()
          & (h.team_a_score + h.team_b_score > 0)].copy()
    h["match_date"] = _naive(h.match_date)
    h["avail"] = np.maximum(h.match_date.to_numpy(), _naive(h.match_time.fillna(h.match_date)).to_numpy())
    return h


def faceit_maps(data: Path, before: pd.Timestamp) -> pd.DataFrame:
    matches = bundle.load(data, "faceit/faceit_matches", before, ["match_id", "finished_at", "has_stats"])
    matches = matches[matches.has_stats.fillna(False).astype(bool)]
    excl = bundle.load(data, "faceit/faceit_exclusions", before)
    whole = set(excl.match_id[excl.map_name.isna()])
    bad_maps = set(zip(excl.match_id[excl.map_name.notna()], excl.map_name[excl.map_name.notna()]))
    f = bundle.load(data, "faceit/faceit_match_history", before,
                    ["match_id", "match_date", "map_name", "map_position_in_series", "team_a_id", "team_b_id",
                     "team_a_faceit_ids", "team_b_faceit_ids", "team_a_score", "team_b_score", "team_a_players",
                     "team_b_players"])
    f = f[f.match_id.isin(matches.match_id) & ~f.match_id.isin(whole)]
    f = f[[(m, n) not in bad_maps for m, n in zip(f.match_id, f.map_name)]]
    f = f[(f.team_a_players == 5) & (f.team_b_players == 5) & (f.team_a_score + f.team_b_score > 0)].copy()
    fin = pd.to_datetime(matches.set_index("match_id").finished_at, unit="s")
    f["match_date"] = _naive(f.match_date)
    f["avail"] = np.maximum(f.match_date.to_numpy(), f.match_id.map(fin).to_numpy())
    return f


def faceit_skill(data: Path, before: pd.Timestamp) -> dict[tuple[str, str], int]:
    """FACEIT skill level of each (match, player) that played, known before the match starts."""
    r = bundle.load(data, "faceit/faceit_rosters", before, ["match_id", "faceit_id", "skill_level", "played"])
    r = r[r.played.fillna(False).astype(bool) & (r.skill_level > 0)]
    return dict(zip(zip(r.match_id, r.faceit_id), r.skill_level.astype(int)))


def build_stream(data: Path, before: pd.Timestamp, use_faceit: bool) -> MapStream:
    h = hltv_maps(data, before)
    parts = [pd.DataFrame({
        "avail": h.avail.to_numpy(), "day": h.match_date.to_numpy(), "source": HLTV,
        "team_a": h.team_a_id.astype(np.int64).to_numpy(), "team_b": h.team_b_id.astype(np.int64).to_numpy(),
        "players_a": [_ids(x) for x in h.team_a_player_ids], "players_b": [_ids(x) for x in h.team_b_player_ids],
        "rounds_a": h.team_a_score.astype(int).to_numpy(), "rounds_b": h.team_b_score.astype(int).to_numpy(),
        "series_key": h.match_id.astype(str).to_numpy(),
        "pos": h.map_position_in_series.fillna(0).astype(int).to_numpy(),
        "skill_a": np.nan, "skill_b": np.nan,
    })]
    if use_faceit:
        f = faceit_maps(data, before)
        link = dict(bundle.load(data, "faceit/hltv_faceit_map")[["faceit_id", "hltv_id"]].itertuples(index=False))
        skill = faceit_skill(data, before)

        def keys(ids) -> tuple[int, ...]:
            return tuple(int(link[x]) if x in link else faceit_key(x) for x in ids)

        def mean_skill(match_id, ids) -> float:
            s = [skill[(match_id, x)] for x in ids if (match_id, x) in skill]
            return float(np.mean(s)) if s else np.nan

        parts.append(pd.DataFrame({
            "avail": f.avail.to_numpy(), "day": f.match_date.to_numpy(), "source": FACEIT,
            "team_a": f.team_a_id.fillna(-1).astype(np.int64).to_numpy(),
            "team_b": f.team_b_id.fillna(-1).astype(np.int64).to_numpy(),
            "players_a": [keys(x) for x in f.team_a_faceit_ids], "players_b": [keys(x) for x in f.team_b_faceit_ids],
            "rounds_a": f.team_a_score.astype(int).to_numpy(), "rounds_b": f.team_b_score.astype(int).to_numpy(),
            "series_key": f.match_id.astype(str).to_numpy(),
            "pos": f.map_position_in_series.fillna(0).astype(int).to_numpy(),
            "skill_a": [mean_skill(m, x) for m, x in zip(f.match_id, f.team_a_faceit_ids)],
            "skill_b": [mean_skill(m, x) for m, x in zip(f.match_id, f.team_b_faceit_ids)],
        }))
    s = pd.concat(parts, ignore_index=True).sort_values(["avail", "source", "series_key", "pos"], kind="stable")
    return MapStream(
        avail=s.avail.to_numpy("datetime64[ns]"),
        day=(s.day.to_numpy("datetime64[D]") - EPOCH).astype(np.int64),
        source=s.source.to_numpy(), team_a=s.team_a.to_numpy(), team_b=s.team_b.to_numpy(),
        players_a=s.players_a.tolist(), players_b=s.players_b.tolist(),
        rounds_a=s.rounds_a.to_numpy(), rounds_b=s.rounds_b.to_numpy(), series_key=s.series_key.to_numpy(),
        skill_a=s.skill_a.to_numpy(float), skill_b=s.skill_b.to_numpy(float),
    )
