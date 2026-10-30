"""FACEIT player form read as of a timestamp: practice volume, recency, shrunk win rate, K/D, ADR and skill level.

A lineup's HLTV players are linked to FACEIT ids and their FACEIT matches are cut at the requested instant, so the
features only see matches that finished before it. Shrinkage toward a population prior keeps players with a few
matches from looking extreme.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

DAY = 86400
WIN_30, WIN_60, WIN_90 = 30 * DAY, 60 * DAY, 90 * DAY
RUSTY_GAP_DAYS = 14
NEWISH_MATCHES_90 = 10
WIN_PRIOR_MATCHES = 5
KD_PRIOR_DEATHS = 50.0
ADR_PRIOR_MAPS = 5
ADR_PRIOR = 70.0
MAX_GAP_DAYS = 365.0
STREAM_ROWS = 500_000

# Same organised-competition definition as the organiser's faceit/scope.py, copied so this package does not depend
# on the organiser's working folder.
ORGANISED = re.compile(
    r"ESEA|ECL|Challenger League|CCT|Qualifier|IEM|ESL|BLAST|PGL|Pro League|Stake|Thunderpick|Elisa|United21|Exort|"
    r"European Pro League|Major|RMR|Fragadelphia|Skyesports|DACH|Winline|Galaxy Battle|YGames|1XBET|Liquid|UKIC|UESF|"
    r"Conquest of Prague|^S\d+ (EU|NA|SA|SEA|OCE|MENA|AS|ASIA)\b.*(Central|Open|Main|Intermediate|Advanced|Entry)", re.I)

FEATURE_COLUMNS = ["n30", "n90", "gap_days", "form", "skill", "kd", "adr", "maps90"]


def is_organised(name) -> bool:
    name = name or ""
    return bool(ORGANISED.search(name)) and "retake" not in name.lower()


@dataclass(frozen=True)
class _Player:
    ev_t: np.ndarray
    cum_won: np.ndarray
    ev_skill: np.ndarray
    ms_t: np.ndarray
    cum_kills: np.ndarray
    cum_deaths: np.ndarray
    cum_adr: np.ndarray
    cum_adr_n: np.ndarray


_EMPTY = _Player(
    ev_t=np.array([], dtype=np.int64), cum_won=np.zeros(1), ev_skill=np.array([], dtype=float),
    ms_t=np.array([], dtype=np.int64), cum_kills=np.zeros(1), cum_deaths=np.zeros(1),
    cum_adr=np.zeros(1), cum_adr_n=np.zeros(1, dtype=np.int64),
)


def _cum(x: np.ndarray, dtype=float) -> np.ndarray:
    return np.concatenate([np.zeros(1, dtype=dtype), np.cumsum(x, dtype=dtype)])


def _numpy(a, dtype) -> np.ndarray:
    return pc.cast(a, pa.from_numpy_dtype(dtype)).to_numpy(zero_copy_only=False)


def _concat(parts: list[np.ndarray], dtype) -> np.ndarray:
    return np.concatenate(parts) if parts else np.array([], dtype=dtype)


@dataclass(frozen=True)
class Timelines:
    players: dict[str, _Player]
    hltv_to_faceit: dict[int, str]

    @classmethod
    def build(cls, data_dir: str | Path, max_cut: pd.Timestamp, hltv_to_faceit: dict[int, str]) -> "Timelines":
        data_dir = Path(data_dir)
        faceit_ids = sorted(set(hltv_to_faceit.values()))
        ids_arr = pa.array(faceit_ids, type=pa.string())
        cut_s = int(pd.Timestamp(max_cut).tz_localize("UTC").timestamp())

        hm = pq.read_table(data_dir / "faceit/history_matches.parquet",
                           columns=["match_id", "game_mode", "finished_at", "winner"],
                           filters=[("finished_at", "<", cut_s)]).combine_chunks()
        hm = hm.filter(pc.and_(pc.equal(hm["game_mode"], "5v5"),
                               pc.is_in(hm["winner"], value_set=pa.array(["faction1", "faction2"]))))
        ev_codes, ev_t, ev_won, ev_skill = _stream_history_players(
            data_dir / "faceit/history_players.parquet", ids_arr,
            hm["match_id"].combine_chunks(), hm["finished_at"].combine_chunks(), hm["winner"].combine_chunks())
        del hm

        fm = pq.read_table(data_dir / "faceit/faceit_matches.parquet",
                           columns=["match_id", "competition_name", "finished_at"],
                           filters=[("finished_at", "<", cut_s)]).combine_chunks()
        organised = [n for n in fm["competition_name"].unique().to_pylist() if is_organised(n)]
        fm = fm.filter(pc.is_in(fm["competition_name"], value_set=pa.array(organised, type=pa.string())))
        ms_codes, ms_t, ms_kills, ms_deaths, ms_adr = _stream_map_stats(
            data_dir / "faceit/faceit_player_map_stats.parquet", ids_arr,
            fm["match_id"].combine_chunks(), fm["finished_at"].combine_chunks())
        del fm

        n = len(faceit_ids)
        ev_by = _by_player(ev_codes, ev_t, {"won": ev_won, "skill": ev_skill}, n)
        ms_by = _by_player(ms_codes, ms_t, {"kills": ms_kills, "deaths": ms_deaths, "adr": ms_adr}, n)
        players = {faceit_ids[i]: _player(ev_by.get(i), ms_by.get(i)) for i in set(ev_by) | set(ms_by)}
        return cls(players=players, hltv_to_faceit=hltv_to_faceit)


def _stream_history_players(path: Path, ids_arr: pa.Array, hm_ids: pa.Array, hm_finished: pa.Array,
                            hm_winner: pa.Array):
    codes_parts, t_parts, won_parts, skill_parts = [], [], [], []
    for batch in pq.ParquetFile(path).iter_batches(
            batch_size=STREAM_ROWS, columns=["match_id", "faction", "faceit_id", "skill_level"]):
        codes = pc.index_in(batch["faceit_id"], value_set=ids_arr)
        keep = pc.is_valid(codes)
        sub = batch.filter(keep)
        pos = pc.index_in(sub["match_id"], value_set=hm_ids)
        ok = pc.is_valid(pos)
        pos = pc.filter(pos, ok)
        codes_parts.append(_numpy(pc.filter(pc.filter(codes, keep), ok), np.int64))
        t_parts.append(_numpy(pc.take(hm_finished, pos), np.int64))
        won_parts.append(_numpy(pc.equal(pc.take(hm_winner, pos), pc.filter(sub["faction"], ok)), float))
        skill_parts.append(_numpy(pc.filter(sub["skill_level"], ok), float))
    return (_concat(codes_parts, np.int64), _concat(t_parts, np.int64),
            _concat(won_parts, float), _concat(skill_parts, float))


def _stream_map_stats(path: Path, ids_arr: pa.Array, fm_ids: pa.Array, fm_finished: pa.Array):
    codes_parts, t_parts, kills_parts, deaths_parts, adr_parts = [], [], [], [], []
    for batch in pq.ParquetFile(path).iter_batches(
            batch_size=STREAM_ROWS, columns=["match_id", "faceit_id", "kills", "deaths", "adr"]):
        codes = pc.index_in(batch["faceit_id"], value_set=ids_arr)
        keep = pc.is_valid(codes)
        sub = batch.filter(keep)
        pos = pc.index_in(sub["match_id"], value_set=fm_ids)
        ok = pc.is_valid(pos)
        pos = pc.filter(pos, ok)
        codes_parts.append(_numpy(pc.filter(pc.filter(codes, keep), ok), np.int64))
        t_parts.append(_numpy(pc.take(fm_finished, pos), np.int64))
        kills_parts.append(_numpy(pc.filter(sub["kills"], ok), float))
        deaths_parts.append(_numpy(pc.filter(sub["deaths"], ok), float))
        adr_parts.append(_numpy(pc.filter(sub["adr"], ok), float))
    return (_concat(codes_parts, np.int64), _concat(t_parts, np.int64), _concat(kills_parts, float),
            _concat(deaths_parts, float), _concat(adr_parts, float))


def _by_player(codes: np.ndarray, t: np.ndarray, values: dict[str, np.ndarray], n_players: int) -> dict[int, dict]:
    order = np.lexsort((t, codes))
    bounds = np.searchsorted(codes[order], np.arange(n_players + 1))
    t_sorted = t[order]
    sorted_values = {k: v[order] for k, v in values.items()}
    out: dict[int, dict] = {}
    for i in range(n_players):
        lo, hi = bounds[i], bounds[i + 1]
        if hi > lo:
            out[i] = {"t": t_sorted[lo:hi], **{k: v[lo:hi] for k, v in sorted_values.items()}}
    return out


def _player(ev: dict | None, ms: dict | None) -> _Player:
    p = _EMPTY
    if ev is not None:
        p = replace(p, ev_t=ev["t"], cum_won=_cum(ev["won"]), ev_skill=ev["skill"])
    if ms is not None:
        adr = ms["adr"]
        p = replace(p, ms_t=ms["t"], cum_kills=_cum(ms["kills"]), cum_deaths=_cum(ms["deaths"]),
                    cum_adr=_cum(np.nan_to_num(adr)), cum_adr_n=_cum(np.isfinite(adr).astype(np.int64), np.int64))
    return p


def _player_block(p: _Player, t: np.ndarray) -> np.ndarray:
    out = np.full((len(t), len(FEATURE_COLUMNS)), np.nan)
    hi = np.searchsorted(p.ev_t, t, side="left")
    lo30 = np.searchsorted(p.ev_t, t - WIN_30, side="left")
    lo60 = np.searchsorted(p.ev_t, t - WIN_60, side="left")
    lo90 = np.searchsorted(p.ev_t, t - WIN_90, side="left")
    n30 = hi - lo30
    n90 = hi - lo90
    won60 = p.cum_won[hi] - p.cum_won[lo60]
    n60 = hi - lo60
    if len(p.ev_t):
        last = np.clip(hi - 1, 0, len(p.ev_t) - 1)
        gap = np.where(hi > 0, (t - p.ev_t[last]) / DAY, MAX_GAP_DAYS)
        skill = np.where(hi > 0, p.ev_skill[last], np.nan)
    else:
        gap = np.full(len(t), MAX_GAP_DAYS)
        skill = np.full(len(t), np.nan)

    mhi = np.searchsorted(p.ms_t, t, side="left")
    mlo = np.searchsorted(p.ms_t, t - WIN_90, side="left")
    kills = p.cum_kills[mhi] - p.cum_kills[mlo]
    deaths = p.cum_deaths[mhi] - p.cum_deaths[mlo]
    adr_sum = p.cum_adr[mhi] - p.cum_adr[mlo]
    adr_n = p.cum_adr_n[mhi] - p.cum_adr_n[mlo]

    out[:, 0] = n30
    out[:, 1] = n90
    out[:, 2] = np.minimum(gap, MAX_GAP_DAYS)
    out[:, 3] = (won60 + WIN_PRIOR_MATCHES / 2) / (n60 + WIN_PRIOR_MATCHES) - 0.5
    out[:, 4] = skill
    out[:, 5] = (kills + KD_PRIOR_DEATHS) / (deaths + KD_PRIOR_DEATHS)
    out[:, 6] = (adr_sum + ADR_PRIOR_MAPS * ADR_PRIOR) / (adr_n + ADR_PRIOR_MAPS)
    out[:, 7] = mhi - mlo
    return out


def player_features(tl: Timelines, faceit_ids: np.ndarray, asof_s: np.ndarray) -> np.ndarray:
    out = np.full((len(faceit_ids), len(FEATURE_COLUMNS)), np.nan)
    if len(faceit_ids) == 0:
        return out
    positions = pd.Series(np.arange(len(faceit_ids))).groupby(faceit_ids).indices
    for pid, idx in positions.items():
        out[idx] = _player_block(tl.players.get(pid, _EMPTY), asof_s[idx])
    return out


def team_features(tl: Timelines, lineups: pd.DataFrame, keys: np.ndarray) -> pd.DataFrame:
    """One row per key in keys; lineups has columns key, hltv_id, asof_s (int64 seconds).

    Keys with no linked player come back as NaN rows, so the caller can tell "no FACEIT coverage" from "not asked".
    """
    lk = lineups.assign(faceit_id=lineups["hltv_id"].map(tl.hltv_to_faceit))
    linked = lk[lk["faceit_id"].notna()].reset_index(drop=True)
    feats = player_features(tl, linked["faceit_id"].to_numpy(object), linked["asof_s"].to_numpy(np.int64))
    frame = pd.DataFrame(feats, columns=FEATURE_COLUMNS)
    per_player = pd.DataFrame({
        "key": linked["key"].to_numpy(),
        "vol30": np.log1p(frame["n30"]),
        "vol90": np.log1p(frame["n90"]),
        "rusty": (frame["gap_days"] > RUSTY_GAP_DAYS).astype(float),
        "newish": (frame["n90"] < NEWISH_MATCHES_90).astype(float),
        "form": frame["form"],
        "kd": np.log(frame["kd"]),
        "adr": frame["adr"] - ADR_PRIOR,
        "skill": frame["skill"],
    })
    agg = per_player.groupby("key").mean()
    agg.insert(0, "n_link", per_player.groupby("key").size())
    return agg.reindex(keys)
