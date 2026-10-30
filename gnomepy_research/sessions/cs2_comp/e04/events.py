from __future__ import annotations

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.hltv.config import showmatch_mask
from gnomepy_research.sessions.cs2_comp.bundle import load

DAY_NS = 86400 * 10**9


def _utc_day(ts: pd.Series) -> np.ndarray:
    t = pd.to_datetime(ts, utc=True).dt.tz_convert(None)
    return (t.to_numpy().astype("datetime64[ns]").astype("int64") // DAY_NS).astype("int64")


def _hltv_ids(values) -> tuple[str, ...]:
    if values is None:
        return ()
    return tuple(f"h{int(v)}" for v in values if pd.notna(v))


def hltv_maps(data_dir: str, before: pd.Timestamp) -> pd.DataFrame:
    h = load(data_dir, "hltv/cs2_match_history", before)
    h = h[~showmatch_mask(h)]
    h = h[h.team_a_id.notna() & h.team_b_id.notna() & h.team_a_score.notna() & h.team_b_score.notna()]
    h = h[(h.team_a_score + h.team_b_score) > 0]
    time_col = h.match_time.where(h.match_time.notna(), h.match_date)
    event_day = np.maximum(_utc_day(h.match_date), _utc_day(time_col))
    return pd.DataFrame({
        "event_day": event_day,
        "match_key": h.match_id.astype(str).to_numpy(),
        "map_pos": h.map_position_in_series.fillna(0).astype("int64").to_numpy(),
        "faceit": False,
        "key_a": ["h" + str(int(x)) for x in h.team_a_id.to_numpy()],
        "key_b": ["h" + str(int(x)) for x in h.team_b_id.to_numpy()],
        "lineup_a": [_hltv_ids(x) for x in h.team_a_player_ids.to_numpy()],
        "lineup_b": [_hltv_ids(x) for x in h.team_b_player_ids.to_numpy()],
        "share_a": (h.team_a_score / (h.team_a_score + h.team_b_score)).to_numpy(),
    })


def announced_lineups(data_dir: str) -> pd.DataFrame:
    h = pd.read_parquet(f"{data_dir}/hltv/cs2_match_history.parquet",
                        columns=["match_id", "team_a_lineup_ids", "team_b_lineup_ids"])
    h = h.drop_duplicates("match_id")
    return pd.DataFrame({
        "match_id": h.match_id.to_numpy(),
        "lineup_a": [_hltv_ids(x) for x in h.team_a_lineup_ids.to_numpy()],
        "lineup_b": [_hltv_ids(x) for x in h.team_b_lineup_ids.to_numpy()],
    })


def faceit_maps(data_dir: str, before: pd.Timestamp) -> pd.DataFrame:
    fm = load(data_dir, "faceit/faceit_matches", before, ["match_id", "finished_at", "has_stats"])
    fm = fm[fm.has_stats.astype(bool)]
    exc = pd.read_parquet(f"{data_dir}/faceit/faceit_exclusions.parquet")
    whole = set(exc.loc[exc.map_name.isna(), "match_id"])
    link = pd.read_parquet(f"{data_dir}/faceit/faceit_hltv_series_link.parquet", columns=["faceit_match_id"])
    fm = fm[~fm.match_id.isin(whole | set(link.faceit_match_id))]

    hist = pd.read_parquet(f"{data_dir}/faceit/faceit_match_history.parquet", columns=[
        "match_id", "map_name", "map_position_in_series", "bo_type",
        "team_a_id", "team_b_id", "team_a_faceit_id", "team_b_faceit_id",
        "team_a_faceit_ids", "team_b_faceit_ids", "team_a_score", "team_b_score",
    ])
    hist = hist[hist.match_id.isin(fm.match_id)]
    map_out = exc[exc.map_name.notna()]
    if len(map_out):
        drop = pd.MultiIndex.from_frame(map_out[["match_id", "map_name"]])
        hist = hist[~pd.MultiIndex.from_frame(hist[["match_id", "map_name"]]).isin(drop)]
    hist = hist[hist.team_a_score.notna() & hist.team_b_score.notna() & ((hist.team_a_score + hist.team_b_score) > 0)]
    hist = hist[hist.team_a_faceit_ids.map(len).eq(5) & hist.team_b_faceit_ids.map(len).eq(5)]
    hist = hist.merge(fm[["match_id", "finished_at"]], on="match_id", how="inner")
    return hist


def faceit_map_table(data_dir: str, before: pd.Timestamp) -> pd.DataFrame:
    hist = faceit_maps(data_dir, before)
    hf = pd.read_parquet(f"{data_dir}/faceit/hltv_faceit_map.parquet", columns=["hltv_id", "faceit_id"])
    to_hltv = dict(zip(hf.faceit_id.astype(str), hf.hltv_id.astype("int64")))

    def team_key(team_id, faceit_id) -> str:
        if pd.notna(team_id):
            return f"h{int(team_id)}"
        return f"f{faceit_id}"

    def player_keys(ids) -> tuple[str, ...]:
        return tuple(f"h{to_hltv[p]}" if p in to_hltv else f"f{p}" for p in map(str, ids))

    return pd.DataFrame({
        "event_day": (hist.finished_at.astype("int64").to_numpy() // 86400),
        "match_key": hist.match_id.astype(str).to_numpy(),
        "map_pos": hist.map_position_in_series.fillna(0).astype("int64").to_numpy(),
        "faceit": True,
        "key_a": [team_key(a, b) for a, b in zip(hist.team_a_id.to_numpy(), hist.team_a_faceit_id.to_numpy())],
        "key_b": [team_key(a, b) for a, b in zip(hist.team_b_id.to_numpy(), hist.team_b_faceit_id.to_numpy())],
        "lineup_a": [player_keys(x) for x in hist.team_a_faceit_ids.to_numpy()],
        "lineup_b": [player_keys(x) for x in hist.team_b_faceit_ids.to_numpy()],
        "share_a": (hist.team_a_score / (hist.team_a_score + hist.team_b_score)).to_numpy(),
    })
