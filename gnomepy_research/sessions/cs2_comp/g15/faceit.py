"""
FACEIT maps on the HLTV player id space, for the batch skill fit.

A FACEIT player linked through `hltv_faceit_map` takes the HLTV id; an unlinked one keeps a negative hash of the
FACEIT id, so FACEIT-only opponents still connect the rating graph. Series also on HLTV (`faceit_hltv_series_link`)
are dropped so each counts once; exclusions and non-5v5 maps are dropped as the cleaning rules say. A map is known
once FACEIT finalised its match (`finished_at`), which can be weeks after it was played.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.bundle import load
from gnomepy_research.sessions.cs2_comp.g15.data import FIVE, naive


def faceit_uid(faceit_id: str) -> int:
    return -int(hashlib.blake2b(faceit_id.encode(), digest_size=7).hexdigest(), 16)


class FaceitLink:
    def __init__(self, data_dir: str | Path):
        link = load(data_dir, "faceit/hltv_faceit_map")
        self.f2h = dict(zip(link.faceit_id, link.hltv_id.astype(np.int64)))
        self.h2f = {h: f for f, h in self.f2h.items()}

    def uid(self, faceit_id: str) -> int:
        h = self.f2h.get(faceit_id)
        return int(h) if h is not None else faceit_uid(faceit_id)

    def five(self, ids) -> tuple[int, ...] | None:
        if ids is None or isinstance(ids, float) or len(ids) != FIVE:
            return None
        out = tuple(self.uid(x) for x in ids)
        return out if len(set(out)) == FIVE else None


def load_faceit_maps(data_dir: str | Path, until: pd.Timestamp, link: FaceitLink) -> pd.DataFrame:
    """Decided 5v5 FACEIT maps known before `until`, shaped like `data.load_maps` plus `known` and `competition`."""
    fm = load(data_dir, "faceit/faceit_matches", until,
              ["match_id", "finished_at", "has_stats", "competition_name", "faction1_hltv_team", "faction2_hltv_team"])
    fm = fm[fm.has_stats.fillna(False).astype(bool)]
    fm["known"] = pd.to_datetime(fm.finished_at, unit="s")
    linked = load(data_dir, "faceit/faceit_hltv_series_link", until, ["faceit_match_id", "hltv_match_id"])
    fm = fm[~fm.match_id.isin(set(linked.faceit_match_id))]
    ex = load(data_dir, "faceit/faceit_exclusions", until)
    whole = set(ex.match_id[ex.map_name.isna()])
    bad_maps = set(zip(ex.match_id[ex.map_name.notna()], ex.map_name[ex.map_name.notna()]))
    cols = ["match_id", "map_name", "match_time", "match_date", "team_a_score", "team_b_score", "team_a_ot_score", "team_b_ot_score",
            "team_a_faceit_ids", "team_b_faceit_ids", "team_a_players", "team_b_players", "map_position_in_series"]
    fh = load(data_dir, "faceit/faceit_match_history", until, cols)
    fh = fh.merge(fm[["match_id", "known", "competition_name", "faction1_hltv_team", "faction2_hltv_team"]], on="match_id")
    fh = fh[~fh.match_id.isin(whole)]
    fh = fh[[(m, n) not in bad_maps for m, n in zip(fh.match_id, fh.map_name)]]
    fh = fh[(fh.team_a_players == FIVE) & (fh.team_b_players == FIVE) & (fh.team_a_score != fh.team_b_score)]
    fh = fh[np.maximum(fh.team_a_score, fh.team_b_score) >= 13]
    fh["five_a"] = [link.five(v) for v in fh.team_a_faceit_ids]
    fh["five_b"] = [link.five(v) for v in fh.team_b_faceit_ids]
    fh = fh[fh.five_a.notna() & fh.five_b.notna()].copy()
    ot = (fh.team_a_ot_score.fillna(0) + fh.team_b_ot_score.fillna(0) > 0) | ((fh.team_a_score + fh.team_b_score) > 24)
    a_won = fh.team_a_score > fh.team_b_score
    fh["rounds_a"] = np.where(ot, np.where(a_won, 13, 12), fh.team_a_score).astype(int)
    fh["rounds_b"] = np.where(ot, np.where(a_won, 12, 13), fh.team_b_score).astype(int)
    fh["time"] = naive(fh.match_time.fillna(fh.match_date))
    fh["known"] = np.maximum(fh.known, fh.time)
    out = pd.DataFrame({"match_id": fh.match_id.to_numpy(), "time": fh.time.to_numpy(), "known": fh.known.to_numpy(),
                        "team_a": fh.faction1_hltv_team.fillna(-1).astype(np.int64).to_numpy(),
                        "team_b": fh.faction2_hltv_team.fillna(-1).astype(np.int64).to_numpy(),
                        "five_a": fh.five_a.to_numpy(), "five_b": fh.five_b.to_numpy(),
                        "rounds_a": fh.rounds_a.to_numpy(), "rounds_b": fh.rounds_b.to_numpy(),
                        "competition": fh.competition_name.to_numpy()})
    return out.sort_values(["time", "match_id"], kind="stable").reset_index(drop=True)
