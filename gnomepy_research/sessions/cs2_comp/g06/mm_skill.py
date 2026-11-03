"""
FACEIT matchmaking skill: a player-skill fit on the FACEIT matchmaking, hub and championship games of linked players
and everyone they played with or against. Pros play thousands of these games, which no competitive-map fit sees, and a
fit (unlike a raw win rate) credits a win by the strength of the opponents. Championship games overlap e12's FACEIT
maps but still helped on 2025 outcomes.

    logit P(faction1 wins) = mean(s over faction1) - mean(s over faction2),   s_p ~ N(0, 1 / RIDGE)

each game weighted by 0.5^(age / HALF_LIFE_DAYS) over the WINDOW_DAYS before the cutoff.

    python -m gnomepy_research.sessions.cs2_comp.g06.mm_skill --data <bundle> --cutoffs 2025-01-06,... --out <dir>
writes each cutoff's linked-player skills, for development.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import scipy.sparse as sp
from scipy.optimize import minimize
from scipy.special import expit

from gnomepy_research.sessions.cs2_comp.bundle import load

HALF_LIFE_DAYS = 90.0
WINDOW_DAYS = 365
RIDGE = 0.5
TEAM = 5
COMPETITION_TYPES = ("matchmaking", "hub", "championship")


class MatchmakingGames:
    """Five-a-side matchmaking and hub games known before `before`, players as integer codes."""

    def __init__(self, data_dir: str | Path, before: pd.Timestamp, competition_types: tuple[str, ...] = COMPETITION_TYPES):
        link = load(data_dir, "faceit/hltv_faceit_map")
        m = load(data_dir, "faceit/history_matches", before, ["match_id", "competition_type", "finished_at", "winner"])
        m = m[m.competition_type.isin(competition_types) & m.winner.isin(["faction1", "faction2"])]
        # history_players is cut through its matches, as bundle.load does, without materialising every id string.
        t = pq.read_table(Path(data_dir) / "faceit/history_players.parquet", columns=["match_id", "faction", "faceit_id"],
                          read_dictionary=["match_id", "faction", "faceit_id"])
        mid = t["match_id"].unify_dictionaries().combine_chunks()
        code_in_m = pc.index_in(mid.dictionary, value_set=pa.array(m.match_id.to_numpy())).to_numpy(zero_copy_only=False)
        code_in_m = np.where(np.isnan(code_in_m.astype(float)), -1, code_in_m).astype(np.int64)
        match_codes = code_in_m[mid.indices.to_numpy()]
        ids = t["faceit_id"].unify_dictionaries().combine_chunks()
        player = ids.indices.to_numpy().astype(np.int64)
        names = ids.dictionary.to_pylist()
        fac = t["faction"].unify_dictionaries().combine_chunks()
        side = (np.asarray(fac.dictionary.to_pylist()) == "faction1")[fac.indices.to_numpy()]
        known = match_codes >= 0
        match_codes, player, side = match_codes[known], player[known], side[known]
        del t, mid, ids, fac
        counts = np.bincount(match_codes * 2 + (~side), minlength=2 * len(m)).reshape(-1, 2)
        full = (counts == TEAM).all(axis=1)
        keep = full[match_codes]
        self.match = match_codes[keep]
        self.player = player[keep]
        self.side = np.where(side[keep], 1.0, -1.0)
        self.time = pd.to_datetime(m.finished_at.to_numpy(), unit="s").to_numpy()
        self.y = (m.winner.to_numpy() == "faction1").astype(float)
        code_of = {f: i for i, f in enumerate(names)}
        self.linked = {int(h): code_of[f] for f, h in zip(link.faceit_id, link.hltv_id) if f in code_of}

    def fit(self, cutoff: pd.Timestamp, half_life_days: float = HALF_LIFE_DAYS, window_days: int = WINDOW_DAYS,
            ridge: float = RIDGE) -> dict[int, float]:
        """HLTV player id -> matchmaking skill from games before the cutoff (linked players who played any)."""
        c = np.datetime64(cutoff)
        in_window = (self.time < c) & (self.time >= c - np.timedelta64(window_days, "D"))
        rows = in_window[self.match]
        games = np.flatnonzero(in_window)
        g_index = np.full(len(self.time), -1)
        g_index[games] = np.arange(len(games))
        r = g_index[self.match[rows]]
        players, col = np.unique(self.player[rows], return_inverse=True)
        A = sp.csr_matrix((self.side[rows] / TEAM, (r, col)), shape=(len(games), len(players)))
        At = A.T.tocsr()
        y = self.y[games]
        w = 0.5 ** (((c - self.time[games]) / np.timedelta64(1, "D")) / half_life_days)

        def objective(s: np.ndarray) -> tuple[float, np.ndarray]:
            d = A @ s
            nll = -(w * (y * d - np.logaddexp(0.0, d))).sum() + 0.5 * ridge * (s @ s)
            return nll, -(At @ (w * (y - expit(d)))) + ridge * s

        s = minimize(objective, np.zeros(len(players)), jac=True, method="L-BFGS-B", options={"maxiter": 500}).x
        out = {}
        for h, code in self.linked.items():
            i = np.searchsorted(players, code)
            if i < len(players) and players[i] == code:
                out[h] = float(s[i])
        return out


def five_diff(skills: dict[int, float], five_a, five_b) -> float:
    """Mean skill of five A minus five B; players without matchmaking games count as zero."""
    def mean(five) -> float:
        five = [] if five is None else list(five)
        return float(np.mean([skills.get(int(p), 0.0) for p in five])) if five else 0.0
    return mean(five_a) - mean(five_b)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--cutoffs", required=True)
    ap.add_argument("--out", default=os.environ.get("OUTPUT_DIR", "."))
    ap.add_argument("--half-life", type=float, default=HALF_LIFE_DAYS)
    ap.add_argument("--window", type=int, default=WINDOW_DAYS)
    ap.add_argument("--ridge", type=float, default=RIDGE)
    ap.add_argument("--types", default=",".join(COMPETITION_TYPES))
    a = ap.parse_args()
    cutoffs = pd.to_datetime(a.cutoffs.split(","))
    games = MatchmakingGames(a.data, cutoffs.max(), tuple(a.types.split(",")))
    rows = []
    for c in cutoffs:
        skills = games.fit(c, a.half_life, a.window, a.ridge)
        rows += [{"cutoff": c, "player_id": p, "skill": v} for p, v in skills.items()]
        print(f"{c:%Y-%m-%d} {len(skills)} linked players", flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(Path(a.out) / f"mm_skill_{cutoffs[0]:%Y%m%d}.parquet", index=False)


if __name__ == "__main__":
    main()
