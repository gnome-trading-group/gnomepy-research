"""
FACEIT matchmaking edge: how much more a linked player wins pick-up games than the skill levels in the lobby predict.
Pros sit at the top level, so lobbies of equals say nothing through the level; who keeps winning them does.

    python -m gnomepy_research.sessions.cs2_comp.g09.pug [--data /work/in/comp] [--cutoff 2026-08-01]

pug_features(data_dir, cutoff, rows) returns per row the A-minus-B mean edge of the two fives as of each row's snap.
The main entry writes the full feature rows with the edge for offline evaluation.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.bundle import load
from gnomepy_research.sessions.cs2_comp.g09.core import build_rows
from gnomepy_research.sessions.cs2_comp.g09.predict import EXTRA_ELO, MAIN_ELO

WINDOW_DAYS = 180.0
SHRINK = 30.0
LEVEL_SLOPE = 0.04


def player_edges(data_dir: str | Path, cutoff: pd.Timestamp) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """HLTV id -> (finish times, residuals) of the player's 5v5 matchmaking games known before cutoff."""
    link = load(data_dir, "faceit/hltv_faceit_map")
    to_hltv = dict(zip(link.faceit_id, link.hltv_id.astype(int)))
    hm = load(data_dir, "faceit/history_matches", cutoff,
              ["match_id", "competition_type", "teams_size", "status", "winner", "finished_at"])
    hm = hm[(hm.teams_size == 5) & hm.winner.notna() & (hm.competition_type.astype(str).str.lower() == "matchmaking")]
    hp = load(data_dir, "faceit/history_players", cutoff, ["match_id", "faction", "faceit_id", "skill_level"])
    hp = hp[hp.match_id.isin(hm.match_id)]
    lv = hp.groupby(["match_id", "faction"]).skill_level.mean().unstack()
    focus = hp[hp.faceit_id.isin(to_hltv)].merge(hm[["match_id", "winner", "finished_at"]], on="match_id")
    f1, f2 = lv.columns[:2]
    lvl = lv.reindex(focus.match_id)
    first = (focus.faction == f1).to_numpy()
    own = np.where(first, lvl[f1].to_numpy(float), lvl[f2].to_numpy(float))
    opp = np.where(first, lvl[f2].to_numpy(float), lvl[f1].to_numpy(float))
    won = (focus.winner.astype(str) == focus.faction.astype(str)).to_numpy(float)
    exp = np.clip(0.5 + LEVEL_SLOPE * np.nan_to_num(own - opp), 0.05, 0.95)
    focus = focus.assign(res=won - exp, t=pd.to_datetime(focus.finished_at, unit="s"), hltv=focus.faceit_id.map(to_hltv))
    focus = focus.sort_values("t", kind="stable")
    return {int(p): (g.t.to_numpy(), g.res.to_numpy()) for p, g in focus.groupby("hltv")}


def pug_features(edges: dict, rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for r in rows.itertuples(index=False):
        snap = np.datetime64(r.snap)
        lo = snap - np.timedelta64(int(WINDOW_DAYS), "D")
        side = []
        for five in (r.lineup_a, r.lineup_b):
            vals = []
            for p in five:
                if p in edges:
                    t, res = edges[p]
                    i, j = np.searchsorted(t, lo), np.searchsorted(t, snap)
                    vals.append(res[i:j].sum() / (j - i + SHRINK))
                else:
                    vals.append(0.0)
            side.append(float(np.mean(vals)) if vals else 0.0)
        out.append({"key": r.key, "pug_a": side[0], "pug_b": side[1], "pug": side[0] - side[1]})
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/work/in/comp")
    ap.add_argument("--cutoff", default="2026-08-01")
    a = ap.parse_args()
    cutoff = pd.Timestamp(a.cutoff)
    hm = load(a.data, "faceit/history_matches", cutoff, ["competition_type", "teams_size", "status", "game_mode", "match_type"])
    for c in hm.columns:
        print(hm[c].value_counts(dropna=False).head(8), flush=True)
    t = pd.read_parquet(Path(a.data) / "targets_tune.parquet").iloc[:0]
    train, _ = build_rows(a.data, cutoff, t, elo_cfg=MAIN_ELO, extra_elo=EXTRA_ELO)
    edges = player_edges(a.data, cutoff)
    print("players with matchmaking history", len(edges), flush=True)
    f = pug_features(edges, train)
    train = train.merge(f, on="key")
    print(train[["pug_a", "pug_b", "pug"]].describe(), flush=True)
    train.drop(columns=["lineup_a", "lineup_b"]).to_parquet(Path(os.environ.get("OUTPUT_DIR", ".")) / "rows_pug.parquet", index=False)


if __name__ == "__main__":
    main()
