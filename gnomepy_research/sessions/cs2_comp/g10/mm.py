"""FACEIT matchmaking form of HLTV players: how often a linked player's side wins their 5v5 matchmaking games beyond
what the two sides' FACEIT skill levels imply. Pros sit at the top level, where the level says little, but they play
many matchmaking games, so the win residual is a cheap, independent measurement of current individual form."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scipy.special import expit

from gnomepy_research.sessions.cs2_comp.bundle import load


@dataclass(frozen=True)
class MMConfig:
    half_life_days: float = 60.0
    window_days: int = 240
    level_slope: float = 0.25
    prior_games: float = 40.0


def mm_rows(data_dir, until: pd.Timestamp, verbose: bool = False) -> pd.DataFrame:
    """One row per (linked HLTV player, finished 5v5 matchmaking game before `until`)."""
    hm = load(data_dir, "faceit/history_matches", until,
              ["match_id", "game_mode", "match_type", "teams_size", "status", "winner", "finished_at", "competition_type"])
    if verbose:
        for c in ["game_mode", "match_type", "teams_size", "status", "winner", "competition_type"]:
            print(c, hm[c].value_counts().head(6).to_dict(), flush=True)
    hm = hm[(hm.teams_size == 5) & hm.winner.isin(["faction1", "faction2"])]
    link = load(data_dir, "faceit/hltv_faceit_map")
    secs = int(pd.Timestamp(until).tz_localize("UTC").timestamp())
    ids = pq.read_table(f"{data_dir}/faceit/history_matches.parquet", columns=["match_id"],
                        filters=[("finished_at", "<", secs)]).column("match_id")
    hp = pq.read_table(f"{data_dir}/faceit/history_players.parquet", columns=["match_id", "faction", "faceit_id", "skill_level"])
    hp = hp.filter(pc.is_in(hp["match_id"], value_set=ids))
    lvl = hp.group_by(["match_id", "faction"]).aggregate([("skill_level", "mean")]).to_pandas()
    mine = hp.filter(pc.is_in(hp["faceit_id"], value_set=pa.array(link.faceit_id.astype(str).unique()))).to_pandas()
    if verbose:
        print("history_players", hp.num_rows, "linked rows", len(mine), "factions", mine.faction.value_counts().head().to_dict(), flush=True)
    lvl = lvl.rename(columns={"skill_level_mean": "lvl"})
    mine = mine.merge(hm[["match_id", "winner", "finished_at"]], on="match_id")
    mine = mine.merge(lvl.rename(columns={"lvl": "lvl_own"}), on=["match_id", "faction"])
    opp = lvl.assign(faction=np.where(lvl.faction == "faction1", "faction2", "faction1")).rename(columns={"lvl": "lvl_opp"})
    mine = mine.merge(opp, on=["match_id", "faction"])
    f2h = dict(zip(link.faceit_id.astype(str), link.hltv_id.astype(np.int64)))
    return pd.DataFrame({"hltv_id": mine.faceit_id.map(f2h).to_numpy(np.int64),
                         "time": pd.to_datetime(mine.finished_at, unit="s").to_numpy(),
                         "win": (mine.winner == mine.faction).to_numpy(float),
                         "lvl_diff": (mine.lvl_own.fillna(10) - mine.lvl_opp.fillna(10)).to_numpy(float)}).sort_values("time")


def player_form(rows: pd.DataFrame, cutoff: pd.Timestamp, cfg: MMConfig) -> pd.DataFrame:
    """Per player: shrunken win residual (logit units) and the recency-weighted game count, from games before cutoff."""
    r = rows[(rows.time < cutoff) & (rows.time >= cutoff - pd.Timedelta(days=cfg.window_days))]
    p = expit(cfg.level_slope * r.lvl_diff.to_numpy())
    w = 0.5 ** ((cutoff - r.time).dt.total_seconds().to_numpy() / 86400.0 / cfg.half_life_days)
    g = pd.DataFrame({"hltv_id": r.hltv_id.to_numpy(), "g": w * (r.win.to_numpy() - p), "h": w * p * (1 - p), "n": w})
    g = g.groupby("hltv_id")[["g", "h", "n"]].sum()
    g["form"] = g.g / (g.h + cfg.prior_games * 0.25)
    return g[["form", "n"]]


def five_form(form: pd.DataFrame, five) -> tuple[float, int]:
    if five is None or len(five) == 0:
        return 0.0, 0
    vals = [form.form.get(int(p), np.nan) for p in five]
    known = [v for v in vals if not np.isnan(v)]
    return (float(np.sum(known)) / len(five) if known else 0.0), len(known)
