"""FACEIT matchmaking form of HLTV players: how often a linked player's side wins their 5v5 FACEIT games, and how many
they play. Pros sit at the top FACEIT level, where the level says little, but they play many games, so the win rate is
a cheap measurement of current individual form that neither the HLTV maps nor the organised FACEIT maps carry, and the
volume says how much the player is practising."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from gnomepy_research.sessions.cs2_comp.bundle import load


@dataclass(frozen=True)
class MMConfig:
    half_life_days: float = 120.0
    window_days: int = 240
    prior_games: float = 200.0


def mm_rows(data_dir, until: pd.Timestamp) -> pd.DataFrame:
    """One row per (linked HLTV player, finished 5v5 game before `until`): whether the player's side won. Only linked
    players' rows are read; their opponents' FACEIT levels barely move the expectation at the top level, where these
    players sit."""
    link = load(data_dir, "faceit/hltv_faceit_map")
    secs = int(pd.Timestamp(until).tz_localize("UTC").timestamp())
    hm = pq.read_table(f"{data_dir}/faceit/history_matches.parquet",
                       columns=["match_id", "teams_size", "winner", "finished_at", "score_faction1", "score_faction2",
                                "competition_type"],
                       filters=[("finished_at", "<", secs), ("teams_size", "=", 5)]).to_pandas()
    hm = hm[hm.winner.isin(["faction1", "faction2"])]
    hp = pq.read_table(f"{data_dir}/faceit/history_players.parquet", columns=["match_id", "faction", "faceit_id", "skill_level"],
                       filters=[("faceit_id", "in", list(link.faceit_id.astype(str).unique()))]).to_pandas()
    hp = hp.merge(hm[["match_id", "winner", "finished_at", "score_faction1", "score_faction2", "competition_type"]], on="match_id")
    s1, s2 = pd.to_numeric(hp.score_faction1, errors="coerce"), pd.to_numeric(hp.score_faction2, errors="coerce")
    mine = np.where(hp.faction == "faction1", s1, s2)
    total = (s1 + s2).to_numpy(float)
    ok = np.isfinite(total) & (total >= 13) & (total <= 60)
    f2h = dict(zip(link.faceit_id.astype(str), link.hltv_id.astype(np.int64)))
    return pd.DataFrame({"hltv_id": hp.faceit_id.map(f2h).to_numpy(np.int64),
                         "time": pd.to_datetime(hp.finished_at, unit="s").to_numpy(),
                         "win": (hp.winner == hp.faction).to_numpy(float),
                         "rounds": np.where(ok, total, np.nan), "rounds_won": np.where(ok, mine, np.nan),
                         "kind": hp.competition_type.to_numpy(), "level": hp.skill_level.to_numpy(float)}
                        ).sort_values("time", kind="stable").reset_index(drop=True)


def player_form(rows: pd.DataFrame, cutoff: pd.Timestamp, cfg: MMConfig, kind: str | None = None) -> pd.DataFrame:
    """Per player: shrunken win residual (logit units) and the recency-weighted game count, from games before cutoff
    (of one competition type when `kind` is given)."""
    r = rows[(rows.time < cutoff) & (rows.time >= cutoff - pd.Timedelta(days=cfg.window_days))]
    if kind is not None:
        r = r[r.kind == kind]
    p = np.full(len(r), 0.5)
    w = 0.5 ** ((cutoff - r.time).dt.total_seconds().to_numpy() / 86400.0 / cfg.half_life_days)
    g = pd.DataFrame({"hltv_id": r.hltv_id.to_numpy(), "g": w * (r.win.to_numpy() - p), "h": w * p * (1 - p), "n": w})
    ok = r.rounds.notna().to_numpy()
    n = np.where(ok, r.rounds.to_numpy(), 0.0)
    g["gr"] = np.where(ok, w * (r.rounds_won.to_numpy() - 0.5 * n), 0.0)
    g["hr"] = w * 0.25 * n
    g = g.groupby("hltv_id")[["g", "h", "n", "gr", "hr"]].sum()
    g["form"] = g.g / (g.h + cfg.prior_games * 0.25)
    g["activity"] = np.log1p(g.n)
    g["round_form"] = g.gr / (g.hr + cfg.prior_games * 0.25 * 22)
    g["net"] = g.g
    return g[["form", "activity", "round_form", "net"]]


def level_trend(rows: pd.DataFrame, cutoff: pd.Timestamp, lookback_days: int = 90) -> pd.DataFrame:
    """Per player: latest FACEIT level before cutoff and its change over the lookback."""
    r = rows[(rows.time < cutoff) & rows["level"].notna()]
    last = r.groupby("hltv_id")["level"].last()
    old = r[r.time < cutoff - pd.Timedelta(days=lookback_days)].groupby("hltv_id")["level"].last()
    return pd.DataFrame({"level": last, "level_change": (last - old.reindex(last.index)).fillna(0.0)})


def five_mean(form: pd.DataFrame, five, col: str) -> float:
    """Mean over the five, a player without matchmaking games counting as zero."""
    if five is None or len(five) == 0:
        return 0.0
    v = form[col]
    return float(np.mean([v.get(int(p), 0.0) for p in five]))
