"""One lineup rating over HLTV and FACEIT maps, refit at each cutoff.

Each map is a logistic comparison of the two five-player lineups: a player's skill enters with weight 1/5, and the
map's log-odds is the difference of the two lineup sums. HLTV and FACEIT maps share one player scale because linked
FACEIT players carry their HLTV id; FACEIT maps are down-weighted, since FACEIT is a noisier, lower-tier signal.
Unlinked FACEIT players keep their own id so that their results still connect the rating graph.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression

from gnomepy_research.models.cs2.hltv.config import showmatch_mask
from gnomepy_research.sessions.cs2_comp.bundle import load

MIN_LINEUP = 4
HLTV, FACEIT = 0, 1


@dataclass(frozen=True)
class Config:
    faceit: bool = True
    faceit_weight: float = 0.3
    half_life_days: float = 365.0
    window_days: int = 730
    lineup_days: int = 120
    C: float = 1.0
    dev_scale: float = 0.0
    team_scale: float = 0.0
    round_weighted: bool = False


def naive(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_convert(None) if ts.tzinfo is not None else ts


def _hltv_maps(data: str, before: pd.Timestamp) -> pd.DataFrame:
    cols = ["match_id", "match_date", "match_time", "team_a_id", "team_b_id", "team_a_player_ids",
            "team_b_player_ids", "team_a_won", "team_a_score", "team_b_score", "event_name", "format"]
    h = load(data, "hltv/cs2_match_history", before, cols)
    h = h[~showmatch_mask(h)].dropna(subset=["team_a_won", "team_a_player_ids", "team_b_player_ids", "team_a_score",
                                             "team_b_score"])
    a = [tuple(str(int(x)) for x in ids if pd.notna(x)) for ids in h.team_a_player_ids]
    b = [tuple(str(int(x)) for x in ids if pd.notna(x)) for ids in h.team_b_player_ids]
    out = pd.DataFrame({
        "match_id": h.match_id.to_numpy(),
        "date": pd.to_datetime(h.match_date).to_numpy(),
        "src": HLTV,
        "a": a,
        "b": b,
        "y": h.team_a_won.astype(int).to_numpy(),
        "ra": h.team_a_score.astype(float).to_numpy(),
        "rb": h.team_b_score.astype(float).to_numpy(),
        "ta": [str(int(x)) if pd.notna(x) else "" for x in h.team_a_id],
        "tb": [str(int(x)) if pd.notna(x) else "" for x in h.team_b_id],
    })
    return out[(out.a.map(len) >= MIN_LINEUP) & (out.b.map(len) >= MIN_LINEUP)]


def _faceit_series(data: str, before: pd.Timestamp) -> pd.Index:
    fm = load(data, "faceit/faceit_matches", before, ["match_id", "has_stats"])
    return fm[fm.has_stats.astype(bool)].match_id


def _faceit_maps(data: str, before: pd.Timestamp) -> pd.DataFrame:
    ids = _faceit_series(data, before)
    cols = ["match_id", "match_date", "map_name", "map_position_in_series", "team_a_faceit_ids", "team_b_faceit_ids",
            "team_a_id", "team_b_id", "team_a_faceit_id", "team_b_faceit_id", "team_a_won", "team_a_score",
            "team_b_score"]
    fh = load(data, "faceit/faceit_match_history", before, cols)
    fh = fh[fh.match_id.isin(ids)].drop_duplicates(["match_id", "map_position_in_series"])

    ex = load(data, "faceit/faceit_exclusions", before, ["match_id", "map_name"])
    whole = ex.match_id[ex.map_name.isna()]
    fh = fh[~fh.match_id.isin(whole)]
    per_map = ex.dropna(subset=["map_name"]).assign(bad=1)
    fh = fh.merge(per_map, on=["match_id", "map_name"], how="left")
    fh = fh[fh.bad.isna()]

    lut = {r.faceit_id: int(r.hltv_id) for r in load(data, "faceit/hltv_faceit_map").itertuples()}

    def keys(ids_list) -> tuple:
        out = []
        for f in ids_list:
            if f is None or (isinstance(f, float) and np.isnan(f)):
                continue
            h = lut.get(f)
            out.append(str(h) if h is not None else f"f{f}")
        return tuple(out)

    a = [keys(x) for x in fh.team_a_faceit_ids]
    b = [keys(x) for x in fh.team_b_faceit_ids]
    out = pd.DataFrame({
        "match_id": fh.match_id.to_numpy(),
        "date": pd.to_datetime(fh.match_date).to_numpy(),
        "src": FACEIT,
        "a": a,
        "b": b,
        "y": fh.team_a_won.astype(int).to_numpy(),
        "ra": fh.team_a_score.astype(float).to_numpy(),
        "rb": fh.team_b_score.astype(float).to_numpy(),
        "ta": [str(int(x)) if pd.notna(x) else f"f{f}" for x, f in zip(fh.team_a_id, fh.team_a_faceit_id)],
        "tb": [str(int(x)) if pd.notna(x) else f"f{f}" for x, f in zip(fh.team_b_id, fh.team_b_faceit_id)],
    })
    return out[(out.a.map(len) >= MIN_LINEUP) & (out.b.map(len) >= MIN_LINEUP)]


def build_maps(data: str, before: pd.Timestamp, cfg: Config) -> pd.DataFrame:
    frames = [_hltv_maps(data, before)]
    if cfg.faceit:
        frames.append(_faceit_maps(data, before))
    maps = pd.concat(frames, ignore_index=True)
    return maps[maps.date < before]


@dataclass
class Fitted:
    skill: dict
    team_effect: dict
    lineups: dict


def _design(m: pd.DataFrame, cfg: Config):
    vocab: dict = {}
    rows, cols, vals = [], [], []
    for i, (aa, bb, s, ta, tb) in enumerate(zip(m.a, m.b, m.src.to_numpy(), m.ta, m.tb)):
        for side, sign in ((aa, 1.0), (bb, -1.0)):
            scale = sign / len(side)
            for k in side:
                terms = [(k, scale)]
                if cfg.dev_scale > 0 and s == FACEIT:
                    terms.append((("F", k), scale * cfg.dev_scale))
                for key, v in terms:
                    j = vocab.setdefault(key, len(vocab))
                    rows.append(i)
                    cols.append(j)
                    vals.append(v)
        if cfg.team_scale > 0:
            for team, sign in ((ta, 1.0), (tb, -1.0)):
                if team:
                    j = vocab.setdefault(("T", team), len(vocab))
                    rows.append(i)
                    cols.append(j)
                    vals.append(sign * cfg.team_scale)
    X = sparse.csr_matrix((vals, (rows, cols)), shape=(len(m), len(vocab)))
    return X, vocab


def fit(maps: pd.DataFrame, before: pd.Timestamp, cfg: Config) -> Fitted:
    m = maps[maps.date >= before - pd.Timedelta(days=cfg.window_days)].reset_index(drop=True)
    age = (before - m.date).dt.total_seconds().to_numpy() / 86400.0
    w = 0.5 ** (age / cfg.half_life_days) * np.where(m.src.to_numpy() == FACEIT, cfg.faceit_weight, 1.0)
    if cfg.round_weighted:
        share = (m.ra / (m.ra + m.rb)).to_numpy()
        m = pd.concat([m, m], ignore_index=True)
        y = np.concatenate([np.ones(len(share)), np.zeros(len(share))])
        w = np.concatenate([w * share, w * (1 - share)])
    else:
        y = m.y.to_numpy()
    w = w / w.mean()
    X, vocab = _design(m, cfg)
    clf = LogisticRegression(C=cfg.C, fit_intercept=False, solver="lbfgs", max_iter=2000)
    clf.fit(X, y, sample_weight=w)
    beta = clf.coef_.ravel()
    skill, team_effect = {}, {}
    for key, j in vocab.items():
        if isinstance(key, tuple) and key[0] == "T":
            team_effect[key[1]] = cfg.team_scale * beta[j]
        elif isinstance(key, tuple):
            skill[key[1]] = skill.get(key[1], 0.0) + cfg.dev_scale * beta[j]
        else:
            skill[key] = skill.get(key, 0.0) + beta[j]
    return Fitted(skill=skill, team_effect=team_effect, lineups=_latest_lineups(maps, before, cfg))


def _latest_lineups(maps: pd.DataFrame, before: pd.Timestamp, cfg: Config) -> dict:
    recent = maps[(maps.date >= before - pd.Timedelta(days=cfg.lineup_days)) & (maps.date < before)]
    if not cfg.faceit:
        recent = recent[recent.src == HLTV]
    sides = pd.concat([
        pd.DataFrame({"team": recent.ta, "lineup": recent.a, "date": recent.date, "match_id": recent.match_id}),
        pd.DataFrame({"team": recent.tb, "lineup": recent.b, "date": recent.date, "match_id": recent.match_id}),
    ])
    sides = sides[sides.team != ""].sort_values(["date", "match_id"], ascending=False)
    latest = sides.drop_duplicates("team")
    return dict(zip(latest.team, latest.lineup))


def map_logit(fitted: Fitted, a_keys: tuple, b_keys: tuple) -> float:
    s = 0.0
    for sign, keys in ((1.0, a_keys), (-1.0, b_keys)):
        if not keys:
            continue
        for k in keys:
            s += sign * fitted.skill.get(k, 0.0) / len(keys)
    return s


def series_prob(p: float, bo: int) -> float:
    need = bo // 2 + 1
    return sum(comb(bo, j) * p**j * (1 - p) ** (bo - j) for j in range(need, bo + 1))


def predict(fitted: Fitted, targets: pd.DataFrame) -> np.ndarray:
    q = np.empty(len(targets))
    for i, r in enumerate(targets.itertuples()):
        ta = str(int(r.team_a_id)) if pd.notna(r.team_a_id) else ""
        tb = str(int(r.team_b_id)) if pd.notna(r.team_b_id) else ""
        eta = map_logit(fitted, fitted.lineups.get(ta, ()), fitted.lineups.get(tb, ()))
        eta += fitted.team_effect.get(ta, 0.0) - fitted.team_effect.get(tb, 0.0)
        p = 1.0 / (1.0 + np.exp(-eta))
        q[i] = series_prob(p, int(r.bo_type))
    return q
