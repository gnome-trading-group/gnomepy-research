"""One player-skill scale over HLTV and FACEIT, refit at each cutoff.

Two measurements share one skill per player. Map results are logistic comparisons of the two five-player lineups: a
player's skill enters with weight 1/5 and the map's log-odds is the difference of the lineup sums. Per-player map
performance (HLTV rating, FACEIT K/D) is a second, Gaussian measurement of the same skill, which sharpens skill for
players with few results. FACEIT measurements are down-weighted, since FACEIT is a noisier, lower-tier signal.
Linked FACEIT players carry their HLTV id; unlinked ones keep a FACEIT id so that their results still connect the graph.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit

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
    perf_weight: float = 0.0
    round_weighted: bool = False


def naive(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_convert(None) if ts.tzinfo is not None else ts


def _player_key(faceit_id: str, lut: dict) -> str:
    h = lut.get(faceit_id)
    return str(h) if h is not None else f"f{faceit_id}"


def _faceit_lookup(data: str) -> dict:
    return {r.faceit_id: int(r.hltv_id) for r in load(data, "faceit/hltv_faceit_map").itertuples()}


def _faceit_series(data: str, before: pd.Timestamp) -> pd.Index:
    fm = load(data, "faceit/faceit_matches", before, ["match_id", "has_stats"])
    return fm[fm.has_stats.astype(bool)].match_id


def _faceit_exclusions(data: str, before: pd.Timestamp):
    ex = load(data, "faceit/faceit_exclusions", before, ["match_id", "map_name"])
    return ex.match_id[ex.map_name.isna()], ex.dropna(subset=["map_name"]).assign(bad=1)


def _hltv_maps(data: str, before: pd.Timestamp) -> pd.DataFrame:
    cols = ["match_id", "match_date", "match_time", "team_a_id", "team_b_id", "team_a_player_ids",
            "team_b_player_ids", "team_a_won", "team_a_score", "team_b_score", "event_name", "format"]
    h = load(data, "hltv/cs2_match_history", before, cols)
    h = h[~showmatch_mask(h)].dropna(subset=["team_a_won", "team_a_player_ids", "team_b_player_ids", "team_a_score",
                                             "team_b_score"])
    out = pd.DataFrame({
        "match_id": h.match_id.to_numpy(),
        "date": pd.to_datetime(h.match_date).to_numpy(),
        "src": HLTV,
        "a": [tuple(str(int(x)) for x in ids if pd.notna(x)) for ids in h.team_a_player_ids],
        "b": [tuple(str(int(x)) for x in ids if pd.notna(x)) for ids in h.team_b_player_ids],
        "y": h.team_a_won.astype(int).to_numpy(),
        "ra": h.team_a_score.astype(float).to_numpy(),
        "rb": h.team_b_score.astype(float).to_numpy(),
        "ta": [str(int(x)) if pd.notna(x) else "" for x in h.team_a_id],
        "tb": [str(int(x)) if pd.notna(x) else "" for x in h.team_b_id],
    })
    return out[(out.a.map(len) >= MIN_LINEUP) & (out.b.map(len) >= MIN_LINEUP)]


def _faceit_maps(data: str, before: pd.Timestamp) -> pd.DataFrame:
    ids = _faceit_series(data, before)
    cols = ["match_id", "match_date", "map_name", "map_position_in_series", "team_a_faceit_ids", "team_b_faceit_ids",
            "team_a_id", "team_b_id", "team_a_faceit_id", "team_b_faceit_id", "team_a_won", "team_a_score",
            "team_b_score"]
    fh = load(data, "faceit/faceit_match_history", before, cols)
    fh = fh[fh.match_id.isin(ids)].drop_duplicates(["match_id", "map_position_in_series"])
    whole, per_map = _faceit_exclusions(data, before)
    fh = fh[~fh.match_id.isin(whole)].merge(per_map, on=["match_id", "map_name"], how="left")
    fh = fh[fh.bad.isna()]
    lut = _faceit_lookup(data)

    def keys(ids_list) -> tuple:
        return tuple(_player_key(f, lut) for f in ids_list if isinstance(f, str))

    out = pd.DataFrame({
        "match_id": fh.match_id.to_numpy(),
        "date": pd.to_datetime(fh.match_date).to_numpy(),
        "src": FACEIT,
        "a": [keys(x) for x in fh.team_a_faceit_ids],
        "b": [keys(x) for x in fh.team_b_faceit_ids],
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


def build_performance(data: str, before: pd.Timestamp, cfg: Config) -> pd.DataFrame:
    empty = pd.DataFrame({"date": pd.to_datetime([]), "key": pd.Series([], dtype=object),
                          "src": pd.Series([], dtype=int), "z": pd.Series([], dtype=float)})
    if cfg.perf_weight <= 0:
        return empty
    hist = load(data, "hltv/cs2_match_history", before,
                ["match_id", "match_date", "match_time", "event_name", "format"]).drop_duplicates("match_id")
    keep_ids = hist.match_id[~showmatch_mask(hist).to_numpy()]
    ps = load(data, "hltv/cs2_player_map_stats", before, ["match_id", "match_date", "player_id", "side", "rating"])
    ps = ps[(ps.side == "all") & ps.rating.notna() & ps.player_id.notna() & ps.match_id.isin(keep_ids)]
    hltv_perf = pd.DataFrame({
        "date": pd.to_datetime(ps.match_date).to_numpy(),
        "key": ps.player_id.astype(int).astype(str).to_numpy(),
        "src": HLTV,
        "z": np.log(ps.rating.clip(lower=0.05)).to_numpy(),
    })

    if not cfg.faceit:
        perf = hltv_perf.dropna(subset=["date"])
        return perf[perf.date < before].reset_index(drop=True)

    ids = _faceit_series(data, before)
    fm = load(data, "faceit/faceit_matches", before, ["match_id", "finished_at"])
    finished = dict(zip(fm.match_id, pd.to_datetime(fm.finished_at, unit="s")))
    whole, _ = _faceit_exclusions(data, before)
    fp = load(data, "faceit/faceit_player_map_stats", before, ["match_id", "hltv_id", "faceit_id", "k_d_ratio"])
    fp = fp[fp.match_id.isin(ids) & ~fp.match_id.isin(whole) & fp.k_d_ratio.notna()]
    lut = _faceit_lookup(data)
    faceit_perf = pd.DataFrame({
        "date": [finished.get(m) for m in fp.match_id],
        "key": [str(int(h)) if pd.notna(h) else _player_key(f, lut) for h, f in zip(fp.hltv_id, fp.faceit_id)],
        "src": FACEIT,
        "z": np.log(fp.k_d_ratio.clip(0.05, 5.0)).to_numpy(),
    })
    frames = [hltv_perf] if not cfg.faceit else [hltv_perf, faceit_perf]
    perf = pd.concat(frames, ignore_index=True).dropna(subset=["date"])
    perf["date"] = pd.to_datetime(perf.date)
    return perf[perf.date < before].reset_index(drop=True)


@dataclass
class Fitted:
    skill: dict
    lineups: dict


def _outcome_rows(m: pd.DataFrame, vocab: dict):
    rows, cols, vals = [], [], []
    for i, (aa, bb) in enumerate(zip(m.a, m.b)):
        for side, sign in ((aa, 1.0), (bb, -1.0)):
            scale = sign / len(side)
            for k in side:
                rows.append(i)
                cols.append(vocab.setdefault(k, len(vocab)))
                vals.append(scale)
    return rows, cols, vals


def _joint_fit(X, y, w, D, z, v, src, lam):
    n_beta = X.shape[1]
    n_src = 2

    def objective(theta):
        beta, alpha = theta[:n_beta], theta[n_beta:]
        eta = X @ beta
        loss = np.sum(w * (np.logaddexp(0.0, eta) - y * eta))
        grad = X.T @ (w * (expit(eta) - y))
        grad_alpha = np.zeros(n_src)
        if D is not None:
            r = D @ beta + alpha[src] - z
            loss += 0.5 * np.sum(v * r * r)
            grad += D.T @ (v * r)
            grad_alpha = np.bincount(src, weights=v * r, minlength=n_src)
        loss += 0.5 * lam * beta @ beta
        grad += lam * beta
        return loss, np.concatenate([grad, grad_alpha])

    res = minimize(objective, np.zeros(n_beta + n_src), jac=True, method="L-BFGS-B", options={"maxiter": 1000})
    return res.x[:n_beta]


def fit(maps: pd.DataFrame, perf: pd.DataFrame, before: pd.Timestamp, cfg: Config) -> Fitted:
    m = maps[maps.date >= before - pd.Timedelta(days=cfg.window_days)].reset_index(drop=True)
    age = (before - m.date).dt.total_seconds().to_numpy() / 86400.0
    w = 0.5 ** (age / cfg.half_life_days) * np.where(m.src.to_numpy() == FACEIT, cfg.faceit_weight, 1.0)
    if cfg.round_weighted:
        share = (m.ra / (m.ra + m.rb)).to_numpy()
        m = pd.concat([m, m], ignore_index=True)
        y = np.concatenate([np.ones(len(share)), np.zeros(len(share))])
        w = np.concatenate([w * share, w * (1 - share)])
    else:
        y = m.y.to_numpy().astype(float)
    w = w / w.mean()

    vocab: dict = {}
    p = perf[perf.date >= before - pd.Timedelta(days=cfg.window_days)].reset_index(drop=True)
    p_cols = [vocab.setdefault(k, len(vocab)) for k in p.key]
    rows, cols, vals = _outcome_rows(m, vocab)
    n_beta = len(vocab)
    X = sparse.csr_matrix((vals, (rows, cols)), shape=(len(m), n_beta))

    D = z = v = src = None
    if len(p):
        D = sparse.csr_matrix((np.ones(len(p)), (np.arange(len(p)), p_cols)), shape=(len(p), n_beta))
        z = np.empty(len(p))
        for s in (HLTV, FACEIT):
            mask = p.src.to_numpy() == s
            if mask.any():
                vals_s = p.z[mask].to_numpy()
                z[mask] = (vals_s - vals_s.mean()) / vals_s.std()
        age_p = (before - p.date).dt.total_seconds().to_numpy() / 86400.0
        v = 0.5 ** (age_p / cfg.half_life_days) * np.where(p.src.to_numpy() == FACEIT, cfg.faceit_weight, 1.0)
        v = cfg.perf_weight * len(m) * v / v.sum()
        src = p.src.to_numpy()

    beta = _joint_fit(X, y, w, D, z, v, src, 1.0 / cfg.C)
    skill = {k: float(beta[j]) for k, j in vocab.items()}
    return Fitted(skill=skill, lineups=_latest_lineups(maps, before, cfg))


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
        q[i] = series_prob(1.0 / (1.0 + np.exp(-eta)), int(r.bo_type))
    return q
