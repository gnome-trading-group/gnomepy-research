"""Weekly point-in-time feature pass: ratings update after each week, features read the state at the week's Monday."""
from __future__ import annotations

import math
from collections import defaultdict, deque

import numpy as np
import pandas as pd

from gnomepy_research.models.cs2.hltv.config import showmatch_mask
from gnomepy_research.sessions.cs2_comp import bundle
from gnomepy_research.sessions.cs2_comp.g03 import bt, faceit

PLAYER_K = 6.0
TEAM_K = 24.0
SCALE = 400.0 / math.log(10)
MIN_KNOWN_MAPS = 20
RATING_WINDOW_DAYS = 120
RATING_HALF_LIFE_DAYS = 60.0
RATING_PRIOR_MAPS = 15.0
RATING_MEAN = 1.0


def _naive(s):
    s = pd.to_datetime(s)
    return s.dt.tz_convert(None) if s.dt.tz is not None else s


def monday(ts: pd.Series) -> pd.Series:
    ts = _naive(ts)
    return (ts - pd.to_timedelta(ts.dt.dayofweek, unit="D")).dt.normalize()


def clean_history(hist: pd.DataFrame) -> pd.DataFrame:
    h = hist[~showmatch_mask(hist)].copy()
    h["t"] = _naive(h.match_time.fillna(h.match_date))
    h["week"] = monday(h.t)
    return h.sort_values(["t", "match_id", "map_position_in_series"], kind="stable")


def series_table(h: pd.DataFrame) -> pd.DataFrame:
    g = h.groupby("match_id", sort=False)
    s = g.agg(t=("t", "min"), week=("week", "min"), team_a_id=("team_a_id", "first"), team_b_id=("team_b_id", "first"),
              a_maps=("team_a_won", "sum"), n_maps=("team_a_won", "size"), tier=("event_tier", "first"),
              is_lan=("is_lan", "first"), bo=("bo_type", "first"),
              a_lineup=("team_a_lineup_ids", "first"), b_lineup=("team_b_lineup_ids", "first"),
              a_players=("team_a_player_ids", "first"), b_players=("team_b_player_ids", "first")).reset_index()
    s["b_maps"] = s.n_maps - s.a_maps
    s["y"] = np.where(s.a_maps > s.b_maps, 1.0, np.where(s.a_maps < s.b_maps, 0.0, np.nan))
    return s


def _ids(x):
    return [] if x is None or (isinstance(x, float) and np.isnan(x)) else [int(v) for v in x]


def rolling_player_rating(stats: pd.DataFrame, weeks: list[pd.Timestamp]) -> dict:
    s = stats[stats.side == "all"][["match_date", "player_id", "rating"]].dropna()
    s = s.assign(d=_naive(s.match_date))
    out = {}
    for w in weeks:
        win = s[(s.d < w) & (s.d >= w - pd.Timedelta(days=RATING_WINDOW_DAYS))]
        age = (w - win.d).dt.days.to_numpy()
        wt = 0.5 ** (age / RATING_HALF_LIFE_DAYS)
        g = pd.DataFrame({"p": win.player_id.to_numpy(), "w": wt, "wr": wt * win.rating.to_numpy()}).groupby("p").sum()
        mean = (g.wr + RATING_PRIOR_MAPS * RATING_MEAN * 0.5) / (g.w + RATING_PRIOR_MAPS * 0.5)
        out[w] = (mean.to_dict(), g.w.to_dict())
    return out


class State:
    def __init__(self):
        self.pr = defaultdict(float)
        self.pn = defaultdict(int)
        self.te = defaultdict(float)
        self.tn = defaultdict(int)
        self.t_maps = defaultdict(list)
        self.t_series = defaultdict(lambda: deque(maxlen=12))
        self.last_lineup = {}
        self.last_t = {}
        self.h2h = defaultdict(lambda: [0, 0])
        self.lineup_hist = defaultdict(list)

    def side(self, team, lineup, now, ratings, fc, standins, skills):
        lu = _ids(lineup)
        prs = [self.pr[p] for p in lu] or [0.0]
        pns = [self.pn[p] for p in lu] or [0]
        rmean, rw = ratings
        fm, fw, fl = fc
        sk_h, sk_f, sk_s = skills
        bh = [sk_h.get(p, 0.0) for p in lu] or [0.0]
        bf = [sk_f.get(p, 0.0) for p in lu] or [0.0]
        bs = [sk_s.get(p, 0.0) for p in lu] or [0.0]
        fs = [fm.get(p, faceit.PRIOR_SCORE) for p in lu] or [faceit.PRIOR_SCORE]
        fcov = float(sum(fw.get(p, 0.0) >= 5 for p in lu))
        lv = [fl[p] for p in lu if p in fl]
        hr = [rmean.get(p, RATING_MEAN) for p in lu] or [RATING_MEAN]
        hw = [rw.get(p, 0.0) for p in lu] or [0.0]
        ms = self.t_maps[team]
        recent = [(d, won) for d, won in ms if (now - d).days <= 90]
        wts = [0.5 ** ((now - d).days / 45.0) for d, _ in recent]
        mwr = (sum(w * won for w, (_, won) in zip(wts, recent)) + 3 * 0.5) / (sum(wts) + 3)
        ser = list(self.t_series[team])
        swr = (sum(v for _, v in ser) + 2 * 0.5) / (len(ser) + 2)
        last = self.last_lineup.get(team, [])
        overlap = len(set(lu) & set(last)) if last else -1
        since = (now - self.last_t[team]).days if team in self.last_t else 400
        return {
            "pr_mean": float(np.mean(prs)), "pr_min": float(np.min(prs)), "pr_max": float(np.max(prs)),
            "pn_min": float(np.min(pns)), "n_known": float(sum(n >= MIN_KNOWN_MAPS for n in pns)),
            "pn_mean": float(np.mean(pns)),
            "te": self.te[team], "tn": float(self.tn[team]),
            "hr_mean": float(np.mean(hr)), "hr_min": float(np.min(hr)), "hw_mean": float(np.mean(hw)),
            "mwr": mwr, "n90": float(len(recent)), "swr": swr,
            "fs_mean": float(np.mean(fs)), "fs_min": float(np.min(fs)), "f_cov": fcov,
            "lvl_mean": float(np.mean(lv)) if lv else 0.0, "lvl_n": float(len(lv)),
            "standin": float(standins),
            "bt_h": float(np.sum(bh)), "bt_f": float(np.sum(bf)), "bt_f_min": float(np.min(bf)),
            "bt_s": float(np.sum(bs)), "bt_unseen": float(sum(p not in sk_f for p in lu)),
            "overlap": float(overlap), "since": float(since), "n_stand": float(len(lu) - sum(n >= 5 for n in pns)),
        }

    def update_map(self, row, now):
        a, b = row.team_a_id, row.team_b_id
        la, lb = _ids(row.team_a_player_ids) or _ids(row.team_a_lineup_ids), _ids(row.team_b_player_ids) or _ids(row.team_b_lineup_ids)
        if not la or not lb:
            return
        sa, sb = row.team_a_score, row.team_b_score
        if pd.isna(sa) or pd.isna(sb) or sa + sb == 0:
            return
        share = float(np.clip(sa / (sa + sb), 0.05, 0.95))
        diff = np.mean([self.pr[p] for p in la]) - np.mean([self.pr[p] for p in lb])
        exp = 1 / (1 + math.exp(-diff / SCALE))
        g = share - exp
        for p in la:
            self.pr[p] += PLAYER_K * g * 4
            self.pn[p] += 1
        for p in lb:
            self.pr[p] -= PLAYER_K * g * 4
            self.pn[p] += 1
        te = self.te[a] - self.te[b]
        g2 = share - 1 / (1 + math.exp(-te / SCALE))
        self.te[a] += TEAM_K * g2 * 4
        self.te[b] -= TEAM_K * g2 * 4
        self.tn[a] += 1
        self.tn[b] += 1
        won = float(row.team_a_won)
        self.t_maps[a].append((now, won))
        self.t_maps[b].append((now, 1 - won))
        self.last_lineup[a], self.last_lineup[b] = la, lb
        self.last_t[a] = self.last_t[b] = now


def build(data_dir, targets: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    hist = bundle.load(data_dir, "hltv/cs2_match_history", cutoff)
    h = clean_history(hist)
    ser = series_table(h)
    stats = bundle.load(data_dir, "hltv/cs2_player_map_stats", cutoff, ["match_date", "player_id", "rating", "side"])
    rank = bundle.load(data_dir, "hltv/cs2_team_rankings", cutoff)
    rank = rank.assign(d=_naive(rank.date)).sort_values("d")

    tgt = targets.copy()
    tgt["t"] = _naive(tgt.match_time)
    tgt["week"] = _naive(tgt.week_cutoff)
    ser_all = ser
    all_weeks = sorted(set(ser.week) | set(tgt.week))
    ratings = rolling_player_rating(stats, all_weeks)
    bt_h = bt.weekly_skills(data_dir, cutoff, h, all_weeks, False)
    bt_f = bt.weekly_skills(data_dir, cutoff, h, all_weeks, True)
    bt_s = bt.weekly_skills(data_dir, cutoff, h, all_weeks, True, bt.Params(window_days=365, half_life_days=120.0, c_ridge=0.3))
    fst, flv = faceit.faceit_frames(data_dir, cutoff)
    fcs = faceit.rolling(fst, flv, all_weeks)
    lu = bundle.load(data_dir, "hltv/cs2_match_lineups", cutoff, ["match_id", "match_date", "team_id", "is_standin"])
    stand = lu.groupby(["match_id", "team_id"]).is_standin.sum().to_dict()

    ser_by_week = dict(tuple(ser.groupby("week")))
    ser_all_by_week = dict(tuple(ser_all.groupby("week")))
    maps_by_week = dict(tuple(h.groupby("week")))
    tgt_by_week = dict(tuple(tgt.groupby("week")))
    st = State()
    rows = []

    def rank_lookup(w):
        r = rank[rank.d < w]
        r = r.sort_values("d").groupby("team_id").last()
        return r["points"].to_dict(), r["rank"].to_dict()

    for w in all_weeks:
        pts, rk = rank_lookup(w)
        rt, fc = ratings[w], fcs[w]
        batches = []
        if w in ser_by_week:
            s = ser_by_week[w]
            batches.append((s, "a_lineup", "b_lineup", False))
        if w in tgt_by_week:
            batches.append((tgt_by_week[w], "team_a_lineup_ids", "team_b_lineup_ids", True))
        for df, la, lb, is_t in batches:
            for r in df.itertuples():
                if is_t:
                    sa, sb = len(r.team_a_standin_ids), len(r.team_b_standin_ids)
                else:
                    sa, sb = stand.get((r.match_id, r.team_a_id), 0), stand.get((r.match_id, r.team_b_id), 0)
                skills = (bt_h[w], bt_f[w], bt_s[w])
                fa = st.side(r.team_a_id, getattr(r, la), w, rt, fc, sa, skills)
                fb = st.side(r.team_b_id, getattr(r, lb), w, rt, fc, sb, skills)
                row = {"match_id": r.match_id, "week": w, "t": r.t, "is_target": is_t,
                       "y": np.nan if is_t else r.y, "tier": float(r.event_tier if is_t else r.tier),
                       "is_lan": float(r.is_lan), "bo": float(r.bo_type if is_t else r.bo)}
                for k in fa:
                    row[f"d_{k}"] = fa[k] - fb[k]
                    row[f"s_{k}"] = fa[k] + fb[k]
                pa, pb = pts.get(r.team_a_id), pts.get(r.team_b_id)
                row["pts_miss"] = float(pa is None) + float(pb is None)
                row["d_logpts"] = math.log1p(pa or 0.0) - math.log1p(pb or 0.0)
                ra, rb = rk.get(r.team_a_id), rk.get(r.team_b_id)
                row["d_logrank"] = math.log(ra or 300.0) - math.log(rb or 300.0)
                key = (min(r.team_a_id, r.team_b_id), max(r.team_a_id, r.team_b_id))
                hw = st.h2h[key]
                ia = r.team_a_id == key[0]
                row["d_h2h"] = (hw[0] - hw[1]) * (1 if ia else -1)
                rows.append(row)
        if w in maps_by_week:
            for r in maps_by_week[w].itertuples():
                st.update_map(r, r.t)
            for r in ser_all_by_week[w].itertuples():
                if np.isnan(r.y):
                    continue
                st.t_series[r.team_a_id].append((r.t, r.y))
                st.t_series[r.team_b_id].append((r.t, 1.0 - r.y))
                key = (min(r.team_a_id, r.team_b_id), max(r.team_a_id, r.team_b_id))
                if r.y == 1.0:
                    st.h2h[key][0 if r.team_a_id == key[0] else 1] += 1
                else:
                    st.h2h[key][1 if r.team_a_id == key[0] else 0] += 1
    return pd.DataFrame(rows)
