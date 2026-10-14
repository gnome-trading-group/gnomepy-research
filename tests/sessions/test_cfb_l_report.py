from __future__ import annotations

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cfb_win_probability.evaluation.l_report import (
    bootstrap_diff, confidence_stats, l_cuts, rating_volatility, season_log_loss, tau_phase_rows)


def games() -> pd.DataFrame:
    return pd.DataFrame({
        "season_type": ["regular", "regular", "regular", "regular", "postseason", "regular"],
        "neutral_site": [False, True, False, False, True, False],
        "week": [2, 12, 6, 11, 1, 3],
        "p_open": [0.85, 0.5, 0.15, 0.45, 0.7, 0.3],
        "conference_game": [True, True, False, True, False, False],
        "home_conference": ["SEC", "SEC", "Sun Belt", "ACC", "SEC", "MAC"],
        "away_conference": ["MAC", "SEC", "Big Ten", "ACC", "Sun Belt", "SEC"],
        "home_team": ["a", "b", "c", "d", "e", "f"],
        "away_team": ["g", "h", "i", "j", "k", "l"],
    })


def test_l_cuts_masks_and_sides():
    c = {k: (m.tolist(), f.tolist()) for k, (m, f) in l_cuts(games()).items()}
    assert c["regular home-site"][0] == [True, False, True, True, False, True]
    assert c["regular neutral-site"][0] == [False, True, False, False, False, False]
    assert c["postseason"][0] == [False, False, False, False, True, False]
    assert c["weeks 1-4"][0] == [True, False, False, False, False, True]
    assert c["weeks 5-9"][0] == [False, False, True, False, False, False]
    assert c["weeks 10+"][0] == [False, True, False, True, False, False]
    assert c["big favorites"] == ([True, False, True, False, False, False], [False, False, True, False, False, False])
    assert c["toss-ups"][0] == [False, True, False, True, False, False] and not any(c["toss-ups"][1])
    assert c["P4 vs G5"] == ([True, False, True, False, True, True], [False, False, True, False, False, True])
    assert c["non-conference"][0] == [False, False, True, False, True, True]


def test_bootstrap_diff_paired():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 3000).astype(float)
    p = np.clip(0.5 + (y - 0.5) * 0.2 + rng.normal(0, 0.05, 3000), 0.05, 0.95)
    r = bootstrap_diff(y, p, p, n_boot=200)
    assert r == (0.0, 0.0, 0.0)
    m, lo, hi = bootstrap_diff(y, np.full(3000, 0.5), p, n_boot=200)
    assert lo < m < hi and lo > 0


def test_confidence_stats():
    y = np.array([1.0, 0.0, 1.0, 0.0])
    p = np.array([0.9, 0.9, 0.6, 0.2])
    s = confidence_stats(y, p)
    assert np.isclose(s["mean_conf"], (0.4 + 0.4 + 0.1 + 0.3) / 4)
    assert s["confident_wrong"] == 1 and s["wrong"] == 1 and s["n"] == 4


def test_rating_volatility_buckets_and_pass_unit():
    rows = []
    for team, step in [(1, 0.1), (2, 0.3)]:
        for w in range(1, 13):
            rows.append(dict(season=2023, team_id=team, unit="pass", season_type="regular", week=w,
                             offense_mean=step * w, defense_mean=0.0))
            rows.append(dict(season=2023, team_id=team, unit="rush", season_type="regular", week=w,
                             offense_mean=5.0 * w, defense_mean=0.0))
        rows.append(dict(season=2023, team_id=team, unit="pass", season_type="postseason", week=1,
                         offense_mean=99.0, defense_mean=0.0))
    v = rating_volatility(pd.DataFrame(rows), [2023])
    assert np.allclose([v["wk2-5"], v["wk6-10"], v["wk11+"]], 0.2)
    assert np.isclose(v["final_sd"], np.std([1.2, 3.6], ddof=1))


def test_rating_volatility_without_unit():
    df = pd.DataFrame(dict(season=2024, team_id=1, season_type="regular", week=[1, 2, 3],
                           offense_mean=[0.0, 0.1, 0.3], defense_mean=[0.0, 0.0, -0.1]))
    v = rating_volatility(df, [2024])
    assert np.isclose(v["wk2-5"], 0.1) and np.isnan(v["wk11+"])


def test_season_log_loss():
    g = pd.DataFrame({"season": [2023, 2023, 2024], "home_win": [1.0, 0.0, 1.0],
                      "p_A": [0.5, 0.5, 0.9], "p_open": [0.5, 0.5, 0.5]})
    t = season_log_loss(g, ["A"])
    assert np.isclose(t.loc[2023, "A"], np.log(2)) and np.isclose(t.loc[2024, "A"], -np.log(0.9))
    assert np.isclose(t.loc[2024, "open"], np.log(2))


def test_tau_phase_rows():
    raw = {"hyperparameters": {"tau_off": [0.1, 0.2, 0.3, 0.4], "tau_def": [0.5, 0.6, 0.7, 0.8]}}
    r = tau_phase_rows(2023, raw)
    assert r == [{"season": 2023, "param": "tau_off", "early": 0.1, "mid": 0.2, "late": 0.3, "post": 0.4},
                 {"season": 2023, "param": "tau_def", "early": 0.5, "mid": 0.6, "late": 0.7, "post": 0.8}]
