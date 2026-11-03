import dataclasses

import numpy as np
import pandas as pd
import pytest

from gnomepy_research.models.cs2.hltv.pit import assert_pit_consistent
from gnomepy_research.models.cs2.hltv.player_skill import (
    LINEUPS_AUX,
    PLAYER_SKILL_FEATURES,
    PlayerSkillConfig,
    PlayerSkillFilter,
    _PlayerSkillAccumulator,
    compute_player_skill_features,
)

RANKINGS = pd.DataFrame({
    "date": pd.to_datetime(["2026-01-01"] * 3),
    "team_id": [1, 2, 3],
    "points": [1000.0, 100.0, 10.0],
})
FIVES = {t: tuple(range(10 * t + 1, 10 * t + 6)) for t in (1, 2, 3, 4)}
DAY = pd.Timestamp("2026-01-05")


def _map(match_id, day, team_a, team_b, rounds_a, rounds_b, pos=1, lineup_a=None, lineup_b=None):
    return {
        "match_id": match_id,
        "match_date": pd.Timestamp(day),
        "map_name": f"map{pos}",
        "map_position_in_series": pos,
        "team_a_id": team_a,
        "team_b_id": team_b,
        "team_a_score": rounds_a,
        "team_b_score": rounds_b,
        "team_a_won": float(rounds_a > rounds_b) if rounds_a == rounds_a else np.nan,
        "team_a_lineup_ids": list(FIVES[team_a]) if lineup_a is None else lineup_a,
        "team_b_lineup_ids": list(FIVES[team_b]) if lineup_b is None else lineup_b,
        "team_a_player_ids": list(FIVES[team_a]),
        "team_b_player_ids": list(FIVES[team_b]),
    }


def _history():
    rows, mid = [], 100
    rng = np.random.default_rng(0)
    for d in range(12):
        day = DAY + pd.Timedelta(days=d)
        for a, b in ((1, 2), (3, 4)):
            mid += 1
            for pos in (1, 2):
                ra = int(rng.integers(5, 14))
                rows.append(_map(mid, day, a, b, ra, 13 if ra < 13 else 9, pos))
    return pd.DataFrame(rows)


def test_newcomer_prior_rises_with_debut_team_ranking():
    f = PlayerSkillFilter(PlayerSkillConfig(), RANKINGS)
    prior = [f._state(1, team, DAY)[0] for team in (1, 2, 3, 99)]
    assert prior == sorted(prior, reverse=True)
    assert prior[-1] == PlayerSkillConfig().a


def test_variance_grows_with_idle_days():
    cfg = PlayerSkillConfig()
    f = PlayerSkillFilter(cfg, RANKINGS)
    f.observe_map(DAY, 1, 2, FIVES[1], FIVES[2], 13, 7)
    _, soon = f._state(11, 1, DAY + pd.Timedelta(days=1))
    _, later = f._state(11, 1, DAY + pd.Timedelta(days=101))
    assert later - soon == pytest.approx(100 * cfg.tau ** 2)


def test_near_tied_leaders_go_to_the_first_listed_player():
    f = PlayerSkillFilter(PlayerSkillConfig(), RANKINGS)
    lam, n = f.cfg.lam, 5
    weights = f._weights([0.1, 0.1 + 1e-15, 0.1, 0.05, 0.0])
    assert weights[0] == pytest.approx(1 / n + lam * (1 - 1 / n))
    assert weights[1] == pytest.approx(1 / n - lam / n)


def test_rounding_noise_between_teammates_does_not_change_the_update():
    """Two level teammates a last bit apart either way, as another machine's rounding leaves them, update alike."""
    filters = []
    for nudge in (1e-16, -1e-16):
        f = PlayerSkillFilter(PlayerSkillConfig(), RANKINGS)
        # a loss drops the first-listed leader furthest, leaving the other four level at the top
        f.observe_map(DAY, 1, 2, FIVES[1], FIVES[2], 5, 13)
        second = FIVES[1][1]
        f._mu[second] = f._mu[FIVES[1][2]] + nudge
        f.observe_map(DAY, 1, 2, FIVES[1], FIVES[2], 5, 13)
        filters.append(f)
    for p in FIVES[1] + FIVES[2]:
        assert filters[0]._mu[p] == pytest.approx(filters[1]._mu[p], abs=1e-12)


def test_winners_rise_and_losers_fall():
    f = PlayerSkillFilter(PlayerSkillConfig(), RANKINGS)
    before = {p: f._state(p, t, DAY)[0] for t in (1, 2) for p in FIVES[t]}
    f.observe_map(DAY, 1, 2, FIVES[1], FIVES[2], 5, 13)
    assert all(f._mu[p] < before[p] for p in FIVES[1])
    assert all(f._mu[p] > before[p] for p in FIVES[2])


def test_gap_uncertainty_falls_as_fives_keep_playing():
    """Results only rank players against each other, so a lineup's own SD keeps a common-mode part; the gap
    between two lineups is what is learned. Without teammate covariances even that would barely move. The settings it
    depends on (start spread, drift, round correlation) are pinned, so it tests the filter, not the latest fit."""
    f = PlayerSkillFilter(PlayerSkillConfig(sigma0=0.3, tau=0.002, rho=0.013), RANKINGS)
    sds = []
    for d in range(30):
        day = DAY + pd.Timedelta(days=d)
        _, s2, _, _ = f.gap(f.lineup(FIVES[1], 1, 0.0, day), f.lineup(FIVES[2], 2, 0.0, day))
        sds.append(s2 ** 0.5)
        f.observe_map(day, 1, 2, FIVES[1], FIVES[2], 13, 9)
    assert sds[-1] < 0.5 * sds[0]


def test_joint_covariance_stays_positive_semidefinite():
    f = PlayerSkillFilter(PlayerSkillConfig(), RANKINGS)
    for d, (a, b) in enumerate([(1, 2), (2, 3), (3, 1), (1, 4), (4, 2)] * 6):
        f.observe_map(DAY + pd.Timedelta(days=d), a, b, FIVES[a], FIVES[b], 13, 4 + d % 8)
    day = DAY + pd.Timedelta(days=40)
    _, _, cov, _ = f.gap(f.lineup(FIVES[1], 1, 0.0, day), f.lineup(FIVES[2], 2, 0.0, day))
    assert np.linalg.eigvalsh(cov).min() > -1e-9


def test_chemistry_counts_only_the_exact_five_inside_the_window():
    cfg = dataclasses.replace(PlayerSkillConfig(), chem_days=10)
    f = PlayerSkillFilter(cfg, RANKINGS)
    f.observe_map(DAY, 1, 2, FIVES[1], FIVES[2], 13, 9)
    assert f.lineup(FIVES[1], 1, 0.0, DAY + pd.Timedelta(days=1)).five_maps == 1
    assert f.lineup(FIVES[1][:4] + (99,), 1, 0.0, DAY + pd.Timedelta(days=1)).five_maps == 0
    assert f.lineup(FIVES[1], 1, 0.0, DAY + pd.Timedelta(days=11)).five_maps == 0


def test_swapping_sides_mirrors_every_feature():
    hist = _history()
    target = _map(999, DAY + pd.Timedelta(days=20), 1, 2, np.nan, np.nan)
    swapped = _map(999, DAY + pd.Timedelta(days=20), 2, 1, np.nan, np.nan)
    ab = compute_player_skill_features(pd.concat([hist, pd.DataFrame([target])]), RANKINGS)
    ba = compute_player_skill_features(pd.concat([hist, pd.DataFrame([swapped])]), RANKINGS)
    ab, ba = ab[ab.match_id == 999].iloc[0], ba[ba.match_id == 999].iloc[0]
    for k in ("ps_p_map", "ps_p_series"):
        assert ab[k] == pytest.approx(1 - ba[k])
    assert ab.ps_strength_diff == pytest.approx(-ba.ps_strength_diff)
    assert ab.ps_gap_sd == pytest.approx(ba.ps_gap_sd)
    for k in ("strength", "chem", "sd", "new_players", "best", "worst", "five_maps"):
        assert ab[f"team_a_ps_{k}"] == pytest.approx(ba[f"team_b_ps_{k}"])


def test_standins_lower_the_lineup():
    hist = pd.concat([_history(), pd.DataFrame([_map(999, DAY + pd.Timedelta(days=20), 1, 2, np.nan, np.nan)])])
    standin = pd.DataFrame({"match_id": [999] * 2, "team_id": [1, 1], "player_id": [11, 12], "is_standin": [True, False]})
    plain = compute_player_skill_features(hist, RANKINGS)
    flagged = compute_player_skill_features(hist, RANKINGS, standin)
    assert flagged[flagged.match_id == 999].ps_p_map.iloc[0] < plain[plain.match_id == 999].ps_p_map.iloc[0]


def test_missing_announced_lineup_gives_nan():
    hist = pd.concat([_history(), pd.DataFrame([_map(999, DAY + pd.Timedelta(days=20), 1, 2, 13, 5, lineup_a=[])])])
    row = compute_player_skill_features(hist, RANKINGS).set_index("match_id").loc[999]
    assert row[PLAYER_SKILL_FEATURES].isna().all()


def test_unplayed_maps_are_priced_but_not_learned_from():
    hist = _history()
    upcoming = pd.DataFrame([_map(999, DAY + pd.Timedelta(days=20), 1, 2, np.nan, np.nan),
                             _map(1000, DAY + pd.Timedelta(days=21), 1, 2, np.nan, np.nan)])
    out = compute_player_skill_features(pd.concat([hist, upcoming]), RANKINGS).set_index("match_id")
    assert out.loc[999, "ps_p_map"] == pytest.approx(out.loc[1000, "ps_p_map"], abs=1e-3)
    assert out.loc[999, "team_a_ps_five_maps"] == out.loc[1000, "team_a_ps_five_maps"]


def test_point_in_time():
    hist = _history()
    lineups = pd.DataFrame({"match_id": hist.match_id.unique(), "team_id": 1, "player_id": 11, "is_standin": False})
    assert_pit_consistent(hist, lambda: _PlayerSkillAccumulator(PlayerSkillConfig(), RANKINGS),
                          aux={LINEUPS_AUX: lineups}, sample=6)
