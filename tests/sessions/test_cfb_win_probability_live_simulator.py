from __future__ import annotations

import numpy as np
import pytest

from gnomepy_research.sessions.cfb_win_probability.live.drive_model import N_DUR_Q, DriveModel
from gnomepy_research.sessions.cfb_win_probability.live.rules import (
    END_HALF,
    FG,
    N_OUTCOMES,
    N_SIT,
    N_YTG,
    PUNT,
    SIT_OT,
    TD,
    diff_bin,
    overtime_rules,
    situation,
    time_bin,
    ytg_bin,
)
from gnomepy_research.sessions.cfb_win_probability.live.simulator import AWAY, HOME, KICKOFF, GameState, simulate


def make_model(probs: dict[int, float], duration: float = 150.0, start_ytg: int = 75,
               beta: np.ndarray | None = None, p_xp: float = 1.0, p_two: float = 0.5,
               ot_probs: dict[int, float] | None = None) -> DriveModel:
    p = np.full(N_OUTCOMES, 1e-12)
    for k, v in probs.items():
        p[k] = v
    logits = np.log(p / p.sum())
    base = np.repeat(np.repeat(logits[:, None, None], N_YTG, axis=1), N_SIT, axis=2)
    if ot_probs is not None:
        q = np.full(N_OUTCOMES, 1e-12)
        for k, v in ot_probs.items():
            q[k] = v
        base[:, :, SIT_OT] = np.log(q / q.sum())[:, None]
    cdf = np.zeros(99)
    cdf[start_ytg - 1:] = 1.0
    return DriveModel(
        base_logits=base,
        beta=np.zeros(N_OUTCOMES) if beta is None else beta,
        next_cdf=np.tile(cdf, (N_OUTCOMES, N_YTG, 1)),
        kick_cdf=cdf.copy(),
        safety_kick_cdf=cdf.copy(),
        duration_q=np.full((N_OUTCOMES, N_SIT, N_DUR_Q), duration),
        p_xp=p_xp,
        p_two=p_two,
    )


BALANCED = make_model({TD: 0.3, FG: 0.1, PUNT: 0.6})


class TestBins:
    def test_ytg_bins_are_inclusive_upper_bounds(self):
        assert ytg_bin(1) == 0 and ytg_bin(5) == 0 and ytg_bin(6) == 1
        assert ytg_bin(75) == 8 and ytg_bin(76) == 9 and ytg_bin(99) == N_YTG - 1

    def test_time_bins(self):
        assert time_bin(0) == 0 and time_bin(30) == 0 and time_bin(31) == 1
        assert time_bin(900) == 4 and time_bin(1800) == 5

    def test_diff_bins_are_symmetric_around_tie(self):
        assert diff_bin(0) == 4
        assert diff_bin(-1) == 3 and diff_bin(1) == 5
        assert diff_bin(-17) == 0 and diff_bin(-16) == 1
        assert diff_bin(16) == 7 and diff_bin(17) == 8

    def test_overtime_situation_ignores_clock_and_score(self):
        assert situation(3, 0, 0) == SIT_OT == situation(4, 1800, 21)


class TestOvertimeRules:
    @pytest.mark.parametrize("season,two_from,shootout_from", [
        (2018, 3, 10**6), (2019, 3, 5), (2020, 3, 5), (2021, 2, 3), (2025, 2, 3),
    ])
    def test_rules_by_season(self, season, two_from, shootout_from):
        r = overtime_rules(season)
        assert (r.two_point_try_from, r.shootout_from) == (two_from, shootout_from)


class TestEndOfRegulation:
    def test_leader_at_zero_clock_wins(self):
        state = GameState(season=2024, half=2, half_clock=0, home_score=24, away_score=21,
                          possession=AWAY, ytg=1)
        assert simulate(state, BALANCED, n_sims=500).p_home == 1.0

    def test_tie_at_zero_clock_goes_to_overtime_and_never_ends_tied(self):
        state = GameState(season=2024, half=2, half_clock=0, home_score=21, away_score=21,
                          possession=HOME, ytg=75)
        res = simulate(state, BALANCED, n_sims=20_000)
        assert np.all(res.home_margin != 0)
        assert abs(res.p_home - 0.5) < 4 * res.mc_se

    def test_end_of_first_half_hands_second_half_kickoff_to_receiver(self):
        # Only the 2nd-half receiver ever gets the ball and every drive is a TD.
        model = make_model({TD: 1.0}, duration=1801)
        state = GameState(season=2024, half=1, half_clock=0, home_score=0, away_score=0,
                          possession=HOME, ytg=KICKOFF, second_half_receiver=AWAY)
        res = simulate(state, model, n_sims=200)
        assert res.p_home == 0.0
        assert np.all(res.home_margin == -7)

    def test_end_half_outcome_stops_the_clock_without_scoring(self):
        model = make_model({END_HALF: 1.0})
        state = GameState(season=2024, half=2, half_clock=600, home_score=10, away_score=3,
                          possession=AWAY, ytg=50)
        res = simulate(state, model, n_sims=200)
        assert np.all(res.home_margin == 7)


class TestOvertime:
    def test_shootout_terminates_when_no_one_can_score(self):
        model = make_model({PUNT: 1.0}, p_two=0.0)
        state = GameState(season=2024, half=2, half_clock=0, home_score=7, away_score=7,
                          possession=HOME, ytg=75)
        res = simulate(state, model, n_sims=2_000)
        assert np.all(np.abs(res.home_margin) == 1)
        assert 0.4 < res.p_home < 0.6

    def test_stronger_ot_offense_wins_more_often(self):
        beta = np.zeros(N_OUTCOMES)
        beta[TD] = 1.0
        model = make_model({TD: 0.4, FG: 0.2, PUNT: 0.4}, beta=beta)
        state = GameState(season=2024, half=2, half_clock=0, home_score=7, away_score=7,
                          possession=HOME, ytg=75)
        res = simulate(state, model, s_home=1.5, s_away=-1.5, n_sims=20_000)
        assert res.p_home > 0.6

    def test_trailing_team_does_not_settle_for_a_field_goal(self):
        # Home has the first OT possession and always scores a TD (huge strength). Away trails by 7,
        # so converting part of its FG mass into TD attempts must lower the home win probability.
        # 2021+ rules, so a tied game ends in a shootout instead of an endless TD-for-TD loop.
        beta = np.zeros(N_OUTCOMES)
        beta[TD] = 1.0
        state = GameState(season=2024, half=3, half_clock=0, home_score=21, away_score=21,
                          possession=HOME, ytg=25)
        p = {}
        for conv in (0.0, 0.5):
            model = make_model({TD: 0.3, FG: 0.7}, beta=beta, p_xp=1.0)
            model.ot_fg_to_td = conv
            p[conv] = simulate(state, model, s_home=50.0, s_away=0.0, n_sims=20_000).p_home
        assert p[0.5] < p[0.0] - 0.05


class TestStrengthAndState:
    def test_symmetric_pregame_is_a_coin_flip(self):
        res = simulate(GameState(season=2024), BALANCED, n_sims=20_000)
        assert abs(res.p_home - 0.5) < 4 * res.mc_se

    def test_win_probability_increases_with_home_strength(self):
        beta = np.zeros(N_OUTCOMES)
        beta[TD] = 0.8
        beta[PUNT] = -0.5
        model = make_model({TD: 0.3, FG: 0.1, PUNT: 0.6}, beta=beta)
        ps = [simulate(GameState(season=2024), model, s_home=s, s_away=-s, n_sims=10_000, seed=1).p_home
              for s in (-0.5, 0.0, 0.5)]
        assert ps[0] < ps[1] < ps[2]

    def test_late_lead_is_safe_when_opponent_cannot_finish_in_time(self):
        # 10 seconds left, every drive takes 150s: the trailing team's drive ends the game.
        state = GameState(season=2024, half=2, half_clock=10, home_score=17, away_score=14,
                          possession=AWAY, ytg=75)
        model = make_model({PUNT: 0.7, FG: 0.3})
        assert simulate(state, model, n_sims=5_000).p_home > 0.65

    def test_posterior_samples_are_cycled_and_paired(self):
        beta = np.zeros(N_OUTCOMES)
        beta[TD] = 2.0
        model = make_model({TD: 0.3, PUNT: 0.7}, beta=beta)
        res = simulate(GameState(season=2024), model, s_home=np.array([2.0, -2.0]),
                       s_away=np.array([-2.0, 2.0]), n_sims=20_000)
        assert abs(res.p_home - 0.5) < 4 * res.mc_se
        with pytest.raises(ValueError):
            simulate(GameState(season=2024), model, s_home=np.zeros(3), s_away=np.zeros(2))

    def test_same_seed_reproduces(self):
        a = simulate(GameState(season=2024), BALANCED, n_sims=2_000, seed=7)
        b = simulate(GameState(season=2024), BALANCED, n_sims=2_000, seed=7)
        assert np.array_equal(a.home_margin, b.home_margin)
