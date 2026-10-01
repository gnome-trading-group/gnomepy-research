import math

import numpy as np
import pytest

from gnomepy_research.sessions.cs2_win_probability.map_dp import (
    DEFAULT_FORMAT,
    MapFormat,
    logit,
    map_win_prob,
    map_win_prob_from_pre_map,
    overtime_value,
    side_probs,
    sigmoid,
    solve_theta,
)


def test_even_teams_even_map_is_a_coinflip():
    assert map_win_prob(0.5, 0.5, 0, 0, True) == pytest.approx(0.5)
    assert map_win_prob(0.5, 0.5, 6, 6, True) == pytest.approx(0.5)


def test_twelve_all_is_overtime_not_a_single_round():
    # the previous implementation returned p here, pricing MR3 overtime as one round
    assert map_win_prob(0.5, 0.5, 12, 12, True) == pytest.approx(0.5)
    dominant = map_win_prob(0.9, 0.9, 12, 12, True)
    assert dominant > 0.99, "MR3 overtime must compound a per-round edge, not apply it once"
    assert dominant != pytest.approx(0.9, abs=1e-3)
    assert overtime_value(0.5, 0.5, True) == pytest.approx(0.5)


def test_terminal_states():
    assert map_win_prob(0.3, 0.3, 13, 5, True) == 1.0
    assert map_win_prob(0.9, 0.9, 5, 13, True) == 0.0


def test_monotone_in_score_and_strength():
    p = [map_win_prob(0.55, 0.55, a, 6, True) for a in range(0, 13)]
    assert all(x < y for x, y in zip(p, p[1:]))
    q = [map_win_prob(s, s, 6, 6, True) for s in (0.40, 0.45, 0.50, 0.55, 0.60)]
    assert all(x < y for x, y in zip(q, q[1:]))


def test_starting_side_is_irrelevant_at_zero_zero():
    """
    A map cannot be won inside 12 rounds, so both halves always complete and each
    ordering deals a team the same 12 CT and 12 T rounds. The starting side is
    therefore worth exactly nothing before the first round.
    """
    p_ct, p_t = side_probs(0.0, logit(0.56))
    assert map_win_prob(p_ct, p_t, 0, 0, True) == pytest.approx(
        map_win_prob(p_ct, p_t, 0, 0, False), abs=1e-12
    )


def test_starting_side_matters_mid_map():
    """Once the halves are unbalanced, which side is still to come dominates."""
    p_ct, p_t = side_probs(0.0, logit(0.56))
    started_ct = map_win_prob(p_ct, p_t, 6, 6, True)
    started_t = map_win_prob(p_ct, p_t, 6, 6, False)
    # level at 6-6 having already spent the favourable CT half is much worse
    assert started_ct < 0.40 < 0.60 < started_t
    assert started_ct + started_t == pytest.approx(1.0, abs=1e-9)


def test_overtime_fixed_point_matches_brute_force_unrolling():
    p_ct, p_t = side_probs(0.25, logit(0.54))
    exact = overtime_value(p_ct, p_t, True)

    # unroll many periods explicitly; the closed form is the limit
    from gnomepy_research.sessions.cs2_win_probability.map_dp import _period_value
    value = 0.5
    for _ in range(400):
        value = _period_value(p_ct, p_t, True, 0, value, DEFAULT_FORMAT)
    assert exact == pytest.approx(value, abs=1e-9)


def test_inversion_round_trips():
    delta = logit(0.545)
    for target in (0.05, 0.2, 0.5, 0.7, 0.95):
        for started in (True, False, None):
            theta = solve_theta(target, delta, started)
            p_ct, p_t = side_probs(theta, delta)
            if started is None:
                got = 0.5 * (map_win_prob(p_ct, p_t, 0, 0, True)
                             + map_win_prob(p_ct, p_t, 0, 0, False))
            else:
                got = map_win_prob(p_ct, p_t, 0, 0, started)
            assert got == pytest.approx(target, abs=1e-6)


def test_pre_map_prob_is_reproduced_at_zero_zero():
    for target in (0.25, 0.5, 0.8):
        got = map_win_prob_from_pre_map(target, logit(0.53), 0, 0, True)
        assert got == pytest.approx(target, abs=1e-6)


def test_no_overtime_format_uses_draw_value():
    fmt = MapFormat(overtime=False, draw_value=0.5)
    assert map_win_prob(0.9, 0.9, 12, 12, True, fmt) == pytest.approx(0.5)


def test_ot_offset_does_not_leak_into_regulation():
    p_ct, p_t = side_probs(0.3, logit(0.57))
    for a, b in ((0, 0), (6, 6), (11, 9)):
        assert map_win_prob(p_ct, p_t, a, b, True, DEFAULT_FORMAT, 0) == pytest.approx(
            map_win_prob(p_ct, p_t, a, b, True, DEFAULT_FORMAT, 1), abs=1e-12
        )


def test_ot_offset_matters_only_inside_an_unbalanced_period():
    """
    Entering overtime is parity-free for the same reason 0-0 is: a period needs 4
    to win over 3+3 rounds, so both overtime halves always complete. Parity only
    bites once a period is part-played.
    """
    p_ct, p_t = side_probs(0.3, logit(0.60))
    assert map_win_prob(p_ct, p_t, 12, 12, True, DEFAULT_FORMAT, 0) == pytest.approx(
        map_win_prob(p_ct, p_t, 12, 12, True, DEFAULT_FORMAT, 1), abs=1e-12
    )
    assert map_win_prob(p_ct, p_t, 13, 12, True, DEFAULT_FORMAT, 0) != pytest.approx(
        map_win_prob(p_ct, p_t, 13, 12, True, DEFAULT_FORMAT, 1), abs=1e-6
    )


def test_overtime_scorelines_are_priced():
    """Reaching 13 in overtime does not win the map; the period must be won."""
    assert map_win_prob(0.5, 0.5, 13, 12, True) == pytest.approx(0.65625)
    assert map_win_prob(0.5, 0.5, 13, 13, True) == pytest.approx(0.5)
    assert map_win_prob(0.5, 0.5, 15, 15, True) == pytest.approx(0.5)  # 3-3 starts a fresh period
    assert map_win_prob(0.5, 0.5, 15, 12, True) == pytest.approx(1 - 0.5 ** 4)
    assert map_win_prob(0.5, 0.5, 16, 12, True) == 1.0
    assert map_win_prob(0.5, 0.5, 12, 16, True) == 0.0


def test_calibrated_probabilities_never_saturate():
    """An exact 0/1 on a map winner makes log-loss and Kelly sizing degenerate."""
    import numpy as np

    from gnomepy_research.sessions.cs2_win_probability.pre_map_model import (
        _PROB_EPS,
        _calibrated,
    )

    raw = np.array([0.0, 1e-9, 0.5, 1.0 - 1e-9, 1.0])
    for temperature in (0.5, 1.0, 2.0):
        out = _calibrated(temperature, raw)
        assert out.min() >= _PROB_EPS
        assert out.max() <= 1.0 - _PROB_EPS


def test_temperature_scaling_is_exactly_symmetric():
    """
    Scaling log-odds commutes with p -> 1-p, so calibration cannot undo the
    A/B order-invariance. Isotonic needed a both-orientations fit to approximate
    this; temperature scaling gets it to machine precision.
    """
    import numpy as np

    from gnomepy_research.sessions.cs2_win_probability.pre_map_model import _calibrated

    p = np.linspace(0.02, 0.98, 49)
    for temperature in (0.5, 0.9, 1.0, 1.4, 3.0):
        out = _calibrated(temperature, p)
        mirror = _calibrated(temperature, 1.0 - p)
        np.testing.assert_allclose(out + mirror, 1.0, atol=1e-12)


def test_temperature_recovers_a_known_distortion():
    """A deliberately over-confident input should fit T > 1 and be pulled back."""
    import numpy as np

    from gnomepy_research.sessions.cs2_win_probability.pre_map_model import (
        _calibrated,
        _fit_temperature,
    )

    rng = np.random.default_rng(0)
    true_p = rng.uniform(0.15, 0.85, 4000)
    y = (rng.uniform(size=4000) < true_p).astype(int)
    over = 1.0 / (1.0 + np.exp(-np.log(true_p / (1 - true_p)) * 1.6))

    temperature = _fit_temperature(over, y)
    assert temperature > 1.2, temperature
    from sklearn.metrics import log_loss
    assert log_loss(y, _calibrated(temperature, over)) < log_loss(y, np.clip(over, 0.01, 0.99))
