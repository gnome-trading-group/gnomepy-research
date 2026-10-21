import numpy as np
import pandas as pd
import pytest

from gnomepy_research.models.cs2.hltv.bayes_series import (
    UNKNOWN_REGION,
    RankStrength,
    SeriesBlend,
    SeriesPosterior,
    event_region,
    home_regions,
)
from gnomepy_research.models.cs2.hltv.player_skill import regulation_map_win

S = 400
A_FIVE, B_FIVE = [1, 2, 3, 4, 5], [6, 7, 8, 9, 10]


def _posterior(skill_a=0.0, skill_b=0.0, sigma_s=0.15, rho_na=0.0, beta=(0.0, 0.0), a_new=0.0, rho_lan_na=None):
    rng = np.random.default_rng(0)
    theta = np.concatenate([np.full((S, 5), skill_a), np.full((S, 5), skill_b)], axis=1) + 0.01 * rng.standard_normal((S, 10))
    rho = np.zeros((S, 7))
    rho[:, 2] = rho_na
    return SeriesPosterior(
        player_ids=np.arange(1, 11), theta=theta, tau=0.055, b=np.full(S, 0.35), sigma0=np.full(S, 0.5),
        strength_mu=-4.0, strength_sd=1.0, rho=rho, sigma_m=np.full(S, 0.37), sigma_s=np.full(S, sigma_s),
        home_team_ids=np.array([100, 200]), home_region=np.array([1, 2]),
        tm_team_ids=np.array([100, 200]), tm_maps=np.array(["de_nuke", "de_nuke"]), beta=np.tile(np.array(beta), (S, 1)),
        plays_team_ids=np.array([100, 200]), plays_maps=np.array(["de_nuke", "de_nuke"]), plays_counts=np.array([4, 4]),
        a_new=np.full(S, a_new), sigma_new=np.full(S, 0.4), career_ids=np.array([*range(1, 11), 555]),
        as_of="2026-10-01",
        rho_lan=None if rho_lan_na is None else np.zeros((S, 7)) + np.eye(7)[2] * rho_lan_na,
    )


def _price(post, five_a=A_FIVE, five_b=B_FIVE, sa=-4.0, sb=-4.0):
    return post.price(100, five_a, sa, 200, five_b, sb)


def test_regulation_map_win_is_symmetric_and_amplifies_round_edges():
    d = np.array([-1.0, -0.2, 0.0, 0.2, 1.0])
    p = regulation_map_win(d)
    assert p[2] == pytest.approx(0.5)
    np.testing.assert_allclose(p + p[::-1], 1.0)
    assert regulation_map_win(np.array(0.2)) > 1 / (1 + np.exp(-0.2))


def test_equal_lineups_price_even_and_swapping_sides_complements():
    assert _price(_posterior()) == pytest.approx(0.5, abs=1e-3)
    post = _posterior(skill_a=0.3)
    swapped = post.price(200, B_FIVE, -4.0, 100, A_FIVE, -4.0)
    assert _price(post) == pytest.approx(1 - swapped, abs=1e-9)
    assert _price(post) > 0.6


def test_series_noise_pulls_prices_toward_even():
    assert _price(_posterior(skill_a=0.3, sigma_s=0.6)) < _price(_posterior(skill_a=0.3, sigma_s=0.05))


def test_region_and_map_terms_move_the_gap():
    base, _ = _posterior().gap(100, A_FIVE, -4.0, 200, B_FIVE, -4.0)
    na_weaker, _ = _posterior(rho_na=-0.3).gap(100, A_FIVE, -4.0, 200, B_FIVE, -4.0)
    nuke_a, _ = _posterior(beta=(0.2, 0.0)).gap(100, A_FIVE, -4.0, 200, B_FIVE, -4.0)
    assert na_weaker.mean() - base.mean() == pytest.approx(0.3, abs=1e-6)
    assert nuke_a.mean() - base.mean() == pytest.approx(0.2, abs=1e-6)


def test_lan_region_offsets_apply_on_lan_only_and_survive_saving(tmp_path):
    post = _posterior(rho_lan_na=-0.4)
    online, _ = post.gap(100, A_FIVE, -4.0, 200, B_FIVE, -4.0)
    lan, _ = post.gap(100, A_FIVE, -4.0, 200, B_FIVE, -4.0, is_lan=True)
    assert lan.mean() - online.mean() == pytest.approx(0.4, abs=1e-6), "team B is North American"
    post.save(tmp_path / "p.npz")
    again = SeriesPosterior.load(tmp_path / "p.npz")
    assert again.price(100, A_FIVE, -4.0, 200, B_FIVE, -4.0, is_lan=True) == pytest.approx(
        post.price(100, A_FIVE, -4.0, 200, B_FIVE, -4.0, is_lan=True))
    assert _posterior().gap(100, A_FIVE, -4.0, 200, B_FIVE, -4.0, is_lan=True)[0].mean() == pytest.approx(online.mean())


def test_unseen_player_takes_the_newcomer_prior_and_widens_the_gap():
    post = _posterior()
    known_mean, known_var = post.gap(100, A_FIVE, -4.0, 200, B_FIVE, -4.0)
    mean, var = post.gap(100, [1, 2, 3, 4, 999], -2.0, 200, B_FIVE, -4.0)
    assert mean.mean() - known_mean.mean() == pytest.approx(0.35 * 2.0 / 5, abs=0.01)
    assert (var > known_var).all()


def test_debutant_takes_the_debut_prior_and_a_returning_player_the_regular_one():
    post = _posterior(a_new=-0.35)
    base, _ = post.gap(100, [1, 2, 3, 4, 999], -4.0, 200, B_FIVE, -4.0)
    returning, _ = post.gap(100, [1, 2, 3, 4, 555], -4.0, 200, B_FIVE, -4.0)
    assert base.mean() - returning.mean() == pytest.approx(-0.35 / 5, abs=1e-6)


def test_save_and_load_price_identically(tmp_path):
    post = _posterior(skill_a=0.2, rho_na=-0.1)
    post.save(tmp_path / "p.npz")
    again = SeriesPosterior.load(tmp_path / "p.npz")
    assert _price(again) == _price(post)


def test_thinning_keeps_prices_and_shrinks_the_draws():
    post = _posterior(skill_a=0.3)
    thin = post.thinned(100)
    assert thin.n_draws == 100 and thin.theta.dtype == np.float32 and thin.as_of == post.as_of
    assert _price(thin) == pytest.approx(_price(post), abs=0.005)


def test_age_is_counted_from_the_fit_windows_end():
    assert _posterior().age_days(pd.Timestamp("2026-10-15 13:00", tz="UTC")) == 14


def test_event_region_and_home_regions():
    assert event_region("ESL Challenger League Season 51 Europe Cup 4") == 1
    assert event_region("CCT North America Series") == 2
    assert event_region("IEM Cologne Major 2026") == UNKNOWN_REGION
    series = pd.DataFrame({"event_name": ["X Europe", "Y Europe", "Z North America", "IEM Cologne"],
                           "team_a_id": [1, 1, 1, 1], "team_b_id": [2, 3, 4, 5]})
    homes = home_regions(series)
    assert homes[1] == 1 and homes[4] == 2 and 5 not in homes


def test_rank_strength_treats_stale_rankings_as_unranked():
    ranks = RankStrength(pd.DataFrame({"date": ["2026-01-05"], "team_id": [7], "rank": [10]}))
    assert ranks(7, pd.Timestamp("2026-02-01")) == pytest.approx(-np.log(10))
    assert ranks(7, pd.Timestamp("2026-06-01")) == pytest.approx(-np.log(300))
    assert ranks(8, pd.Timestamp("2026-02-01")) == pytest.approx(-np.log(300))


def test_blend_recovers_weights_and_stays_symmetric():
    rng = np.random.default_rng(1)
    za, zb = rng.normal(0, 1, 20000), rng.normal(0, 1, 20000)
    y = (rng.random(20000) < 1 / (1 + np.exp(-(0.4 * za + 0.7 * zb)))).astype(float)
    blend = SeriesBlend.fit(1 / (1 + np.exp(-za)), 1 / (1 + np.exp(-zb)), y)
    assert blend.w_bayes == pytest.approx(0.4, abs=0.06) and blend.w_xgb == pytest.approx(0.7, abs=0.06)
    p = blend.predict(np.array([0.7]), np.array([0.6]))
    assert p + blend.predict(np.array([0.3]), np.array([0.4])) == pytest.approx(1.0)
    assert SeriesBlend.from_dict(blend.to_dict()) == blend
