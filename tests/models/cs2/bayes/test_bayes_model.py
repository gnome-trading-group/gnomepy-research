import numpy as np
import pandas as pd
import pytest

from gnomepy_research.models.cs2.bayes.data import build_fit_data, prepare_maps
from gnomepy_research.models.cs2.bayes.model import BayesSeriesConfig, build_model, posterior_from, sample
from gnomepy_research.models.cs2.hltv.bayes_series import RankStrength

DAY0 = pd.Timestamp("2026-01-01")
# Ranked in skill order: team 6 (strongest) is #1.
RANKS = RankStrength(pd.DataFrame({"date": [DAY0] * 6, "team_id": [1, 2, 3, 4, 5, 6], "rank": [60, 40, 25, 12, 5, 1]}))


def _world(n_teams=6, n_days=60, seed=0):
    """Teams of five fixed players with known skills; each day pairs teams for one map, rounds drawn binomially."""
    rng = np.random.default_rng(seed)
    skill = {p: v for p, v in zip(range(n_teams * 5), np.repeat(np.linspace(-0.4, 0.4, n_teams), 5) + rng.normal(0, 0.05, n_teams * 5))}
    hist, lineups, stats = [], [], []
    mid = 0
    for day in range(n_days):
        order = rng.permutation(n_teams)
        for a, b in zip(order[::2], order[1::2]):
            mid += 1
            fa, fb = list(range(a * 5, a * 5 + 5)), list(range(b * 5, b * 5 + 5))
            d = np.mean([skill[p] for p in fa]) - np.mean([skill[p] for p in fb])
            won_a = 0
            while True:
                won_a = rng.binomial(24, 1 / (1 + np.exp(-d)))
                if won_a != 12:
                    break
            sa, sb = (13, max(0, 24 - won_a)) if won_a > 12 else (won_a, 13)
            sa, sb = (13, min(sb, 11)) if won_a > 12 else (min(sa, 11), 13)
            hist.append({"match_id": mid, "match_date": DAY0 + pd.Timedelta(days=day), "map_name": "de_nuke",
                         "team_a_id": a + 1, "team_b_id": b + 1, "team_a_score": sa, "team_b_score": sb,
                         "team_a_won": int(sa > sb), "bo_type": 1, "event_name": "Test Europe Cup"})
            for team, five in ((a + 1, fa), (b + 1, fb)):
                mean = np.mean([skill[p] for p in five])
                for slot, p in enumerate(five):
                    lineups.append({"match_id": mid, "team_id": team, "player_id": p, "slot": slot, "is_standin": False})
                    stats.append({"match_id": mid, "map_name": "de_nuke", "player_id": p, "team_id": team, "side": "all",
                                  "rating": 1.05 + 0.3 * (skill[p] - mean) + rng.normal(0, 0.05)})
    return pd.DataFrame(hist), pd.DataFrame(lineups), pd.DataFrame(stats), skill


def test_overtime_counts_as_twelve_all_and_ties_are_dropped():
    hist, lineups, _, _ = _world(n_days=2)
    hist = pd.concat([hist, hist.iloc[[0]].assign(match_id=999, team_a_score=16, team_b_score=14),
                      hist.iloc[[0]].assign(match_id=998, team_a_score=1, team_b_score=1)])
    lineups = pd.concat([lineups, lineups[lineups.match_id == 1].assign(match_id=999),
                         lineups[lineups.match_id == 1].assign(match_id=998)])
    maps = prepare_maps(hist, lineups).set_index("match_id")
    assert maps.loc[999, "rounds_a"] == 12 and maps.loc[999, "rounds"] == 24
    assert 998 not in maps.index
    assert maps.loc[1, "rounds"] == maps.loc[1, "team_a_score"] + maps.loc[1, "team_b_score"]


def test_fit_window_excludes_its_end_and_ratings_center_within_team():
    hist, lineups, stats, _ = _world(n_days=20)
    maps = prepare_maps(hist, lineups)
    end = DAY0 + pd.Timedelta(days=10)
    data = build_fit_data(maps, stats, RANKS, DAY0, end)
    assert len(data.rounds) == (maps.day < end).sum()
    devs = pd.DataFrame({"lineup": [tuple(sorted(r)) for r in data.rating_lineup], "dev": data.rating_dev})
    assert np.abs(devs.groupby("lineup").dev.sum()).max() < 1e-9
    assert data.rating_lineup.shape == (len(data.rating_dev), 5)


def test_debutants_are_players_first_seen_inside_the_window():
    hist, lineups, stats, _ = _world(n_days=120)
    late = hist.match_id.max()
    lineups.loc[(lineups.match_id == late) & (lineups.slot == 0), "player_id"] = 777
    maps = prepare_maps(hist, lineups)
    data = build_fit_data(maps, stats, RANKS, DAY0 + pd.Timedelta(days=90), DAY0 + pd.Timedelta(days=121))
    flagged = dict(zip(data.player_ids, data.debutant))
    assert flagged[777] and not any(v for p, v in flagged.items() if p != 777)
    assert 777 in set(data.career_ids) and 0 in set(data.career_ids)


@pytest.mark.parametrize("debut_prior", [False, True])
def test_fit_recovers_player_skill_order(debut_prior):
    hist, lineups, stats, skill = _world()
    maps = prepare_maps(hist, lineups)
    data = build_fit_data(maps, stats, RANKS, DAY0, DAY0 + pd.Timedelta(days=60))
    cfg = BayesSeriesConfig(draws=200, tune=300, chains=2, debut_prior=debut_prior)
    post = posterior_from(sample(build_model(data, cfg), cfg), data, cfg, DAY0 + pd.Timedelta(days=60))
    est = pd.Series(post.theta.mean(axis=0), index=post.player_ids)
    truth = pd.Series(skill).reindex(est.index)
    assert np.corrcoef(est, truth)[0, 1] > 0.8
    team = est.index // 5
    within = np.corrcoef(est - est.groupby(team).transform("mean"), truth - truth.groupby(team).transform("mean"))[0, 1]
    assert within > 0.6, "ratings against teammates should order players inside a fixed five"
    assert post.price(6, list(range(25, 30)), RANKS(6, DAY0), 1, list(range(0, 5)), RANKS(1, DAY0)) > 0.6


def _extra(rows):
    return pd.DataFrame(rows, columns=["date", "team_a_id", "team_b_id", "maps_a", "maps_b", "roster_a", "roster_b"])


def test_extra_results_need_the_window_and_enough_known_players():
    hist, lineups, stats, _ = _world(n_days=20)
    maps = prepare_maps(hist, lineups)
    start, end = DAY0 + pd.Timedelta(days=5), DAY0 + pd.Timedelta(days=20)
    six = [0, 1, 2, 3, 4, 900]
    extra = _extra([
        (start - pd.Timedelta(days=1), 1, 2, 2, 0, six, [5, 6, 7, 8, 9]),
        (end, 1, 2, 2, 0, six, [5, 6, 7, 8, 9]),
        (start, 1, 2, 2, 1, [0, 1, 901, 902, 903], [5, 6, 7, 8, 9]),
        (start, -1, 2, 1, 2, six, [5, 6, 7, 8, 9]),
    ])
    data = build_fit_data(maps, stats, RANKS, start, end, extra)
    assert len(data.extra_maps) == 1 and data.extra_maps[0] == 3 and data.extra_maps_a[0] == 1
    known_a = data.player_ids[data.extra_lineup_a[0][data.extra_weight_a[0] > 0]]
    assert sorted(known_a) == [0, 1, 2, 3, 4], "a roster player the fit has never seen is left out"
    assert np.allclose(data.extra_weight_a.sum(axis=1), 1) and np.allclose(data.extra_weight_b.sum(axis=1), 1)

    plain = build_fit_data(maps, stats, RANKS, start, end)
    assert len(plain.extra_maps) == 0
    assert "eps_extra_z" not in build_model(plain, BayesSeriesConfig()).named_vars


def test_extra_results_move_a_thinly_covered_team_toward_its_skill():
    hist, lineups, stats, skill = _world(n_days=60)
    # Team 6, the strongest, keeps two HLTV maps; its other results arrive only as extra bo3 map counts.
    thin = hist[(hist.team_a_id == 6) | (hist.team_b_id == 6)].match_id
    drop = set(thin.iloc[2:])
    hist, lineups, stats = (x[~x.match_id.isin(drop)] for x in (hist, lineups, stats))
    rng = np.random.default_rng(1)
    team = {t: list(range((t - 1) * 5, t * 5)) for t in range(1, 7)}
    rows = []
    for day in range(60):
        b = int(rng.integers(1, 6))
        d = np.mean([skill[p] for p in team[6]]) - np.mean([skill[p] for p in team[b]])
        p_map = 1 / (1 + np.exp(-3.22 * d))
        a_maps = b_maps = 0
        while a_maps < 2 and b_maps < 2:
            a_maps, b_maps = (a_maps + 1, b_maps) if rng.random() < p_map else (a_maps, b_maps + 1)
        rows.append((DAY0 + pd.Timedelta(days=day), 6, b, a_maps, b_maps, team[6], team[b]))
    maps = prepare_maps(hist, lineups)
    end = DAY0 + pd.Timedelta(days=60)
    cfg = BayesSeriesConfig(draws=200, tune=300, chains=2)

    def team6(extra):
        data = build_fit_data(maps, stats, RANKS, DAY0, end, extra)
        post = posterior_from(sample(build_model(data, cfg), cfg), data, cfg, end)
        est = pd.Series(post.theta.mean(axis=0), index=post.player_ids)
        return est[team[6]].mean() - est.drop(team[6]).mean()

    truth = np.mean([skill[p] for p in team[6]]) - np.mean([v for p, v in skill.items() if p not in team[6]])
    without, with_extra = team6(None), team6(_extra(rows))
    assert abs(with_extra - truth) < abs(without - truth)
