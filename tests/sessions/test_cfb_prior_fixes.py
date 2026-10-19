from __future__ import annotations

import numpy as np
import pandas as pd

from gnomepy_research.models.cfb.efficiency.data import (
    RETURNING_FEATURES, TRANSITION_MASKED, UNITS_SPLIT, build_model_data, first_fbs_seasons, prior_fixes)
from gnomepy_research.models.cfb.efficiency.posterior_snapshots import draws_name, hyper_name
from gnomepy_research.sessions.cfb_win_probability.evaluation import n_selection
from gnomepy_research.sessions.cfb_win_probability.evaluation.backtest import MODEL_CONFIGS
from tests.sessions.test_cfb_win_probability_models import SEASON, synthetic


def schedule(first_fbs: dict[int, int], seasons=range(2018, 2026)) -> pd.DataFrame:
    """One FBS-vs-FBS game per team-season from its first FBS season on (FCS before that)."""
    rows, gid = [], 0
    for t, first in first_fbs.items():
        for s in seasons:
            gid += 1
            rows.append({"game_id": gid, "season": s, "home_id": t, "away_id": 999,
                         "home_classification": "fbs" if s >= first else "fcs", "away_classification": "fbs"})
    return pd.DataFrame(rows)


def preseason(teams, seasons=range(2019, 2026)) -> pd.DataFrame:
    rows = [{"season": s, "team_id": t, "prior_off_rating": 0.01, "talent_offense": 500.0, "talent_defense": 500.0,
             "talent_qb": 80.0, "recruiting_points": 200.0, "returning_pct_ppa": 0.6, "returning_pct_passing_ppa": 0.5,
             "qb_returning_starter": True}
            for t in teams for s in seasons]
    return pd.DataFrame(rows)


class TestPriorFixes:
    def test_first_fbs_season(self):
        assert first_fbs_seasons(schedule({1: 2018, 2: 2022})).to_dict() == {1: 2018, 2: 2022, 999: 2018}

    def test_fcs_move_up_masked_for_three_seasons_only(self):
        out = prior_fixes(preseason([1, 2]), schedule({1: 2018, 2: 2022})).set_index(["team_id", "season"])
        for s in (2022, 2023, 2024):
            assert out.loc[(2, s), TRANSITION_MASKED].isna().all()
        assert out.loc[(2, 2025), TRANSITION_MASKED].notna().all()
        assert out.loc[(2, 2021), TRANSITION_MASKED].notna().all()  # still FCS: not an FBS team-season yet
        assert out.loc[1, TRANSITION_MASKED].notna().all().all()  # FBS from the first season on record

    def test_returning_production_unknown_without_prior_season(self):
        pre = preseason([1])
        pre.loc[pre["season"] == 2021, "prior_off_rating"] = np.nan
        out = prior_fixes(pre, schedule({1: 2018})).set_index("season")
        assert out.loc[2021, ["returning_pct_ppa", "returning_pct_passing_ppa"]].isna().all()
        assert out.loc[2021, ["talent_offense", "recruiting_points"]].notna().all()
        assert out.loc[2022, "returning_pct_ppa"] == 0.6

    def test_returning_production_from_a_stale_roster_is_unknown(self):
        # CFBD says 90% of passing PPA returns, yet last season's primary passer is not on this season's roster.
        pre = preseason([1, 2, 3], seasons=[2024]).assign(qb_returning_starter=[False, True, False],
                                                           returning_pct_passing_ppa=[0.9, 0.9, 0.5])
        out = prior_fixes(pre, schedule({1: 2018, 2: 2018, 3: 2018})).set_index("team_id")
        assert out.loc[1, RETURNING_FEATURES].isna().all()
        assert out.loc[2, "returning_pct_passing_ppa"] == 0.9 and out.loc[3, "returning_pct_passing_ppa"] == 0.5

    def test_talent_missing_indicator_ignores_transition_masking(self):
        pre = preseason([1, 2])
        pre.loc[pre["team_id"] == 1, "talent_offense"] = np.nan
        out = prior_fixes(pre, schedule({1: 2018, 2: 2022}))
        assert (out.loc[out["team_id"] == 1, "talent_missing"] == 1.0).all()
        assert (out.loc[out["team_id"] == 2, "talent_missing"] == 0.0).all()

    def test_input_not_mutated(self):
        pre = preseason([2])
        prior_fixes(pre, schedule({2: 2022}))
        assert pre[TRANSITION_MASKED].notna().all().all() and "talent_missing" not in pre


def test_model_data_adds_indicator_only_with_prior_extension():
    games, eff, pre, _, _ = synthetic()
    plain = build_model_data(pre, eff, games, [SEASON], units=UNITS_SPLIT)
    fixed = build_model_data(pre, eff, games, [SEASON], units=UNITS_SPLIT, extensions=("prior",))
    assert not any("talent_missing" in n for n in plain.off_names)
    assert "pass:talent_missing" in fixed.off_names and "rush:talent_missing" in fixed.def_names
    assert fixed.x_off.shape[1] == plain.x_off.shape[1] + 2
    keep = [i for i, n in enumerate(fixed.off_names) if "talent_missing" not in n]
    np.testing.assert_allclose(fixed.x_off[:, keep], plain.x_off)  # no transitions or gaps in the synthetic season


def test_naming_configs_and_frozen_ladder():
    assert hyper_name("normal", 2023, UNITS_SPLIT, True, drives=True, independents_separate=True,
                      extensions=("prior",)) == "normal_split_conf_ind_drv_pfx_2019_2022"
    assert draws_name(2024, UNITS_SPLIT, True, drives=True, independents_separate=True,
                      extensions=("prior", "couple", "fcs", "finish")) == \
        "cfb_team_week_draws_split_conf_ind_drv_cpl_fcs_fin_pfx_2024"
    assert hyper_name("normal", 2023, extensions=("prior",)) == "normal_pfx_2019_2022"
    assert MODEL_CONFIGS["Kp"] == {**MODEL_CONFIGS["K"], "extensions": ("prior",)}
    assert MODEL_CONFIGS["Np"] == {**MODEL_CONFIGS["N"], "extensions": ("couple", "fcs", "finish", "prior")}
    assert n_selection.LADDER == ("couple", "fcs", "finish")
    assert n_selection.ladder_step(("couple", "fcs"), "finish") == ("couple", "fcs", "finish")


def test_expected_starters_zeroes_only_scored_seasons():
    from gnomepy_research.sessions.cfb_win_probability.evaluation.backtest import expected_starters
    games = pd.DataFrame({"game_id": [1, 2, 3], "season": [2024, 2025, 2025]})
    qb = pd.DataFrame({"game_id": [1, 2, 3], "home_qb_delta": [0.1, 0.2, -0.3], "away_qb_delta": [-0.1, 0.0, 0.4]})
    out = expected_starters(qb, games, [2025])
    assert out["home_qb_delta"].tolist() == [0.1, 0.0, 0.0] and out["away_qb_delta"].tolist() == [-0.1, 0.0, 0.0]
    assert qb["home_qb_delta"].tolist() == [0.1, 0.2, -0.3]


def test_calibrator_seasons_guard(tmp_path):
    import pytest
    from gnomepy_research.sessions.cfb_win_probability.evaluation.backtest import load_calibrator
    cfg = {"calibrator": "tail", "calibrator_base": "K"}
    with pytest.raises(SystemExit, match="2023, 2024"):
        load_calibrator(cfg, [2024], results=tmp_path, cal_seasons=[2023, 2024])
    with pytest.raises(SystemExit, match="missing"):
        load_calibrator(cfg, [2025], results=tmp_path, cal_seasons=[2024, 2023])


class TestTightPrior:
    def test_names_share_hyperparameters_but_not_draws(self):
        np_ext = ("couple", "fcs", "finish", "prior")
        kw = dict(units=UNITS_SPLIT, conference=True, drives=True, independents_separate=True)
        assert hyper_name("normal", 2023, **kw, extensions=np_ext + ("tight",)) == hyper_name("normal", 2023, **kw, extensions=np_ext)
        assert draws_name(2024, **kw, extensions=np_ext + ("tight",)) == \
            "cfb_team_week_draws_split_conf_ind_drv_cpl_fcs_fin_pfx_tgt_2024"
        assert MODEL_CONFIGS["Npt"] == {**MODEL_CONFIGS["Np"], "extensions": np_ext + ("tight",)}

    def test_model_data_ignores_fit_time_extension(self):
        games, eff, pre, _, _ = synthetic()
        a = build_model_data(pre, eff, games, [SEASON], extensions=("prior",))
        b = build_model_data(pre, eff, games, [SEASON], extensions=("prior", "tight"))
        assert a.extensions == b.extensions == ("prior",)
        np.testing.assert_array_equal(a.x_off, b.x_off)

    def test_snapshot_scales_prior_spread_only_with_tight(self, monkeypatch, tmp_path):
        from dataclasses import replace
        from gnomepy_research.models.cfb.efficiency import posterior_snapshots as ps
        from gnomepy_research.models.cfb.efficiency.data import TIGHT_PRIOR_SCALE
        from tests.sessions.test_cfb_win_probability_models import hypers
        games, eff, pre, _, _ = synthetic()
        h0 = replace(hypers(build_model_data(pre, eff, games, [SEASON])), likelihood="normal")
        seen = []
        monkeypatch.setattr(ps, "ensure_hyperparameters", lambda *a, **k: "x")
        monkeypatch.setattr(ps, "load_hyperparameters", lambda name: h0)

        def fake_fit(inputs, season, cutoff, h, seed=0):
            seen.append((h.s_off, h.s_def))
            raise StopIteration

        monkeypatch.setattr(ps, "fit_cutoff_exact", fake_fit)
        from gnomepy_research.models.cfb.efficiency.inference import Inputs
        inputs = Inputs(pre, eff, games, as_of=pd.Timestamp("2025-01-01", tz="UTC").to_pydatetime())
        for ext in ((), ("tight",)):
            try:
                ps.snapshot_season(inputs, SEASON, "normal", out_dir=tmp_path, extensions=ext)
            except StopIteration:
                pass
        assert seen[0] == (h0.s_off, h0.s_def)
        assert seen[1] == (h0.s_off * TIGHT_PRIOR_SCALE, h0.s_def * TIGHT_PRIOR_SCALE)
