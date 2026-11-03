from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from gnomepy_research.models.cfb.features.coach_effects import coach_features
from gnomepy_research.models.cfb.features.cutoffs import game_cutoffs, week_cutoffs
from gnomepy_research.models.cfb.features.ratings import FCS_TEAM, ridge_ratings
from gnomepy_research.models.cfb.features.source import SourceData, with_coach_availability

UTC = "UTC"


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz=UTC)


def make_games() -> pd.DataFrame:
    rows = [
        (1, 2024, 1, "2024-08-31 16:00", 10, 20, 31, 17),
        (2, 2024, 1, "2024-09-01 00:00", 30, 40, 21, 24),
        (3, 2024, 2, "2024-09-07 16:00", 10, 30, 14, 10),
    ]
    return pd.DataFrame([{
        "game_id": gid, "season": s, "week": w, "season_type": "regular", "kickoff": ts(k), "completed": True,
        "home_id": h, "away_id": a, "home_team": f"T{h}", "away_team": f"T{a}", "home_points": hp,
        "away_points": ap, "periods": 4, "home_pregame_elo": 1500, "away_pregame_elo": 1500,
        "home_classification": "fbs", "away_classification": "fbs", "neutral_site": False,
    } for gid, s, w, k, h, a, hp, ap in rows])


def make_coaches() -> pd.DataFrame:
    # Team 10: coach 1 continues from 2023 and is fired mid-2024; coach 2 is the interim.
    return coach_rows([
        (1, 10, 2023, "2019-12-01", 12, 5.0),
        (1, 10, 2024, "2019-12-01", 6, 2.0),
        (2, 10, 2024, "2024-10-20", 6, 2.0),
    ])


def coach_rows(rows: list[tuple]) -> pd.DataFrame:
    df = pd.DataFrame([{
        "coach_id": cid, "team_id": tid, "season": s, "hire_date": ts(h), "games": g, "wins": 0, "losses": 0,
        "srs": srs, "sp_overall": 0.0, "sp_offense": 0.0, "sp_defense": 0.0, "postseason_rank": None,
        "results_available_at": ts(f"{s + 1}-02-01"),
    } for cid, tid, s, h, g, srs in rows])
    df["team_id"] = df["team_id"].astype("Int64")
    return df


def make_source() -> SourceData:
    games = make_games()
    empty_eff = pd.DataFrame({
        "game_id": [1], "season": [2024], "team_id": [10], "opponent_id": [20], "plays": [60], "off_ppa": [0.2],
        "available_at": [ts("2024-09-01 16:00")], "usable": [True],
    })
    portal = pd.DataFrame({
        "season": [2024, 2024], "origin_id": [20, 30], "destination_id": [10, 40], "destination": ["T10", "T40"],
        "origin_available_at": [ts("2024-01-05"), ts("2024-09-05")],
        "destination_available_at": [ts("2024-08-30"), ts("2024-08-30")],
    })

    def avail(df: pd.DataFrame, at: str) -> pd.DataFrame:
        return df.assign(available_at=ts(at))

    t = {
        "cfb_games": games,
        "cfb_team_game_efficiency": empty_eff,
        "cfb_passers": pd.DataFrame({"game_id": [1], "season": [2024], "offense": ["T10"], "passer_key": ["a.b"],
                                     "ppa": [0.1], "play_seq": [1]}),
        "cfb_talent": avail(pd.DataFrame({"season": [2024], "team_id": [10], "talent": [700.0]}), "2024-08-30"),
        "cfb_recruiting_teams": avail(pd.DataFrame({"class_year": [2025], "team_id": [10], "points": [200.0]}), "2025-02-15"),
        "cfb_recruits": avail(pd.DataFrame({"class_year": [2024], "athlete_id": ["1"], "rating": [0.9]}), "2024-02-15"),
        "cfb_returning": avail(pd.DataFrame({"season": [2024], "team_id": [10], "pct_ppa": [0.5]}), "2024-08-30"),
        "cfb_portal": portal,
        "cfb_coaches": make_coaches(),
        "cfb_rosters": avail(pd.DataFrame({"season": [2024], "team_id": [10], "player_id": ["1"]}), "2024-08-30"),
    }
    return SourceData.from_tables(t)


class TestAsOf:
    def test_future_results_and_pregame_elo_are_hidden_but_schedule_is_public(self):
        v = make_source().as_of(ts("2024-09-07 16:00"))
        g = v.games.set_index("game_id")
        assert g.loc[1, "home_points"] == 31 and bool(g.loc[1, "completed"])
        # Game 2 kicked off 6 days before the cutoff, so its result is public.
        assert g.loc[2, "home_points"] == 21
        # Game 3 is this week's: scheduled (visible) but unplayed.
        assert 3 in g.index
        assert np.isnan(g.loc[3, "home_points"]) and not bool(g.loc[3, "completed"])
        assert np.isnan(g.loc[3, "home_pregame_elo"])

    def test_result_is_hidden_until_a_day_after_kickoff(self):
        v = make_source().as_of(ts("2024-09-01 12:00"))
        g = v.games.set_index("game_id")
        assert np.isnan(g.loc[1, "home_points"])
        assert v.efficiency.empty

    def test_tables_filter_strictly_before_cutoff(self):
        v = make_source().as_of(ts("2024-08-30"))
        assert v.talent.empty and v.rosters.empty and v.returning.empty
        assert len(v.recruits) == 1 and v.recruiting_teams.empty

    def test_portal_destination_is_masked_until_announced(self):
        src = make_source()
        before = src.as_of(ts("2024-08-01"))
        assert len(before.portal) == 1
        assert pd.isna(before.portal["destination_id"].iloc[0])
        after = src.as_of(ts("2024-08-31"))
        assert after.portal["destination_id"].iloc[0] == 10

    def test_interim_coach_and_season_results_stay_hidden_during_the_season(self):
        src = make_source()
        mid = src.as_of(ts("2024-11-01"))
        c2024 = mid.coaches[mid.coaches["season"] == 2024]
        assert c2024["coach_id"].tolist() == [1]
        assert c2024["srs"].isna().all()
        assert mid.coaches.loc[mid.coaches["season"] == 2023, "srs"].iloc[0] == 5.0
        after = src.as_of(ts("2025-03-01"))
        assert sorted(after.coaches.loc[after.coaches["season"] == 2024, "coach_id"]) == [1, 2]

    def test_cannot_widen_a_view(self):
        v = make_source().as_of(ts("2024-09-01"))
        with pytest.raises(ValueError):
            v.as_of(ts("2024-10-01"))


class TestCoachNew:
    def coach_new(self, rows: list[tuple], season: int) -> bool:
        src = replace(make_source(), coaches=with_coach_availability(coach_rows(rows), make_games()))
        return bool(coach_features(src, season).set_index("team_id").loc[10, "coach_new"])

    def test_bowl_coach_kept_on_is_new_against_the_coach_of_most_of_last_season(self):
        # CFBD's hire_date is a coach's first head-coaching hire anywhere, so coach 3 (hired elsewhere in 2013, bowl
        # game only) outranks coach 4 (12 games) as 2024's season-start pick. His 2025 is still a coaching change.
        rows = [(4, 10, 2024, "2023-12-01", 12, 2.0), (3, 10, 2024, "2013-12-19", 1, 2.0), (3, 10, 2025, "2013-12-19", 12, 0.0)]
        assert self.coach_new(rows, 2025)

    def test_coach_of_most_of_last_season_is_not_new(self):
        rows = [(1, 10, 2024, "2019-12-01", 12, 2.0), (2, 10, 2024, "2024-11-20", 1, 2.0), (1, 10, 2025, "2019-12-01", 12, 0.0)]
        assert not self.coach_new(rows, 2025)


def test_season_start_coach_prefers_continuing_coach_over_later_hire():
    c = with_coach_availability(make_coaches(), make_games())
    start = c[(c["season"] == 2024) & c["season_start_coach"]]
    assert start["coach_id"].tolist() == [1]


class TestCutoffs:
    def test_cutoff_is_first_kickoff_of_the_week(self):
        c = week_cutoffs(make_games()).set_index("week")["cutoff"]
        assert c.loc[1] == ts("2024-08-31 16:00")
        gc = game_cutoffs(make_games()).set_index("game_id")["cutoff"]
        assert gc.loc[2] == ts("2024-08-31 16:00")


class TestRidgeRatings:
    @staticmethod
    def synthetic(n_teams: int = 12, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        o = rng.normal(0, 0.15, n_teams)
        d = rng.normal(0, 0.15, n_teams)
        rows, gid = [], 0
        for i in range(n_teams):
            for j in range(n_teams):
                if i == j:
                    continue
                gid += 1
                for off, de, venue in ((i, j, 1), (j, i, -1)):
                    rows.append({"game_id": gid, "team_id": off, "opponent_id": de, "venue": venue, "plays": 65,
                                 "team_classification": "fbs", "opponent_classification": "fbs", "usable": True,
                                 "off_ppa": 0.1 + o[off] - d[de] + 0.03 * venue + rng.normal(0, 0.02)})
        return pd.DataFrame(rows), o, d

    def test_signs_and_recovery(self):
        eff, o, d = self.synthetic()
        r, glob = ridge_ratings(eff, lam=1.0)
        r = r.set_index("team_id").sort_index()
        assert np.corrcoef(r["off_rating"], o)[0, 1] > 0.98
        assert np.corrcoef(r["def_rating"], d)[0, 1] > 0.98
        assert glob["hfa"] == pytest.approx(0.03, abs=0.01)

    def test_row_order_does_not_change_ratings(self):
        eff, _, _ = self.synthetic()
        a, _ = ridge_ratings(eff)
        b, _ = ridge_ratings(eff.sample(frac=1.0, random_state=3))
        pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-6)

    def test_fcs_opponents_share_one_pooled_rating(self):
        eff, _, _ = self.synthetic(4)
        fcs = eff.iloc[:2].copy()
        fcs["opponent_id"] = [901, 902]
        fcs["opponent_classification"] = "fcs"
        r, _ = ridge_ratings(pd.concat([eff, fcs]))
        assert FCS_TEAM in set(r["team_id"]) and not ({901, 902} & set(r["team_id"]))

    def test_unusable_rows_are_ignored(self):
        eff, _, _ = self.synthetic(4)
        bad = eff.iloc[:1].assign(off_ppa=50.0, usable=False)
        a, _ = ridge_ratings(eff)
        b, _ = ridge_ratings(pd.concat([eff, bad]))
        pd.testing.assert_frame_equal(a, b)


def test_coach_history_before_the_first_modelled_season_is_visible():
    old = make_coaches().iloc[[0]].assign(season=2010, results_available_at=ts("2011-02-01"))
    c = with_coach_availability(pd.concat([old, make_coaches()]), make_games())
    assert c["assignment_available_at"].notna().all()
    assert c.loc[c["season"] == 2010, "assignment_available_at"].iloc[0] < ts("2010-09-01")
