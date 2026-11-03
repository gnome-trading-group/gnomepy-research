"""CFB early-week launch: the slate from the weekly model file, games whose decision window has passed, and the
direct_launch triggers."""
from __future__ import annotations

import pandas as pd
import pytest

from gnomepy_research.sessions.cfb_pregame import launch_early_week as lew

NOW = pd.Timestamp("2026-11-04T12:00:00Z")
SAT = pd.Timestamp("2026-11-07T17:00:00Z")


def _slate(tmp_path, model="e11"):
    path = tmp_path / "week10_e11_e15.csv"
    pd.DataFrame({"game_id": [1, 2, 3], "kickoff (ET)": ["Sat 12:00 PM"] * 3, "home_team": ["A", "B", "C"],
                  "away_team": ["X", "Y", "Z"], "C2": [0.5, 0.6, 0.7], "e11": [0.61, 0.62, 0.63],
                  "e15": [0.6, 0.6, 0.6], "e11+e15": [0.71, 0.72, 0.73]}).to_csv(path, index=False)
    games = tmp_path / "preds_C2.parquet"
    pd.DataFrame({"game_id": [1, 2, 3, 4], "kickoff": [SAT, SAT + pd.Timedelta(hours=3), SAT, SAT],
                  "home_team": ["A", "B", "C", "D"], "away_team": ["X", "Y", "Z", "W"],
                  "home_qb_p": [0.9, 0.95, 0.9, 0.9], "away_qb_p": [0.8, float("nan"), 0.9, 0.9]}).to_parquet(games)
    return path, games


def _legs(ids):
    return pd.DataFrame({"game_id": ids, "venue": "kalshi", "esid_home": [f"H{g}" for g in ids],
                         "esid_away": [f"A{g}" for g in ids], "listing_home": [100 + g for g in ids],
                         "listing_away": [200 + g for g in ids]})


def test_the_slate_takes_the_model_column_and_the_parquets_kickoffs(tmp_path):
    path, games = _slate(tmp_path)
    s = lew.load_slate(path, games, "e11").set_index("game_id")
    assert list(s.index) == [1, 2, 3], "game 4 has no model price"
    assert s.loc[2, "p_home"] == 0.62 and s.loc[2, "kickoff"] == SAT + pd.Timedelta(hours=3)
    assert lew.load_slate(path, games, "e11+e15").set_index("game_id").loc[1, "p_home"] == 0.71
    with pytest.raises(SystemExit):
        lew.load_slate(path, games, "e12")


def test_triggers_hold_the_frozen_model_price_in_paper_mode(tmp_path):
    slate = lew.load_slate(*_slate(tmp_path), "e11")
    out, drops = lew.triggers(_legs([1, 2, 3]), slate, strategy_ids={"sell": 12}, research_commit="abc", model="e11", now=NOW,
                              max_games=15, args={"contracts": 1.0})
    assert drops == []
    [t] = out
    d = t["data"]
    assert d["strategy_class"] == "gnomepy_research.sessions.cfb_pregame.early_week:CFBEarlyWeek"
    assert d["mode"] == "paper" and d["strategy_id"] == 12 and d["research_commit"] == "abc"
    assert d["listings"] == [101, 201, 103, 203, 102, 202], "kickoff order"
    g = d["strategy_args"]["games"][0]
    assert g == {"game_id": 1, "listing_home": 101, "listing_away": 201, "kickoff": SAT.isoformat(), "p_home": 0.61,
                 "off_protocol": False}
    assert d["strategy_args"]["model"] == "e11" and d["strategy_args"]["contracts"] == 1.0
    assert t["deadline"] == (SAT + pd.Timedelta(hours=3) - pd.Timedelta(hours=6)).isoformat()


def test_games_past_their_decision_window_are_left_out(tmp_path):
    slate = lew.load_slate(*_slate(tmp_path), "e11")
    late = SAT - pd.Timedelta(hours=72 - 6) + pd.Timedelta(minutes=1)
    out, drops = lew.triggers(_legs([1, 2, 3]), slate, strategy_ids={"sell": 12}, research_commit="abc", model="e11", now=late,
                              max_games=15, args={})
    assert [g["game_id"] for g in out[0]["data"]["strategy_args"]["games"]] == [2]
    assert sorted(d[0] for d in drops) == [1, 3] and all("decision window" in d[2] for d in drops)


def test_a_slate_splits_into_sessions_of_at_most_max_games(tmp_path):
    slate = lew.load_slate(*_slate(tmp_path), "e11")
    out, _ = lew.triggers(_legs([1, 2, 3]), slate, strategy_ids={"sell": 12}, research_commit="abc", model="e11", now=NOW,
                          max_games=2, args={})
    assert [[g["game_id"] for g in t["data"]["strategy_args"]["games"]] for t in out] == [[1, 3], [2]]


def test_with_include_late_closed_windows_trade_flagged_off_protocol_until_seven_hours_out(tmp_path):
    slate = lew.load_slate(*_slate(tmp_path), "e11")
    now = SAT - pd.Timedelta(hours=8)
    out, drops = lew.triggers(_legs([1, 2, 3]), slate, strategy_ids={"sell": 12}, research_commit="abc", model="e11", now=now,
                              max_games=15, args={}, include_late=True)
    games = {g["game_id"]: g for t in out for g in t["data"]["strategy_args"]["games"]}
    assert sorted(games) == [1, 2, 3] and drops == []
    assert all(g["off_protocol"] for g in games.values())
    out, drops = lew.triggers(_legs([1, 2, 3]), slate, strategy_ids={"sell": 12}, research_commit="abc", model="e11",
                              now=SAT - pd.Timedelta(hours=6.5), max_games=15, args={}, include_late=True)
    assert [g["game_id"] for t in out for g in t["data"]["strategy_args"]["games"]] == [2]
    assert sorted(d[0] for d in drops) == [1, 3] and all("entry window" in d[2] for d in drops)


def test_a_game_with_an_uncertain_starter_is_left_out(tmp_path):
    slate = lew.load_slate(*_slate(tmp_path), "e11")
    assert slate.set_index("game_id").loc[1, "away_qb_p"] == 0.8, "the starters' p_start come from the C2 run"
    slate.loc[slate["game_id"] == 3, "away_qb_p"] = 0.55
    out, drops = lew.triggers(_legs([1, 2, 3]), slate, strategy_ids={"sell": 12}, research_commit="abc", model="e11", now=NOW,
                              max_games=15, args={}, min_qb_p=0.6)
    assert [g["game_id"] for g in out[0]["data"]["strategy_args"]["games"]] == [1, 2], "no snapshot (NaN) is no doubt"
    assert drops == [(3, "kalshi", "uncertain QB: home 0.90, away 0.55")]


def test_the_hold_variant_launches_as_its_own_strategy_on_the_same_games(tmp_path):
    slate = lew.load_slate(*_slate(tmp_path), "e11")
    out, _ = lew.triggers(_legs([1, 2, 3]), slate, strategy_ids={"sell": 12, "hold": 13}, research_commit="abc",
                          model="e11", now=NOW, max_games=15, args={"contracts": 1.0})
    by = {t["variant"]: t["data"] for t in out}
    assert sorted(by) == ["hold", "sell"]
    assert by["hold"]["strategy_id"] == 13 and by["hold"]["strategy_args"]["hold_to_settlement"] is True
    assert "hold_to_settlement" not in by["sell"]["strategy_args"]
    assert by["hold"]["strategy_args"]["games"] == by["sell"]["strategy_args"]["games"]
