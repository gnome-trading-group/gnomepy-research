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
                  "home_team": ["A", "B", "C", "D"], "away_team": ["X", "Y", "Z", "W"]}).to_parquet(games)
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
    out, drops = lew.triggers(_legs([1, 2, 3]), slate, strategy_id=12, research_commit="abc", model="e11", now=NOW,
                              max_games=15, args={"contracts": 1.0})
    assert drops == []
    [t] = out
    d = t["data"]
    assert d["strategy_class"] == "gnomepy_research.sessions.cfb_pregame.early_week:CFBEarlyWeek"
    assert d["mode"] == "paper" and d["strategy_id"] == 12 and d["research_commit"] == "abc"
    assert d["listings"] == [101, 201, 103, 203, 102, 202], "kickoff order"
    g = d["strategy_args"]["games"][0]
    assert g == {"game_id": 1, "listing_home": 101, "listing_away": 201, "kickoff": SAT.isoformat(), "p_home": 0.61}
    assert d["strategy_args"]["model"] == "e11" and d["strategy_args"]["contracts"] == 1.0
    assert t["deadline"] == (SAT + pd.Timedelta(hours=3) - pd.Timedelta(hours=6)).isoformat()


def test_games_past_their_decision_window_are_left_out(tmp_path):
    slate = lew.load_slate(*_slate(tmp_path), "e11")
    late = SAT - pd.Timedelta(hours=72 - 6) + pd.Timedelta(minutes=1)
    out, drops = lew.triggers(_legs([1, 2, 3]), slate, strategy_id=12, research_commit="abc", model="e11", now=late,
                              max_games=15, args={})
    assert [g["game_id"] for g in out[0]["data"]["strategy_args"]["games"]] == [2]
    assert sorted(d[0] for d in drops) == [1, 3] and all("decision window" in d[2] for d in drops)


def test_a_slate_splits_into_sessions_of_at_most_max_games(tmp_path):
    slate = lew.load_slate(*_slate(tmp_path), "e11")
    out, _ = lew.triggers(_legs([1, 2, 3]), slate, strategy_id=12, research_commit="abc", model="e11", now=NOW,
                          max_games=2, args={})
    assert [[g["game_id"] for g in t["data"]["strategy_args"]["games"]] for t in out] == [[1, 3], [2]]
