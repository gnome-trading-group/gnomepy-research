from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gnomepy_research.sessions.cfb_win_probability.evaluation.error_lab.build import build_rows, parse_spec, render


def test_parse_spec_label_name_and_path():
    assert parse_spec("K") == ("K", "K", None)
    assert parse_spec("K:Full priors + drives") == ("K", "Full priors + drives", None)
    assert parse_spec("MINE:My model=out/p.parquet") == ("MINE", "My model", Path("out/p.parquet"))
    with pytest.raises(ValueError):
        parse_spec("bad label!")


def frames():
    games = pd.DataFrame({
        "game_id": [1, 2, 3], "season": 2024, "week": [1, 5, 16], "season_type": ["regular", "regular", "postseason"],
        "neutral_site": [False, False, True], "conference_game": [True, False, False],
        "home_id": [10, 11, 12], "away_id": [20, 21, 22], "home_team": ["A", "Army", "C"], "away_team": ["D", "E", "F"],
        "home_conference": ["SEC", None, "ACC"], "away_conference": ["SEC", "MAC", "Big Ten"],
    })
    preds = {
        "K": pd.DataFrame({"home_win": [1, 0, 1], "p_home": [0.7, 0.4, 0.55]}, index=pd.Index([1, 2, 3], name="game_id")),
        "G": pd.DataFrame({"home_win": [1, 0, 1], "p_home": [0.65, 0.45, 0.5]}, index=pd.Index([1, 2, 3], name="game_id")),
    }
    market = pd.DataFrame({"p_open": [0.68, 0.42, np.nan], "p_close": [0.7, 0.4, 0.6]}, index=pd.Index([1, 2, 3], name="game_id"))
    ranks = pd.Series({(2024, 10): 5.0})
    return preds, games, market, ranks


def test_build_rows_keeps_games_with_both_lines_and_all_models():
    rows = build_rows(*frames())
    assert [r["id"] for r in rows] == [1, 2]
    assert rows[0]["p"] == {"K": 0.7, "G": 0.65, "Open": 0.68, "Close": 0.7}
    assert rows[0]["hr"] == 5 and rows[0]["ar"] is None
    assert rows[1]["hc"] == "" and rows[1]["y"] == 0


def test_render_embeds_models_and_escapes_script_close():
    rows = build_rows(*frames())
    rows[0]["h"] = "</script>"
    html = render(rows, [("K", "Full priors + drives"), ("G", "Conference-aware")], "2024")
    payload = html.split('id="data">')[1].split("</script>")[0]
    data = json.loads(payload.replace("<\\/", "</"))
    assert [m["label"] for m in data["models"]] == ["K", "G"] and data["tag"] == "2024"
    assert "__DATA__" not in html
