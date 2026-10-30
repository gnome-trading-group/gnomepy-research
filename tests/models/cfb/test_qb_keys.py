from __future__ import annotations

import pandas as pd

from gnomepy_research.models.cfb.features.qb import roster_keys


def roster(*players: tuple[str, str, str, str]) -> pd.DataFrame:
    return pd.DataFrame([{"season": 2025, "team_id": 2483, "first_name": f, "last_name": l, "position": pos,
                          "player_id": pid} for f, l, pos, pid in players])


def test_shared_key_links_to_the_only_quarterback():
    # Oregon 2025: Dante Moore (QB) and Dakorien Moore (WR) are both "d.moore"; the passer is the quarterback.
    keys = roster_keys(roster(("Dante", "Moore", "QB", "4870921"), ("Dakorien", "Moore", "WR", "5141586")))
    assert keys.set_index("passer_key").loc["d.moore", "player_id"] == "4870921"


def test_shared_key_between_two_quarterbacks_stays_unlinked():
    keys = roster_keys(roster(("Dante", "Moore", "QB", "1"), ("Drew", "Moore", "QB", "2")))
    assert "d.moore" not in set(keys["passer_key"])


def test_shared_key_with_no_quarterback_stays_unlinked():
    keys = roster_keys(roster(("Dakorien", "Moore", "WR", "1"), ("Devon", "Moore", "RB", "2")))
    assert "d.moore" not in set(keys["passer_key"])
