"""CFB pregame: the prediction feed's view of time, and the anchored price and skip rules (test_cfb_pregame_strategy
covers the strategy)."""
from __future__ import annotations

import math
import time
from dataclasses import replace

import pandas as pd
import pytest

from gnomepy_research.sessions.cfb_pregame.prediction_feed import Prediction, PredictionFeed
from gnomepy_research.sessions.cfb_pregame.pricing import anchored_fair, skip_reason, twin_mid

HOUR = 3_600_000_000_000
MINUTE = HOUR // 60
KICKOFF = pd.Timestamp("2026-10-24T19:30:00Z")
K_NS = KICKOFF.value
EARLY = KICKOFF - pd.Timedelta(hours=6)


def _row(at, game_id=1, kickoff=KICKOFF, p_home=0.70, week=10, home_qb_p=0.8, away_qb_p=0.8, home_rank=20,
         away_rank=40, fbs_teams=136, home_newcomer=False, away_newcomer=False):
    return {"game_id": game_id, "season": 2026, "week": week, "season_type": "regular", "kickoff": kickoff,
            "home_id": 10 * game_id, "away_id": 10 * game_id + 1, "home_team": "Home", "away_team": "Away",
            "neutral_site": False, "p_home": p_home, "p_home_lo90": p_home - 0.1, "p_home_hi90": p_home + 0.1,
            "home_qb_id": "1", "away_qb_id": "2", "home_qb_p": home_qb_p, "away_qb_p": away_qb_p,
            "home_qb_delta": 0.0, "away_qb_delta": 0.0, "home_rank": home_rank, "away_rank": away_rank,
            "fbs_teams": fbs_teams, "home_newcomer": home_newcomer, "away_newcomer": away_newcomer, "model": "B0",
            "model_version": "abc", "qb_status_at": at, "generated_at": at}


def _write(tmp_path, rows) -> str:
    path = tmp_path / "predictions.parquet"
    pd.DataFrame(rows).to_parquet(path)
    return str(path)


def _pred(**kw) -> Prediction:
    base = Prediction(at_ns=EARLY.value, game_id=1, kickoff_ns=K_NS, week=10, p_home=0.7, home_qb_p=0.8,
                      away_qb_p=0.8, home_rank=20, away_rank=40, fbs_teams=136, home_newcomer=False,
                      away_newcomer=False)
    return replace(base, **kw)


# ---- the prediction feed --------------------------------------------------------------

def test_the_feed_is_point_in_time(tmp_path):
    t1, t2 = EARLY, EARLY + pd.Timedelta(hours=1)
    src = _write(tmp_path, [_row(t1, p_home=0.6), _row(t2, p_home=0.65, kickoff=KICKOFF + pd.Timedelta(hours=1))])
    feed = PredictionFeed([1], src, reload_seconds=0)
    assert feed.latest(1, t1.value - 1) is None, "nothing before it was written"
    assert feed.latest(1, t1.value).p_home == pytest.approx(0.6)
    assert feed.latest(1, (t1 + pd.Timedelta(minutes=30)).value).p_home == pytest.approx(0.6)
    assert feed.latest(1, t2.value).p_home == pytest.approx(0.65)
    assert feed.kickoffs() == {1: (KICKOFF + pd.Timedelta(hours=1)).value}, "the newest kickoff, whatever the time"


def test_the_feed_reads_only_its_games(tmp_path):
    src = _write(tmp_path, [_row(EARLY, game_id=1), _row(EARLY, game_id=3, p_home=0.1)])
    feed = PredictionFeed([1], src, reload_seconds=0)
    assert feed.latest(3, K_NS) is None and feed.latest(1, K_NS).game_id == 1
    assert set(feed.kickoffs()) == {1}


def test_the_feed_carries_every_field_the_rules_need(tmp_path):
    src = _write(tmp_path, [_row(EARLY, home_rank=None, home_qb_p=float("nan"), away_newcomer=True, week=8)])
    p = PredictionFeed([1], src, reload_seconds=0).latest(1, K_NS)
    assert p.home_rank is None and p.away_rank == 40 and p.fbs_teams == 136
    assert math.isnan(p.home_qb_p) and p.away_newcomer and not p.home_newcomer and p.week == 8


def test_a_failed_reload_keeps_the_last_snapshot(tmp_path):
    src = _write(tmp_path, [_row(EARLY, p_home=0.6)])
    feed = PredictionFeed([1], src, reload_seconds=0.05)
    later = EARLY + pd.Timedelta(hours=1)
    _write(tmp_path, [_row(EARLY, p_home=0.6), _row(later, p_home=0.62)])
    deadline = time.monotonic() + 5
    while feed.latest(1, K_NS).at_ns != later.value and time.monotonic() < deadline:
        time.sleep(0.02)
    assert feed.latest(1, K_NS).p_home == pytest.approx(0.62)
    (tmp_path / "predictions.parquet").write_bytes(b"not a parquet file")
    time.sleep(0.2)
    assert feed.latest(1, K_NS).at_ns == later.value, "a failed reload keeps the rows already held"
    _write(tmp_path, [_row(EARLY, p_home=0.6)])


# ---- pricing and skip rules -------------------------------------------------------------

def test_twin_mid_combines_both_legs():
    assert twin_mid(0.40, 0.44, 0.56, 0.60) == pytest.approx(0.42)


@pytest.mark.parametrize("legs", [(math.nan, 0.44, 0.56, 0.60), (0.40, 0.44, 0.56, math.nan)])
def test_twin_mid_is_nan_on_a_one_sided_leg(legs):
    assert math.isnan(twin_mid(*legs))


def test_anchored_fair_moves_w_of_the_way_in_logit():
    expected = 1 / (1 + math.exp(-0.25 * math.log(0.6 / 0.4)))
    assert anchored_fair(0.6, 0.5, 0.25) == pytest.approx(expected)
    assert anchored_fair(0.6, 0.5, 0.25) == pytest.approx(0.5253, abs=1e-4)


def test_anchor_extremes_are_the_model_and_the_market():
    assert anchored_fair(0.6, 0.5, 1.0) == pytest.approx(0.6)
    assert anchored_fair(0.6, 0.5, 0.0) == pytest.approx(0.5)
    assert anchored_fair(0.999, 0.5, 1.0) == pytest.approx(0.99), "clipped"


RULES = dict(max_age_ns=150 * MINUTE, min_qb_p=0.6, skip_bottom=35, skip_weeks=(5, 9), skip_newcomers=True)


@pytest.mark.parametrize("pred, now, reason", [
    (_pred(), EARLY.value + 151 * MINUTE, "stale prediction"),
    (_pred(away_qb_p=0.5), EARLY.value, "uncertain QB"),
    (_pred(week=8, home_rank=110), EARLY.value, "bottom team"),
    (_pred(home_newcomer=True), EARLY.value, "newcomer"),
    (_pred(), EARLY.value + 149 * MINUTE, None),
    (_pred(home_qb_p=math.nan), EARLY.value, None),
])
def test_skip_reasons(pred, now, reason):
    assert skip_reason(pred, now, **RULES) == reason


def test_skip_reasons_come_in_order():
    pred = _pred(away_qb_p=0.5, week=8, home_rank=120, home_newcomer=True)
    assert skip_reason(pred, EARLY.value + 200 * MINUTE, **RULES) == "stale prediction"
    assert skip_reason(pred, EARLY.value, **RULES) == "uncertain QB"


def test_bottom_team_rule_only_in_its_weeks():
    assert skip_reason(_pred(week=10, away_rank=120), EARLY.value, **RULES) is None
    assert skip_reason(_pred(week=9, away_rank=102), EARLY.value, **RULES) == "bottom team"
    assert skip_reason(_pred(week=9, away_rank=101), EARLY.value, **RULES) is None, "rank 101 of 136 is not bottom 35"
    assert skip_reason(_pred(week=9, away_rank=None), EARLY.value, **RULES) is None
