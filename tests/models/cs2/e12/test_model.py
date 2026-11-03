import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gnomepy_research.models.cs2.e12.data import UNRANKED
from gnomepy_research.models.cs2.e12.model import GAMMA, E12Model, TeamState
from gnomepy_research.models.cs2.e12.skill import SkillFit
from gnomepy_research.models.cs2.e12.tables import BundleTables, naive

CUTOFF = pd.Timestamp("2026-05-04")
TEAMS = {10: list(range(100, 105)), 20: list(range(200, 205)), 30: list(range(300, 305)), 40: list(range(400, 405))}
STRENGTH = {10: 1.0, 20: 0.4, 30: -0.2, 40: -0.8}


def _write(root: Path, rel: str, df: pd.DataFrame) -> None:
    path = root / f"{rel}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _bundle(root: Path, extra_after_cutoff: bool = False) -> Path:
    """A tiny bundle: four HLTV teams of known strength, FACEIT maps and matchmaking games between their players.
    With extra_after_cutoff, rows dated after CUTOFF are added to every table; a fit at CUTOFF must ignore them."""
    rng = np.random.default_rng(0)
    days = pd.date_range("2025-09-01", "2026-06-15", freq="3D")
    if not extra_after_cutoff:
        days = days[days < CUTOFF]
    pairs = [(a, b) for a in TEAMS for b in TEAMS if a < b]
    hist, stats, lineups, fm, fh, ro, hm, hp = [], [], [], [], [], [], [], []
    mid = 1000
    for i, day in enumerate(days):
        a, b = pairs[i % len(pairs)]
        mid += 1
        p = 1 / (1 + np.exp(-(STRENGTH[a] - STRENGTH[b]) / 2))
        sa = 13 if rng.random() < p else int(rng.integers(5, 12))
        sb = 13 if sa != 13 else int(rng.integers(5, 12))
        hist.append({"match_id": mid, "match_date": day, "match_time": day.tz_localize("UTC") + pd.Timedelta(hours=18),
                     "event_name": "Cup", "format": "Best of 3", "bo_type": 3, "is_lan": 0, "team_a_id": a, "team_b_id": b,
                     "team_a_rank": float(a), "team_b_rank": float(b), "map_name": "Mirage", "team_a_score": sa,
                     "team_b_score": sb, "team_a_won": int(sa > sb), "team_a_player_ids": TEAMS[a],
                     "team_b_player_ids": TEAMS[b], "stage_text": ""})
        for team in (a, b):
            for pl in TEAMS[team]:
                stats.append({"match_id": mid, "match_date": day, "map_name": "Mirage", "team_id": team, "player_id": pl,
                              "side": "all", "rating": 1.0 + 0.05 * (pl % 5)})
                lineups.append({"match_id": mid, "match_date": day, "team_id": team, "player_id": pl,
                                "is_standin": pl % 5 == 4 and i % 7 == 0})
        fid = f"f{mid}"
        fin = int((day + pd.Timedelta(days=1)).tz_localize("UTC").timestamp())
        fm.append({"competition_name": "ESEA Main", "faction1_hltv_team": float(a), "faction2_hltv_team": float(b),
                   "finished_at": fin, "has_stats": True, "match_id": fid, "started_at": float(fin - 7200), "winner": "faction1"})
        fh.append({"bo_type": 1, "event_name": "ESEA", "is_lan": 0, "map_name": "Nuke", "match_date": day,
                   "match_id": fid, "match_time": day.tz_localize("UTC"), "stage_text": "",
                   "team_a_faceit_id": f"t{a}", "team_a_faceit_ids": [f"u{x}" for x in TEAMS[a]], "team_a_id": float(a),
                   "team_a_ot_score": 0.0, "team_a_player_ids": [float(x) for x in TEAMS[a]], "team_a_players": 5,
                   "team_a_score": sa, "team_a_won": int(sa > sb), "team_b_faceit_id": f"t{b}",
                   "team_b_faceit_ids": [f"u{x}" for x in TEAMS[b]], "team_b_id": float(b), "team_b_ot_score": 0.0,
                   "team_b_player_ids": [float(x) for x in TEAMS[b]], "team_b_players": 5, "team_b_score": sb})
        for team, faction in ((a, "faction1"), (b, "faction2")):
            for pl in TEAMS[team]:
                ro.append({"faceit_id": f"u{pl}", "faction": faction, "hltv_id": float(pl), "match_id": fid,
                           "played": True, "skill_level": 10})
        for g in range(3):
            gid = f"g{mid}_{g}"
            hm.append({"match_id": gid, "competition_type": "matchmaking", "competition_name": "5v5", "started_at": fin - 3600,
                       "finished_at": fin, "winner": "faction1" if rng.random() < p else "faction2"})
            for team, faction in ((a, "faction1"), (b, "faction2")):
                for pl in TEAMS[team]:
                    hp.append({"match_id": gid, "faction": faction, "faceit_id": f"u{pl}", "skill_level": 10})
    ranks = [{"date": d, "team_id": t, "rank": t} for d in pd.date_range("2025-09-01", "2026-06-15", freq="W-MON")
             if extra_after_cutoff or d < CUTOFF for t in TEAMS]
    _write(root, "hltv/cs2_match_history", pd.DataFrame(hist))
    _write(root, "hltv/cs2_player_map_stats", pd.DataFrame(stats))
    _write(root, "hltv/cs2_match_lineups", pd.DataFrame(lineups))
    _write(root, "hltv/cs2_team_rankings", pd.DataFrame(ranks))
    _write(root, "faceit/hltv_faceit_map", pd.DataFrame({"hltv_id": [p for ps in TEAMS.values() for p in ps],
                                                         "faceit_id": [f"u{p}" for ps in TEAMS.values() for p in ps]}))
    _write(root, "faceit/faceit_matches", pd.DataFrame(fm))
    _write(root, "faceit/faceit_match_history", pd.DataFrame(fh))
    _write(root, "faceit/faceit_rosters", pd.DataFrame(ro))
    _write(root, "faceit/faceit_exclusions", pd.DataFrame({"match_id": pd.Series([], dtype=str), "map_name": pd.Series([], dtype=str)}))
    _write(root, "faceit/history_matches", pd.DataFrame(hm))
    _write(root, "faceit/history_players", pd.DataFrame(hp))
    return root


@pytest.fixture(scope="module")
def model(tmp_path_factory) -> E12Model:
    return E12Model.build(BundleTables(_bundle(tmp_path_factory.mktemp("bundle"))), CUTOFF)


def _hand_model() -> E12Model:
    gamma = np.array([0.0, 0.1, 0.05, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0])
    players = np.arange(1, 11)
    fit = SkillFit(players, np.linspace(-1, 1, 10), np.full(10, 20), gamma)
    teams = {1: TeamState(rank=5.0, standin=0.2, idle_days=3.0, five_hltv=(1, 2, 3, 4, 5), five_any=(1, 2, 3, 4, 5)),
             2: TeamState(rank=40.0, standin=0.0, idle_days=10.0, five_hltv=(6, 7, 8, 9, 10), five_any=(6, 7, 8, 9, 10))}
    return E12Model(CUTOFF, fit, fit, teams, {p: 0.1 * p for p in range(1, 11)})


def test_price_is_antisymmetric_in_the_teams():
    m = _hand_model()
    a, b = [1, 2, 3, 4, 5], [6, 7, 8, 9, 10]
    assert m.price(1, a, 2, b) + m.price(2, b, 1, a) == pytest.approx(1.0, abs=1e-12)


def test_matchmaking_skill_moves_the_price_by_gamma():
    m = _hand_model()
    a, b = [1, 2, 3, 4, 5], [6, 7, 8, 9, 10]
    m.mm = {}
    flat = m.price(1, a, 2, b)
    m.mm = {1: 5.0}
    raised = m.price(1, a, 2, b)
    assert np.log(raised / (1 - raised)) - np.log(flat / (1 - flat)) == pytest.approx(GAMMA * 5.0 / 5, abs=1e-9)


def test_a_side_without_a_full_five_plays_its_last_five():
    m = _hand_model()
    full = m.price(1, [1, 2, 3, 4, 5], 2, [6, 7, 8, 9, 10])
    assert m.e12(1, [1, 2, 3], 2, [6, 7, 8, 9, 10]) == m.e12(1, [1, 2, 3, 4, 5], 2, [6, 7, 8, 9, 10])
    assert 0 < full < 1


def test_unknown_players_are_priced_from_their_teams_rank():
    m = _hand_model()
    unknown = [91, 92, 93, 94, 95]
    assert m.e12(1, unknown, 2, [6, 7, 8, 9, 10]) > m.e12(99, unknown, 2, [6, 7, 8, 9, 10])
    assert m.teams.get(99) is None and TeamState().rank == UNRANKED


def test_save_load_prices_identically(tmp_path, model):
    path = tmp_path / "e12.npz"
    model.save(path)
    loaded = E12Model.load(path)
    for a, b in ((10, 20), (20, 30), (40, 10)):
        assert loaded.price(a, TEAMS[a], b, TEAMS[b]) == model.price(a, TEAMS[a], b, TEAMS[b])
    assert loaded.cutoff == model.cutoff


def test_load_refuses_an_artifact_fitted_with_other_constants(tmp_path, model):
    path = tmp_path / "e12.npz"
    model.save(path)
    z = dict(np.load(path, allow_pickle=False))
    meta = json.loads(str(z["meta"]))
    meta["gamma"] = 0.5
    z["meta"] = np.array(json.dumps(meta))
    np.savez_compressed(path, **z)
    with pytest.raises(ValueError, match="different constants"):
        E12Model.load(path)


def test_fit_orders_teams_by_strength(model):
    assert model.price(10, TEAMS[10], 40, TEAMS[40]) > 0.5
    assert model.price(10, TEAMS[10], 40, TEAMS[40]) > model.price(20, TEAMS[20], 30, TEAMS[30])
    assert model.mm and model.info["faceit"]["maps"] > 0


def test_rows_after_the_cutoff_change_nothing(tmp_path_factory, model):
    later = E12Model.build(BundleTables(_bundle(tmp_path_factory.mktemp("later"), extra_after_cutoff=True)), CUTOFF)
    for a, b in ((10, 20), (30, 40), (20, 40)):
        assert later.price(a, TEAMS[a], b, TEAMS[b]) == model.price(a, TEAMS[a], b, TEAMS[b])


def test_bundle_tables_cut_by_time(tmp_path):
    root = _bundle(tmp_path, extra_after_cutoff=True)
    t = BundleTables(root)
    assert (naive(t.load("match_history", CUTOFF).match_date) < CUTOFF).all()
    assert (t.load("faceit_matches", CUTOFF).finished_at < CUTOFF.tz_localize("UTC").timestamp()).all()
    known = set(t.load("faceit_matches", CUTOFF, ["match_id"]).match_id)
    assert set(t.load("faceit_rosters", CUTOFF).match_id) <= known


PARITY_BUNDLE = os.environ.get("E12_PARITY_BUNDLE")
PARITY_REF = os.environ.get("E12_PARITY_REF")


@pytest.mark.skipif(not (PARITY_BUNDLE and PARITY_REF),
                    reason="needs E12_PARITY_BUNDLE (a research bundle) and E12_PARITY_REF (that model's research predictions)")
def test_matches_the_validated_research_prices():
    """The release gate: one dev week priced from this package equals the validated research implementation."""
    t = pd.read_parquet(Path(PARITY_BUNDLE) / "targets_tune.parquet")
    t["week_cutoff"] = naive(t.week_cutoff)
    week = t.week_cutoff.min()
    ref = pd.read_parquet(PARITY_REF).set_index("match_id").q
    m = E12Model.build(BundleTables(PARITY_BUNDLE), week)
    wk = t[t.week_cutoff == week]
    q = np.array([m.price(r.team_a_id, r.team_a_lineup_ids, r.team_b_id, r.team_b_lineup_ids) for r in wk.itertuples()])
    assert np.abs(q - ref.reindex(wk.match_id).to_numpy()).max() < 1e-9
