"""Series pricing: every BO3 score priced as an unknown map, composed by the series DP."""
import numpy as np
import pandas as pd
import pytest

from gnomepy_research.models.cs2.win_probability.pre_map_features import PREVETO_FEATURE_NAMES
from gnomepy_research.models.cs2.win_probability.pre_map_model import CS2PreMapModel, _matrix, fit_production, write_bundle
from gnomepy_research.models.cs2.win_probability.series_pricing import node_frame, price
from gnomepy_research.models.cs2.win_probability.symmetry import build_swap_plan, swap_features


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    rng = np.random.default_rng(0)
    n = 1200
    df = pd.DataFrame(rng.normal(size=(n, len(PREVETO_FEATURE_NAMES))), columns=PREVETO_FEATURE_NAMES)
    df["match_id"] = np.repeat(np.arange(n // 2), 2)
    df["match_date"] = pd.Timestamp("2026-01-01") + pd.to_timedelta(df.match_id // 4, unit="D")
    df["map_name"] = ""
    df["team_a_won"] = (rng.uniform(size=n) < 1 / (1 + np.exp(-df.elo_diff))).astype(int)
    path = tmp_path_factory.mktemp("m") / "model.json"
    write_bundle(fit_production(df, PREVETO_FEATURE_NAMES, n_estimators=60, max_depth=3), path)
    return CS2PreMapModel(str(path))


def _mirror(row: pd.Series) -> pd.Series:
    plan = build_swap_plan(PREVETO_FEATURE_NAMES)
    swapped = swap_features(_matrix(pd.DataFrame([row]), PREVETO_FEATURE_NAMES), plan)[0]
    return row.copy().pipe(lambda r: r.where(~r.index.isin(PREVETO_FEATURE_NAMES), pd.Series(swapped, PREVETO_FEATURE_NAMES)))


def test_lattice_covers_every_bo3_score():
    frame, nodes = node_frame(pd.Series({"match_id": 1, "elo_diff": 30.0}))
    assert sorted(nodes) == [(0, 0), (0, 1), (1, 0), (1, 1)]
    for (a, b), r in zip(nodes, frame.itertuples()):
        assert (r.team_a_series_score, r.team_b_series_score) == (a, b)
        assert r.map_position_in_series == a + b + 1 and r.is_decider == (a + b == 2)
    assert (frame.elo_diff == 30.0).all() and (frame.map_name == "").all()


def test_prices_are_order_invariant(model):
    rng = np.random.default_rng(1)
    row = pd.Series(rng.normal(size=len(PREVETO_FEATURE_NAMES)), index=PREVETO_FEATURE_NAMES)
    row["match_id"] = 1
    for c in ("team_a_series_score", "team_b_series_score", "map_position_in_series", "is_decider"):
        row[c] = 0.0
    a = price(model, pd.DataFrame([row]))
    b = price(model, pd.DataFrame([_mirror(row)]))
    assert a.series.iloc[0] + b.series.iloc[0] == pytest.approx(1.0, abs=0.02)
    assert a.game1.iloc[0] + b.game1.iloc[0] == pytest.approx(1.0, abs=0.02)


def test_a_series_is_more_lopsided_than_one_map(model):
    rows = []
    for elo in (-2.5, 2.5):
        r = pd.Series(0.0, index=PREVETO_FEATURE_NAMES)
        r["elo_diff"], r["match_id"] = elo, int(elo > 0)
        rows.append(r)
    out = price(model, pd.DataFrame(rows)).set_index("match_id")
    assert out.series[1] > out.game1[1] > 0.5 > out.game1[0] > out.series[0]
