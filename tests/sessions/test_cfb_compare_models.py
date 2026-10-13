from __future__ import annotations

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cfb_win_probability.evaluation.compare_models import compare, market_probs


def test_market_probs_drops_missing_open():
    lines = pd.DataFrame({"game_id": [1, 1, 2], "spread": [-7, -6.5, 3], "spread_open": [-6, -6, np.nan],
                          "home_prob_novig": [0.7, 0.71, np.nan]})
    m = market_probs(lines)
    assert m.loc[m.game_id == 1, "p_open"].iloc[0] > 0.5
    assert m.loc[m.game_id == 2, "p_open"].isna().all()


def _synthetic(n=600, seed=0):
    rng = np.random.default_rng(seed)
    true_p = rng.uniform(0.2, 0.8, n)
    y = (rng.random(n) < true_p).astype(int)
    good = np.clip(true_p + rng.normal(0, 0.03, n), 0.01, 0.99)
    base = pd.DataFrame({"game_id": np.arange(n), "season": np.where(np.arange(n) % 2, 2023, 2024), "home_win": y})
    preds = {"A": base.assign(p_home=0.5), "B": base.assign(p_home=good)}
    market = pd.DataFrame({"game_id": np.arange(n), "p_open": true_p, "p_close": true_p})
    market.loc[:9, "p_open"] = np.nan
    return preds, market


def test_compare_reports_better_model_and_dropped_count():
    preds, market = _synthetic()
    out = compare(preds, market)
    assert "dropped 10" in out
    line = next(l for l in out.splitlines() if l.startswith("diff B - A"))
    assert float(line.split()[4]) < 0
