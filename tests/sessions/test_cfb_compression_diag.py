from __future__ import annotations

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cfb_win_probability.evaluation.compression_diag import band_table, diagnose, slope


def _frame():
    return pd.DataFrame({"open_margin": [-30.0, -10.0, 1.0, 5.0, 30.0],
                         "model_margin": [-24.0, -8.0, 0.5, 4.0, 24.0],
                         "actual": [-28.0, -9.0, 3.0, 6.0, 27.0],
                         "margin_mean": [-27.0, -9.0, 1.0, 4.5, 27.0]})


def test_slope_recovers_linear_relation():
    x = np.arange(10.0)
    assert abs(slope(x, 0.8 * x + 3) - 0.8) < 1e-12


def test_band_table_uses_favorite_side_and_model_mean():
    t = band_table(_frame())
    assert list(t.index) == ["0-3", "3-7", "7-14", "21+"]
    assert t.loc["21+", "n"] == 2
    assert t.loc["21+", "open"] == 30.0
    assert t.loc["21+", "model_p"] == 24.0
    assert t.loc["21+", "model_mean"] == 27.0
    assert t.loc["21+", "actual"] == 27.5


def test_band_table_without_margin_mean_has_no_model_mean_column():
    assert "model_mean" not in band_table(_frame().drop(columns="margin_mean")).columns


def test_diagnose_compression_slope():
    d = diagnose(_frame())
    assert 0.7 < d["model_vs_open"] < 0.9
