"""compute_adverse_selection must fail loudly on a frame with no mid_price.

Returning an empty DataFrame made a wrong call (passing BacktestResults frames
instead of BacktestReport ones) look like a successful run with no adverse
selection, which is how a broken docs example shipped unnoticed.
"""
from __future__ import annotations

import pandas as pd
import pytest

from gnomepy_research.reporting.backtest.adverse_selection import compute_adverse_selection


def _fills() -> pd.DataFrame:
    ts = pd.to_datetime(["2026-01-23 10:30:00"], utc=True)
    return pd.DataFrame(
        {"side": ["Bid"], "fill_price": [0.52], "fill_qty": [10.0]},
        index=pd.DatetimeIndex(ts, name="timestamp"),
    )


def _market(with_mid: bool) -> pd.DataFrame:
    ts = pd.to_datetime(["2026-01-23 10:30:00", "2026-01-23 10:30:01"], utc=True)
    data = {"bid_price_0": [0.51, 0.52], "ask_price_0": [0.53, 0.54]}
    if with_mid:
        data["mid_price"] = [0.52, 0.53]
    return pd.DataFrame(data, index=pd.DatetimeIndex(ts, name="timestamp"))


def test_raises_when_mid_price_missing():
    with pytest.raises(ValueError, match="mid_price"):
        compute_adverse_selection(_fills(), _market(with_mid=False))


def test_error_names_the_right_accessor():
    """The message has to point at the fix, not just the symptom."""
    with pytest.raises(ValueError, match="report.market_df"):
        compute_adverse_selection(_fills(), _market(with_mid=False))


def test_empty_inputs_still_return_empty_not_raise():
    assert compute_adverse_selection(pd.DataFrame(), _market(with_mid=True)).empty
    assert compute_adverse_selection(_fills(), pd.DataFrame()).empty


def test_valid_input_produces_rows():
    out = compute_adverse_selection(_fills(), _market(with_mid=True))
    assert not out.empty
    assert "mid_at_fill" in out.columns
