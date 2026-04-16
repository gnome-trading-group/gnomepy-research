"""Vectorized backtest metrics built on cash + inventory accounting.

All functions operate on the pandas DataFrames returned by
``gnomepy.BacktestResults`` (``execution_records_df`` and
``market_records_df``).  No row-level Python loops — everything is
cumsums, groupbys, and ``merge_asof``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


def _is_buy(side) -> bool:
    """Recognize a buy side regardless of whether the value is a Java enum,
    a string, or a Python ``Side`` enum.  A 'BID' (bid order) is a buy."""
    return "BID" in str(side).upper()


# ---------------------------------------------------------------------------
# Internal vectorized primitives
# ---------------------------------------------------------------------------

def _signed_fills(executions_df: pd.DataFrame) -> pd.DataFrame:
    """Filter to actual fills and add ``signed_qty`` and ``cash_flow`` columns.

    ``signed_qty``  = +fill_qty for buys, -fill_qty for sells.
    ``cash_flow``   = -signed_qty * fill_price - fee
                      (cash *increases* when selling, *decreases* when buying,
                       and fees always reduce cash).
    """
    if executions_df is None or executions_df.empty:
        return pd.DataFrame(
            columns=[
                "exchange_id", "security_id", "side",
                "fill_qty", "fill_price", "fee",
                "signed_qty", "cash_flow",
            ],
        )

    df = executions_df[executions_df["fill_qty"] > 0].copy()
    if df.empty:
        return df.assign(signed_qty=pd.Series(dtype=float), cash_flow=pd.Series(dtype=float))

    df = df.sort_index()
    is_buy = df["side"].map(_is_buy)
    df["signed_qty"] = df["fill_qty"].where(is_buy, -df["fill_qty"]).astype(float)
    fee = df["fee"].fillna(0.0).astype(float)
    df["cash_flow"] = -df["signed_qty"] * df["fill_price"].astype(float) - fee
    return df


def _per_symbol_cumsums(fills: pd.DataFrame) -> pd.DataFrame:
    """Compute running ``position``, ``cash``, ``cum_fees``, ``cum_volume``
    per ``(exchange_id, security_id)`` via grouped cumsums.

    Returns a DataFrame indexed like *fills* with the four cumulative columns
    appended.
    """
    if fills.empty:
        return fills.assign(
            position=pd.Series(dtype=float),
            cash=pd.Series(dtype=float),
            cum_fees=pd.Series(dtype=float),
            cum_volume=pd.Series(dtype=float),
            cum_notional=pd.Series(dtype=float),
        )

    grp = fills.groupby(["exchange_id", "security_id"], sort=False)
    fills = fills.copy()
    fills["position"] = grp["signed_qty"].cumsum()
    fills["cash"] = grp["cash_flow"].cumsum()
    fills["cum_fees"] = fills["fee"].fillna(0.0).groupby(
        [fills["exchange_id"], fills["security_id"]], sort=False,
    ).cumsum()
    fills["cum_volume"] = grp["fill_qty"].cumsum()
    fills["cum_notional"] = (fills["fill_qty"].astype(float) * fills["fill_price"].astype(float)).groupby(
        [fills["exchange_id"], fills["security_id"]], sort=False,
    ).cumsum()
    return fills


# ---------------------------------------------------------------------------
# Curves dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Curves:
    """Tick-level time series produced by :func:`build_curves`.

    Every metric is available as both a total (``pd.Series``) and a
    per-symbol breakdown (``pd.DataFrame`` with ``(eid, sid)`` columns).
    """

    pnl: pd.Series
    """Net PnL (cash + position * mid) summed across all symbols."""

    pnl_by_symbol: pd.DataFrame
    """Net PnL per ``(exchange_id, security_id)``."""

    position: pd.Series
    """Total position summed across all symbols."""

    position_by_symbol: pd.DataFrame
    """Position per ``(exchange_id, security_id)``."""

    fees: pd.Series
    """Cumulative fees summed across all symbols."""

    fees_by_symbol: pd.DataFrame
    """Cumulative fees per ``(exchange_id, security_id)``."""

    volume: pd.Series
    """Cumulative traded quantity (unsigned) summed across all symbols."""

    volume_by_symbol: pd.DataFrame
    """Cumulative traded quantity per ``(exchange_id, security_id)``."""

    notional: pd.Series
    """Cumulative notional volume (qty * price) summed across all symbols."""

    notional_by_symbol: pd.DataFrame
    """Cumulative notional volume per ``(exchange_id, security_id)``."""

    fills: pd.DataFrame
    """The filtered, signed fills frame (for downstream use)."""

    fill_count: int
    """Total number of fills."""


def build_curves(
    market_df: pd.DataFrame,
    executions_df: pd.DataFrame,
) -> Curves:
    """Build tick-level PnL, position, fees, and volume curves.

    Uses cash + inventory accounting:

    .. code-block:: text

        signed_qty_t = +fill_qty if BID else -fill_qty
        position_t   = cumsum(signed_qty)                      # per symbol
        cash_t       = -cumsum(signed_qty * fill_price) - cumsum(fee)
        pnl_t        = cash_t + position_t * mid_t             # net of fees

    The fill-level cumsums are lifted onto every market tick via
    ``pd.merge_asof`` on a timestamp *column* (not the index), so
    duplicate timestamps in market data are handled correctly — every
    market row gets a PnL value.
    """
    empty_fills = pd.DataFrame()

    if market_df is None or market_df.empty:
        idx = pd.DatetimeIndex([])
        zeros = pd.Series(0.0, index=idx, dtype=float)
        empty_df = pd.DataFrame(index=idx, dtype=float)
        return Curves(
            pnl=zeros.copy(), pnl_by_symbol=empty_df.copy(),
            position=zeros.copy(), position_by_symbol=empty_df.copy(),
            fees=zeros.copy(), fees_by_symbol=empty_df.copy(),
            volume=zeros.copy(), volume_by_symbol=empty_df.copy(),
            notional=zeros.copy(), notional_by_symbol=empty_df.copy(),
            fills=empty_fills, fill_count=0,
        )

    fills = _signed_fills(executions_df)
    if not fills.empty:
        fills = _per_symbol_cumsums(fills)

    # Sort market data and assign a monotonic sequence number so we can
    # align across symbols without relying on (possibly duplicate) timestamps.
    mkt = market_df.sort_index().copy()
    # The gnomepy recorder writes top-of-book as bid_price_0 / ask_price_0
    # (depth-indexed), not a pre-computed mid. Derive mid_price here so the
    # rest of this function can treat it as a scalar column.
    if "mid_price" not in mkt.columns:
        if "bid_price_0" in mkt.columns and "ask_price_0" in mkt.columns:
            mkt["mid_price"] = (mkt["bid_price_0"].astype(float) + mkt["ask_price_0"].astype(float)) / 2.0
        else:
            mkt["mid_price"] = 0.0
    mkt["_seq"] = range(len(mkt))
    n = len(mkt)

    # Per-symbol PnL, fees, volume — indexed by _seq. Each symbol only has
    # values at its own market ticks; we forward-fill and sum at the end
    # so total PnL at any row reflects the latest state of ALL symbols.
    full_seq = range(n)
    sym_pnl_cols: dict[tuple, pd.Series] = {}
    sym_fees_cols: dict[tuple, pd.Series] = {}
    sym_vol_cols: dict[tuple, pd.Series] = {}
    sym_notional_cols: dict[tuple, pd.Series] = {}
    position_cols: dict[tuple, pd.Series] = {}

    for (eid, sid), mkt_grp in mkt.groupby(["exchange_id", "security_id"], sort=False):
        mkt_rows = mkt_grp.reset_index()
        seq_vals = mkt_rows["_seq"].values

        zeros = pd.Series(0.0, index=seq_vals)
        if fills.empty:
            sym_pnl_cols[(eid, sid)] = zeros.copy()
            sym_fees_cols[(eid, sid)] = zeros.copy()
            sym_vol_cols[(eid, sid)] = zeros.copy()
            sym_notional_cols[(eid, sid)] = zeros.copy()
            position_cols[(eid, sid)] = zeros.copy()
            continue

        sym_fills = fills[
            (fills["exchange_id"] == eid) & (fills["security_id"] == sid)
        ]

        if sym_fills.empty:
            sym_pnl_cols[(eid, sid)] = zeros.copy()
            sym_fees_cols[(eid, sid)] = zeros.copy()
            sym_vol_cols[(eid, sid)] = zeros.copy()
            sym_notional_cols[(eid, sid)] = zeros.copy()
            position_cols[(eid, sid)] = zeros.copy()
            continue

        state = sym_fills[["position", "cash", "cum_fees", "cum_volume", "cum_notional"]].reset_index()
        state.columns = ["timestamp", "position", "cash", "cum_fees", "cum_volume", "cum_notional"]

        merged = pd.merge_asof(
            mkt_rows.sort_values("timestamp"),
            state.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        ).fillna(0.0)

        mid = merged["mid_price"].astype(float)
        sym_pnl_cols[(eid, sid)] = pd.Series(
            merged["cash"].values + merged["position"].values * mid.values,
            index=seq_vals,
        )
        sym_fees_cols[(eid, sid)] = pd.Series(merged["cum_fees"].values, index=seq_vals)
        sym_vol_cols[(eid, sid)] = pd.Series(merged["cum_volume"].values, index=seq_vals)
        sym_notional_cols[(eid, sid)] = pd.Series(merged["cum_notional"].values, index=seq_vals)
        position_cols[(eid, sid)] = pd.Series(merged["position"].values, index=seq_vals)

    # Reindex each symbol's series to the full _seq range and forward-fill.
    # Return both the per-symbol DataFrame and the total Series.
    ts_index = mkt.index

    def _combine(cols: dict[tuple, pd.Series]) -> tuple[pd.Series, pd.DataFrame]:
        if not cols:
            zeros = pd.Series(0.0, index=full_seq, dtype=float)
            return zeros, pd.DataFrame(index=ts_index, dtype=float)
        df = pd.DataFrame(
            {k: v.reindex(full_seq).ffill().fillna(0.0) for k, v in cols.items()},
        )
        total = df.sum(axis=1)
        df.index = ts_index
        return total, df

    pnl_total, pnl_by_sym = _combine(sym_pnl_cols)
    fees_total, fees_by_sym = _combine(sym_fees_cols)
    vol_total, vol_by_sym = _combine(sym_vol_cols)
    notional_total, notional_by_sym = _combine(sym_notional_cols)
    pos_total, pos_by_sym = _combine(position_cols)

    return Curves(
        pnl=pd.Series(pnl_total.values, index=ts_index, dtype=float),
        pnl_by_symbol=pnl_by_sym,
        position=pd.Series(pos_total.values, index=ts_index, dtype=float),
        position_by_symbol=pos_by_sym,
        fees=pd.Series(fees_total.values, index=ts_index, dtype=float),
        fees_by_symbol=fees_by_sym,
        volume=pd.Series(vol_total.values, index=ts_index, dtype=float),
        volume_by_symbol=vol_by_sym,
        notional=pd.Series(notional_total.values, index=ts_index, dtype=float),
        notional_by_symbol=notional_by_sym,
        fills=fills,
        fill_count=len(fills),
    )


# ---------------------------------------------------------------------------
# Sharpe ratio and stability
# ---------------------------------------------------------------------------

# Crypto 24/7: 365 days * 24 hours * 3600 seconds = 31_536_000 seconds/year.
_CRYPTO_SECONDS_PER_YEAR = 365 * 24 * 3600


def compute_sharpe(
    pnl_curve: pd.Series,
    bar: str = "10s",
    annualize: bool = False,
    annualize_factor: float | None = None,
    n_buckets: int = 5,
) -> dict:
    """Compute Sharpe, Sortino, and per-bucket stability metrics.

    By default returns the raw (non-annualized) bar-level Sharpe:
    ``mean(bar_pnl) / std(bar_pnl)``.  Set ``annualize=True`` to scale
    by ``sqrt(bars_per_year)`` (assumes 24/7 crypto markets unless
    ``annualize_factor`` is provided).

    Parameters
    ----------
    pnl_curve : pd.Series
        Cumulative PnL indexed by timestamp.
    bar : str
        Pandas resample frequency for computing bar returns (e.g. ``"10s"``,
        ``"1min"``).
    annualize : bool
        Whether to annualize the Sharpe ratio.
    annualize_factor : float, optional
        ``sqrt(bars_per_year)``.  Only used when ``annualize=True``.
        If *None*, auto-computed assuming 24/7 crypto markets.
    n_buckets : int
        Number of equal-sized chunks for stability analysis.

    Returns
    -------
    dict with keys: ``sharpe``, ``sortino``, ``sharpe_per_bucket``,
    ``sharpe_std``, ``sharpe_min``, ``pct_positive_buckets``, ``bar``,
    ``n_bars``.
    """
    empty = {
        "sharpe": 0.0, "sortino": 0.0,
        "sharpe_per_bucket": [], "sharpe_std": 0.0,
        "sharpe_min": 0.0, "pct_positive_buckets": 0.0,
        "bar": bar, "n_bars": 0,
    }

    if pnl_curve.empty or len(pnl_curve) < 2:
        return empty

    # Resample to fixed bars — take last PnL value per bar.
    resampled = pnl_curve.resample(bar).last().dropna()
    bar_pnl = resampled.diff().dropna()

    if len(bar_pnl) < 2:
        return empty

    # Annualization factor.
    if annualize:
        if annualize_factor is None:
            bar_seconds = pd.Timedelta(bar).total_seconds()
            bars_per_year = _CRYPTO_SECONDS_PER_YEAR / bar_seconds
            annualize_factor = math.sqrt(bars_per_year)
    else:
        annualize_factor = 1.0

    mean_pnl = float(bar_pnl.mean())
    std_pnl = float(bar_pnl.std(ddof=1))
    downside = bar_pnl[bar_pnl < 0]
    downside_std = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0

    sharpe = (mean_pnl / std_pnl * annualize_factor) if std_pnl > 0 else 0.0
    sortino = (mean_pnl / downside_std * annualize_factor) if downside_std > 0 else 0.0

    # Per-bucket stability.
    bucket_sharpes = []
    chunk_size = max(len(bar_pnl) // n_buckets, 1)
    for i in range(n_buckets):
        chunk = bar_pnl.iloc[i * chunk_size : (i + 1) * chunk_size]
        if len(chunk) < 2:
            continue
        s = float(chunk.std(ddof=1))
        if s > 0:
            bucket_sharpes.append(float(chunk.mean()) / s * annualize_factor)
        else:
            bucket_sharpes.append(0.0)

    sharpe_std = float(np.std(bucket_sharpes, ddof=1)) if len(bucket_sharpes) > 1 else 0.0
    sharpe_min = float(min(bucket_sharpes)) if bucket_sharpes else 0.0
    pct_positive = sum(1 for s in bucket_sharpes if s > 0) / len(bucket_sharpes) if bucket_sharpes else 0.0

    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "sharpe_per_bucket": bucket_sharpes,
        "sharpe_std": sharpe_std,
        "sharpe_min": sharpe_min,
        "pct_positive_buckets": pct_positive,
        "bar": bar,
        "n_bars": len(bar_pnl),
    }
