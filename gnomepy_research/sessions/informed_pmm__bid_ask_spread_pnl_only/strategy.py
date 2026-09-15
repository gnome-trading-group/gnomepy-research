from __future__ import annotations

from gnomepy_research.sessions.informed_pmm__bid_ask_spread_pnl_only.informed_prediction_market_maker import SpreadCapturePMM

strategy = SpreadCapturePMM(
    ref_yes_listing_id=97203,
    ref_no_listing_id=97202,
    quote_yes_listing_id=222852,
    quote_no_listing_id=222853,
    value_function_path="gnomepy_research/solvers/value_function_kalshi_cal.npz",
    size=5_000_000,
    max_position=20,
    inventory_fade=0.5,
    warmup_ticks=50,
    max_ref_staleness_ns=5_000_000_000,
    kalman_Q=1e-4,
    kalman_R=1e-2,
    maker_fee_rate=0.0,
    processing_time_ns=5_000_000,
    min_quote_interval_ns=1_000_000_000,
    resolution_time_override_ns=1_787_541_060_000_000_000,
)
