from __future__ import annotations

from gnomepy_research.strategies.informed_prediction_market_maker import InformedPredictionMarketMaker

strategy = InformedPredictionMarketMaker(
    ref_listing_id=129651,
    quote_listing_id=130435,
    value_function_path="gnomepy_research/solvers/value_function_kalshi_cal.npz",
    size=1_000_000,
    max_position=100,
    warmup_ticks=50,
    max_ref_staleness_ns=5_000_000_000,
    kalman_Q=1e-4,
    kalman_R=1e-2,
    maker_fee_rate=0.0,
    processing_time_ns=5_000_000,
)
