from __future__ import annotations

from gnomepy_research.strategies.prediction_market_maker import PredictionMarketMaker

strategy = PredictionMarketMaker(
    listing_id=19757,
    value_function_path="artifact://value_function/Q10",
    size=1_000_000,
    max_position=10,
    warmup_ticks=50,
    processing_time_ns=5_000_000,
)
