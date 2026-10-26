"""run_old.py on the Phase 3 code: the same quoter, finding its listing's market data by stream."""
import json
import sys
from datetime import datetime
from pathlib import Path

from gnomepy.java.backtest.config import BacktestConfig, ListingSimConfig
from gnomepy.java.backtest.runner import run_backtest
from gnomepy.java.backtest.strategy import Strategy
from gnomepy.java.enums import OrderType, Side
from gnomepy.java.oms import Intent
from gnomepy.java.statics import Scales
from gnomepy.reporting.report import BacktestReport

SIMULATION_AS_OF = "2026-10-25T00:00:00.000Z"
TAKE_EVERY = 50


class TouchQuoter(Strategy):
    """Joins the touch, and crosses the spread every TAKE_EVERY updates (alternately buying and selling) so a run
    fills even on a thin book."""

    def __init__(self, listing_id: int, size: float):
        self.listing_id = listing_id
        self.size = int(size * Scales.SIZE)
        self.updates = 0
        self.venue = None

    def on_market_data(self, data):
        if self.streams.listing_id(data.stream_id) != self.listing_id:
            return []
        if self.venue is None:
            listing = self.streams.listing(self.listing_id)
            self.venue = (int(listing.exchange().exchangeId()), int(listing.security().securityId()))
        bid, ask = data.bid_price(0), data.ask_price(0)
        if bid <= 0 or ask <= 0 or bid >= ask:
            return []
        self.updates += 1
        take = {}
        if self.updates % TAKE_EVERY == 0:
            buy = (self.updates // TAKE_EVERY) % 2 == 1
            take = dict(take_side=Side.BID if buy else Side.ASK, take_size=self.size,
                        take_order_type=OrderType.LIMIT, take_limit_price=ask if buy else bid)
        return [Intent(self.venue[0], self.venue[1], bid_price=bid, bid_size=self.size, ask_price=ask,
                       ask_size=self.size, **take)]

    def on_execution_report(self, report):
        return []


def main(name: str, listing_id: int, start: str, end: str, size: float) -> None:
    out = Path(__file__).parent / "results" / name / "new"
    out.mkdir(parents=True, exist_ok=True)
    config = BacktestConfig(
        start_date=datetime.fromisoformat(start),
        end_date=datetime.fromisoformat(end),
        listings=[ListingSimConfig(listing_id=listing_id)],
        simulation_as_of=SIMULATION_AS_OF,
    )
    results = run_backtest(config, TouchQuoter(listing_id, size), progress=False, cache=False)
    fills = results.fills_df()
    fills.to_parquet(out / "fills.parquet")
    summary = BacktestReport(results).summary()
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(name, "fills", len(fills), "final_pnl", summary.get("final_pnl"), "market rows", results.market_record_count)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], float(sys.argv[5]))
