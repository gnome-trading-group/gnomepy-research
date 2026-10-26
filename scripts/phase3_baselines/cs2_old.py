"""cs2_series replay on released code over the synthetic books in cs2_spec.json, served from a local cache."""
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import zstandard

from gnomepy.java._jvm import ensure_jvm_started
from gnomepy.java.backtest.config import BacktestConfig, ListingSimConfig
from gnomepy.java.backtest.runner import run_backtest
from gnomepy.java.schemas import Mbp10Schema
from gnomepy.registry.api import RegistryClient
from gnomepy.reporting.report import BacktestReport
from gnomepy_research.sessions.cs2_series.strategy import CS2Series
from local_s3 import local_s3_client

HERE = Path(__file__).parent
SIMULATION_AS_OF = "2026-10-25T00:00:00.000Z"
LISTING_TEAM_A = 1744070
LISTING_TEAM_B = 1744069
ARTIFACTS = dict(
    model_path="artifact://xgboost_model/cs2_prematch_preveto:20261021T074549660300Z",
    posterior_path="artifact://bayes_posterior/cs2_series:20261021T081858647867Z",
    blend_path="artifact://blend/cs2_series:20261021T182637380113Z",
)

# A large stake cap and no adverse-move stop keep it trading through the sweep, so the replay exercises many orders.
STRATEGY_ARGS = dict(bankroll_usd=20000.0, max_stake_usd=5000.0, adverse_move=1.0, reload_minutes=0, **ARTIFACTS)


def old_key(listing, minute: str) -> str:
    t = datetime.fromisoformat(minute)
    return f"{listing.security_id}/{listing.exchange_id}/{t.year}/{t.month}/{t.day}/{t.hour}/{t.minute}/mbp-10.zst"


def main(name: str) -> None:
    spec = json.loads((HERE / "cs2_spec.json").read_text())
    out = HERE / "results" / name / "old"
    out.mkdir(parents=True, exist_ok=True)
    ensure_jvm_started()
    registry = RegistryClient()
    listings = {"a": registry.get_listing(listing_id=LISTING_TEAM_A)[0],
                "b": registry.get_listing(listing_id=LISTING_TEAM_B)[0]}
    files = out / "files"
    for team, records in spec["records"].items():
        listing = listings[team]
        by_minute = defaultdict(list)
        for record in records:
            by_minute[record["minute"]].append(
                Mbp10Schema(exchange_id=listing.exchange_id, security_id=listing.security_id, **record["fields"]))
        for minute, schemas in by_minute.items():
            raw = b"".join(s.encode() for s in schemas)
            path = files / old_key(listing, minute)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(zstandard.ZstdCompressor().compress(raw))

    config = BacktestConfig(
        start_date=datetime.fromisoformat(spec["start"]),
        end_date=datetime.fromisoformat(spec["end"]),
        listings=[ListingSimConfig(listing_id=LISTING_TEAM_A), ListingSimConfig(listing_id=LISTING_TEAM_B)],
        simulation_as_of=SIMULATION_AS_OF,
    )
    strategy = CS2Series(spec["match_id"], LISTING_TEAM_A, LISTING_TEAM_B, **STRATEGY_ARGS)
    results = run_backtest(config, strategy, progress=False, cache=False, s3_client=local_s3_client(files))
    fills = results.fills_df()
    fills.to_parquet(out / "fills.parquet")
    results.orders_df().to_parquet(out / "orders.parquet")
    summary = BacktestReport(results).summary()
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("cs2 fills", len(fills), "orders", len(results.orders_df()), "final_pnl", summary.get("final_pnl"),
          "market rows", results.market_record_count)


if __name__ == "__main__":
    main(sys.argv[1])
