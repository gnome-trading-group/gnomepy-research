"""
Score CS2CrossVenueArb backtest jobs at each venue's real settlement price, not the run's last mark.

    python -m gnomepy_research.sessions.cs2_cross_venue_arb.score RESULTS_DIR MARKETS [--csv OUT]

RESULTS_DIR is a `gnomepy backtest run --output` directory; MARKETS is the market map make_config.py read, whose legs
carry each listing's settlement price (scaled by 1e9; null while unsettled).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from gnomepy_research.sessions.cs2_cross_venue_arb.decision import exposure

BUY_SIDES = {"Bid", "BID", "B", "Side.BID"}


def score_job(job: Path, settle: dict[int, float | None], names: dict[int, str]) -> dict:
    config = yaml.safe_load(open(job / "config.yaml"))
    legs = config["strategy"]["args"]["legs"]
    team = {}
    for leg in legs:
        team[int(leg["listing_team_a"])] = 0
        team[int(leg["listing_team_b"])] = 1
    fills_path = job / "fills.parquet"
    fills = pd.read_parquet(fills_path) if fills_path.exists() else pd.DataFrame()
    summary = json.load(open(job / "summary.json"))
    row = dict(job=job.name, scenario=names.get(next(iter(team)), job.name),
               fills=len(fills), mark_pnl=summary.get("final_pnl"), fees=0.0, cash=0.0, payout=0.0,
               pays_a=0.0, pays_b=0.0, unsettled=False)
    if fills.empty:
        row["settle_pnl"] = 0.0
        return row
    # fills.parquet is already in dollars and contracts.
    qty, px = fills.fill_qty, fills.fill_price
    sign = fills.side.astype(str).map(lambda s: 1 if s.split(".")[-1] in BUY_SIDES else -1)
    row["fees"] = float(fills.fee.sum())
    row["cash"] = float(-(sign * qty * px).sum()) - row["fees"]
    net = (sign * qty).groupby(fills.listing_id).sum()
    holdings = []
    for listing_id, n in net.items():
        s = settle.get(int(listing_id))
        if s is None:
            row["unsettled"] = True
            continue
        row["payout"] += float(n * s)
        holdings.append((team[int(listing_id)], float(n)))
    row["pays_a"], row["pays_b"] = exposure(holdings)
    row["settle_pnl"] = row["cash"] + row["payout"]
    return row


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("results")
    p.add_argument("markets")
    p.add_argument("--csv")
    args = p.parse_args()
    settle, names = {}, {}
    for m in json.load(open(args.markets)):
        for leg in m["legs"]:
            settle[int(leg["listing"])] = None if leg["settle"] is None else int(leg["settle"]) / 1e9
            names[int(leg["listing"])] = f"{'|'.join(m['teams'])}/{m['scope']}/{m['start'][:16]}"
    jobs = sorted(d for d in (Path(args.results) / "jobs").iterdir() if (d / "summary.json").exists())
    df = pd.DataFrame([score_job(j, settle, names) for j in jobs])
    df["unhedged"] = (df.pays_a - df.pays_b).abs()
    pd.set_option("display.width", 200)
    print(df[["scenario", "fills", "mark_pnl", "settle_pnl", "fees", "pays_a", "pays_b", "unhedged", "unsettled"]]
          .round(2).to_string(index=False))
    print(f"\ntotal settle_pnl {df.settle_pnl.sum():.2f} over {len(df)} markets; mark_pnl {df.mark_pnl.sum():.2f}; "
          f"fees {df.fees.sum():.2f}; unhedged contracts {df.unhedged.sum():.1f}")
    if args.csv:
        df.to_csv(args.csv, index=False)


if __name__ == "__main__":
    main()
