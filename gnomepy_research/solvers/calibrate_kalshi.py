"""
Calibrate Feil-Nendel intensity params from Kalshi Mbp10 data, then solve the
HJB and save a new value function.

Usage:
    poetry run python -m gnomepy_research.solvers.calibrate_kalshi \\
        --market <market.parquet> \\
        --listing-id <kalshi_listing_id> \\
        --out <output.npz>
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from gnomepy.registry import RegistryClient
from gnomepy_research.solvers.calibrate_intensity import calibrate
from gnomepy_research.solvers.prediction_market_hjb import (
    HJBParams,
    save_solution,
    solve_hjb,
)

WINDOW_NS = 60_000_000_000  # 60s lookahead for fill detection
SPREAD_LEVELS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10]
# Downsample to every Nth tick to keep calibration tractable
DOWNSAMPLE = 50


def build_calibration_df(
    kalshi: pd.DataFrame,
    resolution_ns: int,
    listing_start_ns: int,
) -> pd.DataFrame:
    """
    For each (downsampled) tick, simulate passive bids at SPREAD_LEVELS above mid
    and detect fills via ask_price_0 crossing below the bid within WINDOW_NS.

    Returns a DataFrame with columns [t_norm, mid, spread_from_mid, fill_count,
    exposure_ns] aggregated over (t_norm_bucket, mid_bucket, spread) cells.
    """
    df = kalshi.copy()
    df['ts_ns'] = df.index.astype('int64')
    df['mid'] = (df.bid_price_0 + df.ask_price_0) / 2.0

    # Use session-relative t_norm (0=session start, 1=session end) so that
    # calibrated params reflect actual activity during this market period,
    # independent of the event's total lifetime (which may be months long).
    session_end_ns = df['ts_ns'].max()
    session_span_ns = max(session_end_ns - listing_start_ns, 1)
    df['t_norm'] = (df['ts_ns'] - listing_start_ns) / session_span_ns
    df['t_norm'] = df['t_norm'].clip(0.0, 1.0)

    # Keep valid mid prices (exclude boundaries and negative spreads)
    df = df[(df.mid > 0.05) & (df.mid < 0.95) & (df.ask_price_0 > df.bid_price_0)]

    # Downsample to reduce computation
    df = df.iloc[::DOWNSAMPLE].reset_index(drop=True)

    ts_ns = df['ts_ns'].to_numpy()
    ask = df['ask_price_0'].to_numpy()
    mid = df['mid'].to_numpy()
    t_norm = df['t_norm'].to_numpy()

    rows = []
    for spread in SPREAD_LEVELS:
        bid_prices = mid - spread

        for i in range(len(df)):
            bid_price = bid_prices[i]
            if bid_price <= 0:
                continue

            t0 = ts_ns[i]
            deadline = t0 + WINDOW_NS

            # Find ticks within the lookahead window
            j = i + 1
            while j < len(df) and ts_ns[j] <= deadline:
                j += 1
            window_ask = ask[i + 1:j]

            if len(window_ask) == 0:
                exposure = deadline - t0
                filled = 0
            else:
                crossed = window_ask <= bid_price
                if crossed.any():
                    fill_tick = np.argmax(crossed)
                    exposure = ts_ns[i + 1 + fill_tick] - t0
                    filled = 1
                else:
                    exposure = deadline - t0
                    filled = 0

            rows.append({
                't_norm': t_norm[i],
                'mid': mid[i],
                'spread_from_mid': spread,
                'fill_count': filled,
                'exposure_ns': exposure,
            })

    raw = pd.DataFrame(rows)

    # Bucket and aggregate
    raw['t_bucket'] = (raw.t_norm * 10).astype(int) / 10
    raw['mid_bucket'] = (raw.mid * 20).astype(int) / 20

    agg = (raw.groupby(['t_bucket', 'mid_bucket', 'spread_from_mid'])
           .agg(t_norm=('t_norm', 'mean'),
                mid=('mid', 'mean'),
                fill_count=('fill_count', 'sum'),
                exposure_ns=('exposure_ns', 'sum'))
           .reset_index())

    return agg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--market', required=True)
    parser.add_argument('--listing-id', type=int, required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    registry = RegistryClient()
    listings = registry.get_listing(listing_id=args.listing_id)
    if not listings:
        raise ValueError(f'listing {args.listing_id} not found')
    exchange_id = listings[0].exchange_id
    security_id = listings[0].security_id

    contracts = registry.get_event_contracts(security_id=security_id)
    events = registry.get_event(event_id=contracts[0].event_id)
    from datetime import datetime, timezone
    expiry_dt = datetime.fromisoformat(events[0].expiry.replace('Z', '+00:00'))
    if expiry_dt.tzinfo is None:
        expiry_dt = expiry_dt.replace(tzinfo=timezone.utc)
    resolution_ns = int(expiry_dt.timestamp() * 1e9)

    mkt = pd.read_parquet(args.market)
    kalshi = mkt[(mkt.exchange_id == exchange_id) & (mkt.security_id == security_id)].copy()
    if kalshi.empty:
        raise ValueError('no Kalshi data found in market parquet')

    listing_start_ns = kalshi.index.astype('int64').min()
    session_span_min = (kalshi.index.astype('int64').max() - listing_start_ns) / 6e10
    print(f'Kalshi rows: {len(kalshi)}  (downsampled to ~{len(kalshi)//DOWNSAMPLE})')
    print(f'Session span: {session_span_min:.1f} min  (t_norm 0→1 within session)')

    print('Building calibration dataset...')
    cal_df = build_calibration_df(kalshi, resolution_ns, listing_start_ns)
    print(f'Calibration rows: {len(cal_df)}')
    print(cal_df.groupby('spread_from_mid')[['fill_count', 'exposure_ns']].sum().assign(
        fill_rate=lambda d: d.fill_count / (d.exposure_ns / 1e9)
    ).to_string())

    print('\nCalibrating intensity parameters...')
    base_params = HJBParams()
    result = calibrate(cal_df, params=base_params)
    print('Calibrated params:')
    for k, v in result.items():
        base_v = getattr(base_params, k)
        print(f'  {k}: {base_v:.3f} -> {v:.3f}')

    params = HJBParams(**result)
    print('\nSolving HJB...')
    V, tau_grid, p_grid, q_levels = solve_hjb(params)
    save_solution(V, tau_grid, p_grid, q_levels, params, args.out)
    print(f'Saved to {args.out}')


if __name__ == '__main__':
    main()
