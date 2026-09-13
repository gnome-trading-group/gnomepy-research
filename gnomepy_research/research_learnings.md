# Research Learnings

Cross-session knowledge base. Appended automatically by `/research` when a session completes or stalls,
or when a significant milestone (first threshold-passing iteration) is reached.

Read by `/research` Step 2 before forming each hypothesis — check for matching `strategy_type` and listing IDs.

## Learnings

<!-- Entries appended here by /research Step 7 -->

### [2026-09-13] equivalent_event_arb__fix_unfilled_legs (arb, polymarket+kalshi)
- STATUS: running (first threshold pass at iter 3)
- WORKED: Maker orders (join_best_bid, 120s imbalance timeout) fixed the fill rate problem. Taker orders (iter 1) failed because 55ms latency meant orders arrived after brief ask dips had closed. Maker bids rest in the book and fill when sellers cross us, capturing the same opportunities without latency race.
- FAILED: Taker-first mode (iter 1) caused 60% leg imbalance — order cancelled on (4,254608) when ask dipped then recovered before order arrived. DepthCoverageCost penalty (iter 2) had zero effect because raw arb edge (20%+) was too large for 188bps penalty to push below threshold.
- INSIGHT: These CS:GO prediction market arbs on Kalshi+Polymarket have very brief (sub-second) ask dips that create entry signals. With 50ms latency, taker orders consistently miss. Maker bids are the right approach. Strategy is VERY latency-sensitive: 2x latency (100ms) → threshold fails (PnL -26.71 vs +6.63 at 50ms). The Polymarket ask book for low-probability outcomes is thin (12.74 shares within 2 cents), so taker orders for large qty fail immediately. Maker bids avoid this problem entirely.

### [2026-09-13] equivalent_event_arb__fix_unfilled_legs — SESSION COMPLETE (iter 020, best=iter_019)
- STATUS: complete — best accepted: iter_019, final_pnl=+6.653, Sharpe=0.056
- WORKED: `min_contract_price=0.25` parameter in compute_target blocks new entries when any leg's BID price is below the threshold. Reduced end-of-match pair 2 adverse selection losses from -2.864 to -0.475 (iter_013 → iter_019). Pair 1 entries (both legs bid ≥ 0.31) unaffected.
- FAILED: cancel_grace_ns (iter_014) — keeps BOTH legs' targets active during grace, accumulating more fills. Selective cancel in compute_target (iter_015/016) — engine batches fill events before next compute_target cycle. Pre-entry ASK check (iter_017) — both pairs have ASK edges of 60-70bps at entry, indistinguishable. High min_pure_arb_bps=250 (iter_018) — affects both entry AND exit validity checks, causes premature exits and wipes pair 1 profit.
- INSIGHT: End-of-match adverse selection in prediction markets: near resolution, one contract converges to 0, creating stale bids that get filled immediately. The contract's BID price itself is the right filter — below 0.25 means the market is nearly decided. Note: 6.06 shares residual persists (structural race condition: order placed when bid=0.25-0.30, filled after bid drops to 0.20 in ~1.7s window post-pair-1-exit). Cannot be eliminated with a simple price threshold.
- DATA: Tennis markets (Zverev vs Paul, 46-min window) produce fewer/smaller arbs than CS:GO (12-min window). Sharpe is severely limited by short data windows (0.056 vs target 1.0) — focus on multi-event datasets for Sharpe improvement.
- MARKET: Listings 117905/117906 (Polymarket), 115763/115764 (Kalshi), event_ids [22765,22167], 2026-08-19. The match resolution triggers the end-of-match dynamics around 22:09 UTC (46 min into the window).
