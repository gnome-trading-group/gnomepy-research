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
