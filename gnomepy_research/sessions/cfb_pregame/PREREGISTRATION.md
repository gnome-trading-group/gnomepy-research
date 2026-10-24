# CFB pregame paper sessions: pre-registration

Written Oct 24, 2026, about 04:00 UTC, before any session ran. Paper only; not financial advice.

## Why paper only

- **B0 trails the 2026 Kalshi close by +0.0365 nats per game,** about 3× its 2021–24 backtest gap.
- **Its 2026 information weight is negative:** w = −0.52 ± 0.29.
- **About 210 pregame rule variants on 2026 weeks 1–7 found nothing** that survives Kalshi costs. The raw-model taker
  lost 27% on stake.

These sessions measure; they are not expected to make money.

## Sessions

There is one session per variant per venue, over the whole slate:

| Variant | Strategy | `anchor_weight` | `min_edge` | Everything else |
|---|---|---|---|---|
| ANCHOR | `cfb_pregame_anchor` | 0.25 | 0.01 | `CFBPregame` defaults |
| RAW | `cfb_pregame_raw` | 1.0 | 0.03 | `CFBPregame` defaults |

**Venues:** Kalshi and Polymarket US, with fees from the registry's schedules.

**Defaults:**
- bankroll_usd 10,000;
- kelly_fraction 0.25;
- max_game_stake_usd 100; max_total_stake_usd 2,000;
- price band [0.15, 0.85); max_spread 0.06;
- entry cutoff 30 min before kickoff;
- predictions at most 150 min old;
- min_qb_p 0.6;
- skip bottom-35 teams in weeks 5–9;
- skip FBS newcomers;
- adverse_move 0.08.

**Universe:** FBS vs FBS games in `cfb_pregame_predictions` whose two listings resolve in the registry, with
consistent orientation (`launch.py`'s cross-check).

**Amended Oct 24, 15:25 UTC, before any session launched (the user's call):** entries run up to each kickoff
(`entry_cutoff_minutes` 0), not 30 minutes before. The model is C2 (B0 + prior bundle), the best model in the
2025–26 error lab, built as `live/picks.py` builds it.

## Endpoints, per variant × venue and pooled

1. **Primary: the live information weight w.**
   - A logistic fit of the home result on offset logit(m) + w·(logit p − logit m), with no intercept.
   - It uses each game's first `cfb_pregame_marks` row with `skipped == 0`.
   - p and m are clipped to [0.01, 0.99]. Report w, its Fisher SE and n.
2. **Settlement P&L, return on stake, and bets won against those implied by the entry price.**
3. **Closing-line value:** each fill's price against the venue's twin mid 5 min before kickoff.

## Use

- Weekly results are logged and pooled with later weeks.
- Nothing is promoted to real money from these sessions alone. The model-health gate still stands: season-to-date
  LL(B0) − LL(close), against the backtest benchmark.
