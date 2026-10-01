# Holding: CS2 market evaluation scripts

Research scripts from the 2026-09-30 / 10-01 sessions, kept out of the session's
main modules on purpose: they are analyses, not pipeline code. Working data lives
in `$CS2_RESEARCH_DATA` (default `~/.cache/gnomepy/cs2_research`) — see `_paths.py`.
That directory is local only; the HLTV datasets are published to the DatasetStore,
the Polymarket files are not yet.

Run from the repo root with `poetry run python <script>`.

| script | does | reads | writes |
|---|---|---|---|
| `run_full.py` | priors → per-block ablation → Polymarket gate, in one pass | reparsed HLTV frames, `cs2_team_rankings.parquet` | `priors_full.parquet` |
| `pm_traj_v2.py` | matches BO3 test series to Polymarket **series moneylines** and fetches 1-min prices anchored on HLTV kickoff | `priors_full`, `cs2_match_history`, `pm_cs2_h2h` | `pm_traj_v2.parquet` |
| `pm_final.py` | the gate: 34-feature baseline vs 95-feature model vs market at each horizon | `priors_full`, `pm_traj_v2` | `cs2_l1_{baseline,harvested}.xgb` |
| `series_diag.py` | why the series probability looked overconfident (split leakage, error stacking, calibrator drift) | L1 bundle, `priors_full` | — |
| `series_wf.py` | six-fold walk-forward: monthly-retrained, out-of-sample series predictions | `priors_full` | `series_wf_oos.parquet` |
| `series_pnl.py` | walk-forward series vs market: log-loss, model+market blend, fee-aware P&L, liquidity split | `series_wf_oos`, `pm_traj_v2` | — |
| `pm_maps_fetch.py` | matches maps to Polymarket **map-winner** markets (`-game1/-game2`) and fetches price paths | `priors_full`, `pm_cs2_h2h` | `pm_maps_{matched,paths}.parquet` |
| `pm_maps_eval.py` | map 1 at kickoff and map 2 at map-1's end vs market, with fee-aware P&L | L1 bundles, `pm_maps_*` | `pm_maps_eval.parquet` |
| `map2_players.py` | whether map-1 margin / per-player performance improves map-2 prediction (it does not) | `cs2_player_map_stats`, `priors_full` | — |

Hard-won rules these scripts encode — do not undo them:

- **Polymarket market selection**: an h2h event holds many markets across several meetings.
  The series moneyline is the slug ending `-YYYY-MM-DD`; map winners end `-gameN`. Never pick by volume.
  Check `final` against the HLTV result.
- **Polymarket `end` is kickoff + 6.0h.** Anchor horizons on HLTV `match_time`.
- **Report fit noise as well as the clustered bootstrap** (`--n-repeats` in `ablate.py`).
- **No series-level calibrator fit on a short window** — 577 series pointed the wrong way.
- **Fills are simulated**: Polymarket price history is one price per minute, not a book.
