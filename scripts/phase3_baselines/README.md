# TEMPORARY: Phase 3 before/after baselines

**Delete this folder once Phase 3 (schemas v2, stream-keyed storage) is cut over in prod and every comparison below
says MATCH.** It exists only to check that cutover; nothing imports it.

Each baseline was run on released code before the change (gnome-backtest 1.3.29, gnomepy 2.31.1, this repo at
`3ded4f7`), twice, with identical fills both times. `results/<name>/old/` holds its fills (and orders), and
`summary.json`. After the change the same run goes in `results/<name>/new/`, then:

```bash
python scripts/phase3_baselines/compare.py kalshi hyperliquid cs2
```

| Name | What it checks | Data | Old result |
|---|---|---|---|
| `kalshi` | backtest engine + migrated data | prod merged, Kalshi listing 97202, 2026-08-24 00:00–03:00 | 9,491 fills, PnL −7,810.20 |
| `hyperliquid` | backtest engine + migrated data | dev merged, Hyperliquid DYDX listing 1, 2026-09-28 00:00–06:00 | 2,254 fills, PnL −42.28 |
| `cs2` | the cs2_series rewrite to listing keys | synthetic books (`cs2_spec.json`) served from local files; match 2399331, Kalshi listings 1744070/1744069; pinned model, posterior and blend; prod `cs2_upcoming_*` rows | 93 orders, 38 fills, PnL −631.27 |

- `run_old.py`: `kalshi` and `hyperliquid`. A quoter that joins the touch and crosses the spread every 50th update.
  `simulation_as_of` is pinned so both runs read the same registry simulation parameters.
- `cs2_old.py`: `cs2`. Writes `cs2_spec.json` as old-layout files and serves them through `local_s3.py`, a Java
  S3Client over a local folder, so no collected books are needed.
- `cs2_spec.py`: regenerates `cs2_spec.json` (deterministic).

The new runs (`run_new.py`, `cs2_new.py`) are the same scripts on the Phase 3 code: market data keyed by stream
(`self.streams.listing_id(data.stream_id)`), and `cs2_new.py` writes the spec in the new key layout
(`{streamId}/book/mbp-10/YYYY/MM/DD/hh/mm.zst`). `kalshi` reruns after the prod merged bucket is migrated,
`hyperliquid` after dev's.

Running the old scripts again needs an environment with the released code: gnomepy 2.31.1
(`pip install "gnomepy[backtest]==2.31.1"`), this repo at `3ded4f7` with the `xgboost` and `pymc` extras, and
`GNOME_JARS` pointing at gnome-backtest 1.3.29's `-all` jar (built from the `gnome-backtest-1.3.29` tag).
`kalshi` and `cs2` run with `STAGE=prod AWS_PROFILE=prod`; `hyperliquid` with `STAGE=dev AWS_PROFILE=dev` and the
dev registry key (`REGISTRY_API_KEY_ID=rb0pbivke8`, `GNOME_REGISTRY_API_KEY` unset).
