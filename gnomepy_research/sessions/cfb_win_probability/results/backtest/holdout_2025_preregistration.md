# 2025 holdout: pre-registration

Written 2026-10-15, before any 2025 outcome was scored. These rules are fixed. Nothing is re-run and no model is changed after the results are seen.

## What is scored

- **Models:** every model is scored once on all 2025 FBS-vs-FBS games:
  - A, B, C, D, E, F, G, H, I0, I, J0, J, K, L, Kc, M, N1, N2, N, Kp and Np;
  - the opening line and the closing line.
- **Training:** each model uses only data from before 2025:
  - efficiency hyperparameters fitted on 2019–2024;
  - the outcome layer trained on 2020–2024 (rolling origin);
  - Kc's tail calibrator fitted on K's out-of-sample 2023–24 predictions;
  - F's configuration as chosen on 2022.
- **Reported metrics:**
  - log loss, overall and for weeks 1–4;
  - paired bootstrap differences against K and against the open;
  - calibration slope;
  - blend weight against the open (information beyond the open);
  - CLV slope.
- **QB sensitivity:** K, N, Kp and Np are also scored with expected starters (`--qb-expected`, QB deltas zeroed). Every backtest otherwise uses the QB who actually started, which is not always known before kickoff. The sensitivity figure is the honest lower bound.

## Decision rules

1. **The live model is Np**, chosen before the holdout. Pooled 2021–24 out of sample it scored −0.0040 against K. It also carries the most information beyond the open (blend weight +0.40 in 2023–24).
2. **Fall back to K** if K beats Np on 2025 by more than 0.005 log loss, about one season's noise.
3. **Fall back to N** if Np is worse than N by more than 0.005. That would mean the prior fix doesn't generalize to teams it never saw (Delaware, Missouri State, and the in-window Jacksonville State, Sam Houston and Kennesaw State).
4. **No other model is promoted** for scoring best on 2025. The other models are reported for information only.
5. **No betting go-ahead from this run.** It answers which model goes live. Real-money decisions wait for live 2026 games logged before kickoff.

## Afterwards

2025 becomes training data. Live 2026 games, logged before kickoff, are the ongoing out-of-sample test.
