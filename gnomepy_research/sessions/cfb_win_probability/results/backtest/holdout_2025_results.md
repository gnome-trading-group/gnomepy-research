# 2025 holdout: results

Scored once on 2026-10-15 under `holdout_2025_preregistration.md`. 808 FBS-vs-FBS games, every model and both market lines present. Nothing was changed after scoring. Full tables are in `comparison_2025.txt` and `report_2025_*.txt`.

## Decision

**Np stays the live model.**

| Rule | Result | Outcome |
|---|---|---|
| 2. Fall back to K if K beats Np by more than 0.005 | Np − K = +0.0000 [−0.0095, +0.0094] | not triggered |
| 3. Fall back to N if N beats Np by more than 0.005 | Np − N = −0.0034 [−0.0079, +0.0011] | not triggered |
| 4. No other model promoted | Kp scored best (0.5333; Kp − K −0.0013 [−0.0048, +0.0019]) | information only |
| 5. No betting go-ahead | — | live 2026 games logged before kickoff decide |

## Log loss

| | Overall | Weeks 1–4 (n=195) | Weeks 5–9 (n=269) | Weeks 10+ (n=298) | Postseason (n=46) | Calibration slope | Blend weight vs open |
|---|---|---|---|---|---|---|---|
| K | 0.5346 | 0.4842 | 0.5402 | 0.5469 | 0.6353 | 1.08 | +0.27 [−0.10, +0.62] |
| N | 0.5379 | 0.5033 | 0.5455 | 0.5371 | 0.6460 | 1.10 | +0.16 [−0.23, +0.56] |
| **Np** | **0.5346** | 0.4957 | 0.5442 | 0.5345 | 0.6439 | 1.11 | +0.27 [−0.06, +0.65] |
| Kp | 0.5333 | 0.4796 | 0.5399 | 0.5467 | 0.6350 | 1.09 | +0.32 [+0.00, +0.69] |
| Npt | 0.5393 | | | | | 1.11 | +0.21 [−0.14, +0.57] |
| Open | 0.5211 | 0.4729 | 0.5348 | 0.5205 | 0.6496 | 1.13 | |
| Close | 0.5205 | 0.4781 | 0.5348 | 0.5198 | 0.6217 | 1.10 | |

- **Gap to the open is wider than in 2023–24.** Np − open is +0.0135 [+0.0007, +0.0267], against +0.006 in 2023–24.
- **Weeks 1–4: Np trails K.** Np − K is +0.0115 [−0.0085, +0.0304]. The preseason fixes did not help early weeks on unseen 2025 teams.
- **Weeks 10+: Np leads K by 0.012.**
- **Npt, the tighter prior, is worse than Np** by +0.0047 [+0.0021, +0.0074], confirming the Phase 15 result.

## QB sensitivity

Scoring with expected starters instead of actual starters costs +0.0013 (K), +0.0020 (N), +0.0020 (Np), +0.0012 (Kp) and +0.0020 (Npt).

## Closing-line value (2025)

Slope of (close − open) on (model − open):

| Model | Slope | 95% CI | Correlation |
|---|---|---|---|
| K | +0.185 | [+0.137, +0.234] | 0.38 |
| Kp | +0.176 | [+0.128, +0.225] | 0.36 |
| N | +0.133 | [+0.089, +0.185] | 0.26 |
| Np | +0.128 | [+0.076, +0.178] | 0.25 |
| Np, expected QBs | +0.109 | [+0.065, +0.160] | 0.22 |

Every model's disagreement with the open predicts line movement. K's disagreement predicts it better than Np's.
