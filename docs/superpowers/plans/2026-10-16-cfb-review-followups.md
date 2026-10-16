# CFB Model Review Follow-ups Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Act on the five-reviewer audit of Npvk. The plan has four parts:
- **A.** Measure the model the way it is used: score information beyond the market price, with no QB look-ahead.
- **B.** Test the model changes the reviewers converged on, under that measurement.
- **C.** Build a market-aware trading layer and paper-trade it.
- **D.** Scope the larger structural work as follow-on plans.

**Architecture:** Parts A–C are session code under `gnomepy_research/sessions/cfb_win_probability/`; production `models/` is not touched.
- **Part A** adds a trading scorecard (`evaluation/market_metrics.py`) and a new pre-registered selection rule (`evaluation/trading_selection.py`). It then re-baselines with `--qb-expected` and re-scores every past candidate.
- **Part B** adds six candidates on Npvk. Five are outcome-layer and scoreboard-channel options in `win_probability.py`, `margin_filter.py` and `backtest.py`. The sixth (`NpvkAc`) is a post-processor, `evaluation/academy_scale.py`. All six are judged by the Part A rule.
- **Part C** adds `trading/decision.py`: a partially pooled market blend, skip rules, fractional Kelly and a team-season cap. On top of it sit a 2022–24 trading backtest and a paper-trade logger with an anytime-valid confidence sequence.

**Tech Stack:** Python 3.13, numpy/pandas/scipy, scikit-learn (already used by `compare_models`), PyMC + nutpie (existing outcome layer), pytest, Poetry. Data via `gnomepy.data.DatasetStore` (read-only).

**Spec:** the five reviews and the shared briefing, all in the worktree's git-ignored `.superpowers/reviews/2026-10-16-model-review/`:
- `BRIEFING.md`
- `review_mit_ml.md`
- `review_cal_frequentist.md`
- `review_utaustin_bayesian.md`
- `review_oxford_frequentist.md`
- `review_harvard_bayesian.md`

The research log `docs/superpowers/plans/2026-10-13-cfb-split-efficiency-and-qb-status.md` (Phases 1–26) holds the history every task builds on. Two corrections to the briefing are binding here:
- The outcome layer trains on an **expanding window from 2020**, not "3 prior seasons".
- `qb_delta_diff` uses each game's **actual first passer** unless `--qb-expected` is passed.

## Global Constraints

- **Production code:** `gnomepy_research/models/` is production code, and this plan does not modify it. Every change lives under `gnomepy_research/sessions/cfb_win_probability/` and `tests/sessions/`.
- **Defaults:** every new parameter defaults to today's behaviour. Existing tests must pass unmodified, and Npvk's configuration must produce the same design arrays as before.
- **Point-in-time:** blend weights, cell weights, academy factors and prior maps are fitted only on seasons before the season they score.
- **2025:** never read it for any decision. Never run `backtest 2025 --holdout`. The evidence base is 2021–24; 2026 is live and only for paper trading.
- **Scored seasons:** from Phase 27 on, the scored seasons are **2022–24**. 2021 is reported as a stress test only, because its outcome layer saw only the 2020 COVID season and the blend has no earlier season to fit on.
- **QB information:** from Phase 28 on, every market-facing comparison uses `--qb-expected` predictions (file suffix `_qbexp`).
- **Public repo:** the repo is PUBLIC. Never commit any of these: per-game market data, win totals, Kalshi/Polymarket data, paper-trade logs or built HTML. Text reports of aggregates under `results/backtest/` are fine. Win totals stay in the scratchpad CSV `/private/tmp/claude-501/-Users-aannestrand/555aa52e-cdb5-40ed-bed0-8c3d3128a4f7/scratchpad/market_preseason_strength.csv`, passed by path.
- **AWS:** only `AWS_PROFILE=AWSAdministratorAccess-241533121172`. Never read credential files or try another profile. On a credential error, stop and ask the user to run `aws sso login --profile AWSAdministratorAccess-241533121172`.
- **Staging:** stage files by explicit path only. Never commit the untracked hyperparameter JSONs in `gnomepy_research/models/cfb/hyperparameters/`.
- **Remote and history:** no push, tag or release without the user's explicit OK. No force-push.
- **Commit messages** end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Code style:** imports go at the top of the file, never inside functions. No comments unless the why is non-obvious.
- **Paper trading only:** no orders, ever. Analysis, not financial advice.
- **Test command:** `poetry run pytest tests/sessions/test_cfb_*.py -q --noconftest` (the JAR conftest is excluded).
- **Launching backtests:** zsh does not word-split variables, so always pass season pairs as two literal arguments. Example: `AWS_PROFILE=AWSAdministratorAccess-241533121172 poetry run python -m gnomepy_research.sessions.cfb_win_probability.evaluation.backtest 2021 2022 --model Npvk --qb-expected`.

## Review Focus

1. **A scored season with no earlier season to fit the blend on** (`--seasons` including 2021, or a model with no 2021 file). Expected: `ValueError` naming the season, never a blend fitted on the season it scores. Test in Task 1.
2. **Games missing an open or a close line.** Expected: dropped, with the count printed in the report, never filled with 0.5. Test in Task 2.
3. **Probabilities of exactly 0 or 1** (clipped calibrators, FCS blowouts). Expected: logits are clipped, so blends, beliefs and stakes stay finite. Tests in Tasks 1 and 10.
4. **Swapping home and away at a neutral site with the new `kf_draws` term.** Expected: the term is exactly negated draw for draw, so p + p_swapped = 1 still holds. Test in Task 4.
5. **One team in several bets in a slate, home in one and away in another.** Expected: the team-season cap holds on both teams in every bet, and stakes are cut, never reversed. Test in Task 10.

---

## File map

| File | Change |
|---|---|
| `sessions/cfb_win_probability/evaluation/market_metrics.py` | **new**: forward-chained blend, trading value, candidate-vs-reference blend comparison, clustered CLV slope, empirical-Bayes shrinkage, pass probability, confidence sequence |
| `sessions/cfb_win_probability/evaluation/trading_selection.py` | **new**: the Phase 27 pre-registered rule and CLI |
| `sessions/cfb_win_probability/evaluation/rescore_candidates.py` | **new**: re-scores the past candidates and the QB re-baseline report |
| `sessions/cfb_win_probability/evaluation/academy_scale.py` | **new**: academy margin-scale post-processor (`NpvkAc`) |
| `sessions/cfb_win_probability/models/win_probability.py` | `kf_draws`, `coach_fade` and `games_before` |
| `sessions/cfb_win_probability/models/margin_filter.py` | `fcs_var` and `q_post` parameters in `run_filter` and `kf_table` |
| `sessions/cfb_win_probability/evaluation/backtest.py` | configs `NpvkU`, `NpvkF`, `NpvkN2`, `NpvkS`, `NpvkR`; flags; the `kf_table` arguments |
| `sessions/cfb_win_probability/trading/__init__.py` | **new**, empty |
| `sessions/cfb_win_probability/trading/decision.py` | **new**: cells, partially pooled blend, skip rules, Kelly stakes, team-season cap, returns |
| `sessions/cfb_win_probability/trading/trading_backtest.py` | **new**: 2022–24 trading backtest report |
| `sessions/cfb_win_probability/trading/paper_trade.py` | **new**: `log` and `score` subcommands |
| `tests/sessions/test_cfb_market_metrics.py`, `test_cfb_trading_selection.py`, `test_cfb_academy_scale.py`, `test_cfb_trading_decision.py`, `test_cfb_paper_trade.py` | **new** tests |
| `tests/sessions/test_cfb_margin_filter.py` | Part B tests appended |
| `docs/superpowers/plans/2026-10-13-cfb-split-efficiency-and-qb-status.md` | Phase 27–29 pre-registrations and results |

All paths below are relative to the worktree root `/Users/aannestrand/Desktop/gnomepy-research-worktrees/cfb-split`. `S` abbreviates `gnomepy_research/sessions/cfb_win_probability`.

---

# Part A — Measure what trading needs (Phase 27)

### Task 1: Trading scorecard module

**Files:**
- Create: `S/evaluation/market_metrics.py`
- Test: `tests/sessions/test_cfb_market_metrics.py`

**Interfaces:**
- Consumes these existing functions:
  - `compare_models._blend_fit(y, p_model, p_ref) -> (w_model, w_ref)` (no-intercept logistic);
  - `compare_models.logit(p)`, which clips to [1e-4, 1 − 1e-4];
  - `metrics.log_loss(y, p) -> per-game array`;
  - `metrics.paired_bootstrap(y, p_a, p_b, n_boot, seed) -> {diff, se, lo95, hi95, p_a_worse}`, where diff = LL(a) − LL(b).
- Produces:
  - `FIRST_BLEND_SEASON = 2021`
  - `forward_blend(d, model, market, scored) -> pd.Series`
  - `trading_value(d, model, market, scored) -> {"value", "se", "p_positive"}`
  - `compare_trading(d, cand, ref, market, scored) -> {"diff", "se", "lo95", "hi95", "p_worse", "per_season"}`
  - `clv_slope(move, lean, cluster) -> {"slope", "se", "t"}`
  - `shrink_effects(deltas, ses) -> np.ndarray`
  - `pass_probability(effect, se, threshold, p_max) -> float`
  - `confidence_sequence(x, alpha=0.10, t0=200.0) -> pd.DataFrame[n, mean, lo, hi]`

- [ ] **Step 1: Write the failing tests**

```python
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gnomepy_research.sessions.cfb_win_probability.evaluation import market_metrics as mm


def synthetic(n=3000, seasons=(2021, 2022, 2023, 2024), seed=0):
    """Market logit L0, hidden signal s the market lacks; 'cand' knows s, 'ref' does not."""
    rng = np.random.default_rng(seed)
    rows = []
    for season in seasons:
        l0, s = rng.normal(0, 1.2, n), rng.normal(0, 0.6, n)
        y = (rng.random(n) < 1 / (1 + np.exp(-(l0 + s)))).astype(float)
        sig = lambda z: 1 / (1 + np.exp(-z))
        rows.append(pd.DataFrame({"season": season, "week": rng.integers(1, 14, n), "season_type": "regular", "home_win": y,
                                  "p_open": sig(l0), "p_close": sig(l0 + 0.3 * s),
                                  "ref": sig(l0 + rng.normal(0, 0.5, n)), "cand": sig(l0 + s + rng.normal(0, 0.5, n))}))
    return pd.concat(rows, ignore_index=True)


def test_forward_blend_uses_earlier_seasons_only_and_refuses_without_one():
    d = synthetic(n=800)
    poisoned = d.copy()
    poisoned.loc[poisoned.season == 2023, "home_win"] = 1.0 - poisoned.loc[poisoned.season == 2023, "home_win"]
    a = mm.forward_blend(d, "cand", "p_open", [2023])
    b = mm.forward_blend(poisoned, "cand", "p_open", [2023])
    np.testing.assert_allclose(a[d.season == 2023], b[d.season == 2023])      # 2023 outcomes never used for 2023
    assert a[d.season != 2023].isna().all()
    with pytest.raises(ValueError, match="2021"):
        mm.forward_blend(d, "cand", "p_open", [2021])


def test_trading_value_sees_information_the_market_lacks():
    d = synthetic()
    v_cand = mm.trading_value(d, "cand", "p_open", [2022, 2023, 2024])
    v_ref = mm.trading_value(d, "ref", "p_open", [2022, 2023, 2024])
    assert v_cand["value"] > 0.01 and v_cand["p_positive"] > 0.99
    assert abs(v_ref["value"]) < 0.002


def test_compare_trading_and_extreme_probabilities_stay_finite():
    d = synthetic()
    r = mm.compare_trading(d, "cand", "ref", "p_open", [2022, 2023, 2024])
    assert r["diff"] < -0.01 and r["p_worse"] < 0.01 and set(r["per_season"]) == {2022, 2023, 2024}
    d.loc[:10, "cand"] = 1.0
    d.loc[11:20, "cand"] = 0.0
    assert np.isfinite(mm.forward_blend(d, "cand", "p_open", [2022, 2023, 2024]).dropna()).all()


def test_clv_slope_recovers_a_planted_slope_with_clustered_se():
    rng = np.random.default_rng(3)
    lean = rng.normal(0, 0.08, 4000)
    move = 0.4 * lean + rng.normal(0, 0.02, 4000)
    r = mm.clv_slope(move, lean, rng.integers(0, 60, 4000))
    assert abs(r["slope"] - 0.4) < 0.02 and r["t"] > 20


def test_shrinkage_and_pass_probability():
    same = mm.shrink_effects([-0.001, -0.001, -0.001], [0.0006] * 3)
    np.testing.assert_allclose(same, -0.001)
    spread = mm.shrink_effects([-0.0030, -0.0005, 0.0010, -0.0002], [0.0006] * 4)
    assert spread[0] > -0.0030 and spread[2] < 0.0010                       # pulled toward the mean
    assert mm.pass_probability(-0.001, 0.0006, -0.001, 0.10) == pytest.approx(0.5)
    assert mm.pass_probability(0.0, 0.0006, -0.001, 0.10) < 0.06


def test_confidence_sequence_covers_the_mean_and_detects_a_real_edge():
    null = mm.confidence_sequence(np.random.default_rng(4).normal(0, 1, 3000))
    assert ((null.lo <= 0) & (null.hi >= 0)).all()                           # holds at every t at once
    assert np.isinf(null.hi.iloc[mm.CS_MIN_N - 2]) and np.isfinite(null.hi.iloc[mm.CS_MIN_N - 1])
    edge = mm.confidence_sequence(np.random.default_rng(4).normal(0.15, 1, 3000))
    assert edge.lo.iloc[-1] > 0 and edge.lo.iloc[-1] < 0.15 < edge.hi.iloc[-1]
    assert list(edge.columns) == ["n", "mean", "lo", "hi"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `poetry run pytest tests/sessions/test_cfb_market_metrics.py -q --noconftest`
Expected: FAIL with `ModuleNotFoundError: ... market_metrics`.

- [ ] **Step 3: Write the implementation**

```python
"""Trading scorecard (Phase 27): what a forecast adds beyond the market price.

A log-utility (Kelly) bettor with belief p at a fair price q grows wealth by LL(q) - LL(p) per game, so a model's
trading value is what it adds to the price: the log loss of the market-conditional belief
logit p* = w_model * logit(model) + w_mkt * logit(market), its weights fitted on earlier seasons only, against the
market's own log loss. Closing-line value (does the line move from open to close toward the model?) is the
lower-variance companion measure.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from gnomepy_research.sessions.cfb_win_probability.evaluation.compare_models import _blend_fit, logit
from gnomepy_research.sessions.cfb_win_probability.evaluation.metrics import log_loss, paired_bootstrap

FIRST_BLEND_SEASON = 2021
N_BOOT = 2000
CS_MIN_N = 20


def forward_blend(d: pd.DataFrame, model: str, market: str, scored: list[int]) -> pd.Series:
    """Blended probability for games of the `scored` seasons, the weights fitted on seasons FIRST_BLEND_SEASON..s-1
    of `d` (columns season, home_win and the two probability columns); NaN elsewhere."""
    out = pd.Series(np.nan, index=d.index, dtype=float)
    for s in scored:
        train = d[(d["season"] >= FIRST_BLEND_SEASON) & (d["season"] < s)]
        if train.empty:
            raise ValueError(f"no season from {FIRST_BLEND_SEASON} before {s} to fit the blend on")
        w_model, w_mkt = _blend_fit(train["home_win"].to_numpy(), train[model].to_numpy(), train[market].to_numpy())
        cur = (d["season"] == s).to_numpy()
        z = w_model * logit(d.loc[cur, model].to_numpy()) + w_mkt * logit(d.loc[cur, market].to_numpy())
        out[cur] = 1 / (1 + np.exp(-z))
    return out


def trading_value(d: pd.DataFrame, model: str, market: str, scored: list[int]) -> dict:
    """Per-game LL(market) - LL(blend of model and market) over the scored seasons: positive = the model adds
    information the price lacks."""
    blend = forward_blend(d, model, market, scored)
    m = d["season"].isin(scored).to_numpy()
    r = paired_bootstrap(d["home_win"].to_numpy()[m], d[market].to_numpy()[m], blend.to_numpy()[m], n_boot=N_BOOT)
    return {"value": r["diff"], "se": r["se"], "p_positive": r["p_a_worse"]}


def compare_trading(d: pd.DataFrame, cand: str, ref: str, market: str, scored: list[int]) -> dict:
    """Candidate's blend minus the reference's blend in log loss over the scored seasons (negative = the candidate
    adds more beyond the market), with the per-season differences."""
    bc = forward_blend(d, cand, market, scored).to_numpy()
    br = forward_blend(d, ref, market, scored).to_numpy()
    y, season = d["home_win"].to_numpy(), d["season"].to_numpy()
    m = np.isin(season, scored)
    r = paired_bootstrap(y[m], bc[m], br[m], n_boot=N_BOOT)
    per = {int(s): float(log_loss(y[season == s], bc[season == s]).mean() - log_loss(y[season == s], br[season == s]).mean())
           for s in scored}
    return {"diff": r["diff"], "se": r["se"], "lo95": r["lo95"], "hi95": r["hi95"], "p_worse": r["p_a_worse"],
            "per_season": per}


def clv_slope(move: np.ndarray, lean: np.ndarray, cluster: np.ndarray) -> dict:
    """OLS slope (with intercept) of the line move (close - open) on a lean (model - open, or candidate - reference),
    with a standard error clustered on `cluster` (season-week): games in one week share news shocks."""
    move, lean = np.asarray(move, float), np.asarray(lean, float)
    x = lean - lean.mean()
    sxx = float(x @ x)
    b = float(x @ move) / sxx
    e = move - move.mean() - b * x
    s = pd.Series(x * e).groupby(np.asarray(cluster)).sum().to_numpy()
    se = float(np.sqrt(s @ s) / sxx)
    return {"slope": b, "se": se, "t": b / se if se > 0 else float("nan")}


def shrink_effects(deltas, ses) -> np.ndarray:
    """Empirical-Bayes (method of moments) estimates of each candidate's true effect: each estimate is pulled toward
    the precision-weighted mean by tau^2 / (tau^2 + se^2), tau^2 the spread of true effects beyond sampling noise.
    Corrects the winner's curse of reporting the best of many tested candidates."""
    d, s2 = np.asarray(deltas, float), np.asarray(ses, float) ** 2
    w = 1 / s2
    mu = float((w * d).sum() / w.sum())
    tau2 = max(float(d.var(ddof=1) - s2.mean()), 0.0) if len(d) > 1 else 0.0
    return mu + tau2 / (tau2 + s2) * (d - mu)


def pass_probability(effect: float, se: float, threshold: float, p_max: float) -> float:
    """Chance a candidate with this true effect passes 'diff <= threshold and bootstrap P(worse) <= p_max', using
    the normal approximation P(worse) = Phi(diff / se)."""
    cut = min(threshold, float(norm.ppf(p_max)) * se)
    return float(norm.cdf((cut - effect) / se))


def confidence_sequence(x: np.ndarray, alpha: float = 0.10, t0: float = 200.0) -> pd.DataFrame:
    """Running mean of x with a two-sided normal-mixture confidence sequence (Robbins): it holds at every t at once,
    so a live record can be checked after each game without inflating the error rate. The variance proxy is the
    running sample variance (a plug-in), unbounded before CS_MIN_N games, where it is still near zero; t0 sets where
    the boundary is tightest. Measured null violation rate over 400 seeds: 4% at alpha 0.10."""
    x = np.asarray(x, float)
    t = np.arange(1, len(x) + 1, dtype=float)
    mean = np.cumsum(x) / t
    var = np.maximum(pd.Series(x).expanding().var().fillna(0.0).to_numpy(), 1e-12)
    v, rho = t * var, t0 * var
    radius = np.where(t < CS_MIN_N, np.inf, np.sqrt((v + rho) * np.log((v + rho) / (rho * alpha ** 2))) / t)
    return pd.DataFrame({"n": t.astype(int), "mean": mean, "lo": mean - radius, "hi": mean + radius})
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `poetry run pytest tests/sessions/test_cfb_market_metrics.py -q --noconftest`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add gnomepy_research/sessions/cfb_win_probability/evaluation/market_metrics.py tests/sessions/test_cfb_market_metrics.py
git commit -m "Trading scorecard: forward-chained market blend, trading value, clustered CLV, shrinkage, confidence sequence

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 2: The Phase 27 selection rule (pre-registered)

**Files:**
- Create: `S/evaluation/trading_selection.py`
- Test: `tests/sessions/test_cfb_trading_selection.py`
- Modify: `docs/superpowers/plans/2026-10-13-cfb-split-efficiency-and-qb-status.md` (Phase 27 section, written in Step 5 **before** any Phase 27/28 run)

**Interfaces:**
- Consumes: Task 1's `compare_trading`, `trading_value`, `clv_slope` and `pass_probability`; `candidate_selection.TAGS`; `candidate_selection.log_loss`; `backtest.RESULTS`; `compare_models.market_probs(lines) -> DataFrame[game_id, p_open, p_close]`.
- Produces:
  - `SCORED = [2022, 2023, 2024]`
  - the constants `TRADE_GAIN`, `MAX_P_WORSE`, `CLOSE_HARM`, `STANDALONE_HARM`, `MAX_SEASON_LOSS`, `UNION_GAIN`
  - `load_predictions(models, suffix) -> DataFrame` (index game_id)
  - `attach_market(d, market) -> (DataFrame, n_dropped)`
  - `evaluate(d, cand, ref) -> dict` (keys `passes`, `open`, `close`, `standalone`, `per_season`, `clv_change`, `stress_2021`)
  - the CLI `--label --ref --candidates [--suffix] [--nested A:B ...]`, which writes `RESULTS/{label}_trading.txt` and `.json`

**The rule** (these constants are the pre-registration):
- **Primary:** the candidate's forward-chained blend with the **open** beats the reference's by Δ ≤ **−0.0003**, with bootstrap P(worse) ≤ **0.10**, pooled over 2022–24.
- **Guards:**
  - the blend with the **close** is not worse by more than **+0.0003**;
  - standalone log loss is not worse by more than **+0.0005** pooled;
  - no season's standalone log loss is worse by more than **0.003**.
- **Nested candidates** (unions): need a further blend Δ ≤ **−0.0003** against the best single candidate.

- [ ] **Step 1: Write the failing tests**

```python
from __future__ import annotations

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cfb_win_probability.evaluation import trading_selection as ts
from tests.sessions.test_cfb_market_metrics import synthetic


def test_rule_passes_real_information_and_rejects_noise():
    d = synthetic()
    d["noise"] = 1 / (1 + np.exp(-(np.log(d.ref / (1 - d.ref)) + np.random.default_rng(9).normal(0, 0.3, len(d)))))
    good = ts.evaluate(d, "cand", "ref")
    assert good["passes"] and good["open"]["diff"] <= ts.TRADE_GAIN and good["clv_change"]["slope"] > 0
    assert set(good["per_season"]) == set(ts.SCORED) and "stress_2021" in good
    assert not ts.evaluate(d, "noise", "ref")["passes"]


def test_standalone_guard_blocks_a_candidate_that_wrecks_raw_accuracy():
    d = synthetic()
    lg = np.log(d.cand / (1 - d.cand))
    d["wild"] = 1 / (1 + np.exp(-3.0 * lg))                    # same information, badly over-confident on its own
    r = ts.evaluate(d, "wild", "ref")
    assert r["standalone"]["diff"] > ts.STANDALONE_HARM and not r["passes"]


def test_attach_market_drops_and_counts_missing_lines():
    d = pd.DataFrame({"season": 2022, "home_win": 1.0, "Npvk": 0.6}, index=pd.Index([1, 2, 3], name="game_id"))
    market = pd.DataFrame({"game_id": [1, 2, 4], "p_open": [0.55, np.nan, 0.5], "p_close": [0.57, 0.5, 0.5]})
    out, dropped = ts.attach_market(d, market)
    assert out.index.tolist() == [1] and dropped == 2
    assert out.loc[1, "p_open"] == 0.55 and out.loc[1, "p_close"] == 0.57
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `poetry run pytest tests/sessions/test_cfb_trading_selection.py -q --noconftest`
Expected: FAIL with `ModuleNotFoundError: ... trading_selection`.

- [ ] **Step 3: Write the implementation**

```python
"""Pre-registered trading rule (Phase 27): select candidates by what they add beyond the market.

    poetry run python -m gnomepy_research.sessions.cfb_win_probability.evaluation.trading_selection \
        --label p28 --ref Npvk --candidates NpvkU NpvkF --suffix _qbexp [--nested NpvkUF:NpvkU]

Scored seasons are 2022-24. 2021 is a stress test only: there is no earlier season to fit its blend on, and its
outcome layer saw only 2020. A candidate passes against the reference when its forward-chained blend with the open
beats the reference's by at least TRADE_GAIN with bootstrap P(worse) <= MAX_P_WORSE, its blend with the close is
not worse by more than CLOSE_HARM, and its standalone log loss is not worse by more than STANDALONE_HARM pooled or
MAX_SEASON_LOSS in any scored season. A nested candidate (A:B, A extends B) beats B only with a further UNION_GAIN.
2025 is never read.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from gnomepy.data import DatasetStore
from gnomepy_research.sessions.cfb_win_probability.evaluation.backtest import RESULTS
from gnomepy_research.sessions.cfb_win_probability.evaluation.candidate_selection import TAGS, log_loss
from gnomepy_research.sessions.cfb_win_probability.evaluation.compare_models import market_probs
from gnomepy_research.sessions.cfb_win_probability.evaluation.market_metrics import (
    clv_slope, compare_trading, pass_probability, trading_value)
from gnomepy_research.sessions.cfb_win_probability.evaluation.metrics import paired_bootstrap

SCORED = [2022, 2023, 2024]
TRADE_GAIN = -0.0003
MAX_P_WORSE = 0.10
CLOSE_HARM = 0.0003
STANDALONE_HARM = 0.0005
MAX_SEASON_LOSS = 0.003
UNION_GAIN = -0.0003
CONTEXT = ["season", "week", "season_type", "home_win", "home_team", "away_team", "home_conference", "away_conference",
           "home_id", "away_id", "kickoff"]


def load_predictions(models: list[str], suffix: str) -> pd.DataFrame:
    """p_home of every model (files predictions_{tag}_{model}{suffix}.parquet for both tags) beside the game context,
    index game_id; column names are the bare model names."""
    frames = {m: pd.concat([pd.read_parquet(RESULTS / f"predictions_{t}_{m}{suffix}.parquet") for t in TAGS])
              .drop_duplicates("game_id").set_index("game_id") for m in models}
    base = frames[models[0]][[c for c in CONTEXT if c in frames[models[0]].columns]]
    return base.join(pd.DataFrame({m: f["p_home"] for m, f in frames.items()}), how="inner").dropna(subset=["home_win", *models])


def attach_market(d: pd.DataFrame, market: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Join p_open and p_close by game_id; games missing either line are dropped and counted, never filled."""
    j = d.join(market.set_index("game_id")[["p_open", "p_close"]], how="left")
    keep = j["p_open"].notna() & j["p_close"].notna()
    return j[keep], int((~keep).sum())


def _standalone(d: pd.DataFrame, cand: str, ref: str, seasons: list[int]) -> tuple[dict, dict]:
    m = d["season"].isin(seasons).to_numpy()
    y = d["home_win"].to_numpy()
    r = paired_bootstrap(y[m], d[cand].to_numpy()[m], d[ref].to_numpy()[m])
    per = {int(s): float(log_loss(g["home_win"].to_numpy(), g[cand].to_numpy()).mean()
                         - log_loss(g["home_win"].to_numpy(), g[ref].to_numpy()).mean()) for s, g in d[m].groupby("season")}
    return r, per


def week_cluster(d: pd.DataFrame) -> np.ndarray:
    return (d["season"].to_numpy() * 100 + np.where(d["season_type"] == "postseason", 99, d["week"].to_numpy())).astype(int)


def evaluate(d: pd.DataFrame, cand: str, ref: str) -> dict:
    open_ = compare_trading(d, cand, ref, "p_open", SCORED)
    close = compare_trading(d, cand, ref, "p_close", SCORED)
    sa, per = _standalone(d, cand, ref, SCORED)
    stress, _ = _standalone(d, cand, ref, [2021]) if (d["season"] == 2021).any() else ({"diff": float("nan")}, {})
    m = d["season"].isin(SCORED).to_numpy()
    clv = clv_slope((d["p_close"] - d["p_open"]).to_numpy()[m], (d[cand] - d[ref]).to_numpy()[m], week_cluster(d)[m])
    ok = (open_["diff"] <= TRADE_GAIN and open_["p_worse"] <= MAX_P_WORSE and close["diff"] <= CLOSE_HARM
          and sa["diff"] <= STANDALONE_HARM and max(per.values()) <= MAX_SEASON_LOSS)
    return {"passes": bool(ok), "open": open_, "close": close, "standalone": sa, "per_season": per,
            "clv_change": clv, "stress_2021": stress["diff"]}


def model_lines(d: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    m = d["season"].isin(SCORED).to_numpy()
    y = d["home_win"].to_numpy()[m]
    rows = {}
    for x in models:
        vo, vc = trading_value(d, x, "p_open", SCORED), trading_value(d, x, "p_close", SCORED)
        clv = clv_slope((d["p_close"] - d["p_open"]).to_numpy()[m], (d[x] - d["p_open"]).to_numpy()[m], week_cluster(d)[m])
        rows[x] = {"log loss": log_loss(y, d[x].to_numpy()[m]).mean(), "value vs open": vo["value"],
                   "se": vo["se"], "value vs close": vc["value"], "CLV slope": clv["slope"], "CLV t": clv["t"]}
    rows["open"] = {"log loss": log_loss(y, d["p_open"].to_numpy()[m]).mean()}
    rows["close"] = {"log loss": log_loss(y, d["p_close"].to_numpy()[m]).mean()}
    return pd.DataFrame(rows).T


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--candidates", nargs="+", required=True)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--nested", nargs="*", default=[], help="A:B pairs where A extends B")
    args = ap.parse_args()
    models = [args.ref, *args.candidates]
    d, dropped = attach_market(load_predictions(models, args.suffix), market_probs(DatasetStore().load("cfb_lines")))
    res = {c: evaluate(d, c, args.ref) for c in args.candidates}
    nested = dict(p.split(":") for p in args.nested)
    for a, b in nested.items():
        res[f"{a} vs {b}"] = evaluate(d, a, b)
    passing = [c for c in args.candidates if res[c]["passes"]]
    for a, b in nested.items():
        if a in passing and b in passing and res[f"{a} vs {b}"]["open"]["diff"] > UNION_GAIN:
            passing.remove(a)
    chosen = min(passing, key=lambda c: res[c]["open"]["diff"]) if passing else args.ref
    se = res[args.candidates[0]]["open"]["se"]
    power = ", ".join(f"{e:+.4f}: {pass_probability(e, se, TRADE_GAIN, MAX_P_WORSE):.0%}" for e in (-0.0003, -0.0006, -0.0010))
    lines = [f"{k}: blend vs open {v['open']['diff']:+.4f} [{v['open']['lo95']:+.4f}, {v['open']['hi95']:+.4f}] "
             f"P(worse) {v['open']['p_worse']:.2f}; vs close {v['close']['diff']:+.4f}; standalone {v['standalone']['diff']:+.4f} "
             f"(per season " + ", ".join(f"{s} {x:+.4f}" for s, x in v["per_season"].items()) + f"); 2021 stress "
             f"{v['stress_2021']:+.4f}; CLV of the change {v['clv_change']['slope']:+.3f} (t {v['clv_change']['t']:+.1f}); "
             f"passes: {v['passes']}" for k, v in res.items()]
    text = "\n\n".join([
        f"{args.label} trading selection vs {args.ref}{args.suffix}: scored 2022-24, {int(d['season'].isin(SCORED).sum())} FBS "
        f"games with open and close lines ({dropped} games dropped for a missing line); 2025 not read",
        "per model (scored seasons; value = LL(market) - LL(forward blend), positive = information beyond the price):\n"
        + model_lines(d, models).round(4).to_string(),
        "\n".join(lines),
        f"rule power at the observed se {se:.4f} for true blend effects {power}",
        f"chosen: {chosen}"])
    (RESULTS / f"{args.label}_trading.txt").write_text(text + "\n")
    (RESULTS / f"{args.label}_trading.json").write_text(json.dumps({"chosen": chosen, "results": res}, indent=2, default=float) + "\n")
    print(text)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `poetry run pytest tests/sessions/test_cfb_trading_selection.py tests/sessions/test_cfb_market_metrics.py -q --noconftest`
Expected: 9 passed.

- [ ] **Step 5: Pre-register Phase 27 in the research log**

Append this section to `docs/superpowers/plans/2026-10-13-cfb-split-efficiency-and-qb-status.md` just before `### Phase 21 (proposed): availability data`. Its first line, `## Phase 27: …`, is a level-two heading, so it ends Phase 26.

```markdown
## Phase 27: Measuring what trading needs (the five-reviewer audit)

Added 2026-10-16. The audit's reviews live in `.superpowers/reviews/2026-10-16-model-review/` (git-ignored).

**The findings that change the protocol:**
- **The old rule scores standalone log loss, which rewards becoming the market.** A Kelly bettor's growth against a fair price q is LL(q) − LL(p). So the trading value of a model is LL(market) − LL(market-conditional belief).
- **Every backtest so far uses the actual first passer** (`historical_starters`), which the open never knew.
- **The outcome layer is an expanding window from 2020,** so 2021 was trained on the COVID season alone.

**The new rule** (`evaluation/trading_selection.py`, pre-registered here):
- **Scored seasons:** 2022–24. 2021 is reported as a stress test.
- **Blend:** forward-chained, logit p* = w_model·logit(model) + w_mkt·logit(market), with the weights fitted on seasons from 2021 up to the season before.
- **Pass:** blend with the open better than the reference's by Δ ≤ −0.0003, with bootstrap P(worse) ≤ 0.10.
- **Guards:**
  - blend with the close not worse by more than +0.0003;
  - standalone not worse by more than +0.0005 pooled;
  - no scored season worse standalone by more than 0.003.
- **Unions:** a further Δ ≤ −0.0003 against the best single candidate.
- **Reported:** closing-line value of the change (week-clustered), the rule's power, and empirical-Bayes shrunk effects.
- **QB information:** from Phase 28 on, all comparisons use `--qb-expected` predictions.
```

- [ ] **Step 6: Commit**

```bash
git add gnomepy_research/sessions/cfb_win_probability/evaluation/trading_selection.py tests/sessions/test_cfb_trading_selection.py docs/superpowers/plans/2026-10-13-cfb-split-efficiency-and-qb-status.md
git commit -m "Phase 27 pre-registration: trading selection rule (blend beyond the market, 2022-24, qb-expected)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 3: QB re-baseline and re-score of past candidates

**Files:**
- Create: `S/evaluation/rescore_candidates.py`
- Test: `tests/sessions/test_cfb_trading_selection.py` (append)
- Modify: the research log (Phase 27 result)

**Interfaces:**
- Consumes: Task 2's `load_predictions`, `attach_market`, `evaluate`, `SCORED` and `week_cluster`; Task 1's `shrink_effects`.
- Produces:
  - `PAIRS: list[tuple[str, str]]` (candidate, the reference it was tested against)
  - `rescore(d, pairs) -> DataFrame` (columns cand, ref, standalone, standalone_shrunk, blend_open, blend_open_shrunk, blend_close, clv_t, passes_new)
  - `qb_table(d_post, d_exp, models) -> DataFrame`

- [ ] **Step 1: Write the failing test** (append to `tests/sessions/test_cfb_trading_selection.py`)

```python
def test_rescore_reports_raw_and_shrunk_effects():
    from gnomepy_research.sessions.cfb_win_probability.evaluation.rescore_candidates import qb_table, rescore
    d = synthetic()
    d["noise"] = 1 / (1 + np.exp(-(np.log(d.ref / (1 - d.ref)) + np.random.default_rng(9).normal(0, 0.3, len(d)))))
    t = rescore(d, [("cand", "ref"), ("noise", "ref")]).set_index("cand")
    assert t.loc["cand", "passes_new"] and not t.loc["noise", "passes_new"]
    assert abs(t.loc["cand", "blend_open_shrunk"]) <= abs(t.loc["cand", "blend_open"]) + 1e-12
    q = qb_table(d, d.assign(ref=d.p_open), ["ref"])
    assert list(q.columns) == ["ex-post QB", "qb-expected", "change"] and q.loc["ref", "change"] != 0
```

- [ ] **Step 2: Run it to verify it fails**

Run: `poetry run pytest tests/sessions/test_cfb_trading_selection.py::test_rescore_reports_raw_and_shrunk_effects -q --noconftest`
Expected: FAIL with `ModuleNotFoundError: ... rescore_candidates`.

- [ ] **Step 3: Write the implementation**

```python
"""Phase 27 re-score: every past candidate under the trading rule, and the QB re-baseline.

    poetry run python -m gnomepy_research.sessions.cfb_win_probability.evaluation.rescore_candidates

The candidate files use ex-post QB starters (they predate Phase 27); differences between candidates share that
information, so the ranking is informative, but adoption needs a qb-expected run under the Phase 28 rule.
"""
from __future__ import annotations

import pandas as pd

from gnomepy.data import DatasetStore
from gnomepy_research.sessions.cfb_win_probability.evaluation.backtest import RESULTS
from gnomepy_research.sessions.cfb_win_probability.evaluation.candidate_selection import log_loss
from gnomepy_research.sessions.cfb_win_probability.evaluation.compare_models import market_probs
from gnomepy_research.sessions.cfb_win_probability.evaluation.market_metrics import shrink_effects
from gnomepy_research.sessions.cfb_win_probability.evaluation.trading_selection import SCORED, attach_market, evaluate, load_predictions

PAIRS = [("Kp", "K"), ("Np", "N"), ("Npt", "Np"), ("Npm", "Np"), ("Npmt", "Np"), ("Npx", "Np"), ("Npc", "Np"), ("Npxc", "Np"),
         ("Npv", "Np"), ("Npvl", "Npv"), ("Npv2", "Npv"), ("Npvs", "Npv"), ("Npvg", "Npv"), ("Npvk", "Npv"), ("Npvk0", "Npv"),
         ("NpvkA", "Npvk"), ("NpvkB", "Npvk"), ("NpvkC", "Npvk"), ("NpvkN", "Npvk"), ("NpvkO", "Npvk"), ("NpvkW", "Npvk"),
         ("NpvkWA", "Npvk")]
QB_MODELS = ["Np", "Npv", "Npvk"]


def rescore(d: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    rows = []
    for cand, ref in pairs:
        r = evaluate(d, cand, ref)
        rows.append({"cand": cand, "ref": ref, "standalone": r["standalone"]["diff"], "se_sa": r["standalone"]["se"],
                     "blend_open": r["open"]["diff"], "se_bo": r["open"]["se"], "blend_close": r["close"]["diff"],
                     "clv_t": r["clv_change"]["t"], "passes_new": r["passes"]})
    t = pd.DataFrame(rows)
    t["standalone_shrunk"] = shrink_effects(t["standalone"], t["se_sa"])
    t["blend_open_shrunk"] = shrink_effects(t["blend_open"], t["se_bo"])
    return t[["cand", "ref", "standalone", "standalone_shrunk", "blend_open", "blend_open_shrunk", "blend_close", "clv_t", "passes_new"]]


def qb_table(d_post: pd.DataFrame, d_exp: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Scored-season log loss with ex-post starters vs expected starters, per model."""
    rows = {}
    for m in models:
        a, b = (x[x["season"].isin(SCORED)] for x in (d_post, d_exp))
        lp = log_loss(a["home_win"].to_numpy(), a[m].to_numpy()).mean()
        le = log_loss(b["home_win"].to_numpy(), b[m].to_numpy()).mean()
        rows[m] = {"ex-post QB": lp, "qb-expected": le, "change": le - lp}
    return pd.DataFrame(rows).T


def main() -> None:
    market = market_probs(DatasetStore().load("cfb_lines"))
    models = sorted({m for p in PAIRS for m in p})
    d, dropped = attach_market(load_predictions(models, ""), market)
    t = rescore(d, PAIRS)
    d_post, _ = attach_market(load_predictions(QB_MODELS, ""), market)
    d_exp, _ = attach_market(load_predictions(QB_MODELS, "_qbexp"), market)
    common = d_post.index.intersection(d_exp.index)
    q = qb_table(d_post.loc[common], d_exp.loc[common], QB_MODELS)
    from_exp = evaluate(d_exp.loc[common], "Npvk", "Np")
    text = "\n\n".join([
        f"p27 re-score: past candidates under the Phase 27 trading rule (ex-post QB files), scored 2022-24, "
        f"{int(d['season'].isin(SCORED).sum())} games ({dropped} dropped for a missing line); 2025 not read",
        t.round(4).to_string(index=False),
        "QB re-baseline (scored 2022-24 log loss; qb-expected zeroes the starter delta in scored seasons):\n" + q.round(4).to_string(),
        f"qb-expected lineage, Npvk vs Np: blend vs open {from_exp['open']['diff']:+.4f} (P(worse) {from_exp['open']['p_worse']:.2f}), "
        f"standalone {from_exp['standalone']['diff']:+.4f}"])
    (RESULTS / "p27_rescore.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `poetry run pytest tests/sessions/test_cfb_trading_selection.py -q --noconftest`
Expected: 4 passed.

- [ ] **Step 5: Run the QB re-baseline backtests** (six runs, about 30–40 minutes each; at most four in parallel)

```bash
cd /Users/aannestrand/Desktop/gnomepy-research-worktrees/cfb-split
L=.superpowers/sdd/2026-10-16-cfb-review-followups/logs; mkdir -p $L
for m in Np Npv Npvk; do
  AWS_PROFILE=AWSAdministratorAccess-241533121172 poetry run python -m gnomepy_research.sessions.cfb_win_probability.evaluation.backtest 2021 2022 --model $m --qb-expected > $L/p27_${m}_2021_2022.log 2>&1 &
  AWS_PROFILE=AWSAdministratorAccess-241533121172 poetry run python -m gnomepy_research.sessions.cfb_win_probability.evaluation.backtest 2023 2024 --model $m --qb-expected > $L/p27_${m}_2023_2024.log 2>&1 &
done; wait
ls gnomepy_research/sessions/cfb_win_probability/results/backtest/ | grep _qbexp
```

Expected: six `predictions_*_{Np,Npv,Npvk}_qbexp.parquet` files and their reports. Then check that no log contains `Traceback`.

- [ ] **Step 6: Run the re-score**

Run: `AWS_PROFILE=AWSAdministratorAccess-241533121172 poetry run python -m gnomepy_research.sessions.cfb_win_probability.evaluation.rescore_candidates`
Expected: `results/backtest/p27_rescore.txt` containing:
- a 22-row candidate table;
- the QB table, where every `change` should be ≥ 0 (expected starters lose the look-ahead);
- the lineage line.

- [ ] **Step 7: Record the Phase 27 result and commit**

Append a `**Phase 27 result (date, results/backtest/p27_rescore.txt)**` block under the Phase 27 section with:
- the QB table;
- the Npvk_qbexp gap to the open and the close (the log loss of each from the file);
- the five candidates with the most negative `blend_open_shrunk`;
- which old adoptions (Npv, Npvk) and old rejections (NpvkN, Npc, NpvkC, Npvg) flip under the new rule.

Mark it **informational**: adoption goes through Phase 28.

```bash
R=gnomepy_research/sessions/cfb_win_probability/results/backtest
git add gnomepy_research/sessions/cfb_win_probability/evaluation/rescore_candidates.py tests/sessions/test_cfb_trading_selection.py docs/superpowers/plans/2026-10-13-cfb-split-efficiency-and-qb-status.md $R/p27_rescore.txt $R/report_2021_2022_Np_qbexp.txt $R/report_2023_2024_Np_qbexp.txt $R/report_2021_2022_Npv_qbexp.txt $R/report_2023_2024_Npv_qbexp.txt $R/report_2021_2022_Npvk_qbexp.txt $R/report_2023_2024_Npvk_qbexp.txt
git commit -m "Phase 27 result: QB re-baseline and re-score of past candidates under the trading rule

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

# Part B — Model candidates the reviewers converged on (Phase 28)

Each candidate below extends Npvk and is run with `--qb-expected`. The math for each, as pre-registered in Task 9:

| Candidate | Change | Why (reviewers) |
|---|---|---|
| `NpvkU` | The Kalman channel enters per rating draw as N(kf_diff, kf_var), so its uncertainty is integrated the same way the efficiency ratings' is. | `kf_var` is computed but unused, and teams with 1–2 games played lose with the channel (UT Austin, Harvard, Cal) |
| `NpvkF` | FCS opponents get variance 144 (not 36), and bowl layoff drift is 16 (not 1); the raw-margin cap is kept. | FCS residual sd 17.5 vs the assumed 14.3; a month-long layoff (Harvard, UT Austin, Cal). Oxford found the innovation-cap change worse, so it is not included. |
| `NpvkN2` | First-year-coach reset `nc_reg` fades as k/(k+n), with n the team's games played and k = 4 (the scoreboard prior's worth in games, σ²/τ² ≈ 169/42); `nc_trend` is dropped. | NpvkN helped weeks 5–9 and hurt weeks 10+; it was the best candidate on the blend metric (UT Austin, Cal) |
| `NpvkS` | Small outcome layer of six terms: pass_edge, rush_edge, venue, bowl_edge, qb_delta_diff, kf_edge. | 17 collinear terms; Oxford's 3-parameter stack matched Npvk (all five) |
| `NpvkR` | Npvk with the roster-continuity extension (`cont`). | Npc was better in every season and carries the most information beyond the open (Oxford, MIT, Cal) |
| `NpvkAc` | Academy games' implied margin scaled by c = slope(capped margin on our margin, academy games) / slope(same, other games), fitted on earlier seasons. | Calibration slope 0.59 in option games; factor about 0.72 (Oxford) |

### Task 4: `kf_draws` — the scoreboard channel with its uncertainty integrated (NpvkU)

**Files:**
- Modify: `S/models/win_probability.py`, in the `game_draws` signature and the `if kf is not None:` block
- Modify: `S/evaluation/backtest.py` (`OUTCOME_FLAGS` and `MODEL_CONFIGS`)
- Test: `tests/sessions/test_cfb_margin_filter.py` (append)

**Interfaces:**
- Consumes: the `kf` DataFrame with `game_id, kf_diff, kf_var, kf_prior` (`margin_filter.kf_table`).
- Produces:
  - `game_draws(..., kf_draws: bool = False)`
  - config `MODEL_CONFIGS["NpvkU"] = {**MODEL_CONFIGS["Npvk"], "kf_draws": True}`
  - `"kf_draws"` added to `OUTCOME_FLAGS`

- [ ] **Step 1: Write the failing test**

```python
def test_kf_draws_integrate_channel_uncertainty_and_keep_swap_symmetry(tmp_path):
    model_k.TestPhaseTerms.write(tmp_path)
    games = TestResultsTerm.games()
    kf = pd.DataFrame({"game_id": [1, 2, 3], "kf_diff": [5.0, -10.0, 0.0], "kf_var": [0.0, 100.0, 100.0], "kf_prior": 0.0})
    gd = game_draws(games, [SEASON], directory=tmp_path, units=UNITS_SPLIT, kf=kf, kf_draws=True)
    assert gd.terms[-1] == KF_TERM
    x = {g: gd.x[i, :, -1] for i, g in enumerate(gd.games["game_id"])}
    np.testing.assert_allclose(x[1], 0.5)                                   # no channel uncertainty: exact
    assert x[2].std() > 0.1                                                  # 10-point sd -> about 1 unit per draw
    swapped = games.assign(home_id=games.away_id, away_id=games.home_id, home_points=games.away_points, away_points=games.home_points)
    gs = game_draws(swapped, [SEASON], directory=tmp_path, units=UNITS_SPLIT, kf=kf.assign(kf_diff=-kf.kf_diff), kf_draws=True)
    xs = {g: gs.x[i, :, -1] for i, g in enumerate(gs.games["game_id"])}
    np.testing.assert_allclose(xs[2], -x[2])                                 # exactly negated draw for draw
    with pytest.raises(ValueError, match="kf_split"):
        game_draws(games, [SEASON], directory=tmp_path, units=UNITS_SPLIT, kf=kf, kf_draws=True, kf_split=True)
    assert MODEL_CONFIGS["NpvkU"] == {**MODEL_CONFIGS["Npvk"], "kf_draws": True}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `poetry run pytest tests/sessions/test_cfb_margin_filter.py::test_kf_draws_integrate_channel_uncertainty_and_keep_swap_symmetry -q --noconftest`
Expected: FAIL with `TypeError: game_draws() got an unexpected keyword argument 'kf_draws'`.

- [ ] **Step 3: Implement**

In the `game_draws` signature, after `kf_split: bool = False,`, add `kf_draws: bool = False,`. Then replace the whole `if kf is not None:` block with:

```python
    if kf is not None:
        if kf_split and kf_draws:
            raise ValueError("kf_split and kf_draws are separate candidates")
        kk = out[["game_id"]].merge(kf, on="game_id", how="left").fillna(0.0)
        total = kk["kf_diff"].to_numpy() / KF_SCALE
        if kf_draws:
            # Orientation sign: swapping home and away flips eps, so the term is exactly antisymmetric per draw.
            orient = np.where(out["home_id"].to_numpy() < out["away_id"].to_numpy(), 1.0, -1.0)
            eps = rng.standard_normal(x.shape[:2]) * orient[:, None]
            k = (total[:, None] + np.sqrt(kk["kf_var"].to_numpy())[:, None] / KF_SCALE * eps)[..., None]
            x = np.concatenate([x, k], axis=-1)
            terms = terms + [KF_TERM]
        else:
            if kf_split:
                prior = kk["kf_prior"].to_numpy() / KF_SCALE
                early = ((out["season_type"] == "regular") & (out["week"] <= EARLY_LAST_WEEK)).to_numpy(float)
                cols, names = [prior, total - prior, (total - prior) * early], KF_SPLIT_TERMS
            else:
                cols, names = [total], [KF_TERM]
            k = np.stack(cols, axis=-1)[:, None, :]
            x = np.concatenate([x, np.broadcast_to(k, (*x.shape[:2], len(cols)))], axis=-1)
            terms = terms + names
```

In `backtest.py`:
- append `"kf_draws"` to `OUTCOME_FLAGS`;
- under the Phase 26 configs, add:

```python
# Phase 28 on Npvk (reviewer follow-ups): U integrates the scoreboard channel's own uncertainty per draw.
MODEL_CONFIGS["NpvkU"] = {**MODEL_CONFIGS["Npvk"], "kf_draws": True}
```

- [ ] **Step 4: Run the margin-filter tests**

Run: `poetry run pytest tests/sessions/test_cfb_margin_filter.py -q --noconftest`
Expected: 14 passed (the 13 existing tests, unmodified, plus this one).

- [ ] **Step 5: Commit**

```bash
git add gnomepy_research/sessions/cfb_win_probability/models/win_probability.py gnomepy_research/sessions/cfb_win_probability/evaluation/backtest.py tests/sessions/test_cfb_margin_filter.py
git commit -m "NpvkU: scoreboard channel drawn from N(kf_diff, kf_var) per rating draw, antisymmetric under team swap

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 5: FCS variance and bowl drift as filter parameters (NpvkF)

**Files:**
- Modify: `S/models/margin_filter.py` (`run_filter`, `kf_table`)
- Modify: `S/evaluation/backtest.py` (the import, the `kf_table` call and the config)
- Test: `tests/sessions/test_cfb_margin_filter.py` (append)

**Interfaces:**
- Produces:
  - `run_filter(games, priors, fcs_var: float = FCS_VAR, q_post: float = Q)`
  - `kf_table(games, srs, strengths, tiered, market=None, fcs_var: float = FCS_VAR, q_post: float = Q)`
  - config `MODEL_CONFIGS["NpvkF"] = {**MODEL_CONFIGS["Npvk"], "kf_fcs_var": 144.0, "kf_q_post": 16.0}`
- With the defaults, output is identical to today.

- [ ] **Step 1: Write the failing test**

```python
def test_fcs_variance_and_bowl_drift_are_parameters_with_unchanged_defaults():
    g = schedule([(1, 1, 1, 9, 50, 3, True), (2, 2, 1, 2, np.nan, np.nan, False)])
    g.loc[1, ["season_type", "week"]] = ["postseason", 1]
    g.loc[1, "kickoff"] = K + pd.Timedelta(days=120)
    pri = {1: (0.0, 36.0), 2: (0.0, 36.0)}
    base = mf.run_filter(g, pri).set_index("game_id")
    pd.testing.assert_frame_equal(base, mf.run_filter(g, pri, fcs_var=mf.FCS_VAR, q_post=mf.Q).set_index("game_id"))
    np.testing.assert_allclose(base.loc[2, "kf_var"], (36 - 36 ** 2 / (36 + 36 + mf.SIGMA ** 2) + mf.Q) + 36.0)
    out = mf.run_filter(g, pri, fcs_var=144.0, q_post=16.0).set_index("game_id")
    total = 36 + 144 + mf.SIGMA ** 2
    innov = mf.CAP - (0.0 - mf.FCS_MEAN)                                    # neutral site, 47-point win capped at 28
    np.testing.assert_allclose(out.loc[2, "kf_diff"], 36 / total * innov)
    np.testing.assert_allclose(out.loc[2, "kf_var"], (36 - 36 ** 2 / total + 16.0) + 36.0)
    assert out.loc[1, "kf_var"] == 36 + 144
    assert MODEL_CONFIGS["NpvkF"] == {**MODEL_CONFIGS["Npvk"], "kf_fcs_var": 144.0, "kf_q_post": 16.0}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `poetry run pytest tests/sessions/test_cfb_margin_filter.py::test_fcs_variance_and_bowl_drift_are_parameters_with_unchanged_defaults -q --noconftest`
Expected: FAIL with `TypeError: run_filter() got an unexpected keyword argument 'fcs_var'`.

- [ ] **Step 3: Implement**

Replace `run_filter` in `margin_filter.py` with:

```python
def run_filter(games: pd.DataFrame, priors: dict, fcs_var: float = FCS_VAR, q_post: float = Q) -> pd.DataFrame:
    """Pre-game mean and variance of theta_home - theta_away for every game in `games` (one season), updating on each
    completed game's capped margin. Teams without a prior (FCS) sit at FCS_MEAN with fcs_var and are not updated.
    A team's first postseason game adds q_post of drift (the layoff); later games add Q per week elapsed."""
    state = {t: list(mv) for t, mv in priors.items()}
    start = {t: mv[0] for t, mv in priors.items()}
    last, last_post = {}, {}
    rows = []
    g = games.sort_values(["kickoff", "game_id"])
    for r in g.itertuples(index=False):
        post = r.season_type == "postseason"
        step = 1 if post else int(r.week)
        for t in (r.home_id, r.away_id):
            if t in state and t in last:
                if post:
                    state[t][1] += Q if last_post[t] else q_post
                else:
                    state[t][1] += Q * max(step - last[t], 0)
        mh, vh = state.get(r.home_id, (FCS_MEAN, fcs_var))
        ma, va = state.get(r.away_id, (FCS_MEAN, fcs_var))
        rows.append((r.game_id, mh - ma, vh + va, start.get(r.home_id, FCS_MEAN) - start.get(r.away_id, FCS_MEAN)))
        if not (bool(r.completed) and pd.notna(r.home_points) and pd.notna(r.away_points)):
            continue
        venue = 0.0 if r.neutral_site else 1.0
        innov = float(np.clip(r.home_points - r.away_points, -CAP, CAP)) - (mh - ma + HFA * venue)
        total = vh + va + SIGMA ** 2
        for t, v, sign in ((r.home_id, vh, 1.0), (r.away_id, va, -1.0)):
            if t in state:
                state[t][0] += sign * v / total * innov
                state[t][1] -= v ** 2 / total
                last[t], last_post[t] = step, post
    return pd.DataFrame(rows, columns=["game_id", "kf_diff", "kf_var", "kf_prior"])
```

**Why the defaults are unchanged:** the old code added `Q * max(step - last, 1)` with step = 1 in the postseason, which is Q. The new code adds Q there too when `q_post = Q`. Regular weeks are unchanged.

In `kf_table`:
- add `fcs_var: float = FCS_VAR, q_post: float = Q` to the signature after `market`;
- change the call to `out.append(run_filter(games[games["season"] == season], priors, fcs_var, q_post))`;
- append to the docstring: `fcs_var, q_post: see run_filter.`

In `backtest.py`:
- change the import to `from ...margin_filter import FCS_VAR as KF_FCS_VAR, Q as KF_Q, kf_table, preseason_strength, turnover_margins`;
- add `fcs_var=cfg.get("kf_fcs_var", KF_FCS_VAR), q_post=cfg.get("kf_q_post", KF_Q)` to the `kf_table(...)` call;
- add the config:

```python
# F: FCS opponents N(-26, 144) (their residual sd is about 17.5 = sqrt(144 + 169)), and 16 points^2 of drift
# over the bowl layoff instead of 1.
MODEL_CONFIGS["NpvkF"] = {**MODEL_CONFIGS["Npvk"], "kf_fcs_var": 144.0, "kf_q_post": 16.0}
```

- [ ] **Step 4: Run the margin-filter tests**

Run: `poetry run pytest tests/sessions/test_cfb_margin_filter.py -q --noconftest`
Expected: 15 passed. The existing closed-form and cap tests are unchanged and must still pass.

- [ ] **Step 5: Commit**

```bash
git add gnomepy_research/sessions/cfb_win_probability/models/margin_filter.py gnomepy_research/sessions/cfb_win_probability/evaluation/backtest.py tests/sessions/test_cfb_margin_filter.py
git commit -m "NpvkF: FCS opponent variance and bowl-layoff drift as filter parameters (defaults unchanged)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 6: First-year-coach reset that fades with games played (NpvkN2)

**Files:**
- Modify: `S/models/win_probability.py` (new `games_before`, the `coach_fade` parameter and the coach block)
- Modify: `S/evaluation/backtest.py` (`OUTCOME_FLAGS`, config)
- Test: `tests/sessions/test_cfb_margin_filter.py` (append)

**Interfaces:**
- Produces:
  - `games_before(out, games) -> np.ndarray` of shape (G, 2): completed games the home and away side played earlier the same season, against any opponent
  - `game_draws(..., coach_fade: float | None = None)`
  - config `MODEL_CONFIGS["NpvkN2"] = {**MODEL_CONFIGS["Npvk"], "newcoach": True, "coach_fade": 4.0}`
  - `"coach_fade"` added to `OUTCOME_FLAGS`

- [ ] **Step 1: Write the failing test**

```python
def test_coach_reset_fades_with_games_played(tmp_path):
    from gnomepy_research.sessions.cfb_win_probability.models.win_probability import games_before
    games = TestResultsTerm.games()                         # team 1 v 2 (wk 1), 2 v 1 (wk 4), 1 v 2 (wk 5), all completed
    n = games_before(games, games)
    np.testing.assert_array_equal(n, [[0, 0], [1, 1], [2, 2]])
    model_k.TestPhaseTerms.write(tmp_path)
    coach = pd.DataFrame({"season": SEASON, "team_id": [1, 2], "coach_new": [1.0, 0.0], "prior_z": [-1.5, 0.5]})
    gd = game_draws(games, [SEASON], directory=tmp_path, units=UNITS_SPLIT, coach=coach, coach_fade=4.0)
    assert gd.terms[-1] == "nc_reg" and "nc_trend" not in gd.terms
    got = dict(zip(gd.games["game_id"], gd.x[:, 0, -1]))
    np.testing.assert_allclose([got[1], got[2], got[3]], [1.5 * 4 / 4, -1.5 * 4 / 5, 1.5 * 4 / 6])
    assert MODEL_CONFIGS["NpvkN2"] == {**MODEL_CONFIGS["Npvk"], "newcoach": True, "coach_fade": 4.0}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `poetry run pytest tests/sessions/test_cfb_margin_filter.py::test_coach_reset_fades_with_games_played -q --noconftest`
Expected: FAIL with `ImportError: cannot import name 'games_before'`.

- [ ] **Step 3: Implement**

Add `games_before` to `win_probability.py`, directly after `mov_diffs`:

```python
def games_before(out: pd.DataFrame, games: pd.DataFrame) -> np.ndarray:
    """(G, 2) completed games the home and the away side played earlier the same season (any opponent)."""
    g = games[games["season"].isin(out["season"].unique())]
    done = (g["completed"].fillna(False).astype(bool) & g["home_points"].notna() & g["away_points"].notna()).astype(float)
    long = pd.concat([pd.DataFrame({"game_id": g["game_id"], "season": g["season"], "kickoff": g["kickoff"], "team": g[c], "n": done})
                      for c in ("home_id", "away_id")]).sort_values(["season", "team", "kickoff", "game_id"])
    long["before"] = long.groupby(["season", "team"])["n"].cumsum() - long["n"]
    b = long.set_index(["game_id", "team"])["before"]
    pick = lambda col: b.reindex(pd.MultiIndex.from_arrays([out["game_id"], out[col]])).fillna(0.0).to_numpy()
    return np.column_stack([pick("home_id"), pick("away_id")])
```

In the `game_draws` signature, after `coach: pd.DataFrame | None = None`, add `coach_fade: float | None = None`. Then replace the `if coach is not None:` block with:

```python
    if coach is not None:
        c = coach.set_index(["season", "team_id"])
        side = lambda col: c.reindex(pd.MultiIndex.from_arrays([out["season"], out[col]])).fillna(0.0)[["coach_new", "prior_z"]].to_numpy(float)
        h, a = side("home_id"), side("away_id")                 # positional arrays: (G, [coach_new, prior_z])
        if coach_fade is None:
            reg = -(h[:, 0] * h[:, 1] - a[:, 0] * a[:, 1])
            frac = np.where(out["season_type"] == "postseason", 1.0, np.minimum(out["week"].to_numpy(float), SEASON_WEEKS) / SEASON_WEEKS)
            cols, names = [reg, (h[:, 0] - a[:, 0]) * frac], NC_TERMS
        else:
            n = games_before(out, games)
            fh, fa = coach_fade / (coach_fade + n[:, 0]), coach_fade / (coach_fade + n[:, 1])
            cols, names = [-(h[:, 0] * h[:, 1] * fh - a[:, 0] * a[:, 1] * fa)], NC_TERMS[:1]
        cc = np.stack(cols, axis=-1)[:, None, :]
        x = np.concatenate([x, np.broadcast_to(cc, (*x.shape[:2], len(cols)))], axis=-1)
        terms = terms + names
```

In `backtest.py`:
- append `"coach_fade"` to `OUTCOME_FLAGS`;
- add the config:

```python
# N2: the first-year-coach reset fades as k / (k + games played), k = 4 (the scoreboard prior is worth
# sigma^2 / tau^2 = 169 / 42 games); the learning-curve term is dropped (its intervals covered 0 in every season).
MODEL_CONFIGS["NpvkN2"] = {**MODEL_CONFIGS["Npvk"], "newcoach": True, "coach_fade": 4.0}
```

- [ ] **Step 4: Run the tests**

Run: `poetry run pytest tests/sessions/test_cfb_margin_filter.py -q --noconftest`
Expected: 16 passed. `test_new_coach_terms_and_table` is unchanged and still passes.

- [ ] **Step 5: Commit**

```bash
git add gnomepy_research/sessions/cfb_win_probability/models/win_probability.py gnomepy_research/sessions/cfb_win_probability/evaluation/backtest.py tests/sessions/test_cfb_margin_filter.py
git commit -m "NpvkN2: first-year-coach reset fading with games played, k = 4

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 7: Small outcome layer and roster continuity on Npvk (NpvkS, NpvkR)

**Files:**
- Modify: `S/evaluation/backtest.py` (configs)
- Test: `tests/sessions/test_cfb_margin_filter.py` (append)

**Interfaces:**
- Consumes:
  - the existing `keep_terms` flag (in `OUTCOME_FLAGS`; `game_draws` raises on unknown names);
  - the existing `cont` efficiency extension (draws exist for 2020–24, from Npc).
- Produces:
  - `MODEL_CONFIGS["NpvkS"] = {**MODEL_CONFIGS["Npvk"], "keep_terms": SMALL_TERMS}`
  - `MODEL_CONFIGS["NpvkR"] = {**MODEL_CONFIGS["Npvk"], "extensions": ("couple", "fcs", "finish", "prior", "cont")}`
  - `SMALL_TERMS = ["pass_edge", "rush_edge", "venue", "bowl_edge", "qb_delta_diff", "kf_edge"]`

- [ ] **Step 1: Write the failing test**

```python
def test_small_layer_and_continuity_configs(tmp_path):
    from gnomepy_research.sessions.cfb_win_probability.evaluation.backtest import SMALL_TERMS
    assert SMALL_TERMS == ["pass_edge", "rush_edge", "venue", "bowl_edge", "qb_delta_diff", "kf_edge"]
    assert MODEL_CONFIGS["NpvkS"] == {**MODEL_CONFIGS["Npvk"], "keep_terms": SMALL_TERMS}
    assert MODEL_CONFIGS["NpvkR"]["extensions"] == ("couple", "fcs", "finish", "prior", "cont") and MODEL_CONFIGS["NpvkR"]["kf"] == "tier"
    model_k.TestPhaseTerms.write(tmp_path)
    kf = pd.DataFrame({"game_id": [1, 2, 3], "kf_diff": [5.0, -10.0, 0.0], "kf_var": 50.0, "kf_prior": 0.0})
    qb = pd.DataFrame({"game_id": [1, 2, 3], "home_qb_delta": 0.1, "away_qb_delta": 0.0})
    gd = game_draws(TestResultsTerm.games(), [SEASON], directory=tmp_path, units=UNITS_SPLIT, qb=qb, kf=kf, keep_terms=SMALL_TERMS)
    assert gd.terms == SMALL_TERMS and gd.x.shape[-1] == 6
```

- [ ] **Step 2: Run it to verify it fails**

Run: `poetry run pytest tests/sessions/test_cfb_margin_filter.py::test_small_layer_and_continuity_configs -q --noconftest`
Expected: FAIL with `ImportError: cannot import name 'SMALL_TERMS'`.

- [ ] **Step 3: Implement** (in `backtest.py`, after the NpvkN2 config)

```python
# S: the outcome layer cut to six terms (the two channels' strength, venue, bowl, QB); the other eleven are
# collinear versions of net strength or never distinguishable from zero.
SMALL_TERMS = ["pass_edge", "rush_edge", "venue", "bowl_edge", "qb_delta_diff", "kf_edge"]
MODEL_CONFIGS["NpvkS"] = {**MODEL_CONFIGS["Npvk"], "keep_terms": SMALL_TERMS}
# R: roster continuity (the Npc extension) under the scoreboard channel.
MODEL_CONFIGS["NpvkR"] = {**MODEL_CONFIGS["Npvk"], "extensions": ("couple", "fcs", "finish", "prior", "cont")}
```

- [ ] **Step 4: Run the tests**

Run: `poetry run pytest tests/sessions/test_cfb_margin_filter.py -q --noconftest`
Expected: 17 passed.

- [ ] **Step 5: Commit**

```bash
git add gnomepy_research/sessions/cfb_win_probability/evaluation/backtest.py tests/sessions/test_cfb_margin_filter.py
git commit -m "NpvkS (six-term outcome layer) and NpvkR (roster continuity) configs

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 8: Academy margin-scale post-processor (NpvkAc)

**Files:**
- Create: `S/evaluation/academy_scale.py`
- Test: `tests/sessions/test_cfb_academy_scale.py`

**Interfaces:**
- Consumes: Task 1's `FIRST_BLEND_SEASON`; `compare_models.SPREAD_SD` (15.5); `backtest.RESULTS`; `candidate_selection.TAGS`.
- Produces:
  - `ACADEMIES`, `CAP`
  - `is_academy(df) -> np.ndarray[bool]`
  - `margin_slope(y, m) -> float | None`
  - `scale_factor(train) -> float`
  - `apply_scale(df) -> DataFrame` (p_home rescaled; new column `academy_scale`)
  - the CLI `--base Npvk --out NpvkAc --suffix _qbexp`, which writes `predictions_{tag}_{out}{suffix}.parquet` for both tags

**The math:**
- Our implied margin is m = 15.5·Φ⁻¹(p).
- For each scored season s, fit on seasons 2021..s−1:
  - c_acad = Σ clip(y, ±28)·m / Σ m² over academy games;
  - c_other is the same over all other games;
  - c = c_acad / c_other.
- The academy games of s get p′ = Φ(c·Φ⁻¹(p)).
- c = 1 when there is no earlier season or either slope is undefined.

- [ ] **Step 1: Write the failing tests**

```python
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from gnomepy_research.sessions.cfb_win_probability.evaluation import academy_scale as ac


def frame(seed=0, per=400):
    rng = np.random.default_rng(seed)
    rows = []
    for season in (2021, 2022, 2023):
        p = rng.uniform(0.1, 0.9, per)
        acad = np.arange(per) < per // 4
        m = 15.5 * norm.ppf(p)
        y = np.where(acad, 0.7, 1.0) * m + rng.normal(0, 8, per)
        rows.append(pd.DataFrame({"season": season, "p_home": p, "home_team": np.where(acad, "Navy", "Tulsa"),
                                  "away_team": "Rice", "home_points": 30 + y, "away_points": 30.0, "completed": True}))
    return pd.concat(rows, ignore_index=True)


def test_scale_is_fitted_on_earlier_seasons_and_shrinks_academy_games_only():
    d = frame()
    out = ac.apply_scale(d)
    assert (out.loc[out.season == 2021, "academy_scale"] == 1.0).all()            # nothing earlier
    c23 = out.loc[(out.season == 2023) & ac.is_academy(out), "academy_scale"].iloc[0]
    assert c23 == pytest.approx(0.7, abs=0.08)
    other = (out.season == 2023) & ~ac.is_academy(out)
    np.testing.assert_allclose(out.loc[other, "p_home"], d.loc[other, "p_home"])
    acad = (out.season == 2023) & ac.is_academy(out)
    assert (np.abs(out.loc[acad, "p_home"] - 0.5) <= np.abs(d.loc[acad, "p_home"] - 0.5) + 1e-12).all()


def test_degenerate_training_gives_factor_one():
    d = frame()
    d.loc[d.season == 2021, "home_team"] = "Tulsa"                                 # no academy games in 2021
    assert ac.apply_scale(d).loc[lambda t: t.season == 2022, "academy_scale"].eq(1.0).all()
    assert ac.margin_slope(np.array([1.0]), np.array([0.0])) is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `poetry run pytest tests/sessions/test_cfb_academy_scale.py -q --noconftest`
Expected: FAIL with `ModuleNotFoundError: ... academy_scale`.

- [ ] **Step 3: Write the implementation**

```python
"""Academy margin scale (Phase 28, NpvkAc): option academies play fewer, longer possessions, so a given per-play
edge is worth fewer points. Our implied margin m = SPREAD_SD * Phi^-1(p) is scaled by c = (slope of capped margin
on m in academy games) / (the same slope in other games), fitted on earlier seasons only; p' = Phi(c * Phi^-1(p)).

    poetry run python -m gnomepy_research.sessions.cfb_win_probability.evaluation.academy_scale \
        --base Npvk --out NpvkAc --suffix _qbexp
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import norm

from gnomepy_research.sessions.cfb_win_probability.evaluation.backtest import RESULTS
from gnomepy_research.sessions.cfb_win_probability.evaluation.candidate_selection import TAGS
from gnomepy_research.sessions.cfb_win_probability.evaluation.compare_models import SPREAD_SD
from gnomepy_research.sessions.cfb_win_probability.evaluation.market_metrics import FIRST_BLEND_SEASON

ACADEMIES = frozenset({"Army", "Navy", "Air Force"})
CAP = 28.0


def is_academy(df: pd.DataFrame) -> np.ndarray:
    return (df["home_team"].isin(ACADEMIES) | df["away_team"].isin(ACADEMIES)).to_numpy()


def _margin(p: np.ndarray) -> np.ndarray:
    return SPREAD_SD * norm.ppf(np.clip(p, 1e-6, 1 - 1e-6))


def margin_slope(y: np.ndarray, m: np.ndarray) -> float | None:
    """No-intercept least-squares slope of the capped margin on our implied margin; None when undefined."""
    mm = float(m @ m)
    return None if len(m) == 0 or mm == 0.0 else float(np.clip(y, -CAP, CAP) @ m / mm)


def scale_factor(train: pd.DataFrame) -> float:
    done = train["completed"].fillna(False).astype(bool) & train["home_points"].notna() & train["away_points"].notna()
    t = train[done.to_numpy()]
    y, m, acad = (t["home_points"] - t["away_points"]).to_numpy(float), _margin(t["p_home"].to_numpy()), is_academy(t)
    a, o = margin_slope(y[acad], m[acad]), margin_slope(y[~acad], m[~acad])
    return a / o if a is not None and o is not None and o > 0 else 1.0


def apply_scale(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["academy_scale"] = 1.0
    acad = is_academy(out)
    for s in sorted(out["season"].unique()):
        train = out[(out["season"] >= FIRST_BLEND_SEASON) & (out["season"] < s)]
        c = scale_factor(df.loc[train.index]) if len(train) else 1.0
        rows = (out["season"] == s).to_numpy() & acad
        out.loc[rows, "academy_scale"] = c
        out.loc[rows, "p_home"] = norm.cdf(c * norm.ppf(np.clip(df.loc[rows, "p_home"].to_numpy(), 1e-6, 1 - 1e-6)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()
    parts = {t: pd.read_parquet(RESULTS / f"predictions_{t}_{args.base}{args.suffix}.parquet") for t in TAGS}
    both = apply_scale(pd.concat([parts[t].assign(_tag=t) for t in TAGS], ignore_index=True))
    for t in TAGS:
        both[both["_tag"] == t].drop(columns="_tag").to_parquet(RESULTS / f"predictions_{t}_{args.out}{args.suffix}.parquet", index=False)
    print(both.groupby("season")["academy_scale"].max().to_string())


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests**

Run: `poetry run pytest tests/sessions/test_cfb_academy_scale.py -q --noconftest`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add gnomepy_research/sessions/cfb_win_probability/evaluation/academy_scale.py tests/sessions/test_cfb_academy_scale.py
git commit -m "NpvkAc: academy margin-scale post-processor fitted on earlier seasons

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 9: Pre-register, run and select Phase 28

**Files:**
- Modify: the research log (the Phase 28 pre-registration **before** any run, then the result)
- Results: `S/results/backtest/p28_trading.txt/.json`, `report_*_{cand}_qbexp.txt`

- [ ] **Step 1: Run the full suite**

Run: `poetry run pytest tests/sessions/test_cfb_*.py -q --noconftest`
Expected: all pass (394 before this plan, plus the new tests).

- [ ] **Step 2: Pre-register Phase 28** (append after the Phase 27 section)

```markdown
## Phase 28: The reviewers' model changes, under the trading rule (NpvkU, NpvkF, NpvkN2, NpvkS, NpvkR, NpvkAc)

Pre-registered 2026-10-16, before any Phase 28 run. Every model is run with `--qb-expected`, and the baseline is
Npvk_qbexp. The rule is Phase 27's (`trading_selection --suffix _qbexp`).

| Candidate | Change |
|---|---|
| NpvkU | Scoreboard channel drawn per rating draw from N(kf_diff, kf_var) |
| NpvkF | FCS opponents N(−26, 144); 16 points² of drift over the bowl layoff |
| NpvkN2 | First-year-coach reset × 4/(4 + games played); no learning-curve term |
| NpvkS | Six-term outcome layer: pass_edge, rush_edge, venue, bowl_edge, qb_delta_diff, kf_edge |
| NpvkR | Npvk + roster continuity (`cont`) |
| NpvkAc | Academy games' implied margin × c, fitted on earlier seasons |

- **Two or more pass:**
  - Their union is built as one config. NpvkAc is applied last, as a post-processor on the union.
  - The union is run and judged with `--nested Union:BestSingle`. It is adopted only with a further blend Δ ≤ −0.0003.
- **Reported:**
  - each candidate's trading value against the open and the close;
  - the closing-line value of each change;
  - first-year-coach, academy, FCS-game, bowl and weeks 1–4 G5 cuts.
- **Caveat:** each candidate came from the reviewers' 2021–24 analyses, so the backtest flatters them. Live 2026 paper trading (Phase 29) is the confirmation.
```

Commit it before running:

```bash
git add docs/superpowers/plans/2026-10-13-cfb-split-efficiency-and-qb-status.md
git commit -m "Phase 28 pre-registration: reviewer model changes under the trading rule

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 3: Run the backtests** (ten runs; at most four at a time)

```bash
cd /Users/aannestrand/Desktop/gnomepy-research-worktrees/cfb-split
L=.superpowers/sdd/2026-10-16-cfb-review-followups/logs
run() { AWS_PROFILE=AWSAdministratorAccess-241533121172 poetry run python -m gnomepy_research.sessions.cfb_win_probability.evaluation.backtest $2 $3 --model $1 --qb-expected > $L/p28_$1_$2_$3.log 2>&1; }
run NpvkU 2021 2022 & run NpvkU 2023 2024 & run NpvkF 2021 2022 & run NpvkF 2023 2024 & wait
run NpvkN2 2021 2022 & run NpvkN2 2023 2024 & run NpvkS 2021 2022 & run NpvkS 2023 2024 & wait
run NpvkR 2021 2022 & run NpvkR 2023 2024 & wait
grep -l Traceback $L/p28_*.log
AWS_PROFILE=AWSAdministratorAccess-241533121172 poetry run python -m gnomepy_research.sessions.cfb_win_probability.evaluation.academy_scale --base Npvk --out NpvkAc --suffix _qbexp
```

Expected: 12 new `*_qbexp.parquet` files (two per candidate, NpvkAc included), and `grep` prints nothing.

- [ ] **Step 4: Select**

Run: `AWS_PROFILE=AWSAdministratorAccess-241533121172 poetry run python -m gnomepy_research.sessions.cfb_win_probability.evaluation.trading_selection --label p28 --ref Npvk --suffix _qbexp --candidates NpvkU NpvkF NpvkN2 NpvkS NpvkR NpvkAc`
Expected: `results/backtest/p28_trading.txt`, ending in `chosen: <model>`.

- [ ] **Step 5: Run the union if two or more pass**

- Add `MODEL_CONFIGS["NpvkX"]` merging the passing configs' flags (leave out NpvkAc, which is a post-processor). Add a config test asserting the merge.
- Run `NpvkX` (both pairs, `--qb-expected`).
- If NpvkAc passed, also run `academy_scale --base NpvkX --out NpvkXAc --suffix _qbexp`, and use `NpvkXAc` as the union below.
- Rerun selection: `--candidates <passers> NpvkX --nested NpvkX:<best single>`.

- [ ] **Step 6: Record and commit**

Append a `**Phase 28 result**` block to the research log: the selection table, the cuts named in Step 2, and the adopted model.

If a candidate is adopted, update these to name it as live:
- the memory file `/Users/aannestrand/.claude/projects/-Users-aannestrand/memory/cfb-model-lineup.md`;
- the summary line in `MEMORY.md`.

Then commit:

```bash
R=gnomepy_research/sessions/cfb_win_probability/results/backtest
git add docs/superpowers/plans/2026-10-13-cfb-split-efficiency-and-qb-status.md $R/p28_trading.txt $R/p28_trading.json $R/report_2021_2022_NpvkU_qbexp.txt $R/report_2023_2024_NpvkU_qbexp.txt $R/report_2021_2022_NpvkF_qbexp.txt $R/report_2023_2024_NpvkF_qbexp.txt $R/report_2021_2022_NpvkN2_qbexp.txt $R/report_2023_2024_NpvkN2_qbexp.txt $R/report_2021_2022_NpvkS_qbexp.txt $R/report_2023_2024_NpvkS_qbexp.txt $R/report_2021_2022_NpvkR_qbexp.txt $R/report_2023_2024_NpvkR_qbexp.txt
git commit -m "Phase 28 result: reviewer model changes under the trading rule

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

# Part C — Trading layer and paper trading (Phase 29)

### Task 10: Decision layer

**Files:**
- Create: `S/trading/__init__.py` (empty), `S/trading/decision.py`
- Test: `tests/sessions/test_cfb_trading_decision.py`

**Interfaces:**
- Consumes: `win_probability.matchup_tier(df) -> 0/1/2`; `compare_models.logit`.
- Produces:
  - the constants `TAU_CELL`, `KELLY_FRACTION`, `FEE_RATE`, `TEAM_SEASON_CAP`, `BOTTOM_N`, `ACADEMIES`
  - `cells(df) -> np.ndarray[str]` ("P4-P4|wk1-4" and so on)
  - `fit_cell_blend(y, p_mkt, p_model, cell, tau=TAU_CELL) -> {"b0", "cells"}`
  - `blend_belief(p_mkt, p_model, cell, fit) -> np.ndarray`
  - `bottom_teams(win_totals, teams, n=BOTTOM_N) -> set[tuple[int, int]]`
  - `skip_reasons(df, bottom) -> np.ndarray[str]` ("" means trade)
  - `kelly_stakes(p, q, fee_rate=FEE_RATE, fraction=KELLY_FRACTION) -> np.ndarray` (+ home, − away)
  - `cap_team_season(stakes, df, cap=TEAM_SEASON_CAP) -> np.ndarray`
  - `bet_returns(stakes, q, y, fee_rate=FEE_RATE) -> np.ndarray`

**The math:**
- **Belief:** logit p* = logit q + (b0 + d_cell)·(logit m − logit q), with d_cell ~ N(0, 0.2²). This is a MAP fit by Newton's method, on earlier seasons only.
- **Stakes:** a $1 contract costs its price plus 0.07·q(1−q). Full Kelly is (p − c)/(1 − c); a quarter of it is staked.
- **Team-season cap:** total stake involving a team is capped at 5% of bankroll per team-season, because a team's games share one rating error. Both teams in a bet count against the cap.
- **Skips:**
  - G5-vs-G5 games in regular-season weeks 1–4;
  - weeks 5–9 games with a bottom-35 team (by preseason win-total strength; teams without a total count as bottom);
  - the postseason;
  - academy games.

- [ ] **Step 1: Write the failing tests**

```python
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gnomepy_research.sessions.cfb_win_probability.trading import decision as dc


def games(n=6):
    return pd.DataFrame({"season": 2026, "week": [1, 2, 6, 7, 11, 1], "season_type": ["regular"] * 5 + ["postseason"],
                         "home_id": [1, 3, 5, 1, 7, 9], "away_id": [2, 4, 6, 8, 1, 10],
                         "home_team": ["A", "Navy", "C", "A", "E", "F"], "away_team": ["B", "D", "G", "H", "A", "I"],
                         "home_conference": ["SEC", "American Athletic", "Sun Belt", "SEC", "MAC", "SEC"],
                         "away_conference": ["SEC", "American Athletic", "Sun Belt", "MAC", "SEC", "Big Ten"],
                         "kickoff": pd.to_datetime(["2026-09-01", "2026-09-08", "2026-10-06", "2026-10-13", "2026-11-10", "2026-12-20"], utc=True)})


def test_cell_blend_recovers_cell_weights_with_pooling():
    rng = np.random.default_rng(0)
    n = 6000
    cell = np.where(np.arange(n) < n // 2, "P4-P4|wk1-4", "G5-G5|wk1-4")
    q = 1 / (1 + np.exp(-rng.normal(0, 1, n)))
    s = rng.normal(0, 0.8, n)
    lq = np.log(q / (1 - q))
    b_true = np.where(cell == "P4-P4|wk1-4", 0.6, 0.0)
    y = (rng.random(n) < 1 / (1 + np.exp(-(lq + b_true * s)))).astype(float)
    m = 1 / (1 + np.exp(-(lq + s)))
    fit = dc.fit_cell_blend(y, q, m, cell)
    assert fit["cells"]["P4-P4|wk1-4"] == pytest.approx(0.6, abs=0.12) and fit["cells"]["G5-G5|wk1-4"] < 0.2
    p = dc.blend_belief(np.array([0.0, 1.0, 0.5]), np.array([1.0, 0.0, 0.5]), np.array(["X|wk1-4"] * 3), fit)
    assert np.isfinite(p).all() and p[2] == pytest.approx(0.5)            # unseen cell falls back to b0


def test_skip_rules_and_cells():
    g = games()
    bottom = dc.bottom_teams(pd.DataFrame({"season": 2026, "team_id": [6, 1, 2], "wt_strength": [-20.0, 10.0, 5.0]}),
                             pd.DataFrame({"season": 2026, "team_id": [1, 2, 5, 6]}), n=1)
    assert bottom == {(2026, 6), (2026, 5)}                                 # team 5 has no win total: counted as bottom
    r = dc.skip_reasons(g, bottom)
    assert r.tolist() == ["", "academy", "bottom-35 wk5-9", "", "", "postseason"]
    g2 = g.assign(home_team="X", away_team="Y")
    g2.loc[1, ["home_conference", "away_conference"]] = ["Sun Belt", "MAC"]
    assert dc.skip_reasons(g2, set())[1] == "G5-G5 wk1-4"
    assert dc.cells(g)[0] == "P4-P4|wk1-4" and dc.cells(g)[5] == "P4-P4|post"


def test_kelly_stakes_fee_and_sides():
    s = dc.kelly_stakes(np.array([0.60, 0.40, 0.505, 1.0]), np.array([0.50, 0.50, 0.50, 0.999]))
    fee = dc.FEE_RATE * 0.25
    assert s[0] == pytest.approx(dc.KELLY_FRACTION * (0.60 - (0.5 + fee)) / (1 - (0.5 + fee)))
    assert s[1] == pytest.approx(-s[0]) and s[2] == 0.0                    # edge below the fee: no bet
    assert np.isfinite(s).all()


def test_team_season_cap_holds_on_both_teams_and_never_flips_sign():
    g = games()
    stakes = np.array([0.04, 0.0, 0.0, 0.04, -0.04, 0.0])                   # team 1 in games 0 (home), 3 (home), 4 (away)
    out = dc.cap_team_season(stakes, g, cap=0.05)
    assert out[0] == pytest.approx(0.04) and out[3] == pytest.approx(0.01) and out[4] == 0.0
    assert (np.sign(out) * np.sign(stakes) >= 0).all()


def test_bet_returns():
    q = np.array([0.5, 0.5, 0.5])
    r = dc.bet_returns(np.array([0.1, 0.1, -0.1]), q, np.array([1.0, 0.0, 0.0]))
    c = 0.5 + dc.FEE_RATE * 0.25
    np.testing.assert_allclose(r, [0.1 * (1 - c) / c, -0.1, 0.1 * (1 - c) / c])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `poetry run pytest tests/sessions/test_cfb_trading_decision.py -q --noconftest`
Expected: FAIL with `ModuleNotFoundError: ... trading`.

- [ ] **Step 3: Write the implementation** (`S/trading/__init__.py` is an empty file)

```python
"""Trading layer (Phase 29): the market-conditional belief, games to skip, and stake sizing. Paper trading only;
nothing here places orders, and none of it is financial advice.

Belief: logit p* = logit q + (b0 + d_cell) * (logit m - logit q), q the market price and m the model, one weight
per matchup tier x season phase cell, partially pooled toward b0 (d_cell ~ N(0, TAU_CELL^2)) and fitted on earlier
seasons only. Stakes are fractional Kelly on a $1 binary contract after a Kalshi-style fee, capped per team-season
because a team's games share one rating error.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cfb_win_probability.evaluation.compare_models import logit
from gnomepy_research.sessions.cfb_win_probability.models.win_probability import EARLY_LAST_WEEK, MID_LAST_WEEK, matchup_tier

TAU_CELL = 0.2
KELLY_FRACTION = 0.25
FEE_RATE = 0.07
TEAM_SEASON_CAP = 0.05
BOTTOM_N = 35
ACADEMIES = frozenset({"Army", "Navy", "Air Force"})
TIER_NAMES = np.array(["P4-P4", "P4-G5", "G5-G5"])


def _phase(df: pd.DataFrame) -> np.ndarray:
    week = df["week"].to_numpy()
    return np.where(df["season_type"] == "postseason", "post",
                    np.where(week <= EARLY_LAST_WEEK, "wk1-4", np.where(week <= MID_LAST_WEEK, "wk5-9", "wk10+")))


def cells(df: pd.DataFrame) -> np.ndarray:
    return np.array([f"{t}|{p}" for t, p in zip(TIER_NAMES[matchup_tier(df)], _phase(df))])


def fit_cell_blend(y, p_mkt, p_model, cell, tau: float = TAU_CELL, iters: int = 50) -> dict:
    """MAP of b0 (flat prior) and the cell offsets d_cell ~ N(0, tau^2) by Newton's method."""
    y, cell = np.asarray(y, float), np.asarray(cell)
    names = sorted(set(cell))
    u = logit(np.asarray(p_model, float)) - logit(np.asarray(p_mkt, float))
    off = logit(np.asarray(p_mkt, float))
    A = np.column_stack([u] + [u * (cell == c) for c in names])
    pen = np.r_[0.0, np.full(len(names), 1 / tau ** 2)]
    th = np.zeros(A.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(off + A @ th)))
        g = A.T @ (y - p) - pen * th
        H = (A * (p * (1 - p))[:, None]).T @ A + np.diag(pen)
        step = np.linalg.solve(H, g)
        th += step
        if np.abs(step).max() < 1e-10:
            break
    return {"b0": float(th[0]), "cells": {c: float(th[0] + d) for c, d in zip(names, th[1:])}}


def blend_belief(p_mkt, p_model, cell, fit: dict) -> np.ndarray:
    b = np.array([fit["cells"].get(c, fit["b0"]) for c in cell])
    lq = logit(np.asarray(p_mkt, float))
    return 1 / (1 + np.exp(-(lq + b * (logit(np.asarray(p_model, float)) - lq))))


def bottom_teams(win_totals: pd.DataFrame, teams: pd.DataFrame, n: int = BOTTOM_N) -> set[tuple[int, int]]:
    """Per season, the n FBS teams with the lowest preseason win-total strength, plus every team without a total."""
    t = teams[["season", "team_id"]].drop_duplicates().merge(win_totals[["season", "team_id", "wt_strength"]], how="left")
    out = set()
    for season, g in t.groupby("season"):
        out |= {(int(season), int(x)) for x in g.loc[g["wt_strength"].isna(), "team_id"]}
        out |= {(int(season), int(x)) for x in g.dropna(subset=["wt_strength"]).nsmallest(n, "wt_strength")["team_id"]}
    return out


def skip_reasons(df: pd.DataFrame, bottom: set[tuple[int, int]]) -> np.ndarray:
    ph, tier = _phase(df), TIER_NAMES[matchup_tier(df)]
    acad = (df["home_team"].isin(ACADEMIES) | df["away_team"].isin(ACADEMIES)).to_numpy()
    low = np.array([(s, h) in bottom or (s, a) in bottom for s, h, a in zip(df["season"], df["home_id"], df["away_id"])])
    return np.select([ph == "post", acad, (tier == "G5-G5") & (ph == "wk1-4"), (ph == "wk5-9") & low],
                     ["postseason", "academy", "G5-G5 wk1-4", "bottom-35 wk5-9"], "")


def kelly_stakes(p, q, fee_rate: float = FEE_RATE, fraction: float = KELLY_FRACTION) -> np.ndarray:
    """Signed stake as a share of bankroll: + buys the home contract at q, - the away contract at 1 - q. A $1
    contract costs its price plus fee_rate * q * (1 - q); full Kelly at cost c and win probability p is (p - c)/(1 - c)."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    fee = fee_rate * q * (1 - q)
    ch, ca = q + fee, (1 - q) + fee
    fh = np.where(p > ch, (p - ch) / np.maximum(1 - ch, 1e-9), 0.0)
    fa = np.where(1 - p > ca, ((1 - p) - ca) / np.maximum(1 - ca, 1e-9), 0.0)
    return fraction * (fh - fa)


def cap_team_season(stakes, df: pd.DataFrame, cap: float = TEAM_SEASON_CAP) -> np.ndarray:
    """Cut stakes, in kickoff order, so the total stake involving any team in a season stays within `cap`."""
    stakes = np.asarray(stakes, float)
    season, home, away = df["season"].to_numpy(), df["home_id"].to_numpy(), df["away_id"].to_numpy()
    used, out = {}, np.zeros(len(stakes))
    for i in np.argsort(df["kickoff"].to_numpy(), kind="stable"):
        if stakes[i] == 0:
            continue
        kh, ka = (season[i], home[i]), (season[i], away[i])
        room = max(min(cap - used.get(kh, 0.0), cap - used.get(ka, 0.0)), 0.0)
        out[i] = np.sign(stakes[i]) * min(abs(stakes[i]), room)
        used[kh] = used.get(kh, 0.0) + abs(out[i])
        used[ka] = used.get(ka, 0.0) + abs(out[i])
    return out


def bet_returns(stakes, q, y, fee_rate: float = FEE_RATE) -> np.ndarray:
    """Profit per game as a share of bankroll: a winning contract bought at cost c returns stake * (1 - c) / c."""
    stakes, q, y = np.asarray(stakes, float), np.asarray(q, float), np.asarray(y, float)
    fee = fee_rate * q * (1 - q)
    home = stakes > 0
    cost = np.where(home, q + fee, 1 - q + fee)
    win = np.where(home, y == 1, y == 0)
    f = np.abs(stakes)
    return np.where(f == 0, 0.0, np.where(win, f * (1 - cost) / cost, -f))
```

- [ ] **Step 4: Run the tests**

Run: `poetry run pytest tests/sessions/test_cfb_trading_decision.py -q --noconftest`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add gnomepy_research/sessions/cfb_win_probability/trading/__init__.py gnomepy_research/sessions/cfb_win_probability/trading/decision.py tests/sessions/test_cfb_trading_decision.py
git commit -m "Trading layer: pooled market blend by cell, skip rules, fractional Kelly, team-season cap

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 11: 2022–24 trading backtest

**Files:**
- Create: `S/trading/trading_backtest.py`
- Test: `tests/sessions/test_cfb_trading_decision.py` (append)
- Modify: the research log (Phase 29 pre-registration, then the result)

**Interfaces:**
- Consumes: Task 10, plus Task 2's `load_predictions` and `attach_market`.
- Produces:
  - `run_policy(d, model, policy, bottom) -> DataFrame[game_id, season, week, stake, ret]`, with policy ∈ {"raw", "pooled", "cells"}
  - `summarize(bets) -> DataFrame` (rows per season and "all": n bets, staked, profit, ROI, ROI 90% lo/hi by week-block bootstrap, log growth)
  - the CLI `--model Npvk --suffix _qbexp --win-totals PATH`, which writes `RESULTS/p29_trading_backtest.txt`

**The policies**, all priced at the no-vig open (a proxy: the traded Kalshi/Polymarket price differs, which the report states):

| Policy | Belief | Skips |
|---|---|---|
| `raw` | the model itself | none |
| `pooled` | one blend weight (`TAU_CELL` → 1e-3) | yes |
| `cells` | the pre-registered layer | yes |

All three use quarter Kelly and the team-season cap. Blends are fitted on seasons 2021..s−1.

- [ ] **Step 1: Write the failing test** (append)

```python
def test_trading_backtest_policies_on_synthetic_edge():
    from gnomepy_research.sessions.cfb_win_probability.trading.trading_backtest import run_policy, summarize
    from tests.sessions.test_cfb_market_metrics import synthetic
    d = synthetic(n=1500)
    d = d.assign(home_id=np.arange(len(d)) % 300, away_id=300 + np.arange(len(d)) % 300, home_team="A", away_team="B",
                 home_conference="SEC", away_conference="SEC",
                 kickoff=pd.Timestamp("2022-09-01", tz="UTC") + pd.to_timedelta(np.arange(len(d)) % 13 * 7, unit="D"))
    d.index.name = "game_id"
    bets = run_policy(d, "cand", "cells", set())
    s = summarize(bets).set_index("season")
    assert set(s.index) == {2022, 2023, 2024, "all"} and s.loc["all", "roi"] > 0
    assert (bets.season >= 2022).all()
    raw = summarize(run_policy(d, "ref", "raw", set())).set_index("season")
    assert raw.loc["all", "roi"] < s.loc["all", "roi"]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `poetry run pytest tests/sessions/test_cfb_trading_decision.py::test_trading_backtest_policies_on_synthetic_edge -q --noconftest`
Expected: FAIL with `ModuleNotFoundError: ... trading_backtest`.

- [ ] **Step 3: Write the implementation**

```python
"""Phase 29 trading backtest, 2022-24, priced at the no-vig open (a proxy for the traded Kalshi/Polymarket price).

    poetry run python -m gnomepy_research.sessions.cfb_win_probability.trading.trading_backtest \
        --model Npvk --suffix _qbexp --win-totals /path/to/market_preseason_strength.csv

Policies: raw (Kelly on the model itself, no skips), pooled (one blend weight, skips), cells (the pre-registered
layer). All use quarter Kelly and the team-season cap, with blends fitted on seasons 2021..s-1. Paper analysis only.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy.data import DatasetStore
from gnomepy_research.sessions.cfb_win_probability.evaluation.backtest import RESULTS
from gnomepy_research.sessions.cfb_win_probability.evaluation.compare_models import market_probs
from gnomepy_research.sessions.cfb_win_probability.evaluation.market_metrics import FIRST_BLEND_SEASON
from gnomepy_research.sessions.cfb_win_probability.evaluation.trading_selection import SCORED, attach_market, load_predictions
from gnomepy_research.sessions.cfb_win_probability.trading import decision as dc

N_BOOT = 2000


def run_policy(d: pd.DataFrame, model: str, policy: str, bottom: set) -> pd.DataFrame:
    frames = []
    for s in SCORED:
        cur = d[d["season"] == s]
        if policy == "raw":
            belief, skip = cur[model].to_numpy(), np.full(len(cur), "")
        else:
            train = d[(d["season"] >= FIRST_BLEND_SEASON) & (d["season"] < s)]
            tau = dc.TAU_CELL if policy == "cells" else 1e-3
            fit = dc.fit_cell_blend(train["home_win"], train["p_open"], train[model], dc.cells(train), tau=tau)
            belief, skip = dc.blend_belief(cur["p_open"], cur[model], dc.cells(cur), fit), dc.skip_reasons(cur, bottom)
        stake = np.where(skip == "", dc.kelly_stakes(belief, cur["p_open"].to_numpy()), 0.0)
        stake = dc.cap_team_season(stake, cur)
        frames.append(pd.DataFrame({"game_id": cur.index, "season": s, "week": cur["week"].to_numpy(), "stake": stake,
                                    "ret": dc.bet_returns(stake, cur["p_open"].to_numpy(), cur["home_win"].to_numpy())}))
    return pd.concat(frames, ignore_index=True)


def _roi_ci(b: pd.DataFrame, seed: int = 0) -> tuple[float, float]:
    keys = pd.factorize(b["season"] * 100 + b["week"])[0]
    nb = keys.max() + 1
    r, st = np.bincount(keys, weights=b["ret"], minlength=nb), np.bincount(keys, weights=b["stake"].abs(), minlength=nb)
    idx = np.random.default_rng(seed).integers(0, nb, (N_BOOT, nb))
    roi = r[idx].sum(1) / np.maximum(st[idx].sum(1), 1e-12)
    return float(np.quantile(roi, 0.05)), float(np.quantile(roi, 0.95))


def summarize(bets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, b in [*bets.groupby("season"), ("all", bets)]:
        b = b[b["stake"] != 0]
        staked = float(b["stake"].abs().sum())
        lo, hi = _roi_ci(b) if len(b) else (np.nan, np.nan)
        rows.append({"season": key, "bets": len(b), "staked": staked, "profit": float(b["ret"].sum()),
                     "roi": float(b["ret"].sum()) / staked if staked else np.nan, "roi_lo90": lo, "roi_hi90": hi,
                     "log_growth": float(np.log1p(b["ret"]).sum())})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--win-totals", type=Path, required=True)
    args = ap.parse_args()
    d, dropped = attach_market(load_predictions([args.model], args.suffix), market_probs(DatasetStore().load("cfb_lines")))
    teams = pd.concat([d[["season", "home_id"]].set_axis(["season", "team_id"], axis=1), d[["season", "away_id"]].set_axis(["season", "team_id"], axis=1)])
    bottom = dc.bottom_teams(pd.read_csv(args.win_totals), teams)
    parts = [f"p29 trading backtest, {args.model}{args.suffix}, priced at the no-vig open (proxy), quarter Kelly, fee "
             f"{dc.FEE_RATE}*q(1-q), team-season cap {dc.TEAM_SEASON_CAP}; {dropped} games dropped for a missing line; paper analysis only"]
    for policy in ("raw", "pooled", "cells"):
        parts.append(f"policy {policy}:\n" + summarize(run_policy(d, args.model, policy, bottom)).round(4).to_string(index=False))
    text = "\n\n".join(parts)
    (RESULTS / "p29_trading_backtest.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test**

Run: `poetry run pytest tests/sessions/test_cfb_trading_decision.py -q --noconftest`
Expected: 6 passed.

- [ ] **Step 5: Pre-register Phase 29** (append to the research log, then commit), then run

```markdown
## Phase 29: Trading layer and paper trading

Pre-registered before the trading backtest runs.

**The layer** (`trading/decision.py`):
- **Belief:** a blend weight per tier × phase cell, partially pooled (τ = 0.2), fitted on seasons 2021..s−1.
- **Skips:**
  - G5–G5 games in weeks 1–4;
  - weeks 5–9 games with a bottom-35 team by preseason win-total strength (teams without a total count as bottom);
  - the postseason;
  - academy games.
- **Sizing:** quarter Kelly after a fee of 0.07·q(1−q), with a 5% cap per team-season.

**The backtest** (`trading/trading_backtest.py`):
- **Scored:** 2022–24, on the adopted Phase 28 model `_qbexp`, priced at the no-vig open.
- **Proxy price:** the open is not the traded price. Kalshi and Polymarket fills differ, so the result is an upper bound before spread.

**The decision:**
- **Go:** paper trading starts if the `cells` policy has ROI > 0 pooled and a 90% lower bound > −2%.
- **Either way:** live orders are never placed by this code.
- **Paper record:** scored with the confidence sequence on LL(open) − LL(belief) per game (α = 0.10).
```

```bash
git add gnomepy_research/sessions/cfb_win_probability/trading/trading_backtest.py tests/sessions/test_cfb_trading_decision.py docs/superpowers/plans/2026-10-13-cfb-split-efficiency-and-qb-status.md
git commit -m "Phase 29 pre-registration and trading backtest

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
AWS_PROFILE=AWSAdministratorAccess-241533121172 poetry run python -m gnomepy_research.sessions.cfb_win_probability.trading.trading_backtest --model <adopted> --suffix _qbexp --win-totals /private/tmp/claude-501/-Users-aannestrand/555aa52e-cdb5-40ed-bed0-8c3d3128a4f7/scratchpad/market_preseason_strength.csv
```

Here `<adopted>` is the model chosen in Task 9 (`Npvk` if none passed).
Expected: `results/backtest/p29_trading_backtest.txt` with three policy tables.

Record the result and the go/no-go under Phase 29, then commit the txt file and the research log.

### Task 12: Paper-trade logger and scorer

**Files:**
- Create: `S/trading/paper_trade.py`
- Test: `tests/sessions/test_cfb_paper_trade.py`

**Interfaces:**
- Consumes: Tasks 1 (`confidence_sequence`) and 10; `market_probs`.
- Produces:
  - `new_rows(preds, market, fit, bottom, now, logged) -> DataFrame`: one row per upcoming, not-yet-logged game, with columns logged_at, game_id, season, week, home_team, away_team, kickoff, p_model, p_open, p_belief, cell, skip, stake
  - `score(log, games) -> (bets DataFrame, confidence-sequence DataFrame)`
  - the CLI subcommands:
    - `log --predictions PATH --fit-model NAME --suffix _qbexp --win-totals PATH [--out PATH]`
    - `score [--out PATH]`
  - default log path `.superpowers/paper_trades/cfb_2026.csv` (git-ignored)

- [ ] **Step 1: Write the failing tests**

```python
from __future__ import annotations

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cfb_win_probability.trading import paper_trade as pt

NOW = pd.Timestamp("2026-10-20 12:00", tz="UTC")


def preds():
    return pd.DataFrame({"game_id": [1, 2, 3], "season": 2026, "week": [8, 8, 8], "season_type": "regular",
                         "home_id": [1, 3, 5], "away_id": [2, 4, 6], "home_team": ["A", "C", "E"], "away_team": ["B", "D", "Navy"],
                         "home_conference": "SEC", "away_conference": "SEC", "p_home": [0.70, 0.55, 0.60],
                         "kickoff": pd.to_datetime(["2026-10-24", "2026-10-19", "2026-10-24"], utc=True)})


def test_new_rows_logs_only_upcoming_unlogged_games_with_skips():
    market = pd.DataFrame({"game_id": [1, 2, 3], "p_open": [0.60, 0.50, 0.55], "p_close": np.nan})
    fit = {"b0": 0.3, "cells": {}}
    rows = pt.new_rows(preds(), market, fit, set(), NOW, logged=set())
    assert rows["game_id"].tolist() == [1, 3]                                 # game 2 already kicked off
    assert rows.set_index("game_id").loc[3, "skip"] == "academy" and rows.set_index("game_id").loc[3, "stake"] == 0.0
    r1 = rows.set_index("game_id").loc[1]
    assert 0.60 < r1.p_belief < 0.70 and r1.stake > 0 and r1.logged_at == NOW
    assert pt.new_rows(preds(), market, fit, set(), NOW, logged={1, 3}).empty


def test_score_settles_bets_and_tracks_the_confidence_sequence():
    log = pd.DataFrame({"game_id": [1, 3], "p_open": [0.60, 0.55], "p_belief": [0.65, 0.55], "stake": [0.02, 0.0],
                        "skip": ["", "academy"]})
    games = pd.DataFrame({"game_id": [1, 3], "completed": True, "home_points": [30, 10], "away_points": [20, 14]})
    bets, cs = pt.score(log, games)
    assert bets.set_index("game_id").loc[1, "ret"] > 0 and bets.set_index("game_id").loc[3, "ret"] == 0.0
    assert len(cs) == 1 and cs.iloc[0]["lo"] < cs.iloc[0]["mean"] < cs.iloc[0]["hi"]   # only traded games enter
```

- [ ] **Step 2: Run them to verify they fail**

Run: `poetry run pytest tests/sessions/test_cfb_paper_trade.py -q --noconftest`
Expected: FAIL with `ModuleNotFoundError: ... paper_trade`.

- [ ] **Step 3: Write the implementation**

```python
"""Paper trading (Phase 29): log the trading layer's decisions before kickoff, and score them afterwards.
Never places orders; analysis, not financial advice. The log is git-ignored (per-game market data stays private).

    poetry run python -m gnomepy_research.sessions.cfb_win_probability.trading.paper_trade log \
        --predictions PATH --fit-model Npvk --suffix _qbexp --win-totals PATH
    poetry run python -m gnomepy_research.sessions.cfb_win_probability.trading.paper_trade score
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy.data import DatasetStore
from gnomepy_research.sessions.cfb_win_probability.evaluation.candidate_selection import log_loss
from gnomepy_research.sessions.cfb_win_probability.evaluation.compare_models import market_probs
from gnomepy_research.sessions.cfb_win_probability.evaluation.market_metrics import FIRST_BLEND_SEASON, confidence_sequence
from gnomepy_research.sessions.cfb_win_probability.evaluation.trading_selection import SCORED, attach_market, load_predictions
from gnomepy_research.sessions.cfb_win_probability.trading import decision as dc

DEFAULT_LOG = Path(__file__).resolve().parents[4] / ".superpowers" / "paper_trades" / "cfb_2026.csv"
COLUMNS = ["logged_at", "game_id", "season", "week", "home_team", "away_team", "kickoff", "p_model", "p_open", "p_belief",
           "cell", "skip", "stake"]


def new_rows(preds: pd.DataFrame, market: pd.DataFrame, fit: dict, bottom: set, now: pd.Timestamp, logged: set) -> pd.DataFrame:
    g = preds[(preds["kickoff"] > now) & ~preds["game_id"].isin(logged)].merge(market[["game_id", "p_open"]], on="game_id")
    g = g[g["p_open"].notna()].reset_index(drop=True)
    if g.empty:
        return pd.DataFrame(columns=COLUMNS)
    cell = dc.cells(g)
    belief = dc.blend_belief(g["p_open"], g["p_home"], cell, fit)
    skip = dc.skip_reasons(g, bottom)
    stake = dc.cap_team_season(np.where(skip == "", dc.kelly_stakes(belief, g["p_open"].to_numpy()), 0.0), g)
    return pd.DataFrame({"logged_at": now, "game_id": g["game_id"], "season": g["season"], "week": g["week"],
                         "home_team": g["home_team"], "away_team": g["away_team"], "kickoff": g["kickoff"],
                         "p_model": g["p_home"], "p_open": g["p_open"], "p_belief": belief, "cell": cell, "skip": skip,
                         "stake": stake})[COLUMNS]


def score(log: pd.DataFrame, games: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    done = games[games["completed"].fillna(False).astype(bool) & games["home_points"].notna()]
    b = log.merge(done[["game_id", "home_points", "away_points"]], on="game_id")
    y = (b["home_points"] > b["away_points"]).to_numpy(float)
    b = b.assign(home_win=y, ret=dc.bet_returns(b["stake"].to_numpy(), b["p_open"].to_numpy(), y))
    traded = b[b["skip"].fillna("") == ""]
    gain = log_loss(traded["home_win"].to_numpy(), traded["p_open"].to_numpy()) - log_loss(traded["home_win"].to_numpy(), traded["p_belief"].to_numpy())
    return b, confidence_sequence(gain)


def _fit(model: str, suffix: str) -> dict:
    d, _ = attach_market(load_predictions([model], suffix), market_probs(DatasetStore().load("cfb_lines")))
    d = d[d["season"].between(FIRST_BLEND_SEASON, max(SCORED))]
    return dc.fit_cell_blend(d["home_win"], d["p_open"], d[model], dc.cells(d))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    lg = sub.add_parser("log")
    lg.add_argument("--predictions", type=Path, required=True)
    lg.add_argument("--fit-model", required=True)
    lg.add_argument("--suffix", default="")
    lg.add_argument("--win-totals", type=Path, required=True)
    lg.add_argument("--out", type=Path, default=DEFAULT_LOG)
    sc = sub.add_parser("score")
    sc.add_argument("--out", type=Path, default=DEFAULT_LOG)
    args = ap.parse_args()
    ds = DatasetStore()
    if args.cmd == "log":
        prior = pd.read_csv(args.out) if args.out.exists() else pd.DataFrame(columns=COLUMNS)
        preds = pd.read_parquet(args.predictions)
        teams = pd.concat([preds[["season", "home_id"]].set_axis(["season", "team_id"], axis=1),
                           preds[["season", "away_id"]].set_axis(["season", "team_id"], axis=1)])
        rows = new_rows(preds, market_probs(ds.load("cfb_lines")), _fit(args.fit_model, args.suffix),
                        dc.bottom_teams(pd.read_csv(args.win_totals), teams), pd.Timestamp.now(tz="UTC"), set(prior["game_id"]))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        pd.concat([prior, rows], ignore_index=True).to_csv(args.out, index=False)
        print(f"logged {len(rows)} games ({int((rows['stake'] != 0).sum())} paper bets) to {args.out}")
    else:
        bets, cs = score(pd.read_csv(args.out), ds.load("cfb_games"))
        traded = bets[bets["stake"] != 0]
        print(f"settled {len(bets)} logged games, {len(traded)} paper bets; profit {traded['ret'].sum():+.4f} bankroll, "
              f"ROI {traded['ret'].sum() / max(traded['stake'].abs().sum(), 1e-12):+.1%}")
        if len(cs):
            last = cs.iloc[-1]
            print(f"LL(open) - LL(belief) per traded game: {last['mean']:+.4f}, 90% confidence sequence [{last['lo']:+.4f}, {last['hi']:+.4f}] after {int(last['n'])} games")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests and the full suite**

Run: `poetry run pytest tests/sessions/test_cfb_*.py -q --noconftest`
Expected: all pass.

- [ ] **Step 5: Confirm the source of the weekly predictions file**

`paper_trade log` needs a parquet of the week's upcoming games, with the columns of a backtest predictions file (`game_id, season, week, season_type, kickoff, home/away id, team and conference, p_home`).

1. Run `AWS_PROFILE=AWSAdministratorAccess-241533121172 poetry run python -m gnomepy_research.sessions.cfb_win_probability.evaluation.backtest 2026 --model <adopted>`.
   - **No** `--qb-expected` here: live QB deltas come from `qb_status.csv` rows, which is the live information state.
2. Check whether `predictions_2026_<adopted>.parquet` contains the upcoming week's games (`home_win` NaN).
3. If it scores completed games only, add a `--upcoming` flag to the scratchpad `predict_game_m.py` path that writes the same columns for every game of the coming week. Record which route works in the research log.

- [ ] **Step 6: Document the weekly paper-trade step and commit**

Append to `## Weekly operating procedure` in the research log:

> 3. Paper trading (Phase 29):
>    - Before the slate, run `paper_trade log` with that week's predictions file, the adopted model, `--suffix _qbexp` and the win-total CSV.
>    - After the slate, run `paper_trade score`.
>    - The log is `.superpowers/paper_trades/cfb_2026.csv` (git-ignored). Never place orders from it.

```bash
git add gnomepy_research/sessions/cfb_win_probability/trading/paper_trade.py tests/sessions/test_cfb_paper_trade.py docs/superpowers/plans/2026-10-13-cfb-split-efficiency-and-qb-status.md
git commit -m "Paper-trade logger and scorer with an anytime-valid confidence sequence

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

# Part D — Follow-on plans (separate plans, each needing a decision or a large compute or pipeline commitment)

These items touch production `models/`, require a rebuild of the efficiency snapshots (hours of compute per window), or need gnome-pipelines work. Each gets its own plan once Parts A–C report. Order is by reviewer consensus and expected value.

| # | Plan | What | Reviewers | Needs |
|---|---|---|---|---|
| D1 | **Efficiency priors** (`models/cfb/efficiency`) | Sign-constrained "less is known" flags (HalfNormal γ ≥ 0; today `prior_missing` fits −0.18 and narrows move-up priors). Lag-2/3 program history (SRS from `cfb_coaches`) as G5-interacted prior features: G5 R² 0.32 → 0.37 out of sample, market-free. Production-weighted portal arrivals. Optionally win totals as a prior feature (four of five: if anywhere, here). Mixing over hyperparameter draws. | MIT, UT Austin, Oxford, Harvard | Snapshot and hyperparameter rebuilds for 2020–26; a release tag for gnome-pipelines (your OK) |
| D2 | **Event-driven dynamics** (Phase 18) | τ²(team, week) = τ₀² + τ_QB²·[starter changed] + τ_coach²·[firing or interim] + τ_post²·[bowl], instead of uniform drift. σ_game prior centred on the measured 0.06. Selection by the one-step predictive score. A timestamped coach-firing table, replacing "interim hidden until February 1". | UT Austin, Harvard, Cal | Snapshot rebuild; a small hand-built firing table |
| D3 | **Joint state-space model** | One latent strength per team, observed through per-unit PPA, points per drive and the capped margin, with learned loadings. Replaces the bolted-on Kalman channel and fixes the prior/drift balance: today the prior keeps about 40% of the weight at week 13. | UT Austin, Harvard | Weeks of work; after D1 and D2 |
| D4 | **Backfill 2014–19** | CFBD games, lines, play-by-play and recruiting back to 2014, so the outcome layer has ≥ 6 training seasons everywhere. Enables a discovery split (2014–20) and a confirmation split (2021–24), which makes P(worse) a valid error rate again. | MIT, Cal, UT Austin, Harvard | gnome-pipelines ingestion and CFBD API budget (your OK) |
| D5 | **QB value and availability** | A hierarchical QB prior (recruiting, class, career dropbacks, previous level) with replacement level calibrated on in-season starter changes. The conference availability-report logger (Phase 21). About 70% of what the market learns between open and close is in QB-change games. | Harvard, Cal, MIT | gnome-pipelines scraper (your OK) |
| D6 | **Closing line as an auxiliary training label** | Train the outcome layer on wins plus λ·(closing margin), never as an input. | MIT | Your decision on independence |
| D7 | **Bowls** | Refresh posteriors per CFP round; portal entries before the game as an opt-out proxy; the January 2–16 portal window from 2026. | UT Austin, Oxford, MIT | After Phase 29 shows whether bowls stay skipped |
| D8 | **Real traded prices** | Capture Kalshi and Polymarket prices at decision time, so the trading backtest and paper trades use executable prices rather than the Bovada open. | Cal, Harvard | A price logger (your OK) |
