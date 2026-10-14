# CFB Split Efficiency + QB Status Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and compare four pregame models against each other and the sportsbook opening and closing lines:
- **A (current):** aggregate offense/defense ratings.
- **B (split):** pass and rush ratings, so matchups like strong run offense vs weak run defense move the win probability.
- **C (split + form):** B plus early-down success, luck regression, field position / special teams, and pace.
- **D (C + QB):** C plus a pregame QB-status adjustment, so a starter change is priced before kickoff.

**Architecture:**
- **Efficiency model:** the dynamic Bayesian efficiency model gains a `units` dimension. `("all",)` reproduces today's model exactly. `("pass", "rush")` gives each team four latent random walks (O_pass, O_rush, D_pass, D_rush), observed through per-game pass and rush PPA.
- **Outcome (win-probability) layer:** turns those ratings into two matchup edges weighted by each team's point-in-time rush share.
- **Form terms (C):** computed per team at each weekly cutoff from play-by-play and drive data, and added to the outcome layer.
- **QB (D):** a per-game term, `qb_delta_diff` = the announced starter's shrunk dropback PPA minus the QB baseline the ratings were learned from.
- **Comparison:** one script scores every model's backtest predictions against each other and against the opening and closing lines.

**Phases:**
1. **Phase 1 (Tasks 1–6):** split model, then backtest A and B.
2. **Phase 2 (Tasks 7–9):** form features, then backtest C.
3. **Phase 3 (Task 10):** compare A/B/C vs open and close, then **checkpoint with the user**.
4. **Phase 4 (Tasks 11–13):** QB, then backtest D and add it to the comparison.
5. **Task 14:** publish whichever model wins. **User approval required.**

**Tech Stack:** Python 3.13, PyMC + nutpie, numpy/pandas/scipy, pytest, Poetry. Data via `gnomepy.data.DatasetStore` (read-only here; writers live in `~/Desktop/gnome-pipelines`).

**Spec:** No separate spec doc; requirements come from the 2026-10-13 conversation:
1. Split the efficiency observation into pass and rush so matchups (good rush O vs bad rush D) are captured.
2. Then add the extra features from the user's advanced-stats sheet: early-down success, 3rd/4th-down and drive-finishing luck regression, field position / special teams, and pace.
3. Compare all three models (A, B, C) to each other and to the sportsbook opening and closing lines, by log loss.
4. Finally add QB injuries / starter changes pregame, with a manual status input, and add that model (D) to the comparison.

## Global Constraints

- `gnomepy_research/models/` is production code (gnome-pipelines imports it, pinned by git tag). Nothing in `models/` may import from `sessions/`. Changes there must be tested.
- The default behaviour of every existing function must be unchanged: `units=("all",)` must reproduce current `ModelData`, hyperparameter JSONs, posterior draws, and outcome-model terms. Existing tests must pass unmodified except where a task explicitly edits them.
- Point-in-time: anything feeding a prediction at a weekly cutoff may only use data public strictly before that cutoff (`SourceData.as_of(cutoff)` / `Inputs.visible(cutoff)`). The one documented exception is the historical QB starter backfill (Task 11).
- 2025 is the untouched holdout. Never run `backtest 2025 --holdout` while tuning. Compare on 2023–2024 only.
- Imports at the top of the file, never inside functions. No comments unless the WHY is non-obvious.
- Do not push, tag, or release without the user's explicit OK (Task 14).
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Team with zero rush (or pass) plays in a game** (all-pass comeback, kneel-downs only) → that unit's observation is dropped for that game, not fed `plays=0` (division by zero in `sigma`). Test in Task 1.
2. **`units=("all",)` regression** → identical arrays/names to today, so the old hyperparameter JSONs (scalar `alpha`) still load and fit. Tests in Tasks 1 and 2.
3. **Team-swap symmetry** at a neutral site must still give `p + p_swapped == 1` with the new edge and QB terms. Test in Tasks 5, 9 and 13.
4. **Backup QB with no dropbacks** → priced at replacement level (mean PPA of QBs in their first 150 dropbacks), not league mean, and not NaN. Test in Task 11.
5. **Manual QB status row whose team name or passer can't be matched** → raise a clear error naming the row, never silently ignore an injury. Test in Task 12.
6. **Week 1 / team with no games yet** → form features fall back to the league mean (z-score 0), never NaN. Test in Task 7.
7. **Games missing a sportsbook line** → dropped from the market comparison and the count reported, never filled with 0.5. Test in Task 10.

---

## File map

| File | Change |
|---|---|
| `gnomepy_research/models/cfb/efficiency/data.py` | `units`, per-unit feature blocks, alpha groups, per-unit observations |
| `gnomepy_research/models/cfb/efficiency/dynamic_efficiency.py` | `alpha` per unit, `alpha_group` coords, `Hyperparameters.units` |
| `gnomepy_research/models/cfb/efficiency/gaussian_solver.py` | use alpha groups instead of seasons |
| `gnomepy_research/models/cfb/efficiency/inference.py` | `CutoffPosterior.units`, per-unit alpha, `units` threaded through fits |
| `gnomepy_research/models/cfb/efficiency/posterior_snapshots.py` | split naming, `__units` and `__rush_share` in draws, `--units` CLI |
| `gnomepy_research/models/cfb/features/rush_share.py` (new) | point-in-time shrunk rush share per team |
| `gnomepy_research/models/cfb/features/team_form.py` (new) | per team-game form stats from plays/drives; point-in-time shrunk team form per cutoff |
| `gnomepy_research/sessions/cfb_win_probability/evaluation/compare_models.py` (new) | A/B/C/D vs each other and vs open/close lines |
| `gnomepy_research/models/cfb/features/qb_status.py` (new) | replacement-level QB quality, baselines, starters, deltas, manual overrides |
| `gnomepy_research/sessions/cfb_win_probability/models/win_probability.py` | term sets, edge terms, QB term, priors by term |
| `gnomepy_research/sessions/cfb_win_probability/evaluation/backtest.py` | `--model {A,B,C,D}` flag |
| `gnomepy_research/sessions/cfb_win_probability/data/qb_status.csv` (new) | manual starter announcements |
| `tests/sessions/test_cfb_win_probability_models.py` | new tests for split model, solver, outcome terms |
| `tests/sessions/test_cfb_qb_status.py` (new) | QB status tests |
| `tests/sessions/test_cfb_team_form.py` (new) | form feature tests |
| `~/Desktop/gnome-pipelines/gnome_pipelines/cfb/posterior.py`, `datasets/cfb.py` | publish the winning model's posteriors (Task 14) |

Test command for everything below: `poetry run pytest tests/sessions -v` (run `poetry install` once first; the venv isn't installed on this machine yet).

---

### Task 1: Per-unit model data

**Files:**
- Modify: `gnomepy_research/models/cfb/efficiency/data.py`
- Test: `tests/sessions/test_cfb_win_probability_models.py`

**Interfaces:**
- Produces:
  - `UNITS_ALL = ("all",)`, `UNITS_SPLIT = ("pass", "rush")`
  - `build_model_data(preseason, efficiency, games, seasons, units: tuple[str, ...] = UNITS_ALL) -> ModelData`
  - `ModelData` new fields: `units: tuple[str, ...]`, `alpha_keys: pd.DataFrame` (columns `season`, `unit`), `obs_alpha: np.ndarray` (N,), `obs_unit: np.ndarray` (N,) of unit index. `ts_keys` gains a `unit` column. The ts index is unit-major: all team-seasons (FBS then FCS) for unit 0, then unit 1.
  - With `UNITS_ALL`, `off_names`/`def_names` are unchanged; with split they are `"pass:<feat>"…, "rush:<feat>"…, "pass:is_fcs", "rush:is_fcs"` (block-diagonal design so each unit has its own betas).

- [ ] **Step 1: Write the failing tests.** Add to `synthetic()` in the test file per-unit columns so split data exists (pass ~ +0.05 over rush), then add tests:

```python
# in synthetic(), replace the eff.append for FBS games with:
                p_pass = 0.15 + true_o[off - 1] - true_d[de - 1] + 0.015 * venue + rng.normal(0, 1.6 / np.sqrt(30))
                p_rush = 0.05 + true_o[off - 1] - true_d[de - 1] + 0.015 * venue + rng.normal(0, 1.2 / np.sqrt(35))
                y = (30 * p_pass + 35 * p_rush) / 65
                eff.append({"game_id": gid, "season": SEASON, "week": w, "season_type": "regular", "team_id": off,
                            "opponent_id": de, "team_classification": "fbs", "opponent_classification": "fbs",
                            "venue": venue, "plays": 65, "off_ppa": y, "pass_plays": 30, "rush_plays": 35,
                            "off_pass_ppa": p_pass, "off_rush_ppa": p_rush, "usable": True,
                            "available_at": kickoff + pd.Timedelta(days=1)})
# and give the FCS row: "pass_plays": 30, "rush_plays": 35, "off_pass_ppa": 0.7, "off_rush_ppa": 0.5
```

```python
from gnomepy_research.models.cfb.efficiency.data import UNITS_SPLIT

class TestSplitModelData:
    def test_all_units_matches_default(self):
        games, eff, pre, _, _ = synthetic()
        a = build_model_data(pre, eff, games, [SEASON])
        b = build_model_data(pre, eff, games, [SEASON], units=("all",))
        np.testing.assert_array_equal(a.x_off, b.x_off)
        assert a.off_names == b.off_names
        assert list(a.alpha_keys["unit"]) == ["all"]
        np.testing.assert_array_equal(a.obs_alpha, np.zeros(len(a.obs_y), dtype=int))

    def test_split_doubles_states_and_observations(self):
        games, eff, pre, _, _ = synthetic()
        d = build_model_data(pre, eff, games, [SEASON], units=UNITS_SPLIT)
        assert len(d.ts_keys) == 2 * (N_TEAMS + 1)
        assert list(d.ts_keys["unit"].unique()) == ["pass", "rush"]
        assert len(d.obs_y) == 2 * len(eff)
        assert set(d.obs_unit) == {0, 1}
        # pass rows only reference pass states
        assert (d.ts_keys["unit"].to_numpy()[d.obs_off[d.obs_unit == 0]] == "pass").all()
        assert d.x_off.shape[1] == 2 * (len(OFF_FEATURES) + 2)  # +prior_missing +is_fcs per unit
        assert d.off_names[0].startswith("pass:") and d.off_names[-1] == "rush:is_fcs"

    def test_zero_play_unit_observation_is_dropped(self):
        games, eff, pre, _, _ = synthetic()
        eff.loc[0, "rush_plays"] = 0
        eff.loc[0, "off_rush_ppa"] = np.nan
        d = build_model_data(pre, eff, games, [SEASON], units=UNITS_SPLIT)
        assert len(d.obs_y) == 2 * len(eff) - 1
        assert (d.obs_plays > 0).all() and np.isfinite(d.obs_y).all()
```

- [ ] **Step 2: Run, expect FAIL** (`ImportError: UNITS_SPLIT`).

Run: `poetry run pytest tests/sessions/test_cfb_win_probability_models.py -v -k "Split or ModelData"`

- [ ] **Step 3: Implement in `data.py`.**

```python
UNITS_ALL = ("all",)
UNITS_SPLIT = ("pass", "rush")
UNIT_COLUMNS = {"all": ("plays", "off_ppa"), "pass": ("pass_plays", "off_pass_ppa"),
                "rush": ("rush_plays", "off_rush_ppa")}
```

Add fields to `ModelData` (after `seasons`): `units: tuple[str, ...]`, `alpha_keys: pd.DataFrame`, `obs_alpha: np.ndarray`, `obs_unit: np.ndarray`. Keep `obs_season`.

Add a helper and rewrite the tail of `build_model_data` (everything from `fcs = ...` down):

```python
def _block(x: np.ndarray, u: int, n_units: int) -> np.ndarray:
    out = np.zeros((x.shape[0], x.shape[1] * n_units))
    out[:, u * x.shape[1]:(u + 1) * x.shape[1]] = x
    return out


def _unit_names(names: list[str], units: tuple[str, ...]) -> list[str]:
    return names if units == UNITS_ALL else [f"{u}:{n}" for u in units for n in names]
```

```python
def build_model_data(preseason, efficiency, games, seasons, units: tuple[str, ...] = UNITS_ALL) -> ModelData:
    # ... unchanged through `n_fbs = len(pre)`; base_keys = pd.concat([pre[["season","team_id"]], fcs])
    base_x_off, base_x_def = with_fcs(xo), with_fcs(xd)
    base_z = np.vstack([unc[UNCERTAINTY_FLAGS].to_numpy(float), np.ones((len(seasons), len(UNCERTAINTY_FLAGS)))])
    n_u = len(units)
    ts_keys = pd.concat([base_keys.assign(unit=u) for u in units], ignore_index=True)
    x_off = np.vstack([_block(base_x_off, i, n_u) for i in range(n_u)])
    x_def = np.vstack([_block(base_x_def, i, n_u) for i in range(n_u)])
    z_unc = np.vstack([base_z] * n_u)
    alpha_keys = pd.DataFrame([(s, u) for u in units for s in seasons], columns=["season", "unit"])
    a_idx = {(s, u): i for i, (s, u) in enumerate(zip(alpha_keys["season"], alpha_keys["unit"]))}

    periods = season_periods(games, seasons)
    e = efficiency[efficiency["season"].isin(seasons) & efficiency["usable"]]
    e = e.merge(periods[[*WEEK_KEYS, "period"]], on=WEEK_KEYS)
    idx = {(s, t, u): i for i, (s, t, u) in enumerate(zip(ts_keys["season"], ts_keys["team_id"], ts_keys["unit"]))}
    off_team = np.where(e["team_classification"] == "fbs", e["team_id"], FCS_TEAM)
    def_team = np.where(e["opponent_classification"] == "fbs", e["opponent_id"], FCS_TEAM)

    parts = []
    for ui, u in enumerate(units):
        plays_col, y_col = UNIT_COLUMNS[u]
        keep = (e[plays_col].to_numpy(float) > 0) & np.isfinite(e[y_col].to_numpy(float))
        eu = e[keep]
        fcs_row = np.array([idx[(s, FCS_TEAM, u)] for s in eu["season"]], dtype=int)
        # An FBS team missing from the preseason table is pooled with FCS rather than silently dropped.
        o = np.array([idx.get((s, t, u), -1) for s, t in zip(eu["season"], off_team[keep])], dtype=int)
        d = np.array([idx.get((s, t, u), -1) for s, t in zip(eu["season"], def_team[keep])], dtype=int)
        parts.append(pd.DataFrame({
            "off": np.where(o < 0, fcs_row, o), "def": np.where(d < 0, fcs_row, d),
            "period": eu["period"].to_numpy(int), "venue": eu["venue"].to_numpy(float),
            "plays": eu[plays_col].to_numpy(float), "y": eu[y_col].to_numpy(float), "game": eu["game_id"].to_numpy(),
            "season": eu["season"].map({s: i for i, s in enumerate(seasons)}).to_numpy(int),
            "alpha": [a_idx[(s, u)] for s in eu["season"]], "unit": ui,
        }))
    obs = pd.concat(parts, ignore_index=True)

    return ModelData(
        ts_keys=ts_keys,
        n_periods=int(periods.groupby("season")["period"].max().max()) + 1,
        x_off=x_off, x_def=x_def, z_unc=z_unc,
        off_names=_unit_names([*off_names, "is_fcs"], units), def_names=_unit_names([*def_names, "is_fcs"], units),
        obs_off=obs["off"].to_numpy(int), obs_def=obs["def"].to_numpy(int), obs_period=obs["period"].to_numpy(int),
        obs_venue=obs["venue"].to_numpy(float), obs_plays=obs["plays"].to_numpy(float), obs_y=obs["y"].to_numpy(float),
        obs_game=obs["game"].to_numpy(), obs_season=obs["season"].to_numpy(int), seasons=list(seasons), periods=periods,
        units=tuple(units), alpha_keys=alpha_keys, obs_alpha=obs["alpha"].to_numpy(int), obs_unit=obs["unit"].to_numpy(int),
    )
```

(Keep `_standardize`, the `unc` frame and `with_fcs` exactly as they are; just assign their results to `base_*` names.)

- [ ] **Step 4: Run, expect PASS** — the new tests and every pre-existing `TestModelData` test.

Run: `poetry run pytest tests/sessions/test_cfb_win_probability_models.py -v -k "Split or ModelData"`

- [ ] **Step 5: Commit**

```bash
git add gnomepy_research/models/cfb/efficiency/data.py tests/sessions/test_cfb_win_probability_models.py
git commit -m "feat(cfb): per-unit (pass/rush) efficiency model data"
```

---

### Task 2: Per-unit league baseline in the PyMC model

**Files:**
- Modify: `gnomepy_research/models/cfb/efficiency/dynamic_efficiency.py`
- Modify: `tests/sessions/test_cfb_win_probability_models.py` (`hypers()` helper)

**Interfaces:**
- Consumes: `ModelData.units`, `alpha_keys`, `obs_alpha` (Task 1).
- Produces:
  - `Hyperparameters.alpha: np.ndarray` shape `(len(units),)`; new trailing field `units: list[str] = field(default_factory=lambda: ["all"])`. `from_dict` accepts the old scalar `alpha` and missing `units`.
  - PyMC vars: `alpha` dims `unit`; `z_alpha` and deterministic `alpha_season` dims `alpha_group` (one per row of `alpha_keys`).

- [ ] **Step 1: Write the failing tests.** Change `hypers(data)` to `alpha=np.full(len(data.units), 0.1)` and add `units=list(data.units)`. Add:

```python
def test_old_scalar_alpha_json_still_loads():
    d = json.loads((HYPER_DIR / "normal_2019_2024.json").read_text())["hyperparameters"]
    h = Hyperparameters.from_dict(d)
    assert h.alpha.shape == (1,) and h.units == ["all"]


def test_split_recovers_unit_baselines_and_ratings():
    games, eff, pre, true_o, true_d = synthetic()
    d = build_model_data(pre, eff, games, [SEASON], units=UNITS_SPLIT)
    h = hypers(d)
    h.alpha = np.array([0.15, 0.05])
    idata = sample(build_model(d, hypers=h), draws=300, tune=300, chains=2)
    post = idata.posterior.to_dataset()
    o = post["O"].mean(("chain", "draw")).values[:, -1]
    n = N_TEAMS + 1
    assert np.corrcoef(o[:N_TEAMS], true_o)[0, 1] > 0.7          # pass O
    assert np.corrcoef(o[n:n + N_TEAMS], true_o)[0, 1] > 0.7     # rush O
    a = post["alpha_season"].mean(("chain", "draw")).values
    assert a[0] > a[1]                                           # pass baseline above rush
```

(imports: `json`, `HYPER_DIR` from `inference`, `UNITS_SPLIT` from `data`.)

- [ ] **Step 2: Run, expect FAIL.** `poetry run pytest tests/sessions/test_cfb_win_probability_models.py -v`

- [ ] **Step 3: Implement.**

In `Hyperparameters`: type `alpha: np.ndarray`, append `units: list[str] = field(default_factory=lambda: ["all"])` (import `field`). In `from_dict`:

```python
        d["alpha"] = np.atleast_1d(np.asarray(d["alpha"], dtype=float))
        d.setdefault("units", ["all"])
```

In `build_model`: coords replace `"season": data.seasons` with `"alpha_group": np.arange(len(data.alpha_keys)), "unit": list(data.units)`. Then:

```python
        if hypers is None:
            alpha = pm.Normal("alpha", 0.1, 0.1, dims="unit")
            ...
        else:
            alpha = pt.as_tensor_variable(np.asarray(h.alpha, dtype=float))
            ...
        group_unit = data.alpha_keys["unit"].map({u: i for i, u in enumerate(data.units)}).to_numpy(int)
        z_alpha = pm.Normal("z_alpha", 0.0, 1.0, dims="alpha_group")
        alpha_season = pm.Deterministic("alpha_season", alpha[group_unit] + sigma_alpha * z_alpha, dims="alpha_group")
        ...
        mu = (alpha_season[data.obs_alpha] + off[data.obs_off, data.obs_period] - de[data.obs_def, data.obs_period]
              + hfa * data.obs_venue)
```

In `hyperparameters_from`: `alpha=np.atleast_1d(m("alpha"))`, and pass `units=list(data.units)`.

- [ ] **Step 4: Run, expect PASS** (including the existing `test_recovers_rating_signs_and_order`).

- [ ] **Step 5: Commit** `feat(cfb): per-unit league baseline in efficiency model`.

---

### Task 3: Exact Gaussian solver over alpha groups

**Files:**
- Modify: `gnomepy_research/models/cfb/efficiency/gaussian_solver.py`
- Modify: `tests/sessions/test_cfb_win_probability_models.py`

**Interfaces:**
- Consumes: `ModelData.alpha_keys`, `obs_alpha`; `Hyperparameters.alpha` array.
- Produces: `solve(data, h)` unchanged signature; layout key `"alpha"` now spans `len(data.alpha_keys)`.

- [ ] **Step 1: Failing test.** Update `layout_size_matches` to `len(d.alpha_keys) + 2 * n_ts + 2 * n_ts * (d.n_periods - 1)` and add:

```python
def test_exact_solver_matches_mcmc_split():
    games, eff, pre, _, _ = synthetic(seed=1)
    d = build_model_data(pre, eff, games, [SEASON], units=UNITS_SPLIT)
    h = hypers(d)
    h.likelihood = "normal"
    h.alpha = np.array([0.15, 0.05])
    mean, _, lay = solve(d, h)
    ts = np.arange(len(d.ts_keys))
    c_off, B_off = _state_map(d, h, lay, ts, np.full(len(ts), d.n_periods - 1), offense=True)
    exact = c_off + B_off @ mean
    idata = sample(build_model(d, hypers=h, likelihood="normal"), draws=1000, tune=500, chains=2)
    mcmc = idata.posterior.to_dataset()["O"].mean(("chain", "draw")).values[:, -1]
    assert layout_size_matches(lay, d)
    np.testing.assert_allclose(exact, mcmc, atol=0.01)
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement.** In `_layout`: `lay = {"alpha": 0, "z0_off": len(data.alpha_keys)}`. In `solve`:

```python
    A[np.arange(len(data.obs_y)), lay["alpha"] + data.obs_alpha] = h.sigma_alpha
    group_unit = data.alpha_keys["unit"].map({u: i for i, u in enumerate(data.units)}).to_numpy(int)
    c = np.asarray(h.alpha)[group_unit][data.obs_alpha] + c_off - c_def + h.hfa * data.obs_venue
```

In `fit_cutoff_exact`, alpha draws become per group: `alpha = (np.asarray(hypers.alpha)[group_unit][:, None] + hypers.sigma_alpha * theta[lay["alpha"]:lay["alpha"] + len(data.alpha_keys)]).T` (shape `(S, G)`), and pass `units` (see Task 4). Also pass `units=units` into `build_model_data` via a new `units` parameter on `fit_cutoff_exact(inputs, season, cutoff, hypers, n_draws=2000, seed=0)` that reads `tuple(hypers.units)`.

- [ ] **Step 4: Run, expect PASS** (both solver tests).

- [ ] **Step 5: Commit** `feat(cfb): exact Gaussian solver supports per-unit baselines`.

---

### Task 4: Cutoff posteriors and snapshots carry units and rush share

**Files:**
- Create: `gnomepy_research/models/cfb/features/rush_share.py`
- Modify: `gnomepy_research/models/cfb/efficiency/inference.py`, `posterior_snapshots.py`, `gaussian_solver.py`
- Test: `tests/sessions/test_cfb_win_probability_models.py`

**Interfaces:**
- Produces:
  - `rush_share(efficiency: pd.DataFrame, season: int, shrink_plays: float = 150.0) -> pd.Series` indexed by `team_id`: `(rush_plays + k * league) / (rush_plays + pass_plays + k)` over the given (already point-in-time) rows of that season; league = that season's pooled share in the rows, 0.5 if no rows. Teams with no rows are absent (callers fill with league via `.reindex(...).fillna(league)`; expose `league_rush_share(efficiency, season) -> float`).
  - `CutoffPosterior` new field `units: np.ndarray` (T,) of unit labels aligned with `team_ids`; `alpha` becomes `(S, U)`; `summary()` adds a `unit` column and `alpha_mean` per unit.
  - `fit_hyperparameters(inputs, seasons, likelihood, units=UNITS_ALL, ...)`, `fit_cutoff(...)` reads `hypers.units`.
  - `hyper_name(likelihood, season, units=UNITS_ALL)` → `"normal_2019_2024"` or `"normal_split_2019_2024"`.
  - Draws file `cfb_team_week_draws_{season}.npz` (all) or `cfb_team_week_draws_split_{season}.npz` (split); new keys `{key}__units` and `{key}__rush_share` (aligned with `__team_ids`).
  - `MODEL_VERSION = "dyn-eff-1"` for all, `"dyn-eff-split-1"` for split.
  - CLI: `python -m ...posterior_snapshots 2020 2024 --likelihood normal --units split`.

- [ ] **Step 1: Failing tests.**

```python
from gnomepy_research.models.cfb.features.rush_share import league_rush_share, rush_share

def test_rush_share_is_shrunk_toward_league():
    e = pd.DataFrame({"season": 2024, "team_id": [1, 1, 2], "rush_plays": [50, 50, 10], "pass_plays": [10, 10, 50]})
    s = rush_share(e, 2024, shrink_plays=150)
    lg = league_rush_share(e, 2024)
    assert lg == pytest.approx(110 / 180)
    assert lg < s[1] < 100 / 120
    assert 10 / 60 < s[2] < lg

def test_rush_share_with_no_rows_is_empty_and_league_half():
    e = pd.DataFrame(columns=["season", "team_id", "rush_plays", "pass_plays"])
    assert rush_share(e, 2024).empty and league_rush_share(e, 2024) == 0.5

def test_split_cutoff_posterior_shapes():
    games, eff, pre, _, _ = synthetic(seed=2)
    inputs = Inputs(preseason=pre, efficiency=eff, games=games, as_of=pd.Timestamp("2025-01-01", tz="UTC"))
    d = build_model_data(pre, eff, games, [SEASON], units=UNITS_SPLIT)
    h = hypers(d); h.likelihood = "normal"; h.alpha = np.array([0.15, 0.05])
    cutoff = d.periods["cutoff"].iloc[-1]
    post, _ = fit_cutoff_exact(inputs, SEASON, cutoff, h, n_draws=200)
    assert post.off.shape == (200, 2 * (N_TEAMS + 1))
    assert list(pd.unique(post.units)) == ["pass", "rush"]
    assert post.alpha.shape == (200, 2)
    s = post.summary()
    assert set(s["unit"]) == {"pass", "rush"} and len(s) == 2 * (N_TEAMS + 1)
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement.**

`rush_share.py`:

```python
"""Point-in-time run/pass mix per team, shrunk toward the league so a team's first game doesn't swing it."""
from __future__ import annotations

import pandas as pd

RUSH_SHARE_SHRINK_PLAYS = 150.0


def league_rush_share(efficiency: pd.DataFrame, season: int) -> float:
    e = efficiency[efficiency["season"] == season]
    total = float(e["rush_plays"].sum() + e["pass_plays"].sum())
    return float(e["rush_plays"].sum()) / total if total > 0 else 0.5


def rush_share(efficiency: pd.DataFrame, season: int, shrink_plays: float = RUSH_SHARE_SHRINK_PLAYS) -> pd.Series:
    e = efficiency[efficiency["season"] == season]
    lg = league_rush_share(efficiency, season)
    g = e.groupby("team_id")[["rush_plays", "pass_plays"]].sum()
    return ((g["rush_plays"] + shrink_plays * lg) / (g["rush_plays"] + g["pass_plays"] + shrink_plays)).rename("rush_share")
```

`inference.py`: add `units: np.ndarray` to `CutoffPosterior` (after `team_ids`). In `fit_cutoff`: build data with `units=tuple(hypers.units)`; `alpha = post["alpha_season"].values.T` (S, G); `units=data.ts_keys["unit"].to_numpy()`. `summary()`:

```python
        units = list(pd.unique(self.units))
        return pd.DataFrame({
            "season": self.season, "cutoff": self.cutoff, "team_id": self.team_ids, "unit": self.units,
            "offense_mean": ..., (unchanged)
            "alpha_mean": [float(self.alpha[:, units.index(u)].mean()) for u in self.units],
        })
```

`predictive_logpdf`: replace `post.alpha[:, None]` with `post.alpha[:, obs["alpha_idx"].to_numpy()]`, and in `week_observations` set `alpha_idx=0` (the all-units path; split scoring is not needed for this plan). `fit_hyperparameters` gets `units` and passes it to `build_model_data`.

`posterior_snapshots.py`:

```python
SPLIT_SUFFIX = "_split"

def _suffix(units: tuple[str, ...]) -> str:
    return "" if units == UNITS_ALL else SPLIT_SUFFIX

def model_version(units: tuple[str, ...]) -> str:
    return "dyn-eff-1" if units == UNITS_ALL else "dyn-eff-split-1"

def hyper_name(likelihood: str, season: int, units: tuple[str, ...] = UNITS_ALL) -> str:
    return f"{likelihood}{_suffix(units)}_{FIRST_TRAIN_SEASON}_{season - 1}"

def draws_name(season: int, units: tuple[str, ...] = UNITS_ALL) -> str:
    return f"cfb_team_week_draws{_suffix(units)}_{season}"
```

Keep `MODEL_VERSION = model_version(UNITS_ALL)` exported for the pipeline. Thread `units` through `ensure_hyperparameters`, `snapshot_season`, `snapshot_seasons`. In `snapshot_season`, after computing `seen`:

```python
        shares = rush_share(seen, season)
        lg = league_rush_share(seen, season)
        draws[f"{key}__units"] = post.units.astype("U8")
        draws[f"{key}__rush_share"] = shares.reindex(post.team_ids).fillna(lg).to_numpy(np.float32)
        draws[f"{key}__alpha"] = post.alpha[keep].astype(np.float32)
```

File path uses `draws_name(season, units)`; table rows get `model_version=model_version(units)`. CLI: `ap.add_argument("--units", choices=["all", "split"], default="all")`, mapped to `UNITS_ALL`/`UNITS_SPLIT`.

- [ ] **Step 4: Run, expect PASS.** `poetry run pytest tests/sessions -v`

- [ ] **Step 5: Commit** `feat(cfb): split posteriors, rush share, and split draws artifacts`.

---

### Task 5: Outcome model with matchup edges

**Files:**
- Modify: `gnomepy_research/sessions/cfb_win_probability/models/win_probability.py`
- Test: `tests/sessions/test_cfb_win_probability_models.py`

**Interfaces:**
- Consumes: draws keys `__off`, `__def`, `__team_ids`, `__units`, `__rush_share` (Task 4); `draws_name` from `posterior_snapshots`.
- Produces:
  - `TERMS_ALL = ["off_diff", "def_diff", "venue", "bowl_net_diff"]` (alias `TERMS = TERMS_ALL` kept), `TERMS_SPLIT = ["pass_edge", "rush_edge", "venue", "bowl_edge"]`.
  - `PRIORS: dict[str, tuple[float, float]]` = `{"off_diff": (8, 5), "def_diff": (8, 5), "venue": (0.3, 0.5), "bowl_net_diff": (0, 5), "pass_edge": (8, 5), "rush_edge": (8, 5), "bowl_edge": (0, 5), "qb_delta_diff": (4, 4)}`.
  - `GameDraws.terms: list[str]`.
  - `edge_terms(op, dp, orr, dr, share, hi, ai) -> tuple[np.ndarray, np.ndarray]`, pass and rush edges, each (G, S):
    - `rush_edge = s_h*(O_r[h]-D_r[a]) - s_a*(O_r[a]-D_r[h])`
    - `pass_edge = (1-s_h)*(O_p[h]-D_p[a]) - (1-s_a)*(O_p[a]-D_p[h])`
    - `s` is the point-in-time rush share. This weighting is what makes a run-heavy team facing a weak run defense count. Without it, the split terms sum back to the aggregate model.
  - `game_draws(games, seasons, n_draws=None, seed=0, directory=DRAWS_DIR, units=UNITS_ALL) -> GameDraws`.
  - `fit(gd, ...)` uses `PRIORS[t]` for `t in gd.terms`, dims `term` = `gd.terms`.

- [ ] **Step 1: Failing tests.**

```python
from gnomepy_research.sessions.cfb_win_probability.models.win_probability import edge_terms

class TestEdgeTerms:
    @staticmethod
    def draws(seed=0):
        rng = np.random.default_rng(seed)
        return [rng.normal(0, 0.1, (40, 6)) for _ in range(4)], rng.uniform(0.3, 0.7, 6)

    def test_swap_negates_edges(self):
        (op, dp, orr, dr), share = self.draws()
        hi, ai = np.array([0, 2, 4]), np.array([1, 3, 5])
        p1, r1 = edge_terms(op, dp, orr, dr, share, hi, ai)
        p2, r2 = edge_terms(op, dp, orr, dr, share, ai, hi)
        np.testing.assert_allclose(p1, -p2); np.testing.assert_allclose(r1, -r2)

    def test_run_heavy_team_gains_more_from_weak_run_defense(self):
        z = np.zeros((1, 2))
        orr = np.array([[0.1, 0.0]]); dr = np.array([[0.0, -0.1]])   # home good rush O, away bad rush D
        _, light = edge_terms(z, z, orr, dr, np.array([0.3, 0.5]), np.array([0]), np.array([1]))
        _, heavy = edge_terms(z, z, orr, dr, np.array([0.7, 0.5]), np.array([0]), np.array([1]))
        assert heavy[0, 0] > light[0, 0] > 0
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement.**

```python
def edge_terms(op, dp, orr, dr, share, hi, ai):
    sh, sa = share[hi][:, None], share[ai][:, None]
    rush = sh * (orr[:, hi] - dr[:, ai]).T - sa * (orr[:, ai] - dr[:, hi]).T
    pas = (1 - sh) * (op[:, hi] - dp[:, ai]).T - (1 - sa) * (op[:, ai] - dp[:, hi]).T
    return pas, rush
```

In `load_season_draws(season, directory=DRAWS_DIR, units=UNITS_ALL)` use `directory / f"{draws_name(season, units)}.npz"`. In `game_draws`, after computing `hi`/`ai`, branch on `units`:

```python
            if units == UNITS_ALL:
                od = (off[pick][:, hi] - off[pick][:, ai]).T
                dd = (de[pick][:, hi] - de[pick][:, ai]).T
                main = [od, dd]
            else:
                u = draws[f"{key}__units"]
                share_all = draws[f"{key}__rush_share"]
                p_cols, r_cols = np.flatnonzero(u == "pass"), np.flatnonzero(u == "rush")
                team_ids = ids[p_cols]
                col = {t: i for i, t in enumerate(team_ids)}
                fcs = col[FCS_TEAM]
                hi = np.array([col.get(t, fcs) for t in gg["home_id"]])
                ai = np.array([col.get(t, fcs) for t in gg["away_id"]])
                pe, re = edge_terms(off[pick][:, p_cols], de[pick][:, p_cols], off[pick][:, r_cols],
                                    de[pick][:, r_cols], share_all[p_cols], hi, ai)
                main = [pe, re]
            venue = ...  # unchanged
            xs.append(np.stack([*main, venue, post * (main[0] + main[1])], axis=-1))
```

Return `GameDraws(games=out, x=..., terms=TERMS_ALL if units == UNITS_ALL else TERMS_SPLIT)`. In `fit`:

```python
    terms = gd.terms
    with pm.Model(coords={"term": terms}):
        g = pm.Normal("gamma", mu=[PRIORS[t][0] for t in terms], sigma=[PRIORS[t][1] for t in terms], dims="term")
```

Import `UNITS_ALL` from `models.cfb.efficiency.data` and `draws_name` from `posterior_snapshots` (both in `models/`, fine to import from a session).

- [ ] **Step 4: Run, expect PASS** including the existing `TestOutcomeModel` tests.

- [ ] **Step 5: Commit** `feat(cfb): pass/rush matchup edges in the win-probability layer`.

---

### Task 6: Backtest models A and B

**Files:**
- Modify: `gnomepy_research/sessions/cfb_win_probability/evaluation/backtest.py`
- Output: `results/backtest/report_2023_2024_A.txt`, `report_2023_2024_B.txt`, `predictions_2023_2024_{A,B}.parquet`, new `hyperparameters/normal_split_2019_*.json`

**Interfaces:**
- Consumes: `game_draws(..., units=...)` (Task 5), snapshot CLI (Task 4).
- Produces:
  - `backtest` CLI flag `--model {A,B,C,D}` (default `A`).
  - `MODELS = {"A": dict(units=UNITS_ALL, form=False, qb=False), "B": dict(units=UNITS_SPLIT, form=False, qb=False), "C": dict(units=UNITS_SPLIT, form=True, qb=False), "D": dict(units=UNITS_SPLIT, form=True, qb=True)}`. Phase 2/4 tasks wire `form`/`qb`; until then `C`/`D` raise `NotImplementedError`.
  - Output files are suffixed `_{model}`.
  - **Predictions parquet contract (consumed by Task 10):** one row per game, with columns `game_id, season, week, season_type, home_id, away_id, home_win, p_home`, plus the existing baseline columns.

**All models use the Gaussian likelihood (`--likelihood normal`)** so the comparison is apples to apples and the exact solver keeps fits fast. If model A's Gaussian log loss differs from the existing report (0.5498), note both numbers in the report.

- [ ] **Step 1: Add the flag** and thread `MODELS[args.model]` into `run_season(...)`, both `game_draws(...)` calls, and the output names. Confirm the predictions parquet has the contract columns above. Add any that are missing from `test_gd.games`.

- [ ] **Step 2: Build posteriors for both unit settings.** Each run writes the hyperparameter JSONs for the 2019..S-1 training windows and the draws for 2020–2024.

```bash
poetry run python -m gnomepy_research.models.cfb.efficiency.posterior_snapshots 2020 2024 --likelihood normal --units all
poetry run python -m gnomepy_research.models.cfb.efficiency.posterior_snapshots 2020 2024 --likelihood normal --units split
```

Expected: per-week lines `rhat 1.000 div 0`; files `results/posterior_draws/cfb_team_week_draws{,_split}_20{20..24}.npz`.

- [ ] **Step 3: Backtest.**

```bash
poetry run python -m gnomepy_research.sessions.cfb_win_probability.evaluation.backtest 2023 2024 --model A
poetry run python -m gnomepy_research.sessions.cfb_win_probability.evaluation.backtest 2023 2024 --model B
```

Expected: both reports are written. Sanity check: B's calibration slope is in [0.8, 1.2] and its log loss is within 0.02 of A's. A bigger gap in either direction means a bug, not a finding.

- [ ] **Step 4: Commit** the flag, the hyperparameter JSONs and the reports (draws and parquets are gitignored). No accept/reject here; Task 10 judges.

```bash
git add gnomepy_research/sessions/cfb_win_probability/evaluation/backtest.py gnomepy_research/models/cfb/hyperparameters/normal_split_*.json gnomepy_research/sessions/cfb_win_probability/results/backtest/report_2023_2024_A.txt gnomepy_research/sessions/cfb_win_probability/results/backtest/report_2023_2024_B.txt
git commit -m "eval(cfb): backtest aggregate (A) and pass/rush split (B) models"
```

---

## Phase 2: Form features (model C)

### Task 7: Team-game form stats and point-in-time team form

**Files:**
- Create: `gnomepy_research/models/cfb/features/team_form.py`
- Test: `tests/sessions/test_cfb_team_form.py`

**Interfaces:**
- Consumes: `cfb_plays` columns `game_id, season, offense, defense, period, offense_score, defense_score, down, distance, yards_gained, play_type, ppa, play_seq`. `cfb_drives` columns `game_id, season, offense_id, defense_id, start_yards_to_goal, off_points, outcome`. `cfb_games` `game_id, kickoff, home_id, away_id, home_team, away_team`. `team_name_ids(games)` from `features/qb.py`.
- Produces:
  - `FORM_FEATURES = ["early_success", "late_down_luck", "finishing_luck", "field_position", "pace"]`
  - `is_success(down, distance, yards_gained) -> np.ndarray[bool]`: standard definition. 1st down gains ≥ 50% of distance, 2nd ≥ 70%, 3rd/4th ≥ 100%.
  - `is_garbage(period, offense_score, defense_score) -> np.ndarray[bool]`: Connelly thresholds, the same as the efficiency observation. Score margin > 38 in Q2, > 28 in Q3, > 22 in Q4.
  - `team_game_form(plays, drives, games) -> pd.DataFrame`, one row per (game_id, team_id) with `season, available_at` (kickoff + 1 day) and:
    - `early_success`: success rate on downs 1–2, non-garbage scrimmage plays (`ppa` not null).
    - `late_down_luck`: 3rd/4th-down success rate minus the rate expected from that team-game's early-down success. Expected = the league linear fit `late ~ a + b * early`, fit on rows passed in, so callers fit on visible data only.
    - `finishing_luck`: points per **quality drive** (a drive that ends at or inside the opponent's 40, `end_yards_to_goal ≤ 40`, or scores) minus the league mean points per quality drive.
    - `field_position`: mean own `start_yards_to_goal` of the opponent's drives minus the team's own (higher = better). This captures special teams and turnovers, which our PPA ignores.
    - `pace`: non-garbage offensive plays per game.
  - `team_form(form_rows: pd.DataFrame, season: int, shrink_games: float = 3.0) -> pd.DataFrame` (index `team_id`, columns `FORM_FEATURES`):
    - Season-to-date means over the (already point-in-time) rows.
    - Each mean is shrunk toward that season's league mean with weight `n / (n + shrink_games)`.
    - Then each column is z-scored by the league sd of the shrunk values.
    - Teams with no rows get 0. With no rows at all, it returns an empty frame and callers fill 0.

  `late_down_luck` and `finishing_luck` are deliberately signed so **positive = running hot**. The outcome model should learn a *negative* coefficient for them, and that sign is the check that this is regression to the mean.

- [ ] **Step 1: Failing tests.**

```python
import numpy as np
import pandas as pd
import pytest

from gnomepy_research.models.cfb.features.team_form import FORM_FEATURES, is_garbage, is_success, team_form


def test_success_definition():
    down = np.array([1, 1, 2, 2, 3, 4])
    dist = np.array([10, 10, 10, 10, 5, 1])
    gain = np.array([5, 4, 7, 6, 5, 0])
    assert is_success(down, dist, gain).tolist() == [True, False, True, False, True, False]


def test_garbage_time_thresholds():
    assert is_garbage(np.array([2, 2, 3, 4, 1]), np.array([45, 44, 35, 30, 50]), np.array([6, 7, 6, 7, 0])).tolist() == [
        True, False, True, True, False]


def rows(team_vals: dict[int, list[float]], col: str) -> pd.DataFrame:
    out = []
    for t, vals in team_vals.items():
        for v in vals:
            out.append({"team_id": t, "season": 2024, **{f: 0.0 for f in FORM_FEATURES}, col: v})
    return pd.DataFrame(out)


def test_team_with_more_games_moves_further_from_league():
    f = team_form(rows({1: [0.6], 2: [0.6] * 6, 3: [0.4] * 6}, "early_success"), 2024)
    assert f.loc[2, "early_success"] > f.loc[1, "early_success"] > 0


def test_no_rows_is_empty():
    assert team_form(rows({}, "pace").reindex(columns=["team_id", "season", *FORM_FEATURES]), 2024).empty


def test_constant_column_gives_zero_not_nan():
    f = team_form(rows({1: [70.0], 2: [70.0]}, "pace"), 2024)
    assert (f["pace"] == 0).all()
```

Also a `team_game_form` test on a hand-built 1-game, 2-team plays/drives frame: assert `field_position`, `finishing_luck` and `pace` match hand-computed values, and that a Q4 play at a 30-point margin is excluded from `pace`.

- [ ] **Step 2: Run, expect FAIL.** `poetry run pytest tests/sessions/test_cfb_team_form.py -v`

- [ ] **Step 3: Implement.**

```python
"""Per team-game form stats beyond PPA: early-down success, luck on late downs and drive finishing,
field position (special teams + turnovers, which PPA ignores) and pace. Luck terms are signed so
positive means running hot; they should regress."""
from __future__ import annotations

import numpy as np
import pandas as pd

from gnomepy_research.models.cfb.features.qb import team_name_ids

FORM_FEATURES = ["early_success", "late_down_luck", "finishing_luck", "field_position", "pace"]
QUALITY_DRIVE_YTG = 40
GARBAGE_MARGIN = {2: 38, 3: 28, 4: 22}
SUCCESS_SHARE = {1: 0.5, 2: 0.7, 3: 1.0, 4: 1.0}
FORM_SHRINK_GAMES = 3.0


def is_success(down, distance, yards_gained) -> np.ndarray:
    need = np.vectorize(SUCCESS_SHARE.get)(down, 1.0) * distance
    return yards_gained >= need


def is_garbage(period, offense_score, defense_score) -> np.ndarray:
    limit = np.vectorize(GARBAGE_MARGIN.get)(period, np.inf).astype(float)
    return np.abs(offense_score - defense_score) > limit


def team_game_form(plays: pd.DataFrame, drives: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    ids = team_name_ids(games)
    p = plays[plays["ppa"].notna() & plays["down"].between(1, 4)].copy()
    p = p[~is_garbage(p["period"].to_numpy(), p["offense_score"].to_numpy(), p["defense_score"].to_numpy())]
    p["team_id"] = p["offense"].map(ids)
    p["success"] = is_success(p["down"].to_numpy(), p["distance"].to_numpy(), p["yards_gained"].to_numpy())
    early = p[p["down"] <= 2].groupby(["game_id", "team_id"])["success"].mean().rename("early_success")
    late = p[p["down"] >= 3].groupby(["game_id", "team_id"])["success"].mean().rename("late_success")
    pace = p.groupby(["game_id", "team_id"]).size().rename("pace").astype(float)

    d = drives.dropna(subset=["offense_id"]).copy()
    d["quality"] = (d["end_yards_to_goal"] <= QUALITY_DRIVE_YTG) | (d["off_points"] > 0)
    q = d[d["quality"]].groupby(["game_id", "offense_id"])["off_points"].mean().rename("pts_per_qd")
    own = d.groupby(["game_id", "offense_id"])["start_yards_to_goal"].mean()
    opp = d.groupby(["game_id", "defense_id"])["start_yards_to_goal"].mean()
    fp = (opp.rename_axis(["game_id", "team_id"]) - own.rename_axis(["game_id", "team_id"])).rename("field_position")

    out = pd.concat([early, late, pace, q.rename_axis(["game_id", "team_id"]), fp], axis=1).reset_index()
    g = games.set_index("game_id")
    out["season"] = out["game_id"].map(g["season"])
    out["available_at"] = out["game_id"].map(g["kickoff"]) + pd.Timedelta(days=1)
    return out


def with_luck(rows: pd.DataFrame) -> pd.DataFrame:
    """Luck residuals against league relationships fit on `rows` only (pass point-in-time rows)."""
    r = rows.copy()
    ok = r[["early_success", "late_success"]].notna().all(axis=1)
    b, a = np.polyfit(r.loc[ok, "early_success"], r.loc[ok, "late_success"], 1) if ok.sum() > 2 else (0.0, r["late_success"].mean())
    r["late_down_luck"] = r["late_success"] - (a + b * r["early_success"])
    r["finishing_luck"] = r["pts_per_qd"] - r["pts_per_qd"].mean()
    return r


def team_form(form_rows: pd.DataFrame, season: int, shrink_games: float = FORM_SHRINK_GAMES) -> pd.DataFrame:
    r = form_rows[form_rows["season"] == season]
    if r.empty:
        return pd.DataFrame(columns=FORM_FEATURES)
    league = r[FORM_FEATURES].mean()
    g = r.groupby("team_id")[FORM_FEATURES]
    n = g.count()
    shrunk = (g.sum() + shrink_games * league) / (n + shrink_games)
    sd = shrunk.std().replace(0, np.nan)
    return ((shrunk - shrunk.mean()) / sd).fillna(0.0)
```

`team_form` takes rows that **already have** the luck columns. Callers run `with_luck(visible_rows)` first, so the league fit uses only point-in-time data.

Fix the test helper so its rows include all `FORM_FEATURES` (it already does). Also add a `with_luck` test: a team-game with late success far above its early-down prediction gets positive `late_down_luck`.

- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** `feat(cfb): team form features (early success, luck, field position, pace)`.

---

### Task 8: Save team form in the posterior draws

**Files:**
- Modify: `gnomepy_research/models/cfb/efficiency/inference.py` (`Inputs`), `posterior_snapshots.py`
- Test: `tests/sessions/test_cfb_win_probability_models.py`

**Interfaces:**
- Consumes: `team_game_form`, `with_luck`, `team_form`, `FORM_FEATURES` (Task 7).
- Produces:
  - `Inputs` gets an optional field `form_rows: pd.DataFrame | None = None`. `Inputs.load(..., with_form: bool = False)` loads `cfb_plays` and `cfb_drives` and sets `form_rows = team_game_form(plays, drives, games)`.
  - When `inputs.form_rows` is set, `snapshot_season` writes `{key}__form` with shape `(T, len(FORM_FEATURES))` float32, aligned with `{key}__team_ids`. It is computed as `team_form(with_luck(form_rows[form_rows.available_at < cutoff]), season).reindex(team_ids).fillna(0)`, and also `{key}__form_names`.
  - Snapshot CLI flag `--form` (sets `with_form=True`).

- [ ] **Step 1: Failing test:** with the synthetic `Inputs` from Task 4 plus a hand-made `form_rows` (two teams, available before/after the cutoff), call a new pure helper `form_at(form_rows, season, cutoff, team_ids) -> np.ndarray`:
  - rows at or after the cutoff don't change the output (perturb them and compare);
  - shape `(len(team_ids), 5)`;
  - the FCS / unseen team row is all zeros.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** `form_at` in `posterior_snapshots.py`:

```python
def form_at(form_rows: pd.DataFrame, season: int, cutoff: pd.Timestamp, team_ids: np.ndarray) -> np.ndarray:
    visible = form_rows[(form_rows["available_at"] < cutoff) & (form_rows["season"] == season)]
    f = team_form(with_luck(visible), season) if len(visible) else pd.DataFrame(columns=FORM_FEATURES)
    return f.reindex(team_ids).fillna(0.0)[FORM_FEATURES].to_numpy(np.float32)
```

Then call it in `snapshot_season` when `inputs.form_rows is not None`. For split draws, `team_ids` repeats per unit, which is fine: the rows are identical across units, and `game_draws` uses the pass columns.

- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Re-snapshot split with form:**

```bash
poetry run python -m gnomepy_research.models.cfb.efficiency.posterior_snapshots 2020 2024 --likelihood normal --units split --form
```

Hyperparameters are reused (they don't depend on form); only the draws files are rewritten with the extra keys.

- [ ] **Step 6: Commit** `feat(cfb): point-in-time team form in posterior draws`.

---

### Task 9: Form terms in the outcome model, then backtest C

**Files:**
- Modify: `gnomepy_research/sessions/cfb_win_probability/models/win_probability.py`, `evaluation/backtest.py`
- Test: `tests/sessions/test_cfb_win_probability_models.py`

**Interfaces:**
- Consumes: `{key}__form`, `{key}__form_names` (Task 8); `MODELS["C"]` (Task 6).
- Produces:
  - `game_draws(..., form: bool = False)`. When true, it appends these terms (each constant across draws):
    - `early_success_diff`, `late_down_luck_diff`, `finishing_luck_diff`, `field_position_diff`: home minus away z-scores.
    - `pace_x_edge`: `(pace_h + pace_a) / 2 * (pass_edge + rush_edge)`. Pace is a z-score, so this scales the edge up in fast games.
  - New `PRIORS`, all N(0, 0.5) on the z-score scale, sign-agnostic: `early_success_diff`, `late_down_luck_diff`, `finishing_luck_diff`, `field_position_diff`, and `pace_x_edge` at N(0, 3) because it multiplies an edge in PPA units.
  - `form_terms(form_h: np.ndarray, form_a: np.ndarray, edge: np.ndarray) -> np.ndarray` returns `(G, S, 5)`. `form_h`/`form_a` are `(G, 5)` in `FORM_FEATURES` order; `edge` is `(G, S)`.

- [ ] **Step 1: Failing tests.**

```python
from gnomepy_research.sessions.cfb_win_probability.models.win_probability import form_terms

def test_form_terms_antisymmetric_under_swap():
    rng = np.random.default_rng(4)
    fh, fa, edge = rng.normal(size=(6, 5)), rng.normal(size=(6, 5)), rng.normal(0, .1, (6, 30))
    a = form_terms(fh, fa, edge)
    b = form_terms(fa, fh, -edge)
    np.testing.assert_allclose(a, -b)

def test_pace_scales_edge():
    fh = np.zeros((1, 5)); fa = np.zeros((1, 5)); fh[0, 4] = fa[0, 4] = 1.0
    t = form_terms(fh, fa, np.full((1, 3), 0.1))
    np.testing.assert_allclose(t[0, :, 4], 0.1)
```

- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.**

```python
FORM_TERMS = ["early_success_diff", "late_down_luck_diff", "finishing_luck_diff", "field_position_diff", "pace_x_edge"]

def form_terms(form_h: np.ndarray, form_a: np.ndarray, edge: np.ndarray) -> np.ndarray:
    diff = (form_h[:, :4] - form_a[:, :4])[:, None, :].repeat(edge.shape[1], 1)
    pace = ((form_h[:, 4] + form_a[:, 4]) / 2)[:, None] * edge
    return np.concatenate([diff, pace[..., None]], axis=-1)
```

In `game_draws` (split branch), when `form`: `f = draws[f"{key}__form"][p_cols]`, `fh, fa = f[hi], f[ai]`, then `xs` gets `form_terms(fh, fa, pe + re)` concatenated after the bowl term. Set `terms = [*TERMS_SPLIT, *FORM_TERMS]`. Raise `ValueError` if `form=True` and the draws lack `__form` (snapshot with `--form` first).

Wire `MODELS["C"]` in `backtest.py` (remove its `NotImplementedError`).

- [ ] **Step 4: Run tests, expect PASS. Then backtest C:**

```bash
poetry run python -m gnomepy_research.sessions.cfb_win_probability.evaluation.backtest 2023 2024 --model C
```

The report must also print the posterior mean and 90% interval of every form coefficient. **Expected signs:** `late_down_luck_diff` and `finishing_luck_diff` negative (regression); `field_position_diff` and `early_success_diff` positive; `pace_x_edge` positive. Note any that come out the other way. That's a finding to discuss, not something to fix by tuning.

- [ ] **Step 5: Commit** code + `report_2023_2024_C.txt`: `feat(cfb): form terms in win probability; backtest model C`.

---

## Phase 3: Compare against each other and the market

### Task 10: Model vs model vs opening/closing line comparison

**Files:**
- Create: `gnomepy_research/sessions/cfb_win_probability/evaluation/compare_models.py`
- Test: `tests/sessions/test_cfb_win_probability_models.py`
- Output: `results/backtest/comparison_2023_2024.txt`

**Interfaces:**
- Consumes: `predictions_2023_2024_{A,B,C[,D]}.parquet` (contract in Task 6); `cfb_lines` (`game_id, spread, spread_open, home_prob_novig`); `paired_bootstrap`, `summarize`, `log_loss` from `evaluation/metrics.py`; `pregame_prior` from `live/wp_data.py` (closing no-vig P(home), moneyline else spread).
- Produces:
  - `market_probs(lines) -> pd.DataFrame[game_id, p_open, p_close]`:
    - `p_open = norm.cdf(-median(spread_open) / 15.5)`, the same constant as `edge_analysis.py`;
    - `p_close = pregame_prior(lines)`.
  - `compare(preds: dict[str, pd.DataFrame], market: pd.DataFrame) -> str`. It runs on games present in every model **and** with both lines, and prints the number dropped. The report contains:
    1. A table per source (each model, open, close): n, log loss, Brier, calibration slope, by season.
    2. Paired log-loss differences with game-bootstrap 95% CI: B−A, C−B, C−A, (D−C), and each model − open, each model − close.
    3. Per model, the **blend weight vs the opening line**: fit `logit(y) ~ a*logit(model) + b*logit(open)` and report `a`. `a > 0` means the model carries information the open lacks, which is the actual alpha question. The same vs close is reported for reference.
    4. Log loss on the subset where `|p_model − p_open| > 0.05` (where we'd actually bet), for each model and the open.
  - CLI: `python -m ...evaluation.compare_models 2023_2024 A B C [D]`.

- [ ] **Step 1: Failing tests:**
  - `market_probs` drops games with no `spread_open` instead of filling them (Review Focus #7);
  - `compare` on two synthetic models where one equals the labels plus noise and the other is 0.5 reports the better one with a negative paired difference;
  - the dropped-games count appears in the output.

```python
def test_market_probs_drops_missing_open():
    lines = pd.DataFrame({"game_id": [1, 1, 2], "spread": [-7, -6.5, 3], "spread_open": [-6, -6, np.nan],
                          "home_prob_novig": [0.7, 0.71, np.nan]})
    m = market_probs(lines)
    assert m.loc[m.game_id == 1, "p_open"].iloc[0] > 0.5
    assert m.loc[m.game_id == 2, "p_open"].isna().all()
```

- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement**, reusing `blend_weight` logic from `edge_analysis.py`. Copy the small function in rather than importing a CLI module.
- [ ] **Step 4: Run tests, expect PASS. Then:**

```bash
poetry run python -m gnomepy_research.sessions.cfb_win_probability.evaluation.compare_models 2023_2024 A B C
```

- [ ] **Step 5: Commit** script + `comparison_2023_2024.txt`.
- [ ] **Step 6: CHECKPOINT. Stop and show the user the comparison table** before Phase 4. Decisions for the user:
  - which of A/B/C carries forward as the base for D;
  - whether any form term with a wrong-signed coefficient should be dropped.

---

## Phase 3b: CatBoost (model E)

### Task 10E: CatBoost win-probability model (model E) and C+E blend

Added 2026-10-13 at the user's request after the Task 10 checkpoint. It uses the same point-in-time inputs as model C, but a gradient-boosted classifier replaces the linear logistic layer, so it can learn interactions (rush O × opponent rush D, uncertainty × edge, week × edge).

**Files:**
- Modify: `pyproject.toml` (add `catboost`; run `poetry add catboost`)
- Create: `gnomepy_research/sessions/cfb_win_probability/models/catboost_wp.py`
- Modify: `gnomepy_research/sessions/cfb_win_probability/evaluation/backtest.py` (`--model E`)
- Modify: `gnomepy_research/sessions/cfb_win_probability/evaluation/compare_models.py` (blend labels like `C+E`)
- Test: `tests/sessions/test_cfb_catboost_wp.py`
- Output: `results/backtest/report_2023_2024_E.txt`, `predictions_2023_2024_E.parquet`, updated `comparison_2023_2024.txt`

**Interfaces:**
- Consumes:
  - `load_season_draws(season, directory, units)` from `win_probability.py`, on split+form draws files with keys `__off`, `__def`, `__team_ids`, `__units`, `__rush_share`, `__form`.
  - `FORM_FEATURES` from `models/cfb/features/team_form.py`.
  - The predictions parquet contract from Task 6.
- Produces:
  - `TEAM_FEATURES = ["o_pass", "o_rush", "d_pass", "d_rush", "o_pass_sd", "o_rush_sd", "d_pass_sd", "d_rush_sd", "rush_share", *FORM_FEATURES]`.
    - The first 8 are posterior **mean and sd over draws** at the game's weekly cutoff.
    - `rush_share` comes from `__rush_share`, and the form features from `__form`, all indexed on the pass columns. Reuse the same `p_cols`/`hi`/`ai` mapping and the same alignment `ValueError` as `game_draws`, via a shared helper; don't duplicate the logic.
  - `game_features(games, seasons, directory=DRAWS_DIR) -> pd.DataFrame`: one row per FBS-vs-FBS game. Columns:
    - `game_id, season, week, season_type, home_win`;
    - `h_<f>` and `a_<f>` for every f in TEAM_FEATURES;
    - matchup columns `pass_matchup_h = h_o_pass - a_d_pass`, `rush_matchup_h = h_o_rush - a_d_rush`, `pass_matchup_a = a_o_pass - h_d_pass`, `rush_matchup_a = a_o_rush - h_d_rush`;
    - `venue` (1 home, 0 neutral), `postseason` (0/1), `week`.
  - `swap(df) -> pd.DataFrame`: exchanges every `h_`/`a_` pair and every `_h`/`_a` matchup pair, sets `home_win = 1 - home_win`, and negates `venue`. At a neutral site, venue stays 0.
  - `fit_catboost(train: pd.DataFrame, seed: int = 0) -> CatBoostWP`:
    - Training data is `pd.concat([train, swap(train)])`. Augmenting with swapped copies teaches the model team-order symmetry.
    - Early stopping is evaluated on the last training season (also augmented). The model is then refit on all training seasons with the best iteration count.
    - Params: `loss_function="Logloss", depth=4, learning_rate=0.03, l2_leaf_reg=10, iterations=3000, random_seed=seed, verbose=False`.
  - `CatBoostWP.predict(df) -> np.ndarray`: `0.5 * (p(df) + (1 - p(swap(df))))`, so swapping teams at a neutral site gives exactly 1 − p.
  - `CatBoostWP.feature_importance() -> pd.Series`.
  - Backtest `MODEL_CONFIGS["E"] = dict(units=UNITS_SPLIT, form=True, qb=False, learner="catboost")`.
    - Existing entries get `learner="bayes"`.
    - For E, `run_season` trains on `game_features(games, 2020..S-1)` rows with a known `home_win` and predicts season S.
    - `p_home` and `p_bayes` are both set to the CatBoost probability, so `report()` works unchanged.
    - Baselines are computed as for the other models.
    - The report adds a top-15 feature-importance section.
  - `compare_models` accepts a label containing `+` (e.g. `C+E`) and scores the mean of those models' `p_home` on the shared game set.

- [ ] **Step 1: Failing tests** (synthetic frames, no npz needed for most):

```python
def test_swap_is_an_involution_and_flips_label(): ...
    # swap(swap(df)) equals df; swap flips home_win and exchanges h_/a_ columns
def test_predict_is_antisymmetric_at_neutral_site(): ...
    # fit on a small synthetic frame where home_win depends on h_o_pass - a_d_pass; at venue 0,
    # predict(df) + predict(swap(df)) == 1 (atol 1e-9)
def test_learns_obvious_signal(): ...
    # 2000 synthetic games, home_win ~ Bernoulli(sigmoid(10*(pass_matchup_h - pass_matchup_a)))
    # held-out log loss < 0.6 and AUC > 0.75
def test_game_features_from_npz(tmp_path): ...
    # reuse the TestSplitGameDraws npz pattern + __form; check h_o_pass equals the mean over draws of the
    # home team's pass offense column, and h_early_success equals its __form row
```

- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** `catboost_wp.py`, the backtest wiring and the blend labels.
- [ ] **Step 4: Run tests, expect PASS. Then run:**

```bash
AWS_PROFILE=AWSAdministratorAccess-241533121172 poetry run python -m gnomepy_research.sessions.cfb_win_probability.evaluation.backtest 2023 2024 --model E
AWS_PROFILE=AWSAdministratorAccess-241533121172 poetry run python -m gnomepy_research.sessions.cfb_win_probability.evaluation.compare_models 2023_2024 A B C E C+E
```

No tuning of CatBoost parameters against 2023–2024. If E is badly calibrated (slope outside [0.8, 1.2]), report it; don't fix it.

- [ ] **Step 5: Commit** code, tests, `report_2023_2024_E.txt` and the updated comparison.
- [ ] **Step 6:** Model D's base is the best of C, E and C+E by 2023–24 log loss. Record the choice in the ledger.

---

## Phase 4: QB status (model D)

### Task 11: QB quality, baseline, and per-game QB delta

**Files:**
- Create: `gnomepy_research/models/cfb/features/qb_status.py`
- Test: `tests/sessions/test_cfb_qb_status.py`

**Interfaces:**
- Consumes: `dropbacks(src)`, `game_starters(db)`, `preseason_qb(src, season)` from `features/qb.py`; `SourceData.as_of`; `week_cutoffs(games)` from `features/cutoffs.py`.
- Produces:
  - `REPLACEMENT_DROPBACKS = 150`
  - `replacement_ppa(db: pd.DataFrame) -> float`: mean `ppa` of dropbacks taken within each QB's first 150 career dropbacks (order by `kickoff`, then `play_seq`). This is the honest level for an unknown backup, not league mean.
  - `qb_quality(db: pd.DataFrame) -> pd.Series` (index `qb_id`): `(sum_ppa + 150 * replacement) / (n + 150)`.
  - `team_qb_baseline(db_season: pd.DataFrame, quality: pd.Series) -> pd.Series` (index `team_id`): dropback-weighted mean quality of QBs who threw for the team this season so far, which is the QB play the ratings learned from.
  - `historical_starters(src_full: SourceData) -> pd.DataFrame[game_id, team_id, qb_id]`: actual first passer of each team-game. **Documented look-ahead exception:** treats the actual starter as known pregame. Real announcements are usually public, but game-time decisions make the backtest somewhat optimistic.
  - `qb_deltas(src_full: SourceData, games: pd.DataFrame, starters: pd.DataFrame) -> pd.DataFrame[game_id, home_qb_delta, away_qb_delta]`: for each weekly cutoff, `view = src_full.as_of(cutoff)`, `db = dropbacks(view)`. `quality = qb_quality(db)`; the starter's quality is `quality.get(qb_id, replacement_ppa(db))`. The baseline is `team_qb_baseline` for teams that have played this season, else the quality of `preseason_qb(view, season).qb_expected_starter`. `delta = starter − baseline`, and 0 when the team has no starter row.

- [ ] **Step 1: Failing tests** (synthetic dropbacks; no datasets):

```python
import numpy as np
import pandas as pd
import pytest

from gnomepy_research.models.cfb.features.qb_status import (
    REPLACEMENT_DROPBACKS, qb_quality, replacement_ppa, team_qb_baseline,
)


def db(rows):
    return pd.DataFrame(rows, columns=["qb_id", "team_id", "season", "kickoff", "play_seq", "ppa"])


def test_replacement_uses_only_early_career_dropbacks():
    k = pd.Timestamp("2024-09-01", tz="UTC")
    vet = [("v", 1, 2024, k, i, 0.5) for i in range(400)]
    rookie = [("r", 2, 2024, k, i, -0.1) for i in range(50)]
    rep = replacement_ppa(db(vet + rookie))
    expected = (REPLACEMENT_DROPBACKS * 0.5 + 50 * -0.1) / (REPLACEMENT_DROPBACKS + 50)
    assert rep == pytest.approx(expected)


def test_unknown_backup_is_replacement_not_nan():
    k = pd.Timestamp("2024-09-01", tz="UTC")
    d = db([("v", 1, 2024, k, i, 0.4) for i in range(300)])
    q = qb_quality(d)
    assert np.isfinite(q.get("backup", replacement_ppa(d)))
    assert q["v"] > replacement_ppa(d) - 1e-12


def test_baseline_is_dropback_weighted():
    k = pd.Timestamp("2024-09-01", tz="UTC")
    d = db([("a", 1, 2024, k, i, 0.0) for i in range(90)] + [("b", 1, 2024, k, 100 + i, 0.0) for i in range(10)])
    q = pd.Series({"a": 0.3, "b": -0.1})
    assert team_qb_baseline(d, q)[1] == pytest.approx(0.9 * 0.3 + 0.1 * -0.1)
```

Plus one integration-style test with a tiny `SourceData` built like `make_source()` in `test_cfb_win_probability_features.py`. Team 1's starter throws in weeks 1–2, then a backup with no dropbacks starts week 3. Assert `home_qb_delta < 0` for the week-3 game and `== 0` for week 2.

- [ ] **Step 2: Run, expect FAIL.** `poetry run pytest tests/sessions/test_cfb_qb_status.py -v`

- [ ] **Step 3: Implement.**

```python
"""Pregame quarterback adjustment: announced starter vs the QB play a team's ratings were learned from."""
from __future__ import annotations

import pandas as pd

from gnomepy_research.models.cfb.features.cutoffs import week_cutoffs
from gnomepy_research.models.cfb.features.qb import dropbacks, game_starters, preseason_qb
from gnomepy_research.models.cfb.features.source import SourceData

REPLACEMENT_DROPBACKS = 150


def _ordered(db: pd.DataFrame) -> pd.DataFrame:
    return db.sort_values(["kickoff", "play_seq"], kind="stable")


def replacement_ppa(db: pd.DataFrame) -> float:
    d = _ordered(db)
    early = d[d.groupby("qb_id").cumcount() < REPLACEMENT_DROPBACKS]
    return float(early["ppa"].mean())


def qb_quality(db: pd.DataFrame) -> pd.Series:
    rep = replacement_ppa(db)
    g = db.groupby("qb_id")["ppa"].agg(["sum", "size"])
    return ((g["sum"] + REPLACEMENT_DROPBACKS * rep) / (g["size"] + REPLACEMENT_DROPBACKS)).rename("qb_quality")


def team_qb_baseline(db_season: pd.DataFrame, quality: pd.Series) -> pd.Series:
    n = db_season.groupby(["team_id", "qb_id"]).size().rename("n").reset_index()
    n["q"] = n["qb_id"].map(quality)
    return n.groupby("team_id").apply(lambda t: (t["n"] * t["q"]).sum() / t["n"].sum()).rename("qb_baseline")


def historical_starters(src_full: SourceData) -> pd.DataFrame:
    return game_starters(dropbacks(src_full))[["game_id", "team_id", "qb_id"]]


def qb_deltas(src_full: SourceData, games: pd.DataFrame, starters: pd.DataFrame) -> pd.DataFrame:
    cuts = week_cutoffs(games)
    g = games.merge(cuts, on=["season", "season_type", "week"])
    start = starters.set_index(["game_id", "team_id"])["qb_id"]
    rows = []
    for (season, cutoff), wk in g.groupby(["season", "cutoff"]):
        view = src_full.as_of(cutoff)
        db = dropbacks(view)
        db = db.assign(kickoff=db["game_id"].map(view.games.set_index("game_id")["kickoff"]))
        quality, rep = qb_quality(db), replacement_ppa(db)
        base = team_qb_baseline(db[db["season"] == season], quality)
        pre = preseason_qb(view, season).set_index("team_id")["qb_expected_starter"]

        def delta(game_id: int, team_id: int) -> float:
            qb = start.get((game_id, team_id))
            if qb is None:
                return 0.0
            baseline = base.get(team_id)
            if baseline is None:
                baseline = quality.get(pre.get(team_id), rep)
            return float(quality.get(qb, rep) - baseline)

        for r in wk.itertuples():
            rows.append({"game_id": r.game_id, "home_qb_delta": delta(r.game_id, r.home_id),
                         "away_qb_delta": delta(r.game_id, r.away_id)})
    return pd.DataFrame(rows)
```

(Check `dropbacks()` output columns at implementation time. If `kickoff` is already present, drop the `assign`. `week_cutoffs` returns `season, season_type, week, cutoff`; confirm against `features/cutoffs.py`.)

- [ ] **Step 4: Run, expect PASS.**

- [ ] **Step 5: Commit** `feat(cfb): pregame QB delta vs ratings baseline`.

---

### Task 12: Manual QB status overrides

**Files:**
- Create: `gnomepy_research/sessions/cfb_win_probability/data/qb_status.csv` (header only)
- Modify: `gnomepy_research/models/cfb/features/qb_status.py`
- Test: `tests/sessions/test_cfb_qb_status.py`

**Interfaces:**
- Produces:
  - CSV columns: `season,season_type,week,team,passer_key,recorded_at`. `team` is the CFBD school name as in `cfb_games`. `passer_key` is first initial + surname, lowercase (e.g. `j.fagnano`), matching `features/qb.py`.
  - `load_qb_status(path: Path, src_full: SourceData) -> pd.DataFrame[game_id, team_id, qb_id]`. It resolves the team via `team_name_ids(games)` and the game via (season, season_type, week, team). `qb_id` comes from `roster_keys(rosters)` (player_id), else the synthetic `f"t{team_id}:{passer_key}"`, the same convention as `dropbacks()`. It raises `ValueError` naming the CSV row if the team or the game can't be resolved.
  - `apply_overrides(starters, overrides) -> pd.DataFrame`: override rows replace the starter for the same `(game_id, team_id)` and add rows for upcoming games.

- [ ] **Step 1: Failing tests:** an unknown team raises `ValueError` with the team name in the message; an override replaces a historical starter; an override for an upcoming game (no plays yet) produces a row; a passer not on the roster gets the synthetic id.

```python
def test_unknown_team_raises_with_row(tmp_path, src):
    p = tmp_path / "qb.csv"
    p.write_text("season,season_type,week,team,passer_key,recorded_at\n2024,regular,3,Nowhere St,j.doe,2024-09-13T12:00Z\n")
    with pytest.raises(ValueError, match="Nowhere St"):
        load_qb_status(p, src)


def test_override_replaces_starter():
    starters = pd.DataFrame({"game_id": [1, 1], "team_id": [10, 20], "qb_id": ["a", "b"]})
    over = pd.DataFrame({"game_id": [1, 2], "team_id": [10, 10], "qb_id": ["backup", "backup"]})
    out = apply_overrides(starters, over).set_index(["game_id", "team_id"])["qb_id"]
    assert out[(1, 10)] == "backup" and out[(1, 20)] == "b" and out[(2, 10)] == "backup"
```

(`src` fixture: reuse `make_source()` from `test_cfb_win_probability_features.py`. Move it into `tests/sessions/conftest.py` if needed.)

- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** `load_qb_status` and `apply_overrides`:

```python
def apply_overrides(starters: pd.DataFrame, overrides: pd.DataFrame) -> pd.DataFrame:
    keep = starters.merge(overrides[["game_id", "team_id"]], on=["game_id", "team_id"], how="left", indicator=True)
    keep = keep[keep["_merge"] == "left_only"].drop(columns="_merge")
    return pd.concat([keep, overrides[["game_id", "team_id", "qb_id"]]], ignore_index=True)
```

- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** `feat(cfb): manual QB status overrides`.

---

### Task 13: QB term in the outcome model, backtest D, and final comparison

**Files:**
- Modify: `win_probability.py`, `evaluation/backtest.py`
- Test: `tests/sessions/test_cfb_win_probability_models.py`

**Interfaces:**
- Consumes: `qb_deltas`, `historical_starters`, `load_qb_status`, `apply_overrides` (Tasks 11–12); `PRIORS["qb_delta_diff"] = (4, 4)` (Task 5).
- Produces: `game_draws(..., qb: pd.DataFrame | None = None)`. When given, it merges on `game_id`, appends the term `qb_delta_diff = home_qb_delta − away_qb_delta` (constant across draws; missing → 0), and adds `"qb_delta_diff"` to `terms`. Enabled by `MODELS[...]["qb"]`.

- [ ] **Step 1: Failing tests:** `game_draws`-level unit tests need draws files, so test the pieces directly. A design with the qb column at zero gives identical `predict` output to the design without it (with gamma's qb column set to anything). The neutral-site swap still gives `p + p_swapped == 1` when the qb column is negated too.

```python
def test_qb_term_swap_symmetry():
    rng = np.random.default_rng(3)
    base = rng.normal(0, 0.1, (5, 40, 2))
    qb = np.repeat(rng.normal(0, 0.05, (5, 1)), 40, axis=1)[..., None]
    z = np.zeros((5, 40, 1))
    x = np.concatenate([base, z, z, qb], axis=-1)
    xs = np.concatenate([-base, z, z, -qb], axis=-1)
    m = OutcomeModel(gamma=np.column_stack([rng.normal(8, .5, 200), rng.normal(7, .5, 200),
                                            rng.normal(.3, .05, 200), rng.normal(-1, .5, 200), rng.normal(4, .5, 200)]))
    np.testing.assert_allclose(m.predict(x)["p_home"] + m.predict(xs)["p_home"], 1.0, atol=1e-12)
```

- [ ] **Step 2: Run, expect FAIL** (only if the implementation has a bug; this may pass immediately since `predict` is generic. That's fine: keep it as a regression guard).
- [ ] **Step 3: Implement** the `qb` param in `game_draws`:

```python
    if qb is not None:
        q = out[["game_id"]].merge(qb, on="game_id", how="left").fillna(0.0)
        qd = (q["home_qb_delta"] - q["away_qb_delta"]).to_numpy()[:, None, None]
        x = np.concatenate([x, np.broadcast_to(qd, (*x.shape[:2], 1))], axis=-1)
        terms = [*terms, "qb_delta_diff"]
```

In `backtest.py`, when `MODELS[model]["qb"]` is true, build `src_full = SourceData.load()`, then `starters = apply_overrides(historical_starters(src_full), load_qb_status(QB_STATUS_CSV, src_full))` and `qb = qb_deltas(src_full, games, starters)` once, passed to every `game_draws` call.

- [ ] **Step 4: Run tests, expect PASS.** Then run:

Wire `MODEL_CONFIGS["D"]` = the base chosen in Task 10E Step 6 (C, E, or C+E), plus `qb=True`. If the base is E or C+E, add `h_qb_delta`/`a_qb_delta` as CatBoost features (and the `qb_delta_diff` term for the C side of a blend) (edit the `MODELS` entry accordingly). Then:

```bash
poetry run python -m gnomepy_research.sessions.cfb_win_probability.evaluation.backtest 2023 2024 --model D
poetry run python -m gnomepy_research.sessions.cfb_win_probability.evaluation.compare_models 2023_2024 A B C D
```

Report, in addition to the updated comparison table:
- the paired log-loss difference D − C (D minus its base model);
- the same restricted to games where `|qb_delta_diff| > 0.05`, which is where it should matter;
- the fitted `qb_delta_diff` coefficient (posterior mean and 90% interval). It should be clearly positive.

- [ ] **Step 5: Commit** code + `report_2023_2024_D.txt` + the updated `comparison_2023_2024.txt`: `feat(cfb): pregame QB status term; backtest model D`.
- [ ] **Step 6: Show the user the final A/B/C/D vs open/close table** and recommend which model to publish.

---

### Task 14: Publish the chosen model from gnome-pipelines (user approval required)

**Files:**
- Modify: `~/Desktop/gnome-pipelines/gnome_pipelines/datasets/cfb.py`: if the chosen model is split, add a new dataset spec `TEAM_WEEK_POSTERIOR_SPLIT` = the existing posterior schema + `pa.field("unit", pa.string())`.
- Modify: `~/Desktop/gnome-pipelines/gnome_pipelines/cfb/posterior.py`

- [ ] **Step 1:** Add `--units {all,split}` and `--form` flags. Pick `UNITS_ALL`/`UNITS_SPLIT`, the dataset spec, `model_version(units)`, `hyper_name(likelihood, season, units)` and the artifact name `draws_name(season, units)`. Load `Inputs.load(ds, with_form=args.form)`.
- [ ] **Step 2:** Run the gnome-pipelines tests: `poetry run pytest -v` in `~/Desktop/gnome-pipelines`.
- [ ] **Step 3: STOP and ask the user** before:
  - tagging gnomepy-research (`scripts/release.sh minor`);
  - bumping the pin in gnome-pipelines;
  - pushing either repo;
  - publishing to the prod dataset store;
  - the QB status CSV going live.

  These are shared, hard-to-reverse actions.

---

## Weekly operating procedure (after this ships)

1. Before each slate, add one row per known QB change to `sessions/cfb_win_probability/data/qb_status.csv`, with `recorded_at` set to when you learned it.
2. Rebuild that week's predictions with the chosen model (with QB enabled). With no row, a team's QB is assumed to be the one the ratings already reflect, so the delta is 0.
