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

## Phase 5: Residual CatBoost on top of C (model F)

Added 2026-10-13 at the user's request after reviewing E's feature importances.
- About 57% of E's importance sat on the four matchup features. The signal is mostly an additive sum (home matchups − away matchups), and trees approximate a sum poorly with about 3k games a season.
- F keeps C's linear predictor exact by passing C's log-odds as CatBoost's starting value (`baseline`). The trees then learn only corrections, from engineered linear and non-linear interaction features.

### Task 15: Residual CatBoost model F with engineered interactions

**Files:**
- Create: `gnomepy_research/sessions/cfb_win_probability/models/residual_wp.py`
- Modify: `gnomepy_research/sessions/cfb_win_probability/evaluation/backtest.py` (`--model F`, plus the `--validate` mode below)
- Modify: `gnomepy_research/sessions/cfb_win_probability/evaluation/compare_models.py` (pair `F - C`)
- Test: `tests/sessions/test_cfb_residual_wp.py`
- Output: `results/backtest/validation_2022_F.txt`, `report_2023_2024_F.txt`, updated `comparison_2023_2024.txt`

**Interfaces:**
- Consumes:
  - From `catboost_wp.py`: `game_features(games, seasons, directory)`, `swap(df)`, `PARAMS`.
  - From `win_probability.py`: `game_draws(..., units=UNITS_SPLIT, form=True)`, `fit(gd)`, `OutcomeModel.predict(x) -> DataFrame[p_home, p_lo, p_hi]`.
- Produces:
  - `engineer(df: pd.DataFrame) -> pd.DataFrame`. It adds the columns below to a `game_features` frame. Every column must be antisymmetric or symmetric under `swap`; the test checks this.
    - **Linear edges (antisymmetric):**
      - `pass_edge = pass_matchup_h − pass_matchup_a`
      - `rush_edge = rush_matchup_h − rush_matchup_a`
      - `net_edge = pass_edge + rush_edge`
      - `fp_diff = h_field_position − a_field_position`
      - `early_diff = h_early_success − a_early_success`
      - `finish_diff = h_finishing_luck − a_finishing_luck`
      - `late_diff = h_late_down_luck − a_late_down_luck`
    - **Uncertainty:** `unc = sqrt(mean of the 8 h_/a_ *_sd squared)` (symmetric), and `edge_x_unc = net_edge * unc` (antisymmetric).
    - **Season phase:** `edge_x_week = net_edge * week` (antisymmetric).
    - **Blowout:** `abs_edge = |net_edge|` (symmetric).
    - **Pace:**
      - `pace_sum = h_pace + a_pace` (symmetric)
      - `edge_x_pace = net_edge * pace_sum` (antisymmetric)
    - **Style clash:**
      - `style_diff = h_rush_share − a_rush_share` (antisymmetric)
      - `run_heavy_vs_run_d = h_rush_share * rush_matchup_h − a_rush_share * rush_matchup_a` (antisymmetric)
    - **Finishing × field position:** `finish_x_fp = h_finishing_luck * h_field_position − a_finishing_luck * a_field_position` (antisymmetric).
  - Extend `swap()` (or wrap it in `residual_wp`) so that it negates antisymmetric engineered columns and leaves symmetric ones, or simply recompute `engineer(swap(raw))`. Prefer recomputing; it can't drift.
  - `FEATURE_SETS: dict[str, list[str]]`:
    - `"linear"`: the 7 linear edges + venue;
    - `"interactions"`: linear + all the interaction columns above;
    - `"full"`: interactions + every raw `game_features` column.
  - `fit_residual(train: pd.DataFrame, baseline: np.ndarray, features: list[str], params: dict, seed: int = 0) -> ResidualWP`.
    - `baseline` is C's log-odds for the train rows.
    - Swap augmentation: swapped rows get `-baseline`. C has no intercept and venue is negated on swap, so C's log-odds flip sign exactly.
    - Early stopping uses the last training season, then a refit at the best iteration count, as in `fit_catboost`.
    - CatBoost gets `Pool(..., baseline=...)` for both fit and eval.
  - `ResidualWP.predict(df, baseline) -> np.ndarray`: `0.5 * (sigmoid(baseline + raw(df)) + 1 − sigmoid(−baseline + raw(swap(df))))`. Use `prediction_type="RawFormulaVal"` for the raw tree output, since CatBoost doesn't add the baseline at predict time.
  - **C's baseline for a season set:** fit C (`game_draws(..., units=UNITS_SPLIT, form=True)`, then `fit`) on the training seasons. `baseline = logit(clip(p_home, 1e-4, 1 − 1e-4))` for both train rows (in-sample: C has only 9 coefficients, so in-sample bias is small) and test rows. Join to `game_features` rows by `game_id`; raise if any row has no baseline.

**Selection protocol (prevents tuning on the scoring seasons):**
- [ ] `backtest --validate F` runs **only** train 2020–2021 → validate on 2022 for this grid:
  - feature sets `linear`, `interactions`, `full`;
  - `depth` ∈ {2, 3, 4};
  - fixed `learning_rate=0.03`, `l2_leaf_reg=10`.

  That's 9 configs. It writes `validation_2022_F.txt`, which reports log loss on 2022 for each config, C's 2022 log loss, and the chosen config (lowest 2022 log loss).
  - Write the chosen config to `results/backtest/F_config.json` and commit it.
  - Draws for 2020–2022 exist (split+form).
- [ ] `backtest 2023 2024 --model F` reads `F_config.json` and runs once, rolling-origin as usual: train 2020..S−1, early stop on S−1, test S.
  - `MODEL_CONFIGS["F"] = dict(units=UNITS_SPLIT, form=True, qb=False, learner="residual")`.
  - The report includes top-15 feature importances and the chosen config.
- [ ] `compare_models 2023_2024 A B C D E F C+E`, with a pair for `F − C`.
- **Never** change features or parameters after seeing 2023–24 results.

**Tests (TDD):**
- `engineer` symmetry: for each engineered column, `engineer(swap(raw))[col]` equals `−engineer(raw)[col]` for the antisymmetric ones and `+` for the symmetric ones.
- Residual zero-signal: when y is generated exactly from `sigmoid(baseline)`, F's held-out log loss is within 0.005 of the baseline's log loss. The trees shouldn't add noise.
- Residual finds a planted interaction: y ~ `sigmoid(baseline + 2·edge_x_unc)`, and F beats the baseline by > 0.01 log loss on held-out rows.
- `predict` is exactly antisymmetric at a neutral site.
- A missing baseline raises.

Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

---

## Phase 6: Model G, "Conference-aware"

Added 2026-10-13 after the error-lab analysis of model D on 2023–24. That analysis found:
- D compresses the gap between strong and weak teams. Favorites win 3.4 pts more than D expects (6.7 pts for open ≥ 80% favorites; calibration slope 1.33).
- Power 4 teams beat Group of 5 teams 5.5 pts more than expected (D minus open log loss +0.034).
- Non-conference games are the weakest context (+0.030).
- Weeks 1–9 trail the open by about 0.025.
- Service academies lean 5.5 pts toward the academy vs the open.

G = D + four fixes:
1. A hierarchical conference effect in the efficiency model.
2. Conference-mean preseason features.
3. A stretch term in the outcome layer.
4. An option-offense style flag.

Everything is opt-in. With `conference=False` (the default), every existing model and production path is byte-identical.

### Task 16: Conference effect and conference-mean priors in the efficiency model

**Files:**
- Modify: `gnomepy_research/models/cfb/efficiency/data.py`, `dynamic_efficiency.py`, `gaussian_solver.py`, `inference.py`, `posterior_snapshots.py`
- Test: `tests/sessions/test_cfb_win_probability_models.py`

**Interfaces:**
- `build_model_data(..., units=UNITS_ALL, conference: bool = False)`. When `conference=True`:
  - **Conference index:** each FBS team-season gets a conference index from that season's `cfb_games` (`home_conference` / `away_conference`; the most frequent value per team-season). The pooled FCS entity and teams with no conference get their own index per season, named `"none"`. New `ModelData` fields: `conf_keys: pd.DataFrame` (season, conference), and `ts_conf: np.ndarray` (one conference index per ts row, the same across units).
  - **Conference-mean features (fix 2):**
    - Append `conf_prior_off`, `conf_prior_def`, `conf_talent` and `conf_recruiting` to the base features before standardizing: off gets `conf_prior_off`, `conf_talent`, `conf_recruiting`; def gets `conf_prior_def`, `conf_talent`, `conf_recruiting`.
    - Each is the mean, over the team-season's conference members in the same season, of `prior_off_rating`, `prior_def_rating`, `talent_offense` / `talent_defense`, and `recruiting_points`.
    - All of these are preseason features, so point-in-time is unchanged.
    - Standardize within season, as the other features are.
  - With `conference=False`, `conf_keys` is empty, `ts_conf` has length 0, and features are unchanged.
- `build_model`: when `len(data.conf_keys) > 0`, add non-centered conference effects:
  - `z_conf_off ~ N(0,1)` and `z_conf_def ~ N(0,1)`, dims `conf`;
  - `start_off += s_conf_off * z_conf_off[ts_conf]`, and the same for def.
  - New hyperparameters: `s_conf_off` and `s_conf_def` ~ HalfNormal(0.05). In `Hyperparameters` they default to 0.0, and `from_dict` tolerates their absence.
  - `units` and the conference effect compose: in split mode, each unit gets its own conference effect. Index `conf × unit` the same way ts is unit-major.
- `gaussian_solver`: add the `z_conf_off` and `z_conf_def` blocks to `_layout` (size 0 when there are no conferences) and fill `B[row, lay["z_conf_off"] + conf_of_ts] = s_conf_off` in `_state_map`. The default layout size is unchanged.
- `Hyperparameters.conference: bool = False`. The fit and cutoff functions read it, as they read `units`.
- Naming: `hyper_name(..., units, conference)` appends `_conf`, and so does `draws_name`. `model_version` becomes `dyn-eff-split-conf-1`. Snapshot CLI flag `--conference`.
- Defaults are byte-identical. Existing tests must pass unchanged.

**Tests (TDD):**
- With `conference=False`, `ModelData` has the same arrays as before.
- With `conference=True`:
  - `ts_conf` maps every team in a conference to the same index;
  - FCS gets `"none"`;
  - the conference-mean feature equals the hand-computed mean.
- Synthetic recovery: a planted conference shift (+0.1 to every team in conf X) is recovered as the `z_conf` mean sign, with the exact solver matching MCMC (atol 0.01).
- Old hyperparameter JSONs still load.

### Task 17: Stretch and option terms in the outcome layer; snapshots and backtest for G

**Files:**
- Modify: `gnomepy_research/models/cfb/features/team_form.py` or a new `features/style.py`
- Modify: `posterior_snapshots.py`, `sessions/cfb_win_probability/models/win_probability.py`, `evaluation/backtest.py`
- Test: `tests/sessions/test_cfb_win_probability_models.py`, `tests/sessions/test_cfb_team_form.py`

**Interfaces:**
- **Option flag (fix 4):** `option_offense(efficiency, season) -> pd.Series[team_id → 0/1]`.
  - 1 when the team's **previous season** rush share (`rush_plays / (rush_plays + pass_plays)`) is ≥ 0.65.
  - With no previous season, it uses season-to-date rows with at least 3 games before the cutoff.
  - It must be point-in-time: the caller passes visible rows only.
  - Saved in snapshots as `{key}__option` (T,), aligned with team_ids, when the snapshot is built with `--conference`. (G snapshots are always split + form + conference.)
- Outcome terms added by `game_draws(..., stretch: bool = False, option: bool = False)`:
  - `edge_abs_edge = net * |net|`, where `net = pass_edge + rush_edge` (antisymmetric). It lets big edges stretch toward 0 and 1 (fix 3). Prior N(0, 30), since `net` is in PPA/play units of about ±0.3.
  - `option_diff = option_h − option_a` (antisymmetric). Prior N(0, 0.5).
- `MODEL_CONFIGS["G"] = dict(units=UNITS_SPLIT, form=True, qb=True, learner="bayes", conference=True, stretch=True, option=True)`. Thread the new keys through `_bayes_season` and `game_draws`. G reads draws named with the `_conf` suffix.
- Data runs (prefix `AWS_PROFILE=AWSAdministratorAccess-241533121172`):
  1. `posterior_snapshots 2020 2024 --likelihood normal --units split --form --conference`. This fits `normal_split_conf_2019_*` hyperparameters (NUTS; expect minutes per season).
  2. `backtest 2023 2024 --model G`
  3. `compare_models 2023_2024 A B C D E F G C+E`
- The G report must include:
  - every coefficient with its 90% interval;
  - the fitted `s_conf_off` and `s_conf_def`;
  - the top and bottom 3 conference effects per season at the final regular-season cutoff;
  - D vs G on the error-lab cuts that motivated G: favorites, big favorites, P4 vs G5, non-conference, weeks 1–4, weeks 5–9, service academies, toss-ups. For each, give bias, lean vs open and log-loss minus open, with 95% intervals.
- **No tuning against 2023–24.** If a term comes out wrong-signed, report it; don't fix it.

### Task 18: Add G to the CFB Error Lab artifact

- Rebuild the error lab's data with G included (`build_dist_data.py`, `MODELS += ["G"]`).
- In the page, add G as a selectable series named "Conference-aware". It needs its own categorical slot. The palette has 8 slots and E/F are the weakest, so drop F ("Residual CatBoost", identical to C) from the page to stay within 8 validated hues.
- Republish to the same artifact URL.

---

## Phase 7: Model H, "Talent & continuity"

Added 2026-10-13 after model G's cut analysis:
- Big favorites are still underpriced (G leans 3.6 pts toward the underdog vs the open; log loss minus open +0.026).
- Weeks 1–4 still trail the open by 0.022.

H = G + two components:
1. **Physical/talent gap terms** in the win-probability layer: talent composite, **blue-chip ratio**, and trench size from roster weights, plus a gap × edge interaction.
2. **A preseason projection regression** per unit that predicts this season's efficiency from last season's efficiency, weighted by how much of the production returns. Its projection feeds the efficiency model's preseason prior.

Everything is opt-in. Defaults and models A–G stay byte-identical.

### Task 19: Physical and talent-gap features and outcome terms

**Files:**
- Create: `gnomepy_research/models/cfb/features/physical.py`
- Modify: `sessions/cfb_win_probability/models/win_probability.py`, `evaluation/backtest.py`
- Test: `tests/sessions/test_cfb_physical.py`

**Interfaces:**
- `physical_features(src: SourceData, season: int) -> pd.DataFrame`, one row per FBS team_id. It is computed from `src.as_of(season start)` so it's point-in-time. Columns:
  - `talent`: the 247 talent composite (`cfb_talent`).
  - `blue_chip_ratio`: 4- and 5-star signees ÷ all rated signees over the last four high-school classes, `class_year` in [season−3, season]. Uses `cfb_recruits` (`recruit_type == "HighSchool"`, `committed_to`/`team_id` = team) with `available_at` before the cutoff. NaN if there are fewer than 20 signees.
  - `ol_weight`: mean weight of the 5 heaviest players listed at OL (positions OL, OT, OG, C) on that season's roster.
  - `dl_weight`: mean weight of the 4 heaviest DL (DL, DT, DE, NT).
  - Missing weights are NaN. Callers z-score within season and fill NaN with 0.
- `physical_table(src, seasons) -> pd.DataFrame[season, team_id, ...]`.
- Outcome terms via `game_draws(..., physical: pd.DataFrame | None = None)`. Merge home and away rows by (season, team_id); FCS or missing → 0 after z-scoring. All terms are antisymmetric:
  - `talent_diff = z_talent_h − z_talent_a`
  - `bluechip_diff = z_bcr_h − z_bcr_a`
  - `trench_diff = (z_ol_h − z_dl_a) − (z_ol_a − z_dl_h)`
  - `gap_x_edge = talent_diff * net`, where `net = pass_edge + rush_edge`
  - Priors: N(0, 0.5) for the first three and N(0, 10) for `gap_x_edge`. They're appended after G's terms.
- `MODEL_CONFIGS["H"]` = G's config plus `physical=True` and `projection=True` (projection wiring comes in Task 20; until then H raises NotImplementedError).

**Tests:**
- Blue-chip ratio on a hand-built recruits frame.
- OL/DL top-k weights.
- Point-in-time: a recruit class with `available_at` after the season start is excluded.
- Antisymmetry of the outcome terms under a home/away swap.
- Missing → 0.

### Task 20: Preseason projection regression and projection-informed priors

**Files:**
- Create: `gnomepy_research/models/cfb/efficiency/projection.py`
- Modify: `efficiency/data.py`, `inference.py`, `posterior_snapshots.py`
- Test: `tests/sessions/test_cfb_projection.py`

**Design:**
- **Target.** For each (season s, team, unit ∈ pass/rush × off/def), the target is the team's posterior mean rating at the **last regular-season cutoff of season s**, from the split+conference draws. Season-end rating is roughly 12 games of data, so the preseason prior barely affects it.
- **Features for projecting season s+1**, all public before s+1 starts:
  - `last_rating` for that unit: season s's final rating.
  - **Returning share** for that unit:
    - pass offense: `pct_passing_ppa` and `pct_receiving_ppa` (cfb_returning for season s+1);
    - rush offense: `pct_rushing_ppa`;
    - defense (no CFBD data): `def_continuity` = recruiting-rating-weighted share of season s's defensive roster players (positions DL, DT, DE, NT, LB, ILB, OLB, DB, CB, S, FS, SS) whose player_id is on the s+1 roster.
  - **The key interactions:** `last_rating × returning_share`, one per share for that unit.
  - Also `qb_returning_starter × last_pass_rating`, `qb_prior_ppa`, portal in/out stars, `coach_new`, `talent`, `blue_chip_ratio` (from Task 19), and the conference means from G.
- **Model:** one ridge regression per unit (sklearn `Ridge`, alpha fixed at 1.0 on standardized features). Rolling origin: the projection for season S is fitted only on pairs whose target season is < S. Pairs start at s = 2019 → 2020 (2018 has no prior-season rating). For S = 2020 there is a single pair season, so fit on it.
- **Output:** `project(preseason_inputs, season) -> DataFrame[season, team_id, unit, proj, proj_sd]`. `proj_sd` is the fit's residual sd, scaled per team by `1 + (1 − returning_share)`, so teams with low continuity get a wider starting range.
- **Integration**, opt-in via `build_model_data(..., projection: pd.DataFrame | None = None)`:
  - Append `proj` as an extra feature in **that unit's block only**, standardized like the others, so its beta is learned per unit.
  - Add the uncertainty flag `low_continuity` (returning share < 0.4) to `z_unc` only when projection is on.
- Naming: `_proj` suffix, chained as `normal_split_conf_proj_...` / `draws_split_conf_proj_...`. Snapshot CLI flag `--projection`. The projection table is built inside the snapshot run from the existing split+conf draws (2019–2023 seasons), which must exist first; raise a clear error if they don't.
- The report prints each unit's ridge coefficients (most recent fit) and the out-of-sample projection R² for each target season.

**Tests:**
- Rolling-origin: the projection for S doesn't change when season S's target rows are perturbed.
- The interaction is present and the sign is recovered on a planted synthetic relationship.
- `def_continuity` on a hand-built roster pair.
- Default `build_model_data` is unchanged.

### Task 21: Backtest H, cut table, Error Lab

- Snapshots:
  `posterior_snapshots 2020 2024 --likelihood normal --units split --form --conference --projection`
- Then:
  - `backtest 2023 2024 --model H`
  - `compare_models 2023_2024 A B C D E F G H C+E`
  - Add a pair for H − G.
- The H report adds the D/G/H motivating-cut table: big favorites, favorites, P4 vs G5, non-conference, weeks 1–4, weeks 5–9, toss-ups, service academies.
  - Each cut gets a lean vs the open and log loss minus open, with 95% CIs, and a paired H − G per cut.
  - Also break weeks 1–4 down by returning-share tercile.
- Error Lab: add H as a series named "Talent & continuity". To stay within 8 validated hues, drop B ("Pass/rush split"). Republish to the same URL.
- No tuning on 2023–24. Wrong-signed terms are findings.

---

## Phase 8: Model I, "Margin"

Added 2026-10-13. Diagnostic on 2023–24, comparing model probabilities converted to spreads at SD 15.5 with the opening spread and the actual margin:
- Model spreads are uniformly compressed: slope vs the open is 0.83 (A–D), 0.88 (G/H). The open is well scaled (actual margin ≈ 1.01 × open); G is under-scaled (actual ≈ 1.03 × model).
- For open favorites of 21+ points, G implies 25.5 against 27.4 actual (open 28.2).
- Hypothesis: the win/loss-only outcome layer can't learn scale from margin of victory.
- Model I = G with a margin-based outcome layer plus a monotone recalibration fitted on training seasons only.

### Task 22: Margin outcome layer

**Files:**
- Modify: `sessions/cfb_win_probability/models/win_probability.py` (new `fit_margin`, `MarginModel`)
- Modify: `evaluation/backtest.py` (`MODEL_CONFIGS["I"]`, a margin learner)
- Test: `tests/sessions/test_cfb_win_probability_models.py`

**Design:**
- Use the same design arrays `x` (G, S, T) and terms as G (`game_draws` with G's flags).
- Margin model: `margin[g] ~ StudentT(nu, mu[g], sigma)`, where `mu[g] = mean over rating draws of (x[g,s,:] @ beta)`, the same averaging-over-draws idea as `fit`.
  - Priors: beta ~ N(0, 50·prior_sd_scale) on the same terms. Scale the old PRIORS mu/sd by 100/8, since a 0.1 net PPA edge is about 12 points.
  - `sigma ~ HalfNormal(20)`, `nu ~ Gamma(2, 0.1)`.
  - `margin = home_points − away_points` for completed games.
- Prediction: `P(home) = mean over rating draws and posterior draws of (1 − T_cdf(0; nu, mu, sigma))`. That is the probability the margin is above 0, with ties negligible. Same API as `OutcomeModel.predict`: DataFrame with p_home, p_lo, p_hi. Also return `margin_mean` per game.
- No intercept: neutral-site swap symmetry must hold exactly. Venue is a term, as before.
- **Recalibration (optional flag `recal=True`):** after fitting on train seasons, compute in-sample train predictions, then fit a 2-parameter logistic recalibration `logit(p') = a·logit(p) + b` on **the last training season only**, using predictions from a margin model fit on the other training seasons (cross-fitted, like model F's fix). Apply it to test predictions. Report a and b. To keep symmetry, fix b = 0.
- `MODEL_CONFIGS["I"]` = G config + `learner="margin", recal=True`. Also `"I0"` = margin without recalibration, so we can see each piece's contribution.

**Tests:**
- Synthetic: margins generated as Normal(10·edge, 14) are recovered (beta within tolerance).
- P(home) is monotone in edge.
- Swap symmetry: p + p_swapped = 1 at a neutral site.
- The recalibration with b = 0 keeps symmetry.
- The recalibration is fitted only on rows from the last training season, using cross-fitted predictions.

### Task 23: Backtest I and I0, compression diagnostic, cut table, Error Lab
- `backtest 2023 2024 --model I0`, then `--model I`.
- `compare_models 2023_2024 A C D E G H I0 I C+E`, with pairs I−G and I0−G.
- Re-run the compression diagnostic (`compression_diag.py` logic, committed as `evaluation/compression_diag.py`) for G, I0 and I: the slope vs the open, the actual-vs-model slope, and the favorite-size band table.
- Cut table G/I0/I via `g_report` (big favorites, favorites, P4 vs G5, non-conference, weeks 1–4, weeks 5–9, toss-ups, service academies), with paired I−G.
- Error Lab: add I ("Margin"). To stay at 8 hues, drop A.
- No tuning on 2023–24.

---

## Phase 9: Model J, "Lean drives"

Added 2026-10-13. Model I showed that big-favorite compression is regression dilution from noisy ratings: a margin layer fitted on real margins still predicts 25 for 21+ favorites that win by 27. The fix is more information in the ratings, not output reshaping. The user wants an independent model (no blending with the line) and asked to cut feature count for clearer signal.

J = G, pruned (Task 24), + a drive-level observation channel in the efficiency model (Task 25).

### Task 24: Feature ablation on the 2022 validation season → frozen reduced G

**Protocol:** train 2020–2021 and score 2022 only. 2023–24 is never used for selection.

1. **Outcome-layer terms:** start from G's 15 terms. For each term, refit the outcome layer without it (same draws, so this is cheap) and record the 2022 log loss. Use 3 seeds, averaging the log losses, to cut refit noise.
2. **Preseason prior feature groups:** these need hyperparameter refits, so run them as groups (normal likelihood, split + conf, hypers fitted on 2019–2021 only, snapshots for 2020–2022):
   - QB: `qb_prior_ppa`, `qb_returning_starter`
   - coach: `coach_srs_resid`, `coach_new`
   - recruiting/talent: `talent_*`, `recruiting_points`
   - portal: `portal_in_stars`
   - returning: `returning_pct_ppa`, `returning_pct_passing_ppa`
   - conference means: `conf_*`
3. **Selection rule, fixed in advance:**
   - Backward elimination: repeatedly drop the item whose removal *improves or worsens by less than 0.0005* the 2022 log loss. Stop when every remaining removal costs more than 0.0005.
   - Freeze the result in `results/backtest/J_features.json` (kept outcome terms; kept prior features).
   - Report every step.
4. Implement this as `evaluation/ablation.py` (CLI `ablation --validate 2022`) plus a small `features=` / `terms=` selection hook. Concretely: `game_draws(..., keep_terms=None)` filters columns, and `build_model_data(..., drop_features=())` removes base features before standardizing. Defaults are unchanged.

### Task 25: Drive-level observation channel (opt-in `drives=True`)

- **Data:** `cfb_drives` per team-game offense, keyed by offense_id/defense_id and game_id, with `available_at` = kickoff + 1 day (same point-in-time rule as efficiency).
- **Observation:** `ppd` = points per drive for the offense in that game.
  - Exclude end-of-half/game kneel drives (`drive_result` in {"END OF HALF", "END OF GAME", "END OF 4TH QUARTER"} with `n_plays` ≤ 3).
  - Include garbage-time drives at weight 0.3. A drive is garbage time if it starts in Q2 at a margin > 38, Q3 > 28, or Q4 > 22, from `start_offense_score`/`start_defense_score`.
  - The effective drive count `n_eff` = Σ weights; `ppd` = Σ(w · points) / n_eff.
- **Likelihood:** add to `build_model` and to the exact solver (both must agree):
  - `ppd[g] ~ Normal(alpha_ppd[season] + lam_ppd * (Ocomb[off, k] − Dcomb[def, k]) + hfa_ppd * venue, sqrt(s_game_ppd² + s_drive_ppd² / n_eff))`
  - `Ocomb` = the play-share-weighted mean of the team's unit ratings (pass share from the season-to-date efficiency rows; 0.5/0.5 if none), and `Dcomb` likewise. With `units=("all",)`, Ocomb = O.
  - New hyperparameters: `alpha_ppd`, `lam_ppd` (prior N(12, 6): points per drive per PPA/play), `hfa_ppd`, `s_game_ppd`, `s_drive_ppd`. Defaults are off, so default paths stay byte-identical.
- Naming suffix `_drv`; snapshot flag `--drives`.
- Tests:
  - Synthetic recovery: ratings recovered better (higher correlation with truth) when a ppd channel generated from the same latent is added.
  - The exact solver matches MCMC.
  - Point-in-time: drives from games after the cutoff are excluded.
  - The garbage weight is applied.
  - Defaults are unchanged.

### Task 26: Backtest J and compare

- `J` = G's config + the frozen feature selection + `drives=True`. Also run **J0** = G + frozen selection without drives, to separate pruning from drives.
- Snapshots, then `backtest 2023 2024 --model J0` and `--model J`.
- `compare_models 2023_2024 C D G H I J0 J`, with pairs J0−G, J−G, J−J0.
- Compression diagnostic for G/J0/J, including the 21+ band.
- `g_report` cut table G/J0/J with paired J−G.
- Error Lab: add J ("Lean drives"). To keep 8 hues, drop E (CatBoost).
- No tuning on 2023–24.

---

## Phase 10: Model K, "Full priors + drives"

Added 2026-10-14. J's pruning (selected on 2022 alone) hurt early-season (+0.012), P4 vs G5 (+0.019) and non-conference (+0.022) vs G. J's drive channel helped mid-season (weeks 5–9 J−J0 −0.018) and was the first model whose blend weight vs the open excluded 0.

K = G (no pruning, full priors) + drive channel + independents as separate conference groups + season-phase edge weights.

### Task 27: Independents as separate groups, and season-phase edge terms

**Independents:**
- `build_model_data(..., independents_separate: bool = False)`. When True, each team whose conference is "FBS Independents" gets its own conference group per season, named `"FBS Independents:<team_id>"`, instead of one shared group.
- Default False keeps G/H/J byte-identical.
- `Hyperparameters.independents_separate` defaults to False and is dropped from to_dict when off.
- Naming suffix `_ind`, ordered after `_conf`.
- Snapshot flag `--independents-separate`.

**Phase terms:**
- `game_draws(..., phase: bool = False)` appends `pass_edge_early = pass_edge × early` and `rush_edge_early = rush_edge × early`, where `early = 1` for regular-season weeks 1–4 and 0 otherwise.
- Both are antisymmetric under swap. Priors N(0, 5).
- They're appended after the existing terms.

**Tests:**
- Independents map to distinct groups.
- The default is unchanged.
- Phase terms are zero outside weeks 1–4, antisymmetric under swap, and in the right column order.

### Task 28: Snapshots, backtest K, comparison, closing-line value

- Snapshots 2020–2024: `--likelihood normal --units split --form --conference --independents-separate --drives`. No drop features.
- `MODEL_CONFIGS["K"]` = G flags + `drives=True`, `independents_separate=True`, `phase=True`. No keep_terms or drop_features.
- `backtest 2023 2024 --model K`.
- `compare_models 2023_2024 C D G H I J0 J K`, with pairs K−G, K−J.
- `compression_diag` for G/J/K.
- `g_report` cut table G/J/K with paired K−G and K−J. Cuts: big favorites, favorites, P4 vs G5, non-conference, weeks 1–4, weeks 5–9, toss-ups, service academies, FBS independents.
- **New, the closing-line value check** (`evaluation/clv.py`): for each model, regress (p_close − p_open) on (p_model − p_open) across games, with a game-bootstrap 95% CI on the slope, plus the correlation.
  - A positive slope means the market moves toward the model between open and kickoff.
  - Report it overall and for |p_model − p_open| > 0.05.
  - Run it for G, J and K.
- Report per-season (2023, 2024) log loss for K, so we can check the gain isn't one season only.
- Error Lab: add K. To keep 8 hues, drop C.
- No tuning on 2023–24.

---

## Phase 11: Model L, "Settling ratings"

Added 2026-10-14. K vs G diagnostics on 2023–24:
- K is better on regular-season home-site games (−0.011, significant).
- K is worse on regular-season neutral-site games, mostly week 5+ and conference championships (+0.066, significant), and on the postseason (+0.050). In the postseason it is more confident, with 7 confident wrong picks vs G's 1.
- Late-season weekly rating movement is 0.036 for K vs 0.012 for G, and K's end-of-season rating spread is wider (0.254 vs 0.204).

**Hypothesis:** K's single drift rate (tau ≈ 0.034, pushed up by the drive channel) suits September, but keeps ratings chasing recent results all season.

L = K + phase-varying drift rates (with a postseason phase) − the unstable bowl term.

### Task 29: Phase-varying random-walk drift (opt-in `tau_phases=True`)

- **Phase of each random-walk step,** by the destination period's (season_type, week):
  - early: regular weeks ≤ 4
  - mid: regular weeks 5–9
  - late: regular weeks ≥ 10
  - post: postseason
- **Drift parameters:** `tau_off` and `tau_def` become vectors of length 4, indexed by phase. Each phase's step uses that phase's tau.
- **Priors:** HalfNormal(0.03) for early/mid/late; HalfNormal(0.06) for post. The post walk is where bowl-layoff uncertainty enters, so post gets a wider prior.
- **Defaults off:** scalar tau, byte-identical to today. `Hyperparameters.tau_phases` defaults to False. When on, tau_off/tau_def are stored as lists in the JSON; when off, they stay scalar.
- **Exact solver:** `_state_map` uses the step's phase tau for each cumulative step. Each B row sums the tau of each step up to period k, so the eps columns get per-step tau values.
- Naming suffix `_tph`, after `_drv`. Snapshot flag `--tau-phases`.
- **Outcome layer:** add `exclude_terms` to `game_draws`, which drops named terms after all terms are built and is applied before keep_terms. L excludes `bowl_edge`.

**Tests:**
- Default unchanged.
- Phase assignment from (season_type, week).
- Exact solver == MCMC with tau_phases (O at atol 0.01).
- Synthetic recovery: data generated with large early drift and near-zero late drift gives fitted early tau > late tau.
- exclude_terms removes the column and its prior.

### Task 30: Backtest L

- Snapshots 2020–2024: `--likelihood normal --units split --form --conference --independents-separate --drives --tau-phases`.
- `MODEL_CONFIGS["L"]` = K + `tau_phases=True` + `exclude_terms=["bowl_edge"]`.
- `backtest 2023 2024 --model L`.
- `compare_models 2023_2024 D G H I J K L`, with pairs L−K and L−G.
- **The motivating cut table G/K/L,** each with log loss, paired L−K and L−G with 95% CIs: regular-season home-site, regular-season neutral-site, postseason, weeks 1–4, weeks 5–9, weeks 10+, big favorites, toss-ups, P4 vs G5, non-conference.
- **Diagnostics:**
  - Postseason confidence (mean |p − 0.5|) and the count of confident wrong picks (wrong side, |p − 0.5| > 0.25).
  - Weekly rating volatility by phase for G/K/L, from the team-week posterior parquet: mean |Δ(offense_mean + defense_mean)|, pass unit, regular season, weeks 2–5, 6–10 and 11+.
  - The fitted tau per phase.
- CLV (`clv.py`) for G, K and L, plus per-season log loss.
- Error Lab: add L. To keep 8 hues, drop D.
- No tuning on 2023–24.

---

## Phase 12: Model M, "Confident favorites"

Added 2026-10-14.
- **The miss:** on open ≥ 80% favorites, K says about 85% but they win about 90.5%, an under-rating of 5.4 points. The open itself under-rates them by 2.6. K's log loss on this cut is 0.287 vs the open's 0.270.
- **The diagnosis:** Normal team-strength priors shrink true outliers as hard as lucky teams, and garbage-time evidence of dominance is discarded or down-weighted.
- **The constraint:** bowls need less confidence, not more.

### Task 31: Heavy-tailed team strengths (opt-in `heavy_tails=True`)

- **Prior:** the start states use `z0_off, z0_def ~ StudentT(nu_team, 0, 1)` instead of Normal(0, 1). The random-walk innovations stay Normal.
  - In the hyper fit, `nu_team ~ Gamma(2, 0.2)` (mean 10, which allows heavy tails), shared by off/def.
  - `Hyperparameters.heavy_tails` defaults to False; `nu_team` defaults to inf and is omitted from to_dict when off.
- **Fitting:** the exact Gaussian solver no longer applies. Cutoff fits with heavy_tails use MCMC (nutpie; fewer draws are fine, e.g. 300 draws and 300 tune, 2 chains). `fit_cutoff_exact` must raise when heavy_tails is on, so it can't be silently misused.
- Naming suffix `_ht`, after `_tph` in the order split, conf, ind, proj, drv, tph, ht, drop. Snapshot flag `--heavy-tails`.
- **Tests:**
  - Default unchanged.
  - Synthetic recovery: with one planted extreme team (+4 sd), the heavy-tailed fit leaves it closer to truth than the Normal fit, while ordinary teams are recovered equally well.
  - fit_cutoff_exact raises with heavy_tails.
  - Naming.

### Task 32: Out-of-sample tail calibrator with a postseason switch

- **Out-of-sample predictions:** run `backtest 2021 2022 --model K` (outcome layer trained on 2020..S−1, draws already exist).
- **Calibrator** (`models/` is not needed; a session module `models/tail_calibrator.py`):
  - With z = logit(p): `logit(p') = a·z + b·z·|z| + post·(c·z)`, where post is 1 for postseason games. No intercept, so swap symmetry holds.
  - Fit (a, b, c) by maximum likelihood on the pooled 2021–22 out-of-sample predictions. Report the coefficients with a bootstrap CI.
  - Apply to 2023–24 predictions. The calibrator must never be fitted on 2023–24.
- **Model K+cal:** add `MODEL_CONFIGS["Kc"]` = K + `calibrator="tail"`. The backtest loads the calibrator fitted from that model's own 2021–22 out-of-sample predictions; raise if they're missing.
- **Tests:** symmetry (p(swap) = 1 − p); a calibrator fitted on synthetic data with known under-confidence in the tails recovers b > 0; the postseason term is applied only to postseason rows.

### Task 33: Garbage-time weight, chosen on 2022 only

- `team_game_drives(..., garbage_weight=0.3)` becomes a parameter, threaded through `Inputs` and snapshot naming (suffix `_gw<value>` when ≠ 0.3) and the CLI flag `--garbage-weight`.
- **Validation:** for the weights {0.3, 0.5, 1.0}, build K-config snapshots for 2020–2022, train the outcome layer on 2020–2021, and score 2022.
  - Choose the lowest 2022 log loss. Also report the 2022 big-favorites log loss for each weight.
  - Freeze the choice in `results/backtest/M_garbage.json`.
- **Tests:** the weight is applied (a hand-built drive frame); default naming unchanged.

### Task 34: Model M, backtest and comparison

- **Config:** `MODEL_CONFIGS["M"]` = K + `heavy_tails=True` + the frozen garbage weight + `calibrator="tail"`. The calibrator is fitted on M's own 2021–22 out-of-sample predictions.
- **Runs:**
  - snapshots for 2020–2024;
  - `backtest 2021 2022 --model M` (for the calibrator);
  - `backtest 2023 2024 --model Kc` and `--model M`;
  - `compare_models 2023_2024 G K L Kc M`, with pairs M−K, Kc−K, M−Kc;
  - `clv` for K, Kc, M.
- **Report:**
  - The cut table K/Kc/M with paired differences vs K: big favorites, favorites, toss-ups, postseason, regular neutral, weeks 1–4, weeks 10+, P4 vs G5.
  - Big-favorite bias: mean(result − p) for K/Kc/M/open.
  - Postseason confident wrong picks.
  - Fitted nu_team per season.
  - Per-season log loss.
- **Error Lab:** rebuild with M and Kc added.
- No tuning on 2023–24.

---

## Phase 13: Model N, "Honest observation model"

Added 2026-10-14.

**Diagnosis.** K's gap to the open sits in the games where it disagrees strongly with the market. Those games are early-season heavy, and K's rating edges there are twice the usual size with no extra posterior uncertainty. Three measured misspecifications in the efficiency model explain this:

1. **Drives double-count each game.** Within a team-game, the residuals of points per drive and PPA correlate 0.72 (pass 0.55, rush 0.36; pass vs rush −0.04). The model treats them as independent. Adding drives raised τ from about 0.008 to 0.035 per week (`conf_drop` → `conf_drv_drop`). The ratings now chase single games.
2. **FCS games are treated as precise.** FBS-vs-FCS team-games carry about 0.11 PPA/play of extra noise beyond play sampling; FBS mismatches don't. 538 of 622 such games fall in weeks 1–3, but `sigma_game` is about 0.008.
3. **One rating per unit blends per-play efficiency with drive scoring.** In split-half tests, half-season points-per-drive ratings beat PPA ratings on future margins (R² 0.348 vs 0.277; win log loss 0.558 vs 0.586). A single 50/50 blend loses about 0.017 R² against separate components.

**Design.** Add one opt-in tuple, `extensions ⊆ {"couple", "fcs", "finish"}`, threaded like `drop_features`.
- Naming suffix after `ht`, in canonical order `_cpl`, `_fcs`, `_fin`.
- `couple` and `finish` require `drives=True`.
- The default path stays byte-identical.

**Model terms:**
- **couple:** ppd[g] | PPA rows of the same (game, offense) is distributed N(mu_ppd + Σ_u κ_u (y_u − mu_u), σ_ppd). This is the exact Gaussian factorization of jointly distributed PPA and ppd noise. κ ~ N(2, 2) per unit. In the solver the drive row becomes y_d − Σκ(y_u − c_u) = c_d + (A_d − Σκ A_u)θ + e.
- **fcs:** σ² gets + s_fcs² on PPA rows with an FCS side, and + s_fcs_ppd² on drive rows. Priors s_fcs ~ HalfNormal(0.1) and s_fcs_ppd ~ HalfNormal(1.0).
- **finish:** a static team-season finishing effect per side, fin_off and fin_def ~ N(0, s_fin_off/def), with priors HalfNormal(0.5). It enters only the ppd mean as + fin_off[off] − fin_def[def].
  - Snapshots store `__fin_off`/`__fin_def` aligned to the unit columns.
  - The outcome layer adds the term `finish_edge` = (fo_h − fd_a) − (fo_a − fd_h), with prior N(1, 1) in logit per ppd.

**Tasks:**
- **Task 35:** extensions plumbing plus `fcs` (data flags, PyMC model, exact solver, hypers round trip, naming, CLI).
- **Task 36:** `couple` (data row links, PyMC conditional, solver transform).
- **Task 37:** `finish` (latents, posterior draws, snapshots, `game_draws` term).
- Each task includes these tests: the default is unchanged, exact vs NUTS agreement on synthetic data, synthetic recovery (planted coupling, FCS noise, finishing), naming, and the round trip.

**Task 38: selection and scoring.** The rule is fixed before any run:
- **Ablation ladder:** N1 = K+couple, N2 = N1+fcs, N = N2+finish.
- **Validation:** snapshots 2020–22 (hypers 2019..S−1), outcome trained on 2020–21, score 2022, average of 3 refits.
- **Keep rule:** keep each step unless it worsens 2022 log loss by more than 0.002 against the previous kept step. These are diagnostic-driven corrections, so the threshold guards against harm, not for gain.
- Freeze the chosen config in `results/backtest/N_selection.json` before scoring 2023–24.
- **Scoring:** score 2023–24 once with the frozen N (ladder steps reported for information), plus the 2021–22 out-of-sample backtest.
- **Report:**
  - τ before and after;
  - κ, s_fcs and s_fin;
  - the cut table;
  - the big-disagreement bucket table;
  - CLV;
  - the Error Lab rebuild.
- No tuning on 2023–24.

---

## Phase 14: Preseason-feature hygiene ("prior" extension, models Kp and Np)

Added 2026-10-15.

**Diagnosis.** The 15 worst weeks 1–4 games of 2021–24 carry +13.6 of N's +14.2 early-season log-loss gap to the open. Recurring causes:
- **An FCS move-up rated by FCS-era inputs.** James Madison's 4 games are 26% of the gap. Its talent sat in the bottom 1–9% of FBS and its returning production read 4–11%, while it went 8–3, 11–1 and 9–4.
- **A 2020-COVID artifact.** NMSU 2021 had no prior rating but returning production of 100%.
- **Academies.** Their missing talent was silently filled with the league average.

**Fix (opt-in extension `prior`, suffix `_pfx`; default unchanged).** `data.prior_fixes`, applied before conference means:
- (a) No prior-season rating → returning production is unknown (NaN, so the season mean).
- (b) A team first seen as FBS after the games table's first season has, for its first 3 FBS seasons, talent, recruiting and returning production set to NaN.
- (c) A `talent_missing` indicator is added to the offense and defense prior features.

**Task 35p: selection and scoring.** The rule is fixed before any run:
- **Candidates:** Kp = K + prior and Np = N + prior.
- **Validation:** for each, snapshots 2020–22 (hyperparameters 2019..S−1), outcome layer trained on 2020–21, scored on 2022 with 3 seeds.
- **Keep rule:** keep unless 2022 log loss is worse than the base model (K or N) by more than 0.002.
- **Scoring:** then score 2023–24 once, plus 2021–22. Report weeks 1–4 and the 15-game list.
- Both kept or reverted models enter the 2025 holdout comparison.
- No tuning on 2023–24.

---

## Phase 15: Mid-season (weeks 5–9) gap

Added 2026-10-15.

**Diagnosis.** Np's remaining gap to the open is almost entirely weeks 5–9: +0.025 per game in 2021–22 and +0.015 in 2023–24, about 8 of 9 units. It is not a calibration bias, because the slope points in opposite directions in the two periods. Weeks 5–9 is when prior and data weigh about equally, so it is the phase most sensitive to wrong noise assumptions in the observation model.

**Phase 1 diagnostics.** These are analysis only, on 2021–24 out-of-sample Np predictions and posteriors:
- **D1, prior/data balance.** One-week-ahead predictive log density of unit PPA under τ ×{0.5, 1, 1.5, 2.5} and prior sd ×{0.75, 1, 1.5}, using each season's Np hyperparameters, scored by phase.
- **D2, mismatch contamination.** (a) Does a team's season-to-date share of mismatch games (FCS, or Np ≥ 90%) explain Np's weeks 5–9 residuals and disagreement? (b) Are unit PPA residuals compressed for large predicted edges?
- **D3, mid-season signal screen.** Season-to-date success rate, turnover margin and pace, as home − away differences, tested against Np's weeks 5–9 residuals. Fit on one period and score on the other.

**Phase 2: build only what Phase 1 supports.**
- Candidates: a success-rate channel (coupled), turnover-luck neutralisation, mismatch down-weighting, and phase-dependent updating.
- Selection on 2022 with the harm rule, then score 2023–24 once, then the holdout.

---

## Weekly operating procedure (after this ships)

1. Before each slate, add one row per known QB change to `sessions/cfb_win_probability/data/qb_status.csv`, with `recorded_at` set to when you learned it.
2. Rebuild that week's predictions with the chosen model (with QB enabled). With no row, a team's QB is assumed to be the one the ratings already reflect, so the delta is 0.
