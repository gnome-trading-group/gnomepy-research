# cfb_win_probability: college football pregame win probability

A dynamic hierarchical Bayesian model of opponent-adjusted offensive and defensive efficiency.
Structural priors come from talent, recruiting, returning production, transfers and coaching. The
pregame win probability is a logistic model over posterior samples. It trades college football
moneylines on Kalshi, Polymarket US and Polymarket International.

## Layout

| Package | Contents |
|---|---|
| `features/` | point-in-time gate (`source.as_of`), weekly cutoffs, ridge ratings, talent/QB/coach features, snapshots, leak audit |
| `models/` | priors, dynamic efficiency model, win-probability model (milestones 3–4) |
| `evaluation/` | backtests, metrics, baselines (milestone 5) |
| `live/` | **parked** drive-level Markov simulator for the future live engine, with tests |

```bash
# in gnome-pipelines: CFBD ingestion (cached, budgeted), quality checks, and publishing
poetry run python -m gnome_pipelines.cfb.ingest fetch 2018 2026 [--refresh-current 2026]
poetry run python -m gnome_pipelines.cfb.quality 2018 2026
poetry run python -m gnome_pipelines.cfb.ingest publish 2018 2026 [--only cfb_x ...]
poetry run python -m gnome_pipelines.cfb.features 2018 2026
poetry run python -m gnome_pipelines.cfb.posterior 2020 2026

# here: build without publishing, and audits
poetry run python -m gnomepy_research.models.cfb.features.snapshots 2018 2026
poetry run python -m gnomepy_research.sessions.cfb_win_probability.features.leak_audit
```

## Data source: CollegeFootballData (CFBD)

- **Key:** `CFBD_API_KEY` (exported in `~/.zshrc`). Free tier.
- **Rate limit:** 1,000 calls/month; the quota resets on the 1st. `/info` doesn't count against it.
  Every response is cached forever in `~/.cache/gnomepy/cfbd/`, and the client refuses to run
  past a reserve of 100 calls.
- **Cost:** a full 2018–2026 build cost 267 calls. Refreshing the current season costs about 20
  calls plus one per new week of plays.
- **Not on the free tier:** live play-by-play, the scoreboard, weather, and adjusted metrics.
- **Licensing:** the terms of use haven't been checked yet. Read them before any commercial use of
  data derived from CFBD.

## Datasets (`DatasetStore`, gnome-research-prod)

| Dataset | Grain | Available at (point-in-time policy) |
|---|---|---|
| `cfb_games` | game | kickoff (schedule); results after kickoff |
| `cfb_team_game_efficiency` | game × offense | kickoff + 1 day |
| `cfb_plays`, `cfb_drives`, `cfb_passers` | play / drive / dropback | after the play (`wallclock` where valid) |
| `cfb_lines` | game × sportsbook | **closing line, so only after kickoff**; CFBD has no quote timestamps |
| `cfb_talent`, `cfb_returning`, `cfb_rosters` | team-season | season start (day before the first kickoff) |
| `cfb_recruiting_teams`, `cfb_recruits` | class | Feb 15 of the class year |
| `cfb_portal` | transfer | `transfer_date` for leaving; season start for joining |
| `cfb_coaches` | coach-season | `hire_date` for the job; that season's results only from Feb 1 the next year |
| `cfb_team_game_ppa_cfbd` | game × team | validation reference only |

## Known data limitations

- **Drive start/end scores** are sometimes recorded after the score. Drive outcomes trust CFBD's
  label first and fall back to score deltas only for unlabelled drives.
- **Play-by-play:**
  - Plays come back unordered. About 130 games use synthetic negative ids, and play order is
    rebuilt from the ids.
  - About half of pre-2025 games have a few plays with a `wallclock` up to 24 hours early; those
    timestamps are nulled.
  - The play-text format changed in 2025, and passer names switch between `Joe Fagnano` and
    `J.Fagnano`. Passers are keyed by first initial + surname.
- **Moneylines** only exist from 2021. The closing spread converted at sd 15.4 points matches the
  no-vig moneyline's log loss on the same games, so it serves as the market benchmark for every
  season. CFBD lines carry no quote timestamps.
- **Transfer portal** data starts in 2021, and 16–41% of entries have no destination.
- **Talent and rosters** are CFBD's latest snapshots. They may include in-season revisions or
  mid-season additions, so a little look-ahead is possible.
- **Returning production** has a few out-of-range values (`pct_ppa` −3.1 to 1.7). Feature
  engineering winsorizes them and treats talent = 0 as missing.
- **FCS opponents** are 6.7% of efficiency rows. FBS offenses gain +0.357 PPA/play against them vs
  +0.092 against FBS, so FCS teams get their own group prior, never an average-FBS rating.
- **Excluded:** two team-games with incomplete play-by-play (fewer than 10 plays), flagged
  `usable=False`.
- **Deferred:** coordinators (CFBD has head coaches only), injuries, snap counts.

## Team efficiency observation

Offensive PPA per play over all plays CFBD assigns PPA to, excluding garbage time by Connelly's
thresholds (Q2 > 38, Q3 > 28, Q4 > 22). It matches CFBD's `/ppa/games` at corr 0.993 over 13,581
team-games and adds play counts, pass/rush splits, success rate and explosiveness.

## Point-in-time features (milestone 2)

- **One gate.** Feature code only ever receives `SourceData.as_of(cutoff)`, which keeps rows
  public strictly before the cutoff. It also hides:
  - results until kickoff + 1 day
  - future games' CFBD pregame Elo
  - portal destinations before season start
  - mid-season and interim coaches, and every coach-season's results, until Feb 1 of the next year
- **Cutoffs.** One per week, at that week's first kickoff. Every game in the week uses the same
  snapshot.
- **`cfb_team_preseason_features`** (season × team, as of week 1):
  - prior-season ridge-adjusted O and D
  - talent; recruiting points blended over 4 classes
  - position-group talent (top-k roster recruiting ratings; unrated = 0.70)
  - returning production, clipped to [0, 1]
  - portal in/out (NaN before 2021)
  - expected starting QB (returning flag, career dropbacks, career PPA shrunk at 150 dropbacks,
    recruiting rating)
  - head coach: new, tenure, career head-coach seasons, SRS above a regression-to-mean
    expectation (shrunk at 3 seasons)
- **`cfb_team_week_features`** (cutoff × team): season-to-date games and plays, offensive PPA,
  success rate, explosiveness, PPA allowed, ridge-adjusted O and D, current starting QB.
- **`cfb_pregame_features`**: one row per game with both teams' features and `label_*` outcome
  columns.
- **Checks:**
  - `leak_audit` scrambles everything that became public at or after each audited cutoff and
    requires identical features; it also requires identical features from row-shuffled tables.
    7 cutoffs, 20/20 pass, and it catches a planted leak.
  - Unit tests cover the gate's semantics.
- **Known gaps:**
  - 2018 has no prior-season rating (no 2017 data) and no returning-starter QB flag.
  - Rosters list about 200 players on two teams in 2021, from CFBD snapshot timing.
