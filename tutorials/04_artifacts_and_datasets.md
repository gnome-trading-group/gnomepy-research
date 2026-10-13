# Artifacts and Datasets

Research reads two kinds of shared outputs through `gnomepy.data`: **datasets** (tables of features,
labels and raw game data) and **artifacts** (trained models and other files). Research code never
writes them. Every writer lives in gnome-pipelines, next to the table schemas.

---

## Concepts

**Dataset** — an append-only Iceberg table on S3 Tables. Every row carries `ingested_at` and the
`code_version` that wrote it. A correction appends a new row, and readers take the newest row per key,
so a dataset can be read as it was known at any past moment.

**Artifact** — a versioned file (XGBoost model, `.npz` draws) under the research bucket, indexed by
three Iceberg tables: one row per version, aliases such as `production` (the newest alias row wins), and
the datasets each version was trained on. Versions are UTC timestamps like `20261013T165937051464Z`.

Where things live is read from the account's SSM parameters (`/gnome/research/table-bucket-arn`,
`/gnome/research/artifact-root`), so the AWS profile you run under picks dev or prod. Set
`GNOME_TABLE_BUCKET_ARN` and `GNOME_ARTIFACT_ROOT` to override, or `GNOME_LOCAL_WAREHOUSE` to work
against a local SQLite catalog.

---

## Loading Datasets

```python
from datetime import datetime, timezone
from gnomepy.data import DatasetStore

ds = DatasetStore()
df = ds.load("cs2_match_history")                                  # as known now
df = ds.load("cs2_match_history", as_of=datetime(2026, 10, 1, tzinfo=timezone.utc))
df = ds.load("cs2_player_map_stats", columns=["rating"])           # key columns always included
```

`ds.loaded` records every load (dataset, as_of, row count, a hash of the rows). Hand it to the
pipeline that publishes a model so the model records exactly what it was trained on.

---

## Resolving Artifacts

In YAML configs, reference artifacts by URI instead of a local path:

```yaml
model_path: "artifact://xgboost_model/cs2_pre_map_win_prob"                          # latest
model_path: "artifact://xgboost_model/cs2_pre_map_win_prob:20261013T165937051464Z"   # pinned
model_path: "artifact://xgboost_model/cs2_pre_map_win_prob@production"               # alias
```

In Python, call `resolve_artifact_path()` before loading:

```python
from gnomepy.data import ArtifactStore, resolve_artifact_path

path = resolve_artifact_path("artifact://xgboost_model/cs2_pre_map_win_prob")
store = ArtifactStore()
store.list("xgboost_model")
store.inputs(store.latest("xgboost_model", "cs2_prematch_preveto"))   # datasets it was trained on
```

`resolve_artifact_path` handles three forms:
- `artifact://type/name[:version|@alias]` — downloads once, cached under `~/.cache/gnomepy/artifacts/`
- `s3://bucket/key` — downloads directly from S3
- anything else — passed through as a local path

---

## CLI Commands

```bash
poetry run research artifacts list [--type xgboost_model] [--name cs2_pre_map_win_prob]
poetry run research artifacts get xgboost_model/cs2_pre_map_win_prob [-o ./model.xgb]

poetry run research datasets list
poetry run research datasets get cs2_match_history [--as-of 2026-10-01T00:00:00+00:00] [-o ./history.parquet]
```

---

## Publishing

Add or change the writer in gnome-pipelines: declare the table's `TableSpec` in
`gnome_pipelines/datasets/`, and publish through `gnome_pipelines.stores`, which stamps rows with the
build's code version and the run id.
