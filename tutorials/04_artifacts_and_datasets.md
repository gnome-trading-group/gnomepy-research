# Artifacts and Datasets

Research sessions produce two kinds of shareable outputs: **artifacts** (trained models, calibrated value functions) and **datasets** (tabular training data shared across sessions). Both are versioned, stored in S3, and queryable from the web UI.

---

## Concepts

**Artifact** — a versioned binary file produced by a research session (XGBoost model, `.npz` value function, calibration output, etc.). Identified by `type/name:version`. Lives under the producing session or `__global__` if session-independent.

**Dataset** — a versioned Parquet file containing tabular data (features, labels, raw game data). Shared across all sessions. Identified by `name:version`.

Both are stored in `s3://gnome-research-{STAGE}/` and registered in DynamoDB for metadata querying.

---

## Publishing Artifacts

```python
from gnomepy_research.artifacts import ArtifactStore

store = ArtifactStore()

# After training an XGBoost model:
model.save_model("/tmp/cs2_fair_value.xgb")
ref = store.publish(
    "/tmp/cs2_fair_value.xgb",
    artifact_type="xgboost_model",
    name="cs2_fair_value",
    session_name="cs2_xgb",
    description="CS2 fair value model trained on match features",
    params={"n_estimators": 200, "max_depth": 6},
)
print(ref)  # artifact://xgboost_model/cs2_fair_value:1
```

Versions are auto-incremented — republishing the same `type/name` creates version 2, 3, etc.

---

## Resolving Artifacts (using in strategies)

In YAML configs, reference artifacts by URI scheme instead of a local path:

```yaml
# pinned version:
value_function_path: "artifact://value_function/kalshi_cal:3"

# always latest:
value_function_path: "artifact://value_function/kalshi_cal"
```

In Python, call `resolve_artifact_path()` before loading. Old local paths still work unchanged:

```python
from gnomepy_research.artifacts import resolve_artifact_path

path = resolve_artifact_path("artifact://xgboost_model/cs2_fair_value")
model = xgb.Booster()
model.load_model(path)
```

`resolve_artifact_path` handles three URI forms:
- `artifact://type/name[:version]` — downloads from artifact store, caches at `~/.cache/gnomepy/artifacts/`
- `s3://bucket/key` — downloads directly from S3
- any other string — passed through as a local path

---

## Publishing Datasets

```python
import pandas as pd
from gnomepy_research.artifacts import DatasetStore

ds = DatasetStore()

cs2_df = pd.DataFrame(...)  # your features / labels
ref = ds.publish(
    cs2_df,
    name="cs2_match_features",
    description="CS2 match-level features: team stats, map, round outcomes",
    producing_session="cs2_xgb",
)
print(ref)  # dataset://cs2_match_features:1
```

To update with new data, publish again — a new version is created, old versions remain immutable:

```python
updated_df = pd.concat([ds.load("cs2_match_features"), new_games_df])
ds.publish(updated_df, name="cs2_match_features", description="Added Sep 2026 matches")
# → version 2
```

---

## Loading Datasets

```python
from gnomepy_research.artifacts import DatasetStore

ds = DatasetStore()

# latest version:
df = ds.load("cs2_match_features")

# pinned version for reproducibility:
df = ds.load("cs2_match_features:1")
```

Data is cached locally at `~/.cache/gnomepy/datasets/` after first download. Remote Batch jobs have S3 access to the research bucket and download directly.

---

## Composing Multiple Datasets at Training Time

Use multiple named datasets to keep concerns separate:

```python
features = ds.load("cs2_match_features")
labels = ds.load("cs2_labels")
df = features.merge(labels, on="match_id")
```

---

## CLI Commands

```bash
# List all artifacts
poetry run research artifacts list

# Filter by type or name
poetry run research artifacts list --type xgboost_model
poetry run research artifacts list --name cs2_fair_value

# Publish a local file
poetry run research artifacts publish /tmp/model.xgb \
  --type xgboost_model --name cs2_fair_value \
  --session cs2_xgb --description "Initial model"

# Download an artifact (latest version)
poetry run research artifacts get xgboost_model/cs2_fair_value

# Download a specific version
poetry run research artifacts get xgboost_model/cs2_fair_value:2 -o ./model_v2.xgb

# List all datasets
poetry run research datasets list

# Publish a parquet file as a new dataset version
poetry run research datasets publish features.parquet \
  --name cs2_match_features --description "Initial dataset"

# Download a dataset
poetry run research datasets get cs2_match_features -o ./cs2_features.parquet
```

---

## Web UI

The controller web UI has three pages under the **Research** section in the sidebar:

- **Sessions** — existing session list and detail view
- **Artifacts** — all artifacts across sessions, filterable by type and name
- **Datasets** — all shared datasets

The session detail page also shows a compact artifact table listing files produced by that session.

---

## Artifact URI Reference

| Format | Meaning |
|--------|---------|
| `artifact://type/name` | Latest version |
| `artifact://type/name:N` | Specific version N |
| `s3://bucket/key` | Direct S3 path |
| `path/to/file.npz` | Local file (backward compatible) |
