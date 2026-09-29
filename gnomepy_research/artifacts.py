"""Artifact and dataset storage for research sessions.

Artifacts are versioned binary files (models, value functions) stored in S3.
Datasets are versioned tabular DataFrames stored as Parquet in S3.

Both are registered in the gnome-research-sessions DynamoDB table using
reserved partition keys: artifacts under the producing session (or
"__global__"), datasets under "__datasets__".

URI schemes accepted by resolve_artifact_path():
  artifact://type/name       → latest version from artifact store
  artifact://type/name:N     → specific version N
  s3://bucket/key            → direct S3 download
  /local/path or rel/path    → returned as-is
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import boto3
import pandas as pd
from boto3.dynamodb.conditions import Key

from gnomepy._fs import fs_read_bytes, fs_write_bytes, fs_write_parquet, fs_read_parquet, resolve_fs

_TABLE_NAME = "gnome-research-sessions"
_DATASETS_PK = "__datasets__"
_GLOBAL_PK = "__global__"


def _stage() -> str:
    return os.getenv("STAGE", "prod").lower()


def _research_bucket() -> str:
    return os.environ.get("GNOME_RESEARCH_BUCKET", f"gnome-research-{_stage()}")


def _cache_base() -> Path:
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "gnomepy"


def _ddb_table():
    return boto3.resource("dynamodb").Table(_TABLE_NAME)


# ---------------------------------------------------------------------------
# ArtifactRef
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ArtifactRef:
    artifact_type: str
    name: str
    version: int
    s3_uri: str
    session_name: str = "__global__"

    def __str__(self) -> str:
        return f"artifact://{self.artifact_type}/{self.name}:{self.version}"

    @classmethod
    def parse(cls, ref_str: str) -> ArtifactRef:
        """Parse 'artifact://type/name[:version]' into an ArtifactRef (no S3 lookup)."""
        if not ref_str.startswith("artifact://"):
            raise ValueError(f"not an artifact:// reference: {ref_str!r}")
        body = ref_str[len("artifact://"):]
        version = 0
        if ":" in body:
            body, ver_str = body.rsplit(":", 1)
            version = int(ver_str)
        parts = body.split("/", 1)
        if len(parts) != 2:
            raise ValueError(f"expected artifact://type/name[:version], got {ref_str!r}")
        return cls(artifact_type=parts[0], name=parts[1], version=version, s3_uri="")


# ---------------------------------------------------------------------------
# DatasetRef
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetRef:
    dataset_name: str
    version: int
    s3_uri: str

    def __str__(self) -> str:
        return f"dataset://{self.dataset_name}:{self.version}"


# ---------------------------------------------------------------------------
# ArtifactStore
# ---------------------------------------------------------------------------

class ArtifactStore:
    """Upload, download, and resolve versioned ML artifacts in S3."""

    def __init__(self, bucket: str | None = None) -> None:
        self._bucket = bucket or _research_bucket()

    def _s3_prefix(self, artifact_type: str, name: str, version: int) -> str:
        return f"artifacts/{artifact_type}/{name}/{version}"

    def _cache_dir(self, artifact_type: str, name: str, version: int) -> Path:
        return _cache_base() / "artifacts" / artifact_type / name / str(version)

    def _next_version(self, artifact_type: str, name: str) -> int:
        table = _ddb_table()
        sk_prefix = f"ARTIFACT#{artifact_type}#{name}#"
        resp = table.query(
            KeyConditionExpression=Key("session_name").eq(_GLOBAL_PK) & Key("sk").begins_with(sk_prefix),
            ProjectionExpression="sk",
        )
        versions = []
        for item in resp.get("Items", []):
            try:
                versions.append(int(item["sk"].rsplit("#", 1)[-1]))
            except (ValueError, IndexError):
                pass
        # also scan other session partitions for this artifact type/name
        resp2 = table.query(
            IndexName="artifact-type-name-index",
            KeyConditionExpression=Key("artifact_type").eq(artifact_type) & Key("artifact_name").eq(name),
            ProjectionExpression="#v",
            ExpressionAttributeNames={"#v": "version"},
        )
        for item in resp2.get("Items", []):
            try:
                versions.append(int(item["version"]))
            except (ValueError, TypeError):
                pass
        return max(versions, default=0) + 1

    def publish(
        self,
        local_path: str | Path,
        artifact_type: str,
        name: str,
        *,
        session_name: str = "__global__",
        description: str = "",
        params: dict | None = None,
        source_iteration: int | None = None,
    ) -> ArtifactRef:
        """Upload a local file as a new artifact version and register in DynamoDB."""
        local_path = Path(local_path)
        ext = local_path.suffix.lstrip(".")
        version = self._next_version(artifact_type, name)

        prefix = self._s3_prefix(artifact_type, name, version)
        artifact_key = f"{prefix}/artifact.{ext}" if ext else f"{prefix}/artifact"
        s3_uri = f"s3://{self._bucket}/{artifact_key}"

        fs, fs_path = resolve_fs(s3_uri)
        fs_write_bytes(fs, fs_path, local_path.read_bytes())

        now = datetime.now(timezone.utc).isoformat()
        item: dict = {
            "session_name": session_name,
            "sk": f"ARTIFACT#{artifact_type}#{name}#{version}",
            "artifact_type": artifact_type,
            "artifact_name": name,
            "version": version,
            "s3_uri": s3_uri,
            "file_format": ext or "bin",
            "size_bytes": local_path.stat().st_size,
            "created_at": now,
            "description": description,
        }
        if params:
            item["params"] = params
        if source_iteration is not None:
            item["source_iteration"] = source_iteration

        try:
            import subprocess
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            ).decode().strip()
            item["git_commit"] = git_commit
        except Exception:
            pass

        _ddb_table().put_item(Item=item)

        return ArtifactRef(
            artifact_type=artifact_type,
            name=name,
            version=version,
            s3_uri=s3_uri,
            session_name=session_name,
        )

    def resolve(self, ref: str | ArtifactRef) -> str:
        """Resolve an artifact reference to a local cached file path, downloading if needed."""
        if isinstance(ref, str):
            if ref.startswith("artifact://"):
                parsed = ArtifactRef.parse(ref)
                if parsed.version == 0:
                    parsed = self.latest(parsed.artifact_type, parsed.name)
                ref = parsed
            elif ref.startswith("s3://"):
                return self._download_s3(ref)
            else:
                return ref

        cache_dir = self._cache_dir(ref.artifact_type, ref.name, ref.version)
        existing = list(cache_dir.glob("artifact.*")) + list(cache_dir.glob("artifact"))
        if existing:
            return str(existing[0])

        cache_dir.mkdir(parents=True, exist_ok=True)
        fs, fs_path = resolve_fs(ref.s3_uri)
        data = fs_read_bytes(fs, fs_path)
        ext = ref.s3_uri.rsplit(".", 1)[-1] if "." in ref.s3_uri.rsplit("/", 1)[-1] else ""
        local_file = cache_dir / (f"artifact.{ext}" if ext else "artifact")
        local_file.write_bytes(data)
        return str(local_file)

    def _download_s3(self, s3_uri: str) -> str:
        """Download an S3 URI to a temp cache location and return local path."""
        key = s3_uri.split("/", 3)[-1]
        local_path = _cache_base() / "s3" / key
        if local_path.exists():
            return str(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        fs, fs_path = resolve_fs(s3_uri)
        local_path.write_bytes(fs_read_bytes(fs, fs_path))
        return str(local_path)

    def latest(self, artifact_type: str, name: str) -> ArtifactRef:
        """Return the highest-version ArtifactRef for the given type/name."""
        refs = self.list(artifact_type=artifact_type, name=name)
        if not refs:
            raise KeyError(f"no artifact found: {artifact_type}/{name}")
        return max(refs, key=lambda r: r.version)

    def list(
        self,
        artifact_type: str | None = None,
        name: str | None = None,
        session_name: str | None = None,
    ) -> list[ArtifactRef]:
        """List artifacts, optionally filtered by type, name, or session."""
        table = _ddb_table()
        items: list[dict] = []

        if artifact_type and name:
            resp = table.query(
                IndexName="artifact-type-name-index",
                KeyConditionExpression=Key("artifact_type").eq(artifact_type) & Key("artifact_name").eq(name),
            )
            items = resp.get("Items", [])
        elif session_name:
            sk_prefix = "ARTIFACT#"
            if artifact_type:
                sk_prefix += f"{artifact_type}#"
            resp = table.query(
                KeyConditionExpression=Key("session_name").eq(session_name) & Key("sk").begins_with(sk_prefix),
            )
            items = resp.get("Items", [])
        else:
            # scan all artifact items (cross-session)
            from boto3.dynamodb.conditions import Attr
            resp = table.scan(FilterExpression=Attr("sk").begins_with("ARTIFACT#"))
            items = resp.get("Items", [])
            if artifact_type:
                items = [i for i in items if i.get("artifact_type") == artifact_type]

        refs = []
        for item in items:
            try:
                refs.append(ArtifactRef(
                    artifact_type=item["artifact_type"],
                    name=item["artifact_name"],
                    version=int(item["version"]),
                    s3_uri=item["s3_uri"],
                    session_name=item.get("session_name", _GLOBAL_PK),
                ))
            except (KeyError, ValueError):
                pass
        return refs


# ---------------------------------------------------------------------------
# DatasetStore
# ---------------------------------------------------------------------------

class DatasetStore:
    """Publish and load versioned tabular datasets stored as Parquet in S3."""

    def __init__(self, bucket: str | None = None) -> None:
        self._bucket = bucket or _research_bucket()

    def _s3_uri(self, name: str, version: int) -> str:
        return f"s3://{self._bucket}/datasets/{name}/{version}/data.parquet"

    def _cache_path(self, name: str, version: int) -> Path:
        return _cache_base() / "datasets" / name / str(version) / "data.parquet"

    def _next_version(self, name: str) -> int:
        table = _ddb_table()
        sk_prefix = f"DATASET#{name}#"
        resp = table.query(
            KeyConditionExpression=Key("session_name").eq(_DATASETS_PK) & Key("sk").begins_with(sk_prefix),
            ProjectionExpression="#v",
            ExpressionAttributeNames={"#v": "version"},
        )
        versions = [int(item["version"]) for item in resp.get("Items", []) if "version" in item]
        return max(versions, default=0) + 1

    def publish(
        self,
        df: pd.DataFrame,
        name: str,
        *,
        description: str = "",
        producing_session: str | None = None,
    ) -> DatasetRef:
        """Write a DataFrame to S3 as Parquet and register in DynamoDB."""
        version = self._next_version(name)
        s3_uri = self._s3_uri(name, version)

        fs, fs_path = resolve_fs(s3_uri)
        fs_write_parquet(fs, fs_path, df)
        size_bytes = fs.get_file_info(fs_path).size

        now = datetime.now(timezone.utc).isoformat()
        item: dict = {
            "session_name": _DATASETS_PK,
            "sk": f"DATASET#{name}#{version}",
            "dataset_name": name,
            "version": version,
            "s3_uri": s3_uri,
            "file_format": "parquet",
            "size_bytes": size_bytes,
            "row_count": len(df),
            "columns": list(df.columns),
            "column_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "created_at": now,
            "description": description,
        }
        if producing_session:
            item["producing_session"] = producing_session

        _ddb_table().put_item(Item=item)

        return DatasetRef(dataset_name=name, version=version, s3_uri=s3_uri)

    def load(self, ref: str | DatasetRef) -> pd.DataFrame:
        """Load a dataset into a DataFrame, caching locally."""
        if isinstance(ref, str):
            if ":" in ref:
                name, ver_str = ref.rsplit(":", 1)
                version = int(ver_str)
                refs = self.list(name=name)
                matched = [r for r in refs if r.version == version]
                if not matched:
                    raise KeyError(f"dataset not found: {ref}")
                ref = matched[0]
            else:
                ref = self.latest(ref)

        cache_path = self._cache_path(ref.dataset_name, ref.version)
        if cache_path.exists():
            return pd.read_parquet(cache_path)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        fs, fs_path = resolve_fs(ref.s3_uri)
        df = fs_read_parquet(fs, fs_path)
        df.to_parquet(cache_path, index=False)
        return df

    def latest(self, name: str) -> DatasetRef:
        """Return the highest-version DatasetRef for the given name."""
        refs = self.list(name=name)
        if not refs:
            raise KeyError(f"no dataset found: {name}")
        return max(refs, key=lambda r: r.version)

    def list(self, name: str | None = None) -> list[DatasetRef]:
        """List datasets, optionally filtered by name."""
        table = _ddb_table()
        sk_prefix = f"DATASET#{name}#" if name else "DATASET#"
        resp = table.query(
            KeyConditionExpression=Key("session_name").eq(_DATASETS_PK) & Key("sk").begins_with(sk_prefix),
        )
        refs = []
        for item in resp.get("Items", []):
            try:
                refs.append(DatasetRef(
                    dataset_name=item["dataset_name"],
                    version=int(item["version"]),
                    s3_uri=item["s3_uri"],
                ))
            except (KeyError, ValueError):
                pass
        return refs


# ---------------------------------------------------------------------------
# resolve_artifact_path
# ---------------------------------------------------------------------------

def resolve_artifact_path(path: str, *, bucket: str | None = None) -> str:
    """Resolve a path that may be an artifact reference, S3 URI, or local path.

    artifact://type/name        → download latest version, return local cache path
    artifact://type/name:N      → download version N, return local cache path
    s3://bucket/key             → download, return local cache path
    anything else               → returned as-is (local file path)
    """
    if path.startswith("artifact://"):
        return ArtifactStore(bucket=bucket).resolve(path)
    if path.startswith("s3://"):
        return ArtifactStore(bucket=bucket)._download_s3(path)
    return path
