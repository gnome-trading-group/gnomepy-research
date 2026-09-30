"""Tests for artifact reference resolution.

`ArtifactRef.parse` cannot know an artifact's S3 location, so a pinned `artifact://type/name:N`
URI arrives with an empty `s3_uri`. `resolve` only substituted the real ref for unpinned URIs,
so every pinned reference failed on a cold cache — the documented pinning syntax never worked.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from gnomepy_research.artifacts import ArtifactRef, ArtifactStore

REFS = [
    ArtifactRef("value_function", "kalshi_cal", 1, "s3://bucket/v1"),
    ArtifactRef("value_function", "kalshi_cal", 2, "s3://bucket/v2"),
    ArtifactRef("value_function", "kalshi_cal", 3, "s3://bucket/v3"),
]


@pytest.fixture
def cold_cache(tmp_path):
    with patch("gnomepy_research.artifacts._cache_base", return_value=tmp_path):
        yield tmp_path


def _resolve(uri: str) -> str:
    """Resolve a URI, returning the S3 location the downloader was handed."""
    with patch.object(ArtifactStore, "list", return_value=REFS), \
         patch("gnomepy_research.artifacts.resolve_fs", return_value=("fs", "p")) as resolve_fs, \
         patch("gnomepy_research.artifacts.fs_read_bytes", return_value=b"payload"):
        ArtifactStore().resolve(uri)
        return resolve_fs.call_args[0][0]


@pytest.mark.parametrize("version,expected", [(1, "s3://bucket/v1"), (3, "s3://bucket/v3")])
def test_pinned_version_resolves_to_its_own_uri(cold_cache, version, expected):
    assert _resolve(f"artifact://value_function/kalshi_cal:{version}") == expected


def test_unpinned_resolves_to_latest(cold_cache):
    assert _resolve("artifact://value_function/kalshi_cal") == "s3://bucket/v3"


def test_unknown_pinned_version_raises(cold_cache):
    with patch.object(ArtifactStore, "list", return_value=REFS):
        with pytest.raises(KeyError, match="kalshi_cal:9"):
            ArtifactStore().resolve("artifact://value_function/kalshi_cal:9")


def test_cache_hit_skips_the_lookup(cold_cache):
    """A cached artifact must not cost a DynamoDB round trip."""
    cache_dir = cold_cache / "artifacts" / "value_function" / "kalshi_cal" / "3"
    cache_dir.mkdir(parents=True)
    (cache_dir / "artifact.npz").write_bytes(b"cached")
    with patch.object(ArtifactStore, "list") as listing:
        out = ArtifactStore().resolve("artifact://value_function/kalshi_cal:3")
    assert out.endswith("artifact.npz")
    listing.assert_not_called()


def test_local_paths_pass_through(cold_cache):
    assert ArtifactStore().resolve("some/local/file.npz") == "some/local/file.npz"


def test_parse_roundtrip():
    assert str(ArtifactRef.parse("artifact://t/n:7")) == "artifact://t/n:7"
    assert ArtifactRef.parse("artifact://t/n").version == 0
