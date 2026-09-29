"""
Train the CS2 round win probability model from DatasetStore and publish to ArtifactStore.

Usage:
    poetry run python -m gnomepy_research.sessions.cs2_win_probability.train_model
"""
from __future__ import annotations

import logging
import tempfile

from gnomepy_research.artifacts import ArtifactStore, DatasetStore
from gnomepy_research.sessions.cs2_win_probability.model import train

logger = logging.getLogger(__name__)


def main() -> None:
    ds = DatasetStore()
    df = ds.load("cs2_round_features")
    logger.info("Loaded %d round rows from cs2_round_features", len(df))

    with tempfile.NamedTemporaryFile(suffix=".xgb", delete=False) as f:
        out_path = f.name

    metrics = train(df, out_path)
    logger.info("Training complete: %s", metrics)

    ArtifactStore().publish(
        out_path,
        artifact_type="xgboost_model",
        name="cs2_round_win_prob",
        session_name="cs2_win_probability",
    )
    logger.info("Published cs2_round_win_prob to ArtifactStore")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
