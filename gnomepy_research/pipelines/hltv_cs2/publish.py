"""
Publishing for the HLTV datasets, shared by the local backfill and the production pipeline.

One path on purpose: the production pipeline once published only match history,
so its priors were rebuilt without the harvested datasets and player stats, veto,
head-to-head and form went stale. Both callers now publish through here.
"""
from __future__ import annotations

import logging

import pandas as pd

from gnomepy_research.artifacts import DatasetStore
from gnomepy_research.pipelines.hltv_cs2.sides import attach_team_keyed_scores

logger = logging.getLogger(__name__)


def merge_publish(name: str, new_df: pd.DataFrame, key_cols: list[str], date_col: str) -> None:
    """Key-based upsert: replace all rows whose key_cols values appear in new_df, keep everything else."""
    if new_df.empty:
        logger.warning("merge_publish called with empty DataFrame — %s not updated", name)
        return
    ds = DatasetStore()
    try:
        existing = ds.load(name)
        new_keys = new_df[key_cols].drop_duplicates()
        indicator = existing.merge(new_keys, on=key_cols, how="left", indicator=True)
        keep = existing[indicator["_merge"] == "left_only"]
        merged = pd.concat([keep, new_df], ignore_index=True).sort_values(date_col).reset_index(drop=True)
    except KeyError:
        merged = new_df
    min_date = pd.Timestamp(merged[date_col].min()).date().isoformat()
    max_date = pd.Timestamp(merged[date_col].max()).date().isoformat()
    ds.publish(merged, name, description=f"{min_date} to {max_date}")
    logger.info("Published %s: %d rows (%s to %s)", name, len(merged), min_date, max_date)


def publish_match_data(match_rows: list[dict], demo_rows: list[dict], player_rows: list[dict] | None = None, veto_rows: list[dict] | None = None,
                        h2h_rows: list[dict] | None = None,
                        form_rows: list[dict] | None = None) -> None:
    if match_rows:
        df = pd.DataFrame(match_rows)
        df["match_date"] = pd.to_datetime(df["match_date"])
        logger.info("Publishing cs2_match_history (%d rows)...", len(df))
        merge_publish("cs2_match_history", df, ["match_id", "map_name"], "match_date")
    else:
        logger.warning("No match rows — cs2_match_history not updated")

    if player_rows:
        df = pd.DataFrame(player_rows)
        df["match_date"] = pd.to_datetime(df["match_date"])
        logger.info("Publishing cs2_player_map_stats (%d rows)...", len(df))
        merge_publish("cs2_player_map_stats", df,
                      ["match_id", "map_name", "player_id", "side"], "match_date")

    if veto_rows:
        df = pd.DataFrame(veto_rows)
        df["match_date"] = pd.to_datetime(df["match_date"])
        logger.info("Publishing cs2_match_veto (%d rows)...", len(df))
        merge_publish("cs2_match_veto", df, ["match_id", "order"], "match_date")

    if h2h_rows:
        df = pd.DataFrame(h2h_rows)
        df["match_date"] = pd.to_datetime(df["match_date"])
        logger.info("Publishing cs2_h2h_history (%d rows)...", len(df))
        merge_publish("cs2_h2h_history", df, ["match_id", "h2h_date", "map_name"], "match_date")

    if form_rows:
        df = pd.DataFrame(form_rows)
        df["match_date"] = pd.to_datetime(df["match_date"])
        logger.info("Publishing cs2_team_recent_form (%d rows)...", len(df))
        merge_publish("cs2_team_recent_form", df, ["match_id", "team_index", "position"], "match_date")

    if demo_rows:
        df = pd.DataFrame(demo_rows)
        df["match_date"] = pd.to_datetime(df["match_date"])
        if match_rows:
            history = pd.DataFrame(match_rows)
            history["match_date"] = pd.to_datetime(history["match_date"])
        else:
            history = DatasetStore().load("cs2_match_history")
        df, dropped = attach_team_keyed_scores(df, history)
        if len(dropped):
            logger.warning(
                "dropped %d map(s) that could not be reconciled with HLTV scores: %s",
                len(dropped), dropped.reason.value_counts().to_dict(),
            )
        if df.empty:
            logger.warning("No reconcilable round rows — cs2_round_features not updated")
            return
        logger.info("Publishing cs2_round_features (%d rows)...", len(df))
        merge_publish("cs2_round_features", df, ["match_id", "map_name"], "match_date")
