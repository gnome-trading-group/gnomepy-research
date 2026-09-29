"""Weekly HLTV CS2 data collection pipeline — runs on ECS Fargate via gnome-controller.

Params (passed via PIPELINE_PARAMS env var):
  start_date: str  — YYYY-MM-DD
  end_date: str    — YYYY-MM-DD
  min_stars: int   — 0=all matches, 2=top-tier+, 3=big events only (default 2)
"""
from __future__ import annotations

import datetime
import logging

import pandas as pd

from gnomepy_research.artifacts import DatasetStore
from gnomepy_research.pipelines import Pipeline, PipelineResult, register_pipeline
from gnomepy_research.pipelines.hltv_cs2.build_priors import build_priors
from gnomepy_research.pipelines.hltv_cs2.scraper import run_matches, run_rankings

logger = logging.getLogger(__name__)


@register_pipeline
class HltvCs2Pipeline(Pipeline):
    name = "hltv_cs2"

    def run(self, params: dict) -> PipelineResult:
        start_date = datetime.date.fromisoformat(params["start_date"])
        end_date = datetime.date.fromisoformat(params["end_date"])
        min_stars = params.get("min_stars", 2)

        match_df = run_matches(start_date, end_date, min_stars=min_stars)
        rankings_df = run_rankings(start_date, end_date)

        ds = DatasetStore()
        all_matches = ds.load("cs2_match_history")
        all_rankings = ds.load("cs2_team_rankings")
        priors_df = build_priors(all_matches, all_rankings)
        min_date = pd.Timestamp(priors_df["match_date"].min()).date().isoformat()
        max_date = pd.Timestamp(priors_df["match_date"].max()).date().isoformat()
        ds.publish(priors_df, "cs2_match_priors", description=f"{min_date} to {max_date}")

        return PipelineResult(
            status="succeeded",
            outputs={
                "matches": len(match_df),
                "rankings": len(rankings_df),
                "priors": len(priors_df),
            },
        )
