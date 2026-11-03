"""
The fair price a session trades on: the market's own price blended with e12-wide-mm's.

    fair_A = sigmoid(market * logit(mid_A) + model * logit(q_A) + intercept)

Team A is HLTV's team A, so the intercept is not symmetric in the teams; fair_B = 1 - fair_A. The weights are fitted on
past series' kickoff prices and outcomes and published as an artifact, never refitted inside a session. Probabilities
are clipped to [1e-4, 1 - 1e-4] before the logits, as they were when the weights were fitted: the model's prices reach
well below 1%.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

CLIP = 1e-4


def _logit(p: float) -> float:
    p = min(max(p, CLIP), 1 - CLIP)
    return float(np.log(p / (1 - p)))


@dataclass(frozen=True)
class MarketBlend:
    market: float
    model: float
    intercept: float

    def fair(self, mid: float, q: float) -> float:
        """P(team A wins) from team A's market mid and the model's price for team A."""
        z = self.market * _logit(mid) + self.model * _logit(q) + self.intercept
        return float(1 / (1 + np.exp(-z)))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self)))

    @classmethod
    def load(cls, path: str | Path) -> MarketBlend:
        d = json.loads(Path(path).read_text())
        return cls(float(d["market"]), float(d["model"]), float(d["intercept"]))
