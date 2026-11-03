"""g01 refreshed for the confirmation months: the same model as `predict` with the skill-fit constants re-selected on
all data before 2026-08-01 (monthly expanding-window stage 1 over Apr-Jul, one-at-a-time grid, lowest log loss wins)."""
from __future__ import annotations

from gnomepy_research.sessions.cs2_comp.g01 import predict

REFIT_SKILL_KW = {"short_halflife": 60.0, "faceit_weight": 3.0, "halflife": 1500.0}


def main() -> None:
    predict.SKILL_KW = REFIT_SKILL_KW
    predict.main()


if __name__ == "__main__":
    main()
