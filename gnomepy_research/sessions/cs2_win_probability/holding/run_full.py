"""Full-corpus pipeline: priors -> ablation -> Polymarket gate."""
import logging
import pathlib
import subprocess
import sys

import pandas as pd

from gnomepy_research.pipelines.hltv_cs2.build_priors import build_priors
from gnomepy_research.sessions.cs2_win_probability.holding._paths import DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
D = str(DATA_DIR)

history = pd.read_parquet(f"{D}/cs2_match_history.parquet")
history["match_date"] = pd.to_datetime(history["match_date"])
print(f"history: {len(history)} rows, {history.match_id.nunique()} series, "
      f"{history.match_date.min().date()} -> {history.match_date.max().date()}", flush=True)

priors = build_priors(
    history,
    pd.read_parquet(f"{D}/cs2_team_rankings.parquet"),
    player_stats=pd.read_parquet(f"{D}/cs2_player_map_stats.parquet"),
    veto=pd.read_parquet(f"{D}/cs2_match_veto.parquet"),
    h2h=pd.read_parquet(f"{D}/cs2_h2h_history.parquet"),
)
priors.to_parquet(f"{D}/priors_full.parquet")
print(f"priors: {priors.shape}\n", flush=True)

print("=" * 90, "\nABLATION (full corpus, 5 column orders per variant)\n", "=" * 90, flush=True)
subprocess.run([sys.executable, "-m", "gnomepy_research.sessions.cs2_win_probability.ablate",
                "--priors", f"{D}/priors_full.parquet",
                "--history", f"{D}/cs2_match_history.parquet",
                "--n-boot", "500", "--n-repeats", "5"], check=False)

print("\n" + "=" * 90, "\nPOLYMARKET GATE\n", "=" * 90, flush=True)
subprocess.run([sys.executable, str(pathlib.Path(__file__).with_name("pm_final.py"))], check=False)
