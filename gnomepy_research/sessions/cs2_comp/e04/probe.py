"""Schema and coverage probe for the bundle: prints what the FACEIT and HLTV joins can rely on."""
import io
import os
from pathlib import Path

import pandas as pd

ROOT = Path("/work/in/comp")
out = io.StringIO()


def say(*args):
    print(*args, file=out)


def show(name: str, df: pd.DataFrame, n: int = 2) -> None:
    say(f"\n=== {name}: {len(df)} rows")
    say(df.dtypes.to_string())
    with pd.option_context("display.width", 250, "display.max_columns", 200, "display.max_colwidth", 80):
        say(df.head(n).T.to_string())


for table in [
    "hltv/cs2_match_history",
    "faceit/faceit_matches",
    "faceit/faceit_match_history",
    "faceit/faceit_lineups",
    "faceit/faceit_hltv_series_link",
    "faceit/hltv_faceit_map",
    "faceit/faceit_exclusions",
    "faceit/faceit_rosters",
]:
    cols = None
    df = pd.read_parquet(ROOT / f"{table}.parquet", columns=cols)
    show(table, df)

exc = pd.read_parquet(ROOT / "faceit/faceit_exclusions.parquet")
say("\nexclusions by reason:\n" + exc.reason.value_counts(dropna=False).to_string())
say("whole-series exclusions:", int(exc.map_name.isna().sum()))

fm = pd.read_parquet(ROOT / "faceit/faceit_matches.parquet")
say("\nfaceit_matches has_stats:", fm.has_stats.value_counts(dropna=False).to_dict())
say("faction1_hltv_team non-null:", float(fm.faction1_hltv_team.notna().mean()))
say("faction1_hltv_team sample:", fm.faction1_hltv_team.dropna().head(5).tolist())
say("best_of:", fm.best_of.value_counts(dropna=False).to_dict())

fh = pd.read_parquet(ROOT / "faceit/faceit_match_history.parquet")
for side in ("a", "b"):
    say(f"team_{side}_id null share:", float(fh[f"team_{side}_id"].isna().mean()))
    say(f"team_{side}_player_ids sample:", fh[f"team_{side}_player_ids"].dropna().head(2).tolist())
    say(f"team_{side}_faceit_ids sample:", fh[f"team_{side}_faceit_ids"].dropna().head(2).tolist())
say("map_name null share:", float(fh.map_name.isna().mean()))
say("bo_type values:", fh.bo_type.value_counts(dropna=False).head(6).to_dict())

hf = pd.read_parquet(ROOT / "faceit/faceit_lineups.parquet")
say("\nfaceit_lineups player_id null share:", float(hf.player_id.isna().mean()))
say("faceit_lineups source values:", hf.source.value_counts(dropna=False).to_dict())

hm = pd.read_parquet(ROOT / "hltv/cs2_match_history.parquet", columns=["match_id", "match_date", "team_a_lineup_ids", "team_a_player_ids", "bo_type"])
say("\nhltv lineup_ids len dist:", hm.team_a_lineup_ids.map(lambda x: len(x) if x is not None else -1).value_counts().to_dict())
say("hltv player_ids len dist:", hm.team_a_player_ids.map(lambda x: len(x) if x is not None else -1).value_counts().to_dict())

Path(os.environ["OUTPUT_DIR"], "probe.txt").write_text(out.getvalue())
print(out.getvalue()[:4000])
