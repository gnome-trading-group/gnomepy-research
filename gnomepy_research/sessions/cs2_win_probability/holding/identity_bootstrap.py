"""
Seed the cs2_team_aliases table from history, and measure what id-based resolution recovers.

Walks HLTV BO3 series chronologically (Aug-Sep 2026, Polymarket's CS2 coverage),
resolves each one's series market with identity.resolve_match, learns aliases as
it goes and settles them against HLTV's result - the same loop the live pipeline
runs. Compared with plain normalised-name matching (polymarket.match_market).
"""
import pandas as pd

from gnomepy_research.sessions.cs2_prematch.identity import empty_aliases, learn, resolve_match, settle
from gnomepy_research.sessions.cs2_prematch.polymarket import fetch_markets, match_market
from gnomepy_research.sessions.cs2_win_probability.holding._paths import DATA_DIR

START, END = pd.Timestamp("2026-08-07"), pd.Timestamp("2026-09-30")

history = pd.read_parquet(DATA_DIR / "cs2_match_history.parquet")
history["match_date"] = pd.to_datetime(history.match_date)
series = (history[(history.bo_type == 3) & history.match_time.notna()
                  & (history.match_date >= START) & (history.match_date < END)]
          .groupby("match_id")
          .agg(team_a_id=("team_a_id", "first"), team_a_name=("team_a_name", "first"),
               team_b_id=("team_b_id", "first"), team_b_name=("team_b_name", "first"),
               match_time=("match_time", "first"), wins_a=("team_a_won", "sum"), maps=("team_a_won", "size"))
          .dropna(subset=["team_a_id", "team_b_id"]).sort_values("match_time").reset_index())
series["match_time"] = pd.to_datetime(series.match_time, utc=True)
series["team_a_won"] = (series.wins_a * 2 > series.maps).astype(int)

pm = fetch_markets(START - pd.Timedelta(days=10), END + pd.Timedelta(days=2))
pm.to_parquet(DATA_DIR / "pm_markets_aug_sep_kickoff.parquet", index=False)
print(f"{len(series)} HLTV BO3 series, {len(pm)} Polymarket markets ({pm.kickoff.notna().mean():.0%} with kickoff)")

by_slug = pm.set_index("market_slug")
aliases = empty_aliases()
rows = []
for s in series.itertuples():
    match = s._asdict()
    name_hit = match_market(pm, s.team_a_name, s.team_b_name, s.match_time.date(), "series")[3] is None
    r = resolve_match(match, pm, aliases)
    settled = conflict = None
    if r.market_slug:
        mk = by_slug.loc[r.market_slug]
        fin = [float(x) for x in mk.final] if len(mk.final) == 2 else []
        if fin and max(fin) > 0.9:
            settled = True
            conflict = int(fin[r.team_a_outcome] > 0.5) != s.team_a_won
    aliases = learn(aliases, r.learned, s.match_time)
    if settled:
        for name, team_id in r.learned:
            aliases = settle(aliases, "polymarket", name, team_id, agrees=not conflict)
    rows.append({"match_id": s.match_id, "name_hit": name_hit, "id_hit": r.market_slug is not None,
                 "anchored": bool(r.learned), "reason": r.reason, "conflict": conflict})

R = pd.DataFrame(rows)
print(f"name matching: {R.name_hit.mean():.1%}   id resolution: {R.id_hit.mean():.1%}   "
      f"(+{(R.id_hit & ~R.name_hit).sum()} series, lost {(R.name_hit & ~R.id_hit).sum()})")
print(f"resolved by anchor+kickoff: {R.anchored.sum()},  settlement conflicts: {int(R.conflict.fillna(False).sum())} "
      f"of {R.conflict.notna().sum()} settled")
print("unresolved reasons:", R[~R.id_hit].reason.value_counts().to_dict())
print(f"aliases: {len(aliases)} ({aliases.status.value_counts().to_dict()})")
print(aliases.merge(series[["team_a_id", "team_a_name"]].drop_duplicates("team_a_id"),
                    left_on="hltv_team_id", right_on="team_a_id", how="left")
      [["source_name_norm", "team_a_name", "status", "evidence_count"]].to_string(index=False))
aliases.to_parquet(DATA_DIR / "cs2_team_aliases.parquet", index=False)
R.to_parquet(DATA_DIR / "identity_bootstrap_results.parquet", index=False)
