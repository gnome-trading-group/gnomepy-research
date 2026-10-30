from __future__ import annotations

import argparse
import math

import pandas as pd

from gnomepy_research.sessions.cs2_comp.e04.events import announced_lineups, faceit_map_table, hltv_maps
from gnomepy_research.sessions.cs2_comp.e04.ratings import Ladder, Params, series_probability

DAY_SECONDS = 86400


def day_index(ts: pd.Timestamp) -> int:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return int(ts.value // (DAY_SECONDS * 10**9))


def build_stream(data_dir: str, before: pd.Timestamp, faceit: bool) -> pd.DataFrame:
    parts = [hltv_maps(data_dir, before)]
    if faceit:
        parts.append(faceit_map_table(data_dir, before))
    ev = pd.concat(parts, ignore_index=True)
    return ev.sort_values(["event_day", "faceit", "match_key", "map_pos"], kind="mergesort").reset_index(drop=True)


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def predict(data_dir: str, targets: pd.DataFrame, faceit: bool, params: Params) -> pd.DataFrame:
    detail = predict_detail(data_dir, targets, faceit, params)
    return detail[["match_id", "q", "fit_cutoff"]]


def predict_detail(data_dir: str, targets: pd.DataFrame, faceit: bool, params: Params) -> pd.DataFrame:
    targets = targets.copy()
    targets["week_cutoff"] = pd.to_datetime(targets["week_cutoff"], utc=True).dt.tz_convert(None)
    stream = build_stream(data_dir, targets.week_cutoff.max(), faceit)
    lineups = announced_lineups(data_dir).set_index("match_id")
    ladder = Ladder(params)

    ev_day = stream.event_day.to_numpy()
    ev_key_a, ev_key_b = stream.key_a.to_numpy(), stream.key_b.to_numpy()
    ev_lineup_a, ev_lineup_b = stream.lineup_a.to_numpy(), stream.lineup_b.to_numpy()
    ev_share_a, ev_faceit = stream.share_a.to_numpy(), stream.faceit.to_numpy()
    pos = 0
    faceit_seen: set[str] = set()
    detail_rows: dict = {}
    for cutoff, grp in targets.groupby("week_cutoff", sort=True):
        cut_day = day_index(cutoff)
        while pos < len(stream) and ev_day[pos] < cut_day:
            ladder.update_map(ev_key_a[pos], list(ev_lineup_a[pos]), ev_key_b[pos], list(ev_lineup_b[pos]),
                              float(ev_share_a[pos]), bool(ev_faceit[pos]))
            if ev_faceit[pos]:
                faceit_seen.update(ev_lineup_a[pos])
                faceit_seen.update(ev_lineup_b[pos])
            pos += 1
        for row in grp.itertuples():
            lu = lineups.loc[row.match_id] if row.match_id in lineups.index else None
            lineup_a = list(lu.lineup_a) if lu is not None else []
            lineup_b = list(lu.lineup_b) if lu is not None else []
            key_a, key_b = f"h{int(row.team_a_id)}", f"h{int(row.team_b_id)}"
            p_map, _, _ = ladder.price_map(key_a, lineup_a, key_b, lineup_b, faceit_map=False)
            p_series = series_probability(p_map, int(row.bo_type) if pd.notna(row.bo_type) else 3)
            p_series = min(max(p_series, 1e-4), 1.0 - 1e-4)
            q = _sigmoid(params.calibration * _logit(p_series))
            detail_rows[row.match_id] = {
                "q": q,
                "fit_cutoff": cutoff,
                "faceit_players_a": sum(1 for k in lineup_a if k in faceit_seen),
                "faceit_players_b": sum(1 for k in lineup_b if k in faceit_seen),
                "lineup_a_size": len(lineup_a),
                "lineup_b_size": len(lineup_b),
            }

    rows = [detail_rows[m] for m in targets.match_id.values]
    out = pd.DataFrame(rows, index=range(len(rows)))
    out.insert(0, "match_id", targets.match_id.values)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-faceit", action="store_true")
    args = ap.parse_args()
    targets = pd.read_parquet(args.targets)
    result = predict(args.data, targets, faceit=not args.no_faceit, params=Params())
    result.to_parquet(args.out, index=False)
    print(f"wrote {len(result)} predictions to {args.out}")


if __name__ == "__main__":
    main()
