"""One run: the rating pass over the bundle, series rows for the layers, and a price for every target at its own
week_cutoff from the belief and the layers as they stood that Monday."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.g14.data import FIVE, Inputs, load_inputs, target_market
from gnomepy_research.sessions.cs2_comp.g14.layer import LayerConfig, Layers
from gnomepy_research.sessions.cs2_comp.g14.ratings import Engine, RatingConfig

PATH_COLUMNS = ["m0", "m_any", "lead_last_h", "n_prints", "span_h", "drift24", "drift6", "drift1", "flat_half", "vol6", "prints_1h"]


def _five(v) -> tuple[int, ...] | None:
    if v is None or len(v) != FIVE:
        return None
    five = tuple(sorted(int(p) for p in v))
    return five if len(set(five)) == FIVE else None


@dataclass
class Run:
    series_rows: pd.DataFrame
    target_rows: pd.DataFrame
    predictions: pd.DataFrame
    layers: dict[pd.Timestamp, Layers]


def prepare_targets(targets: pd.DataFrame) -> pd.DataFrame:
    t = targets.copy()
    wc = pd.to_datetime(t.week_cutoff)
    t["week_cutoff"] = wc.dt.tz_convert(None) if wc.dt.tz is not None else wc
    mt = pd.to_datetime(t.match_time)
    t["time"] = mt.dt.tz_convert(None) if mt.dt.tz is not None else mt
    return t


def run(data_dir: str | Path, targets: pd.DataFrame, rcfg: RatingConfig, lcfg: LayerConfig,
        verbose: bool = False, fit_layers: bool = True) -> Run:
    targets = prepare_targets(targets)
    cutoffs = sorted(targets.week_cutoff.unique())
    t0 = time.time()
    inputs = load_inputs(data_dir, max(cutoffs), rcfg.faceit)
    if verbose:
        print(f"loaded {len(inputs.maps)} maps, {len(inputs.series)} series in {time.time() - t0:.0f}s", flush=True)
    tm = target_market(data_dir, targets.match_id)
    targets = targets.merge(tm.drop(columns=["kickoff"]), on="match_id", how="left")
    engine = Engine(inputs, rcfg)
    s = inputs.series
    series_rows: list[dict] = []
    target_rows: list[dict] = []
    checkpoints: dict[pd.Timestamp, int] = {}
    week_targets = {pd.Timestamp(c): targets[targets.week_cutoff == c] for c in cutoffs}
    s_lineup_a = [_five(v) for v in s.lineup_a]
    s_lineup_b = [_five(v) for v in s.lineup_b]
    s_fields = s[["match_id", "time", "known", "team_a", "team_b", "rank_a", "rank_b", "is_lan", "tier", "y", "sub_a", "sub_b"]]
    s_records = s_fields.to_dict("records")

    def on_series(j: int, now: float) -> None:
        rec = s_records[j]
        five_a, five_b = s_lineup_a[j], s_lineup_b[j]
        if five_a is None or five_b is None:
            return
        rec = dict(rec)
        rec.update(engine.price(int(rec["team_a"]), five_a, int(rec["team_b"]), five_b, float(rec["rank_a"]), float(rec["rank_b"]), now))
        series_rows.append(rec)

    def on_cutoff(c: pd.Timestamp) -> None:
        checkpoints[c] = len(series_rows)
        week = week_targets[c]
        for r in week.itertuples(index=False):
            five_a, five_b = _five(r.team_a_lineup_ids), _five(r.team_b_lineup_ids)
            now = float(pd.Timestamp(r.time).value // 10 ** 9)
            rank_a, rank_b = inputs.rank(int(r.team_a_id), c), inputs.rank(int(r.team_b_id), c)
            rec = {"match_id": r.match_id, "week_cutoff": c, "time": r.time, "team_a": r.team_a_id, "team_b": r.team_b_id,
                   "rank_a": rank_a, "rank_b": rank_b, "is_lan": int(r.is_lan), "tier": int(r.event_tier),
                   "sub_a": 0 if r.team_a_standin_ids is None else len(r.team_a_standin_ids),
                   "sub_b": 0 if r.team_b_standin_ids is None else len(r.team_b_standin_ids)}
            for col in PATH_COLUMNS:
                rec[col] = getattr(r, col)
            if five_a is None or five_b is None:
                five_a = five_a or (engine.teams[int(r.team_a_id)].last_five if int(r.team_a_id) in engine.teams else None)
                five_b = five_b or (engine.teams[int(r.team_b_id)].last_five if int(r.team_b_id) in engine.teams else None)
            if five_a is None or five_b is None:
                five_a = five_a or tuple(range(-10 ** 9 - 5, -10 ** 9))
                five_b = five_b or tuple(range(-10 ** 9 - 10, -10 ** 9 - 5))
            rec.update(engine.price(int(r.team_a_id), five_a, int(r.team_b_id), five_b, rank_a, rank_b, now))
            target_rows.append(rec)
        if verbose:
            print(f"{c:%Y-%m-%d} rows {len(series_rows)} targets {len(week)} {time.time() - t0:.0f}s", flush=True)

    engine.run(cutoffs, on_series, on_cutoff)
    sr = pd.DataFrame(series_rows)
    sr["match_id"] = sr.match_id.astype(np.int64)
    mk = inputs.market.copy()
    mk["match_id"] = mk.match_id.astype(np.int64)
    sr = sr.merge(mk[["match_id"] + PATH_COLUMNS], on="match_id", how="left")
    tr = pd.DataFrame(target_rows)
    if not fit_layers:
        return Run(sr, tr, pd.DataFrame(), {})
    layers: dict[pd.Timestamp, Layers] = {}
    preds = []
    for c in cutoffs:
        c = pd.Timestamp(c)
        lay = Layers(sr.iloc[:checkpoints[c]], c, lcfg)
        layers[c] = lay
        rows = tr[tr.week_cutoff == c]
        p = lay.predict(rows)
        p["fit_cutoff"] = c
        preds.append(p)
        if verbose:
            print(f"{c:%Y-%m-%d} layer rows {lay.n_model} market rows {lay.n_market}", flush=True)
    return Run(sr, tr, pd.concat(preds, ignore_index=True), layers)
