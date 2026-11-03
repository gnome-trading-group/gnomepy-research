"""One run: the rating pass over the bundle, series rows for the layers, and a price for every target at its own
week_cutoff from the belief and the layers as they stood that Monday."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.g14.data import FIVE, Inputs, load_inputs, logit, target_market
from gnomepy_research.sessions.cs2_comp.g14.layer import LayerConfig, Layers, fit_ridge_logistic, market_design, model_column
from gnomepy_research.sessions.cs2_comp.g14.ratings import Engine, RatingConfig, bo3, map_win
from gnomepy_research.sessions.cs2_comp.g14.skillfit import FitConfig, SkillFit, fit

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


def fit_features(sf: SkillFit, team_a: int, five_a: tuple[int, ...], rank_a: float,
                 team_b: int, five_b: tuple[int, ...], rank_b: float) -> dict:
    """The batch fit's view of a series: skill gap of the announced fives, team offsets, who the fit barely knows."""
    ma, thin_a, spread_a = sf.lineup(five_a, rank_a)
    mb, thin_b, spread_b = sf.lineup(five_b, rank_b)
    oa, ob = sf.team(team_a), sf.team(team_b)
    d = ma + oa - mb - ob
    p = bo3(map_win(d))
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    return {"d_fit": d, "z_fit": float(np.log(p / (1 - p))), "off_fit": oa - ob, "thin_fit_a": thin_a, "thin_fit_b": thin_b,
            "spread_fit_a": spread_a, "spread_fit_b": spread_b}


def add_fit_features(rows: pd.DataFrame, sf: SkillFit, fives: list[tuple[tuple[int, ...], tuple[int, ...]]]) -> pd.DataFrame:
    feats = [fit_features(sf, int(r.team_a), fa, float(r.rank_a), int(r.team_b), fb, float(r.rank_b))
             for r, (fa, fb) in zip(rows.itertuples(index=False), fives)]
    return pd.concat([rows.reset_index(drop=True), pd.DataFrame(feats)], axis=1)


def prepare_targets(targets: pd.DataFrame) -> pd.DataFrame:
    t = targets.copy()
    wc = pd.to_datetime(t.week_cutoff)
    t["week_cutoff"] = wc.dt.tz_convert(None) if wc.dt.tz is not None else wc
    mt = pd.to_datetime(t.match_time)
    t["time"] = mt.dt.tz_convert(None) if mt.dt.tz is not None else mt
    return t


def online_pass(data_dir: str | Path, inputs: Inputs, targets: pd.DataFrame, cutoffs: list, rcfg: RatingConfig,
                verbose: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Series rows (online features as each series started, with its announced fives) and target rows (online
    features at the week cutoff), from one walk over the maps."""
    t0 = time.time()
    tm = target_market(data_dir, targets.match_id)
    targets = targets.merge(tm.drop(columns=["kickoff"]), on="match_id", how="left")
    engine = Engine(inputs, rcfg)
    s = inputs.series
    series_rows: list[dict] = []
    target_rows: list[dict] = []
    target_fives_a: list[tuple[int, ...]] = []
    target_fives_b: list[tuple[int, ...]] = []
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
        rec["five_a"], rec["five_b"] = list(five_a), list(five_b)
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
            target_fives_a.append(five_a)
            target_fives_b.append(five_b)
        if verbose:
            print(f"{c:%Y-%m-%d} rows {len(series_rows)} targets {len(week)} {time.time() - t0:.0f}s", flush=True)

    engine.run(cutoffs, on_series, on_cutoff)
    sr = pd.DataFrame(series_rows)
    sr["match_id"] = sr.match_id.astype(np.int64)
    mk = inputs.market.copy()
    mk["match_id"] = mk.match_id.astype(np.int64)
    sr = sr.merge(mk[["match_id"] + PATH_COLUMNS], on="match_id", how="left")
    tr = pd.DataFrame(target_rows)
    tr["five_a"] = [v for v in target_fives_a]
    tr["five_b"] = [v for v in target_fives_b]
    return sr, tr, checkpoints


def run(data_dir: str | Path, targets: pd.DataFrame, rcfg: RatingConfig, lcfg: LayerConfig,
        verbose: bool = False, fit_layers: bool = True, inputs: Inputs | None = None, fcfg: FitConfig | None = None,
        weights: dict[str, float] | None = None, train_rows: pd.DataFrame | None = None) -> Run:
    targets = prepare_targets(targets)
    cutoffs = sorted(targets.week_cutoff.unique())
    t0 = time.time()
    if inputs is None:
        inputs = load_inputs(data_dir, max(cutoffs), rcfg.faceit)
        if verbose:
            print(f"loaded {len(inputs.maps)} maps, {len(inputs.series)} series in {time.time() - t0:.0f}s", flush=True)
    sr, tr, checkpoints = online_pass(data_dir, inputs, targets, cutoffs, rcfg, verbose)
    if fcfg is not None:
        parts = []
        for c in cutoffs:
            c = pd.Timestamp(c)
            t0 = time.time()
            sf = fit(inputs, c, fcfg)
            week = tr[tr.week_cutoff == c]
            parts.append(add_fit_features(week, sf, list(zip(week.five_a, week.five_b))))
            if verbose:
                print(f"{c:%Y-%m-%d} fit {sf.info} {time.time() - t0:.0f}s", flush=True)
        tr = pd.concat(parts, ignore_index=True)
    tr = tr.drop(columns=["five_a", "five_b"])
    if not fit_layers:
        return Run(sr, tr, pd.DataFrame(), {})
    if weights is not None:
        z = np.column_stack([model_column(tr, f) for f in weights]) @ np.array(list(weights.values()))
        q = 1 / (1 + np.exp(-z))
        preds = pd.DataFrame({"match_id": tr.match_id.to_numpy(), "q_model": q, "q": q, "z_model": z, "fit_cutoff": tr.week_cutoff.to_numpy()})
        return Run(sr, tr, preds, {})
    layers: dict[pd.Timestamp, Layers] = {}
    preds = []
    source = sr if train_rows is None else train_rows
    for c in cutoffs:
        c = pd.Timestamp(c)
        lay = Layers(source if train_rows is not None else sr.iloc[:checkpoints[c]], c, lcfg)
        layers[c] = lay
        rows = tr[tr.week_cutoff == c]
        p = lay.predict(rows)
        p["fit_cutoff"] = c
        preds.append(p)
        if verbose:
            print(f"{c:%Y-%m-%d} layer rows {lay.n_model} market rows {lay.n_market}", flush=True)
    return Run(sr, tr, pd.concat(preds, ignore_index=True), layers)


def market_cutoffs(inputs: Inputs, until: pd.Timestamp) -> list[pd.Timestamp]:
    """The first Monday of each month from the month the market data starts, up to `until`: the batch fits whose
    prices the market layer learns from."""
    if inputs.market.empty:
        return []
    first = pd.Timestamp(inputs.market.kickoff.min()).normalize().replace(day=1)
    out = []
    month = first
    while month <= until:
        monday = month + pd.Timedelta(days=(7 - month.weekday()) % 7)
        if monday <= until:
            out.append(monday)
        month = (month + pd.Timedelta(days=32)).replace(day=1)
    return out


def predict_weeks(data_dir: str | Path, targets: pd.DataFrame, rcfg: RatingConfig, fcfg: FitConfig, lcfg: LayerConfig,
                  weights: dict[str, float], verbose: bool = False, market_layer: bool = True) -> Run:
    """The delivered flow: per week, the batch fit at the cutoff prices the targets, the frozen weights give the
    model's price, and a market layer fitted on the market-era series (priced by the monthly fits before the cutoff)
    adjusts it where the target has a kickoff price."""
    targets = prepare_targets(targets)
    weeks = sorted(pd.Timestamp(c) for c in targets.week_cutoff.unique())
    t0 = time.time()
    inputs = load_inputs(data_dir, max(weeks), rcfg.faceit)
    if verbose:
        print(f"loaded {len(inputs.maps)} maps, {len(inputs.series)} series in {time.time() - t0:.0f}s", flush=True)
    months = market_cutoffs(inputs, max(weeks)) if market_layer else []
    sr, tr, _ = online_pass(data_dir, inputs, targets, sorted(set(weeks) | set(months)), rcfg, verbose)
    sr["time"] = pd.to_datetime(sr.time)
    fits: dict[pd.Timestamp, SkillFit] = {}
    preds = []
    weeks_out = []
    names = list(weights)
    beta = np.array([weights[k] for k in names])
    for c in weeks:
        t0 = time.time()
        sf = fit(inputs, c, fcfg)
        week = tr[tr.week_cutoff == c]
        week = add_fit_features(week, sf, list(zip(week.five_a, week.five_b))).drop(columns=["five_a", "five_b"])
        z = np.column_stack([model_column(week, f) for f in names]) @ beta
        q_model = 1 / (1 + np.exp(-z))
        q = q_model.copy()
        n_market = 0
        if market_layer:
            parts = []
            past = [m for m in months if m <= c]
            for m, nxt in zip(past, past[1:] + [c]):
                if m not in fits:
                    fits[m] = fit(inputs, m, fcfg)
                rows = sr[(sr.time >= m) & (sr.time < nxt)]
                if len(rows):
                    parts.append(add_fit_features(rows, fits[m], [(tuple(a), tuple(b)) for a, b in zip(rows.five_a, rows.five_b)]))
            if parts:
                train = pd.concat(parts, ignore_index=True)
                train = train[(pd.to_datetime(train.known) < c) & train.m0.notna()]
                n_market = len(train)
                if n_market >= lcfg.min_market_rows:
                    zt = np.column_stack([model_column(train, f) for f in names]) @ beta
                    Xm = market_design(train, zt, lcfg.market_features)
                    scale = np.maximum(Xm.std(axis=0), 1e-6)
                    scale[-1] = 1.0
                    bm = fit_ridge_logistic(Xm / scale, train.y.to_numpy(float), np.ones(len(train)), lcfg.l2_market,
                                            offset=logit(train.m0.to_numpy())) / scale
                    has = week.m0.notna().to_numpy()
                    if has.any():
                        eta = logit(week.m0.to_numpy()[has]) + market_design(week[has], z[has], lcfg.market_features) @ bm
                        q[has] = 1 / (1 + np.exp(-eta))
        preds.append(pd.DataFrame({"match_id": week.match_id.to_numpy(), "q_model": q_model, "q": q, "z_model": z, "fit_cutoff": c}))
        weeks_out.append(week)
        if verbose:
            print(f"{c:%Y-%m-%d} fit {sf.info['maps']} maps {sf.info['players']} players, market rows {n_market}, "
                  f"{len(week)} targets, {time.time() - t0:.0f}s", flush=True)
    return Run(sr, pd.concat(weeks_out, ignore_index=True), pd.concat(preds, ignore_index=True), {})
