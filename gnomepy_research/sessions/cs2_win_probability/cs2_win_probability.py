from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import pandas as pd

from gnomepy import ExecutionReport, Intent, OrderType, Scales, Side, Strategy
from gnomepy.java.schemas import Schema
from gnomepy.registry import RegistryClient

from gnomepy_research.artifacts import resolve_artifact_path
from gnomepy_research.sessions.cs2_win_probability.features import extract_features
from gnomepy_research.sessions.cs2_win_probability.game_state import CS2GameState
from gnomepy_research.sessions.cs2_win_probability.map_model import (
    map_win_probs_to_series_win_prob,
    round_win_prob_to_map_win_prob,
)
from gnomepy_research.sessions.cs2_win_probability.model import CS2RoundModel

logger = logging.getLogger(__name__)

PRICE_SCALE = Scales.PRICE  # 1_000_000_000
SIZE_SCALE = Scales.SIZE    # 1_000_000


class BookLevel(NamedTuple):
    price: int
    size: int


@dataclass
class Book:
    bids: list[BookLevel] = field(default_factory=list)
    asks: list[BookLevel] = field(default_factory=list)
    last_ts: int = 0

    def best_bid(self) -> int:
        return self.bids[0].price if self.bids else 0

    def best_ask(self) -> int:
        return self.asks[0].price if self.asks else 0

    def spread(self) -> int:
        b, a = self.best_bid(), self.best_ask()
        return (a - b) if (a > 0 and b > 0) else PRICE_SCALE

    def is_ready(self) -> bool:
        return bool(self.bids) and bool(self.asks)


def _read_book(data: Schema, depth: int = 3) -> Book:
    bids = [BookLevel(data.bid_price(i), data.bid_size(i)) for i in range(depth) if data.bid_size(i) > 0]
    asks = [BookLevel(data.ask_price(i), data.ask_size(i)) for i in range(depth) if data.ask_size(i) > 0]
    return Book(bids=bids, asks=asks, last_ts=data.event_timestamp)


class CS2WinProbability(Strategy):
    """
    CS2 win probability strategy.

    Takes a single match (event_id) and a level ("map" or "match").
    Sources a fair value from either:
      - A live GRID WebSocket feed (live trading, requires grid_api_key + grid_series_id)
      - A pre-computed fair value series parquet file (backtesting)

    In both cases, the strategy compares the model's fair price to the current
    Polymarket/Kalshi book and posts maker orders when edge exceeds the threshold.

    Parameters
    ----------
    listing_id_yes : int
        The listing to trade (YES contract — team A wins).
    listing_id_no : int
        The complementary NO contract (team B wins). Used only for position tracking.
    model_path : str
        Artifact URI or local path to the XGBoost model.
        E.g. "artifact://xgboost_model/cs2_round_win_prob"
    level : str
        "map" or "match" — determines which probability output to use.
    ct_team_is_team_a : bool
        Whether the CT-side team is "team A" (the YES contract side).
        Required to map model output (P(CT wins)) to the YES contract price.
    ct_score : int
        Starting CT score on this map (for in-progress games).
    t_score : int
        Starting T score on this map.
    ct_maps_won : int
        Maps won by CT-side team (for series-level probability in "match" mode).
    t_maps_won : int
        Maps won by T-side team.
    maps_to_win : int
        Maps needed to win the series (2 for BO3, 3 for BO5).
    fair_value_series_path : str | None
        Path or DatasetStore name for pre-computed fair values (backtesting).
        Parquet with columns [timestamp_ns, ct_win_prob].
    grid_api_key : str | None
        GRID API key (live trading only).
    grid_series_id : str | None
        GRID series ID for this match (live trading only).
    team_priors : dict | None
        Optional team priors dict (see features.py). Defaults to neutral (0.5).
    edge_threshold : float
        Minimum absolute probability edge to post a maker order (default 0.03).
    half_spread : float
        Half-spread around fair value for maker quotes (default 0.02 = 2%).
    max_position : int
        Maximum net shares held (YES minus NO).
    maker_size : int
        Order size in shares.
    max_book_spread_pct : float
        Skip trading if market spread exceeds this fraction (default 0.10 = 10%).
    """

    def __init__(
        self,
        listing_id_yes: int,
        listing_id_no: int,
        model_path: str,
        level: str = "map",
        ct_team_is_team_a: bool = True,
        ct_score: int = 0,
        t_score: int = 0,
        ct_maps_won: int = 0,
        t_maps_won: int = 0,
        maps_to_win: int = 2,
        fair_value_series_path: str | None = None,
        grid_api_key: str | None = None,
        grid_series_id: str | None = None,
        team_priors: dict | None = None,
        edge_threshold: float = 0.03,
        half_spread: float = 0.02,
        max_position: int = 100,
        maker_size: int = 10,
        max_book_spread_pct: float = 0.10,
        processing_time_ns: int = 5_000_000,
    ):
        assert level in ("map", "match"), "level must be 'map' or 'match'"

        self._level = level
        self._ct_team_is_team_a = ct_team_is_team_a
        self._maps_to_win = maps_to_win
        self._team_priors = team_priors or {}
        self._edge_threshold = edge_threshold
        self._half_spread_int = int(half_spread * PRICE_SCALE)
        self._max_position = max_position
        self._maker_size_int = maker_size * SIZE_SCALE
        self._max_book_spread_int = int(max_book_spread_pct * PRICE_SCALE)
        self._processing_time_ns = processing_time_ns

        resolved = resolve_artifact_path(model_path)
        self._round_model = CS2RoundModel(resolved)

        registry = RegistryClient()
        yes_listings = registry.get_listing(listing_id=listing_id_yes)
        if not yes_listings:
            raise ValueError(f"No listing found for listing_id_yes={listing_id_yes}")
        self._yes_eid: int = yes_listings[0].exchange_id
        self._yes_sid: int = yes_listings[0].security_id

        no_listings = registry.get_listing(listing_id=listing_id_no)
        if not no_listings:
            raise ValueError(f"No listing found for listing_id_no={listing_id_no}")
        self._no_eid: int = no_listings[0].exchange_id
        self._no_sid: int = no_listings[0].security_id

        specs = registry.get_listing_spec(listing_id=listing_id_yes)
        self._lot_size: int = int(specs[0].lot_size) if specs else SIZE_SCALE

        self._game_state = CS2GameState(
            ct_score=ct_score,
            t_score=t_score,
        )
        self._ct_maps_won = ct_maps_won
        self._t_maps_won = t_maps_won
        self._fair_value: float | None = None
        self._fair_value_lock = threading.Lock()

        self._book_yes = Book()
        self._grid_client = None
        self._fair_value_series: pd.DataFrame | None = None

        if fair_value_series_path:
            self._fair_value_series = self._load_fair_value_series(fair_value_series_path)
        elif grid_api_key and grid_series_id:
            self._start_grid_client(grid_api_key, grid_series_id)
        else:
            logger.warning(
                "No GRID feed or fair value series provided — fair value will be None until state updates"
            )

    def _load_fair_value_series(self, path: str) -> pd.DataFrame:
        from gnomepy_research.artifacts import DatasetStore
        if path.endswith(".parquet") or path.startswith("/"):
            df = pd.read_parquet(path)
        else:
            df = DatasetStore().load(path)
        return df.sort_values("timestamp_ns").reset_index(drop=True)

    def _start_grid_client(self, api_key: str, series_id: str) -> None:
        from gnomepy_research.sessions.cs2_win_probability.grid_client import GridClient
        self._grid_client = GridClient(
            api_key=api_key,
            series_id=series_id,
            on_event=self._on_grid_event,
        )
        self._grid_client.start()

    def _on_grid_event(self, event: dict) -> None:
        with self._fair_value_lock:
            self._game_state.update_from_grid_event(event)
            if self._game_state.is_ready():
                self._recompute_fair_value()

    def _recompute_fair_value(self) -> None:
        features = extract_features(self._game_state, self._team_priors)
        p_ct_round = self._round_model.predict_ct_win_prob(features)

        if self._level == "map":
            p_ct_map = round_win_prob_to_map_win_prob(
                p_ct_wins_round=p_ct_round,
                ct_score=self._game_state.ct_score,
                t_score=self._game_state.t_score,
            )
            p_team_a = p_ct_map if self._ct_team_is_team_a else (1.0 - p_ct_map)
        else:
            p_ct_map = round_win_prob_to_map_win_prob(
                p_ct_wins_round=p_ct_round,
                ct_score=self._game_state.ct_score,
                t_score=self._game_state.t_score,
            )
            p_team_a_series = map_win_probs_to_series_win_prob(
                p_ct_wins_map=p_ct_map if self._ct_team_is_team_a else (1.0 - p_ct_map),
                ct_maps_won=self._ct_maps_won,
                t_maps_won=self._t_maps_won,
                maps_to_win=self._maps_to_win,
            )
            p_team_a = p_team_a_series

        self._fair_value = float(np.clip(p_team_a, 0.01, 0.99))

    def _get_fair_value_from_series(self, timestamp_ns: int) -> float | None:
        if self._fair_value_series is None:
            return None
        df = self._fair_value_series
        idx = np.searchsorted(df["timestamp_ns"].values, timestamp_ns, side="right") - 1
        if idx < 0:
            return None
        return float(df.iloc[idx]["ct_win_prob"])

    def simulate_processing_time(self) -> int:
        return self._processing_time_ns

    def register_metrics(self) -> None:
        buf = self.metrics.create_buffer("cs2_signals")
        self._m_ts = buf.add_long_column("timestamp")
        self._m_fv = buf.add_double_column("fair_value")
        self._m_bid = buf.add_double_column("book_bid")
        self._m_ask = buf.add_double_column("book_ask")
        self._m_pos = buf.add_long_column("position")
        buf.freeze()
        self._metrics_buf = buf

    def on_market_data(self, data: Schema) -> list[Intent]:
        if data.exchange_id != self._yes_eid or data.security_id != self._yes_sid:
            return []

        self._book_yes = _read_book(data)
        if not self._book_yes.is_ready():
            return []

        if self._book_yes.spread() > self._max_book_spread_int:
            return []

        if self._fair_value_series is not None:
            fv = self._get_fair_value_from_series(data.event_timestamp)
        else:
            with self._fair_value_lock:
                fv = self._fair_value

        if fv is None or np.isnan(fv):
            return []

        fv_int = int(fv * PRICE_SCALE)
        yes_qty = self.positions.get_effective_quantity(self._yes_eid, self._yes_sid) // self._lot_size
        no_qty = self.positions.get_effective_quantity(self._no_eid, self._no_sid) // self._lot_size
        net_position = yes_qty - no_qty

        bid_price = fv_int - self._half_spread_int
        ask_price = fv_int + self._half_spread_int

        bid_size = self._maker_size_int if net_position < self._max_position else 0
        ask_size = self._maker_size_int if net_position > -self._max_position else 0

        market_mid = (self._book_yes.best_bid() + self._book_yes.best_ask()) // 2
        edge_abs = abs(fv_int - market_mid)
        if edge_abs < int(self._edge_threshold * PRICE_SCALE):
            bid_size = 0
            ask_size = 0

        if self._metrics_buf is not None:
            row = self._metrics_buf.append_row()
            self._metrics_buf.set_long(row, self._m_ts, data.event_timestamp)
            self._metrics_buf.set_double(row, self._m_fv, fv)
            self._metrics_buf.set_double(row, self._m_bid, self._book_yes.best_bid() / PRICE_SCALE)
            self._metrics_buf.set_double(row, self._m_ask, self._book_yes.best_ask() / PRICE_SCALE)
            self._metrics_buf.set_long(row, self._m_pos, net_position)

        if bid_size == 0 and ask_size == 0:
            return []

        return [Intent(
            exchange_id=self._yes_eid,
            security_id=self._yes_sid,
            bid_price=bid_price,
            bid_size=bid_size,
            ask_price=ask_price,
            ask_size=ask_size,
        )]

    def on_execution_report(self, report: ExecutionReport) -> list[Intent]:
        return []
