"""
CS2 series winner, pre-match, on one venue for one HLTV match.

gnome-launcher starts one session per match per venue, so each session sits near its venue
and a venue's trouble stays in its own sessions. The session prices the match itself from each
cs2_upcoming_features row as the pipeline writes it (match_feed.py): the pre-veto model and the
series DP give one series price, the Bayesian series model (its weekly posterior) another from the
two announced lineups, and the two are blended (series_pricing.price_blended).

Inside the entry window - from window_hours before HLTV's latest kickoff until kickoff, while
HLTV still lists the match as upcoming - it buys whichever team's listing is underpriced by at
least min_edge after the venue's taker fee, sized by the Kelly criterion: the holding that maximises
log-wealth growth on kelly_fraction of its share of bankroll_usd, where the bankroll is split evenly
across the venues with a series market for the match, since a session on each takes the same view.
Every book update re-checks the edge and tops the holding up to the Kelly size the current book
justifies, never past max_stake_usd. An order walks the ask book only as deep as every share still
clears the threshold and the Kelly size; it is an IOC limit at the deepest level it costed, so it only
ever takes liquidity it has already priced.
Positions are held to settlement. Trading stops at kickoff (or once HLTV lists the match live), but
the session runs on through the match so its positions keep being marked, and ends itself, keeping
what it holds, once HLTV drops the match from its listing, which it does when the match ends. If
the pipeline stalls and never says, it ends exit_hours_after_kickoff after kickoff.

It does not trade until both lineups are announced: both models price players, and the Bayesian
one has no price without them. Nor does it trade on stale match state: if the newest cs2_upcoming row is
older than max_state_age_minutes, the pipeline has stalled, and a match that has since started
or moved would look upcoming. Nor on a stale posterior: one fitted more than max_posterior_age_days
ago has missed recent results and debuts, and one fitted after the moment being priced (a replay)
would know the future. Once it holds a position it stops adding if a lineup changes, because a
stand-in the model has barely seen is news it cannot price.

Settings come from the six-month walk-forward backtest (spec.yaml). State that must survive a
restart - which side it holds, money committed, average entry - is read from the engine's
positions rather than kept here, so a restarted session resumes instead of buying again.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from gnomepy import ExecutionReport, Intent, OrderStatus, OrderType, Scales, Side, Strategy
from gnomepy.data import ArtifactQuery, ArtifactStore, DatasetStore, resolve_artifact_path
from gnomepy.java.schemas import Schema
from gnomepy.registry import RegistryClient

from gnomepy_research.fees import registry_fee_rates
from gnomepy_research.models.cs2.win_probability.pre_map_model import CS2PreMapModel
from gnomepy_research.models.cs2.hltv.bayes_series import RankStrength, SeriesBlend, SeriesPosterior
from gnomepy_research.models.cs2.win_probability.series_pricing import price_blended
from gnomepy_research.sessions.cs2_series.decision import Level, choose_entry, cost_per_share, kelly_take
from gnomepy_research.sessions.cs2_series.match_feed import FEATURES, MATCHES, MatchFeed, Quote

logger = logging.getLogger(__name__)

PRICE_SCALE = Scales.PRICE
SIZE_SCALE = Scales.SIZE
HOUR_NS = 3_600_000_000_000
TERMINAL = {OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED}
UPCOMING = "upcoming"
MISSING = "missing"
ARTIFACT = "artifact://"
DEFAULT_MODEL = "artifact://xgboost_model/cs2_prematch_preveto"
DEFAULT_POSTERIOR = "artifact://bayes_posterior/cs2_series"
DEFAULT_BLEND = "artifact://blend/cs2_series"
RANKINGS = "cs2_team_rankings"
DAY_NS = 24 * HOUR_NS


@dataclass
class _Leg:
    listing_id: int
    exchange_id: int
    security_id: int
    tick: float
    lot: int
    min_size: int
    asks: list[Level] = field(default_factory=list)
    in_flight: bool = False
    last_limit: float = 0.0


def _fetch(uri: str) -> tuple[Path, str]:
    """A local copy of an artifact URI or path, and the exact version it resolved to (an unpinned URI or an alias
    is resolved first)."""
    if uri.startswith(ARTIFACT):
        store = ArtifactStore()
        ref = store.find(ArtifactQuery.parse(uri))
        return Path(store.resolve(ref)), str(ref)
    return Path(resolve_artifact_path(uri)), uri


@dataclass(frozen=True)
class Models:
    premap: CS2PreMapModel
    posterior: SeriesPosterior
    blend: SeriesBlend
    versions: dict[str, str]


def load_models(model_path: str, posterior_path: str, blend_path: str) -> Models:
    (m, mv), (p, pv), (b, bv) = _fetch(model_path), _fetch(posterior_path), _fetch(blend_path)
    return Models(CS2PreMapModel(m), SeriesPosterior.load(p), SeriesBlend.from_dict(json.loads(b.read_text())),
                  {"premap": mv, "posterior": pv, "blend": bv})


def read_table(source: str) -> pd.DataFrame:
    """A whole DatasetStore table, or a parquet extract of one (for replays)."""
    return pd.read_parquet(source) if source.endswith(".parquet") else DatasetStore().load(source)


class CS2Series(Strategy):
    """
    Args:
        match_id: the HLTV match.
        listing_team_a, listing_team_b: the venue's registry listings for each HLTV team
            winning the series.
        model_path: the pre-veto model, an artifact URI or path; unpinned means the latest.
        posterior_path: the Bayesian series model's posterior (cs2_bayes_fit, weekly), likewise.
        blend_path: the blend weights of the two (JSON), likewise.
        bankroll_usd: the money the strategy sizes its bets against, across every venue it trades the match on.
        max_posterior_age_days: the oldest posterior it trades on, counted from the end of its fit window.
        min_edge: the least edge after the fee worth buying.
        kelly_fraction: the share of the Kelly size to hold; under 1 because the model's win probability is an
            estimate, and betting full Kelly on an estimate over-bets.
        window_hours: how long before kickoff the entry window opens.
        max_stake_usd: most this session may commit, fees included, however large the Kelly size.
        adverse_move: stop adding once the price has fallen this far below our average entry -
            a big pre-kickoff move is usually news the model cannot see.
        min_order_usd: smallest order worth sending.
        max_state_age_minutes: the oldest match state it trades on; the pipeline writes one every 15 minutes.
        reload_minutes: how often to re-read the match's rows (0 = once, for replays).
        near_kickoff_reload_minutes, near_kickoff_hours: re-read this often inside this long before kickoff,
            so HLTV marking the match live stops trading within a minute of the pipeline writing it.
        exit_hours_after_kickoff: when the session ends if the match state never says the match is over.
        features_source, matches_source, rankings_source: DatasetStore tables, or parquet extracts of them.
    """

    def __init__(
        self,
        match_id: int,
        listing_team_a: int,
        listing_team_b: int,
        bankroll_usd: float,
        model_path: str = DEFAULT_MODEL,
        posterior_path: str = DEFAULT_POSTERIOR,
        blend_path: str = DEFAULT_BLEND,
        max_posterior_age_days: float = 14.0,
        min_edge: float = 0.02,
        kelly_fraction: float = 0.5,
        window_hours: float = 12.0,
        max_stake_usd: float = 50.0,
        adverse_move: float = 0.08,
        min_order_usd: float = 1.0,
        book_depth: int = 10,
        processing_time_ns: int = 0,
        max_state_age_minutes: float = 30.0,
        reload_minutes: float = 5.0,
        near_kickoff_reload_minutes: float = 1.0,
        near_kickoff_hours: float = 1.0,
        exit_hours_after_kickoff: float = 8.0,
        features_source: str = FEATURES,
        matches_source: str = MATCHES,
        rankings_source: str = RANKINGS,
    ):
        self.match_id = int(match_id)
        self._fee_rates = registry_fee_rates(self)
        if bankroll_usd <= 0:
            raise ValueError(f"bankroll_usd must be positive, got {bankroll_usd}")
        self.bankroll_usd = bankroll_usd
        self.min_edge = min_edge
        self.kelly_fraction = kelly_fraction
        self.window_ns = int(window_hours * HOUR_NS)
        self.max_stake_usd = max_stake_usd
        self.adverse_move = adverse_move
        self.min_order_usd = min_order_usd
        self.book_depth = book_depth
        self._processing_time_ns = processing_time_ns
        self.max_state_age_ns = int(max_state_age_minutes * 60 * 1e9)
        self.exit_after_kickoff_ns = int(exit_hours_after_kickoff * HOUR_NS)
        self._metrics_buf = None
        self._entry_lineups = None

        registry = RegistryClient()
        self._legs = (self._resolve(registry, listing_team_a), self._resolve(registry, listing_team_b))
        self._side_of = {(leg.exchange_id, leg.security_id): side for side, leg in enumerate(self._legs)}

        models = load_models(model_path, posterior_path, blend_path)
        logger.info("match %d: pricing with %s", self.match_id, models.versions)
        ranks = RankStrength(read_table(rankings_source))
        self._posterior_as_of_ns = pd.Timestamp(models.posterior.as_of).value
        self.max_posterior_age_ns = int(max_posterior_age_days * DAY_NS)
        # Each priced row's components, by the row's generated_at, so an order records what its price was made of.
        self._components: dict[int, tuple[float, float]] = {}

        def pricer(rows: pd.DataFrame) -> list[float]:
            priced = price_blended(models.premap, models.posterior, models.blend, ranks, rows)
            for at, xgb, bayes in zip(rows.generated_at, priced.xgb, priced.bayes):
                self._components[pd.Timestamp(at).value] = (float(xgb), float(bayes))
            return priced.series.tolist()

        self._feed = MatchFeed(self.match_id, pricer, features_source, matches_source, reload_minutes * 60,
                               near_kickoff_reload_minutes * 60, near_kickoff_hours * 3600)

    @staticmethod
    def _resolve(registry, listing_id: int) -> _Leg:
        listings = registry.get_listing(listing_id=int(listing_id))
        if not listings:
            raise ValueError(f"listing {listing_id} not found in registry")
        specs = registry.get_listing_spec(listing_id=int(listing_id))
        spec = specs[0] if specs else None
        tick = int(spec.tick_size) / PRICE_SCALE if spec and spec.tick_size else 0.01
        lot = int(spec.lot_size) if spec and spec.lot_size else 1
        min_size = int(spec.min_size) if spec and spec.min_size else 0
        return _Leg(listing_id=int(listing_id), exchange_id=listings[0].exchange_id,
                    security_id=listings[0].security_id, tick=tick, lot=lot, min_size=min_size)

    def simulate_processing_time(self) -> int:
        return self._processing_time_ns

    def register_metrics(self) -> None:
        buf = self.metrics.create_stream("cs2_series_orders")
        self._m_ts = buf.add_long_column("timestamp")
        self._m_side = buf.add_long_column("side")
        self._m_fair = buf.add_double_column("fair")
        self._m_fair_xgb = buf.add_double_column("fair_xgb")
        self._m_fair_bayes = buf.add_double_column("fair_bayes")
        self._m_ask = buf.add_double_column("best_ask")
        self._m_edge = buf.add_double_column("edge_at_best")
        self._m_limit = buf.add_double_column("limit_price")
        self._m_shares = buf.add_double_column("shares")
        self._m_cost = buf.add_double_column("cost")
        self._m_kelly_bankroll = buf.add_double_column("kelly_bankroll")
        self._m_held_shares = buf.add_double_column("held_shares")
        buf.freeze()
        self._metrics_buf = buf

    def _taker_rate(self, leg: _Leg) -> float:
        """The leg's taker rate in the registry; the fee per share is rate * p * (1 - p)."""
        return self._fee_rates((leg.exchange_id, leg.security_id))[0]

    # ---- state read from the engine's positions --------------------------------------

    def _leg_state(self, leg: _Leg) -> tuple[float, float, float]:
        """(filled shares, committed USD incl. fees and pending, average entry price)."""
        pos = self.positions.get_position(leg.exchange_id, leg.security_id)
        filled = pos.net_quantity / SIZE_SCALE if pos else 0.0
        avg = pos.avg_entry_price / PRICE_SCALE if pos and pos.net_quantity else 0.0
        fees = float(pos.total_fees) if pos else 0.0
        pending = max(0.0, self.positions.get_effective_quantity(leg.exchange_id, leg.security_id) / SIZE_SCALE - filled)
        return filled, filled * avg + fees + pending * cost_per_share(leg.last_limit, self._taker_rate(leg)), avg

    # ---- engine callbacks ----------------------------------------------------------------

    def on_market_data(self, data: Schema) -> list[Intent]:
        side = self._side_of.get((data.exchange_id, data.security_id))
        if side is None:
            return []
        self._legs[side].asks = [Level(data.ask_price(i) / PRICE_SCALE, data.ask_size(i) / SIZE_SCALE)
                                 for i in range(self.book_depth) if data.ask_size(i) > 0]
        return self._evaluate(data.event_timestamp)

    def _evaluate(self, now_ns: int) -> list[Intent]:
        state, quote = self._feed.state(now_ns), self._feed.quote(now_ns)
        if state is not None and (state.status == MISSING or now_ns >= state.kickoff_ns + self.exit_after_kickoff_ns):
            # Judged on the last state even when it's stale: a stalled pipeline mustn't keep the session up forever.
            self.request_exit(f"match {self.match_id} is {state.status}" if state.status == MISSING
                              else f"match {self.match_id} not over {self.exit_after_kickoff_ns / HOUR_NS:g}h after kickoff")
            return []
        if state is None or quote is None or state.status != UPCOMING:
            return []
        if now_ns - state.at_ns > self.max_state_age_ns:
            return []
        if not (state.kickoff_ns - self.window_ns <= now_ns < state.kickoff_ns):
            return []
        if not quote.lineup_known or any(leg.in_flight for leg in self._legs):
            return []
        if not 0 <= now_ns - self._posterior_as_of_ns <= self.max_posterior_age_ns:
            return []
        if math.isnan(quote.p_team_a):
            # Unpriced: a lineup that isn't a full five gives the Bayesian model nothing to price.
            return []

        states = [self._leg_state(leg) for leg in self._legs]
        held = [side for side, (filled, committed, _) in enumerate(states) if filled > 0 or committed > 0]
        if held and not self._lineups_unchanged(quote):
            return []
        committed = sum(s[1] for s in states)
        if self.max_stake_usd - committed < self.min_order_usd:
            return []

        fee_rates = (self._taker_rate(self._legs[0]), self._taker_rate(self._legs[1]))
        entry = choose_entry(quote.p_team_a, self._legs[0].asks, self._legs[1].asks,
                             min_edge=self.min_edge, fee_rates=fee_rates,
                             tick=self._legs[0].tick, allowed_sides=tuple(held) if held else (0, 1))
        if entry is None:
            return []
        held_shares, held_cost, avg_entry = states[entry.side]
        if avg_entry > 0 and entry.best_ask <= avg_entry - self.adverse_move:
            return []

        leg = self._legs[entry.side]
        kelly_bankroll = self.kelly_fraction * self.bankroll_usd / state.series_venues
        take = kelly_take(entry, kelly_bankroll, held_shares, held_cost, self.max_stake_usd - (committed - held_cost),
                          fee_rates[entry.side], leg.lot / SIZE_SCALE)
        size = round(take.shares * SIZE_SCALE)
        if size <= 0 or size < leg.min_size or take.cost < self.min_order_usd:
            return []

        if not held:
            self._entry_lineups = quote.lineups
        leg.in_flight = True
        leg.last_limit = take.limit_price
        self._record(now_ns, entry, take, self._components.get(quote.at_ns, (float("nan"), float("nan"))),
                     kelly_bankroll, held_shares)
        return [Intent(leg.exchange_id, leg.security_id, take_side=Side.BID, take_size=size,
                       take_order_type=OrderType.LIMIT, take_limit_price=round(take.limit_price * PRICE_SCALE))]

    def _lineups_unchanged(self, quote: Quote) -> bool:
        """Whether the lineups still match the ones the position was entered on."""
        if self._entry_lineups is None:
            # Restarted with a position: what the earlier session entered on isn't stored, so
            # the lineups at restart become the reference.
            self._entry_lineups = quote.lineups
        return quote.lineups == self._entry_lineups

    def on_execution_report(self, report: ExecutionReport) -> list[Intent]:
        side = self._side_of.get((report.exchange_id, report.security_id))
        if side is not None and report.order_status in TERMINAL:
            self._legs[side].in_flight = False
        return []

    def _record(self, now_ns: int, entry, take, components: tuple[float, float], kelly_bankroll: float,
                held_shares: float) -> None:
        if self._metrics_buf is None:
            return
        buf = self._metrics_buf
        buf.begin()
        buf.set_long(self._m_ts, now_ns)
        buf.set_long(self._m_side, entry.side)
        buf.set_double(self._m_fair, entry.fair)
        buf.set_double(self._m_fair_xgb, components[0])
        buf.set_double(self._m_fair_bayes, components[1])
        buf.set_double(self._m_ask, entry.best_ask)
        buf.set_double(self._m_edge, entry.edge_at_best)
        buf.set_double(self._m_limit, take.limit_price)
        buf.set_double(self._m_shares, take.shares)
        buf.set_double(self._m_cost, take.cost)
        buf.set_double(self._m_kelly_bankroll, kelly_bankroll)
        buf.set_double(self._m_held_shares, held_shares)
        buf.commit()
