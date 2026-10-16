"""
Polymarket taker fees for the holding scripts: what the registry says each market charged when the trade happened.

The scripts know outcome tokens, not listings. A registry listing's exchange_security_id is
"<conditionId>:<tokenId>", so each token's conditionId comes from Gamma and its listing from one exact registry
lookup. That mapping never changes and takes one request per token, so it is cached next to the data.

Frames passed to ``for_column`` carry ``token_a``, ``token_b``, the price of team A in ``<col>`` and when that
price was taken, in epoch seconds, in ``t_<col>``.
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

from gnomepy.fees import FeeTimeline, load_fee_timelines
from gnomepy.registry import RegistryClient
from gnomepy_research.models.cs2.prematch.polymarket import GAMMA, resolve_listing
from gnomepy_research.sessions.cs2_win_probability.holding._paths import DATA_DIR

LISTINGS = DATA_DIR / "pm_token_listings.parquet"
GAMMA_BATCH = 50
FEE_BATCH = 200
WORKERS = 8
NANOS_PER_SECOND = 1_000_000_000


class TakerFees:
    """Per-share taker fee for buying Polymarket outcome tokens, from each token's registry fee timeline."""

    def __init__(self, tokens, registry: RegistryClient | None = None):
        registry = registry or RegistryClient()
        self._listing = _listing_ids(sorted({str(t) for t in tokens}), registry)
        ids = sorted(set(self._listing.values()))
        self._timelines: dict[int, FeeTimeline] = {}
        for i in range(0, len(ids), FEE_BATCH):
            self._timelines.update(load_fee_timelines(registry, ids[i:i + FEE_BATCH]))

    def per_share(self, token, at_seconds: float, price: float) -> float:
        listing = self._listing[str(token)]
        schedule = self._timelines.get(listing, FeeTimeline()).at(int(round(at_seconds)) * NANOS_PER_SECOND)
        if schedule is None:
            raise ValueError(f"registry has no fee schedule for listing {listing} (token {token}) "
                             f"at {pd.Timestamp(at_seconds, unit='s', tz='UTC')}")
        return schedule.fee_dollars(True, float(price), 1.0, False)

    def for_column(self, frame: pd.DataFrame, col: str) -> tuple[np.ndarray, np.ndarray]:
        """Fees for buying team A at ``frame[col]`` and team B at ``1 - frame[col]``."""
        if frame.empty:
            return np.zeros(0), np.zeros(0)
        rows = zip(frame.token_a, frame.token_b, frame[f"t_{col}"], frame[col])
        fees = [(self.per_share(a, t, p), self.per_share(b, t, 1 - p)) for a, b, t, p in rows]
        return np.array([f[0] for f in fees], dtype=float), np.array([f[1] for f in fees], dtype=float)


def _listing_ids(tokens: list[str], registry: RegistryClient) -> dict[str, int]:
    cached = pd.read_parquet(LISTINGS) if LISTINGS.exists() else pd.DataFrame(columns=["token", "listing_id"])
    found = {str(t): int(i) for t, i in zip(cached.token, cached.listing_id)}
    missing = [t for t in tokens if t not in found]
    if missing:
        conditions = _condition_ids(missing)
        with ThreadPoolExecutor(WORKERS) as pool:
            resolved = pool.map(lambda t: resolve_listing(conditions.get(t), t, registry), missing)
        new = {t: i for t, i in zip(missing, resolved) if i is not None}
        found.update(new)
        pd.DataFrame({"token": list(found), "listing_id": list(found.values())}).to_parquet(LISTINGS, index=False)
    unresolved = [t for t in tokens if t not in found]
    if unresolved:
        raise ValueError(f"{len(unresolved)} of {len(tokens)} Polymarket tokens have no registry listing, "
                         f"so their fees are unknown; first few: {unresolved[:5]}")
    return {t: found[t] for t in tokens}


def _condition_ids(tokens: list[str]) -> dict[str, str]:
    s = requests.Session()
    s.headers["User-Agent"] = "research/1.0"
    conditions: dict[str, str] = {}
    for i in range(0, len(tokens), GAMMA_BATCH):
        batch = tokens[i:i + GAMMA_BATCH]
        # Gamma's /markets filters on closed either way, so a batch mixing settled and open markets needs both.
        for closed in ("true", "false"):
            todo = [t for t in batch if t not in conditions]
            if not todo:
                break
            params = [("clob_token_ids", t) for t in todo] + [("closed", closed), ("limit", len(todo))]
            resp = s.get(f"{GAMMA}/markets", params=params, timeout=30)
            resp.raise_for_status()
            for m in resp.json():
                for t in json.loads(m.get("clobTokenIds") or "[]"):
                    conditions[t] = m["conditionId"]
        time.sleep(0.2)
    return conditions
