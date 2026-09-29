from __future__ import annotations

import json
import logging
import threading
from typing import Callable

import websocket

logger = logging.getLogger(__name__)

_SERIES_EVENTS_URL = "wss://api.grid.gg/live/series-events/v1"
_SERIES_STATE_URL = "https://api.grid.gg/series-state/graphql"

_SERIES_STATE_QUERY = """
query SeriesState($seriesId: ID!) {
  series(id: $seriesId) {
    id
    title
    maps {
      number
      currentMap
      name
      teams {
        id
        name
        side
        score
        players {
          id
          health
          armor
          money
          equipment_value
          is_alive
        }
      }
      round {
        number
        phase
        bomb {
          state
          countdown
        }
        teams {
          id
          score
          consecutive_losses
        }
      }
    }
  }
}
"""


class GridClient:
    """
    Connects to GRID's Series Events WebSocket for a single CS2 series.
    Calls `on_event(event_dict)` for each incoming game event.

    Usage:
        client = GridClient(api_key="...", series_id="abc-123", on_event=handler)
        client.start()
        # ... strategy runs ...
        client.stop()
    """

    def __init__(
        self,
        api_key: str,
        series_id: str,
        on_event: Callable[[dict], None],
    ):
        self._api_key = api_key
        self._series_id = series_id
        self._on_event = on_event
        self._ws: websocket.WebSocketApp | None = None
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run_ws, daemon=True, name="grid-ws")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._ws:
            self._ws.close()

    def _run_ws(self) -> None:
        url = f"{_SERIES_EVENTS_URL}?seriesId={self._series_id}"
        headers = {"Authorization": f"Bearer {self._api_key}"}
        self._ws = websocket.WebSocketApp(
            url,
            header=headers,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
            on_open=self._on_open,
        )
        while self._running:
            try:
                self._ws.run_forever(ping_interval=30, ping_timeout=10)
            except Exception as exc:
                logger.error("GRID WebSocket error: %s", exc)
            if self._running:
                logger.warning("GRID WebSocket disconnected — reconnecting in 2s")
                import time
                time.sleep(2)

    def _on_open(self, ws) -> None:
        logger.info("GRID WebSocket connected for series %s", self._series_id)

    def _on_message(self, ws, raw: str) -> None:
        try:
            event = json.loads(raw)
            self._on_event(event)
        except Exception as exc:
            logger.error("Failed to parse GRID event: %s — raw: %.200s", exc, raw)

    def _on_error(self, ws, error) -> None:
        logger.error("GRID WebSocket error: %s", error)

    def _on_close(self, ws, code, msg) -> None:
        logger.info("GRID WebSocket closed: code=%s msg=%s", code, msg)
