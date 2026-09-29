"""
HLTV data scraper for CS2 professional match history and team rankings.

Scrapes three page types:
  - /results?startDate=...&endDate=...  — paginated match result listings
  - /matches/{id}/...                   — per-match detail (lineups, veto, stats)
  - /ranking/teams/{year}/{month}/{day} — weekly team rankings

Requires ZENROWS_API_KEY env var.
Uses js_render + premium_proxy (25 credits/page) to bypass Cloudflare on HLTV.

Usage (via HltvCs2Pipeline or CLI):
    python -m gnomepy_research.pipelines.hltv_cs2.scraper \\
        --mode matches --start-date 2023-09-01 --end-date 2026-09-28
"""
from __future__ import annotations

import argparse
import calendar
import datetime
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from bs4 import BeautifulSoup
from zenrows import ZenRowsClient

from gnomepy_research.artifacts import DatasetStore

logger = logging.getLogger(__name__)

_BASE_URL = "https://www.hltv.org"
_REQUEST_DELAY = 3.0
_MAX_WORKERS = 5
_ZENROWS_PARAMS = {"js_render": True, "premium_proxy": True}


class _Session:
    """ZenRows session — residential proxy + JS render bypasses Cloudflare on HLTV."""

    def __init__(self) -> None:
        self._client = ZenRowsClient(os.environ["ZENROWS_API_KEY"])

    def get_html(self, url: str) -> str | None:
        try:
            resp = self._client.get(url, params=_ZENROWS_PARAMS)
            if resp.status_code == 200:
                return resp.text
            logger.warning("ZenRows HTTP %s for %s", resp.status_code, url)
            return None
        except Exception as exc:
            logger.warning("ZenRows error for %s: %s", url, exc)
            return None


def _make_session() -> _Session:
    return _Session()


def _get(session: _Session, url: str, retries: int = 3) -> BeautifulSoup | None:
    for attempt in range(retries):
        html = session.get_html(url)
        if html is None:
            time.sleep(5 * (attempt + 1))
            continue

        soup = BeautifulSoup(html, "html.parser")
        title_el = soup.find("title")
        title_text = title_el.get_text() if title_el else ""
        if "429" in title_text:
            wait = 60 * (attempt + 1)
            logger.warning("Rate limited — sleeping %ds", wait)
            time.sleep(wait)
            continue

        return soup

    return None


def merge_publish(name: str, new_df: pd.DataFrame, key_cols: list[str], date_col: str) -> None:
    """Key-based upsert: replace all rows whose key_cols values appear in new_df, keep everything else."""
    if new_df.empty:
        logger.warning("merge_publish called with empty DataFrame — %s not updated", name)
        return
    ds = DatasetStore()
    try:
        existing = ds.load(name)
        new_keys = new_df[key_cols].drop_duplicates()
        indicator = existing.merge(new_keys, on=key_cols, how="left", indicator=True)
        keep = existing[indicator["_merge"] == "left_only"]
        merged = pd.concat([keep, new_df], ignore_index=True).sort_values(date_col).reset_index(drop=True)
    except KeyError:
        merged = new_df
    min_date = pd.Timestamp(merged[date_col].min()).date().isoformat()
    max_date = pd.Timestamp(merged[date_col].max()).date().isoformat()
    ds.publish(merged, name, description=f"{min_date} to {max_date}")
    logger.info("Published %s: %d rows (%s to %s)", name, len(merged), min_date, max_date)


# ---------------------------------------------------------------------------
# Match result listings
# ---------------------------------------------------------------------------

def scrape_match_ids(
    session: _Session,
    start_date: datetime.date,
    end_date: datetime.date,
    min_stars: int = 2,
) -> dict[int, int]:
    """
    Scrape all match IDs from HLTV results pages between start_date and end_date.
    Returns {match_id: stars}.
    min_stars filters by HLTV star rating (0=all, 2=top-tier+, 3=big events only).
    Note: stars=1 is not a valid HLTV filter value and returns no results.
    """
    matches: dict[int, int] = {}
    offset = 0
    stars_param = f"&stars={min_stars}" if min_stars >= 2 else ""

    while True:
        url = f"{_BASE_URL}/results?startDate={start_date.isoformat()}&endDate={end_date.isoformat()}&offset={offset}{stars_param}"
        soup = _get(session, url)
        if soup is None:
            break

        links = soup.select(".results-all a.a-reset[href*='/matches/']")
        if not links:
            break

        for link in links:
            href = link.get("href", "")
            parts = href.split("/")
            if len(parts) >= 3:
                try:
                    stars_el = link.select_one(".stars")
                    stars = len(stars_el.select(".star")) if stars_el else 0
                    matches[int(parts[2])] = stars
                except ValueError:
                    pass

        if len(links) < 100:
            break
        offset += 100
        time.sleep(_REQUEST_DELAY)

    logger.info("Found %d match IDs between %s and %s (min_stars=%d)", len(matches), start_date.isoformat(), end_date.isoformat(), min_stars)
    return matches


def parse_match_ids_from_html(html: str) -> tuple[list[tuple[int, str, int]], bool]:
    """Parse match IDs, URLs, and star ratings from a results page HTML. Returns ([(id, url, stars)], has_more_pages)."""
    soup = BeautifulSoup(html, "html.parser")
    links = soup.select(".results-all a.a-reset[href*='/matches/']")
    matches = []
    for link in links:
        href = link.get("href", "")
        parts = href.split("/")
        if len(parts) >= 3:
            try:
                stars_el = link.select_one(".stars")
                stars = len(stars_el.select(".star")) if stars_el else 0
                matches.append((int(parts[2]), href, stars))
            except ValueError:
                pass
    return matches, len(links) >= 100


# ---------------------------------------------------------------------------
# Match detail pages
# ---------------------------------------------------------------------------

def _avg_rating(players: list[dict]) -> float:
    rated = [p["rating"] for p in players if p["rating"] is not None]
    return sum(rated) / len(rated) if rated else float("nan")


def _picked_map_value(picked_maps: dict, map_name: str, team_a_name: str) -> float:
    picker = picked_maps.get(map_name)
    if picker is None:
        return float("nan")
    return 1.0 if picker == team_a_name else 0.0


def _parse_veto(soup: BeautifulSoup) -> list[dict]:
    # HLTV renders two .veto-box elements: first is format text, second has picks/bans
    veto_boxes = soup.select(".veto-box")
    veto_box = veto_boxes[-1] if veto_boxes else None
    if not veto_box:
        return []
    rows = []
    for item in veto_box.select(".padding > div"):
        text = item.get_text(strip=True)
        if text:
            rows.append({"text": text})
    return rows


def _parse_player_stats(soup: BeautifulSoup) -> list[dict]:
    players = []
    for table in soup.select(".stats-table"):
        team_header = table.find_previous("div", class_="teamLine")
        team_name = team_header.get_text(strip=True) if team_header else ""
        for row in table.select("tbody tr"):
            cols = row.find_all("td")
            if len(cols) < 6:
                continue
            player_link = cols[0].select_one("a[href*='/player/']")
            if not player_link:
                continue
            href = player_link.get("href", "")
            parts = href.split("/")
            try:
                player_id = int(parts[2]) if len(parts) >= 3 else None
            except ValueError:
                player_id = None

            def _float(cell):
                try:
                    return float(cell.get_text(strip=True).replace("%", "").replace("+", ""))
                except (ValueError, AttributeError):
                    return None

            players.append({
                "team_name": team_name,
                "player_id": player_id,
                "player_name": player_link.get_text(strip=True),
                "kills": _float(cols[1]),
                "deaths": _float(cols[2]),
                "adr": _float(cols[3]) if len(cols) > 3 else None,
                "kast": _float(cols[4]) if len(cols) > 4 else None,
                "rating": _float(cols[5]) if len(cols) > 5 else None,
            })
    return players


def parse_match_detail_html(soup: BeautifulSoup, match_id: int, stars: int = 0) -> dict | None:
    """
    Parse a single HLTV match detail page from a BeautifulSoup object.
    Returns a dict with "rows" (list of per-map dicts) and "demo_url" (str or None).
    """
    try:
        team_boxes = soup.select(".team")
        team_names = []
        team_ids = []
        for box in team_boxes[:2]:
            link = box.select_one("a[href*='/team/']")
            if link:
                parts = link.get("href", "").split("/")
                try:
                    team_ids.append(int(parts[2]))
                except (ValueError, IndexError):
                    team_ids.append(None)
                name_el = box.select_one(".teamName")
                team_names.append(name_el.get_text(strip=True) if name_el else link.get_text(strip=True))
            else:
                name_el = box.select_one(".teamName")
                team_names.append(name_el.get_text(strip=True) if name_el else "")
                team_ids.append(None)

        if len(team_names) < 2:
            return None

        rank_els = soup.select(".teamRanking a")
        ranks = []
        for el in rank_els[:2]:
            text = el.get_text(strip=True).lstrip("#")
            try:
                ranks.append(int(text))
            except ValueError:
                ranks.append(None)
        while len(ranks) < 2:
            ranks.append(None)

        event_el = soup.select_one(".event a")
        event_name = event_el.get_text(strip=True) if event_el else ""


        format_el = soup.select_one(".preformatted-text")
        fmt = format_el.get_text(strip=True) if format_el else ""

        date_el = soup.select_one(".date")
        match_date = None
        if date_el:
            ts = date_el.get("data-unix")
            if ts:
                match_date = datetime.datetime.fromtimestamp(int(ts) / 1000, tz=datetime.timezone.utc).date()

        maps = []
        for map_el in soup.select(".mapholder"):
            name_el = map_el.select_one(".mapname")
            score_els = map_el.select(".results-team-score")
            if name_el and len(score_els) >= 2:
                try:
                    maps.append({
                        "map": name_el.get_text(strip=True),
                        "score_a": int(score_els[0].get_text(strip=True)),
                        "score_b": int(score_els[1].get_text(strip=True)),
                    })
                except ValueError:
                    pass

        veto = _parse_veto(soup)
        players = _parse_player_stats(soup)
        team_a_players = [p for p in players if p["team_name"] == team_names[0]]
        team_b_players = [p for p in players if p["team_name"] == team_names[1]]

        picked_maps: dict[str, str] = {}
        for v in veto:
            text = v["text"].lower()
            for m in ["inferno", "mirage", "nuke", "dust2", "anubis", "ancient", "vertigo"]:
                if m in text and "picked" in text:
                    picker = team_names[0] if team_names[0].lower() in text else (team_names[1] if team_names[1].lower() in text else "")
                    if picker:
                        picked_maps[f"de_{m}"] = picker

        # Demo download link
        demo_link_el = soup.select_one("a[href*='/download/demo/']")
        demo_url = f"{_BASE_URL}{demo_link_el['href']}" if demo_link_el else None

        rows = []
        for m in maps:
            map_name = f"de_{m['map'].lower()}" if not m["map"].startswith("de_") else m["map"].lower()
            team_a_won = m["score_a"] > m["score_b"]
            rows.append({
                "match_id": match_id,
                "match_date": match_date,
                "event_name": event_name,
                "event_tier": stars,
                "format": fmt,
                "team_a_name": team_names[0],
                "team_a_id": team_ids[0] if team_ids else None,
                "team_b_name": team_names[1],
                "team_b_id": team_ids[1] if len(team_ids) > 1 else None,
                "team_a_rank": ranks[0],
                "team_b_rank": ranks[1],
                "map_name": map_name,
                "team_a_score": m["score_a"],
                "team_b_score": m["score_b"],
                "team_a_won": int(team_a_won),
                "team_a_picked_map": _picked_map_value(picked_maps, map_name, team_names[0]),
                "team_a_player_ids": [p["player_id"] for p in team_a_players],
                "team_a_player_names": [p["player_name"] for p in team_a_players],
                "team_b_player_ids": [p["player_id"] for p in team_b_players],
                "team_b_player_names": [p["player_name"] for p in team_b_players],
                "team_a_avg_rating": _avg_rating(team_a_players),
                "team_b_avg_rating": _avg_rating(team_b_players),
            })

        return {"rows": rows, "demo_url": demo_url}

    except Exception as exc:
        logger.warning("Failed to parse match %d: %s", match_id, exc)
        return None


def scrape_match_detail(session: _Session, match_id: int, stars: int = 0) -> dict | None:
    """Scrape a single HLTV match detail page via ZenRows."""
    url = f"{_BASE_URL}/matches/{match_id}/"
    soup = _get(session, url)
    if soup is None:
        return None
    return parse_match_detail_html(soup, match_id, stars=stars)


# ---------------------------------------------------------------------------
# Weekly team rankings
# ---------------------------------------------------------------------------

def _all_mondays(start_date: datetime.date, end_date: datetime.date) -> list[datetime.date]:
    days_ahead = (7 - start_date.weekday()) % 7
    current = start_date + datetime.timedelta(days=days_ahead)
    mondays = []
    while current <= end_date:
        mondays.append(current)
        current += datetime.timedelta(weeks=1)
    return mondays


def parse_team_ranking_html(soup: BeautifulSoup, date: datetime.date) -> list[dict] | None:
    """Parse HLTV team rankings from a BeautifulSoup object."""
    ranked_teams = soup.select(".ranked-team")
    rows = []
    for team_el in ranked_teams:
        pos_el = team_el.select_one(".position")
        name_el = team_el.select_one(".teamLine .name")
        pts_el = team_el.select_one(".teamLine .points")
        change_el = team_el.select_one(".change")

        team_link = team_el.select_one("a[href*='/team/']")
        team_id = None
        if team_link:
            parts = team_link.get("href", "").split("/")
            try:
                team_id = int(parts[2])
            except (ValueError, IndexError):
                pass

        if not (pos_el and name_el):
            continue

        try:
            rank = int(pos_el.get_text(strip=True).lstrip("#"))
        except ValueError:
            continue

        pts_text = pts_el.get_text(strip=True) if pts_el else ""
        pts_match = re.search(r"[\d,]+", pts_text)
        try:
            points = int(pts_match.group(0).replace(",", "")) if pts_match else 0
        except ValueError:
            points = 0

        change_text = change_el.get_text(strip=True) if change_el else "0"
        try:
            change = int(change_text.replace("+", ""))
        except ValueError:
            change = 0

        rows.append({
            "date": pd.Timestamp(date),
            "team_name": name_el.get_text(strip=True),
            "team_id": team_id,
            "rank": rank,
            "points": points,
            "change": change,
        })

    return rows if rows else None


def scrape_team_ranking(session: _Session, date: datetime.date) -> list[dict] | None:
    """Scrape HLTV team rankings for a specific Monday via ZenRows."""
    month_name = calendar.month_name[date.month].lower()
    url = f"{_BASE_URL}/ranking/teams/{date.year}/{month_name}/{date.day}"
    soup = _get(session, url)
    if soup is None:
        return None
    return parse_team_ranking_html(soup, date)


# ---------------------------------------------------------------------------
# Top-level scrape runners (called by HltvCs2Pipeline)
# ---------------------------------------------------------------------------

def run_matches(
    start_date: datetime.date,
    end_date: datetime.date,
    min_stars: int = 2,
) -> pd.DataFrame:
    """Scrape all match detail pages between start_date and end_date with concurrency."""
    session = _make_session()

    match_stars = scrape_match_ids(session, start_date, end_date, min_stars=min_stars)
    logger.info("%d matches to scrape", len(match_stars))

    new_rows: list[dict] = []
    total_done = 0
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        for batch_start in range(0, len(match_stars), 100):
            batch = list(match_stars.items())[batch_start : batch_start + 100]
            futures = {pool.submit(scrape_match_detail, session, mid, stars): mid for mid, stars in batch}
            for future in as_completed(futures):
                result = future.result()
                if result and result["rows"]:
                    new_rows.extend(result["rows"])
                total_done += 1
            logger.info("Progress: %d / %d matches", total_done, len(match_stars))

    df = pd.DataFrame(new_rows)
    df["match_date"] = pd.to_datetime(df["match_date"])
    logger.info("Scraped %d map rows", len(df))
    merge_publish("cs2_match_history", df, ["match_id", "map_name"], "match_date")
    return df


def run_rankings(
    start_date: datetime.date,
    end_date: datetime.date,
) -> pd.DataFrame:
    """Scrape weekly HLTV team rankings with concurrency. Publishes to DatasetStore append-only."""
    session = _make_session()
    mondays = _all_mondays(start_date, end_date)
    logger.info("Scraping %d weekly rankings", len(mondays))

    all_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        futures = {pool.submit(scrape_team_ranking, session, d): d for d in mondays}
        for future in as_completed(futures):
            rows = future.result()
            if rows:
                all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    logger.info("Scraped %d ranking rows", len(df))
    merge_publish("cs2_team_rankings", df, ["date", "team_id"], "date")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["matches", "rankings"])
    ap.add_argument("--start-date", required=True)
    ap.add_argument("--end-date", required=True)
    ap.add_argument("--min-stars", type=int, default=2)
    args = ap.parse_args()

    start = datetime.date.fromisoformat(args.start_date)
    end = datetime.date.fromisoformat(args.end_date)
    if args.mode == "matches":
        run_matches(start, end, min_stars=args.min_stars)
    else:
        run_rankings(start, end)
