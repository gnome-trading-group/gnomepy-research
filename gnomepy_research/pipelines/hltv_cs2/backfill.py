"""
Local HLTV data backfill — runs on your machine with a real Chrome browser.

Three independent subcommands:
  matches   — scrape match stats + demos, one date at a time
  rankings  — scrape weekly team rankings for a date range
  priors    — rebuild cs2_match_priors from full history (no browser needed)

Usage:
    # Matches — loops one day at a time; stops on any failure
    python -m gnomepy_research.pipelines.hltv_cs2.backfill matches \\
        --start-date 2026-09-01 --end-date 2026-09-07 \\
        --min-stars 0 --min-stars-demo 2

    # Rankings
    python -m gnomepy_research.pipelines.hltv_cs2.backfill rankings \\
        --start-date 2026-09-01 --end-date 2026-09-28

    # Priors (no browser)
    python -m gnomepy_research.pipelines.hltv_cs2.backfill priors

Cloudflare: Chrome opens in visible mode. If prompted with "Verify you are human",
click through manually — the session cookies persist for subsequent pages.
"""
from __future__ import annotations

import argparse
import asyncio
import calendar
import datetime
import gzip
import logging
import shutil
import subprocess
import time
from pathlib import Path

import nodriver as uc
import pandas as pd
from bs4 import BeautifulSoup

from demoparser2 import DemoParser

from gnomepy_research.artifacts import DatasetStore
from gnomepy_research.pipelines.hltv_cs2.build_priors import build_priors
from gnomepy_research.pipelines.hltv_cs2.demo_parser import parse_demo
from gnomepy_research.pipelines.hltv_cs2.scraper import (
    _BASE_URL,
    _all_mondays,
    merge_publish,
    parse_match_detail_html,
    parse_match_ids_from_html,
    parse_team_ranking_html,
)

logger = logging.getLogger(__name__)

_DEMO_TMP = Path("/tmp/hltv_demo")
_CHROME_PROFILE = Path.home() / ".cache" / "hltv_backfill_chrome"
_DOWNLOAD_TIMEOUT = 300  # seconds to wait for a demo download
_CF_TIMEOUT = 120        # seconds to wait for CF challenge resolution

_CF_PHRASES = (
    "Just a moment",
    "Verify you are human",
    "Checking your browser",
    "Performing security verification",
)

_download_state: dict = {}


def _notify_cf(label: str) -> None:
    subprocess.Popen([
        "osascript", "-e",
        f'display notification "CF challenge on {label} — solve it in Chrome" with title "HLTV Backfill" sound name "Ping"',
    ])


def _on_download_begin(event: uc.cdp.browser.DownloadWillBegin) -> None:
    _download_state["guid"] = event.guid
    _download_state["filename"] = event.suggested_filename
    _download_state["started"] = True


def _on_download_progress(event: uc.cdp.browser.DownloadProgress) -> None:
    if event.guid != _download_state.get("guid"):
        return
    _download_state["received"] = event.received_bytes
    _download_state["total"] = event.total_bytes
    if event.state == "completed":
        _download_state["done"] = True


async def _wait_for_page(page: uc.Tab, url: str, wait_selector: str | None, base_timeout: float = 30.0) -> str | None:
    """Wait for a page to load, notifying user on CF and waiting for manual solve."""
    cf_seen = False
    notified = False
    deadline = time.monotonic() + base_timeout

    while time.monotonic() < deadline:
        try:
            content = await page.get_content()
        except Exception:
            await asyncio.sleep(2.0)
            continue

        if any(p in content for p in _CF_PHRASES):
            if not cf_seen:
                cf_seen = True
                deadline = time.monotonic() + _CF_TIMEOUT
            if not notified:
                logger.info("CF challenge on %s — solve in Chrome...", url)
                _notify_cf(url)
                notified = True
            await asyncio.sleep(2.0)
            continue

        if not wait_selector or wait_selector in content:
            if cf_seen:
                logger.info("CF resolved on %s", url)
            return content

        await asyncio.sleep(2.0)

    logger.warning("Timed out on %s (cf_seen=%s, selector='%s')", url, cf_seen, wait_selector)
    return None


async def _get_html(page: uc.Tab, url: str, wait_selector: str | None = None, retries: int = 3) -> str | None:
    """Navigate to URL and return page HTML once CF resolves and target element appears."""
    for attempt in range(1, retries + 1):
        try:
            await page.get(url)
            await asyncio.sleep(2.0)
            html = await _wait_for_page(page, url, wait_selector)
            if html is not None:
                return html
            if attempt < retries:
                logger.info("Retrying %s (attempt %d/%d)...", url, attempt + 1, retries)
        except Exception as exc:
            logger.warning("nodriver error for %s (attempt %d/%d): %s", url, attempt, retries, exc)
    return None


async def _download_demo(page: uc.Tab, demo_url: str) -> Path | None:
    """Navigate to demo download URL and wait for Chrome to save the file."""
    for f in _DEMO_TMP.iterdir():
        shutil.rmtree(f) if f.is_dir() else f.unlink()

    _download_state.clear()

    try:
        await page.get(demo_url)
    except Exception as exc:
        logger.warning("Demo download navigation error: %s", exc)
        return None

    cf_seen = False
    notified = False
    deadline = time.monotonic() + _DOWNLOAD_TIMEOUT

    while time.monotonic() < deadline:
        if _download_state.get("started"):
            break
        files = list(_DEMO_TMP.iterdir())
        if any(f.suffix == ".crdownload" for f in files) or any(f.suffix != ".crdownload" for f in files):
            break
        try:
            content = await page.get_content()
            if any(p in content for p in _CF_PHRASES):
                if not cf_seen:
                    cf_seen = True
                if not notified:
                    logger.info("CF challenge on demo download — solve in Chrome...")
                    _notify_cf("demo download")
                    notified = True
            elif cf_seen:
                logger.info("CF resolved on demo download")
                cf_seen = False
        except Exception:
            pass
        await asyncio.sleep(2.0)

    filename = _download_state.get("filename", "")
    if filename:
        logger.info("Download started: %s", filename)
    else:
        logger.info("Download started")

    last_log = -1.0
    while time.monotonic() < deadline:
        if _download_state.get("done"):
            break
        files = list(_DEMO_TMP.iterdir())
        in_progress = [f for f in files if f.suffix == ".crdownload"]
        completed = [f for f in files if f.suffix != ".crdownload"]
        if completed and not in_progress:
            break
        if in_progress:
            total = _download_state.get("total", 0)
            received = _download_state.get("received", 0)
            if total > 0:
                pct = int(received * 100 / total)
                if pct != last_log:
                    logger.info("  %s — %.1f / %.1f MB (%d%%)",
                                filename or in_progress[0].name,
                                received / 1_048_576, total / 1_048_576, pct)
                    last_log = pct
            else:
                size_mb = in_progress[0].stat().st_size / 1_048_576
                if abs(size_mb - last_log) > 1.0:
                    elapsed = _DOWNLOAD_TIMEOUT - (deadline - time.monotonic())
                    logger.info("  %s — %.1f MB (%.0fs elapsed)", in_progress[0].name, size_mb, elapsed)
                    last_log = size_mb
        await asyncio.sleep(2.0)

    completed = [f for f in _DEMO_TMP.iterdir() if f.suffix != ".crdownload"]
    if completed:
        logger.info("Demo download complete: %s (%.1f MB)",
                    completed[0].name, completed[0].stat().st_size / 1_048_576)
        return completed[0]

    logger.warning("Demo download timed out after %ds", _DOWNLOAD_TIMEOUT)
    return None


def _extract_dem(archive_path: Path) -> list[Path]:
    """Decompress archive and return all .dem files. Deletes the archive."""
    if archive_path.suffix == ".gz":
        dem_path = archive_path.with_suffix("")
        with gzip.open(archive_path, "rb") as src, open(dem_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
        archive_path.unlink()
        return [dem_path]

    if archive_path.suffix == ".dem":
        return [archive_path]

    if archive_path.suffix == ".rar":
        result = subprocess.run(
            ["unar", "-o", str(_DEMO_TMP), "-f", str(archive_path)],
            capture_output=True,
        )
        archive_path.unlink()
        if result.returncode != 0:
            logger.warning("unar failed: %s", result.stderr.decode())
            return []
        dem_files = list(_DEMO_TMP.glob("**/*.dem"))
        if not dem_files:
            logger.warning("No .dem found after extracting %s", archive_path.name)
        return dem_files

    logger.warning("Unsupported demo archive format: %s", archive_path.suffix)
    return []


def _publish_match_data(match_rows: list[dict], demo_rows: list[dict]) -> None:
    if match_rows:
        df = pd.DataFrame(match_rows)
        df["match_date"] = pd.to_datetime(df["match_date"])
        logger.info("Publishing cs2_match_history (%d rows)...", len(df))
        merge_publish("cs2_match_history", df, ["match_id", "map_name"], "match_date")
    else:
        logger.warning("No match rows — cs2_match_history not updated")

    if demo_rows:
        df = pd.DataFrame(demo_rows)
        df["match_date"] = pd.to_datetime(df["match_date"])
        logger.info("Publishing cs2_round_features (%d rows)...", len(df))
        merge_publish("cs2_round_features", df, ["match_id", "map_name"], "match_date")


async def backfill_matches(
    page: uc.Tab,
    start_date: datetime.date,
    end_date: datetime.date,
    min_stars_match: int = 0,
    skip_demos: bool = False,
    min_stars_demo: int = 2,
    match_ids: set[int] | None = None,
) -> bool:
    """
    Scrape all matches for every date in [start_date, end_date].
    Accumulates data across all dates and publishes once in the finally block.
    Returns True on full success, False if short-circuited on any failure.
    """
    match_rows: list[dict] = []
    demo_rows: list[dict] = []
    stars_param = f"&stars={min_stars_match}" if min_stars_match >= 2 else ""

    try:
        current = start_date
        while current <= end_date:
            # --- Fetch match listing for this date ---
            matches: list[tuple[int, str]] = []
            offset = 0
            logger.info("Scraping matches for %s (min_stars=%d)", current.isoformat(), min_stars_match)

            while True:
                url = f"{_BASE_URL}/results?startDate={current.isoformat()}&endDate={current.isoformat()}&offset={offset}{stars_param}"
                html = await _get_html(page, url, wait_selector="contentCol")
                if html is None:
                    logger.warning("Match listing failed for %s at offset %d — short-circuiting", current, offset)
                    return False
                if "results-all" not in html:
                    logger.info("No matches for %s at offset %d", current, offset)
                    break
                page_matches, has_more = parse_match_ids_from_html(html)
                matches.extend(page_matches)
                if not has_more:
                    break
                offset += 100

            seen: set[int] = set()
            unique: list[tuple[int, str, int]] = []
            for mid, href, stars in matches:
                if mid not in seen:
                    seen.add(mid)
                    unique.append((mid, href, stars))
            matches = unique
            if match_ids is not None:
                matches = [(mid, href, stars) for mid, href, stars in matches if mid in match_ids]
                logger.info("Filtered to %d matches for %s (--match-ids)", len(matches), current.isoformat())
            else:
                logger.info("Found %d matches for %s", len(matches), current.isoformat())

            # --- Process each match ---
            n_demos_parsed = 0
            n_demos_skipped = 0

            for i, (match_id, match_href, match_stars) in enumerate(matches):
                n = i + 1
                logger.info("[%d/%d] match %d — fetching page", n, len(matches), match_id)
                html = await _get_html(page, f"{_BASE_URL}{match_href}", wait_selector="mapholder")
                if html is None:
                    logger.warning("[%d/%d] match %d — no HTML, short-circuiting", n, len(matches), match_id)
                    return False

                soup = BeautifulSoup(html, "html.parser")
                result = parse_match_detail_html(soup, match_id, stars=match_stars)
                if result is None:
                    logger.warning("[%d/%d] match %d — parse failed, short-circuiting", n, len(matches), match_id)
                    return False

                if result["rows"]:
                    match_rows.extend(result["rows"])
                    logger.info("[%d/%d] match %d — %d maps parsed (stars=%d)", n, len(matches), match_id, len(result["rows"]), match_stars)
                else:
                    logger.warning("[%d/%d] match %d — no map rows", n, len(matches), match_id)

                demo_url = result.get("demo_url")
                if skip_demos or not demo_url:
                    if not skip_demos and not demo_url:
                        logger.info("[%d/%d] match %d — no demo URL", n, len(matches), match_id)
                    continue
                if match_stars < min_stars_demo:
                    logger.info("[%d/%d] match %d — demo skipped (stars=%d < min=%d)", n, len(matches), match_id, match_stars, min_stars_demo)
                    n_demos_skipped += 1
                    continue

                logger.info("[%d/%d] match %d — downloading demo (stars=%d)", n, len(matches), match_id, match_stars)

                archive = await _download_demo(page, demo_url)
                if archive is None:
                    logger.warning("[%d/%d] match %d — demo download failed, short-circuiting", n, len(matches), match_id)
                    return False

                dem_paths = _extract_dem(archive)
                if not dem_paths:
                    logger.warning("[%d/%d] match %d — demo extraction failed, short-circuiting", n, len(matches), match_id)
                    return False

                match_date = result["rows"][0]["match_date"] if result["rows"] else None
                score_by_map = {
                    r["map_name"]: r["team_a_score"] + r["team_b_score"]
                    for r in result["rows"]
                }

                # Group dem files by map (handles multi-part demos)
                map_groups: dict[str, list[Path]] = {}
                for dem_path in dem_paths:
                    try:
                        map_name = DemoParser(str(dem_path)).parse_header().get("map_name", "unknown")
                    except Exception:
                        map_name = "unknown"
                    map_groups.setdefault(map_name, []).append(dem_path)

                for demo_map, group_paths in map_groups.items():
                    df = parse_demo([str(p) for p in group_paths])
                    if df is None:
                        n_demos_skipped += 1
                        logger.info("[%d/%d] match %d — demo parse returned no rows: %s", n, len(matches), match_id, demo_map)
                        for p in group_paths:
                            p.unlink(missing_ok=True)
                        continue

                    actual_rounds = df["round_number"].nunique()
                    expected_rounds = score_by_map.get(demo_map)
                    if expected_rounds is not None and actual_rounds != expected_rounds:
                        logger.warning(
                            "[%d/%d] match %d — demo round count mismatch on %s: got %d, expected %d — skipping",
                            n, len(matches), match_id, demo_map, actual_rounds, expected_rounds,
                        )
                        n_demos_skipped += 1
                        for p in group_paths:
                            p.unlink(missing_ok=True)
                        continue

                    df["match_id"] = match_id
                    df["match_date"] = match_date
                    demo_rows.extend(df.to_dict("records"))
                    n_demos_parsed += 1
                    logger.info("[%d/%d] match %d — demo parsed %s (%d parts, %d/%s rounds, %d snapshots)",
                                n, len(matches), match_id, demo_map, len(group_paths), actual_rounds,
                                str(expected_rounds) if expected_rounds else "?", len(df))
                    for p in group_paths:
                        p.unlink(missing_ok=True)

            logger.info(
                "Done %s: %d maps from %d matches | %d demos parsed | %d skipped",
                current.isoformat(), len(match_rows), len(matches), n_demos_parsed, n_demos_skipped,
            )
            current += datetime.timedelta(days=1)

        return True

    finally:
        _publish_match_data(match_rows, demo_rows)


async def backfill_rankings(
    page: uc.Tab,
    start_date: datetime.date,
    end_date: datetime.date,
) -> None:
    """Scrape weekly HLTV team rankings for all Mondays in [start_date-1week, end_date]."""
    ranking_start = start_date - datetime.timedelta(weeks=1)
    mondays = _all_mondays(ranking_start, end_date)
    logger.info("Scraping %d weekly rankings (%s to %s)", len(mondays), ranking_start, end_date)

    ranking_rows: list[dict] = []
    for j, date in enumerate(mondays):
        month_name = calendar.month_name[date.month].lower()
        url = f"{_BASE_URL}/ranking/teams/{date.year}/{month_name}/{date.day}"
        html = await _get_html(page, url, wait_selector="ranked-team")
        if html:
            soup = BeautifulSoup(html, "html.parser")
            rows = parse_team_ranking_html(soup, date)
            if rows:
                ranking_rows.extend(rows)
                logger.info("Rankings [%d/%d] %s — %d teams", j + 1, len(mondays), date, len(rows))
            else:
                logger.warning("Rankings [%d/%d] %s — no rows", j + 1, len(mondays), date)
        else:
            logger.warning("Rankings [%d/%d] %s — no HTML", j + 1, len(mondays), date)

    if ranking_rows:
        logger.info("Publishing cs2_team_rankings (%d rows)...", len(ranking_rows))
        merge_publish("cs2_team_rankings", pd.DataFrame(ranking_rows), ["date", "team_id"], "date")
    else:
        logger.warning("No ranking rows — cs2_team_rankings not updated")


def backfill_priors() -> None:
    """Rebuild cs2_match_priors from full cs2_match_history + cs2_team_rankings."""
    ds = DatasetStore()
    logger.info("Loading cs2_match_history and cs2_team_rankings...")
    all_matches = ds.load("cs2_match_history")
    all_rankings = ds.load("cs2_team_rankings")
    logger.info("Building priors for %d match rows...", len(all_matches))
    priors_df = build_priors(all_matches, all_rankings)
    min_date = pd.Timestamp(priors_df["match_date"].min()).date().isoformat()
    max_date = pd.Timestamp(priors_df["match_date"].max()).date().isoformat()
    logger.info("Publishing cs2_match_priors (%d rows, %s to %s)...", len(priors_df), min_date, max_date)
    ds.publish(priors_df, "cs2_match_priors", description=f"{min_date} to {max_date}")
    logger.info("Priors backfill complete.")


async def _run_with_browser(coro) -> None:
    """Start a Chrome browser, run coro(page), stop browser when done."""
    _DEMO_TMP.mkdir(parents=True, exist_ok=True)
    _CHROME_PROFILE.mkdir(parents=True, exist_ok=True)

    browser = await uc.start(headless=False, user_data_dir=str(_CHROME_PROFILE))
    page = await browser.get("about:blank")

    await page.send(uc.cdp.browser.set_download_behavior(
        behavior="allow",
        download_path=str(_DEMO_TMP),
        events_enabled=True,
    ))
    browser.add_handler(uc.cdp.browser.DownloadWillBegin, _on_download_begin)
    browser.add_handler(uc.cdp.browser.DownloadProgress, _on_download_progress)

    try:
        await coro(page)
    finally:
        browser.stop()


def _run_cancellable(coro) -> None:
    """Run a coroutine, ensuring finally blocks execute on Ctrl+C."""
    loop = uc.loop()
    task = loop.create_task(coro)
    try:
        loop.run_until_complete(task)
    except KeyboardInterrupt:
        logger.info("Interrupted — flushing collected data...")
        task.cancel()
        try:
            loop.run_until_complete(task)
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    mp = sub.add_parser("matches", help="Scrape match stats and demos")
    mp.add_argument("--start-date", required=True)
    mp.add_argument("--end-date", required=True)
    mp.add_argument("--min-stars", type=int, default=0, help="Min stars to scrape a match (0=all)")
    mp.add_argument("--skip-demos", action="store_true")
    mp.add_argument("--min-stars-demo", type=int, default=2, help="Min stars to download a demo")
    mp.add_argument("--match-ids", type=int, nargs="+", default=None, help="Only process these match IDs")

    rp = sub.add_parser("rankings", help="Scrape weekly team rankings")
    rp.add_argument("--start-date", required=True)
    rp.add_argument("--end-date", required=True)

    sub.add_parser("priors", help="Rebuild match priors from full history (no browser)")

    args = ap.parse_args()

    if args.cmd == "priors":
        backfill_priors()

    elif args.cmd == "rankings":
        start = datetime.date.fromisoformat(args.start_date)
        end = datetime.date.fromisoformat(args.end_date)

        async def _rankings(page: uc.Tab) -> None:
            await backfill_rankings(page, start, end)

        _run_cancellable(_run_with_browser(_rankings))

    elif args.cmd == "matches":
        start = datetime.date.fromisoformat(args.start_date)
        end = datetime.date.fromisoformat(args.end_date)
        min_stars = args.min_stars
        skip_demos = args.skip_demos
        min_stars_demo = args.min_stars_demo
        match_ids = set(args.match_ids) if args.match_ids else None

        async def _matches(page: uc.Tab) -> None:
            await backfill_matches(
                page, start, end,
                min_stars_match=min_stars,
                skip_demos=skip_demos,
                min_stars_demo=min_stars_demo,
                match_ids=match_ids,
            )

        _run_cancellable(_run_with_browser(_matches))
