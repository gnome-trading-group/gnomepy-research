"""Bidirectional sync between API research notes and local markdown files for Obsidian."""
from __future__ import annotations

import re
from pathlib import Path

from gnomepy_research.api import add_note, get_notes


_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


def _ts_to_filename(timestamp: str) -> str:
    safe = timestamp.replace(":", "-").replace("+", "").replace(" ", "T")
    safe = re.sub(r"\.\d+", "", safe)
    return f"note_{safe}.md"


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    fm: dict = {}
    for line in match.group(1).splitlines():
        if ": " in line:
            k, _, v = line.partition(": ")
            fm[k.strip()] = v.strip()
    body = text[match.end():]
    return fm, body


def pull_notes(session_name: str, session_dir: Path) -> int:
    """Download notes from API to sessions/<name>/notes/*.md. Returns count written."""
    notes_dir = Path(session_dir) / "notes"
    notes_dir.mkdir(exist_ok=True)

    notes = get_notes(session_name)
    written = 0
    for note in notes:
        ts = note.get("timestamp", "")
        filename = _ts_to_filename(ts)
        target = notes_dir / filename
        if target.exists():
            continue
        author = note.get("author", "")
        content = note.get("content", "")
        target.write_text(
            f"---\nauthor: {author}\ntimestamp: {ts}\n---\n\n{content}\n"
        )
        written += 1

    return written


def push_notes(session_name: str, session_dir: Path) -> int:
    """Upload new local .md files from notes/ to the API. Returns count uploaded."""
    notes_dir = Path(session_dir) / "notes"
    if not notes_dir.exists():
        return 0

    existing_notes = get_notes(session_name)
    existing_timestamps = {n.get("timestamp") for n in existing_notes}

    uploaded = 0
    for md_file in sorted(notes_dir.glob("*.md")):
        text = md_file.read_text()
        fm, body = _parse_frontmatter(text)
        ts = fm.get("timestamp")
        if ts and ts in existing_timestamps:
            continue
        content = body.strip()
        if not content:
            continue
        add_note(session_name, content)
        uploaded += 1

    return uploaded


