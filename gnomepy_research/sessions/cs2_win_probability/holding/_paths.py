"""
Where the holding scripts read and write their working data.

These scripts began in a job scratch directory and /tmp, both of which are
cleared. Everything now goes through one directory, overridable with
CS2_RESEARCH_DATA, so the analyses survive the session that produced them.
"""
import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("CS2_RESEARCH_DATA", "~/.cache/gnomepy/cs2_research")).expanduser()
DATA_DIR.mkdir(parents=True, exist_ok=True)
