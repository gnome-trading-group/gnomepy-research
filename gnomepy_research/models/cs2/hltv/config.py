import pandas as pd

MAP_POOL = [
    "de_ancient", "de_anubis", "de_cache", "de_dust2",
    "de_inferno", "de_mirage", "de_nuke", "de_overpass",
    "de_train", "de_vertigo",
]

SHOWMATCH_EVENT = "Showmatch CS"


def showmatch_mask(history: pd.DataFrame) -> pd.Series:
    """
    Exhibition series (legends rosters, mixed teams, old CS:GO maps), which say nothing about competitive strength.
    HLTV files every one under its generic "Showmatch CS" event, and the format text names it too.
    """
    return history["event_name"].eq(SHOWMATCH_EVENT) | history["format"].fillna("").str.contains("showmatch", case=False)
