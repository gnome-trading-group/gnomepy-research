import pandas as pd

from gnomepy_research.models.cs2.hltv.config import showmatch_mask


def test_showmatch_is_found_by_event_or_format():
    history = pd.DataFrame({
        "event_name": ["Showmatch CS", "PGL Astana 2026", "IEM Cologne 2026", "ESL Pro League"],
        "format": ["Best of 3 (LAN)", "Best of 1 (LAN) * PGL Astana 2026 Showmatch", "Best of 3 (LAN)", None],
    })
    assert showmatch_mask(history).tolist() == [True, True, False, False]
