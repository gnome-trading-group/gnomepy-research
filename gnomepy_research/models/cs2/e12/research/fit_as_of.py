"""The e12-wide-mm fit cs2_e12_fit would have published at a given time, fitted at that cutoff. HLTV datasets are read
as of that time (rows ingested after it are invisible, so no later rescrape leaks in). FACEIT datasets were first
loaded into prod by a one-off backfill on 2026-11-03, so before then they are read as they are now and cut by FACEIT's
own finished_at, the rule the model applies anyway (a map FACEIT finalised late carries the time it was finalised).

    python -m gnomepy_research.models.cs2.e12.research.fit_as_of --cutoff 2026-11-03T08:30
Writes $OUTPUT_DIR/e12_<cutoff>.npz."""
import argparse
import os
from datetime import timezone
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

from gnomepy.data import DatasetStore
from gnomepy_research.models.cs2.e12.model import E12Model
from gnomepy_research.models.cs2.e12.tables import DatasetTables


# Prod's HLTV history was backfilled on 2026-10-19/20 (back to 2024); read as of an earlier moment it is incomplete in a
# way no prod fit ever saw. --hltv-as-of reads it as of a later moment instead, and --scrape-rule then keeps only the
# series the 06:00 hltv_cs2 scrape before the cutoff would have had: start + format buffer before that 06:00.
BUFFER_H = {1: 3.0, 3: 6.0, 5: 9.0}
SCRAPE_H = 6.0


class AsOf:
    """A DatasetStore whose HLTV reads are as of one moment."""

    def __init__(self, store, as_of, scrape_cutoff=None):
        self.store, self.as_of, self.scrape_cutoff = store, as_of, scrape_cutoff

    def load(self, name, **kw):
        if not (name.startswith("cs2_") and name != "cs2_hltv_faceit_map"):
            return self.store.load(name, **kw)
        d = self.store.load(name, as_of=self.as_of, **kw)
        if name == "cs2_match_history" and self.scrape_cutoff is not None:
            cols = kw.get("columns")
            full = d if cols is None or {"match_time", "bo_type"} <= set(cols) else self.store.load(
                name, as_of=self.as_of, columns=list(dict.fromkeys((cols or []) + ["match_id", "match_time", "bo_type"])))
            t = pd.to_datetime(full.match_time)
            t = t.dt.tz_convert(None) if t.dt.tz is not None else t
            end = t + pd.to_timedelta(full.bo_type.map(BUFFER_H).fillna(9.0), unit="h")
            scrape = self.scrape_cutoff.normalize() + pd.Timedelta(hours=SCRAPE_H)
            scrape = scrape if scrape <= self.scrape_cutoff else scrape - pd.Timedelta(days=1)
            keep = set(full.match_id[end < scrape])
            d = d[d.match_id.isin(keep)] if "match_id" in d else d
        return d


ap = argparse.ArgumentParser()
ap.add_argument("--cutoff", required=True, help="naive UTC; the datasets are read as of this moment too")
ap.add_argument("--hltv-as-of", help="read HLTV as of this moment instead (naive UTC)")
ap.add_argument("--scrape-rule", action="store_true", help="keep only series the morning scrape before the cutoff had")
a = ap.parse_args()
cutoff = pd.Timestamp(a.cutoff)
as_of = pd.Timestamp(a.hltv_as_of) if a.hltv_as_of else cutoff
tables = DatasetTables(AsOf(DatasetStore(), as_of.tz_localize(timezone.utc).to_pydatetime(),
                            cutoff if a.scrape_rule else None))
with threadpool_limits(1):
    model = E12Model.build(tables, cutoff)
print("fit", cutoff, model.info)
out = Path(os.environ.get("OUTPUT_DIR", ".")) / f"e12_{cutoff.strftime('%Y%m%dT%H%M')}.npz"
model.save(out)
print("wrote", out)
