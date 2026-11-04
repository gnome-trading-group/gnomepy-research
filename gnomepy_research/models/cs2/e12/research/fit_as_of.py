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


class AsOf:
    """A DatasetStore whose HLTV reads are as of one moment."""

    def __init__(self, store, as_of):
        self.store, self.as_of = store, as_of

    def load(self, name, **kw):
        if name.startswith("cs2_") and name != "cs2_hltv_faceit_map":
            return self.store.load(name, as_of=self.as_of, **kw)
        return self.store.load(name, **kw)


ap = argparse.ArgumentParser()
ap.add_argument("--cutoff", required=True, help="naive UTC; the datasets are read as of this moment too")
a = ap.parse_args()
cutoff = pd.Timestamp(a.cutoff)
tables = DatasetTables(AsOf(DatasetStore(), cutoff.tz_localize(timezone.utc).to_pydatetime()))
with threadpool_limits(1):
    model = E12Model.build(tables, cutoff)
print("fit", cutoff, model.info)
out = Path(os.environ.get("OUTPUT_DIR", ".")) / f"e12_{cutoff.strftime('%Y%m%dT%H%M')}.npz"
model.save(out)
print("wrote", out)
