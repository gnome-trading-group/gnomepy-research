"""Batch entry points: each writes its output under $OUTPUT_DIR.

    python -m gnomepy_research.sessions.cs2_comp.g08.batch e12 <weeks>      frozen e12 on the dev targets of those weeks
    python -m gnomepy_research.sessions.cs2_comp.g08.batch predict <weeks>  this entry's predictions for those weeks"""
import os
import sys
from pathlib import Path

import pandas as pd

from gnomepy_research.sessions.cs2_comp.g08.e12ref.predict import predict as e12_predict

DATA = Path("/work/in/comp")


def main() -> None:
    what, weeks = sys.argv[1], sys.argv[2]
    out = Path(os.environ["OUTPUT_DIR"])
    t = pd.read_parquet(DATA / "targets_tune.parquet")
    t = t[t.week_cutoff.isin(pd.to_datetime(weeks.split(",")))]
    p = e12_predict(DATA, t, faceit=True, verbose=True)
    p.to_parquet(out / f"{what}_{weeks.split(',')[0]}.parquet", index=False)
    print(f"wrote {len(p)}")


if __name__ == "__main__":
    main()
