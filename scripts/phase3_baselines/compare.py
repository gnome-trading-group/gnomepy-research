"""Compares a baseline's old (released code) and new (Phase 3 code) runs: same fills, orders and final PnL.

Identity columns are left out: the new runs key by listing and stream where the old ones used exchange and security.
"""
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
IDENTITY = {"exchange_id", "security_id", "listing_id", "stream_id"}


def frame(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df[[c for c in df.columns if c not in IDENTITY]].reset_index()


def main(name: str) -> int:
    old, new = HERE / "results" / name / "old", HERE / "results" / name / "new"
    failures = []
    for table in sorted(p.name for p in old.glob("*.parquet")):
        a, b = frame(old / table), frame(new / table)
        if not a.equals(b):
            failures.append(f"{table}: {len(a)} old rows, {len(b)} new rows, differing")
    pnl_old = json.loads((old / "summary.json").read_text())["final_pnl"]
    pnl_new = json.loads((new / "summary.json").read_text())["final_pnl"]
    if pnl_old != pnl_new:
        failures.append(f"final_pnl {pnl_old} -> {pnl_new}")
    print(name, "MATCH" if not failures else "DIFFERS:\n  " + "\n  ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(max(main(name) for name in sys.argv[1:]))
