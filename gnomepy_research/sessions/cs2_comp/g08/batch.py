"""Batch entry point: the predict module for some weeks of the dev targets, written under $OUTPUT_DIR.

    python -m gnomepy_research.sessions.cs2_comp.g08.batch <name> <weeks> [--no-e12]"""
import os
import sys
from pathlib import Path

from gnomepy_research.sessions.cs2_comp.g08.predict import predict, read_targets

DATA = Path("/work/in/comp")


def main() -> None:
    name, weeks = sys.argv[1], sys.argv[2]
    p = predict(DATA, read_targets(DATA / "targets_tune.parquet", weeks), use_e12="--no-e12" not in sys.argv, verbose=True)
    p.to_parquet(Path(os.environ["OUTPUT_DIR"]) / f"{name}_{weeks.split(',')[0]}.parquet", index=False)
    print(f"wrote {len(p)}")


if __name__ == "__main__":
    main()
