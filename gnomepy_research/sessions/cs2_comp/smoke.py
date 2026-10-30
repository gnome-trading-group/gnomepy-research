"""Batch smoke test for the competition bundle: lists the input and writes each table's row count."""
import json
import os
from pathlib import Path

import pyarrow.parquet as pq

root = Path("/work/in/comp")
counts = {str(p.relative_to(root)): pq.ParquetFile(p).metadata.num_rows for p in sorted(root.rglob("*.parquet"))}
Path(os.environ["OUTPUT_DIR"], "counts.json").write_text(json.dumps(counts, indent=1))
print(json.dumps(counts, indent=1))
