"""Evaluate a list of parameter settings in one process, sharing the loaded events (also the Batch worker).

    python -m gnomepy_research.sessions.cs2_comp.e06.sweep --data <bundle> --configs <json> [--out <json>]

configs: a list of {"name": str, "set": {param: value}, "faceit": bool, "oracle": bool, "start": str, "end": str}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from gnomepy_research.sessions.cs2_comp.e06.events import HLTV
from gnomepy_research.sessions.cs2_comp.e06.run import load_inputs
from gnomepy_research.sessions.cs2_comp.e06.skill import Params
from gnomepy_research.sessions.cs2_comp.e06.tune import evaluate


def as_params(values: dict) -> Params:
    return Params().with_values(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in values.items()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--configs", type=Path)
    ap.add_argument("--configs-json")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--slice", default="0/1", help="i/n: take every n-th config starting at i")
    args = ap.parse_args()
    i, n = (int(x) for x in args.slice.split("/"))
    configs = json.loads(args.configs_json or args.configs.read_text())[i::n]
    events, rankings, moves = load_inputs(args.data, True)
    hltv_only = [e for e in events if e.source == HLTV]
    results = []
    for c in configs:
        faceit = c.get("faceit", True)
        summary, _ = evaluate(args.data, as_params(c.get("set", {})), c.get("start", "2025-10-01"), c.get("end", "2026-04-01"),
                              faceit, c.get("oracle", False), preloaded=(events if faceit else hltv_only, rankings, moves))
        results.append({"name": c["name"], **summary})
        print(json.dumps(results[-1]), flush=True)
    out = args.out or (Path(os.environ["OUTPUT_DIR"]) / f"sweep_{i}_{n}.json" if "OUTPUT_DIR" in os.environ else None)
    if out:
        out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
