"""Writes one entry file per sweep configuration (configs/<name>.py) and the research-job array file for them."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
REMOTE = "/opt/gnomepy-research/gnomepy_research/models/cs2/inplay/econ/configs"


def write(configs: dict, array_out: str):
    tasks = []
    for name, over in configs.items():
        (HERE / "configs" / f"{name}.py").write_text(
            f'"""Sweep configuration {name}."""\nfrom gnomepy_research.models.cs2.inplay.econ.model import make\n\n'
            f"fit, predict = make(**{over!r})\n")
        tasks.append([f"{REMOTE}/{name}.py", "--data", "/work/in/data", "--cache", "/work/in/bench",
                      "--out", f"/work/out/{name}.json", "--name", f"econ-{name}"])
    Path(array_out).write_text(json.dumps(tasks, indent=1))


if __name__ == "__main__":
    write(json.loads(sys.argv[1]), sys.argv[2])
