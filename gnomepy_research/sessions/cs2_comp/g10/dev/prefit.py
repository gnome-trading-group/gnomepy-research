"""Fit the correction weights on pre-dev outcomes against e12: y ~ a*logit(q_e12) + sum_k b_k*x_k, months held out;
the weights predict uses are b_k / a. python -m ...g10.dev.prefit --dir D"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from gnomepy_research.sessions.cs2_comp.g10.dev.mmscan import cvfit
from gnomepy_research.sessions.cs2_comp.g10.evaluate import ci, logit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    D = Path(a.dir)
    pre = pd.concat([pd.read_parquet(p) for p in sorted(D.glob("prehist_*.parquet"))], ignore_index=True)
    e12 = pd.concat([pd.read_parquet(p) for p in sorted(D.glob("pree12_*.parquet"))], ignore_index=True)
    mm = pd.read_parquet(D / "mm.parquet")
    d = pre.merge(e12, on="match_id").merge(mm, on="match_id")
    y = d.y.to_numpy(float)
    mo = pd.to_datetime(d.time).dt.strftime("%Y-%m").to_numpy()
    ze, zb = logit(d.q_e12), logit(d.q_base)
    x = {"q_map": logit(d.q_m240s) - zb, "q_setting": logit(d.q_lan) - zb, "q_form": logit(d.q_form) - zb,
         "mm_form": (d.mm_a - d.mm_b).to_numpy(), "mm_activity": (d.mmact_a - d.mmact_b).to_numpy(),
         }
    x["net"] = (d.mmnet_a - d.mmnet_b).to_numpy() / 10.0
    x["net365"] = (d.mm365net_a - d.mm365net_b).to_numpy() / 10.0
    x["form365"] = (d.mm365_a - d.mm365_b).to_numpy()
    x["level"] = (d.level_a - d.level_b).to_numpy()
    x["level_change"] = (d.level_change_a - d.level_change_b).to_numpy()
    base, wb = cvfit([ze], y, mo)
    print(f"n={len(y)} e12 cv {base.mean():.4f} w={wb.round(3)} hltv-only cv {cvfit([zb], y, mo)[0].mean():.4f}")
    for k, v in x.items():
        l, w = cvfit([ze, v], y, mo)
        print(f"{k:14s} {ci(l - base)} w={w.round(3)}", flush=True)
    combos = [["q_map", "q_setting", "mm_form", "mm_activity"],
              ["q_map", "q_setting", "mm_form", "mm_activity", "net"],
              ["q_map", "q_setting", "mm_form", "mm_activity", "net365"],
              ["q_map", "q_setting", "net", "mm_activity"]]
    early = mo < "2025-08"
    for c in combos:
        l, w = cvfit([ze] + [x[k] for k in c], y, mo)
        ratios = {k: round(float(w[i + 1] / w[0]), 4) for i, k in enumerate(c)}
        print(f"{'+'.join(c):50s} {ci(l - base)} early {np.mean((l - base)[early]):+.4f} late {np.mean((l - base)[~early]):+.4f} "
              f"a={w[0]:.3f} ratios={ratios}", flush=True)


if __name__ == "__main__":
    main()
