"""Parameter vectors of the filter and the fitted values the predict module uses.

Vectors are in the unconstrained order below; log_* entries are natural logs. They were fitted by one-step-ahead map log
loss on maps from 2025-01 to 2026-03, which is before the first target, and are fixed constants of the model."""
from __future__ import annotations

import numpy as np

from gnomepy_research.sessions.cs2_comp.e05 import data as D
from gnomepy_research.sessions.cs2_comp.e05.pricing import PriceConfig
from gnomepy_research.sessions.cs2_comp.e05.stream import FIRST_TIERS, Config

HLTV_NAMES = ("log_tau", "lam_best", "hltv_a", "hltv_b", "log_s0", "log_smap_h", "log_rho_h", "log_g_h", "log_noise_h", "log_lam_h")
FACEIT_NAMES = ("log_lam_f", "log_wt_f", "log_rho_f", "log_smap_f", "log_s0_f", "level_c", "a_f", "log_g_f", "log_noise_f")


def build_config(hltv: np.ndarray, faceit: np.ndarray | None) -> Config:
    h = dict(zip(HLTV_NAMES, hltv))
    kw = dict(tau=float(np.exp(h["log_tau"])), lam_best=float(h["lam_best"]), hltv_a=float(h["hltv_a"]), hltv_b=float(h["hltv_b"]),
              hltv_sigma0=float(np.exp(h["log_s0"])), sigma_map={"hltv": float(np.exp(h["log_smap_h"]))},
              rho={"hltv": float(np.exp(h["log_rho_h"]))}, perf_g={"hltv": float(np.exp(h["log_g_h"]))},
              perf_noise={"hltv": float(np.exp(h["log_noise_h"]))}, lam={"hltv": float(np.exp(h["log_lam_h"]))}, faceit=faceit is not None)
    if faceit is not None:
        f = dict(zip(FACEIT_NAMES, faceit))
        for s in D.FACEIT_SOURCES:
            kw["lam"][s] = float(np.exp(h["log_lam_h"] + f["log_lam_f"]))
            kw["weight"] = {**kw.get("weight", {}), s: float(np.exp(f["log_wt_f"]))}
            kw["rho"][s] = float(np.exp(f["log_rho_f"]))
            kw["sigma_map"][s] = float(np.exp(f["log_smap_f"]))
            kw["perf_g"][s] = float(np.exp(f["log_g_f"]))
            kw["perf_noise"][s] = float(np.exp(f["log_noise_f"]))
        kw["faceit_sigma0"] = float(np.exp(f["log_s0_f"]))
        kw["faceit_level_c"] = float(f["level_c"])
        kw["faceit_a"] = {t: float(f["a_f"]) for t in FIRST_TIERS}
    return Config(**kw)


TUNED_HLTV = np.array((
    -6.16948, -0.199104, 0.383746,
    0.293614, -0.00500413, -0.813522,
    -6.99677, 0.163569, -5.42184,
    0.433901,
))
TUNED_FACEIT = np.array((
    0.783948, 1.94953, -7.08931,
    -0.72713, -2.99454, 0.246133,
    0.766792, -0.276515, -5.57302,
))


def final_config(faceit: bool) -> Config:
    return build_config(TUNED_HLTV, TUNED_FACEIT if faceit else None)


def final_price(faceit: bool) -> PriceConfig:
    if faceit:
        return PriceConfig(sigma_series=0.0778, scale=0.999, var_mult=0.270, coh_sigma=0.0, coh_n0=1.0, faceit_lineups=True, link_inflate=0.04)
    return PriceConfig(sigma_series=0.1209, scale=0.997, var_mult=0.2884, coh_sigma=0.191, coh_n0=8.18, faceit_lineups=False, link_inflate=0.0)
