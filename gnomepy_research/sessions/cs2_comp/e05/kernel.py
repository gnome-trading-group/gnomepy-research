"""
Sequential skill filter over HLTV and FACEIT maps: one pass in time order, each map updating the ten players in it.

A player's skill is Gaussian in round-logit units; variance grows tau^2 per idle day. A lineup's strength is the mean of
its players plus lam_best on the best one. Teammates keep their covariance (a fixed five only ever reveals its sum, so
without it each player's variance would never shrink); opponents are treated as independent after each map. Each source
has its own scale (lam), information weight and overdispersion, so a FACEIT map moves the same skills by a different
amount than an HLTV map. Written for numba; the state lives in plain arrays and one typed dict of teammate covariances.
"""
from __future__ import annotations

import numpy as np
from numba import njit, types
from numba.typed import Dict

TEAM = 5
PAIR_SHIFT = 20


def new_cov() -> Dict:
    return Dict.empty(types.int64, types.float64)


@njit(cache=True)
def _pair(p, q):
    return (p << PAIR_SHIFT) | q if p < q else (q << PAIR_SHIFT) | p


GRID_LO = -8.0
GRID_STEP = 0.005


@njit(cache=True)
def _map_loglik(table, d, win):
    x = (min(max(d, GRID_LO), -GRID_LO - 1e-9) - GRID_LO) / GRID_STEP
    i = int(x)
    p = table[i] + (x - i) * (table[i + 1] - table[i])
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return np.log(p) if win == 1 else np.log(1.0 - p)


@njit(cache=True)
def _read_side(players, day, mu, var, seen, cov, tau2, lam_best, means, block, weights):
    for j in range(TEAM):
        p = players[j]
        means[j] = mu[p]
        block[j, j] = var[p] + tau2 * (day - seen[p])
    for j in range(TEAM):
        for k in range(j + 1, TEAM):
            c = cov.get(_pair(players[j], players[k]), 0.0)
            block[j, k] = c
            block[k, j] = c
    best = 0
    for j in range(1, TEAM):
        if means[j] > means[best] + 1e-9:
            best = j
    for j in range(TEAM):
        weights[j] = 1.0 / TEAM + lam_best * ((1.0 if j == best else 0.0) - 1.0 / TEAM)
    strength = 0.0
    for j in range(TEAM):
        strength += weights[j] * means[j]
    return strength


@njit(cache=True)
def _stat_update(m, c, perf, g, noise, ref, gain, row):
    """Condition one side's skills on its players' map-centred performance: perf_j = g * (skill_j - ref) + noise."""
    for j in range(TEAM):
        z = perf[j]
        if np.isnan(z):
            continue
        s = g * g * c[j, j] + noise
        innovation = z - g * (m[j] - ref)
        for k in range(TEAM):
            row[k] = c[j, k]
            gain[k] = g * row[k] / s
        for k in range(TEAM):
            m[k] += gain[k] * innovation
            for l in range(TEAM):
                c[k, l] -= gain[k] * g * row[l]


@njit(cache=True)
def run(mu, var, seen, prior_mu, prior_var, cov, day, src, rounds_a, rounds, win_a, side_a, side_b, perf_a, perf_b, start, stop, tau2,
        lam_best, src_lam, src_weight, src_rho, src_g, src_noise, tables, out_ll):
    ma = np.empty(TEAM)
    mb = np.empty(TEAM)
    ca = np.empty((TEAM, TEAM))
    cb = np.empty((TEAM, TEAM))
    wa = np.empty(TEAM)
    wb = np.empty(TEAM)
    na = np.empty((TEAM, TEAM))
    nb = np.empty((TEAM, TEAM))
    nma = np.empty(TEAM)
    nmb = np.empty(TEAM)
    gain = np.empty(TEAM)
    row = np.empty(TEAM)
    for i in range(start, stop):
        pa = side_a[i]
        pb = side_b[i]
        clash = False
        for j in range(TEAM):
            for k in range(TEAM):
                if pa[j] == pb[k]:
                    clash = True
            for k in range(j + 1, TEAM):
                if pa[j] == pa[k] or pb[j] == pb[k]:
                    clash = True
        if clash:
            continue
        for j in range(TEAM):
            for p in (pa[j], pb[j]):
                if seen[p] < 0:
                    mu[p] = prior_mu[p]
                    var[p] = prior_var[p]
                    seen[p] = day[i]
        sa = _read_side(pa, day[i], mu, var, seen, cov, tau2, lam_best, ma, ca, wa)
        sb = _read_side(pb, day[i], mu, var, seen, cov, tau2, lam_best, mb, cb, wb)
        s = src[i]
        lam = src_lam[s]
        n = rounds[i]
        r = rounds_a[i]
        infl = (1.0 + (n - 1.0) * src_rho[s]) / src_weight[s]
        cwa = ca @ wa
        cwb = -(cb @ wb)
        s2 = 0.0
        for j in range(TEAM):
            s2 += wa[j] * cwa[j] - wb[j] * cwb[j]
        d = lam * (sa - sb)
        pi = 1.0 / (1.0 + np.exp(-d))
        out_ll[i] = _map_loglik(tables[s], d / np.sqrt(1.0 + 0.125 * lam * lam * s2 * 3.141592653589793), win_a[i])
        grad = lam * (r - n * pi) / infl
        curv = lam * lam * n * pi * (1.0 - pi) / infl
        denom = 1.0 + curv * s2
        for j in range(TEAM):
            nma[j] = ma[j] + cwa[j] * grad / denom
            nmb[j] = mb[j] + cwb[j] * grad / denom
            for k in range(TEAM):
                na[j, k] = ca[j, k] - cwa[j] * cwa[k] * curv / denom
                nb[j, k] = cb[j, k] - cwb[j] * cwb[k] * curv / denom
        g = src_g[s]
        if g > 0.0:
            ref = 0.0
            for j in range(TEAM):
                ref += (ma[j] + mb[j]) / (2.0 * TEAM)
            _stat_update(nma, na, perf_a[i], g, src_noise[s], ref, gain, row)
            _stat_update(nmb, nb, perf_b[i], g, src_noise[s], ref, gain, row)
        for j in range(TEAM):
            mu[pa[j]] = nma[j]
            mu[pb[j]] = nmb[j]
            var[pa[j]] = max(1e-6, na[j, j])
            var[pb[j]] = max(1e-6, nb[j, j])
            for k in range(j + 1, TEAM):
                cov[_pair(pa[j], pa[k])] = na[j, k]
                cov[_pair(pb[j], pb[k])] = nb[j, k]
        for j in range(TEAM):
            seen[pa[j]] = day[i]
            seen[pb[j]] = day[i]
