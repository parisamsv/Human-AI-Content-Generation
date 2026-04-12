"""
Sweep functions for generating equilibrium data.
"""
import numpy as np
from simulator import PlatformSimulator

KEYS = ['beta_h', 'r', 'utility', 'q_h', 'q_A', 'cs', 'uc', 'tq']


def _collect(sim, res):
    b, r = res['beta_h'], res['r']
    res['cs'] = sim.consumer_surplus(b, r)
    res['uc'] = sim.creator_welfare(b, r)
    res['tq'] = sim.total_quality(b, r)
    return res


def sweep_Q(model, params, k, Q_min, Q_max, n, n_beta=1001):
    Qs = np.linspace(Q_min, Q_max, n)
    out = {key: np.zeros(n) for key in KEYS}
    for i, Q in enumerate(Qs):
        sim = PlatformSimulator(Q=Q, k=k, model=model, **params)
        res = _collect(sim, sim.optimize(n_beta))
        for key in KEYS:
            out[key][i] = res[key]
    return Qs, out


def sweep_k(model, params, Q, k_min, k_max, n, n_beta=1001):
    ks = np.linspace(k_min, k_max, n)
    out = {key: np.zeros(n) for key in KEYS}
    for i, kv in enumerate(ks):
        sim = PlatformSimulator(Q=Q, k=kv, model=model, **params)
        res = _collect(sim, sim.optimize(n_beta))
        for key in KEYS:
            out[key][i] = res[key]
    return ks, out


def sweep_alpha(model, params, k, Q, a_min, a_max, n, n_beta=1001):
    alphas = np.linspace(a_min, a_max, n)
    out = {key: np.zeros(n) for key in KEYS}
    p = {kk: v for kk, v in params.items() if kk != 'alpha'}
    for i, a in enumerate(alphas):
        sim = PlatformSimulator(alpha=a, Q=Q, k=k, model=model, **p)
        res = _collect(sim, sim.optimize(n_beta))
        for key in KEYS:
            out[key][i] = res[key]
    return alphas, out


def sweep_delta(model, params, k, Q, d_min, d_max, n, n_beta=1001):
    deltas = np.linspace(d_min, d_max, n)
    out = {key: np.zeros(n) for key in KEYS}
    p = {kk: v for kk, v in params.items() if kk != 'delta'}
    for i, d in enumerate(deltas):
        sim = PlatformSimulator(delta=d, Q=Q, k=k, model=model, **p)
        res = _collect(sim, sim.optimize(n_beta))
        for key in KEYS:
            out[key][i] = res[key]
    return deltas, out
