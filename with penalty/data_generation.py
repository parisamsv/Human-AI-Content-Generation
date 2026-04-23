"""
Sweep functions for generating equilibrium data.
"""
import numpy as np
from simulator import PlatformSimulator_ViewComp, PlatformSimulator_EngComp

KEYS = ['beta_h', 'r', 'pu', 'cu', 'tv', 'te']


def _collect(sim, res):
    b, r = res['beta_h'], res['r']
    res['pu'] = sim.utility(b, r)
    res['cu'] = sim.creator_utility(b, r)
    res['tv'] = sim.total_view(b, r)
    res['te'] = sim.total_engagement(b, r)
    return res


def sweep_Q(rev_model, comp_model, params, Q_min, Q_max, n, n_beta=1001):
    Qs = np.linspace(Q_min, Q_max, n)   # list of Q values to sweep over
    out = {key: np.zeros(n) for key in KEYS} # dictionary to store results for each metric for each Q
    if comp_model == 'view':
        for i, Q in enumerate(Qs):
            params['Q'] = Q
            sim = PlatformSimulator_ViewComp(model=rev_model, **params)
            res = _collect(sim, sim.optimize(n_beta))
            for key in KEYS:
                out[key][i] = res[key]
    elif comp_model == 'engagement':
        for i, Q in enumerate(Qs):
            params['Q'] = Q
            sim = PlatformSimulator_EngComp(model=rev_model, **params)
            res = _collect(sim, sim.optimize(n_beta))
            for key in KEYS:
                out[key][i] = res[key]
    return Qs, out


def sweep_k(rev_model, comp_model, params, k_min, k_max, n, n_beta=1001):
    ks = np.linspace(k_min, k_max, n)
    out = {key: np.zeros(n) for key in KEYS}
    if comp_model == 'view':
        for i, kv in enumerate(ks):
            params['k'] = kv
            sim = PlatformSimulator_ViewComp(model=rev_model, **params)
            res = _collect(sim, sim.optimize(n_beta))
            for key in KEYS:
                out[key][i] = res[key]
    elif comp_model == 'engagement':
        for i, kv in enumerate(ks):
            params['k'] = kv
            sim = PlatformSimulator_EngComp(model=rev_model, **params)
            res = _collect(sim, sim.optimize(n_beta))
            for key in KEYS:
                out[key][i] = res[key]
    return ks, out


def sweep_alpha(rev_model, comp_model, params, a_min, a_max, n, n_beta=1001):
    alphas = np.linspace(a_min, a_max, n)
    out = {key: np.zeros(n) for key in KEYS}
    if comp_model == 'view':
        for i, a in enumerate(alphas):
            params['alpha'] = a
            sim = PlatformSimulator_ViewComp(model=rev_model, **params)
            res = _collect(sim, sim.optimize(n_beta))
            for key in KEYS:
                out[key][i] = res[key]
    elif comp_model == 'engagement':
        for i, a in enumerate(alphas):
            params['alpha'] = a
            sim = PlatformSimulator_EngComp(model=rev_model, **params)
            res = _collect(sim, sim.optimize(n_beta))
            for key in KEYS:
                out[key][i] = res[key]
    return alphas, out


def sweep_delta(rev_model, comp_model, params, d_min, d_max, n, n_beta=1001):
    deltas = np.linspace(d_min, d_max, n)
    out = {key: np.zeros(n) for key in KEYS}
    if comp_model == 'view':
        for i, d in enumerate(deltas):
            params['delta'] = d
            sim = PlatformSimulator_ViewComp(model=rev_model, **params)
            res = _collect(sim, sim.optimize(n_beta))
            for key in KEYS:
                out[key][i] = res[key]
    elif comp_model == 'engagement':
        for i, d in enumerate(deltas):
            params['delta'] = d
            sim = PlatformSimulator_EngComp(model=rev_model, **params)
            res = _collect(sim, sim.optimize(n_beta))
            for key in KEYS:
                out[key][i] = res[key]
    return deltas, out
