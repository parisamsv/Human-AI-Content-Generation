"""
Functional interface to the platform model (vectorized over beta).

All economic equations live here; interior_search.py calls these
functions and contains no model logic of its own.

The utility is the engagement-based objective from simulator.py:
    u_p = q_A * D_A + (q_h - r) * D_h
where r is the analytically optimal revenue share for each beta
(derived from simulator.optimal_r_engagement, vectorised).
"""

import numpy as np


def platform_utility_vec(betas, alpha, Q, delta, u0, t=1.0):
    """
    Engagement-based platform utility for an array of promotion weights.

    Parameters
    ----------
    betas : array-like
        Promotion weights for human content, each in [0, 1].
    alpha : float
        AI learning efficiency.
    Q : float
        AI baseline content quality.
    delta : float
        Platform algorithmic sensitivity.
    u0 : float
        Consumer outside option.
    t : float
        Consumer transportation cost (default 1.0).

    Returns
    -------
    ndarray
        Platform utility at each beta, with r optimised analytically
        (mirrors simulator.optimal_r_engagement + platform_utility_engagement).
    """
    betas = np.asarray(betas, dtype=float)

    # ── mismatch costs (simulator.mismatch_costs) ────────────────────────────
    m_H = 1.0 - betas * delta                  # human mismatch
    m_A = 1.0 - delta * (1.0 - betas)          # AI mismatch (= 1 - delta + beta*delta)

    # ── analytically optimal revenue share (simulator.optimal_r_engagement) ──
    one_minus_tmH = 1.0 - t * m_H              # = beta * delta when t = 1
    numer = t * m_H * (u0 * m_A * one_minus_tmH
                        - alpha * m_H * (2.0 * Q - u0))
    denom = 2.0 * (alpha**2 * t * m_H + m_A * one_minus_tmH)
    r = np.maximum(0.0, np.where(denom > 0.0, numer / denom, 0.0))

    # ── creator qualities (simulator.creator_qualities) ──────────────────────
    q_h = r / (t * m_H)
    q_A = alpha * r / (t * m_H) + Q

    # ── consumer demands with boundary conditions (simulator.demands) ─────────
    D_h = np.maximum(0.0, (q_h - u0) / (t * m_H))
    D_A = np.maximum(0.0, (q_A - u0) / (t * m_A))

    # ── engagement utility (simulator.platform_utility_engagement) ────────────
    return q_A * D_A + (q_h - r) * D_h
