"""
model_view.py
-------------
View-based monetization model.

The platform pays the creator r per view (r * D_H).
The platform's per-view revenue is 1 unit regardless of content type.

Platform utility:
    u_P^V = D_A + D_H - r*D_H - (k/2)*(beta_H - 1/2)^2

Creator utility (after optimizing effort):
    u_H^V = (1/2)*q_H^2   [Corollary: creator utility = half squared effort]

---------------------------------------------------------------------
Proposition (Platform's Joint Optimal Design, View-Based):

  r*(beta_H) = min(r_unc, r_bnd)
    r_unc = (m_A + alpha*m_H) / (2*m_A)        [uncovered interior FOC]
    r_bnd = t*m_H^2*(t*m_A - Q) / N            [boundary rate, N = m_A + alpha*m_H]

  Because the platform's covered utility is strictly decreasing in r,
  and the boundary utility is also decreasing in r, the optimal rate
  never exceeds r_bnd.  So r* <= r_bnd always.

  Since r* <= r_bnd, the creator is always in the uncovered or exactly
  at the boundary regime, giving:
    q_H* = r* / (t*m_H)

  M(beta_H) is piecewise across two regimes:

    Uncovered branch (r_unc <= r_bnd, equiv. N^2 <= 2*t*m_A*m_H^2*P):
      M = delta * [(1/m_H + alpha/m_A)*(1/m_H^2 - alpha/m_A^2) / (2*t^2)
                   - Q / (t*m_A^2)]

    Boundary branch (r_unc > r_bnd):
      M = 2*delta*m_H*P / N^3 * (t*m_A^2 - alpha*t*m_H^2 - Q*(2-delta))

  where P = t*m_A - Q > 0 (by Assumption 2).

  beta_H* solves: M(beta_H*) = k*(beta_H* - 0.5), clipped to [0, 1].
---------------------------------------------------------------------
"""

from common import m_A, m_H
from solver import find_beta_star


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _scalars(beta_H: float, Q: float, alpha: float,
             delta: float, t: float) -> tuple:
    """
    Return (mA, mH, N, P) where:
      N = m_A + alpha*m_H
      P = t*m_A - Q  (> 0 by Assumption 2)
    """
    mA = m_A(beta_H, delta)
    mH = m_H(beta_H, delta)
    N  = mA + alpha * mH
    P  = t * mA - Q
    return mA, mH, N, P


def _is_uncovered(N: float, mA: float, mH: float,
                  P: float, t: float) -> bool:
    """
    Uncovered regime condition: r_unc <= r_bnd.

    r_unc = N / (2*mA),  r_bnd = t*mH^2*P / N.
    r_unc <= r_bnd  ↔  N^2 <= 2*t*mA*mH^2*P.
    """
    return N ** 2 <= 2.0 * t * mA * mH ** 2 * P


# ---------------------------------------------------------------------------
# Core functions (used directly and by equilibrium_view)
# ---------------------------------------------------------------------------

def r_star_view(beta_H: float, Q: float, alpha: float,
                delta: float, t: float) -> float:
    """
    Optimal compensation rate for view-based at a given beta_H.
    r* = min(r_unc, r_bnd).
    """
    mA, mH, N, P = _scalars(beta_H, Q, alpha, delta, t)
    r_unc = N / (2.0 * mA)
    r_bnd = t * mH ** 2 * P / N
    return min(r_unc, r_bnd)


def M_view(beta_H: float, Q: float, alpha: float,
           delta: float, t: float) -> float:
    """
    Marginal base revenue M(beta_H) for view-based.

    This is the derivative of the platform's utility (excluding the
    penalty term) with respect to beta_H, evaluated at r = r*(beta_H).

    Piecewise formula from Proposition (Platform's Joint Optimal Design,
    View-Based), Part 2:

    Uncovered branch (N^2 <= 2*t*m_A*m_H^2*P):
      M = delta * [(1/m_H + alpha/m_A)*(1/m_H^2 - alpha/m_A^2) / (2*t^2)
                   - Q / (t*m_A^2)]

    Boundary branch:
      M = 2*delta*m_H*P / N^3 * (t*m_A^2 - alpha*t*m_H^2 - Q*(2-delta))
    """
    mA, mH, N, P = _scalars(beta_H, Q, alpha, delta, t)

    if _is_uncovered(N, mA, mH, P, t):
        # --- Uncovered branch ---
        # Comes from differentiating U_base = (1+rho)^2/(4t^2*mH^2) + Q/(t*mA)
        # w.r.t. beta_H using the chain rule and rho = alpha*mH/mA.
        cross = (1.0 / mH + alpha / mA) * (1.0 / mH ** 2 - alpha / mA ** 2)
        return delta * (cross / (2.0 * t ** 2) - Q / (t * mA ** 2))
    else:
        # --- Boundary branch ---
        # Comes from differentiating U_base = 1 - (mH*P/N)^2 w.r.t. beta_H.
        W = t * mA ** 2 - alpha * t * mH ** 2 - Q * (2.0 - delta)
        return 2.0 * delta * mH * P / N ** 3 * W


# ---------------------------------------------------------------------------
# Solving for beta_H*
# ---------------------------------------------------------------------------

def solve_beta_star_view(Q: float, alpha: float,
                         delta: float, t: float, k: float) -> float:
    """
    Find the optimal algorithmic weight beta_H* for view-based.

    Calls solver.find_beta_star with M_view as the marginal revenue function.
    beta_H* is the unique solution to:
        M(beta_H*) = k*(beta_H* - 0.5)
    clipped to [0, 1].
    """
    def Mf(b: float) -> float:
        return M_view(b, Q, alpha, delta, t)

    return find_beta_star(Mf, k)


# ---------------------------------------------------------------------------
# Full equilibrium
# ---------------------------------------------------------------------------

def equilibrium_view(Q: float, alpha: float,
                     delta: float, t: float, k: float) -> dict:
    """
    Compute the full equilibrium under view-based monetization.

    Steps:
      1. Solve for beta_H* (Proposition, Part 2).
      2. Compute r* (Proposition, Part 1).
      3. Creator effort: q_H* = r* / (t*m_H).
         This formula holds in both the uncovered and boundary regimes
         because at r* = r_bnd the creator's optimal stall effort q_H_bar
         equals r_bnd / (t*m_H) exactly.
      4. AI quality: q_A = alpha*q_H* + Q.
      5. Demands: D_A = q_A/(t*m_A), D_H = q_H*/(t*m_H).
         These uncovered-formula demands also hold at the boundary
         (they sum to 1 there).
      6. Platform and creator utilities.

    Returns
    -------
    dict with keys:
        beta_H  – optimal algorithmic weight
        r       – optimal compensation rate
        q_H     – creator equilibrium effort (= quality)
        q_A     – AI content quality
        D_A     – AI consumer demand
        D_H     – human consumer demand
        u_P     – platform utility
        u_H     – creator utility
        TV      – total views  (D_A + D_H)
        TE      – total engagement  (q_A*D_A + q_H*D_H)
    """
    beta_H = solve_beta_star_view(Q, alpha, delta, t, k)

    mA = m_A(beta_H, delta)
    mH = m_H(beta_H, delta)

    r   = r_star_view(beta_H, Q, alpha, delta, t)
    q_H = r / (t * mH)              # creator effort
    q_A = alpha * q_H + Q           # AI quality

    D_A = q_A / (t * mA)            # AI demand
    D_H = q_H / (t * mH)            # human demand

    TV  = D_A + D_H
    TE  = q_A * D_A + q_H * D_H

    # Platform utility: total views minus compensation minus penalty
    u_P = TV - r * D_H - 0.5 * k * (beta_H - 0.5) ** 2
    # Creator utility: half the square of effort (Corollary lem:uc)
    u_H = 0.5 * q_H ** 2

    return dict(
        beta_H=beta_H, r=r, q_H=q_H, q_A=q_A,
        D_A=D_A, D_H=D_H, u_P=u_P, u_H=u_H, TV=TV, TE=TE,
    )
