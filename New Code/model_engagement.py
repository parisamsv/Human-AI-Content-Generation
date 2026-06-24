"""
model_engagement.py
-------------------
Engagement-based monetization model.

The platform pays the creator r per unit of effort (r * q_H).
The platform's per-consumer revenue equals the quality of content consumed.

Platform utility:
    u_P^E = q_A*D_A + q_H*D_H - r*q_H - (k/2)*(beta_H - 1/2)^2

Creator utility (after optimizing effort):
    u_H^E = (1/2)*q_H^2   [Corollary: same form as view-based]

The creator's FOC gives q_H* = r regardless of beta_H (effort = rate).

---------------------------------------------------------------------
Proposition (Platform's Joint Optimal Design, Engagement-Based):

  Define:
    N      = m_A + alpha*m_H
    P      = t*m_A - Q                             (> 0 by Assumption 2)
    Gamma  = m_A*(t*m_H - 1) - alpha^2*m_H         (> 0 by Assumption 3)
    Lambda = t*(2 - delta) - (1 - alpha)^2
    N_c    = t*(alpha*m_H + m_A) - 2*(1-alpha)*Q
    r_b    = m_H*P / N                             [boundary compensation rate]

  Three candidate optimal rates, one per demand regime:
    r1 = alpha*Q*m_H / Gamma   [uncovered interior FOC]
    r2 = N_c / (2*Lambda)      [covered interior FOC]

  Selection rule (from the proposition):
    r* = r1   if r1 <= r_b   (uncovered optimum is feasible)
    r* = r2   if r2 >= r_b   (covered optimum is feasible, r1 > r_b)
    r* = r_b  otherwise      (optimum is at the demand boundary)

  Matching M(beta_H) by regime:

    Case 1 (r* = r1, uncovered):
      M = -delta*Q^2*[(t*m_H - 1)^2 - alpha^2] / (t*Gamma^2)

    Case 2 (r* = r2, covered):
      M = delta / (2*Lambda*(2-delta)) * [(1-alpha)*N_c - 2*Q*Lambda]

    Case 3 (r* = r_b, boundary):
      M = -delta*[2*alpha + 2*r_b*((1-alpha) - t*m_H)/(t*m_H)]
              * (t*m_A^2 - alpha*t*m_H^2 - Q*(2-delta)) / N^2
          + t*delta * [2*r_b/(t*m_H) - 1]

  beta_H* solves: M(beta_H*) = k*(beta_H* - 0.5), clipped to [0, 1].
---------------------------------------------------------------------
"""
from common import m_A, m_H
from solver import find_beta_star


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _scalars(beta_H: float, Q: float, alpha: float,
             delta: float, t: float) -> dict:
    """
    Collect all the named scalars used across r_star and M computations.
    Returns a dict so callers can pick what they need by name.
    """
    mA     = m_A(beta_H, delta)
    mH     = m_H(beta_H, delta)
    N      = mA + alpha * mH
    P      = t * mA - Q

    Gamma  = mA * (t * mH - 1.0) - alpha ** 2 * mH   # > 0 by Assumption 3
    Lambda = t * (2.0 - delta) - (1.0 - alpha) ** 2
    N_c    = t * (alpha * mH + mA) - 2.0 * (1.0 - alpha) * Q
    r_b    = mH * P / N                               # boundary rate

    return dict(mA=mA, mH=mH, N=N, P=P,
                Gamma=Gamma, Lambda=Lambda, N_c=N_c, r_b=r_b)


def _regime_and_rate(s: dict, Q: float, alpha: float) -> tuple:
    """
    Select the optimal compensation rate and return (regime, r*).

    regime is one of: 'uncov', 'cov', 'bnd'
    """
    Gamma, Lambda = s["Gamma"], s["Lambda"]
    mH, N_c, r_b = s["mH"], s["N_c"], s["r_b"]

    # Guard: if assumptions are violated return boundary rate
    if Gamma <= 0 or Lambda <= 0:
        # return error
        print("Error: Assumptions violated: Gamma <= 0 or Lambda <= 0.")
        return

    r1 = alpha * Q * mH / Gamma
    r2 = N_c / (2.0 * Lambda)

    if r1 <= r_b:
        return "uncov", r1
    if r2 >= r_b:
        return "cov", r2
    return "bnd", r_b


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def r_star_eng(beta_H: float, Q: float, alpha: float,
               delta: float, t: float) -> float:
    """
    Optimal compensation rate for engagement-based at a given beta_H.

    Selects among the three candidates (r1, r2, r_b) following the
    proposition's selection rule.
    """
    s = _scalars(beta_H, Q, alpha, delta, t)
    _, r = _regime_and_rate(s, Q, alpha)
    return r


def M_eng(beta_H: float, Q: float, alpha: float,
          delta: float, t: float) -> float:
    """
    Marginal base revenue M(beta_H) for engagement-based.

    Piecewise formula from Proposition (Platform's Joint Optimal Design,
    Engagement-Based), Part 2.  The case matches that of r_star_eng.
    """
    s = _scalars(beta_H, Q, alpha, delta, t)
    mA, mH, N = s["mA"], s["mH"], s["N"]
    Gamma, Lambda, N_c, r_b = s["Gamma"], s["Lambda"], s["N_c"], s["r_b"]
    regime, _ = _regime_and_rate(s, Q, alpha)

    if regime == "uncov":
        # Case 1: uncovered interior optimum r* = r1
        M = -delta * Q ** 2 * ((t * mH - 1.0) ** 2 - alpha ** 2) / (t * Gamma ** 2)

    elif regime == "cov":
        # Case 2: covered interior optimum r* = r2
        M = (delta / (2.0 * Lambda * (2.0 - delta))
             * ((1.0 - alpha) * N_c - 2.0 * Q * Lambda))

    else:
        # Case 3: boundary optimum r* = r_b
        # W = t*m_A^2 - alpha*t*m_H^2 - Q*(2-delta)
        W = t * mA ** 2 - alpha * t * mH ** 2 - Q * (2.0 - delta)
        # bracket = 2*alpha + 2*r_b*((1-alpha) - t*m_H) / (t*m_H)
        bracket = 2.0 * alpha + 2.0 * r_b * ((1.0 - alpha) - t * mH) / (t * mH)
        M = (-delta * bracket * W / N ** 2
             + t * delta * (2.0 * r_b / (t * mH) - 1.0))

    return M


# ---------------------------------------------------------------------------
# Solving for beta_H*
# ---------------------------------------------------------------------------

def _u_P_eng(beta_H: float, Q: float, alpha: float,
             delta: float, t: float, k: float) -> float:
    """
    Platform utility under engagement-based at a given beta_H.
 
    u_P = TE - r*q_H - (k/2)*(beta_H - 0.5)^2
    """
    mA = m_A(beta_H, delta)
    mH = m_H(beta_H, delta)
    N   = mA + alpha * mH
    P   = t * mA - Q
    r_b = mH * P / N
    r   = r_star_eng(beta_H, Q, alpha, delta, t)
    q_H = r
    q_A = alpha * q_H + Q
    if r <= r_b + 1e-10:
        D_A = q_A / (t * mA)
        D_H = q_H / (t * mH)
    else:
        m_sum = mA + mH
        D_A = max(0.0, (t * mH + q_A - q_H) / (t * m_sum))
        D_H = max(0.0, (t * mA - q_A + q_H) / (t * m_sum))
    TE  = q_A * D_A + q_H * D_H
    return TE - r * q_H - 0.5 * k * (beta_H - 0.5) ** 2
 
 
def solve_beta_star_eng(Q: float, alpha: float,
                        delta: float, t: float, k: float) -> float:
    """
    Find the optimal algorithmic weight beta_H* for engagement-based.
 
    Calls solver.find_beta_star with M_eng as the marginal revenue function
    and _u_P_eng as the utility function for corner comparison.
    beta_H* is the unique solution to:
        M(beta_H*) = k*(beta_H* - 0.5)
    clipped to [0, 1].
    """
    def Mf(b: float) -> float:
        return M_eng(b, Q, alpha, delta, t)
 
    def Uf(b: float) -> float:
        return _u_P_eng(b, Q, alpha, delta, t, k)
 
    return find_beta_star(Mf, k, Uf)


# ---------------------------------------------------------------------------
# Full equilibrium
# ---------------------------------------------------------------------------

def equilibrium_eng(Q: float, alpha: float,
                    delta: float, t: float, k: float) -> dict:
    """
    Compute the full equilibrium under engagement-based monetization.

    Steps:
      1. Solve for beta_H* (Proposition, Part 2).
      2. Compute r* (Proposition, Part 1).
      3. Creator effort: q_H* = r*  (from the creator's FOC r - q_H = 0).
      4. AI quality: q_A = alpha*q_H* + Q.
      5. Demands:
           - If r* <= r_b (uncovered/boundary):
               D_A = q_A/(t*m_A),  D_H = q_H*/(t*m_H)
           - If r* > r_b (covered):
               Hotelling indifferent-consumer split
               D_A = (t*m_H + q_A - q_H) / (t*(m_A + m_H))
               D_H = (t*m_A - q_A + q_H) / (t*(m_A + m_H))
      6. Platform and creator utilities.

    Returns
    -------
    dict with keys:
        beta_H  – optimal algorithmic weight
        r       – optimal compensation rate
        q_H     – creator equilibrium effort
        q_A     – AI content quality
        D_A     – AI consumer demand
        D_H     – human consumer demand
        u_P     – platform utility
        u_H     – creator utility
        TV      – total views  (D_A + D_H)
        TE      – total engagement  (q_A*D_A + q_H*D_H)
    """
    beta_H = solve_beta_star_eng(Q, alpha, delta, t, k)

    mA = m_A(beta_H, delta)
    mH = m_H(beta_H, delta)
    N  = mA + alpha * mH
    P  = t * mA - Q
    r_b = mH * P / N

    r   = r_star_eng(beta_H, Q, alpha, delta, t)
    q_H = r                       # engagement-based: effort = rate
    q_A = alpha * q_H + Q         # AI quality

    if r <= r_b + 1e-10:
        # Uncovered (or exactly at the demand boundary)
        D_A = q_A / (t * mA)
        D_H = q_H / (t * mH)
    else:
        # Covered: Hotelling indifferent-consumer split
        # x* = (q_A - q_H + t*m_H) / (t*(m_A + m_H))
        # D_A = x*,  D_H = 1 - x*
        m_sum = mA + mH             # = 2 - delta
        D_A = max(0.0, (t * mH + q_A - q_H) / (t * m_sum))
        D_H = max(0.0, (t * mA - q_A + q_H) / (t * m_sum))

    TV  = D_A + D_H
    TE  = q_A * D_A + q_H * D_H

    # Platform utility: engagement revenue minus compensation minus penalty
    u_P = TE - r * q_H - 0.5 * k * (beta_H - 0.5) ** 2
    # Creator utility: half the square of effort (Corollary lem:uc)
    u_H = 0.5 * q_H ** 2

    s = _scalars(beta_H, Q, alpha, delta, t)
    regime, _ = _regime_and_rate(s, Q, alpha)

    return dict(
        beta_H=beta_H, r=r, q_H=q_H, q_A=q_A,
        D_A=D_A, D_H=D_H, u_P=u_P, u_H=u_H, TV=TV, TE=TE, Market=regime
    )
