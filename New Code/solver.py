"""
solver.py
---------
Finds the platform's optimal algorithmic weight beta_H* by solving its
first-order condition (FOC), clipped to [0, 1].

Both models (view-based and engagement-based) share the same structure
for beta_H*:

    beta_H* = max(0, min(1, 1/2 + M(beta_H*) / k))

which is equivalent to the bounded FOC:

    M(beta_H) = k * (beta_H - 0.5)      with beta_H in [0, 1].

Rearranged: f(beta_H) = M(beta_H) - k*(beta_H - 0.5) = 0.

Under the large-k assumption (Assumption 1), the platform utility is
strictly concave in beta_H, so f is strictly decreasing and the FOC has
exactly one solution in [0, 1].  We find it with Brent's method.
"""
from scipy.optimize import brentq


def find_beta_star(M_func, k: float, tol: float = 1e-12) -> float:
    """
    Solve the platform's constrained FOC for beta_H*.

    Parameters
    ----------
    M_func : callable
        M_func(beta_H) returns the marginal base revenue at beta_H.
        This is the derivative of the platform's base utility (without
        the penalty term) with respect to beta_H.
    k : float
        Algorithmic-neutrality penalty (must be positive).
    tol : float
        Absolute tolerance for the root-finder.

    Returns
    -------
    float
        Optimal beta_H* in [0, 1].
    """
    def foc(b: float) -> float:
        # Positive ↔ platform wants to increase beta_H
        # Negative ↔ platform wants to decrease beta_H
        return M_func(b) - k * (b - 0.5)

    # --- Corner solutions ---
    # If foc(0) <= 0, the FOC is already non-positive at the left boundary,
    # meaning the platform prefers to go left of 0 → corner at 0.
    if foc(0.0) <= 0.0:
        return 0.0

    # If foc(1) >= 0, the platform prefers to go right of 1 → corner at 1.
    if foc(1.0) >= 0.0:
        return 1.0

    # --- Interior solution ---
    # foc(0) > 0 and foc(1) < 0, so a unique root exists in (0, 1).
    return brentq(foc, 0.0, 1.0, xtol=tol, full_output=False)
