"""
solver.py
---------
Finds the platform's optimal algorithmic weight beta_H* in [0, 1].

The platform's utility is:

    U(beta) = U_base(beta) - (k/2) * (beta - 0.5)^2

Its derivative is the first-order condition (FOC):

    f(beta) = M(beta) - k * (beta - 0.5)       where M = dU_base/dbeta

Under Assumption 1 (k large), U is globally concave, f is strictly
decreasing, and the unique interior root of f is a maximum.

When Assumption 1 fails, U can be globally convex: f is then strictly
increasing and any interior root is a minimum.  The maximum is then at
one of the corners {0, 1}.

Because the utility is either globally concave or globally convex, f
has at most one interior root.  It suffices to:

  1. Look for the unique root of f in (0, 1).
  2. Check the SOC at the root.
       M'(root) < k  →  local maximum  →  return it.
       M'(root) ≥ k  →  local minimum  →  compare corners.
  3. Compare corners by evaluating U directly at 0 and 1.
"""

from scipy.optimize import brentq

_FD_EPS = 1e-6   # step size for the central-difference estimate of M'


def find_beta_star(M_func, k: float, utility_func,
                   tol: float = 1e-12) -> float:
    """
    Find beta_H* in [0, 1] that maximises the platform's utility.

    Parameters
    ----------
    M_func : callable
        M_func(beta) = dU_base/dbeta, the marginal base revenue.
    k : float
        Algorithmic-neutrality penalty (positive).
    utility_func : callable
        utility_func(beta) = U(beta), the full platform utility at beta.
        Used to compare corners when the interior root is a minimum.
    tol : float
        Absolute tolerance for brentq.

    Returns
    -------
    float
        Optimal beta_H* in [0, 1].
    """
    def foc(b: float) -> float:
        return M_func(b) - k * (b - 0.5)

    f0 = foc(0.0)
    f1 = foc(1.0)

    # ── Step 1: look for an interior root ─────────────────────────────────
    if f0 * f1 < 0.0:
        root = brentq(foc, 0.0, 1.0, xtol=tol)

        # ── Step 2: second-order condition ────────────────────────────────
        # M'(root) < k  ↔  d²U/dbeta² < 0  ↔  local maximum.
        M_prime = (M_func(root + _FD_EPS) - M_func(root - _FD_EPS)) / (2.0 * _FD_EPS)
        if M_prime < k:
            return root     # concave case: interior root is the global max

        # M'(root) ≥ k  →  local minimum; fall through to corner comparison.

    # ── Step 3: compare corners ───────────────────────────────────────────
    return 1.0 if utility_func(1.0) > utility_func(0.0) else 0.0