"""
common.py
---------
Shared primitives used by both the view-based and engagement-based models.

The mismatch multipliers come from the Hotelling setup:
  m_A(beta_H) = 1 - delta*(1 - beta_H)   (AI side)
  m_H(beta_H) = 1 - delta*beta_H          (Human side)

When beta_H = 0 the algorithm fully promotes AI, reducing AI mismatch.
When beta_H = 1 the algorithm fully promotes Human, reducing human mismatch.
Note that m_A + m_H = 2 - delta for all beta_H.
"""


def m_A(beta_H: float, delta: float) -> float:
    """AI mismatch multiplier: m_A = 1 - delta*(1 - beta_H)."""
    return 1.0 - delta * (1.0 - beta_H)


def m_H(beta_H: float, delta: float) -> float:
    """Human mismatch multiplier: m_H = 1 - delta*beta_H."""
    return 1.0 - delta * beta_H
