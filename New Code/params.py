"""
params.py
---------
Baseline parameters and experiment ranges (Table 1 in the paper).

Assumption checks at the baseline (beta_H = 0.5 → m_A = m_H = 1 - delta/2 = 0.80):
  Assumption 2 (nonsat):
    Q < t*(1-delta)  →  0.30 < 2.50*0.60 = 1.50  ✓
    t > 1/(1-delta)  →  2.50 > 1/0.60 ≈ 1.67      ✓
  Assumption 3 (concavity-E):
    t > alpha^2 + 1/(1-delta)  →  2.50 > 0.09 + 1.67 = 1.76  ✓
"""

# ---------------------------------------------------------------------------
# Baseline parameter values
# ---------------------------------------------------------------------------
BASELINE = {
    "Q":     1.00,   # AI baseline quality
    "alpha": 0.40,   # AI learning efficiency
    "delta": 0.50,   # algorithmic influence
    "t":     2.50,   # consumer mismatch cost
    "k":     2.50,   # algorithmic-neutrality penalty
}

# ---------------------------------------------------------------------------
# Ranges for one-at-a-time comparative statics
# ---------------------------------------------------------------------------
RANGES = {
    "Q":     (0.05, 1.30),
    "alpha": (0.10, 0.70),
    "delta": (0.10, 0.50),
}

# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------
N_POINTS = 300   # grid points for 1-D comparative statics
N_GRID   = 300    # grid points per axis for 2-D preference maps
