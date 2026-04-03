"""
Baseline parameters for the interior-solution search.
Keys match the keyword arguments of model.platform_utility_vec.
"""

BASELINE = {
    'alpha': 0.5,   # AI learning efficiency
    'Q':     0.5,   # AI baseline content quality
    'delta': 0.8,   # Platform algorithmic sensitivity
    'u0':    0.6,   # Consumer outside option
    't':     1.0,   # Consumer transportation cost (held fixed)
}
