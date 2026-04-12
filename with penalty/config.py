"""
Configuration for the platform model with discrimination penalty.
"""

# View-based: Assumption 1 sufficient condition alpha/(1-delta) + t*u0 < 1
VIEW_PARAMS = {'alpha': 0.5, 'delta': 0.3, 'u_0': 0.2, 't': 1.0}
# Check: 0.5/0.7 + 1.0*0.2 = 0.914 < 1 ✓

# Engagement-based: SOC sufficient condition t > 1/(1-delta) + alpha^2
ENG_PARAMS = {'alpha': 0.5, 'delta': 0.3, 'u_0': 0.2, 't': 2.0}
# Check: 1/0.7 + 0.25 = 1.68 < 2.0 ✓

K_VALUES = [0.0, 1.0, 2.0, 5.0, 10.0]

Q_SWEEP     = {'min': 0.0, 'max': 2.0, 'points': 101}
K_SWEEP     = {'min': 0.0, 'max': 10.0, 'points': 51}
ALPHA_SWEEP = {'min': 0.05, 'max': 0.7, 'points': 51}
DELTA_SWEEP = {'min': 0.05, 'max': 0.8, 'points': 51}

BETA_GRID_POINTS = 1001

PLOT_CONFIG = {'font.family': 'serif', 'font.size': 11,
               'text.usetex': False, 'figure.dpi': 150}
OUTPUT_DIR = 'figures'
