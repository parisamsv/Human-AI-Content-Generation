"""
Configuration for the platform model with discrimination penalty.
"""

PARAMS = {'alpha': 0.5, 'delta': 0.3, 'k': 1.0, 'Q': 0.5}

K_VALUES = [0.0, 1.0, 2.0, 5.0]
K_COLORS = {0.0:'#1f77b4', 1.0:'#2ca02c', 2.0:"#b28b0b",
             5.0:'#d62728'}

ALPHA_VALUES = [0.1, 0.2, 0.4, 0.7]
ALPHA_COLORS = {0.1:'#1f77b4', 0.2:'#2ca02c', 0.4:"#b28b0b",
                0.7:'#d62728'}
DELTA_VALUES = [0.1, 0.2, 0.4, 0.8]
DELTA_COLORS = {0.1:'#1f77b4', 0.2:'#2ca02c', 0.4:"#b28b0b",
                0.8:'#d62728'}

Q_VALUES = [0.0, 0.2, 0.5, 1.0]
Q_COLORS = {0.0:'#1f77b4', 0.2:'#2ca02c', 0.5:"#b28b0b", 1.0:'#d62728'}

Q_SWEEP     = {'min': 0.0, 'max': 2.0, 'points': 101}
K_SWEEP     = {'min': 0.0, 'max': 10.0, 'points': 51}
ALPHA_SWEEP = {'min': 0.05, 'max': 0.7, 'points': 51}
DELTA_SWEEP = {'min': 0.05, 'max': 0.8, 'points': 51}

BETA_GRID_POINTS = 1001

PLOT_CONFIG = {'font.family': 'serif', 'font.size': 11,
               'text.usetex': False, 'figure.dpi': 150}
OUTPUT_DIR = 'figures'
