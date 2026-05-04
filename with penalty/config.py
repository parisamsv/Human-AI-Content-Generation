"""
Configuration for the platform model with discrimination penalty.
"""

PARAMS = {'alpha': 0.3, 'delta': 0.4, 'Q': 0.3, 'k': 1.0, 'u_0': 0}

K_VALUES = [0.0, 1.0, 2.0, 5.0]
K_COLORS = {0.0:"#1c80ab", 1.0:"#d6ca25", 2.0:"#c98622",
             5.0:"#8e1502"}

ALPHA_VALUES = [0.1, 0.2, 0.4, 0.7]
ALPHA_COLORS = {0.1:'#1c80ab', 0.2:'#d6ca25', 0.4:"#c98622",
                0.7:'#8e1502'}
DELTA_VALUES = [0.1, 0.2, 0.4]
DELTA_COLORS = {0.1:'#1c80ab', 0.2:'#d6ca25', 0.4:"#c98622",}

Q_VALUES = [0.0, 0.2, 0.5, 1.0]
Q_COLORS = {0.0:'#1c80ab', 0.2:'#d6ca25', 0.5:"#c98622", 1.0:'#8e1502'}

Q_SWEEP     = {'min': 0.0, 'max': 2.0, 'points': 101}
K_SWEEP     = {'min': 0.0, 'max': 10.0, 'points': 51}
ALPHA_SWEEP = {'min': 0.05, 'max': 0.7, 'points': 51}
DELTA_SWEEP = {'min': 0.05, 'max': 0.5, 'points': 51}

BETA_GRID_POINTS = 1001

PLOT_CONFIG = {'font.family': 'serif', 'font.size': 11,
               'text.usetex': False, 'figure.dpi': 150}
OUTPUT_DIR = 'figures'
