"""
Configuration file for the platform revenue simulation.
Stores fixed parameters and settings for the analysis.
"""

# Fixed parameters for the simulation
PARAMS = {
    't': 10.0,      # Consumer transportation cost
    'alpha': 0.5,  # AI learning efficiency
    'u_0': 0.6,    # Consumer outside option
    'delta': 0.8   # Platform algorithmic sensitivity
}

# Q value sweep configuration
Q_SWEEP = {
    'min': 0,
    'max': 1,
    'points': 21
}

# Matplotlib configuration for plots
PLOT_CONFIG = {
    'font.family': 'serif',
    'font.size': 10,
    'text.usetex': False,  # Set to True if LaTeX is installed
    'figure.dpi': 150
}
