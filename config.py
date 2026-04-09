"""
Configuration file for the platform revenue simulation.
Stores fixed parameters and settings for the analysis.
"""

# Fixed parameters for the simulation
PARAMS = {
    'alpha': 0.5,  # AI learning efficiency
    'u_0': 0,    # Consumer outside option
    'delta': 0.3,   # Platform algorithmic sensitivity
    't_v': 2.0,   # Transportation cost (will be varied in the analysis)
    't_e': 2.0
}

# Matplotlib configuration for plots
PLOT_CONFIG = {
    'font.family': 'serif',
    'font.size': 10,
    'text.usetex': False,  # Set to True if LaTeX is installed
    'figure.dpi': 150
}
