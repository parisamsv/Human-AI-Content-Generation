"""
Configuration file for the platform revenue simulation.
Stores fixed parameters and settings for the analysis.
"""

# Fixed parameters for the simulation
PARAMS = {
    'alpha': 0.5,  # AI learning efficiency
    'u_0': 0.2,    # Consumer outside option
    'delta': 0.6,   # Platform algorithmic sensitivity
    't': 2.0,   # Transportation cost (will be varied in the analysis)
    'K': 1.0    # Platform preference for human creators
}

# Matplotlib configuration for plots
PLOT_CONFIG = {
    'font.family': 'serif',
    'font.size': 10,
    'text.usetex': False,  # Set to True if LaTeX is installed
    'figure.dpi': 150
}
