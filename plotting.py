"""
Plotting functions for the platform revenue simulation.
"""

import matplotlib.pyplot as plt
from matplotlib import rcParams
from config import PLOT_CONFIG

def configure_plots():
    """Apply plot settings from config."""
    for key, value in PLOT_CONFIG.items():
        try:
            rcParams[key] = value
        except Exception as e:
            print(f"Could not set plot config '{key}': {e}")

def create_comparison_plots(Q_values, results_view, results_engagement, save_path=None):
    """
    Create a 2x2 grid of comparison plots.
    """
    configure_plots()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Optimal Promotion Weight
    ax = axes[0, 0]
    ax.plot(Q_values, results_view['beta_h'], 'o-', label='View-Based', 
            linewidth=2, markersize=6, color='#1f77b4')
    ax.plot(Q_values, results_engagement['beta_h'], 's-', label='Engagement-Based', 
            linewidth=2, markersize=6, color='#ff7f0e')
    ax.set_xlabel(r'AI Baseline Quality $Q$')
    ax.set_ylabel(r'Optimal Promotion Weight $\beta_h^*$')
    ax.set_title(r'Panel (a): Optimal Promotion Weight')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best')
    ax.set_ylim([-0.05, 1.05])
    
    # Plot 2: Platform Utility
    ax = axes[0, 1]
    ax.plot(Q_values, results_view['utility'], 'o-', label='View-Based', 
            linewidth=2, markersize=6, color='#1f77b4')
    ax.plot(Q_values, results_engagement['utility'], 's-', label='Engagement-Based', 
            linewidth=2, markersize=6, color='#ff7f0e')
    ax.set_xlabel(r'AI Baseline Quality $Q$')
    ax.set_ylabel(r'Platform Utility $u_p^*$')
    ax.set_title(r'Panel (b): Platform Utility')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best')
    
    # Plot 3: Human Content Quality
    ax = axes[1, 0]
    ax.plot(Q_values, results_view['q_h'], 'o-', label='View-Based', 
            linewidth=2, markersize=6, color='#1f77b4')
    ax.plot(Q_values, results_engagement['q_h'], 's-', label='Engagement-Based', 
            linewidth=2, markersize=6, color='#ff7f0e')
    ax.set_xlabel(r'AI Baseline Quality $Q$')
    ax.set_ylabel(r'Human Content Quality $q_h^*$')
    ax.set_title(r'Panel (c): Human Content Quality')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best')
    
    # Plot 4: AI Content Quality
    ax = axes[1, 1]
    ax.plot(Q_values, results_view['q_A'], 'o-', label='View-Based', 
            linewidth=2, markersize=6, color='#1f77b4')
    ax.plot(Q_values, results_engagement['q_A'], 's-', label='Engagement-Based', 
            linewidth=2, markersize=6, color='#ff7f0e')
    ax.set_xlabel(r'AI Baseline Quality $Q$')
    ax.set_ylabel(r'AI Content Quality $q_A^*$')
    ax.set_title(r'Panel (d): AI Content Quality')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best')
    
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.4)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
        print(f"\nPlot saved to {save_path}")
    
    return fig, axes
