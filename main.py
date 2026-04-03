"""
Main entry point for running the platform revenue simulation.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from config import PARAMS, Q_SWEEP
from simulator import PlatformRevenueSimulator
from plotting import create_comparison_plots

def sweep_Q_values(Q_min, Q_max, n_points, params):
    """
    Sweep across AI baseline quality values and optimize both settings.
    """
    Q_values = np.linspace(Q_min, Q_max, n_points)
    
    results_view = {k: [] for k in ['beta_h', 'r', 'utility', 'q_h', 'q_A']}
    results_engagement = {k: [] for k in ['beta_h', 'r', 'utility', 'q_h', 'q_A']}
    
    print(f"{'Q':<8} {'View β_h':<12} {'Eng β_h':<12} {'View u_p':<12} {'Eng u_p':<12}")
    print("-" * 60)
    
    for Q in Q_values:
        sim = PlatformRevenueSimulator(Q=Q, **params)
        
        # Optimize both settings
        result_view = sim.optimize_view_based()
        result_eng = sim.optimize_engagement_based()
        
        # Store results
        for key in results_view:
            results_view[key].append(result_view[key])
            results_engagement[key].append(result_eng[key])
            
        print(f"{Q:<8.2f} {result_view['beta_h']:<12.4f} {result_eng['beta_h']:<12.4f} "
              f"{result_view['utility']:<12.4f} {result_eng['utility']:<12.4f}")
    
    # Convert to numpy arrays
    for key in results_view:
        results_view[key] = np.array(results_view[key])
        results_engagement[key] = np.array(results_engagement[key])
    
    return Q_values, results_view, results_engagement

def main():
    """Main execution function."""
    print("=" * 60)
    print("Platform Revenue Settings Comparison")
    print("=" * 60)
    
    print("\nFixed Parameters:")
    for key, value in PARAMS.items():
        print(f"  {key}: {value}")
    print(f"\nVarying AI Baseline Quality (Q) from {Q_SWEEP['min']} to {Q_SWEEP['max']}")
    print("=" * 60)
    
    # Sweep across Q values
    Q_values, results_view, results_engagement = sweep_Q_values(
        Q_SWEEP['min'], Q_SWEEP['max'], Q_SWEEP['points'], PARAMS
    )
    
    print("\n" + "=" * 60)
    print("Generating comparison plots...")
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(output_dir, 'platform_revenue_comparison.pdf')
    
    create_comparison_plots(Q_values, results_view, results_engagement, save_path)
    
    plt.show()
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
