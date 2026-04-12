"""
Main entry point for running the platform revenue simulation.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from config import PARAMS
from simulator import PlatformRevenueSimulator
from plotting import create_comparison_plots

def sweep_Q_values(Q_min, Q_max, n_points, params):
    """
    Sweep across platform quality values and optimize both settings.
    """
    Q_values = np.linspace(Q_min, Q_max, n_points)
    
    results_view = {k: [] for k in ['beta_h', 'r', 'utility', 'q_h', 'q_A']}
    results_engagement = {k: [] for k in ['beta_h', 'r', 'utility', 'q_h', 'q_A']}
    
    print(f"{'Q':<8} {'View β_h':<12} {'Eng β_h':<12} {'View u_p':<12} {'Eng u_p':<12}")
    print("-" * 60)
    
    for Q_v in Q_values:
        sim = PlatformRevenueSimulator(Q=Q_v, **params)
        
        # Optimize both settings
        # result_view = sim.optimize_view_based()
        # result_eng = sim.optimize_engagement_based()
        result_view = sim.optimize_general('view')
        result_eng = sim.optimize_general('engagement')
        
        # Store results
        for key in results_view:
            results_view[key].append(result_view[key])
            results_engagement[key].append(result_eng[key])
            
        print(f"{Q_v:<8.2f} {result_view['beta_h']:<12.4f} {result_eng['beta_h']:<12.4f} "
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
    params_dict = {}
    for key, value in PARAMS.items():
        print(f"  {key}: {value}")
        params_dict[key] = value

    Q_SWEEP = {
    'min': 0,
    'max': 1,
    'points': 21
    }
    
    print(f"\nVarying Platform Quality (Q) from {Q_SWEEP['min']} to {Q_SWEEP['max']}")
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
