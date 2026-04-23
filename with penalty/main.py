
from config import *
from data_generation import sweep_Q, sweep_k, sweep_alpha, sweep_delta
from plotting import (plot_Q_sweep, plot_k_sweep, plot_alpha_sweep,
                      plot_delta_sweep, plot_platform_welfare)


def exp1_Q_sweep():
    print("\n=== Exp 1: Q sweep ===")
    for model, params, tag in [('view', VIEW_PARAMS, 'View'),
                                ('engagement', ENG_PARAMS, 'Eng')]:
        data = {}
        for k in K_VALUES:
            print(f"  {tag} k={k}")
            Qs, d = sweep_Q(model, params, k,
                            Q_SWEEP['min'], Q_SWEEP['max'],
                            Q_SWEEP['points'], BETA_GRID_POINTS)
            data[k] = d
        plot_Q_sweep(Qs, data, tag, f'exp1_Q_{tag.lower()}.pdf')
        plot_platform_welfare(Qs, data, tag, f'exp1_platform_{tag.lower()}.pdf')


def exp2_k_sweep():
    print("\n=== Exp 2: k sweep ===")
    Q_fixed = [0.0, 0.2, 0.5, 1.0, 2.0]
    for model, params, tag in [('view', VIEW_PARAMS, 'View'),
                                ('engagement', ENG_PARAMS, 'Eng')]:
        data = {}
        for Q in Q_fixed:
            print(f"  {tag} Q={Q}")
            ks, d = sweep_k(model, params, Q,
                            K_SWEEP['min'], K_SWEEP['max'],
                            K_SWEEP['points'], BETA_GRID_POINTS)
            data[Q] = d
        plot_k_sweep(ks, data, tag, f'exp2_k_{tag.lower()}.pdf')


def exp3_alpha():
    print("\n=== Exp 3: alpha sweep ===")
    Q_fixed = 0.5
    for model, params, tag in [('view', VIEW_PARAMS, 'View'),
                                ('engagement', ENG_PARAMS, 'Eng')]:
        data = {}
        for k in K_VALUES:
            print(f"  {tag} k={k}")
            alphas, d = sweep_alpha(model, params, k, Q_fixed,
                                    ALPHA_SWEEP['min'], ALPHA_SWEEP['max'],
                                    ALPHA_SWEEP['points'], BETA_GRID_POINTS)
            data[k] = d
        plot_alpha_sweep(alphas, data, tag, f'exp3_alpha_{tag.lower()}.pdf')


def exp4_delta():
    print("\n=== Exp 4: delta sweep ===")
    Q_fixed = 0.5
    for model, params, tag in [('view', VIEW_PARAMS, 'View'),
                                ('engagement', ENG_PARAMS, 'Eng')]:
        data = {}
        for k in K_VALUES:
            print(f"  {tag} k={k}")
            deltas, d = sweep_delta(model, params, k, Q_fixed,
                                    DELTA_SWEEP['min'], DELTA_SWEEP['max'],
                                    DELTA_SWEEP['points'], BETA_GRID_POINTS)
            data[k] = d
        plot_delta_sweep(deltas, data, tag, f'exp4_delta_{tag.lower()}.pdf')


# def exp5_cross_model():
#     print("\n=== Exp 5: cross-model comparison ===")
#     for k in [0.0, 1.0, 2.0, 5.0]:
#         print(f"  k={k}")
#         Qs_v, dv = sweep_Q('view', VIEW_PARAMS, k,
#                            Q_SWEEP['min'], Q_SWEEP['max'],
#                            Q_SWEEP['points'], BETA_GRID_POINTS)
#         Qs_e, de = sweep_Q('engagement', ENG_PARAMS, k,
#                            Q_SWEEP['min'], Q_SWEEP['max'],
#                            Q_SWEEP['points'], BETA_GRID_POINTS)
#         plot_cross_model(Qs_v, dv, de, k, f'exp5_cross_k{k:.0f}.pdf')


def main():
    print("=" * 60)
    print("Platform Model with Discrimination Penalty")
    print("r ∈ [0, 1] enforced")
    print("=" * 60)
    print(f"View params:  {VIEW_PARAMS}")
    print(f"Eng params:   {ENG_PARAMS}")
    print(f"k values:     {K_VALUES}")

    exp1_Q_sweep()
    exp2_k_sweep()
    exp3_alpha()
    exp4_delta()
    exp5_cross_model()

    print("\n" + "=" * 60)
    print("All experiments complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
