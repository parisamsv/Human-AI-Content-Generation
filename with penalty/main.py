
from config import *
from data_generation import sweep_Q, sweep_k, sweep_alpha, sweep_delta
from plotting import plot_sweep


def exp1_Q(changing_param, VALUES):
    print("\n=== Exp 1: Q sweep ===")
    for rev_model, comp_model, tag in [ ('view', 'view','Rev: View, Comp: View'),
                                        ('engagement', 'engagement', 'Rev: Engagement, Comp: Engagement'),
                                        ('view', 'engagement', 'Rev: View, Comp: Engagement'),
                                        ('engagement', 'view', 'Rev: Engagement, Comp: View')]:
        params = PARAMS.copy()
        data = {}
        if changing_param == 'k':
            for k in VALUES:
                params['k'] = k
                Qs, d = sweep_Q(rev_model, comp_model, params,
                                Q_SWEEP['min'], Q_SWEEP['max'],
                                Q_SWEEP['points'], BETA_GRID_POINTS)
                data[k] = d
        elif changing_param == 'alpha':
            for alpha in VALUES:
                params['alpha'] = alpha
                Qs, d = sweep_Q(rev_model, comp_model, params,
                                Q_SWEEP['min'], Q_SWEEP['max'],
                                Q_SWEEP['points'], BETA_GRID_POINTS)
                data[alpha] = d

        elif changing_param == 'delta':
            for delta in VALUES:
                params['delta'] = delta
                Qs, d = sweep_Q(rev_model, comp_model, params,
                                Q_SWEEP['min'], Q_SWEEP['max'],
                                Q_SWEEP['points'], BETA_GRID_POINTS)
                data[delta] = d
        
        plot_sweep(Qs, data, tag, 'Q', changing_param, f'exp1_Q_{changing_param}_{tag.lower()}.pdf')


def exp2_k(changing_param, VALUES):
    print("\n=== Exp 2: k sweep ===")
    for rev_model, comp_model, tag in [ ('view', 'view','Rev: View, Comp: View'),
                                        ('engagement', 'engagement', 'Rev: Engagement, Comp: Engagement'),
                                        ('view', 'engagement', 'Rev: View, Comp: Engagement'),
                                        ('engagement', 'view', 'Rev: Engagement, Comp: View')]:
        params = PARAMS.copy()
        data = {}
        if changing_param == 'Q':
            for Q in VALUES:
                params['k'] = Q
                Qs, d = sweep_k(rev_model, comp_model, params,
                                Q_SWEEP['min'], Q_SWEEP['max'],
                                Q_SWEEP['points'], BETA_GRID_POINTS)
                data[Q] = d
        elif changing_param == 'alpha':
            for alpha in VALUES:
                params['alpha'] = alpha
                Qs, d = sweep_k(rev_model, comp_model, params,
                                Q_SWEEP['min'], Q_SWEEP['max'],
                                Q_SWEEP['points'], BETA_GRID_POINTS)
                data[alpha] = d

        elif changing_param == 'delta':
            for delta in VALUES:
                params['delta'] = delta
                Qs, d = sweep_k(rev_model, comp_model, params,
                                Q_SWEEP['min'], Q_SWEEP['max'],
                                Q_SWEEP['points'], BETA_GRID_POINTS)
                data[delta] = d
        
        plot_sweep(Qs, data, tag, 'k', changing_param, f'exp2_k_{changing_param}_{tag.lower()}.pdf')



def exp3_alpha(changing_param, VALUES):
    print("\n=== Exp 3: alpha sweep ===")
    for rev_model, comp_model, tag in [ ('view', 'view','Rev: View, Comp: View'),
                                        ('engagement', 'engagement', 'Rev: Engagement, Comp: Engagement'),
                                        ('view', 'engagement', 'Rev: View, Comp: Engagement'),
                                        ('engagement', 'view', 'Rev: Engagement, Comp: View')]:
        params = PARAMS.copy()
        data = {}
        if changing_param == 'k':
            for k in VALUES:
                params['k'] = k
                Qs, d = sweep_alpha(rev_model, comp_model, params,
                                Q_SWEEP['min'], Q_SWEEP['max'],
                                Q_SWEEP['points'], BETA_GRID_POINTS)
                data[k] = d
        elif changing_param == 'Q':
            for Q in VALUES:
                params['Q'] = Q
                Qs, d = sweep_alpha(rev_model, comp_model, params,
                                Q_SWEEP['min'], Q_SWEEP['max'],
                                Q_SWEEP['points'], BETA_GRID_POINTS)
                data[Q] = d

        elif changing_param == 'delta':
            for delta in VALUES:
                params['delta'] = delta
                Qs, d = sweep_alpha(rev_model, comp_model, params,
                                Q_SWEEP['min'], Q_SWEEP['max'],
                                Q_SWEEP['points'], BETA_GRID_POINTS)
                data[delta] = d
        
        plot_sweep(Qs, data, tag, 'alpha', changing_param, f'exp2_alpha_{changing_param}_{tag.lower()}.pdf')


def exp4_delta(changing_param, VALUES):
    print("\n=== Exp 4: delta sweep ===")
    for rev_model, comp_model, tag in [ ('view', 'view','Rev: View, Comp: View'),
                                        ('engagement', 'engagement', 'Rev: Engagement, Comp: Engagement'),
                                        ('view', 'engagement', 'Rev: View, Comp: Engagement'),
                                        ('engagement', 'view', 'Rev: Engagement, Comp: View')]:
        params = PARAMS.copy()
        data = {}
        if changing_param == 'k':
            for k in VALUES:
                params['k'] = k
                Qs, d = sweep_k(rev_model, comp_model, params,
                                Q_SWEEP['min'], Q_SWEEP['max'],
                                Q_SWEEP['points'], BETA_GRID_POINTS)
                data[k] = d
        elif changing_param == 'alpha':
            for alpha in VALUES:
                params['alpha'] = alpha
                Qs, d = sweep_k(rev_model, comp_model, params,
                                Q_SWEEP['min'], Q_SWEEP['max'],
                                Q_SWEEP['points'], BETA_GRID_POINTS)
                data[alpha] = d

        elif changing_param == 'Q':
            for Q in VALUES:
                params['Q'] = Q
                Qs, d = sweep_k(rev_model, comp_model, params,
                                Q_SWEEP['min'], Q_SWEEP['max'],
                                Q_SWEEP['points'], BETA_GRID_POINTS)
                data[Q] = d
        
        plot_sweep(Qs, data, tag, 'delta', changing_param, f'exp2_delta_{changing_param}_{tag.lower()}.pdf')


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
    print("=" * 60)
    print(f"params:  {PARAMS}")
    print(f"k values:     {K_VALUES}")

    exp1_Q()
    exp2_k()
    exp3_alpha()
    exp4_delta()

    print("\n" + "=" * 60)
    print("All experiments complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
