"""
Plotting functions.
"""
import os, numpy as np, matplotlib.pyplot as plt
from matplotlib import rcParams
from config import DELTA_COLORS, PLOT_CONFIG, OUTPUT_DIR, K_COLORS, Q_COLORS, ALPHA_COLORS

def _setup(): # font and style settings for all plots
    for k, v in PLOT_CONFIG.items():
        try: rcParams[k] = v
        except: pass
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def _save(fig, name): # save a figure to the output directory with a given name
    p = os.path.join(OUTPUT_DIR, name)
    fig.savefig(p, dpi=300, bbox_inches='tight')
    print(f"  Saved {p}")

def set_color(param_type, param_value):
    if param_type == 'k': return K_COLORS.get(param_value, '#333')
    elif param_type == 'alpha': return ALPHA_COLORS.get(param_value, '#333')
    elif param_type == 'delta': return DELTA_COLORS.get(param_value, '#333')
    elif param_type == 'Q': return Q_COLORS.get(param_value, '#333')
    else: return '#333'

def plot_sweep(x, data_by_changing_param, model, main_param, changing_param, filename):
    _setup()

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    panels = [(axes[0,0],'beta_h',r'$\beta_h^*$','Promotion Weight'),
              (axes[0,1],'r',r'$r^*$','Compensation Rate'),
              (axes[1,2],'cu','Creator Utility','Creator Utility'),
              (axes[1,0],'te',r'$q_h * D_h + q_A * D_A$','Total Engagement'),
              (axes[1,1],'tv',r'$D_h + D_A$','Total Views'),
              (axes[0,2],'pu',r'Platform Utility $u_p^*$','Platform Utility')]
    
    for ax, key, yl, ti in panels:
        
        for param_value, d in sorted(data_by_changing_param.items()):   
            ax.plot(x, d[key], '-', color=set_color(changing_param, param_value), lw=2, label=f'{changing_param}={param_value}')

        ax.set_xlabel(f'{main_param}')
        ax.set_ylabel(yl); ax.set_title(f'{model}: {ti}')
        ax.grid(True, alpha=0.3, ls='--'); ax.legend(fontsize=9)

    # axes[0,0].set_ylim([-0.05,1.05])
    # axes[0,1].set_ylim([-0.05,1.05])
    axes[1,2].set_visible(False)

    fig.tight_layout(); _save(fig, filename); return fig



# def plot_cross_model(Q_vals, view_data, eng_data, k, filename):
#     _setup()
#     fig, axes = plt.subplots(2, 3, figsize=(17, 9))
#     panels = [(axes[0,0],'beta_h',r'$\beta_h^*$','Promotion Weight'),
#               (axes[0,1],'r',r'$r^*$','Compensation Rate'),
#               (axes[0,2],'cs','Consumer Surplus','Consumer Surplus'),
#               (axes[1,0],'uc',r'$u_c^*$','Creator Welfare'),
#               (axes[1,1],'utility',r'$u_p^*$','Platform Welfare')]
#     for ax, key, yl, ti in panels:
#         ax.plot(Q_vals, view_data[key], '-', color='#1f77b4', lw=2, label='View-Based')
#         ax.plot(Q_vals, eng_data[key], '-', color='#ff7f0e', lw=2, label='Engagement-Based')
#         ax.set_xlabel(r'$Q$'); ax.set_ylabel(yl)
#         ax.set_title(f'{ti} ($k={k}$)')
#         ax.grid(True, alpha=0.3, ls='--'); ax.legend()
#     axes[0,0].set_ylim([-0.05,1.05])
#     axes[0,1].set_ylim([-0.05,1.05])
#     axes[1,2].set_visible(False)
#     fig.tight_layout(); _save(fig, filename); return fig
