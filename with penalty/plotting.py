"""
Plotting functions.
"""
import os, numpy as np, matplotlib.pyplot as plt
from matplotlib import rcParams
from config import PLOT_CONFIG, OUTPUT_DIR

def _setup():
    for k, v in PLOT_CONFIG.items():
        try: rcParams[k] = v
        except: pass
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def _save(fig, name):
    p = os.path.join(OUTPUT_DIR, name)
    fig.savefig(p, dpi=300, bbox_inches='tight')
    print(f"  Saved {p}")

K_COLORS = {0.0:'#1f77b4', 1.0:'#2ca02c', 2.0:'#ff7f0e',
             5.0:'#d62728', 10.0:'#9467bd'}
def _kc(k): return K_COLORS.get(k, '#333')


def plot_Q_sweep(x, data_by_k, model, filename):
    _setup()
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    panels = [(axes[0,0],'beta_h',r'$\beta_h^*$','Promotion Weight'),
              (axes[0,1],'r',r'$r^*$','Compensation Rate'),
              (axes[1,0],'cs','Consumer Surplus','Consumer Surplus'),
              (axes[1,1],'uc',r'Creator Welfare $u_c^*$','Creator Welfare')]
    for ax, key, yl, ti in panels:
        for k, d in sorted(data_by_k.items()):
            ax.plot(x, d[key], '-', color=_kc(k), lw=2, label=f'$k={k}$')
        ax.set_xlabel(r'AI Baseline Quality $Q$')
        ax.set_ylabel(yl); ax.set_title(f'{model}: {ti}')
        ax.grid(True, alpha=0.3, ls='--'); ax.legend(fontsize=9)
    axes[0,0].set_ylim([-0.05,1.05])
    axes[0,1].set_ylim([-0.05,1.05])
    fig.tight_layout(); _save(fig, filename); return fig


def plot_k_sweep(x, data_by_Q, model, filename):
    _setup()
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(data_by_Q)))
    panels = [(axes[0,0],'beta_h',r'$\beta_h^*$','Promotion Weight'),
              (axes[0,1],'r',r'$r^*$','Compensation Rate'),
              (axes[1,0],'cs','Consumer Surplus','Consumer Surplus'),
              (axes[1,1],'uc',r'Creator Welfare $u_c^*$','Creator Welfare')]
    for ax, key, yl, ti in panels:
        for j,(Q,d) in enumerate(sorted(data_by_Q.items())):
            ax.plot(x, d[key], '-', color=colors[j], lw=2, label=f'$Q={Q}$')
        ax.set_xlabel(r'Penalty Strength $k$')
        ax.set_ylabel(yl); ax.set_title(f'{model}: {ti}')
        ax.grid(True, alpha=0.3, ls='--'); ax.legend(fontsize=9)
    axes[0,0].set_ylim([-0.05,1.05])
    axes[0,1].set_ylim([-0.05,1.05])
    fig.tight_layout(); _save(fig, filename); return fig


def plot_alpha_sweep(x, data_by_k, model, filename):
    _setup()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    panels = [(axes[0],'beta_h',r'$\beta_h^*$','Promotion'),
              (axes[1],'r',r'$r^*$','Compensation'),
              (axes[2],'uc',r'$u_c^*$','Creator Welfare')]
    for ax, key, yl, ti in panels:
        for k, d in sorted(data_by_k.items()):
            ax.plot(x, d[key], '-', color=_kc(k), lw=2, label=f'$k={k}$')
        ax.set_xlabel(r'$\alpha$'); ax.set_ylabel(yl)
        ax.set_title(f'{model}: {ti}')
        ax.grid(True, alpha=0.3, ls='--'); ax.legend(fontsize=9)
    axes[0].set_ylim([-0.05,1.05])
    axes[1].set_ylim([-0.05,1.05])
    fig.tight_layout(); _save(fig, filename); return fig


def plot_delta_sweep(x, data_by_k, model, filename):
    _setup()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    panels = [(axes[0],'beta_h',r'$\beta_h^*$','Promotion'),
              (axes[1],'r',r'$r^*$','Compensation'),
              (axes[2],'uc',r'$u_c^*$','Creator Welfare')]
    for ax, key, yl, ti in panels:
        for k, d in sorted(data_by_k.items()):
            ax.plot(x, d[key], '-', color=_kc(k), lw=2, label=f'$k={k}$')
        ax.set_xlabel(r'$\delta$'); ax.set_ylabel(yl)
        ax.set_title(f'{model}: {ti}')
        ax.grid(True, alpha=0.3, ls='--'); ax.legend(fontsize=9)
    axes[0].set_ylim([-0.05,1.05])
    axes[1].set_ylim([-0.05,1.05])
    fig.tight_layout(); _save(fig, filename); return fig


def plot_cross_model(Q_vals, view_data, eng_data, k, filename):
    _setup()
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    panels = [(axes[0,0],'beta_h',r'$\beta_h^*$','Promotion Weight'),
              (axes[0,1],'r',r'$r^*$','Compensation Rate'),
              (axes[1,0],'cs','Consumer Surplus','Consumer Surplus'),
              (axes[1,1],'uc',r'$u_c^*$','Creator Welfare')]
    for ax, key, yl, ti in panels:
        ax.plot(Q_vals, view_data[key], '-', color='#1f77b4', lw=2, label='View-Based')
        ax.plot(Q_vals, eng_data[key], '-', color='#ff7f0e', lw=2, label='Engagement-Based')
        ax.set_xlabel(r'$Q$'); ax.set_ylabel(yl)
        ax.set_title(f'{ti} ($k={k}$)')
        ax.grid(True, alpha=0.3, ls='--'); ax.legend()
    axes[0,0].set_ylim([-0.05,1.05])
    axes[0,1].set_ylim([-0.05,1.05])
    fig.tight_layout(); _save(fig, filename); return fig
