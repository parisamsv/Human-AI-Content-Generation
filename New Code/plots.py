"""
plots.py
--------
Figure-generation routines for the numerical study (Section 4 of the paper).

Figures produced
----------------
1. fig_Q_comparative_statics.pdf
   Six-panel: β_H*, q_H*, u_P, u_H, TV, TE  vs  AI baseline quality Q.

2. fig_alpha_comparative_statics.pdf
   Six-panel: β_H*, q_H*, u_P, u_H, TV, TE  vs  AI learning efficiency α.

3. fig_delta_comparative_statics.pdf
   Six-panel: β_H*, q_H*, u_P, u_H, TV, TE  vs  algorithmic influence δ.

4. fig_preference_map_Q_alpha.pdf
   Heatmap over (Q, α) showing the four preference regions
   (Region I: both prefer View; II: both prefer Engagement;
    III: Platform→E, Creator→V; IV: Platform→V, Creator→E).

5. fig_preference_map_delta.pdf
   Two side-by-side heatmaps: (Q, δ) and (α, δ) with the same
   four preference regions.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm

from params import BASELINE, RANGES, N_POINTS, N_GRID
from model_view import equilibrium_view
from model_engagement import equilibrium_eng
from common import m_A, m_H


# ── Global style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":     "serif",
    "mathtext.fontset": "dejavuserif",
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.linewidth":  0.8,
    "grid.linewidth":  0.4,
    "grid.color":      "#cccccc",
    "xtick.direction": "in",
    "ytick.direction": "in",
})


# ── Colour scheme ─────────────────────────────────────────────────────────────
COL_VIEW = "#8f07aa"   # pink       – view-based
COL_ENG  = "#0f21b0"   # blue – engagement-based

# Preference-region colours (Regions I–IV)
REGION_COLORS = ["#4292c6", "#f16913", "#41ab5d", "#5d36a0"]
REGION_LABELS = [
    "Region I: both prefer View",
    "Region II: both prefer Engagement",
    "Region III: Platform→Eng, Creator→View",
    "Region IV: Platform→View, Creator→Eng",
]
_RCMAP = ListedColormap(REGION_COLORS)
_RNORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], _RCMAP.N)
_RCMAP.set_bad("#d9d9d9")   # grey for masked (invalid) cells

REGION_COLORS_reg = ["#97cef3", "#f1f0a7", "#f99797"]
REGION_LABELS_reg = [
    "Uncovered",
    "Boundary",
    "Covered"
]
_RCMAP_reg = ListedColormap(REGION_COLORS_reg)
_RNORM_reg = BoundaryNorm([-0.1, 0.5, 1.5, 2.5], _RCMAP_reg.N)
_RCMAP_reg.set_bad("#d9d9d9")   # grey for masked (invalid) cells


# ── Panel specification (same order in every 6-panel figure) ─────────────────
_PANEL_KEYS = ["beta_H", "q_H", "u_P", "u_H", "TV", "TE"]
_PANEL_YLABELS = [
    r"Algorithmic weight $\beta_H^*$",
    r"Creator effort $q_H^*$",
    r"Platform utility $u_P$",
    r"Creator utility $u_H$",
    r"Total views $\mathrm{TV}$",
    r"Total engagement $\mathrm{TE}$",
]

# Axis labels for each varying parameter
_PARAM_LABEL = {
    "Q":     r"AI baseline quality $Q$",
    "alpha": r"AI learning efficiency $\alpha$",
    "delta": r"Algorithmic influence $\delta$",
}

# ── Internal helpers ──────────────────────────────────────────────────────────

def _safe_eq(Q, alpha, delta, t, k):
    """
    Return (eq_view, eq_eng) or (None, None) if any computation fails.
    Failure can happen at extreme parameter values that violate assumptions.
    """
    try:
        ev = equilibrium_view(Q, alpha, delta, t, k)
        ee = equilibrium_eng(Q,  alpha, delta, t, k)
        return ev, ee
    except Exception:
        return None, None

def _run_1d(param_name: str, values: np.ndarray, baseline: dict):
    """
    Sweep `param_name` over `values`, holding all other parameters at baseline.

    Returns
    -------
    data_v, data_e : dict[str, np.ndarray]
        One array per panel key, length = len(values).
        NaN where the equilibrium could not be computed.
    """
    data_v = {k: np.full(len(values), np.nan) for k in _PANEL_KEYS}
    data_e = {k: np.full(len(values), np.nan) for k in _PANEL_KEYS}

    for idx, val in enumerate(values):
        p = {**baseline, param_name: val}
        ev, ee = _safe_eq(p["Q"], p["alpha"], p["delta"], p["t"], p["k"])
        if ev is not None:
            for k in _PANEL_KEYS:
                data_v[k][idx] = ev[k]
                data_e[k][idx] = ee[k]

    return data_v, data_e

def _pref_code(dp: float, dh: float) -> int:
    """
    Convert (Δu_P, Δu_H) = (u_P^E - u_P^V, u_H^E - u_H^V) to region code.

        code 0  Region I   – both prefer View       (dp ≤ 0, dh ≤ 0)
        code 1  Region II  – both prefer Engagement (dp > 0, dh > 0)
        code 2  Region III – Platform→E, Creator→V  (dp > 0, dh ≤ 0)
        code 3  Region IV  – Platform→V, Creator→E  (dp ≤ 0, dh > 0)
    """
    if dp > 0.0001 and dh > 0.0001:
        return 1
    elif dp > 0.0001 and dh <= -0.0001:
        return 2
    elif dh > 0.0001 and dp <= -0.0001:
        return 3
    else:
        return 0

def _pref_code_reg(integer) -> int:
    """
        code 0  Uncovered
        code 1  Boundary
        code 2  Covered
    """
    if integer == 1:
        return 0
    elif integer == 2:
        return 1
    else:
        return 2

def _run_2d(param_x: str, param_y: str,
            range_x: tuple, range_y: tuple,
            baseline: dict, reg=None):
    """
    Build a (N_GRID × N_GRID) preference-region map over the
    (param_x, param_y) plane.

    Returns
    -------
    xs : (N_GRID,) array   – x-axis (param_x) values
    ys : (N_GRID,) array   – y-axis (param_y) values
    region : (N_GRID, N_GRID) int array
        region[j, i] is the preference code at (xs[i], ys[j]).
        Value -1 marks grid points where assumptions were violated.
    """
    xs = np.linspace(*range_x, N_GRID)
    ys = np.linspace(*range_y, N_GRID)
    region = np.full((N_GRID, N_GRID), -1, dtype=int)

    for i, xv in enumerate(xs):
        for j, yv in enumerate(ys):
            p = {**baseline, param_x: xv, param_y: yv}
            ev, ee = _safe_eq(p["Q"], p["alpha"], p["delta"], p["t"], p["k"])
            if ev is not None:
                if reg == "V":
                    v = int(ev["Market"]=="uncov") + 2*int(ev["Market"]=="bnd") + 3*int(ev["Market"]=="cov")
                    region[j, i] = _pref_code_reg(v)
                elif reg == "E":
                    e = int(ee["Market"]=="uncov") + 2*int(ee["Market"]=="bnd") + 3*int(ee["Market"]=="cov")
                    region[j, i] = _pref_code_reg(e)
                else:
                    dp = ee["u_P"] - ev["u_P"]
                    dh = ee["u_H"] - ev["u_H"]
                    region[j, i] = _pref_code(dp, dh)

    return xs, ys, region

# ── Comparative-statics figures ───────────────────────────────────────────────

def plot_comparative_statics(param_name: str,
                              output_path: str,
                              baseline: dict = None) -> None:
    """
    Six-panel comparative-statics figure.

    Varies `param_name` over its full range (from RANGES), holds all other
    parameters at `baseline`, and plots β_H*, q_H*, u_P, u_H, TV, TE.
    One line per monetization base.  A vertical dotted line marks the
    baseline value of the varying parameter.

    Parameters
    ----------
    param_name  : one of "Q", "alpha", "delta"
    output_path : destination for the PDF
    baseline    : parameter dict; defaults to BASELINE from params.py
    """
    if baseline is None:
        baseline = BASELINE

    lo, hi = RANGES[param_name]
    values  = np.linspace(lo, hi, N_POINTS)
    base_v  = baseline[param_name]

    print(f"    sweeping {param_name} over [{lo}, {hi}] …", end=" ", flush=True)
    data_v, data_e = _run_1d(param_name, values, baseline)
    print("done.")

    # ------------------- thresholds
    def Qc (setting, beta_H):
        mA = m_A(beta_H, baseline["delta"])
        mH = m_H(beta_H, baseline["delta"])
        t = baseline["t"]
        alpha = baseline["alpha"]
        if setting == "view":
            return t * m_A - (m_A + alpha * m_H) ** 2 / (2 * t * m_A * m_H ** 2)
        else:
            Gamma  = mA * (t * mH - 1.0) - alpha ** 2 * mH 
            return t * Gamma / (t * m_H - 1 + alpha)
    
    def Qc_prime(beta_H):
        mA = m_A(beta_H, baseline["delta"])
        mH = m_H(beta_H, baseline["delta"])
        t = baseline["t"]
        alpha = baseline["alpha"]
        delta = baseline["delta"]
        Lambda = t * (2.0 - delta) - (1.0 - alpha) ** 2
        neum = t * (2 * Lambda * mA * mH - (mA + alpha * mH) ** 2)
        denom = 2 * (2 - delta) * (t * mH - 1 + alpha)
        return neum / denom
    
    def Q_hat(setting, beta_H):
        mA = m_A(beta_H, baseline["delta"])
        mH = m_H(beta_H, baseline["delta"])
        t = baseline["t"]
        alpha = baseline["alpha"]
        delta = baseline["delta"]
        if setting == "engagement":
            Psi = t * mH - 1
            Gamma  = mA * (t * mH - 1.0) - alpha ** 2 * mH 
            bracket = (mA + alpha * mH) ** 2 + (alpha + Psi) * (mA ** 2 - alpha * mH ** 2) + (2 - delta) * Gamma
            return t * bracket / (2 * (2 - delta ) * (alpha + Psi))
        else:
            bracket = 2 * mA ** 2 + mA * mH - alpha * mH ** 2
            return t * bracket / (2 * (2 - delta))

    

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()

    for ax, key, ylabel in zip(axes, _PANEL_KEYS, _PANEL_YLABELS):
        ax.plot(values, data_v[key], color=COL_VIEW, lw=2.0,
                label="View-based")
        ax.plot(values, data_e[key], color=COL_ENG,  lw=2.0,
                linestyle="--", label="Engagement-based")
        ax.axvline(base_v, color="#666666", lw=0.8, ls=":", alpha=0.9,
                   zorder=0)
        ax.set_xlabel(_PARAM_LABEL[param_name], fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlim(lo, hi)
        ax.tick_params(labelsize=8.5)
        ax.grid(True, axis="y")

    # Shared legend below all panels
    handles = [
        plt.Line2D([0], [0], color=COL_VIEW, lw=2,       label="View-based"),
        plt.Line2D([0], [0], color=COL_ENG,  lw=2, ls="--", label="Engagement-based"),
        plt.Line2D([0], [0], color="#666666", lw=0.8, ls=":", label="Baseline"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.01))

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"    Saved → {output_path}")

# ── Preference-map helpers ────────────────────────────────────────────────────

def _draw_pref_map(ax, xs, ys, region,
                   xlabel: str, ylabel: str,
                   title: str = "", reg = None) -> None:
    """Render a single preference-region heatmap on `ax`."""
    masked = np.ma.masked_where(region < 0, region).astype(float)
    if reg is not None:
        ax.pcolormesh(xs, ys, masked,
                    cmap=_RCMAP_reg, norm=_RNORM_reg,
                    shading="auto", rasterized=True)

    else:
        ax.contourf(
            xs, ys, masked,
            levels=[-0.5, 0.5, 1.5, 2.5, 3.5],
            cmap=_RCMAP,
            hatches=['///', '\\\\\\', '...', 'xxx']
        )
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    if title:
        ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=8.5)

def _region_legend(reg=None):
    hatches = ['///', '\\\\\\', '...', 'xxx']

    if reg is None:
        return [
            mpatches.Patch(
                facecolor=c,
                label=l,
                hatch=h,
                edgecolor="black",   # hatch color comes from edgecolor
                lw=0.4
            )
            for c, l, h in zip(REGION_COLORS, REGION_LABELS, hatches)
        ]
    else:
        return [
            mpatches.Patch(
                facecolor=c,
                label=l,
                hatch=h,
                edgecolor="black",
                lw=0.4
            )
            for c, l, h in zip(REGION_COLORS_reg, REGION_LABELS_reg, hatches)
        ]

# ── fig_preference_map_Q_alpha ─────────────────────────────────────────────────

def plot_preference_map_Q_alpha(output_path: str,
                                 baseline: dict = None, reg = None) -> None:
    """
    Heatmap over (Q, α) showing the four preference regions.
    The x-axis is Q ∈ RANGES["Q"], the y-axis is α ∈ RANGES["alpha"].

    Parameters
    ----------
    output_path : destination for the PDF
    baseline    : parameter dict; defaults to BASELINE
    """
    if baseline is None:
        baseline = BASELINE

    print(f"    computing (Q, α) preference map ({N_GRID}×{N_GRID}) …",
          end=" ", flush=True)
    xs, ys, region = _run_2d("Q", "alpha",
                              RANGES["Q"], RANGES["alpha"], baseline, reg)
    print("done.")

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    _draw_pref_map(ax, xs, ys, region,
                xlabel=_PARAM_LABEL["Q"],
                ylabel=_PARAM_LABEL["alpha"], reg=reg)

    # Mark baseline
    bQ, ba = baseline["Q"], baseline["alpha"]
    ax.plot(bQ, ba, marker="*", ms=9, color="black", zorder=5,
            label=f"Baseline $(Q,\\alpha)=({bQ},{ba})$")

    ax.legend(handles=_region_legend(reg) +
              [mpatches.Patch(facecolor="none", edgecolor="none",
                              label=f"$\\bigstar$ baseline $({bQ},{ba})$")],
              loc="upper left", fontsize=8, framealpha=0.9, edgecolor="#aaaaaa")

    # fig.suptitle(
        # r"Monetization preferences over $(Q,\,\alpha)$",
        # fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"    Saved → {output_path}")

# ── fig_preference_map_delta ───────────────────────────────────────────────────

def plot_preference_map_delta(output_path: str,
                               baseline: dict = None) -> None:
    """
    Two side-by-side preference heatmaps:
      left panel  – (Q, δ) plane
      right panel – (α, δ) plane

    Parameters
    ----------
    output_path : destination for the PDF
    baseline    : parameter dict; defaults to BASELINE
    """
    if baseline is None:
        baseline = BASELINE

    print(f"    computing (Q, δ) preference map ({N_GRID}×{N_GRID}) …",
          end=" ", flush=True)
    xs_Q,  ys_d,  reg_Qd  = _run_2d("Q",    "delta",
                                     RANGES["Q"],     RANGES["delta"], baseline)
    print("done.")

    print(f"    computing (α, δ) preference map ({N_GRID}×{N_GRID}) …",
          end=" ", flush=True)
    xs_al, ys_d2, reg_ald = _run_2d("alpha", "delta",
                                     RANGES["alpha"], RANGES["delta"], baseline)
    print("done.")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    _draw_pref_map(axes[0], xs_Q,  ys_d,  reg_Qd,
                   xlabel=_PARAM_LABEL["Q"],
                   ylabel=_PARAM_LABEL["delta"],
                   title=r"$(Q,\,\delta)$ space")

    _draw_pref_map(axes[1], xs_al, ys_d2, reg_ald,
                   xlabel=_PARAM_LABEL["alpha"],
                   ylabel=_PARAM_LABEL["delta"],
                   title=r"$(\alpha,\,\delta)$ space")

    # Mark baselines
    for ax, bx, by in [
        (axes[0], baseline["Q"],     baseline["delta"]),
        (axes[1], baseline["alpha"], baseline["delta"]),
    ]:
        ax.plot(bx, by, marker="*", ms=9, color="black", zorder=5)

    # Shared legend at the bottom
    fig.legend(handles=_region_legend(),
               loc="lower center", ncol=2, fontsize=8.5,
               frameon=True, framealpha=0.9, edgecolor="#aaaaaa",
               bbox_to_anchor=(0.5, -0.12))

    fig.suptitle(
        r"Monetization preferences as algorithmic influence $\delta$ varies",
        fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"    Saved → {output_path}")
