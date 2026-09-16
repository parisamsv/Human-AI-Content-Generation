"""
Figures produced
----------------
fig_vb_effort_regions.pdf           View-based creator effort q_H*(r) vs r
                                    (Figure 2)

fig_eb_effort_regions.pdf           Engagement-based creator effort q_H*(=r*) vs Q
                                    (Figure 3)

fig_cmp_uH_Q_regions.pdf            Creator utility vs Q, both settings
                                    (Figure 4)

fig_cmp_uH_alpha_regions.pdf        Creator utility vs alpha, both settings
                                    (Figure 5)

fig_preference_map_Q_alpha.pdf      Heatmap over (Q, α) showing the four preference regions
                                    (Region I: both prefer View; II: both prefer Engagement;
                                    III: Platform→E, Creator→V; IV: Platform→V, Creator→E).
                                    (Figure 6)

fig_preference_map_delta.pdf        Two side-by-side heatmaps: (Q, δ) and (α, δ) with the same
                                    four preference regions as above.
                                    (Figure 7)

fig_negative_TE_Q_region.pdf        Heatmap showing the length of the Q-interval where
                                    dTE/dQ < 0 (the negative branch of Corollary cor:Q_TE) over an (alpha,delta) grid 
                                    (Figure 8)

fig_Q_comparative_statics.pdf       Six-panel: β_H*, q_H*, u_P, u_H, TV, TE  vs  AI baseline quality Q.
                                    (Figure 9)
                                        
fig_alpha_comparative_statics.pdf   Six-panel: β_H*, q_H*, u_P, u_H, TV, TE  vs  AI learning efficiency α.
                                    (Figure 10)
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
from model_engagement import equilibrium_eng, r_star_eng
from common import m_A, m_H


# ==============================================================================
#                                  STYLE AND COLOURS
# ==============================================================================

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
GREY     = "#666666"

# Preference-region colours (Regions I–IV)
REGION_COLORS = ["#4292c6", "#f16913", "#41ab5d", "#5d36a0"]
REGION_LABELS = [
    "Region I: both prefer View",
    "Region II: both prefer Engagement",
    "Region III: Platform→Eng, Creator→View",
    "Region IV: Platform→View, Creator→Eng",
]
_RCMAP = ListedColormap(REGION_COLORS)
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

# region tints (uncovered / boundary / covered / capped)
SHADE = [ "#efe9f3", "#e8edf4", "#e9f1ec", "#f7f4e7"]

# resolution for the 2-D sign maps (finite-difference, so each cell solves twice)
SIGNMAP_GRID = 170


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

# ==============================================================================
#                                   FIGURES 6,7,9,10
# ==============================================================================

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


# ==============================================================================
#                                   FIGURES 2,3,4,5
# ==============================================================================

# ────────────────────────────── Local closed-form helpers ──────────────────────
def _vb_effort_thresholds(beta, Q, alpha, delta, t):
    """View-based creator best response: thresholds r1<r2 and the stall effort."""
    mA, mH = m_A(beta, delta), m_H(beta, delta)
    N = mA + alpha * mH
    P = t * mA - Q
    r1   = t * mH ** 2 * P / N                       # uncovered upper edge
    r2   = t * (mA + mH) * mH * P / ((1 - alpha) * N)  # covered lower edge
    qbar = mH * P / N                                # stall effort at coverage
    r_c = t * (mA+mH) * (t*mH+Q) / (1 - alpha)**2
    return r1, r2, qbar, r_c

def _vb_effort_of_r(r, beta, Q, alpha, delta, t):
    """View-based q_H*(r), piecewise (Prop. pro:Optimal:q_H:View-Based)."""
    mA, mH = m_A(beta, delta), m_H(beta, delta)
    r1, r2, qbar, r_c = _vb_effort_thresholds(beta, Q, alpha, delta, t)
    if r <= r1:
        return r / (t * mH)
    if r <= r2:
        return qbar
    if r <= r_c:
        return r * (1 - alpha) / (t * (mA + mH))
    return (t*mH+Q) / (1-alpha)

def _eng_rate_candidates(beta, Q, alpha, delta, t):
    """Engagement candidate rates r1 (unc), r_b (bnd), r2 (cov) and the regime."""
    mA, mH = m_A(beta, delta), m_H(beta, delta)
    N   = mA + alpha * mH
    P   = t * mA - Q
    Gam = mA * (t * mH - 1.0) - alpha ** 2 * mH
    Lam = t * (2.0 - delta) - (1.0 - alpha) ** 2
    Nc  = t * (alpha * mH + mA) - 2.0 * (1.0 - alpha) * Q
    r_b = mH * P / N
    r1  = alpha * Q * mH / Gam
    r2  = Nc / (2.0 * Lam)
    if r1 <= r_b:
        regime = "unc"
    elif r2 >= r_b:
        regime = "cov"
    else:
        regime = "bnd"
    return r1, r_b, r2, regime

def _shade_regions(ax, edges, xmax, model):
    """Shade [0,e1],[e1,e2],[e2,e3],[e3,xmax] with the four SHADE tints."""
    if model == "view":
        e1, e2, e3 = edges
        ax.axvspan(0, e1, color=SHADE[0], zorder=0)
        ax.axvspan(e1, e2, color=SHADE[1], zorder=0)
        ax.axvspan(e2, e3, color=SHADE[2], zorder=0)
        ax.axvspan(e3, xmax, color=SHADE[3], zorder=0)
        for e in edges:
            ax.axvline(e, color=GREY, ls="--", lw=0.9, zorder=1)
    else:
        e1, e2 = edges
        ax.axvspan(0, e1, color=SHADE[0], zorder=0)
        ax.axvspan(e1, e2, color=SHADE[1], zorder=0)
        ax.axvspan(e2, xmax, color=SHADE[2], zorder=0)
        for e in edges:
            ax.axvline(e, color=GREY, ls="--", lw=0.9, zorder=1)

# View-based effort regions  (vs the compensation rate r, at beta_H = 1/2)
def plot_vb_effort_regions(output_path, baseline=None):
    if baseline is None:
        baseline = BASELINE
    Q, alpha, delta, t = baseline["Q"], baseline["alpha"], baseline["delta"], baseline["t"]
    beta = 0.5

    r1, r2, qbar, r_c = _vb_effort_thresholds(beta, Q, alpha, delta, t)
    q_dbar = _vb_effort_of_r(r_c, beta, Q, alpha, delta, t)
    rmax = r_c * 1.4
    rs = np.linspace(0, rmax, 600)
    q  = np.array([_vb_effort_of_r(r, beta, Q, alpha, delta, t) for r in rs])

    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    _shade_regions(ax, (r1, r2, r_c), rmax, "view")
    ax.axhline(qbar, color=GREY, ls=":", lw=0.9)
    ax.axhline(q_dbar, color=GREY, ls=":", lw=0.9)
    ax.plot(rs, q, color=COL_VIEW, lw=2.4)

    ymax = q.max() * 1.16
    ax.set_ylim(0, ymax)
    ax.set_xlim(0, rmax)
    ax.text(r1 / 2, ymax * 0.94, "Uncovered\n",
            ha="center", va="top", fontsize=7)
    ax.text((r1 + r2) / 2, ymax * 0.94, "Boundary\n",
            ha="center", va="top", fontsize=7)
    ax.text((r2 + r_c) / 2, ymax * 0.94,
            "Covered\n" ,
            ha="center", va="top", fontsize=7)
    ax.text((r_c + rmax) / 2, ymax * 0.94,
            "Capped\n" ,
            ha="center", va="top", fontsize=7)
    ax.set_xticks([r1, r2, r_c])
    ax.set_xticklabels([r"$r_1$", r"$r_2$", r"$r_3$"])
    ax.set_yticks([qbar, _vb_effort_of_r(r_c, beta, Q, alpha, delta, t)])
    ax.set_yticklabels([r"$\bar q_H$", r"$\bar{\bar{q}}_H$"])
    ax.set_xlabel(r"Compensation rate $r$")
    ax.set_ylabel(r"Creator effort $q_H^*$")
    # ax.set_title(r"View-based: creator best response has three regions in $r$  "
    #              r"($\beta_H=1/2$)", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"    Saved -> {output_path}")
# 2. Engagement-based effort regions  (vs AI baseline quality Q, at beta_H = 1/2)
# Illustrative parameters chosen so all three regimes fall inside the
# assumptions' admissible Q-range; printed on the figure.
EB_ILLUSTRATIVE = {"alpha": 0.80, "delta": 0.40, "t": 2.40, "k": 1.00}
def plot_eb_effort_regions(output_path, params=None):
    p = dict(EB_ILLUSTRATIVE if params is None else params)
    alpha, delta, t = p["alpha"], p["delta"], p["t"]
    beta = 0.5
    Qmax = t * (1.0 - delta)                 # non-saturation bound (Assumption 2)
    Qs = np.linspace(0.01, Qmax * 0.999, 600)

    r1 = np.array([_eng_rate_candidates(beta, Q, alpha, delta, t)[0] for Q in Qs])
    rb = np.array([_eng_rate_candidates(beta, Q, alpha, delta, t)[1] for Q in Qs])
    r2 = np.array([_eng_rate_candidates(beta, Q, alpha, delta, t)[2] for Q in Qs])
    reg = [_eng_rate_candidates(beta, Q, alpha, delta, t)[3] for Q in Qs]
    # selected r* straight from the user's solver (effort = rate)
    qstar = np.array([r_star_eng(beta, Q, alpha, delta, t) for Q in Qs])

    # regime edges
    reg = np.array(reg)
    QEc  = Qs[np.argmax(reg != reg[0])]
    QEcp = Qs[::-1][np.argmax(reg[::-1] != reg[-1])]

    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    _shade_regions(ax, (QEc, QEcp), Qmax, "eng")
    ax.plot(Qs, r1, color=COL_ENG, lw=1.0, ls="--", alpha=0.55,
            label=r"$r_1=\alpha Q m_H/\Gamma$ (unc.)")
    ax.plot(Qs, rb, color=GREY,   lw=1.0, ls="-.", alpha=0.75,
            label=r"$r_b=m_H P/N$ (bnd.)")
    ax.plot(Qs, r2, color="#7b5cd6", lw=1.0, ls=":", alpha=0.8,
            label=r"$r_2=N_c/2\Lambda$ (cov.)")
    ax.plot(Qs, qstar, color=COL_ENG, lw=2.6,
            label=r"$q_H^*=r^*$ (selected)")

    ymax = max(qstar.max(), rb.max()) * 1.18
    ax.set_ylim(0, ymax)
    ax.set_xlim(0, Qmax)
    ax.text(QEc / 2, ymax * 0.95, "Uncovered\n" r"$q_H^*=r_1$",
            ha="center", va="top", fontsize=9.5)
    ax.text((QEc + QEcp) / 2, ymax * 0.95, "Boundary\n" r"$q_H^*=r_b$",
            ha="center", va="top", fontsize=9.5)
    ax.text((QEcp + Qmax) / 2, ymax * 0.95, "Covered\n" r"$q_H^*=r_2$",
            ha="center", va="top", fontsize=9.5)
    ax.set_xticks([QEc, QEcp])
    ax.set_xticklabels([r"$Q_E^{c}$", r"$Q_E^{c\,\prime}$"])
    ax.set_xlabel(r"AI baseline quality $Q$")
    ax.set_ylabel(r"Creator effort $q_H^*\;(=r^*)$")
    ax.legend(loc="lower left", fontsize=8.3, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"    Saved -> {output_path}")

# ───────────────── Creator utility vs Q, alpha ────────────────────────────────
def _sweep_uH(values, param, baseline):
    uV = np.full(len(values), np.nan)
    uE = np.full(len(values), np.nan)
    from model_engagement import equilibrium_eng
    for i, v in enumerate(values):
        p = {**baseline, param: v}
        try:
            uV[i] = equilibrium_view(p["Q"], p["alpha"], p["delta"], p["t"], p["k"])["u_H"]
            uE[i] = equilibrium_eng(p["Q"], p["alpha"], p["delta"], p["t"], p["k"])["u_H"]
        except Exception:
            pass
    return uV, uE

def plot_cmp_uH_Q_regions(output_path, baseline=None):
    if baseline is None:
        baseline = BASELINE
    baseline = dict(baseline)
    lo, hi = RANGES["Q"]
    Qs = np.linspace(lo, hi, 400)
    baseline["alpha"] = 0.1
    uV_0_1, uE_0_1 = _sweep_uH(Qs, "Q", baseline)
    baseline["alpha"] = 0.2
    uV_0_2, uE_0_2 = _sweep_uH(Qs, "Q", baseline)
    baseline["alpha"] = 0.3
    uV_0_3, uE_0_3 = _sweep_uH(Qs, "Q", baseline)
    baseline["alpha"] = 0.4
    uV_0_4, uE_0_4 = _sweep_uH(Qs, "Q", baseline)
    baseline["alpha"] = 0.5
    uV_0_5, uE_0_5 = _sweep_uH(Qs, "Q", baseline)
    baseline["alpha"] = 0.6
    uV_0_6, uE_0_6 = _sweep_uH(Qs, "Q", baseline)
    baseline["alpha"] = 0.7
    uV_0_7, uE_0_7 = _sweep_uH(Qs, "Q", baseline)
    baseline["alpha"] = BASELINE['alpha']  # restore baseline

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.1), sharex=True)
    view_data = (uV_0_1, uV_0_2, uV_0_3, uV_0_4, uV_0_5, uV_0_6, uV_0_7)
    engagement_data = (uE_0_1, uE_0_2, uE_0_3, uE_0_4, uE_0_5, uE_0_6, uE_0_7)
    alpha_values = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)

    for ax, data in zip(
            axes, (view_data, engagement_data)):
        for values, alpha in zip(data, alpha_values):
            ax.plot(Qs, values, lw=2.2, label=rf"$\alpha={alpha}$")
        ax.set_xlim(lo, hi)
        ax.set_xlabel(r"AI baseline quality $Q$")
        ax.grid(True, axis="y")

    axes[0].set_ylabel(r"Creator utility $u_H^*$")
    axes[1].legend(loc="upper left", fontsize=9.5, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"    Saved -> {output_path}")


def plot_cmp_uH_alpha_regions(output_path, baseline=None):
    if baseline is None:
        baseline = BASELINE
    baseline = dict(baseline)
    lo, hi = RANGES["alpha"]
    alphas = np.linspace(lo, hi, 400)
    baseline["Q"] = 0.3
    uV_0_3, uE_0_3 = _sweep_uH(alphas, "alpha", baseline)
    baseline["Q"] = 0.4
    uV_0_4, uE_0_4 = _sweep_uH(alphas, "alpha", baseline)
    baseline["Q"] = 0.5
    uV_0_5, uE_0_5 = _sweep_uH(alphas, "alpha", baseline)
    baseline["Q"] = 0.6
    uV_0_6, uE_0_6 = _sweep_uH(alphas, "alpha", baseline)
    baseline["Q"] = 0.7
    uV_0_7, uE_0_7 = _sweep_uH(alphas, "alpha", baseline)
    baseline["Q"] = 0.8
    uV_0_8, uE_0_8 = _sweep_uH(alphas, "alpha", baseline)
    baseline["Q"] = 0.9
    uV_0_9, uE_0_9 = _sweep_uH(alphas, "alpha", baseline)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.1), sharex=True)
    view_data = (uV_0_3, uV_0_4, uV_0_5, uV_0_6, uV_0_7, uV_0_8, uV_0_9)
    engagement_data = (uE_0_3, uE_0_4, uE_0_5, uE_0_6, uE_0_7, uE_0_8, uE_0_9)
    q_values = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

    for ax, data in zip(
            axes, (view_data, engagement_data)):
        for values, q in zip(data, q_values):
            ax.plot(alphas, values, lw=2.2, label=rf"$Q={q}$")
        ax.set_xlim(lo, hi)
        ax.set_xlabel(r"AI Efficiency $\alpha$")
        ax.grid(True, axis="y")

    axes[0].set_ylabel(r"Creator utility $u_H^*$")
    axes[1].legend(loc="upper left", fontsize=9.5, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"    Saved -> {output_path}")

# ==============================================================================
#                                   FIGURE 8
# ==============================================================================

# Admissibility, evaluated at beta_H*(Q) 
def _admissible(alpha, delta, t, Q):
    if t <= np.max([1.0, Q]) / (1.0 - delta):           
        return False
    if t <= alpha ** 2 + 1.0 / (1.0 - delta):
        return False
    if (1-delta) * (t * (1-delta) - 1.0) < alpha ** 2:  
        return False
    if t * (2.0 - delta) < (1.0 - alpha) ** 2:  # concavity-E
        return False
    return True

# Core measurement (optimal beta per Q) 
def negative_length(setting, alpha, delta,
                    baseline=None, n_Q=500, rel_tol=1e-6, q_lo=0.03):
    if baseline is None:
        baseline = BASELINE
    t, k = baseline["t"], baseline["k"]

    # generous sweep; per-Q non-saturation handled by the mask below
    q_hi = t * (1.0 - delta) * 0.999
    Qgrid = np.linspace(q_lo, q_hi, n_Q)

    TE = np.full(n_Q, np.nan)
    for i, Q in enumerate(Qgrid):
        try:
            ok = _admissible(alpha, delta, t, Q)
            if ok:
                if setting == "view":
                    eq = equilibrium_view(Q, alpha, delta, t, k)
                else:
                    eq = equilibrium_eng(Q, alpha, delta, t, k)
                TE[i] = eq["TE"]
        except Exception:
            pass

    ok = np.isfinite(TE)
    if ok.sum() < 2:
        return dict(admissible=bool(ok.sum()), neg_len=0.0, span=0.0,
                    frac=0.0, q_lo=q_lo, q_hi=q_hi)

    # measure negative slope only across runs of consecutive admissible points
    neg_len, span = 0.0, 0.0
    idx = np.where(ok)[0]
    # split into contiguous runs
    runs = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
    for run in runs:
        if len(run) < 3:
            continue
        Qr, TEr = Qgrid[run], TE[run]
        dTE = np.gradient(TEr, Qr)
        tol = rel_tol * np.nanmax(np.abs(dTE))
        for j in range(len(Qr) - 1):
            s = 0.5 * (dTE[j] + dTE[j + 1])
            seg = Qr[j + 1] - Qr[j]
            span += seg
            if s < -tol:
                neg_len += seg

    return dict(admissible=True, neg_len=neg_len, span=span,
                frac=(neg_len / span if span > 0 else 0.0),
                q_lo=Qgrid[idx[0]], q_hi=Qgrid[idx[-1]])

# ────────────────── Scan / summary / plot  ────────────────────────────────

def scan_alpha_delta(setting, alpha_vals=None, delta_vals=None,
                     baseline=None, n_Q=500):
    if baseline is None:
        baseline = BASELINE
    if alpha_vals is None:
        alpha_vals = np.linspace(RANGES["alpha"][0], RANGES["alpha"][1], 25)
    if delta_vals is None:
        delta_vals = np.linspace(0.05, 0.65, 28)
    LEN  = np.full((len(delta_vals), len(alpha_vals)), np.nan)
    FRAC = np.full((len(delta_vals), len(alpha_vals)), np.nan)
    for j, d in enumerate(delta_vals):
        for i, a in enumerate(alpha_vals):
            r = negative_length(setting, a, d, baseline, n_Q=n_Q)
            if r["admissible"]:
                LEN[j, i], FRAC[j, i] = r["neg_len"], r["frac"]
    return np.asarray(alpha_vals), np.asarray(delta_vals), LEN, FRAC

def print_summary(setting, alpha_vals, delta_vals, LEN, FRAC):
    valid = np.isfinite(LEN)
    if valid.sum() == 0:
        print(f"\n=== {setting.upper()} : no admissible cells ==="); return
    zero = np.mean(LEN[valid] <= 1e-9)
    print(f"\n=== {setting.upper()} : negative-region scan "
          f"({len(alpha_vals)}x{len(delta_vals)} grid, {valid.sum()} admissible) ===")
    print(f"  admissible cells with NO negative region : {100*zero:5.1f}%")
    print(f"  max negative length (in Q)               : {np.nanmax(LEN):.4f}")
    print(f"  max negative fraction of range           : {100*np.nanmax(FRAC):5.2f}%")
    print(f"  mean negative fraction                   : {100*np.nanmean(FRAC):5.2f}%")

def plot_negative_region_map(output_path, baseline=None, n_Q=400,
                             alpha_vals=None, delta_vals=None, use_fraction=True):
    if baseline is None:
        baseline = BASELINE

    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "dejavuserif",
        "axes.spines.top": False, "axes.spines.right": False,
    })

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    vmax = 0.0
    panels = {}
    for setting in ("view", "eng"):
        a, d, LEN, FRAC = scan_alpha_delta(setting, alpha_vals, delta_vals,
                                           baseline, n_Q=n_Q)
        Z = FRAC if use_fraction else LEN
        panels[setting] = (a, d, Z)
        print_summary(setting, a, d, LEN, FRAC)
        vmax = max(vmax, np.nanmax(Z))
    vmax = max(vmax, 1e-9)

    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad("#d9d9d9")     # grey = inadmissible (assumptions fail at beta*)

    # titles = {"view": "View-based", "eng": "Engagement-based"}
    for ax, setting in zip(axes, ("view", "eng")):
        a, d, Z = panels[setting]
        Zm = np.ma.masked_invalid(Z)
        im = ax.pcolormesh(a, d, Zm, cmap=cmap, vmin=0.0, vmax=vmax,
                           shading="auto", rasterized=True)
        ax.set_xlabel(r"AI learning efficiency $\alpha$")
        ax.set_ylabel(r"Algorithmic influence $\delta$")
        # ax.set_title(f"{titles[setting]}", fontsize=11)
    fig.subplots_adjust(right=0.88, wspace=0.28)
    fig.colorbar(im, ax=axes, location="right", fraction=0.046, pad=0.04,
                 label=(r"share of $Q$-range with $dTE/dQ<0$"
                        if use_fraction
                        else r"length of $Q$-range with $dTE/dQ<0$"))
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\n    Saved -> {output_path}")

