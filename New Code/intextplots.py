"""
---------------
Each figure illustrates a result that is *non-monotone* or that
*partitions* the parameter space, so a picture is more informative than a curve.

This module reuses the existing numerical code:
    common.m_A / m_H              – mismatch multipliers
    model_view.equilibrium_view   – view-based equilibrium
    model_engagement.equilibrium_eng, r_star_eng – engagement-based equilibrium
    params.BASELINE, RANGES       – baseline values and sweep ranges

Figures produced
----------------
  fig_vb_effort_regions.pdf      View-based creator effort q_H*(r): 3 regions in r.
                                 (after Prop. pro:Optimal:q_H:View-Based)

  fig_eb_effort_regions.pdf      Engagement-based creator effort q_H*(=r*) vs Q:
                                 3 regions (uncovered / boundary / covered).
                                 (after Prop. prop:optimal:r_beta:Eng-Based)

  fig_cmp_uH_Q_regions.pdf       Creator utility vs Q, both settings; the
                                 engagement curve is hump-shaped.
                                 (after Cor. cor:Q_uh)

  fig_cmp_signmap_uH_alpha.pdf   Sign of du_H^V/d(alpha) over the (Q, alpha)
                                 plane.  (after Cor. cor:alpha_uh)

  fig_cmp_signmap_beta_Q.pdf     Sign of d(beta_H*)/dQ over the (Q, alpha)
                                 plane, view-based.  (after Cor. cor:Q_beta)

"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches

from params import BASELINE, RANGES
from common import m_A, m_H
from model_view import equilibrium_view
from model_engagement import equilibrium_eng
from model_engagement import r_star_eng


# ── Style (mirrors plots.py) ──────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "mathtext.fontset": "dejavuserif",
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.linewidth":   0.8,
    "grid.linewidth":   0.4,
    "grid.color":       "#cccccc",
    "xtick.direction":  "in",
    "ytick.direction":  "in",
})

COL_VIEW = "#8f07aa"   # view-based   (matches plots.py)
COL_ENG  = "#0f21b0"   # engagement   (matches plots.py)
GREY     = "#666666"

# three soft, neutral region tints (uncovered / boundary / covered / capped)
SHADE = [ "#efe9f3", "#e8edf4", "#e9f1ec", "#f7f4e7"]

# sign-map colours: decreasing / flat-or-clamped / increasing
_SMCOLORS = ["#c44e63", "#e9ecef", "#3a8c5f"]
_SMCMAP   = ListedColormap(_SMCOLORS)
_SMNORM   = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], _SMCMAP.N)

# resolution for the 2-D sign maps (finite-difference, so each cell solves twice)
SIGNMAP_GRID = 170


# ──────────────────────────────────────────────────────────────────────────────
# Local closed-form helpers (Stage-2 / fixed-beta objects not exposed elsewhere)
# ──────────────────────────────────────────────────────────────────────────────

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


def _shade_regions(ax, edges, xmax):
    """Shade [0,e1],[e1,e2],[e2,e3],[e3,xmax] with the four SHADE tints."""
    e1, e2, e3 = edges
    ax.axvspan(0, e1, color=SHADE[0], zorder=0)
    ax.axvspan(e1, e2, color=SHADE[1], zorder=0)
    ax.axvspan(e2, e3, color=SHADE[2], zorder=0)
    ax.axvspan(e3, xmax, color=SHADE[3], zorder=0)
    for e in edges:
        ax.axvline(e, color=GREY, ls="--", lw=0.9, zorder=1)


# ──────────────────────────────────────────────────────────────────────────────
# 1. View-based effort regions  (vs the compensation rate r, at beta_H = 1/2)
# ──────────────────────────────────────────────────────────────────────────────

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
    _shade_regions(ax, (r1, r2, r_c), rmax)
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


# ──────────────────────────────────────────────────────────────────────────────
# 2. Engagement-based effort regions  (vs AI baseline quality Q, at beta_H = 1/2)
#    Effort = rate here (q_H* = r*), so r* selecting among three branches
#    traces three regions in equilibrium effort.
# ──────────────────────────────────────────────────────────────────────────────

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
    _shade_regions(ax, (QEc, QEcp), Qmax)
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
    ax.set_title("Engagement-based: equilibrium effort has three regions in $Q$\n"
                 rf"(illustrative $t={t},\ \alpha={alpha},\ \delta={delta},\ \beta_H=1/2$)",
                 fontsize=10.5)
    ax.legend(loc="lower left", fontsize=8.3, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"    Saved -> {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 3. Creator utility vs Q  (engagement hump)  — Cor. cor:Q_uh
# ──────────────────────────────────────────────────────────────────────────────

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

def _sweep_te(values, param, baseline):
    uV = np.full(len(values), np.nan)
    uE = np.full(len(values), np.nan)
    from model_engagement import equilibrium_eng
    for i, v in enumerate(values):
        p = {**baseline, param: v}
        try:
            uV[i] = equilibrium_view(p["Q"], p["alpha"], p["delta"], p["t"], p["k"])["TE"]
            uE[i] = equilibrium_eng(p["Q"], p["alpha"], p["delta"], p["t"], p["k"])["TE"]
        except Exception:
            pass
    return uV, uE

def _sweep_beta(values, param, baseline):
    uV = np.full(len(values), np.nan)
    uE = np.full(len(values), np.nan)
    from model_engagement import equilibrium_eng
    for i, v in enumerate(values):
        p = {**baseline, param: v}
        try:
            uV[i] = equilibrium_view(p["Q"], p["alpha"], p["delta"], p["t"], p["k"])["beta_H"]
            uE[i] = equilibrium_eng(p["Q"], p["alpha"], p["delta"], p["t"], p["k"])["beta_H"]
        except Exception:
            pass
    return uV, uE

def plot_cmp_uH_Q_regions(output_path, baseline=None):
    if baseline is None:
        baseline = BASELINE
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

    # fig, ax = plt.subplots(figsize=(6.4, 4.1))
    # ax.plot(Qs, uV_0_1, lw=2.2, label=r"$\alpha=0.1$")
    # ax.plot(Qs, uV_0_2, lw=2.2, label=r"$\alpha=0.2$")
    # ax.plot(Qs, uV_0_3, lw=2.2, label=r"$\alpha=0.3$")
    # ax.plot(Qs, uV_0_4, lw=2.2, label=r"$\alpha=0.4$")
    # ax.plot(Qs, uV_0_5, lw=2.2, label=r"$\alpha=0.5$")
    # ax.plot(Qs, uV_0_6, lw=2.2, label=r"$\alpha=0.6$")
    # ax.plot(Qs, uV_0_7, lw=2.2, label=r"$\alpha=0.7$")

    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    ax.plot(Qs, uE_0_1, lw=2.2, label=r"$\alpha=0.1$")
    ax.plot(Qs, uE_0_2, lw=2.2, label=r"$\alpha=0.2$")
    ax.plot(Qs, uE_0_3, lw=2.2, label=r"$\alpha=0.3$")
    ax.plot(Qs, uE_0_4, lw=2.2, label=r"$\alpha=0.4$")
    ax.plot(Qs, uE_0_5, lw=2.2, label=r"$\alpha=0.5$")
    ax.plot(Qs, uE_0_6, lw=2.2, label=r"$\alpha=0.6$")
    ax.plot(Qs, uE_0_7, lw=2.2, label=r"$\alpha=0.7$")

    ax.set_xlim(lo, hi)
    ax.set_xlabel(r"AI baseline quality $Q$")
    ax.set_ylabel(r"Creator utility $u_H^*$")
    ax.grid(True, axis="y")
    ax.legend(loc="upper right", fontsize=9.5, framealpha=0.92)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"    Saved -> {output_path}")

def plot_cmp_te_Q_regions(output_path, baseline=None):
    if baseline is None:
        baseline = BASELINE
    lo, hi = RANGES["Q"]
    Qs = np.linspace(lo, hi, 400)

    baseline["alpha"] = 0.1
    uV_0_1, uE_0_1 = _sweep_te(Qs, "Q", baseline)
    baseline["alpha"] = 0.2
    uV_0_2, uE_0_2 = _sweep_te(Qs, "Q", baseline)
    baseline["alpha"] = 0.3
    uV_0_3, uE_0_3 = _sweep_te(Qs, "Q", baseline)
    baseline["alpha"] = 0.4
    uV_0_4, uE_0_4 = _sweep_te(Qs, "Q", baseline)
    baseline["alpha"] = 0.5
    uV_0_5, uE_0_5 = _sweep_te(Qs, "Q", baseline)
    baseline["alpha"] = 0.6
    uV_0_6, uE_0_6 = _sweep_te(Qs, "Q", baseline)
    baseline["alpha"] = 0.7
    uV_0_7, uE_0_7 = _sweep_te(Qs, "Q", baseline)

    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    ax.plot(Qs, uV_0_1, lw=2.2, label=r"$\alpha=0.1$")
    ax.plot(Qs, uV_0_2, lw=2.2, label=r"$\alpha=0.2$")
    ax.plot(Qs, uV_0_3, lw=2.2, label=r"$\alpha=0.3$")
    ax.plot(Qs, uV_0_4, lw=2.2, label=r"$\alpha=0.4$")
    ax.plot(Qs, uV_0_5, lw=2.2, label=r"$\alpha=0.5$")
    ax.plot(Qs, uV_0_6, lw=2.2, label=r"$\alpha=0.6$")
    ax.plot(Qs, uV_0_7, lw=2.2, label=r"$\alpha=0.7$")

    # fig, ax = plt.subplots(figsize=(6.4, 4.1))
    # ax.plot(Qs, uE_0_1, lw=2.2, label=r"$\alpha=0.1$")
    # ax.plot(Qs, uE_0_2, lw=2.2, label=r"$\alpha=0.2$")
    # ax.plot(Qs, uE_0_3, lw=2.2, label=r"$\alpha=0.3$")
    # ax.plot(Qs, uE_0_4, lw=2.2, label=r"$\alpha=0.4$")
    # ax.plot(Qs, uE_0_5, lw=2.2, label=r"$\alpha=0.5$")
    # ax.plot(Qs, uE_0_6, lw=2.2, label=r"$\alpha=0.6$")
    # ax.plot(Qs, uE_0_7, lw=2.2, label=r"$\alpha=0.7$")

    ax.set_xlim(lo, hi)
    ax.set_xlabel(r"AI baseline quality $Q$")
    ax.set_ylabel(r"Total Engagement $TE^*$")
    ax.grid(True, axis="y")
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.92)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"    Saved -> {output_path}")

def plot_cmp_uH_alpha_regions(output_path, baseline=None):
    if baseline is None:
        baseline = BASELINE
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

    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    ax.plot(alphas, uV_0_3, lw=2.2, label=r"$Q=0.3$")
    ax.plot(alphas, uV_0_4, lw=2.2, label=r"$Q=0.4$")
    ax.plot(alphas, uV_0_5, lw=2.2, label=r"$Q=0.5$")
    ax.plot(alphas, uV_0_6, lw=2.2, label=r"$Q=0.6$")
    ax.plot(alphas, uV_0_7, lw=2.2, label=r"$Q=0.7$")
    ax.plot(alphas, uV_0_8, lw=2.2, label=r"$Q=0.8$")
    ax.plot(alphas, uV_0_9, lw=2.2, label=r"$Q=0.9$")

    # fig, ax = plt.subplots(figsize=(6.4, 4.1))
    # ax.plot(alphas, uE_0_3, lw=2.2, label=r"$Q=0.3$")
    # ax.plot(alphas, uE_0_4, lw=2.2, label=r"$Q=0.4$")
    # ax.plot(alphas, uE_0_5, lw=2.2, label=r"$Q=0.5$")
    # ax.plot(alphas, uE_0_6, lw=2.2, label=r"$Q=0.6$")
    # ax.plot(alphas, uE_0_7, lw=2.2, label=r"$Q=0.7$")
    # ax.plot(alphas, uE_0_8, lw=2.2, label=r"$Q=0.8$")
    # ax.plot(alphas, uE_0_9, lw=2.2, label=r"$Q=0.9$")

    ax.set_xlim(lo, hi)
    ax.set_xlabel(r"AI Efficiency $\alpha$")
    ax.set_ylabel(r"Creator utility $u_H^*$")
    ax.grid(True, axis="y")
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.92)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"    Saved -> {output_path}")

def plot_cmp_te_alpha_regions(output_path, baseline=None):
    if baseline is None:
        baseline = BASELINE
    lo, hi = RANGES["alpha"]
    alphas = np.linspace(lo, hi, 400)
    
    baseline["Q"] = 0.3
    uV_0_3, uE_0_3 = _sweep_te(alphas, "alpha", baseline)
    baseline["Q"] = 0.4
    uV_0_4, uE_0_4 = _sweep_te(alphas, "alpha", baseline)
    baseline["Q"] = 0.5
    uV_0_5, uE_0_5 = _sweep_te(alphas, "alpha", baseline)
    baseline["Q"] = 0.6
    uV_0_6, uE_0_6 = _sweep_te(alphas, "alpha", baseline)
    baseline["Q"] = 0.7
    uV_0_7, uE_0_7 = _sweep_te(alphas, "alpha", baseline)
    baseline["Q"] = 0.8
    uV_0_8, uE_0_8 = _sweep_te(alphas, "alpha", baseline)
    baseline["Q"] = 0.9
    uV_0_9, uE_0_9 = _sweep_te(alphas, "alpha", baseline)

    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    ax.plot(alphas, uV_0_3, lw=2.2, label=r"$Q=0.3$")
    ax.plot(alphas, uV_0_4, lw=2.2, label=r"$Q=0.4$")
    ax.plot(alphas, uV_0_5, lw=2.2, label=r"$Q=0.5$")
    ax.plot(alphas, uV_0_6, lw=2.2, label=r"$Q=0.6$")
    ax.plot(alphas, uV_0_7, lw=2.2, label=r"$Q=0.7$")
    ax.plot(alphas, uV_0_8, lw=2.2, label=r"$Q=0.8$")
    ax.plot(alphas, uV_0_9, lw=2.2, label=r"$Q=0.9$")

    # fig, ax = plt.subplots(figsize=(6.4, 4.1))
    # ax.plot(alphas, uE_0_3, lw=2.2, label=r"$Q=0.3$")
    # ax.plot(alphas, uE_0_4, lw=2.2, label=r"$Q=0.4$")
    # ax.plot(alphas, uE_0_5, lw=2.2, label=r"$Q=0.5$")
    # ax.plot(alphas, uE_0_6, lw=2.2, label=r"$Q=0.6$")
    # ax.plot(alphas, uE_0_7, lw=2.2, label=r"$Q=0.7$")
    # ax.plot(alphas, uE_0_8, lw=2.2, label=r"$Q=0.8$")
    # ax.plot(alphas, uE_0_9, lw=2.2, label=r"$Q=0.9$")

    ax.set_xlim(lo, hi)
    ax.set_xlabel(r"AI Efficiency $\alpha$")
    ax.set_ylabel(r"Total Engagement $TE^*$")
    ax.grid(True, axis="y")
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.92)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"    Saved -> {output_path}")

def plot_cmp_beta_Q_regions(output_path, baseline=None):
    if baseline is None:
        baseline = BASELINE
    lo, hi = RANGES["Q"]
    Qs = np.linspace(lo, hi, 400)
    baseline["alpha"] = 0.1
    uV_0_1, uE_0_1 = _sweep_beta(Qs, "Q", baseline)
    baseline["alpha"] = 0.2
    uV_0_2, uE_0_2 = _sweep_beta(Qs, "Q", baseline)
    baseline["alpha"] = 0.3
    uV_0_3, uE_0_3 = _sweep_beta(Qs, "Q", baseline)
    baseline["alpha"] = 0.4
    uV_0_4, uE_0_4 = _sweep_beta(Qs, "Q", baseline)
    baseline["alpha"] = 0.5
    uV_0_5, uE_0_5 = _sweep_beta(Qs, "Q", baseline)
    baseline["alpha"] = 0.6
    uV_0_6, uE_0_6 = _sweep_beta(Qs, "Q", baseline)
    baseline["alpha"] = 0.7
    uV_0_7, uE_0_7 = _sweep_beta(Qs, "Q", baseline)

    # fig, ax = plt.subplots(figsize=(6.4, 4.1))
    # ax.plot(Qs, uV_0_1, lw=2.2, label=r"$\alpha=0.1$")
    # ax.plot(Qs, uV_0_2, lw=2.2, label=r"$\alpha=0.2$")
    # ax.plot(Qs, uV_0_3, lw=2.2, label=r"$\alpha=0.3$")
    # ax.plot(Qs, uV_0_4, lw=2.2, label=r"$\alpha=0.4$")
    # ax.plot(Qs, uV_0_5, lw=2.2, label=r"$\alpha=0.5$")
    # ax.plot(Qs, uV_0_6, lw=2.2, label=r"$\alpha=0.6$")
    # ax.plot(Qs, uV_0_7, lw=2.2, label=r"$\alpha=0.7$")

    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    ax.plot(Qs, uE_0_1, lw=2.2, label=r"$\alpha=0.1$")
    ax.plot(Qs, uE_0_2, lw=2.2, label=r"$\alpha=0.2$")
    ax.plot(Qs, uE_0_3, lw=2.2, label=r"$\alpha=0.3$")
    ax.plot(Qs, uE_0_4, lw=2.2, label=r"$\alpha=0.4$")
    ax.plot(Qs, uE_0_5, lw=2.2, label=r"$\alpha=0.5$")
    ax.plot(Qs, uE_0_6, lw=2.2, label=r"$\alpha=0.6$")
    ax.plot(Qs, uE_0_7, lw=2.2, label=r"$\alpha=0.7$")

    ax.set_xlim(lo, hi)
    ax.set_xlabel(r"AI baseline quality $Q$")
    ax.set_ylabel(r"Promotion Weight $\beta_H^*$")
    ax.grid(True, axis="y")
    ax.legend(loc="upper right", fontsize=9.5, framealpha=0.92)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"    Saved -> {output_path}")

def plot_cmp_beta_alpha_regions(output_path, baseline=None):
    if baseline is None:
        baseline = BASELINE
    lo, hi = RANGES["alpha"]
    alphas = np.linspace(lo, hi, 400)
    
    baseline["Q"] = 0.3
    uV_0_3, uE_0_3 = _sweep_beta(alphas, "alpha", baseline)
    baseline["Q"] = 0.4
    uV_0_4, uE_0_4 = _sweep_beta(alphas, "alpha", baseline)
    baseline["Q"] = 0.5
    uV_0_5, uE_0_5 = _sweep_beta(alphas, "alpha", baseline)
    baseline["Q"] = 0.6
    uV_0_6, uE_0_6 = _sweep_beta(alphas, "alpha", baseline)
    baseline["Q"] = 0.7
    uV_0_7, uE_0_7 = _sweep_beta(alphas, "alpha", baseline)
    baseline["Q"] = 0.8
    uV_0_8, uE_0_8 = _sweep_beta(alphas, "alpha", baseline)
    baseline["Q"] = 0.9
    uV_0_9, uE_0_9 = _sweep_beta(alphas, "alpha", baseline)

    # fig, ax = plt.subplots(figsize=(6.4, 4.1))
    # ax.plot(alphas, uV_0_3, lw=2.2, label=r"$Q=0.3$")
    # ax.plot(alphas, uV_0_4, lw=2.2, label=r"$Q=0.4$")
    # ax.plot(alphas, uV_0_5, lw=2.2, label=r"$Q=0.5$")
    # ax.plot(alphas, uV_0_6, lw=2.2, label=r"$Q=0.6$")
    # ax.plot(alphas, uV_0_7, lw=2.2, label=r"$Q=0.7$")
    # ax.plot(alphas, uV_0_8, lw=2.2, label=r"$Q=0.8$")
    # ax.plot(alphas, uV_0_9, lw=2.2, label=r"$Q=0.9$")

    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    ax.plot(alphas, uE_0_3, lw=2.2, label=r"$Q=0.3$")
    ax.plot(alphas, uE_0_4, lw=2.2, label=r"$Q=0.4$")
    ax.plot(alphas, uE_0_5, lw=2.2, label=r"$Q=0.5$")
    ax.plot(alphas, uE_0_6, lw=2.2, label=r"$Q=0.6$")
    ax.plot(alphas, uE_0_7, lw=2.2, label=r"$Q=0.7$")
    ax.plot(alphas, uE_0_8, lw=2.2, label=r"$Q=0.8$")
    ax.plot(alphas, uE_0_9, lw=2.2, label=r"$Q=0.9$")

    ax.set_xlim(lo, hi)
    ax.set_xlabel(r"AI Efficiency $\alpha$")
    ax.set_ylabel(r"Promotion Weight $\beta_H^*$")
    ax.grid(True, axis="y")
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.92)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"    Saved -> {output_path}")

# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_OUT = os.path.join(_SCRIPT_DIR, "..", "figures")


def main(output_dir=_DEFAULT_OUT):
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    print("[1/8] View-based effort regions")
    plot_vb_effort_regions(os.path.join(output_dir, "fig_vb_effort_regions.pdf"))

    # print("[2/8] Engagement-based effort regions")
    # plot_eb_effort_regions(os.path.join(output_dir, "fig_eb_effort_regions.pdf"))

    # print("[3/8] Creator utility vs Q")
    # plot_cmp_uH_Q_regions(os.path.join(output_dir, "fig_cmp_uH_Q_regions.pdf"))

    # print("[4/8] Total Engagement vs Q")
    # plot_cmp_te_Q_regions(os.path.join(output_dir, "fig_cmp_te_Q_regions.pdf"))

    # print("[5/8] Creator utility vs α")
    # plot_cmp_uH_alpha_regions(os.path.join(output_dir, "fig_cmp_uH_alpha_regions.pdf"))

    # print("[6/8] Total Engagement vs α)")
    # plot_cmp_te_alpha_regions(os.path.join(output_dir, "fig_cmp_te_alpha_regions.pdf"))

    # print("[7/8] Beta_H vs Q")
    # plot_cmp_beta_Q_regions(os.path.join(output_dir, "fig_cmp_beta_Q_regions.pdf"))

    # print("[8/8] Beta_H vs α)")
    # plot_cmp_beta_alpha_regions(os.path.join(output_dir, "fig_cmp_beta_alpha_regions.pdf"))

    # print(f"\n{'-'*55}\nAll region figures written to '{output_dir}'")

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_OUT
    main(out)