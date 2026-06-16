"""
negative_region.py  (optimal-beta version)
-------------------------------------------
Same purpose as before -- measure the length of the Q-interval where
dTE/dQ < 0 (the negative branch of Corollary cor:Q_TE) over an (alpha,delta)
grid -- but every cell now uses the EQUILIBRIUM algorithmic weight
beta_H*(Q) implied by the parameters, instead of a fixed beta_H = 1/2.

What changed vs. the old script
-------------------------------
1. The equilibrium is solved with the beta_H* fixed point
       beta_H* = clip( 1/2 + M(beta_H*)/k , 0, 1 )
   (Proposition prop:beta_view / prop:beta_eng).  TE(Q) is read off that
   equilibrium, so it is no longer pinned at beta=1/2.

2. Admissibility (non-saturation + concavity-E) is checked AT beta_H*(Q),
   per Q point, not with the worst-case-over-beta rule.  Because beta_H*
   moves with Q, a cell can be partly admissible: inadmissible Q points are
   simply dropped before the slope is measured.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from params import BASELINE, RANGES
from model_view import equilibrium_view
from model_engagement import equilibrium_eng


# ──────────────────────────────────────────────────────────────────────────
# Admissibility, now evaluated at beta_H*(Q)
# ──────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────
# Core measurement (optimal beta per Q)
# ──────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────
# Scan / summary / plot  (unchanged interface)
# ──────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────
# Heatmap figure  (same output as before: fig_negative_TE_Q_region.pdf)
# ──────────────────────────────────────────────────────────────────────────

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
        fig.colorbar(im, ax=ax, label=(r"share of $Q$-range with $dTE/dQ<0$"
                                       if use_fraction
                                       else r"length of $Q$-range with $dTE/dQ<0$"))
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\n    Saved -> {output_path}")


# ──────────────────────────────────────────────────────────────────────────
# Entry point  (unchanged usage:  python negative_region.py [output_dir])
# ──────────────────────────────────────────────────────────────────────────

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_OUT = os.path.join(_SCRIPT_DIR, "..", "figures")


def main(output_dir=_DEFAULT_OUT):
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    plot_negative_region_map(
        os.path.join(output_dir, "fig_negative_TE_Q_region.pdf"),
        use_fraction=True,
    )


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_OUT
    main(out)