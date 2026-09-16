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
from params import BASELINE, RANGES

# Allow running from any working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from plot_functions import (
    plot_comparative_statics,
    plot_preference_map_Q_alpha,
    plot_preference_map_delta,
    plot_vb_effort_regions,
    plot_eb_effort_regions,
    plot_cmp_uH_Q_regions,
    plot_cmp_uH_alpha_regions,
    plot_negative_region_map
)


# ── Default output directory ──────────────────────────────────────────────────
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_OUT = os.path.join(_SCRIPT_DIR, "..", "figures")

def prompt_figure(figures):
    """Ask which figure to generate. Returns a choice key or 'all'."""
    print("\nWhich figure do you want to generate?")
    for key in sorted(figures, key=int):
        print(f"  {key}: {figures[key][0]}")
    while True:
        choice = input("Figure number: ").strip().lower()
        if choice in figures:
            return choice
        SystemExit("Not a valid choice, try again.")

def prompt_baseline(fig, defaults=BASELINE):
    """Ask the user for parameters for one figure. Enter keeps the default."""
    print("Press Enter to keep the value in [brackets].")
    chosen = {}
    for name, default in defaults.items():
        while True:
            raw = input(f"  {name} [{default}]: ").strip()
            if raw == "":
                chosen[name] = default          # keep default
                break
            try:
                chosen[name] = float(raw)        # use what they typed
                break
            except ValueError:
                print(f"    '{raw}' is not a number, try again.")
    ok, problems = check_assumptions(chosen, fig=fig)
    if ok:
        return chosen
    print("  Assumptions violated:")
    for msg in problems:
        print(f"    - {msg}")
    raise SystemExit("Fix the parameters and run again.")

def check_assumptions(p, fig):
    # fig = str(fig)
    # if fig == "2":
    #     Q = p["Q"]
    #     alpha = p["alpha"]
    # elif fig == "3":
    #     Q = RANGES["Q"][1]
    #     alpha = p["alpha"]
    # elif fig == "5":
    #     Q = 0.9
    #     alpha = RANGES["alpha"][1]
    # else:
    Q = RANGES["Q"][1]
    alpha = RANGES["alpha"][1]
    delta = p["delta"]
    t = p["t"]
    problems = []
    if not (Q < t * (1 - delta)):
        problems.append(f"non-saturation fails at Q={Q}: need Q < t(1-delta) = {t*(1-delta):.3f}")
    if not (t > alpha**2 + 1 / (1 - delta)):
        problems.append(f"concavity fails at alpha={alpha}: need t > alpha**2 + 1/(1-delta) = {alpha**2 + 1/(1-delta):.3f}")
    return (len(problems) == 0), problems

def main(output_dir: str = _DEFAULT_OUT) -> None:
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}\n")
    
    figures = {
        "2":  ("View-based effort regions",
               lambda p: plot_vb_effort_regions(
                   os.path.join(output_dir, "fig_vb_effort_regions.pdf"), baseline=p)),
        "3":  ("Engagement-based effort regions",
               lambda p: plot_eb_effort_regions(
                   os.path.join(output_dir, "fig_eb_effort_regions.pdf"), baseline=p)),
        "4":  ("Creator utility vs Q",
               lambda p: plot_cmp_uH_Q_regions(
                   os.path.join(output_dir, "fig_cmp_uH_Q_regions.pdf"), baseline=p)),
        "5":  ("Creator utility vs alpha",
               lambda p: plot_cmp_uH_alpha_regions(
                   os.path.join(output_dir, "fig_cmp_uH_alpha_regions.pdf"), baseline=p)),
        "6":  ("Preference map (Q, alpha)",
               lambda p: plot_preference_map_Q_alpha(
                   os.path.join(output_dir, "fig_preference_map_Q_alpha.pdf"), baseline=p)),
        "7":  ("Preference maps (Q, delta) and (alpha, delta)",
               lambda p: plot_preference_map_delta(
                   os.path.join(output_dir, "fig_preference_map_delta.pdf"), baseline=p)),
        "8":  ("Negative-TE region map",
               lambda p: plot_negative_region_map(
                   os.path.join(output_dir, "fig_negative_TE_Q_region.pdf"),
                   baseline=p, use_fraction=True)),
        "9":  ("Comparative statics in Q",
               lambda p: plot_comparative_statics(
                   "Q", os.path.join(output_dir, "fig_Q_comparative_statics.pdf"), baseline=p)),
        "10": ("Comparative statics in alpha",
               lambda p: plot_comparative_statics(
                   "alpha", os.path.join(output_dir, "fig_alpha_comparative_statics.pdf"), baseline=p)),
    }
    
    choice = prompt_figure(figures)
    p = prompt_baseline(fig=choice)

    label, runner = figures[choice]
    print(f"[{label}]")
    runner(p)

    print(f"\nDone. Figures written to '{output_dir}'")

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_OUT
    main(out)
