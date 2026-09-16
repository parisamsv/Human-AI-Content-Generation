"""
Figures generated
-----------------
  fig_Q_comparative_statics.pdf     — β_H*, q_H*, u_P, u_H, TV, TE vs Q
  fig_alpha_comparative_statics.pdf — same panels vs α
  fig_preference_map_Q_alpha.pdf    — heatmap: four regions in (Q, α) space
  fig_preference_map_delta.pdf      — two heatmaps: (Q,δ) and (α,δ) side by side
"""

import sys
import os

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


def main(output_dir: str = _DEFAULT_OUT) -> None:
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}\n")
    
    # # ── Figure 2: View-based creator effort q_H*(r) vs r ───────────────────────
    # print("[Figure 2] View-based effort regions")
    # plot_vb_effort_regions(os.path.join(output_dir, "fig_vb_effort_regions.pdf"))

    # # ── Figure 3: Engagement-based creator effort q_H*(=r*) vs Q ───────────────
    # print("[Figure 3] Engagement-based effort regions")
    # plot_eb_effort_regions(os.path.join(output_dir, "fig_eb_effort_regions.pdf"))

    # ── Figure 4: Creator utility vs Q, both settings ──────────────────────────       
    print("[Figure 4] Creator utility vs Q")
    plot_cmp_uH_Q_regions(os.path.join(output_dir, "fig_cmp_uH_Q_regions.pdf"))

    # # ── Figure 5: Creator utility vs alpha, both settings ──────────────────────
    # print("[Figure 5] Creator utility vs α")
    # plot_cmp_uH_alpha_regions(os.path.join(output_dir, "fig_cmp_uH_alpha_regions.pdf"))

    # # ── Figure 6: Preference maps: (Q, α) plane ──────────────────────────────────────
    # print("\n[Figure 6] Preference map: (Q, α) plane")
    # plot_preference_map_Q_alpha(os.path.join(output_dir, "fig_preference_map_Q_alpha.pdf"))

    # # ── Figure 7: Preference maps: (Q, δ) and (α, δ) planes ─────────────────────────
    # print("\n[Figure 7] Preference maps: (Q, δ) and (α, δ) planes")
    # plot_preference_map_delta(os.path.join(output_dir, "fig_preference_map_delta.pdf"))

    # # ── Figure 8: measure the length of the Q-interval where dTE/dQ < 0 over an (alpha,delta) grid
    # print("[Figure 8] length of Q-interval where dTE/dQ < 0 over an (α, δ) grid")
    # plot_negative_region_map(
    #     os.path.join(output_dir, "fig_negative_TE_Q_region.pdf"),
    #     use_fraction=True)

    # # ── Figure 9: Comparative statics: AI baseline quality Q ────────────────────────
    # print("[Figure 9] Comparative statics: AI baseline quality Q")
    # plot_comparative_statics(
    #     "Q",
    #     os.path.join(output_dir, "fig_Q_comparative_statics.pdf"))

    # # ── Figure 10: Comparative statics: AI learning efficiency α ────────────────────
    # print("\n[Figure 10] Comparative statics: AI learning efficiency α")
    # plot_comparative_statics(
    #     "alpha",
    #     os.path.join(output_dir, "fig_alpha_comparative_statics.pdf"))

    print(f"All figures written to '{output_dir}'")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_OUT
    main(out)
