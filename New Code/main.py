"""
main.py
-------
Entry point for the numerical study.
Generates all five figures for Section 4 of the paper.

Usage
-----
    cd numerical
    python main.py [output_dir]

`output_dir` defaults to  ../figures/  relative to this script.
Every figure is saved as a PDF.

Figures generated
-----------------
  fig_Q_comparative_statics.pdf     — β_H*, q_H*, u_P, u_H, TV, TE vs Q
  fig_alpha_comparative_statics.pdf — same panels vs α
  fig_delta_comparative_statics.pdf — same panels vs δ
  fig_preference_map_Q_alpha.pdf    — heatmap: four regions in (Q, α) space
  fig_preference_map_delta.pdf      — two heatmaps: (Q,δ) and (α,δ) side by side
"""

import sys
import os
import time

# Allow running from any working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from params import BASELINE
from plots import (
    plot_comparative_statics,
    plot_preference_map_Q_alpha,
    plot_preference_map_delta,
)


# ── Default output directory ──────────────────────────────────────────────────
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_OUT = os.path.join(_SCRIPT_DIR, "..", "figures")


def main(output_dir: str = _DEFAULT_OUT) -> None:
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    t0 = time.time()

    # ── 1. Comparative statics: AI baseline quality Q ────────────────────────
    print("[1/5] Comparative statics: AI baseline quality Q")
    plot_comparative_statics(
        "Q",
        os.path.join(output_dir, "fig_Q_comparative_statics.pdf"),
    )

    # ── 2. Comparative statics: AI learning efficiency α ────────────────────
    print("\n[2/5] Comparative statics: AI learning efficiency α")
    plot_comparative_statics(
        "alpha",
        os.path.join(output_dir, "fig_alpha_comparative_statics.pdf"),
    )

    # ── 3. Comparative statics: algorithmic influence δ ──────────────────────
    print("\n[3/5] Comparative statics: algorithmic influence δ")
    plot_comparative_statics(
        "delta",
        os.path.join(output_dir, "fig_delta_comparative_statics.pdf"),
    )

    # ── 4. Preference map: (Q, α) plane ──────────────────────────────────────
    print("\n[4/5] Preference map: (Q, α) plane")
    plot_preference_map_Q_alpha(
        os.path.join(output_dir, "fig_preference_map_Q_alpha.pdf"),
    )

    # ── 5. Preference maps: (Q, δ) and (α, δ) planes ─────────────────────────
    print("\n[5/5] Preference maps: (Q, δ) and (α, δ) planes")
    plot_preference_map_delta(
        os.path.join(output_dir, "fig_preference_map_delta.pdf"),
    )

    elapsed = time.time() - t0
    print(f"\n{'─'*55}")
    print(f"All 5 figures written to '{output_dir}'")
    print(f"Total runtime: {elapsed:.1f} s")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_OUT
    main(out)
