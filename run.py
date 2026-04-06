"""
Entry point: runs existing experiments then the interior-solution search.
"""

from main import main as run_main

# ── existing experiments ──────────────────────────────────────────────────────
# run_main()

# # ── interior-solution search ──────────────────────────────────────────────────
from interior_search import scan_2d, plot_interior_regions

pairs = [
    ("alpha", "Q"),
    ("alpha", "delta"),
    ("alpha", "u0"),
    ("Q",     "delta"),
    ("Q",     "u0"),
    ("delta", "u0"),
]

for type in ["engagement", "view"]:
    results = {(x, y): scan_2d(type, x, y) for x, y in pairs}
    plot_interior_regions(type, results)
