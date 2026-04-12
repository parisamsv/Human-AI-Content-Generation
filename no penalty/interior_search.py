"""
Search for parameter regions where the optimal promotion weight beta* is interior,
i.e. strictly between 0.01 and 0.99.

Three public functions
----------------------
scan_2d              -- sweep two parameters over a grid and find beta* at each point
plot_interior_regions -- heatmap figure for all six parameter pairs
report_interior_ranges -- stdout summary of interior regions
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from params import BASELINE
from simulator import PlatformRevenueSimulator


# ── ordered list used by both plot and report ─────────────────────────────────
_PAIRS = [
    ("alpha", "Q"),
    ("alpha", "delta"),
    ("alpha", "u0"),
    ("Q",     "delta"),
    ("Q",     "u0"),
    ("delta", "u0"),
]


# ─────────────────────────────────────────────────────────────────────────────
def scan_2d(type, param_x, param_y, grid_size=100, threshold=0.01):
    """
    Sweep two parameters simultaneously over [0.01, 0.99]^2.

    For every (param_x, param_y) grid point, compute

        beta* = argmax_{beta in [0.01, 0.99]} u_p(beta)

    holding all other parameters at BASELINE.

    Parameters
    ----------
    param_x, param_y : str
        Names of the two parameters to sweep (keys in BASELINE).
    grid_size : int
        Number of points along each axis (default 100 → 10 000 evaluations).
    threshold : float
        beta* is classified as interior when threshold < beta* < 1 - threshold.

    Returns
    -------
    beta_star_grid : ndarray, shape (grid_size, grid_size)
        Optimal beta* at each grid point.
        Axis 0 ↔ param_x values, axis 1 ↔ param_y values.
    is_interior : ndarray of bool, shape (grid_size, grid_size)
        True where the solution is interior.
    interior_share : float
        Fraction of grid points with interior solutions.
    """
    xs = np.linspace(0.01, 0.99, grid_size)
    ys = np.linspace(0.01, 0.99, grid_size)

    beta_star_grid = np.empty((grid_size, grid_size))

    for i, xv in enumerate(xs):
        for j, yv in enumerate(ys):
            params = dict(BASELINE)
            params[param_x] = xv
            params[param_y] = yv
            sim = PlatformRevenueSimulator(t=params['t'], alpha=params['alpha'], Q=params['Q'], u_0=params['u0'], delta=params['delta'])
            # Optimize both settings
            # result_view = sim.optimize_view_based()
            if type == "engagement":
                result_eng = sim.optimize_engagement_based()
                beta_star_grid[i, j] = result_eng['beta_h']
            elif type == "view":
                result_view = sim.optimize_view_based()
                beta_star_grid[i, j] = result_view['beta_h']    

    is_interior = (beta_star_grid > 0) & (beta_star_grid < 1.0)
    interior_share = float(is_interior.mean())
    return beta_star_grid, is_interior, interior_share


# ─────────────────────────────────────────────────────────────────────────────
def plot_interior_regions(type, results):
    """
    2x3 grid of beta* heatmaps, one subplot per parameter pair.

    Parameters
    ----------
    results : dict
        Mapping (param_x, param_y) → (beta_star_grid, is_interior, interior_share)
        as returned by scan_2d.
    filename : str
        Output path for the saved figure (directory is created if absent).
    """
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    for idx, (px, py) in enumerate(_PAIRS):
        ax = axes[idx // 3, idx % 3]
        beta_star_grid, is_interior, interior_share = results[(px, py)]

        grid_size = beta_star_grid.shape[0]
        xs = np.linspace(0.01, 0.99, grid_size)
        ys = np.linspace(0.01, 0.99, grid_size)

        # heatmap: axis 0 of beta_star_grid → param_x (x-axis of plot)
        #          axis 1                   → param_y (y-axis of plot)
        # pcolormesh expects [row=y, col=x], so transpose.
        mesh = ax.pcolormesh(
            xs, ys, beta_star_grid.T,
            cmap="RdBu_r", vmin=0.0, vmax=1.0, shading="auto",
        )

        # # white contour at the interior / corner boundary
        # if is_interior.any() and not is_interior.all():
        #     ax.contour(
        #         xs, ys, is_interior.T.astype(float),
        #         levels=[0.5], colors="white", linewidths=1.5,
        #     )

        ax.set_xlabel(px, fontsize=11)
        ax.set_ylabel(py, fontsize=11)
        ax.set_title(f"Interior: {interior_share:.0%}", fontsize=11)
        ax.grid(False)

        cb = plt.colorbar(mesh, ax=ax)
        cb.set_label(r"$\beta^*$", fontsize=10)

    fig.suptitle(
        r"Optimal Promotion Weight $\beta^*$ — Interior vs Corner Regions",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()

    if type == "engagement":
        filename = "results/interior_regions_engagement.png"
    elif type == "view":
        filename = "results/interior_regions_view.png"

    out_dir = os.path.dirname(os.path.abspath(filename))
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Saved: {filename}")
    return fig

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Running 2D parameter sweeps (this may take a moment)...")
    for type in ["engagement", "view"]:
        results = {(x, y): scan_2d(type, x, y) for x, y in _PAIRS}
        plot_interior_regions(type, results)
        plt.show()
