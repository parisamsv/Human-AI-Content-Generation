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
from model import platform_utility_vec


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
def scan_2d(param_x, param_y, grid_size=100, threshold=0.01):
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
    betas = np.linspace(0.01, 0.99, 200)

    beta_star_grid = np.empty((grid_size, grid_size))

    for i, xv in enumerate(xs):
        for j, yv in enumerate(ys):
            params = dict(BASELINE)
            params[param_x] = xv
            params[param_y] = yv
            utils = platform_utility_vec(betas, **params)
            beta_star_grid[i, j] = betas[int(np.argmax(utils))]

    is_interior = (beta_star_grid > threshold) & (beta_star_grid < 1.0 - threshold)
    interior_share = float(is_interior.mean())
    return beta_star_grid, is_interior, interior_share


# ─────────────────────────────────────────────────────────────────────────────
def plot_interior_regions(results, filename="figures/interior_regions.png"):
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

        # white contour at the interior / corner boundary
        if is_interior.any() and not is_interior.all():
            ax.contour(
                xs, ys, is_interior.T.astype(float),
                levels=[0.5], colors="white", linewidths=1.5,
            )

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

    out_dir = os.path.dirname(os.path.abspath(filename))
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Saved: {filename}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
def report_interior_ranges(results):
    """
    Print a human-readable summary of interior solution regions.

    For each parameter pair, reports
    - the interior share
    - the marginal range of each parameter across all interior grid points
      (i.e. the projection of the interior region onto each axis).

    Parameters
    ----------
    results : dict
        Same structure as the input to plot_interior_regions.
    """
    for px, py in _PAIRS:
        beta_star_grid, is_interior, interior_share = results[(px, py)]

        grid_size = beta_star_grid.shape[0]
        xs = np.linspace(0.01, 0.99, grid_size)
        ys = np.linspace(0.01, 0.99, grid_size)

        print(f"\n{'='*52}")
        print(f"Pair         : ({px}, {py})")
        print(f"Interior share: {interior_share:.1%}")

        if is_interior.any():
            # For param_x: a value xs[i] is "in range" if ANY j gives interior
            x_has_interior = np.any(is_interior, axis=1)   # shape (grid_size,)
            # For param_y: a value ys[j] is "in range" if ANY i gives interior
            y_has_interior = np.any(is_interior, axis=0)   # shape (grid_size,)

            print(f"  {px:8s} interior range: "
                  f"[{xs[x_has_interior].min():.3f}, "
                  f"{xs[x_has_interior].max():.3f}]")
            print(f"  {py:8s} interior range: "
                  f"[{ys[y_has_interior].min():.3f}, "
                  f"{ys[y_has_interior].max():.3f}]")
        else:
            print("  No interior solutions found in this parameter region.")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Running 2D parameter sweeps (this may take a moment)...")
    results = {(x, y): scan_2d(x, y) for x, y in _PAIRS}
    plot_interior_regions(results)
    report_interior_ranges(results)
    plt.show()
