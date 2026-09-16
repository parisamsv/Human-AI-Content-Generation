# Content Generation: Human Creators, AI Learning, and Platform Design

Replication code for the paper **"Content Generation: Human Creators, AI Learning, and Platform Design"** by S. P. Moosavi, A. Malekian (Rotman School of Management, University of Toronto), and A. Makhdoumi (Fuqua School of Business, Duke University).

This repository reproduces the figures in the paper. It solves a three-stage game between a platform, a human creator, and consumers, under two ways the platform can make money: **view-based** (revenue and pay scale with audience size) and **engagement-based** (revenue and pay scale with the quality consumed). For each set of parameters, the code computes the equilibrium under both designs and draws the comparative-statics figures.

## Repository layout

```
.
├── README.md
├── code/                 # all source files
│   ├── main.py           # entry point: pick a figure, enter parameters, generate it
│   ├── params.py         # baseline parameters and ranges (Table 1)
│   ├── common.py         # mismatch multipliers m_A, m_H
│   ├── solver.py         # solves for the optimal algorithmic weight beta_H*
│   ├── model_view.py     # view-based equilibrium
│   ├── model_engagement.py  # engagement-based equilibrium
│   └── plot_functions.py # all figure-drawing routines
└── figures/              # output folder; generated PDFs are saved here
```

`main.py` writes its output to the `figures/` folder one level up from `code/`, so keep the two folders side by side as shown.

## Requirements

- Python 3.8 or newer
- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/)
- [Matplotlib](https://matplotlib.org/)

Install them with:

```bash
pip install numpy scipy matplotlib
```

## How to run

From the `code/` folder:

```bash
cd code
python main.py
```

The script works in two steps:

1. **Pick a figure.** It prints a numbered menu of the figures and asks for one number. It generates a single figure per run, so to make several figures just run it again for each one.
2. **Enter the parameters.** It then asks for the five model parameters. Press **Enter** on any line to keep the paper's baseline value shown in brackets.

The chosen figure is saved as a PDF in the `figures/` folder.


### Parameter prompt

After you pick a figure, you are asked for:

| Symbol | Meaning                     | Baseline |
|--------|-----------------------------|----------|
| `Q`    | AI baseline quality         | 1.00     |
| `alpha`| AI learning efficiency      | 0.40     |
| `delta`| Algorithmic influence       | 0.50     |
| `t`    | Consumer mismatch cost      | 2.50     |
| `k`    | Algorithmic-neutrality cost | 2.50     |

Before drawing, the code checks that your values satisfy the paper's two key assumptions over the full range being plotted (it checks the hardest point in the range, the largest `Q` and largest `alpha`):

- **Non-saturation:** `Q < t*(1 - delta)`
- **Concavity:** `t > alpha^2 + 1/(1 - delta)`

If a value breaks either assumption, the script prints which one and stops, so no figure is produced with invalid settings. Fix the parameters and run again.

## Figures produced

The menu number is what you type at the first prompt.

| Menu | File                              | Content                                                        |
|------|-----------------------------------|----------------------------------------------------------------|
| 2    | `fig_vb_effort_regions.pdf`       | View-based creator effort `q_H*` as the pay rate `r` varies    |
| 3    | `fig_eb_effort_regions.pdf`       | Engagement-based creator effort `q_H* (= r*)` as `Q` varies    |
| 4    | `fig_cmp_uH_Q_regions.pdf`        | Creator utility vs `Q`, both designs                           |
| 5    | `fig_cmp_uH_alpha_regions.pdf`    | Creator utility vs `alpha`, both designs                       |
| 6    | `fig_preference_map_Q_alpha.pdf`  | Which design each party prefers over the `(Q, alpha)` plane    |
| 7    | `fig_preference_map_delta.pdf`    | Preference maps over `(Q, delta)` and `(alpha, delta)`         |
| 8    | `fig_negative_TE_Q_region.pdf`    | Where total engagement falls in `Q`, over an `(alpha, delta)` grid |
| 9    | `fig_Q_comparative_statics.pdf`   | Six panels vs `Q`: `beta_H*`, `q_H*`, `u_P`, `u_H`, TV, TE     |
| 10   | `fig_alpha_comparative_statics.pdf`| Six panels vs `alpha`: same six outcomes                      |

## Reproducing specific figures

The paper does not use one single `k` for every figure. To match a particular figure exactly, enter the `k` from that figure's caption at the prompt:

- Main comparative-statics and preference maps: `k = 2.5`
- Creator utility vs `Q`: `k = 1.5`
- Creator utility vs `alpha`: `k = 1.0`

All other baseline values are as in the table above.
