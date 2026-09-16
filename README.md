# Content Generation: Human Creators, AI Learning, and Platform Design

Replication code for the paper **"Content Generation: Human Creators, AI Learning, and Platform Design"** by S. P. Moosavi, A. Malekian (Rotman School of Management, University of Toronto), and A. Makhdoumi (Fuqua School of Business, Duke University).

This repository reproduces every figure in the paper. It solves a three-stage game between a platform, a human creator, and consumers, under two ways the platform can make money: **view-based** (revenue and pay scale with audience size) and **engagement-based** (revenue and pay scale with the quality consumed). For each set of parameters, the code computes the equilibrium under both designs and draws the comparative-statics figures.

## Repository layout

```
.
├── README.md
├── code/                 # all source files
│   ├── main.py           # entry point: prompts for parameters, generates figures
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

The script will ask you for the five model parameters. Press **Enter** on any line to keep the paper's baseline value shown in brackets. It then generates all figures and saves them as PDFs in the `figures/` folder.

To write the figures somewhere else, pass a path:

```bash
python main.py /path/to/output_folder
```

### Parameter prompt

When the script starts, you are asked for:

| Symbol | Meaning                     | Baseline |
|--------|-----------------------------|----------|
| `Q`    | AI baseline quality         | 1.00     |
| `alpha`| AI learning efficiency      | 0.40     |
| `delta`| Algorithmic influence       | 0.50     |
| `t`    | Consumer mismatch cost      | 2.50     |
| `k`    | Algorithmic-neutrality cost | 2.50     |

Before running, the code checks that your values satisfy the paper's two key assumptions over the full range being plotted:

- **Non-saturation:** `Q < t*(1 - delta)`
- **Concavity:** `t > alpha^2 + 1/(1 - delta)`

If a value breaks either assumption, the script tells you which one and does not produce a figure with invalid settings.

## Figures produced

| File                              | Content                                                        |
|-----------------------------------|----------------------------------------------------------------|
| `fig_vb_effort_regions.pdf`       | View-based creator effort `q_H*` as the pay rate `r` varies    |
| `fig_eb_effort_regions.pdf`       | Engagement-based creator effort `q_H* (= r*)` as `Q` varies    |
| `fig_cmp_uH_Q_regions.pdf`        | Creator utility vs `Q`, both designs                           |
| `fig_cmp_uH_alpha_regions.pdf`    | Creator utility vs `alpha`, both designs                       |
| `fig_preference_map_Q_alpha.pdf`  | Which design each party prefers over the `(Q, alpha)` plane    |
| `fig_preference_map_delta.pdf`    | Preference maps over `(Q, delta)` and `(alpha, delta)`         |
| `fig_negative_TE_Q_region.pdf`    | Where total engagement falls in `Q`, over an `(alpha, delta)` grid |
| `fig_Q_comparative_statics.pdf`   | Six panels vs `Q`: `beta_H*`, `q_H*`, `u_P`, `u_H`, TV, TE     |
| `fig_alpha_comparative_statics.pdf`| Six panels vs `alpha`: same six outcomes                      |

## Reproducing specific figures

The paper does not use one single `k` for every figure. If you want to match a particular figure exactly, enter the `k` from that figure's caption at the prompt:

- Main comparative-statics and preference maps: `k = 2.5`
- Creator utility vs `Q`: `k = 1.5`
- Creator utility vs `alpha`: `k = 1.0`

All other baseline values are as in the table above.

## What the code does, step by step

For a given set of parameters, each equilibrium is found by backward induction:

1. **Creator's effort.** The creator picks effort from her own first-order condition: `q_H = r` under engagement-based, and `q_H = r/(t*m_H)` under view-based.
2. **Optimal pay rate.** For a fixed algorithmic weight, the platform's best pay rate `r*` is chosen among the interior and boundary candidates in the relevant proposition.
3. **Optimal algorithmic weight.** `solver.py` solves `M(beta_H) = k*(beta_H - 1/2)` for the weight `beta_H*`, then clips it to `[0, 1]`.
4. **Outcomes.** Demands, platform utility, creator utility, total views, and total engagement are computed from the equilibrium values.

## Citation

If you use this code, please cite the paper:

> Moosavi, S. P., Malekian, A., and Makhdoumi, A. "Content Generation: Human Creators, AI Learning, and Platform Design." *Management Science* (forthcoming).

_(Update the volume, year, and DOI once available.)_