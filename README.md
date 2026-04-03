# Content Generation: Human Creators, AI Learning, and Platform Design

A LaTeX research paper analyzing platform competition between human content creators and AI generators, with a focus on algorithmic promotion, compensation design, and equilibrium behavior.

---

## Project Structure

```
.
├── main.tex               # Main paper file
├── references.bib         # BibTeX bibliography
└── README.md              # This file
```

---

## Paper Overview

The paper develops a two-stage game-theoretic model of a platform where:
- A **human creator** chooses effort level (content quality)
- An **AI generator** learns from the human's output via `q_A = α q_h + Q`
- The **platform** sets an algorithmic promotion weight `β_h` and a compensation ratio `r`

The equilibrium is solved by backward induction across three stages:
1. Human chooses optimal effort given `r` and `β_h`
2. Platform chooses optimal compensation rate `r*`
3. Platform chooses optimal promotion weight `β_h*`

---

## Key Model Parameters

| Parameter | Description |
|-----------|-------------|
| `α ∈ [0,1]` | AI learning efficiency from human content |
| `Q ≥ 0` | AI baseline quality without human input |
| `t > 0` | Consumer mismatch cost parameter |
| `δ ∈ (0,1)` | Algorithm's influence on mismatch costs |
| `β_h ∈ [0,1]` | Platform's promotion weight for the human |
| `r ∈ [0,1]` | Compensation ratio paid to human per view |
| `u_0 > 0` | Consumer outside option |

---

## Mismatch Cost Structure

Consumer utilities are:
```
u_A = q_A - t·x·(β_h·δ + (1−δ))        [AI]
u_h = q_h - t·(1−x)·(1 − β_h·δ)        [Human]
```

This gives mismatch weights:
```
m_A(β_h) = 1 − δ(1−β_h)    ∈ [1−δ, 1]
m_H(β_h) = 1 − β_h·δ        ∈ [1−δ, 1]
```

Both weights are bounded away from zero by the floor `1−δ > 0`, ensuring human effort and all equilibrium quantities remain finite at both corners `β_h = 0` and `β_h = 1`.

---

## Main Results

### Proposition 1 — Optimal Human Effort
```
q_h* = r / (t · m_H)
q_A* = α·r / (t · m_H) + Q
```

### Proposition 2 — Optimal Compensation Rate
```
r* = (1/2) · [1 + α·m_H/m_A + t·m_H·u_0]
```
Increases in `α`: platform pays more when AI learns more efficiently from human effort.

### Proposition 3 — Optimal Promotion when Q ≥ u_0
The platform's value function `V(β_h)` is globally strictly convex. Corner solution:
```
β_h* = 0    if u_p(0) ≥ u_p(1)
β_h* = 1    if u_p(1) > u_p(0)
```
Convexity is established by expressing `V(β_h) = (1/4)·W(β_h)² + (Q−u_0)/t·y_A − u_0/t·y_H`
where `W`, `y_A = 1/m_A`, and `y_H = 1/m_H` are all strictly convex, and invoking the
participation constraint `W > 2u_0`.

### Proposition 4 — Optimal Promotion when Q < u_0
Define the survival threshold `β̃` where `q_A*(β̃) = u_0`. An interior optimum `β_h* ∈ (β̃, 1)` exists if and only if the **friction dominance condition** holds:
```
(q̄_A − u_0)·δ / t  >  Ω̄
```
where `q̄_A = α·r*(1)/[t(1−δ)] + Q` and `Ω̄` collects all positive terms in `du_p/dβ_h` at `β_h = 1`. If the condition fails, `β_h* = 1`.

---

## Proof Strategy

| Proposition | Method |
|-------------|--------|
| 1 | First-order condition + second-order check |
| 2 | FOC w.r.t. `r`, multiply through by `t²m_H²` |
| 3 | Express `V(β_h)` as closed-form quadratic in `r`, substitute `r*`, show `V''> 0` using convexity of `1/m_A` and `1/m_H` plus participation constraint |
| 4 | Sign derivative at `β̃` (positive by definition) and at `β_h = 1` (negative iff friction dominance), apply IVT |

---

## Compilation

The paper uses the `informs3_NoOR` document class. Compile with:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or with `latexmk`:

```bash
latexmk -pdf main.tex
```

---

## Dependencies

### LaTeX Packages
- `amsmath`, `mathtools`, `mathrsfs` — mathematics
- `amsthm`, `thmtools` — theorem environments
- `natbib` — author-year citations (`plainnat` style)
- `hyperref` — colored hyperlinks
- `enumitem` — custom list formatting
- `xcolor` (dvipsnames) — extended color names
- `todonotes` — inline todo notes (set to inline by default)
- `soul` — highlighting (`\hl{}`)
- `lmodern`, `fontenc` (T1) — font encoding

### Bibliography
References are managed in `references.bib`. Key cited works include:
- Ai, Simchi-Levi & Xu (2026) — GenAI vs. Human Creators
- Esmaeili et al. (2025) — Strategic Human Content Creation
- Taitler & Ben-Porat (2024, 2025) — Braess's Paradox, Selective Response
- Taitler et al. (2025) — Data Sharing with GenAI Competitor
- Yao et al. (2023, 2024a, 2024b) — Recommender system incentives
- Yu et al. (2025) — Group strategies in recommendation platforms

---

## Notes

- The document class option `nonblindrev` shows author information; switch to `blindrev` for anonymous submission.
- Line spacing is set to `\OneAndAHalfSpacedXI`; change to `\DoubleSpacedXI` if the journal requires double spacing.
- Todo notes are forced inline via a `regexpatch` on `\@todo`.
- Highlight color is set to yellow via `\sethlcolor{yellow}`.
