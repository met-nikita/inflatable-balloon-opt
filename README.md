# Inflatable Balloon Method (метод «надувного шарика», МНШ / MIB)

A Python implementation of **G.B. Bronfeld's "Inflatable Balloon Method"** for
the *optimal distribution of a production plan* — Kantorovich's assortment /
machine-loading problem — and its nonlinear, multi-criteria generalisation.

> 🇷🇺 Версия на русском: [`README.ru.md`](README.ru.md)

The algorithm is reconstructed from:

* **Бронфельд Г.Б. — «Алгоритм решения задачи оптимального распределения плана
  производства»** (1977) — the detailed algorithm. (https://sirius-2.narod.ru/tw.htm)
* **Бронфельд Г.Б. — «Метод "надувного шарика" …»**, *Вестник ТвГТУ*, 2019
  — history and the nonlinear multi-criteria generalisation. (https://sirius-2.narod.ru/tw.htm)

---

## The problem

There are `m` enterprises (aggregates / machines) `I = {1..m}` that must make
`n` products `J = {1..n}` in a fixed assortment. Enterprise `i` produces product
`j` at rate `a[i][j]` (output per unit time; `0` ⇒ cannot make it) and has a
working-time budget `z[i]`. One complete *assortment set* needs `b0[j]` units of
product `j`. We choose the times `x[i][j]` to maximise the number of complete
sets `mu`:

```
maximize    mu
subject to  sum_i a[i][j] * x[i][j]  >=  b0[j] * mu     for every product j     (2)
            sum_j x[i][j]            <=  z[i]           for every enterprise i  (3)
            x[i][j] >= 0                                                        (4)
```

### The "balloon" idea

`mu` acts like air pressure. Inflating `mu` expands the demand surface
`b0[j]·mu` until the balloon presses against the resource walls (constraint 3).
The largest `mu` for which a feasible allocation still exists is the optimum
`mu*`. The **generalised** method keeps bulging *individual* criteria into any
leftover free space afterwards (see below).

---

## How the method works

### Outer loop — inflation (`balloon/solver.py`)

Iterative search for `mu*` (eq. 9, 10, 26–31): start from the paper's boundary
estimate `mu_gr/10`, grow while feasible (`alpha = 0.1·mu/10^k`, eq. 30) and
step back by dichotomy on overshoot (`alpha = -|Δmu|/2`, eq. 31) until the step
is below `eps` (eq. 9). The iterate is kept inside a known
feasible/infeasible bracket so convergence to `mu*` is guaranteed.

### Inner oracle — estimate-class fixing + branch & bound (`balloon/feasibility.py`)

For a fixed demand vector this decides whether a feasible allocation exists and
builds one. Estimates `s[i][j] = a[i][j]·d[i][j]` (eq. 11) measure how much of
product `j` enterprise `i` could still make. They are classified and *fixed*
("закрепление оценок"):

| Class | Definition (paper) | Meaning | Action |
|-------|--------------------|---------|--------|
| **A** | unique positive estimate in a **column** (def. 1) | only one enterprise can make this product | forced assignment (eq. 15) |
| **B** | unique positive estimate in a **row** (def. 2) | enterprise can make only one product | forced assignment (eq. 16) |
| **C** | strict column-maximum (def. 3) | best enterprise for the product — *maximum-element* heuristic | branch (eq. 22–25) |
| **D** | global maximum estimate (def. 4) | greedy maximum element | branch (eq. 22–25) |

A/B are deterministic, never-wrong reductions. When only C/D choices remain the
solver **branches** on the most-constrained product, trying its suppliers in
maximum-element order, with branch-and-bound pruning via the necessary bound `P`
(eq. 17) and the per-product bound `Q` (eq. 18).

Fixing `(k, l)` commits enterprise `k`'s time to product `l`: if `k` can cover
the remaining demand it spends exactly enough and keeps the rest; otherwise it
pours all its time in and the shortfall is left to others.

### Generalised / nonlinear balloon (`balloon/nonlinear.py`)

After the uniform optimum `mu*`, leftover resources are used by inflating each
criterion along its own monotone non-decreasing *desired-level* function
`b[j] = f_j(b0[j], mu, t)` (eq. 8, Fig. 1–3 of the 2019 paper) — round-robin,
re-checking full feasibility so already-granted criteria are never violated.
The outcome is a point on the Pareto frontier; ceilings / order express the
multi-criteria preferences.

---

## Accuracy: exact where it counts

The class-fixing method is **combinatorial and faithful to the paper** — and the
paper notes its efficiency rises for **sparsely-filled matrices**. Measured
against the exact LP optimum over random instances:

| matrix density | exact-match rate | mean gap | max gap |
|---------------:|-----------------:|---------:|--------:|
| 0.2            | 100 %            | 0.000    | 0.000   |
| 0.3            | 100 %            | 0.000    | 0.000   |
| 0.4            | 97 %             | 0.0005   | 0.034   |
| 0.5            | 93 %             | 0.0005   | 0.016   |
| 1.0 (dense)    | 46 %             | 0.0043   | 0.032   |

So on **sparse** problems — the method's intended domain, and the shape of real
production plans like the paper's 20×16 example — it reaches the exact optimum.
On dense matrices it is a fast heuristic and an always-valid **lower bound**
(it never over-reports `mu*`). For a guaranteed optimum on *any* instance, plug
in the exact oracle:

```python
from balloon import BalloonSolver, LPOracle
res = BalloonSolver(p, oracle=LPOracle(p)).solve()   # exact on any density
```

---

## Usage

```python
from balloon import Problem, BalloonSolver, solve_generalized

p = Problem.from_lists(
    a=[[2.0, 1.0, 0.0],
       [1.0, 2.0, 1.0],
       [0.0, 1.0, 3.0]],
    z=[1.0, 1.0, 1.0],
    b0=[1.0, 1.0, 1.0],
)

res = BalloonSolver(p, eps=1e-5).solve()
print(res.mu)                      # optimal number of assortment sets
print(res.allocation.x)            # the time-allocation matrix x[i][j]
print(res.allocation.slack_time)   # unused time per enterprise

gen = solve_generalized(p)         # uniform mu* + per-criterion inflation
print(gen.demand)                  # inflated demand levels b[j]
```

### Command line

```bash
python run.py examples/sample_problem.json            # solve
python run.py examples/sample_problem.json --trace    # show the inflation trace
python run.py examples/sample_problem.json --exact    # exact LP oracle
python run.py examples/sample_problem.json --generalized
```

### The paper's worked example

`examples/nelidovo_1977.py` rebuilds the 20-aggregate × 16-product example from
Tables 1 & 2 and reproduces the paper's reported `mu = 1.08`:

```bash
python -m examples.nelidovo_1977
```

```
Balloon Method  mu* = 1.0809   (paper reports 1.08; ...)
Exact LP        mu* = 1.0809
  match: True
```

---

## Project layout

```
balloon/
  problem.py      Problem & Allocation models, constraint checking
  feasibility.py  class-fixing (A/B/C/D) + branch-&-bound feasibility oracle
  solver.py       outer balloon-inflation search for mu*  (eq. 9,10,26-31)
  nonlinear.py    generalised per-criterion (nonlinear) inflation  (2019 paper)
  exact.py        scipy-LP reference: exact optimum, LPOracle (ground truth)
examples/
  nelidovo_1977.py   the paper's 20x16 example  (mu = 1.08)
  sample_problem.json
mib_tests/         test suite (run with: python -m mib_tests.run_all)
run.py             command-line driver
```

## Requirements

* Python ≥ 3.9, **numpy** (the method itself).
* **scipy** — only for the exact LP reference (`balloon.exact`) and the tests.

## Tests

```bash
python -m mib_tests.run_all
```

The suite checks the inner oracle against an independent LP feasibility oracle,
confirms the inflation reaches the LP optimum on sparse problems (and is a valid
lower bound otherwise), reproduces the paper's `mu = 1.08`, and verifies the
generalised inflation stays feasible and reaches the resource frontier.
