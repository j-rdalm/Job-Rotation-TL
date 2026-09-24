# Sequential Optimization of Assembly Line Balancing and Job Rotation for Ergonomic Workload Equity

Code accompanying the paper:

> J. R. Almeida, A. Moura, A. R. Xambre and J. L. Oliveira. *Sequential Optimization of Assembly Line Balancing and Job Rotation for Ergonomic Workload Equity.* (under review)

The paper proposes a two-stage decision-support framework:

1. **Assembly line balancing (ALB).** Tasks are assigned to operators, which gives the group of workstations (a "loop") each operator handles. A greedy heuristic with local improvement is compared with a MILP model (SALBP-1).
2. **Job rotation.** For each rotation period, operators are assigned to loops so that the spread of cumulative ergonomic exposure across operators is as small as possible. Exposure is scored with a traffic-light index (green / yellow / red) derived from EAWS, SWR and BSHA assessments. Operator skills and medical restrictions are hard constraints. A MILP model is compared with a greedy heuristic based on bipartite matching and local search.

Everything runs on the real case-study data from a thermotechnology company (124 tasks, 11 workstations, 9 operators).

## Repository structure

| File | Contents | Paper |
|---|---|---|
| `alb_balancing.py` | Loads task data, greedy + local-search heuristic, ALB MILP, smoothness index, heuristic-vs-MILP comparison for 3 to 12 operators | Section 3.1; Section 4.1 (workstation input data table, ALB heuristic-vs-MILP table) |
| `job_rotation.py` | Builds loops from the ALB solution, loop workloads, skill/medical eligibility, rotation MILP and heuristic, Lorenz curves and Gini index | Section 3.2; Sections 4.2 to 4.4 (skills and restriction matrices, loop eligibility, risk before/after rotation, heuristic vs MILP over R) |
| `analysis.py` | Heuristic variability over repeated runs, bootstrap confidence intervals, traffic-light weight sensitivity, empirical CPU scaling plot | Sections 4.4 to 4.6 (empirical scaling figure, sensitivity table, statistical analysis) |
| `Dados_Bal.xlsx` | Task list with MTM times per workstation (sheet `856_CPT1`) | Input data |
| `Dados_ergonomia.xlsx` | Traffic-light risk level per workstation (`TL`), operator skills (`Skill`), medical restrictions (`medical_restrictions`) | Input data |

## Installation

Python 3.11 is recommended.

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

PuLP ships with the open-source CBC solver, so no other solver is needed.

## Usage

Run every command from the repository root, since the scripts read the `.xlsx` files with relative paths.

```bash
python alb_balancing.py   # ALB input summary + heuristic vs MILP for 3-12 operators (writes resultados.xlsx)
python job_rotation.py    # job rotation for 9 operators and R = 3 periods; Lorenz curves / Gini figures
python analysis.py        # heuristic variability, bootstrap CI, weight sensitivity, empirical_scaling.png
```

Figures are saved as `.png` files in the repository root.

### Runtime

Each MILP solve has a 120-second time limit. Several instances reach it: ALB with 9 or more operators, and job rotation with R ≥ 5 periods. Running all three scripts therefore takes roughly 15 to 30 minutes. The heuristics finish in milliseconds.

CPU times are wall-clock (`time.perf_counter`) because CBC runs as a separate process. Absolute times depend on your machine, but the relative gap between heuristic and MILP should match the paper.

## Citation

If you use this code, please cite the paper above. Full reference to be added after publication.

## License

Code released under the [MIT License](LICENSE).
