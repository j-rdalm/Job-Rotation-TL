"""Ergonomic job rotation (Section 3.2): loop construction, risk scoring, rotation MILP, fairness.

Input data: Dados_ergonomia.xlsx provides the traffic-light risk level per
workstation (`TL`) and, for the 9 operators, the skill matrix (`Skill`) and the
medical restriction matrix (`medical_restrictions`). Both matrices are binary
(1 = qualified / medically cleared, 0 = not qualified / restricted) and are
enforced as hard eligibility constraints.

"Loops": operators do not rotate between the 11 individual workstations, but
between ALB-derived groups of workstations ("loops") -- whatever grouping the
greedy/local-search heuristic in alb_balancing.py produces for a chosen operator
count N. N must match the number of real operators (here 9, from
Dados_ergonomia.xlsx), since the rotation MILP assigns real people to loops.
"""
import numpy as np
import pandas as pd
import pulp
from matplotlib import pyplot as plt

import alb_balancing

ERGONOMIA_FILE = "Dados_ergonomia.xlsx"
NUM_OPERATORS = 9  # matches Dados_ergonomia.xlsx's Skill/medical_restrictions row count
NUM_ROTATIONS = 3  # three 2h30 rotation periods (per case study description)


def build_alb_scenario(num_operators=NUM_OPERATORS):
    """Run the ALB heuristic for num_operators and return per-operator station groups."""
    df = alb_balancing.load_tasks()
    operators, _ = alb_balancing.greedy_local_search(df, num_operators)
    return {op: sorted(dados["tempos_postos"].keys()) for op, dados in operators.items()}, operators


def compute_loop_workloads(operators, ergonomia_file=ERGONOMIA_FILE, risk_map=None):
    """Time-weighted ergonomic score per loop. risk_map maps traffic-light levels to
    numeric weights (default {"Red": 3, "Yellow": 2, "Green": 1}); analysis.py varies
    it as a sensitivity check.

    Each workstation's time is normalized by the total time of the whole scenario
    (summed over all operators), so scores are on a common scale across loops."""
    tl = pd.read_excel(ergonomia_file, sheet_name="TL")
    risk_map = risk_map or {"Red": 3, "Yellow": 2, "Green": 1}
    tl_score = dict(zip(tl["Posto"], tl["TL"].map(risk_map)))

    scenario_total_time = sum(dados["tempo_total"] for dados in operators.values())
    loop_workload = {}
    for op, dados in operators.items():
        loop_workload[op] = sum(
            (tempo / scenario_total_time) * tl_score[posto] for posto, tempo in dados["tempos_postos"].items()
        )
    return loop_workload


def build_loop_skill_medical(operators, ergonomia_file=ERGONOMIA_FILE):
    """Aggregate per-workstation Skill/medical_restrictions (Dados_ergonomia.xlsx) to
    per-loop eligibility. Both sheets use 1 = qualified / cleared and
    0 = not qualified / restricted. An operator is eligible for a loop only if
    eligible (1) at every workstation in it.
    """
    skill = pd.read_excel(ergonomia_file, sheet_name="Skill").set_index("Posto")
    medical = pd.read_excel(ergonomia_file, sheet_name="medical_restrictions").set_index("Posto")
    skill.columns = skill.columns.astype(int)  # column headers are a str/int mix in the source file
    medical.columns = medical.columns.astype(int)

    loops = {op: dados for op, dados in operators.items()}
    people = skill.index.tolist()

    skill_matrix = pd.DataFrame(index=people, columns=loops.keys(), dtype=int)
    medical_matrix = pd.DataFrame(index=people, columns=loops.keys(), dtype=int)
    for person in people:
        for loop_id, dados in loops.items():
            stations = list(dados["tempos_postos"].keys())
            skill_matrix.loc[person, loop_id] = int(all(skill.loc[person, s] == 1 for s in stations))
            medical_matrix.loc[person, loop_id] = int(all(medical.loc[person, s] == 1 for s in stations))
    return skill_matrix, medical_matrix


def job_rotation_milp(loop_workload, skill_matrix, medical_matrix, rotations=NUM_ROTATIONS, time_limit=120):
    """Minimizes E_max - E_min (the spread of cumulative ergonomic exposure across
    operators), not just E_max: std-dev and Gini -- the fairness metrics actually
    reported downstream -- are both spread measures, so this directly optimizes
    what gets reported instead of relying on min-max as an unstated proxy for it.

    NP-hard (a fair multi-period assignment problem, generalizing bipartite
    matching). O(P^2*R) binary assignment variables x_psr (P operators/loops, R
    rotations) plus O(P) continuous exposure variables, with O(P*R) one-loop/
    one-operator-per-rotation constraints, O(P^2) eligibility constraints, and O(P)
    exposure-definition constraints. Solve time is worst-case exponential in P and
    R -- see run_comparison's CPU_milp/optimality_gap columns for the empirical
    picture, which shows this in practice for R >= 5 on the real 9-operator case."""
    people = skill_matrix.index.tolist()
    loops = list(loop_workload.keys())
    assert len(people) == len(loops), (
        f"MILP requires one loop per operator per rotation ({len(people)} people vs {len(loops)} loops)"
    )

    prob = pulp.LpProblem("JobRotation", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", (people, loops, range(rotations)), cat=pulp.LpBinary)
    exposure = pulp.LpVariable.dicts("E", people, lowBound=0)
    e_max = pulp.LpVariable("E_max", lowBound=0)
    e_min = pulp.LpVariable("E_min", lowBound=0)

    prob += e_max - e_min

    for p in people:
        for r in range(rotations):
            prob += pulp.lpSum(x[p][s][r] for s in loops) == 1
    for s in loops:
        for r in range(rotations):
            prob += pulp.lpSum(x[p][s][r] for p in people) == 1
    for p in people:
        for s in loops:
            if skill_matrix.loc[p, s] == 0 or medical_matrix.loc[p, s] == 0:
                for r in range(rotations):
                    prob += x[p][s][r] == 0
    for p in people:
        prob += exposure[p] == pulp.lpSum(
            loop_workload[s] * x[p][s][r] for s in loops for r in range(rotations)
        )
        prob += exposure[p] <= e_max
        prob += exposure[p] >= e_min

    log_path = f"/tmp/cbc_rotation_{len(people)}p_{rotations}r.log"
    prob.solve(pulp.PULP_CBC_CMD(msg=1, timeLimit=time_limit, logPath=log_path))
    with open(log_path) as f:
        log = f.read()
    proven_optimal = "Optimal solution found" in log or "Search completed" in log
    range_val = pulp.value(e_max) - pulp.value(e_min)

    assignments = {
        r: {p: s for p in people for s in loops if pulp.value(x[p][s][r]) > 0.5} for r in range(rotations)
    }
    exposures = {p: pulp.value(exposure[p]) for p in people}
    return {
        "status": pulp.LpStatus[prob.status],
        "proven_optimal": proven_optimal,
        "optimality_gap": 0.0 if proven_optimal else alb_balancing._cbc_optimality_gap(log, range_val),
        "assignments": assignments,
        "exposures": exposures,
        "E_max": pulp.value(e_max),
        "E_min": pulp.value(e_min),
        "range": range_val,
    }


def _eligible(skill_matrix, medical_matrix, p, s):
    return skill_matrix.loc[p, s] == 1 and medical_matrix.loc[p, s] == 1


def _assign_period(loop_workload, skill_matrix, medical_matrix, priority_order, rng=None):
    """Kuhn's algorithm (augmenting-path bipartite matching): each operator, in
    priority order, greedily claims its most preferred (highest-w_s) eligible loop,
    displacing a lower-priority operator to its next-best alternative if needed.
    Guarantees a perfect matching whenever one exists in the eligibility graph
    (already proven to exist by job_rotation_milp finding a feasible solution) --
    unlike a one-shot greedy pass, which can dead-end on infeasible leftovers.
    O(P*E) per call (P operators, E = eligible operator-loop pairs), since each of
    the P augmenting-path searches visits at most E edges once.

    With rng set, loops tied on workload are shuffled before sorting (stable sort
    keeps rng's order among ties) so repeated calls explore different matchings
    among equally-attractive options instead of always preferring the same one."""
    loop_order = list(loop_workload.keys())
    if rng is not None:
        rng.shuffle(loop_order)
    preferences = {
        p: sorted(
            (s for s in loop_order if _eligible(skill_matrix, medical_matrix, p, s)),
            key=lambda s: loop_workload[s],
            reverse=True,
        )
        for p in priority_order
    }
    loop_to_operator = {}

    def try_assign(p, visited):
        for s in preferences[p]:
            if s in visited:
                continue
            visited.add(s)
            if s not in loop_to_operator or try_assign(loop_to_operator[s], visited):
                loop_to_operator[s] = p
                return True
        return False

    for p in priority_order:
        if not try_assign(p, set()):
            raise RuntimeError(f"No feasible per-period matching for operator {p}: eligibility graph has no perfect matching")

    return {p: s for s, p in loop_to_operator.items()}


def _greedy_construct(loop_workload, skill_matrix, medical_matrix, rotations, rng=None):
    people = skill_matrix.index.tolist()
    cumulative = {p: 0.0 for p in people}
    assignments = {}

    for r in range(rotations):
        order = list(people)
        if rng is not None:
            rng.shuffle(order)  # randomizes ties among equal-cumulative operators
        order.sort(key=lambda p: cumulative[p])
        assignments[r] = _assign_period(loop_workload, skill_matrix, medical_matrix, order, rng=rng)
        for p, s in assignments[r].items():
            cumulative[p] += loop_workload[s]

    return assignments, cumulative


def _local_search_swap(assignments, cumulative, loop_workload, skill_matrix, medical_matrix, rotations, max_iters=500):
    """Repeatedly scans all operator pairs (worst-first) and all rotation periods for
    a feasible loop swap that reduces E_max - E_min, applying the first improving move
    found. Scanning all pairs -- not just the single global worst/best -- matters here:
    the global worst and best are often not cross-eligible for each other's loops in any
    shared period, so restricting to that one pair stalls immediately even when other
    improving swaps exist."""
    people = list(cumulative.keys())
    for _ in range(max_iters):
        pairs = sorted(
            ((a, b) for i, a in enumerate(people) for b in people[i + 1 :]),
            key=lambda ab: abs(cumulative[ab[0]] - cumulative[ab[1]]),
            reverse=True,
        )
        improved = False
        for a, b in pairs:
            current_range = max(cumulative.values()) - min(cumulative.values())
            for r in range(rotations):
                s_a, s_b = assignments[r][a], assignments[r][b]
                if s_a == s_b:
                    continue
                if not _eligible(skill_matrix, medical_matrix, a, s_b):
                    continue
                if not _eligible(skill_matrix, medical_matrix, b, s_a):
                    continue

                new_cumulative = dict(cumulative)
                new_cumulative[a] = cumulative[a] - loop_workload[s_a] + loop_workload[s_b]
                new_cumulative[b] = cumulative[b] - loop_workload[s_b] + loop_workload[s_a]
                new_range = max(new_cumulative.values()) - min(new_cumulative.values())
                if new_range < current_range:
                    assignments[r][a], assignments[r][b] = s_b, s_a
                    cumulative.update(new_cumulative)
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return assignments, cumulative


def job_rotation_heuristic(loop_workload, skill_matrix, medical_matrix, rotations=NUM_ROTATIONS, rng=None):
    """Greedy construction + local-search swap, mirroring alb_balancing.greedy_local_search's
    'sequential construction, then repeatedly fix the worst case' structure. Optimizes the
    same E_max - E_min range as job_rotation_milp, so the two are directly comparable.

    Construction is O(R*P*E) (R rotations, each an O(P*E) _assign_period call). Local
    search is O(k*P^2*R) for k improving-swap iterations found (P^2 pairs scanned per
    iteration, R periods checked per pair); k is empirically small and bounded by
    max_iters. With rng set, both stages randomize their tie-breaking (see
    _greedy_construct, _assign_period), so repeated calls measure variability."""
    assignments, cumulative = _greedy_construct(loop_workload, skill_matrix, medical_matrix, rotations, rng=rng)
    assignments, cumulative = _local_search_swap(
        assignments, cumulative, loop_workload, skill_matrix, medical_matrix, rotations
    )
    e_max, e_min = max(cumulative.values()), min(cumulative.values())
    return {
        "assignments": assignments,
        "exposures": cumulative,
        "E_max": e_max,
        "E_min": e_min,
        "range": e_max - e_min,
    }


def baseline_exposures(loop_workload, people, rotations=NUM_ROTATIONS):
    """No-rotation scenario: each operator stays at the loop matching their index
    for every rotation period."""
    loops = list(loop_workload.keys())
    return {p: rotations * loop_workload[loops[i % len(loops)]] for i, p in enumerate(people)}


def lorenz_and_gini(values):
    values_sorted = np.sort(np.asarray(values, dtype=float))
    n = len(values_sorted)
    cum_share = np.cumsum(values_sorted) / values_sorted.sum()
    cum_pop = np.arange(1, n + 1) / n
    gini = 1 - 2 * np.trapz(cum_share, dx=1 / n)
    return cum_pop, cum_share, gini


def fairness_analysis(baseline, rotated, out_path="lorenz_curve_gini_effect.png"):
    base_vals = list(baseline.values())
    rot_vals = list(rotated.values())

    cum_pop_base, cum_share_base, gini_base = lorenz_and_gini(base_vals)
    cum_pop_rot, cum_share_rot, gini_rot = lorenz_and_gini(rot_vals)

    plt.figure(figsize=(7, 7))
    plt.plot(np.insert(cum_pop_rot, 0, 0), np.insert(cum_share_rot, 0, 0), "-o", label=f"Post-rotation (G={gini_rot:.3f})")
    plt.plot(np.insert(cum_pop_base, 0, 0), np.insert(cum_share_base, 0, 0), "-s", label=f"Without rotation (G={gini_base:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Equality line")
    plt.xlabel("Cumulative share of operators")
    plt.ylabel("Cumulative share of ergonomic risk")
    plt.title("Lorenz curves of ergonomic risk distribution")
    plt.legend()
    plt.grid(True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "std_baseline": float(np.std(base_vals, ddof=1)),
        "std_rotated": float(np.std(rot_vals, ddof=1)),
        "gini_baseline": float(gini_base),
        "gini_rotated": float(gini_rot),
    }


def run_comparison(rotation_counts=range(2, 9)):
    """Job rotation's analogue of alb_balancing's Table-3 comparison: heuristic vs
    MILP, swept over the number of rotation periods R using the real case-study
    data (no synthetic instances)."""
    import time

    loops, operators = build_alb_scenario()
    loop_workload = compute_loop_workloads(operators)
    skill_matrix, medical_matrix = build_loop_skill_medical(operators)
    people = skill_matrix.index.tolist()

    rows = []
    for r in rotation_counts:
        # Wall-clock time (perf_counter): the CBC solver runs as an external subprocess.
        start = time.perf_counter()
        heur = job_rotation_heuristic(loop_workload, skill_matrix, medical_matrix, rotations=r)
        cpu_heur = time.perf_counter() - start

        start = time.perf_counter()
        milp = job_rotation_milp(loop_workload, skill_matrix, medical_matrix, rotations=r)
        cpu_milp = time.perf_counter() - start

        baseline = baseline_exposures(loop_workload, people, rotations=r)
        fair_heur = fairness_analysis(baseline, heur["exposures"], out_path=f"/tmp/lorenz_heur_R{r}.png")
        fair_milp = (
            fairness_analysis(baseline, milp["exposures"], out_path=f"/tmp/lorenz_milp_R{r}.png")
            if milp["status"] == "Optimal" and milp["proven_optimal"]
            else None
        )

        rows.append(
            {
                "R": r,
                "CPU_heuristic": cpu_heur,
                "range_heuristic": heur["range"],
                "std_heuristic": fair_heur["std_rotated"],
                "gini_heuristic": fair_heur["gini_rotated"],
                "CPU_milp": cpu_milp,
                "MILP_status": milp["status"],
                "MILP_proven_optimal": milp["proven_optimal"],
                "MILP_optimality_gap": milp["optimality_gap"],
                "range_milp": milp["range"],
                "std_milp": fair_milp["std_rotated"] if fair_milp else None,
                "gini_milp": fair_milp["gini_rotated"] if fair_milp else None,
            }
        )

    return pd.DataFrame(rows)


def run():
    loops, operators = build_alb_scenario()
    loop_workload = compute_loop_workloads(operators)
    skill_matrix, medical_matrix = build_loop_skill_medical(operators)
    people = skill_matrix.index.tolist()
    baseline = baseline_exposures(loop_workload, people)

    heuristic = job_rotation_heuristic(loop_workload, skill_matrix, medical_matrix)
    milp = job_rotation_milp(loop_workload, skill_matrix, medical_matrix)

    print("Loop workloads (w_s):", loop_workload)
    print("\nHeuristic: range =", heuristic["range"])
    print("Heuristic exposures:", heuristic["exposures"])
    if milp["status"] == "Optimal":
        print("\nMILP status:", milp["status"], "range =", milp["range"])
        print("MILP exposures:", milp["exposures"])
    else:
        print("\nMILP did not solve to optimality (status=", milp["status"], ")")

    fairness_heur = fairness_analysis(baseline, heuristic["exposures"], out_path="lorenz_curve_gini_effect_heuristic.png")
    print("\nBaseline exposures:", baseline)
    print("Fairness (heuristic):", fairness_heur)
    fairness_milp = None
    if milp["status"] == "Optimal":
        fairness_milp = fairness_analysis(baseline, milp["exposures"], out_path="lorenz_curve_gini_effect.png")
        print("Fairness (MILP):", fairness_milp)

    return loop_workload, heuristic, milp, baseline, fairness_heur, fairness_milp


if __name__ == "__main__":
    run()
