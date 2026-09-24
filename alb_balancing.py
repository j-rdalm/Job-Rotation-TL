"""Assembly line balancing (Section 3.1): greedy heuristic with local improvement vs. MILP.

Both methods are compared across operator counts 3-12 on the case-study data
(Dados_Bal.xlsx), producing the workstation input data and the heuristic vs. MILP
comparison reported in Section 4.1 of the paper.
"""
import re
import time

import pandas as pd
import pulp

_CBC_LOWER_BOUND_RE = re.compile(r"^Lower bound:\s*([\d.eE+-]+)", re.MULTILINE)


def _cbc_optimality_gap(log_text, objective_value):
    """CBC always writes a final 'Lower bound: X' line, even when the solve was cut
    off by the time limit. Compute the relative gap ourselves from that bound and
    the full-precision objective PuLP reports, rather than trusting CBC's own
    'Gap:' line (which is pre-rounded to 2 decimals and reads '0.00' for any gap
    under 0.5%, even when not formally proven optimal). Returns 0.0 once proven
    optimal, None if the log doesn't contain a lower bound (e.g. solve failed)."""
    match = _CBC_LOWER_BOUND_RE.search(log_text)
    if not match or objective_value is None:
        return None
    lower_bound = float(match.group(1))
    if lower_bound == 0:
        return 0.0 if abs(objective_value) < 1e-9 else float("inf")
    return abs(objective_value - lower_bound) / abs(lower_bound)


DATA_FILE = "Dados_Bal.xlsx"
SHEET = "856_CPT1"
RESULTS_FILE = "resultados.xlsx"


def load_tasks(data_file=DATA_FILE, sheet=SHEET):
    df = pd.read_excel(data_file, sheet_name=sheet)
    df["Posto"] = df["Posto"].astype(str).str.extract(r"(\d+)").astype(int)
    return df


def workstation_summary(df):
    """Per-workstation task count and cycle time (workstation input data table)."""
    return (
        df.groupby("Posto", sort=False)
        .agg(**{"No. of tasks": ("Tarefa", "count"), "Time (s)": ("Tempo", "sum")})
        .reindex(df["Posto"].drop_duplicates())
    )


def smoothness_index(station_times, c_max):
    return sum((c_max - t) ** 2 for t in station_times) ** 0.5


def greedy_local_search(df, num_operators, rng=None):
    """Construction is a single O(n) pass over the n tasks in fixed MTM sequence
    order (not a free choice, so not randomized even when rng is given). Local
    search repeatedly rebalances the busiest operator against a neighbor; each
    iteration is O(m) to find the busiest operator (m operators), so the whole
    loop is O(k*m) for k improving moves found (empirically small and roughly
    constant across the operator counts tested -- see run_comparison's CPU_greedy
    column). With rng set, ties for "busiest operator" and the order neighbors are
    tried are broken randomly instead of by insertion/tempo order, so repeated
    calls can be used to measure the heuristic's variability (analysis.py)."""
    total_time = df["Tempo"].sum()
    target_time = total_time / num_operators + 2  # 2s slack margin

    operators = {
        i + 1: {"tarefas": [], "tempo_total": 0.0, "tempos_postos": {}}
        for i in range(num_operators)
    }

    current = 1
    for _, task in df.iterrows():
        if operators[current]["tempo_total"] + task["Tempo"] > target_time and current < num_operators:
            current += 1
        op = operators[current]
        op["tarefas"].append(task.to_dict())
        op["tempo_total"] += task["Tempo"]
        posto = task["Posto"]
        op["tempos_postos"][posto] = op["tempos_postos"].get(posto, 0) + task["Tempo"]

    def efficiency():
        c_max = max(op["tempo_total"] for op in operators.values())
        return total_time / (num_operators * c_max) * 100

    eff = efficiency()
    improved = True
    while improved:
        improved = False
        max_time = max(op["tempo_total"] for op in operators.values())
        tied_busiest = [p for p in operators if operators[p]["tempo_total"] == max_time]
        busiest = rng.choice(tied_busiest) if rng is not None else tied_busiest[0]
        neighbors = [p for p in (busiest - 1, busiest + 1) if 1 <= p <= num_operators]
        if rng is not None:
            rng.shuffle(neighbors)
        else:
            neighbors.sort(key=lambda p: operators[p]["tempo_total"])
        for neighbor in neighbors:
            if not operators[busiest]["tarefas"]:
                continue
            move_from_front = neighbor == busiest - 1
            task = operators[busiest]["tarefas"].pop(0 if move_from_front else -1)
            _move_task(operators, busiest, neighbor, task, insert_front=not move_from_front)

            if efficiency() > eff:
                eff = efficiency()
                improved = True
                break
            _move_task(operators, neighbor, busiest, task, insert_front=move_from_front)

    return operators, eff


def _move_task(operators, src, dst, task, insert_front=False):
    posto = task["Posto"]
    src_op, dst_op = operators[src], operators[dst]
    if src is not dst:
        src_op["tempo_total"] -= task["Tempo"]
        src_op["tempos_postos"][posto] -= task["Tempo"]
        if src_op["tempos_postos"][posto] <= 1e-9:
            del src_op["tempos_postos"][posto]
        if task in src_op["tarefas"]:
            src_op["tarefas"].remove(task)
    if insert_front:
        dst_op["tarefas"].insert(0, task)
    else:
        dst_op["tarefas"].append(task)
    dst_op["tempo_total"] += task["Tempo"]
    dst_op["tempos_postos"][posto] = dst_op["tempos_postos"].get(posto, 0) + task["Tempo"]


def milp_alb(df, num_operators, time_limit=120, verbose=False):
    """SALBP-1 (simple assembly line balancing, type 1) MILP. NP-hard in general
    (Garey & Johnson-class partitioning problem); this formulation uses O(n*m)
    binary assignment variables x_ij (n tasks, m stations) plus m continuous
    station-time variables, with O(n + m + n*m) constraints (one assignment
    constraint per task, two per station, one precedence inequality per
    consecutive task pair). Solve time is worst-case exponential in n and m;
    see run_comparison's CPU/optimality_gap columns for the empirical picture."""
    task_times = dict(zip(df["Num tarefa"], df["Tempo"]))  # "Tarefa" labels are not unique
    tasks = list(task_times.keys())
    precedence = list(zip(tasks, tasks[1:]))  # sequential MTM order = precedence chain
    stations = list(range(1, num_operators + 1))

    prob = pulp.LpProblem("ALB_min_makespan", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", (tasks, stations), cat=pulp.LpBinary)
    ct = pulp.LpVariable.dicts("CT", stations, lowBound=0, cat=pulp.LpContinuous)
    c_max = pulp.LpVariable("Cmax", lowBound=0, cat=pulp.LpContinuous)

    prob += c_max
    for i in tasks:
        prob += pulp.lpSum(x[i][j] for j in stations) == 1
    for j in stations:
        prob += ct[j] == pulp.lpSum(task_times[i] * x[i][j] for i in tasks)
        prob += ct[j] <= c_max
    for i, k in precedence:
        prob += pulp.lpSum(j * x[i][j] for j in stations) <= pulp.lpSum(j * x[k][j] for j in stations)

    log_path = f"/tmp/cbc_alb_{num_operators}.log"
    prob.solve(pulp.PULP_CBC_CMD(msg=1, timeLimit=time_limit, logPath=log_path))
    with open(log_path) as f:
        log = f.read()
    proven_optimal = "Optimal solution found" in log or "Search completed" in log

    ct_values = {j: pulp.value(ct[j]) for j in stations}
    total_time = sum(task_times.values())
    c_max_val = pulp.value(c_max)
    return {
        "status": pulp.LpStatus[prob.status],
        "proven_optimal": proven_optimal,
        "optimality_gap": 0.0 if proven_optimal else _cbc_optimality_gap(log, c_max_val),
        "Cmax": c_max_val,
        "CT": ct_values,
        "efficiency": total_time / (num_operators * c_max_val) * 100,
        "SI": smoothness_index(ct_values.values(), c_max_val),
    }


def run_comparison(operator_counts=range(3, 13), data_file=DATA_FILE, sheet=SHEET, results_file=RESULTS_FILE):
    df = load_tasks(data_file, sheet)

    detail_rows = []
    comparison_rows = []
    for n in operator_counts:
        # Wall-clock time (perf_counter): the CBC solver runs as an external
        # subprocess, so wall-clock time is needed to capture the full solve time.
        start = time.perf_counter()
        operators, eff_greedy = greedy_local_search(df, n)
        cpu_greedy = time.perf_counter() - start
        si_greedy = smoothness_index(
            (op["tempo_total"] for op in operators.values()), max(op["tempo_total"] for op in operators.values())
        )

        start = time.perf_counter()
        milp_result = milp_alb(df, n, time_limit=120)
        cpu_milp = time.perf_counter() - start

        comparison_rows.append(
            {
                "Operators": n,
                "CPU_greedy": cpu_greedy,
                "Efficiency_greedy": eff_greedy,
                "SI_greedy": si_greedy,
                "CPU_milp": cpu_milp,
                "Efficiency_milp": milp_result["efficiency"],
                "SI_milp": milp_result["SI"],
                "MILP_proven_optimal": milp_result["proven_optimal"],
                "MILP_optimality_gap": milp_result["optimality_gap"],
            }
        )

        for op, dados in operators.items():
            for posto, tempo in dados["tempos_postos"].items():
                detail_rows.append(
                    {
                        "Cenario": n,
                        "Operador": f"Operador {op}",
                        "Tempo Total": dados["tempo_total"],
                        "Posto": posto,
                        "Tempo no Posto": tempo,
                        "Número de Tarefas": len(dados["tarefas"]),
                        "Eficiência": eff_greedy,
                    }
                )

    with pd.ExcelWriter(results_file) as writer:
        pd.DataFrame(detail_rows).to_excel(writer, sheet_name="Sheet1", index=False)
        pd.DataFrame(comparison_rows).to_excel(writer, sheet_name="ALB_comparison", index=False)

    return pd.DataFrame(comparison_rows)


if __name__ == "__main__":
    df = load_tasks()
    print("Workstation summary:")
    print(workstation_summary(df))
    print(f"\nTotal tasks: {len(df)}, total cycle time: {df['Tempo'].sum():.2f}s\n")

    comparison = run_comparison()
    print("ALB comparison (heuristic vs. MILP):")
    print(comparison.to_string(index=False))
