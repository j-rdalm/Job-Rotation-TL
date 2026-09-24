"""Statistical rigor, sensitivity, and complexity analysis on top of alb_balancing.py
and job_rotation.py -- kept separate from those two so the core ALB/rotation modeling
files stay focused on the models themselves, not validation/reporting concerns.

Provides the statistical analysis (heuristic variability, bootstrap confidence
intervals), the sensitivity analysis (traffic-light weight schemes) and the
empirical CPU scaling plot reported in the paper. Big-O derivations are documented
in the docstrings of the relevant functions in alb_balancing.py and job_rotation.py.
All analyses use the case-study data.
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import alb_balancing as ab
import job_rotation as jr


def alb_heuristic_variability(operator_counts=(3, 6, 9, 12), n_reps=30, seed=0):
    """Variability of the ALB heuristic. Runs it n_reps times per instance size
    with randomized tie-breaking (see greedy_local_search's rng parameter) and
    reports the resulting distribution, alongside the deterministic MILP result for
    the same instance as a reference point."""
    df = ab.load_tasks()
    rows = []
    for n in operator_counts:
        effs, sis = [], []
        for i in range(n_reps):
            rng = np.random.default_rng(seed + i)
            operators, eff = ab.greedy_local_search(df, n, rng=rng)
            si = ab.smoothness_index(
                (op["tempo_total"] for op in operators.values()),
                max(op["tempo_total"] for op in operators.values()),
            )
            effs.append(eff)
            sis.append(si)

        milp = ab.milp_alb(df, n, time_limit=120)
        rows.append(
            {
                "Operators": n,
                "Efficiency_mean": np.mean(effs),
                "Efficiency_std": np.std(effs, ddof=1),
                "Efficiency_min": np.min(effs),
                "Efficiency_max": np.max(effs),
                "SI_mean": np.mean(sis),
                "SI_std": np.std(sis, ddof=1),
                "Efficiency_milp": milp["efficiency"],
                "MILP_proven_optimal": milp["proven_optimal"],
                "MILP_optimality_gap": milp["optimality_gap"],
            }
        )
    return pd.DataFrame(rows)


def _dispersion(values):
    _, _, gini = jr.lorenz_and_gini(values)
    return float(np.std(values, ddof=1)), float(gini)


def rotation_heuristic_variability(rotation_counts=(2, 3, 5, 8), n_reps=30, seed=0):
    """Same idea as alb_heuristic_variability, for the job-rotation heuristic
    (job_rotation.job_rotation_heuristic's rng parameter). Unlike the ALB heuristic,
    this one shows genuine variability (verified: range spans ~2.2-6.0 across seeds
    at R=3 on the real 9-operator instance), so this is where the distribution
    actually matters for judging heuristic reliability, not just a formality."""
    loops, operators = jr.build_alb_scenario()
    loop_workload = jr.compute_loop_workloads(operators)
    skill_matrix, medical_matrix = jr.build_loop_skill_medical(operators)
    people = skill_matrix.index.tolist()

    rows = []
    for r in rotation_counts:
        ranges, stds, ginis = [], [], []
        baseline = jr.baseline_exposures(loop_workload, people, rotations=r)
        for i in range(n_reps):
            rng = np.random.default_rng(seed + i)
            heur = jr.job_rotation_heuristic(loop_workload, skill_matrix, medical_matrix, rotations=r, rng=rng)
            std, gini = _dispersion(list(heur["exposures"].values()))
            ranges.append(heur["range"])
            stds.append(std)
            ginis.append(gini)

        milp = jr.job_rotation_milp(loop_workload, skill_matrix, medical_matrix, rotations=r, time_limit=120)
        rows.append(
            {
                "R": r,
                "range_mean": np.mean(ranges),
                "range_std": np.std(ranges, ddof=1),
                "range_min": np.min(ranges),
                "range_max": np.max(ranges),
                "std_mean": np.mean(stds),
                "gini_mean": np.mean(ginis),
                "range_milp": milp["range"],
                "MILP_proven_optimal": milp["proven_optimal"],
                "MILP_optimality_gap": milp["optimality_gap"],
            }
        )
    return pd.DataFrame(rows)


def bootstrap_dispersion_reduction(baseline, rotated, n_boot=1000, seed=0):
    """Bootstrap confidence interval for the reduction in std-dev and Gini index
    achieved by rotation. Resamples the N paired (baseline, rotated) exposures with replacement n_boot
    times, recomputing the std-dev and Gini reduction each time, and reports a 95%
    percentile confidence interval -- not just the point estimate."""
    people = list(baseline.keys())
    n = len(people)
    base_vals = np.array([baseline[p] for p in people])
    rot_vals = np.array([rotated[p] for p in people])

    rng = np.random.default_rng(seed)
    std_reductions, gini_reductions = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        std_base, gini_base = _dispersion(base_vals[idx])
        std_rot, gini_rot = _dispersion(rot_vals[idx])
        std_reductions.append(std_base - std_rot)
        gini_reductions.append(gini_base - gini_rot)

    def summarize(values):
        values = np.array(values)
        return {
            "point_estimate": float(values.mean()),
            "ci_low": float(np.percentile(values, 2.5)),
            "ci_high": float(np.percentile(values, 97.5)),
        }

    return {"std_reduction": summarize(std_reductions), "gini_reduction": summarize(gini_reductions)}


DEFAULT_TL_SCHEMES = {
    "baseline (3,2,1)": {"Red": 3, "Yellow": 2, "Green": 1},
    "spread (5,3,1)": {"Red": 5, "Yellow": 3, "Green": 1},
    "compressed (2,1.5,1)": {"Red": 2, "Yellow": 1.5, "Green": 1},
}


def tl_weight_sensitivity(schemes=None, rotations=jr.NUM_ROTATIONS):
    """Sensitivity of the main result (rotation reduces ergonomic-exposure
    dispersion) to the Red/Yellow/Green -> numeric weight mapping. Together with
    job_rotation.run_comparison's sweep over the number of rotation periods R, this
    gives two sensitivity dimensions."""
    schemes = schemes or DEFAULT_TL_SCHEMES
    loops, operators = jr.build_alb_scenario()
    skill_matrix, medical_matrix = jr.build_loop_skill_medical(operators)
    people = skill_matrix.index.tolist()

    rows = []
    for name, risk_map in schemes.items():
        loop_workload = jr.compute_loop_workloads(operators, risk_map=risk_map)
        baseline = jr.baseline_exposures(loop_workload, people, rotations=rotations)
        milp = jr.job_rotation_milp(loop_workload, skill_matrix, medical_matrix, rotations=rotations, time_limit=120)

        std_base, gini_base = _dispersion(list(baseline.values()))
        if milp["status"] == "Optimal" and milp["proven_optimal"]:
            std_rot, gini_rot = _dispersion(list(milp["exposures"].values()))
        else:
            std_rot, gini_rot = None, None

        rows.append(
            {
                "scheme": name,
                "std_baseline": std_base,
                "std_rotated": std_rot,
                "gini_baseline": gini_base,
                "gini_rotated": gini_rot,
                "conclusion_holds": (std_rot is not None) and (std_rot < std_base) and (gini_rot < gini_base),
            }
        )
    return pd.DataFrame(rows)


def empirical_scaling(alb_comparison=None, rotation_comparison=None, out_path="empirical_scaling.png"):
    """Empirical CPU time vs. instance size, complementing the Big-O
    docstrings on greedy_local_search/milp_alb (alb_balancing.py) and
    job_rotation_heuristic/job_rotation_milp (job_rotation.py). Reuses the sweeps
    already produced by both files' run_comparison() -- no new instance sizes."""
    alb_comparison = alb_comparison if alb_comparison is not None else ab.run_comparison()
    rotation_comparison = rotation_comparison if rotation_comparison is not None else jr.run_comparison()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(alb_comparison["Operators"], alb_comparison["CPU_greedy"], "-o", label="Greedy heuristic")
    axes[0].plot(alb_comparison["Operators"], alb_comparison["CPU_milp"], "-s", label="MILP")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Number of operators (stations)")
    axes[0].set_ylabel("CPU time (s, log scale)")
    axes[0].set_title("ALB: CPU vs. instance size")
    axes[0].legend()
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].plot(rotation_comparison["R"], rotation_comparison["CPU_heuristic"], "-o", label="Greedy+local-search heuristic")
    axes[1].plot(rotation_comparison["R"], rotation_comparison["CPU_milp"], "-s", label="MILP")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Number of rotation periods R")
    axes[1].set_ylabel("CPU time (s, log scale)")
    axes[1].set_title("Job rotation: CPU vs. instance size")
    axes[1].legend()
    axes[1].grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return alb_comparison, rotation_comparison


if __name__ == "__main__":
    print("=== ALB heuristic variability ===")
    print(alb_heuristic_variability().to_string(index=False))

    print("\n=== Rotation heuristic variability ===")
    print(rotation_heuristic_variability().to_string(index=False))

    print("\n=== TL weight sensitivity ===")
    print(tl_weight_sensitivity().to_string(index=False))

    loops, operators = jr.build_alb_scenario()
    loop_workload = jr.compute_loop_workloads(operators)
    skill_matrix, medical_matrix = jr.build_loop_skill_medical(operators)
    people = skill_matrix.index.tolist()
    baseline = jr.baseline_exposures(loop_workload, people)
    milp = jr.job_rotation_milp(loop_workload, skill_matrix, medical_matrix)
    print("\n=== Bootstrap CI on dispersion reduction ===")
    print(bootstrap_dispersion_reduction(baseline, milp["exposures"]))

    print("\n=== Empirical scaling ===")
    empirical_scaling()
    print("Saved empirical_scaling.png")
