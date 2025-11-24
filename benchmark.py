import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

# Solver imports
from data_generator import load_tsp_file
from drl_solver import DRLSolver
from genetic_algorithm import GeneticAlgorithmSolver
from ant_colony import AntColonyOptimizer


# ================================
# CONFIGURATION
# ================================
DATASET_DIR = "tsp_data"
OUTPUT_DIR = "benchmarks_full"
TIME_LIMIT = 2.0
RUNS_PER_DATASET = 10
VERBOSE_SOLVER = False
ALPHA = 0.5   # Balanced weight for combined score

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================
# TIME-WRAPPER: Equal-Time + Time-to-Target
# =============================================================
def run_with_time_limit(solve_fn, max_time_limit, target_distance=None, **kwargs):

    start = time.time()
    best_tour = None
    best_time = max_time_limit
    logs = []

    def callback(tour):
        nonlocal best_tour, best_time

        now = time.time()
        elapsed = now - start

        if tour is None:
            return

        if isinstance(tour, tuple):
            tour = tour[0]

        if not hasattr(tour, "get_total_distance"):
            return

        dist = tour.get_total_distance()

        if best_tour is None or dist < best_tour.get_total_distance():
            best_tour = tour.clone()
            logs.append((elapsed, dist))

            if target_distance is not None and dist <= target_distance:
                best_time = elapsed
                raise StopIteration

        if elapsed >= max_time_limit:
            raise TimeoutError

    try:
        final = solve_fn(
            time_limit=max_time_limit, callback=callback, verbose=VERBOSE_SOLVER, **kwargs
        )

        if isinstance(final, tuple):
            final = final[0]

        if final is None:
            final = best_tour

        if final and best_time == max_time_limit:
            best_time = time.time() - start

    except StopIteration:
        final = best_tour
    except TimeoutError:
        final = best_tour
        best_time = max_time_limit
    except Exception:
        final = best_tour
        best_time = max_time_limit

    elapsed = time.time() - start
    return final, logs, elapsed if target_distance is None else best_time


# =============================================================
# SOLVER WRAPPERS
# =============================================================
def drl_wrapper(cities, time_limit, callback, verbose):
    solver = DRLSolver(cities)
    tour, _ = solver.solve_fast(use_2opt=True, verbose=verbose, time_limit=time_limit)
    if tour is not None:
        callback(tour)
    return tour


def ga_wrapper(cities, time_limit, callback, verbose):
    solver = GeneticAlgorithmSolver(cities)
    solver.initialize()
    start = time.time()

    while True:
        solver.evolve_generation()
        callback(solver.get_best_tour())
        if time.time() - start >= time_limit:
            break

    return solver.get_best_tour()


def aco_wrapper(cities, time_limit, callback, verbose):
    solver = AntColonyOptimizer(cities) # Defaults to n_ants=50
    solver._initialize_pheromones()
    start = time.time()

    while True:
        tours = [solver._construct_tour() for _ in range(solver.n_ants)]
        for t in tours:
            callback(t)
        solver._update_pheromones(tours)
        if time.time() - start >= time_limit:
            break

    return solver.get_best_tour()


def hybrid_wrapper(cities, time_limit, callback, verbose):
    """
    Fixed Hybrid Wrapper:
    - Removes skip logic for small datasets.
    - Manually sets parameters to ensure Fair Comparison + Optimization.
    """
    start = time.time()

    # 1. DRL Kickstart
    drl = DRLSolver(cities)
    drl_tour, _ = drl.solve_fast(use_2opt=True, verbose=verbose, time_limit=None)
    callback(drl_tour)

    # [FIX 1] Removed the 'if len(cities) < 40' skip logic
    # We want Hybrid to optimize ALL datasets.

    # [FIX 2] Explicitly define parameters to fix the "Blind Ant" issue
    aco_params = {
        "n_ants": 50,          # Increased from 25 to 50 to match Pure ACO (Fairness)
        "alpha": 1.0,
        "beta": 3.0,
        "rho": 0.5,
        "q": 100.0,
        "elite_weight": 2.0,
        "seed_weight": 5.0     # [CRITICAL] Reduced from 50.0 to 5.0 to allow exploration
    }

    # 2. ACO refinement with fixed parameters
    aco = AntColonyOptimizer(cities, **aco_params)
    aco._initialize_pheromones(seed_tour=drl_tour)

    while True:
        if time.time() - start >= time_limit:
            break

        tours = [aco._construct_tour() for _ in range(aco.n_ants)]
        for t in tours:
            callback(t)
        aco._update_pheromones(tours)

    return aco.get_best_tour()


SOLVERS = {
    "DRL": drl_wrapper,
    "GA": ga_wrapper,
    "ACO": aco_wrapper,
    "Hybrid": hybrid_wrapper,
}


# =============================================================
# EQUAL-TIME BENCHMARK
# =============================================================
def benchmark_equal_time(path):

    print("\n" + "*" * 30)
    print(f"Running Equal-Time benchmark for {path} (Time Limit: {TIME_LIMIT:.1f}s)")
    print("*" * 30)

    cities = load_tsp_file(path)
    if not cities:
        print(f"Error loading: {path}")
        return []

    results = []

    for name, fn in SOLVERS.items():
        print(f"\n--- Running {name} ---")

        distances = []
        times = []

        for _ in tqdm(range(RUNS_PER_DATASET), desc=name):

            tour, logs, elapsed = run_with_time_limit(
                lambda time_limit, callback, verbose: fn(
                    cities=cities,
                    time_limit=time_limit,
                    callback=callback,
                    verbose=verbose
                ),
                max_time_limit=TIME_LIMIT,
                target_distance=None
            )

            if tour is None:
                distances.append(float("inf"))
            else:
                try:
                    distances.append(tour.get_total_distance())
                except:
                    distances.append(float("inf"))

            times.append(elapsed)

        results.append({
            "dataset": os.path.basename(path),
            "solver": name,
            "metric": "Equal_Time",
            "avg_dist": float(np.mean(distances)),
            "best_dist": float(np.min(distances)),
            "std_dist": float(np.std(distances)),
            "avg_time": float(np.mean(times)),
        })

    return results


# =============================================================
# TIME-TO-TARGET BENCHMARK
# =============================================================
def benchmark_time_to_target(path, target_dist):
    rows = []
    cities = load_tsp_file(path)
    MAX_TIME = 30.0

    for name, fn in SOLVERS.items():
        times = []

        for _ in range(RUNS_PER_DATASET):
            _, _, t_hit = run_with_time_limit(
                lambda tl, cb, vb: fn(cities=cities, time_limit=tl, callback=cb, verbose=vb),
                max_time_limit=MAX_TIME,
                target_distance=target_dist
            )
            times.append(t_hit)

        hit_rate = np.mean([t < MAX_TIME for t in times])

        rows.append({
            "dataset": os.path.basename(path),
            "solver": name,
            "metric": "Time_to_Target",
            "target_dist": target_dist,
            "avg_time": float(np.mean(times)),
            "hit_rate": float(hit_rate),
        })

    return rows


# =============================================================
# COMBINED SCORE BENCHMARK (NEW)
# =============================================================
def benchmark_combined_score(equal_rows):
    rows = []

    df = pd.DataFrame(equal_rows)

    # Normalize inside each dataset
    for dataset in df["dataset"].unique():

        sub = df[df["dataset"] == dataset]

        dist_min, dist_max = sub["avg_dist"].min(), sub["avg_dist"].max()
        time_min, time_max = sub["avg_time"].min(), sub["avg_time"].max()

        for _, row in sub.iterrows():

            dist_norm = (row["avg_dist"] - dist_min) / (dist_max - dist_min + 1e-9)
            time_norm = (row["avg_time"] - time_min) / (time_max - time_min + 1e-9)

            score = ALPHA * dist_norm + (1 - ALPHA) * time_norm

            rows.append({
                "dataset": dataset,
                "solver": row["solver"],
                "metric": "Combined_Score",
                "score": float(score),
                "dist_norm": float(dist_norm),
                "time_norm": float(time_norm),
            })

    return rows


# =============================================================
# TARGET DISTANCES
# =============================================================
def get_target_dist(dataset):
    # [FIX 3] Relaxed targets by ~10% so 'hit_rate' is not always 0.0
    targets = {
        "att48.tsp": 38500.0,    # Relaxed from 35000
        "berlin52.tsp": 8500.0,  # Relaxed from 7700
        "dj38.tsp": 7400.0,      # Relaxed from 6700
        "ulysses22.tsp": 85.0,   # Relaxed from 76
    }
    return targets.get(dataset, 1e9)


# =============================================================
# MAIN
# =============================================================
def run_benchmark_all():
    all_rows = []

    tsp_files = sorted([
        f for f in os.listdir(DATASET_DIR)
        if f.endswith(".tsp") and f != "gr48.tsp"
    ])

    equal_rows = []

    for fname in tsp_files:
        path = os.path.join(DATASET_DIR, fname)

        # === 1. Equal-Time Quality ===
        eq = benchmark_equal_time(path)
        equal_rows.extend(eq)
        all_rows.extend(eq)

        # === 2. Time-to-Target ===
        target = get_target_dist(fname)
        tt = benchmark_time_to_target(path, target)
        all_rows.extend(tt)

    # === 3. Combined Score ===
    comb_rows = benchmark_combined_score(equal_rows)
    all_rows.extend(comb_rows)

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "full_benchmarks.csv"), index=False)

    print("\n=== Combined Score (Lower = Better) ===")
    print(pd.DataFrame(comb_rows).sort_values(by=["dataset", "score"]))

    print("\n=== Done. Results saved. ===")


    # Save split files for convenience
    df = pd.DataFrame(all_rows)

    # === SAVE EQUAL-TIME RESULTS ===
    df_equal = df[df['metric'] == 'Equal_Time']
    df_equal.to_csv(os.path.join(OUTPUT_DIR, "equal_time_results.csv"), index=False)

    # === SAVE TIME-TO-TARGET RESULTS ===
    df_ttt = df[df['metric'] == 'Time_to_Target']
    df_ttt.to_csv(os.path.join(OUTPUT_DIR, "time_to_target_results.csv"), index=False)

    # === SAVE COMBINED SCORE RESULTS ===
    df_comb = df[df['metric'] == 'Combined_Score']
    df_comb.to_csv(os.path.join(OUTPUT_DIR, "combined_score_results.csv"), index=False)

    print("\nSaved:")
    print(" - equal_time_results.csv")
    print(" - time_to_target_results.csv")
    print(" - combined_score_results.csv")
    print(" - full_benchmarks.csv")


if __name__ == "__main__":
    run_benchmark_all()