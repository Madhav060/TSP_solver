"""
TSP Experiment Runner (Full Version)
---------------------
This script automates the process of benchmarking the DRL, GA, ACO,
and DRL-ACO-Hybrid solvers on a given TSPLIB problem.

It runs stochastic algorithms multiple times to gather reliable
statistics (Best, Average, Std. Deviation) and compares them
against the deterministic DRL solver.

All results are printed to the console and appended to
'experiment_results.csv'.
"""

import numpy as np
import time
import os
import re
import csv
from typing import List, Optional

# --- Import your solver code ---
# (Assumes these files are in the same directory)
try:
    from tsp_core import City, Tour
    from genetic_algorithm import GeneticAlgorithmSolver
    from ant_colony import AntColonyOptimizer
    from drl_solver import DRLSolver
except ImportError as e:
    print(f"Error: Missing solver files. Make sure all .py files are in the same directory.")
    print(f"Details: {e}")
    exit(1)

# --- Experiment Configuration ---
TSPLIB_FILE = 'att48.tsp'         # The problem file to test (e.g., 'att48.tsp', 'berlin52.tsp')
N_RUNS = 20                       # Number of times to run stochastic algos (20-30 is good)

# Iterations for "pure" solvers
GA_GENERATIONS = 800
ACO_ITERATIONS = 800

# Iterations for "hybrid" solver
HYBRID_ACO_ITERATIONS = 400

# --- TSPLIB File Loader ---

def load_tsplib_problem(filepath: str) -> Optional[List[City]]:
    """
    Parses a .tsp file (EUC_2D format) and returns a list of City objects.
    """
    cities = []
    in_coord_section = False
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line == "NODE_COORD_SECTION":
                    in_coord_section = True
                    continue
                if line == "EOF" or not line:
                    in_coord_section = False
                    continue
                
                if in_coord_section:
                    # Line format: "index x_coord y_coord"
                    parts = re.split(r'\s+', line)
                    if len(parts) == 3:
                        try:
                            index = int(parts[0])
                            x = float(parts[1])
                            y = float(parts[2])
                            cities.append(City(x, y, f"City_{index}"))
                        except ValueError:
                            print(f"Warning: Skipping malformed line: {line}")
                            
    except FileNotFoundError:
        print(f"Error: TSPLIB file not found: {filepath}")
        print("Please download it and place it in the same directory.")
        return None
    
    if not cities:
        print(f"Error: No cities were loaded from {filepath}.")
        print("Check if the file is a valid .tsp file with NODE_COORD_SECTION.")
        return None
        
    return cities

# --- Statistics Helper ---

def process_and_print_results(
    name: str, 
    distances: List[float], 
    times: List[float], 
    optimal_dist: Optional[float],
    # --- New arguments for CSV logging ---
    csv_writer: csv.DictWriter,
    problem_name: str,
    n_runs: int
):
    """
    Calculates stats, prints them to console, AND writes them to a CSV row.
    """
    
    # --- Calculations ---
    best_dist = np.min(distances)
    avg_dist = np.mean(distances)
    std_dist = np.std(distances)
    avg_time = np.mean(times)
    
    gap_str = "---"
    best_gap = "N/A"
    if optimal_dist:
        gap_calc = ((best_dist - optimal_dist) / optimal_dist) * 100
        gap_str = f"{gap_calc:6.2f}%"
        best_gap = gap_calc

    # --- 1. Print to console (as before) ---
    print(f"| {name:<20} | {best_dist:<12.2f} | {avg_dist:<15.2f} | {std_dist:<10.2f} | {avg_time:<15.2f} | {gap_str:<10} |")

    # --- 2. Write to CSV file ---
    row = {
        "Problem": problem_name,
        "N_Runs": n_runs,
        "Algorithm": name,
        "Best_Dist": f"{best_dist:.2f}",
        "Avg_Dist": f"{avg_dist:.2f}",
        "Std_Dev": f"{std_dist:.2f}",
        "Avg_Time_s": f"{avg_time:.2f}",
        "Best_Gap_Percent": f"{best_gap:.2f}" if isinstance(best_gap, float) else best_gap
    }
    csv_writer.writerow(row)

# --- Main Experiment ---

def main():
    print(f"🚀 Starting TSP Solver Experiment")
    print(f"Problem: {TSPLIB_FILE}")
    print(f"Stochastic Runs: {N_RUNS}")
    print(f"GA Generations: {GA_GENERATIONS} | ACO Iterations: {ACO_ITERATIONS} | Hybrid Iterations: {HYBRID_ACO_ITERATIONS}")
    print("="*80)

    # 1. Load the problem
    cities = load_tsplib_problem(TSPLIB_FILE)
    if not cities:
        return
    print(f"✓ Loaded {len(cities)} cities from {TSPLIB_FILE}.")
    
    # Optional: Add known optimal solutions to calculate the "gap"
    optimal_solutions = {
        'berlin52.tsp': 7542.0,
        'att48.tsp': 10628.0,
        'eil76.tsp': 538.0,
        'kroA100.tsp': 21282.0,
        'bays29.tsp': 2020.0
    }
    optimal_dist = optimal_solutions.get(TSPLIB_FILE)

    # 2. Run DRL Solver (Deterministic)
    print("\n[Phase 1] Running DRL (w/ 2-opt) for baseline and seed tour...")
    drl_solver = DRLSolver(cities)
    
    start_drl = time.time()
    # Note: solve_fast() also applies 2-opt by default
    drl_tour = drl_solver.solve_fast(use_2opt=True, verbose=False)
    drl_time = time.time() - start_drl
    drl_distance = drl_tour.get_total_distance()
    
    print(f"✓ DRL complete. Found tour of {drl_distance:.2f} in {drl_time:.2f}s.")
    
    # These lists will store results from all N_RUNS
    ga_results = {'distances': [], 'times': []}
    aco_results = {'distances': [], 'times': []}
    hybrid_results = {'distances': [], 'times': []}

    # 3. Run Stochastic Solvers (N_RUNS times)
    print(f"\n[Phase 2] Running GA, ACO, and Hybrid solvers {N_RUNS} times...")
    for i in range(N_RUNS):
        print(f"--- Run {i+1}/{N_RUNS} ---")
        
        # --- Genetic Algorithm ---
        print("  Running GA...")
        ga_solver = GeneticAlgorithmSolver(
            cities, population_size=100, mutation_rate=0.015
        )
        start_ga = time.time()
        ga_tour = ga_solver.solve(generations=GA_GENERATIONS, verbose=False)
        ga_time = time.time() - start_ga
        ga_results['distances'].append(ga_tour.get_total_distance())
        ga_results['times'].append(ga_time)

        # --- Ant Colony (Pure) ---
        print("  Running ACO (pure)...")
        aco_solver = AntColonyOptimizer(cities, n_ants=30)
        start_aco = time.time()
        aco_tour = aco_solver.solve(iterations=ACO_ITERATIONS, verbose=False, seed_tour=None)
        aco_time = time.time() - start_aco
        aco_results['distances'].append(aco_tour.get_total_distance())
        aco_results['times'].append(aco_time)
        
        # --- Hybrid (DRL + ACO) ---
        print("  Running Hybrid (DRL-Seeded ACO)...")
        hybrid_aco = AntColonyOptimizer(
            cities,
            n_ants=30,
            seed_weight=10.0 # Use the strong seed weight
        )
        # Time *only* the refinement part
        start_hybrid_refine = time.time()
        hybrid_tour = hybrid_aco.solve(
            iterations=HYBRID_ACO_ITERATIONS, 
            verbose=False, 
            seed_tour=drl_tour.clone() # Use the DRL tour as seed
        )
        hybrid_refine_time = time.time() - start_hybrid_refine
        
        # Total hybrid time = DRL inference time + ACO refinement time
        hybrid_total_time = drl_time + hybrid_refine_time
        
        hybrid_results['distances'].append(hybrid_tour.get_total_distance())
        hybrid_results['times'].append(hybrid_total_time)
        
        print(f"  Run {i+1} complete.")

    # 4. Process, Print, and Save Final Results
    print("\n" + "="*80)
    print(f"📊 FINAL RESULTS - {TSPLIB_FILE} ({N_RUNS} Runs)")
    print("="*80)
    
    # --- Define CSV file and headers ---
    RESULTS_FILE = 'experiment_results.csv'
    fieldnames = [
        "Problem", "N_Runs", "Algorithm", "Best_Dist", 
        "Avg_Dist", "Std_Dev", "Avg_Time_s", "Best_Gap_Percent"
    ]
    
    # Check if file exists to decide on writing header
    file_exists = os.path.isfile(RESULTS_FILE)

    # --- Open file in "append" mode ('a') ---
    with open(RESULTS_FILE, 'a', newline='') as f:
        csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            csv_writer.writeheader()  # Write header only if file is new
            
        # --- Print console header ---
        header = f"| {'Algorithm':<20} | {'Best Dist':<12} | {'Avg. Dist':<15} | {'Std. Dev':<10} | {'Avg. Time (s)':<15} | {'Gap (Best)':<10} |"
        print(header)
        print("-" * len(header))

        # --- Process DRL ---
        process_and_print_results(
            "DRL (w/ 2-opt)", 
            [drl_distance], 
            [drl_time], 
            optimal_dist,
            csv_writer, TSPLIB_FILE, N_RUNS # Pass new args
        )
        
        # --- Process GA ---
        process_and_print_results(
            "GA", 
            ga_results['distances'], 
            ga_results['times'],
            optimal_dist,
            csv_writer, TSPLIB_FILE, N_RUNS # Pass new args
        )
        
        # --- Process ACO ---
        process_and_print_results(
            "ACO (pure)", 
            aco_results['distances'], 
            aco_results['times'],
            optimal_dist,
            csv_writer, TSPLIB_FILE, N_RUNS # Pass new args
        )
        
        # --- Process Hybrid ---
        process_and_print_results(
            "Hybrid (DRL+ACO)", 
            hybrid_results['distances'], 
            hybrid_results['times'],
            optimal_dist,
            csv_writer, TSPLIB_FILE, N_RUNS # Pass new args
        )
        
        print("-" * len(header))
    
    print(f"\n✓ Results for this run have been appended to: {RESULTS_FILE}")


if __name__ == "__main__":
    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    main()