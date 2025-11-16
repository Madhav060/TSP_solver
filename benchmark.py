"""
benchmark.py

Runs a formal benchmark comparing all available TSP solvers (DRL, GA, ACO, Hybrid)
on a single instance (berlin52.tsp) over multiple runs.
"""

import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- Import Your Project's Solvers ---
try:
    from data_generator import load_tsp_file
except ImportError:
    print("Error: Could not import 'load_tsp_file' from 'data_generator.py'")
    print("Please ensure that function exists and the file is in your path.")
    exit()

from tsp_core import City
from drl_solver import DRLSolver
from genetic_algorithm import GeneticAlgorithmSolver
from ant_colony import AntColonyOptimizer
from hybrid_solver import HybridSolverACO


# --- Benchmark Configuration ---
TSP_FILE = 'berlin52.tsp'
OPTIMAL_DISTANCE = 7542.0  # Known optimal for berlin52
N_RUNS = 10                # Number of times to run each stochastic algorithm
OUTPUT_DIR = 'benchmarks'

# Solver parameters for a 52-city problem (rigorous settings)
GA_GENERATIONS = 1000
ACO_ITERATIONS = 1000
HYBRID_ITERATIONS = 500 # DRL seed + 500 ACO iterations


def run_benchmark():
    """
    Main benchmark function.
    """
    # --- 1. Setup ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("="*70)
    print(f"🚀 RUNNING TSP SOLVER BENCHMARK")
    print("="*70)
    print(f"Problem:         {TSP_FILE} (Optimal: {OPTIMAL_DISTANCE})")
    print(f"Stochastic Runs: {N_RUNS} per solver")
    print(f"Solvers:         DRL, GA, ACO, Hybrid")
    print("="*70)

    # --- 2. Load Cities ---
    print(f"\nLoading cities from {TSP_FILE}...")
    cities = load_tsp_file(TSP_FILE)
    if not cities:
        print(f"Error: Could not load cities from {TSP_FILE}.")
        return
    print(f"✓ Loaded {len(cities)} cities.")

    # --- 3. Pre-load DRL Model ---
    # This is a one-time cost. We load it here so the DRL and Hybrid
    # solvers can use the already-loaded model instance.
    print("\nPre-loading DRL model (from tsp_checkpoints)...")
    try:
        drl_solver_instance = DRLSolver(cities)
        if not drl_solver_instance.model:
            print("⚠️  DRL model not loaded (running in heuristic mode).")
        else:
            print("✓ DRL model loaded successfully.")
    except Exception as e:
        print(f"✗ CRITICAL ERROR: Could not load DRL model: {e}")
        print("   Cannot run DRL or Hybrid benchmarks. Exiting.")
        return

    # --- 4. Define Solver Configurations ---
    solver_configs = [
        {
            'name': 'DRL (Model + 2-Opt)',
            'solver_class': None, # We use the pre-loaded instance
            'instance': drl_solver_instance,
            'solve_method': 'solve_fast',
            'solve_params': {'use_2opt': True, 'verbose': False}
        },
        {
            'name': f'Genetic Algorithm ({GA_GENERATIONS} gen)',
            'solver_class': GeneticAlgorithmSolver,
            'instance': None, # We create a new one each run
            'init_params': {
                'cities': cities,
                'population_size': 100,
                'mutation_rate': 0.015,
                'tournament_size': 5
            },
            'solve_method': 'solve',
            'solve_params': {'generations': GA_GENERATIONS, 'verbose': False}
        },
        {
            'name': f'Ant Colony ({ACO_ITERATIONS} iter)',
            'solver_class': AntColonyOptimizer,
            'instance': None, # We create a new one each run
            'init_params': {'cities': cities, 'n_ants': 30},
            'solve_method': 'solve',
            'solve_params': {'iterations': ACO_ITERATIONS, 'verbose': False, 'seed_tour': None}
        },
        {
            'name': f'Hybrid (DRL + ACO {HYBRID_ITERATIONS} iter)',
            'solver_class': HybridSolverACO,
            'instance': None, # We create a new one each run
            'init_params': {
                'cities': cities,
                'drl_solver': drl_solver_instance # Pass the pre-loaded solver
            },
            'solve_method': 'solve',
            'solve_params': {'quick_iterations': HYBRID_ITERATIONS, 'verbose': False, 'use_2opt': True}
        }
    ]

    # --- 5. Run Benchmark Loop ---
    all_results = []
    
    for config in solver_configs:
        solver_name = config['name']
        print(f"\n--- Benchmarking: {solver_name} ---")
        
        run_distances = []
        run_times = []
        
        # Use tqdm for a progress bar over N_RUNS
        for i in tqdm(range(N_RUNS), desc=f"Running {solver_name}"):
            # Instantiate solver
            if config['instance'] is not None:
                solver = config['instance'] # Use pre-loaded DRL
            else:
                solver = config['solver_class'](**config['init_params'])

            # Run and time the solve method
            start_time = time.time()
            solve_method_to_call = getattr(solver, config['solve_method'])
            tour = solve_method_to_call(**config['solve_params'])
            end_time = time.time()
            
            run_distances.append(tour.get_total_distance())
            run_times.append(end_time - start_time)
            
            # For deterministic solvers, no need to run N times
            if solver_name == 'DRL (Model + 2-Opt)':
                print("\n   (DRL is deterministic, only running once)")
                run_distances = run_distances * N_RUNS # Fill list with same result
                run_times = run_times * N_RUNS
                break 

        # Calculate stats
        all_results.append({
            'Solver': solver_name,
            'Avg. Distance': np.mean(run_distances),
            'Best Distance': np.min(run_distances),
            'Std. Dev.': np.std(run_distances),
            'Avg. Time (s)': np.mean(run_times)
        })

    # --- 6. Process and Save Results ---
    print("\n" + "="*70)
    print("📊 Benchmark Complete: Processing Results")
    print("="*70)
    
    df = pd.DataFrame(all_results)
    
    # Add comparison columns
    df['Optimal'] = OPTIMAL_DISTANCE
    df['Avg. Error (%)'] = ((df['Avg. Distance'] - OPTIMAL_DISTANCE) / OPTIMAL_DISTANCE) * 100.0
    df['Best Error (%)'] = ((df['Best Distance'] - OPTIMAL_DISTANCE) / OPTIMAL_DISTANCE) * 100.0
    
    # Sort by best average distance
    df = df.sort_values(by='Avg. Distance')
    
    # Re-order columns for clarity
    df = df[[
        'Solver', 
        'Avg. Distance', 
        'Best Distance', 
        'Avg. Time (s)', 
        'Avg. Error (%)', 
        'Best Error (%)',
        'Std. Dev.'
    ]]
    
    # --- 7. Save Files ---
    report_md_path = os.path.join(OUTPUT_DIR, 'solver_comparison_report.md')
    results_csv_path = os.path.join(OUTPUT_DIR, 'solver_comparison_results.csv')
    results_json_path = os.path.join(OUTPUT_DIR, 'solver_comparison_results.json')
    
    # Save raw data
    df.to_csv(results_csv_path, index=False, float_format='%.3f')
    df.to_json(results_json_path, orient='records', indent=4)
    
    # Format for Markdown report
    df_md = df.copy()
    df_md['Avg. Distance'] = df_md['Avg. Distance'].map('{:,.2f}'.format)
    df_md['Best Distance'] = df_md['Best Distance'].map('{:,.2f}'.format)
    df_md['Avg. Time (s)'] = df_md['Avg. Time (s)'].map('{:,.3f}'.format)
    df_md['Avg. Error (%)'] = df_md['Avg. Error (%)'].map('{:+.2f}%'.format)
    df_md['Best Error (%)'] = df_md['Best Error (%)'].map('{:+.2f}%'.format)
    df_md['Std. Dev.'] = df_md['Std. Dev.'].map('{:,.2f}'.format)
    
    best_solver = df.iloc[0] # Get the top row after sorting

    summary = f"""# 🏆 TSP Solver Benchmark: berlin52.tsp

**Test Run:** {time.strftime("%Y-%m-%d %H:%M:%S")}
**Problem:** `{TSP_FILE}` (52 cities)
**Known Optimal Distance:** `{OPTIMAL_DISTANCE}`
**Runs per Solver:** `{N_RUNS}`

## 🥇 Best Performing Solver (by Avg. Distance)

**{best_solver['Solver']}**
* **Average Distance:** {best_solver['Avg. Distance']:.2f} ({best_solver['Avg. Error (%)']:+.2f}% from optimal)
* **Best Distance:** {best_solver['Best Distance']:.2f} ({best_solver['Best Error (%)']:+.2f}% from optimal)
* **Average Time:** {best_solver['Avg. Time (s)']:.3f}s

## 📊 Full Comparison Table
*(Sorted by `Avg. Distance`)*

{df_md.to_markdown(index=False)}
"""
    
    with open(report_md_path, 'w') as f:
        f.write(summary)

    # --- 8. Print Final Summary to Console ---
    print("\n" + df_md.to_string(index=False))
    
    print("\n" + "="*70)
    print("✓ Benchmark Proof Saved!")
    print(f"Human-readable report: {report_md_path}")
    print(f"CSV data:              {results_csv_path}")
    print("="*70)


if __name__ == "__main__":
    run_benchmark()