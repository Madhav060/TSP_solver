"""
TSP Solver - Main Application
Comprehensive demonstration of all solving methods.
"""

import argparse
import time
import sys
from typing import List

from tsp_core import City, Tour
from genetic_algorithm import GeneticAlgorithmSolver
from ant_colony import AntColonyOptimizer
from drl_solver import DRLSolver, HybridSolver
from visualization import (
    TSPVisualizer, 
    generate_random_cities, 
    generate_circle_cities
)


def benchmark_all_solvers(cities: List[City], visualize: bool = True):
    """
    Benchmark all solvers on the same problem and compare results.
    
    Args:
        cities: List of cities to solve
        visualize: Whether to create visualizations
    """
    print(f"\n{'='*70}")
    print(f"TSP SOLVER BENCHMARK")
    print(f"{'='*70}")
    print(f"Problem Size: {len(cities)} cities")
    print(f"{'='*70}\n")
    
    results = {}
    
    # ==========================================
    # 1. DRL Fast Solver (Heuristic)
    # ==========================================
    print("\n" + "="*70)
    print("1. DRL FAST SOLVER (Nearest Neighbor + 2-opt)")
    print("="*70)
    
    drl_solver = DRLSolver(cities)
    start_time = time.time()
    drl_tour = drl_solver.solve_fast(use_2opt=True, verbose=True)
    drl_time = time.time() - start_time
    
    results['DRL Fast'] = {
        'tour': drl_tour,
        'distance': drl_tour.get_total_distance(),
        'time': drl_time
    }
    
    # ==========================================
    # 2. Genetic Algorithm
    # ==========================================
    print("\n" + "="*70)
    print("2. GENETIC ALGORITHM SOLVER")
    print("="*70)
    
    ga_solver = GeneticAlgorithmSolver(
        cities=cities,
        population_size=100,
        mutation_rate=0.015,
        tournament_size=5,
        elitism=True,
        elite_size=2
    )
    
    start_time = time.time()
    ga_tour = ga_solver.solve(generations=500, verbose=True)
    ga_time = time.time() - start_time
    
    results['Genetic Algorithm'] = {
        'tour': ga_tour,
        'distance': ga_tour.get_total_distance(),
        'time': ga_time,
        'history': ga_solver.best_distance_history
    }
    
    # ==========================================
    # 3. Ant Colony Optimization
    # ==========================================
    print("\n" + "="*70)
    print("3. ANT COLONY OPTIMIZATION")
    print("="*70)
    
    aco_solver = AntColonyOptimizer(
        cities=cities,
        n_ants=30,
        n_iterations=300,
        alpha=1.0,
        beta=3.0,
        rho=0.5
    )
    
    start_time = time.time()
    aco_tour = aco_solver.solve(verbose=True)
    aco_time = time.time() - start_time
    
    results['Ant Colony'] = {
        'tour': aco_tour,
        'distance': aco_tour.get_total_distance(),
        'time': aco_time,
        'history': aco_solver.best_distance_history
    }
    
    # ==========================================
    # 4. Hybrid Solver (DRL + GA)
    # ==========================================
    print("\n" + "="*70)
    print("4. HYBRID SOLVER (DRL + GA)")
    print("="*70)
    
    hybrid_solver = HybridSolver(
        cities=cities,
        drl_solver=drl_solver,
        ga_params={
            'population_size': 100,
            'mutation_rate': 0.015,
            'tournament_size': 5,
            'elitism': True,
            'elite_size': 5
        }
    )
    
    start_time = time.time()
    hybrid_tour = hybrid_solver.solve(quick_generations=300, verbose=True)
    hybrid_time = time.time() - start_time
    
    results['Hybrid (DRL+GA)'] = {
        'tour': hybrid_tour,
        'distance': hybrid_tour.get_total_distance(),
        'time': hybrid_time
    }
    
    # ==========================================
    # Results Summary
    # ==========================================
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)
    print(f"{'Method':<25} {'Distance':<15} {'Time (s)':<15} {'Quality':<10}")
    print("-"*70)
    
    best_distance = min(r['distance'] for r in results.values())
    
    for method, result in results.items():
        distance = result['distance']
        time_taken = result['time']
        quality = (best_distance / distance) * 100
        
        print(f"{method:<25} {distance:<15.2f} {time_taken:<15.3f} {quality:<10.1f}%")
    
    print("="*70)
    
    best_method = min(results.items(), key=lambda x: x[1]['distance'])[0]
    print(f"\nBest Solution: {best_method} with distance {results[best_method]['distance']:.2f}")
    
    fastest_method = min(results.items(), key=lambda x: x[1]['time'])[0]
    print(f"Fastest Method: {fastest_method} ({results[fastest_method]['time']:.3f}s)")
    
    # ==========================================
    # Visualizations
    # ==========================================
    if visualize:
        visualizer = TSPVisualizer()
        
        print("\nGenerating visualizations...")
        
        # Compare all tours
        tours = [results[m]['tour'] for m in ['DRL Fast', 'Genetic Algorithm', 'Ant Colony', 'Hybrid (DRL+GA)']]
        titles = ['DRL Fast', 'Genetic Algorithm', 'Ant Colony', 'Hybrid (DRL+GA)']
        visualizer.plot_comparison(tours, titles)
        
        # Plot convergence histories
        histories = {
            'Genetic Algorithm': results['Genetic Algorithm']['history'],
            'Ant Colony': results['Ant Colony']['history']
        }
        visualizer.plot_multiple_convergence(histories, title="Algorithm Convergence Comparison")
        
        # Plot best tour in detail
        best_tour = results[best_method]['tour']
        visualizer.plot_tour(best_tour, title=f"Best Tour ({best_method})", show_arrows=True)


def demo_single_solver(solver_type: str, cities: List[City], visualize: bool = True):
    """
    Demonstrate a single solver.
    
    Args:
        solver_type: Type of solver ('ga', 'aco', 'drl', 'hybrid')
        cities: List of cities to solve
        visualize: Whether to visualize results
    """
    visualizer = TSPVisualizer() if visualize else None
    
    if solver_type == 'ga':
        print("\nRunning Genetic Algorithm...")
        solver = GeneticAlgorithmSolver(
            cities=cities,
            population_size=100,
            mutation_rate=0.015,
            tournament_size=5
        )
        tour = solver.solve(generations=1000, verbose=True)
        
        if visualizer:
            visualizer.plot_tour(tour, title="Genetic Algorithm Solution")
            visualizer.plot_convergence(solver.best_distance_history)
    
    elif solver_type == 'aco':
        print("\nRunning Ant Colony Optimization...")
        solver = AntColonyOptimizer(
            cities=cities,
            n_ants=50,
            n_iterations=500
        )
        tour = solver.solve(verbose=True)
        
        if visualizer:
            visualizer.plot_tour(tour, title="Ant Colony Optimization Solution")
            visualizer.plot_convergence(solver.best_distance_history, xlabel="Iteration")
    
    elif solver_type == 'drl':
        print("\nRunning DRL Fast Solver...")
        solver = DRLSolver(cities)
        tour = solver.solve_fast(use_2opt=True, verbose=True)
        
        if visualizer:
            visualizer.plot_tour(tour, title="DRL Fast Solution")
    
    elif solver_type == 'hybrid':
        print("\nRunning Hybrid Solver (DRL + GA)...")
        solver = HybridSolver(cities)
        tour = solver.solve(quick_generations=500, verbose=True)
        
        if visualizer:
            visualizer.plot_tour(tour, title="Hybrid Solution (DRL + GA)")
    
    else:
        print(f"Unknown solver type: {solver_type}")
        sys.exit(1)


def main():
    """Main entry point for the TSP solver application."""
    parser = argparse.ArgumentParser(
        description="TSP Solver - Solve the Traveling Salesman Problem using multiple algorithms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark all solvers on 30 random cities
  python main.py --benchmark --cities 30
  
  # Run Genetic Algorithm on 50 cities
  python main.py --solver ga --cities 50
  
  # Run Hybrid solver on circle pattern
  python main.py --solver hybrid --cities 25 --pattern circle
  
  # Quick DRL solution for 100 cities
  python main.py --solver drl --cities 100 --no-viz
        """
    )
    
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Benchmark all solvers and compare results'
    )
    
    parser.add_argument(
        '--solver',
        type=str,
        choices=['ga', 'aco', 'drl', 'hybrid'],
        help='Run a specific solver (ga=Genetic Algorithm, aco=Ant Colony, drl=DRL Fast, hybrid=DRL+GA)'
    )
    
    parser.add_argument(
        '--cities',
        type=int,
        default=30,
        help='Number of cities to generate (default: 30)'
    )
    
    parser.add_argument(
        '--pattern',
        type=str,
        choices=['random', 'circle'],
        default='random',
        help='City placement pattern (default: random)'
    )
    
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Disable visualizations'
    )
    
    args = parser.parse_args()
    
    # Generate cities
    print(f"\nGenerating {args.cities} cities in {args.pattern} pattern...")
    
    if args.pattern == 'circle':
        cities = generate_circle_cities(args.cities, radius=50)
    else:
        cities = generate_random_cities(args.cities, width=100, height=100)
    
    visualize = not args.no_viz
    
    # Run requested mode
    if args.benchmark:
        benchmark_all_solvers(cities, visualize=visualize)
    elif args.solver:
        demo_single_solver(args.solver, cities, visualize=visualize)
    else:
        # Default: run benchmark
        print("\nNo mode specified. Running benchmark...")
        benchmark_all_solvers(cities, visualize=visualize)


if __name__ == "__main__":
    main()