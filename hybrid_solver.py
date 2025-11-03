"""
Hybrid Solver: DRL + Ant Colony Optimization
Uses DRL for fast initialization, then refines with ACO
"""

from typing import List, Optional, Callable
from tsp_core import City, Tour
from drl_solver import DRLSolver
from ant_colony import AntColonyOptimizer


class HybridSolverACO:
    """
    Hybrid solver combining Deep RL and Ant Colony Optimization.
    DRL provides a high-quality initial solution quickly.
    ACO refines it further with its exploration capabilities.
    """
    
    def __init__(
        self,
        cities: List[City],
        drl_solver: Optional[DRLSolver] = None,
        aco_params: Optional[dict] = None
    ):
        """
        Initialize the hybrid solver.
        
        Args:
            cities: List of City objects
            drl_solver: Pre-loaded DRL solver (optional, will create if None)
            aco_params: Parameters for ACO solver
        """
        self.cities = cities
        self.drl_solver = drl_solver  # May be None
        
        # Default ACO parameters optimized for refinement
        self.aco_params = aco_params or {
            'n_ants': 25,
            'alpha': 1.0,
            'beta': 3.0,
            'rho': 0.5,
            'q': 100.0,
            'elite_weight': 2.0,
            'seed_weight': 10.0
        }
    
    def solve(
        self,
        quick_iterations: int = 500,
        verbose: bool = True,
        use_2opt: bool = True
    ) -> Tour:
        """
        Solve the TSP using the hybrid approach.
        
        Args:
            quick_iterations: Number of ACO iterations
            verbose: Print progress information
            use_2opt: Apply 2-opt improvement to DRL solution
            
        Returns:
            Best tour found
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"HYBRID SOLVER: DRL + Ant Colony Optimization")
            print(f"{'='*60}")
            print(f"Phase 1: DRL Fast Inference")
            print(f"Phase 2: ACO Refinement ({quick_iterations} iterations)")
            print(f"{'='*60}\n")
        
        # Phase 1: Get initial solution from DRL
        if self.drl_solver is None:
            if verbose:
                print("Initializing DRL solver...")
            self.drl_solver = DRLSolver(self.cities)
        
        initial_tour = self.drl_solver.solve_fast(use_2opt=use_2opt, verbose=verbose)
        
        if verbose:
            print(f"\n[Phase 1 Complete]")
            print(f"DRL Initial Distance: {initial_tour.get_total_distance():.2f}")
            print(f"\n[Phase 2: ACO Refinement Starting...]")
        
        # Phase 2: Refine with ACO
        aco_solver = AntColonyOptimizer(
            cities=self.cities,
            **self.aco_params
        )
        
        final_tour = aco_solver.solve(
            iterations=quick_iterations,
            seed_tour=initial_tour,
            verbose=verbose
        )
        
        if verbose:
            improvement = (
                (initial_tour.get_total_distance() - final_tour.get_total_distance()) 
                / initial_tour.get_total_distance() * 100
            )
            print(f"\n{'='*60}")
            print(f"HYBRID SOLVER COMPLETE")
            print(f"{'='*60}")
            print(f"Initial (DRL):     {initial_tour.get_total_distance():.2f}")
            print(f"Final (DRL+ACO):   {final_tour.get_total_distance():.2f}")
            print(f"Improvement:       {improvement:.2f}%")
            print(f"{'='*60}\n")
        
        return final_tour
    
    def solve_step_by_step(
        self,
        quick_iterations: int = 500,
        callback: Optional[Callable] = None,
        initial_tour: Optional[Tour] = None  # *** CRITICAL: Accept pre-calculated tour ***
    ) -> Tour:
        """
        Solve with step-by-step callbacks for visualization.
        
        *** IMPORTANT: For thread safety, pass initial_tour pre-calculated in main thread ***
        
        Args:
            quick_iterations: Number of ACO iterations
            callback: Function called with (solver, phase, iteration, tour)
            initial_tour: Pre-calculated DRL tour (REQUIRED for thread safety)
            
        Returns:
            Best tour found
        """
        # Phase 1: DRL
        # *** CRITICAL FIX: DO NOT call DRL in this thread if running in background ***
        if initial_tour is None:
            # Only calculate DRL tour if not provided
            # This should ONLY happen when called from main thread
            if self.drl_solver is None:
                self.drl_solver = DRLSolver(self.cities)
            
            if callback:
                callback(self, 'drl', 0, None)
            
            initial_tour = self.drl_solver.solve_fast(use_2opt=True, verbose=False)
        
        # Send the DRL tour to callback
        if callback:
            callback(self, 'drl', 1, initial_tour)
        
        # Phase 2: ACO Refinement
        aco_solver = AntColonyOptimizer(
            cities=self.cities,
            **self.aco_params
        )
        
        # Callback wrapper for ACO
        def aco_callback(solver, iteration):
            if callback:
                best_tour = solver.get_best_tour()
                # Pass 1-indexed iteration
                callback(self, 'aco', iteration + 1, best_tour)
        
        # Run ACO with the initial tour as seed
        final_tour = aco_solver.solve(
            iterations=quick_iterations,
            verbose=False,
            callback=aco_callback,
            seed_tour=initial_tour
        )
        
        return final_tour


def solve_tsp_hybrid_aco(
    cities: List[City],
    iterations: int = 500,
    verbose: bool = True
) -> Tour:
    """
    Convenience function to solve TSP with hybrid approach.
    
    Args:
        cities: List of City objects
        iterations: Number of ACO iterations for refinement
        verbose: Print progress information
        
    Returns:
        Best tour found
    """
    solver = HybridSolverACO(cities)
    return solver.solve(quick_iterations=iterations, verbose=verbose)