import time
from typing import List, Optional, Tuple

# Assuming these modules are available and imported correctly
from tsp_core import City, Tour
from drl_solver import DRLSolver
from ant_colony import AntColonyOptimizer


class HybridSolverACO:
    """
    Hybrid TSP Solver:
    - Phase 1: DRL for fast high-quality initialization
    - Phase 2: ACO refinement ONLY within remaining time_limit
    - Adaptive ACO iterations based on TSP size
    """

    def __init__(self, cities: List[City], aco_params=None):
        self.cities = cities
        self.drl = DRLSolver(cities)

        # Default ACO parameters
        # --- FIX APPLIED HERE ---
        # seed_weight reduced from 50.0 to 5.0 to prevent strong initial bias
        self.aco_params = aco_params or {
            "n_ants": 25,
            "alpha": 1.0,
            "beta": 3.0,
            "rho": 0.5,
            "q": 100.0,
            "elite_weight": 2.0,
            "seed_weight": 5.0,   # Lowered bias for more exploration
        }

    # --------------------------------------------------------
    #  ADAPTIVE ACO ITERATIONS BASED ON #CITIES
    # --------------------------------------------------------
    def compute_iterations(self):
        n = len(self.cities)

        if 10 <= n < 20:
            return 100
        elif 20 <= n < 30:
            return 200
        elif 30 <= n < 40:
            return 300
        elif 40 <= n < 50:
            return 400
        elif 50 <= n < 60:
            return 500

        # Generic fallback rule for large inputs
        return min(max(n * 8, 200), 1000)

    # --------------------------------------------------------
    #  MAIN HYBRID SOLVER
    # --------------------------------------------------------
    def solve(
        self,
        quick_iterations: Optional[int] = None,
        callback=None,  # Added callback for external logging/timing
        verbose: bool = False,
        use_2opt: bool = True,
        time_limit: Optional[float] = None
    ) -> Tuple[Tour, list]:

        start_time = time.time()
        log = []
        best_tour = None # Track best tour across both phases

        # --------------------------------------------------------
        #  PHASE 1 — DRL INITIALIZATION
        # --------------------------------------------------------
        drl_tour, _ = self.drl.solve_fast(
            use_2opt=use_2opt,
            verbose=False,
            time_limit=None
        )

        best_tour = drl_tour
        init_dist = drl_tour.get_total_distance()
        
        # Log and callback DRL result immediately
        log.append((0.0, init_dist))
        if callback:
            callback(best_tour) 

        elapsed = time.time() - start_time

        # Remaining time for refinement
        remaining = None
        if time_limit is not None:
            remaining = max(0.0, time_limit - elapsed)
            if remaining <= 0:
                return best_tour, log # No time for ACO

        # --------------------------------------------------------
        #  PHASE 2 — ACO REFINEMENT
        # --------------------------------------------------------

        # Determine ACO iterations adaptively if not given
        if quick_iterations is None:
            quick_iterations = self.compute_iterations()

        aco = AntColonyOptimizer(self.cities, **self.aco_params)

        aco_tour, aco_log = aco.solve(
            iterations=quick_iterations,
            verbose=False,
            seed_tour=drl_tour,
            time_limit=remaining
        )

        # Merge ACO log (shift by DRL time) and find the true best tour
        for t, d in aco_log:
            current_time = elapsed + t
            log.append((current_time, d))
            
            # Update best_tour
            if d < best_tour.get_total_distance():
                best_tour = aco_tour
                
            # Call back for external logging/time-to-target tracking
            if callback:
                callback(aco_tour) 


        # Ensure the final returned tour is the one with the best distance found
        return best_tour, log