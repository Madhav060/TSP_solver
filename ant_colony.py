"""
TSP Solver - Ant Colony Optimization Implementation
(Modified for Hybrid Seeding)
"""

import numpy as np
import random
from typing import List
from tsp_core import City, Tour, DistanceMatrix


class AntColonyOptimizer:
    """
    Ant Colony Optimization solver for TSP.
    (Modified to accept a seed_tour for hybrid initialization)
    """
    
    def __init__(
        self,
        cities: List[City],
        n_ants: int = 50,
        # n_iterations removed from here
        alpha: float = 1.0,      # Pheromone influence
        beta: float = 3.0,       # Distance influence
        rho: float = 0.5,        # Evaporation rate
        q: float = 100.0,        # Pheromone deposit factor
        elite_weight: float = 2.0,  # Extra pheromone for best ant
        seed_weight: float = 10.0 # *** NEW: How much to boost the seed tour
    ):
        self.cities = cities
        self.n_cities = len(cities)
        self.n_ants = n_ants
        # self.n_iterations = n_iterations # Moved to solve()
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.elite_weight = elite_weight
        self.seed_weight = seed_weight # *** NEW: Store seed weight
        
        self.distance_matrix = DistanceMatrix(cities)
        self.pheromones = None # Will be initialized in solve()
        
        self.best_tour = None
        self.best_distance = float('inf')
        self.best_distance_history = []
    
    def _initialize_pheromones(self, seed_tour: Tour = None):
        """
        *** NEW: Initialize pheromone matrix.
        If a seed_tour is provided, strongly boost its edges.
        """
        # Start with a base, low pheromone level everywhere
        base_pheromone = 0.1
        self.pheromones = np.ones((self.n_cities, self.n_cities)) * base_pheromone
        
        if seed_tour:
            # If we have a seed tour, give it a large initial deposit
            deposit = (self.q / seed_tour.get_total_distance()) * self.seed_weight
            
            for i in range(len(seed_tour.cities)):
                city1 = seed_tour.cities[i]
                city2 = seed_tour.cities[(i + 1) % len(seed_tour.cities)]
                idx1 = self.distance_matrix.city_to_index[city1]
                idx2 = self.distance_matrix.city_to_index[city2]
                
                # Add the large seed deposit
                self.pheromones[idx1][idx2] += deposit
                self.pheromones[idx2][idx1] += deposit

    # ... (Keep _calculate_probabilities and _construct_tour exactly as they were) ...
    def _calculate_probabilities(
        self,
        current_city_idx: int,
        unvisited: set,
        pheromones: np.ndarray
    ) -> dict:
        """
        Calculate probabilities for choosing the next city.
        """
        probabilities = {}
        total = 0.0
        
        for city_idx in unvisited:
            pheromone = pheromones[current_city_idx][city_idx] ** self.alpha
            distance = self.distance_matrix.get_distance_by_index(current_city_idx, city_idx)
            heuristic = (1.0 / distance) ** self.beta
            
            prob = pheromone * heuristic
            probabilities[city_idx] = prob
            total += prob
        
        if total > 0:
            for city_idx in probabilities:
                probabilities[city_idx] /= total
        
        return probabilities
    
    def _construct_tour(self) -> Tour:
        """
        An ant constructs a complete tour by probabilistically choosing cities.
        """
        start_idx = random.randint(0, self.n_cities - 1)
        tour_indices = [start_idx]
        unvisited = set(range(self.n_cities))
        unvisited.remove(start_idx)
        
        current_idx = start_idx
        
        while unvisited:
            probabilities = self._calculate_probabilities(
                current_idx,
                unvisited,
                self.pheromones
            )
            
            cities_list = list(probabilities.keys())
            probs_list = [probabilities[c] for c in cities_list]
            
            next_idx = np.random.choice(cities_list, p=probs_list)
            
            tour_indices.append(next_idx)
            unvisited.remove(next_idx)
            current_idx = next_idx
        
        tour_cities = [self.cities[idx] for idx in tour_indices]
        return Tour(tour_cities)
    
    def _update_pheromones(self, ant_tours: List[Tour]):
        """
        Update pheromone levels based on ant tours.
        """
        self.pheromones *= (1 - self.rho)
        
        for tour in ant_tours:
            distance = tour.get_total_distance()
            pheromone_deposit = self.q / distance
            
            for i in range(len(tour.cities)):
                city1 = tour.cities[i]
                city2 = tour.cities[(i + 1) % len(tour.cities)]
                idx1 = self.distance_matrix.city_to_index[city1]
                idx2 = self.distance_matrix.city_to_index[city2]
                
                self.pheromones[idx1][idx2] += pheromone_deposit
                self.pheromones[idx2][idx1] += pheromone_deposit
        
        if self.best_tour:
            elite_deposit = (self.q / self.best_tour.get_total_distance()) * self.elite_weight
            
            for i in range(len(self.best_tour.cities)):
                city1 = self.best_tour.cities[i]
                city2 = self.best_tour.cities[(i + 1) % len(self.best_tour.cities)]
                idx1 = self.distance_matrix.city_to_index[city1]
                idx2 = self.distance_matrix.city_to_index[city2]
                
                self.pheromones[idx1][idx2] += elite_deposit
                self.pheromones[idx2][idx1] += elite_deposit
    
    def solve(
        self,
        iterations: int = 1000, # *** MODIFIED: Iterations passed in here
        verbose: bool = True,
        callback=None,
        seed_tour: Tour = None   # *** NEW: Accept a seed tour
    ) -> Tour:
        """
        Run the Ant Colony Optimization algorithm.
        """
        # *** NEW: Initialize pheromones, using the seed if provided
        self._initialize_pheromones(seed_tour)
        
        # Reset bests
        self.best_tour = seed_tour.clone() if seed_tour else None
        self.best_distance = seed_tour.get_total_distance() if seed_tour else float('inf')
        self.best_distance_history = []
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Starting Ant Colony Optimization")
            if seed_tour:
                print(f"Mode: Hybrid (Seeded with DRL tour, dist: {self.best_distance:.2f})")
            print(f"{'='*60}")
            print(f"Number of Ants: {self.n_ants}")
            print(f"Iterations: {iterations}") # Use local var
            print(f"Alpha (pheromone): {self.alpha}")
            print(f"Beta (distance): {self.beta}")
            print(f"Rho (evaporation): {self.rho}")
            print(f"{'='*60}\n")
        
        
        for iteration in range(iterations): # *** MODIFIED: Use local var
            ant_tours = []
            for _ in range(self.n_ants):
                tour = self._construct_tour()
                ant_tours.append(tour)
                
                distance = tour.get_total_distance()
                if distance < self.best_distance:
                    self.best_distance = distance
                    self.best_tour = tour.clone()
            
            self._update_pheromones(ant_tours)
            
            self.best_distance_history.append(self.best_distance)
            
            if callback:
                callback(self, iteration)
            
            if verbose and (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1:4d} | "
                      f"Best Distance: {self.best_distance:8.2f}")
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Optimization Complete!")
            print(f"Best Distance Found: {self.best_distance:.2f}")
            print(f"{'='*60}\n")
        
        return self.best_tour
    
    def get_best_tour(self) -> Tour:
        """Get the current best tour."""
        return self.best_tour