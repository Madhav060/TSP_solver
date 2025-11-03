"""
TSP Solver - Genetic Algorithm Implementation
Goal 1: Create the Best Quality Solution (Near-Perfect)
"""

import random
import numpy as np
from typing import List, Tuple
from tsp_core import City, Tour


class Population:
    """Represents a population of tour solutions."""
    
    def __init__(self, population_size: int, cities: List[City]):
        self.population_size = population_size
        self.cities = cities
        self.tours: List[Tour] = []
    
    def initialize(self):
        """Initialize population with random tours."""
        print(f"Initializing population with {self.population_size} random tours...")
        self.tours = []
        
        for _ in range(self.population_size):
            shuffled_cities = self.cities.copy()
            random.shuffle(shuffled_cities)
            self.tours.append(Tour(shuffled_cities))
    
    def seed_with_tour(self, seed_tour: Tour, copies: int = 1):
        """Seed the population with a good starting tour (for hybrid approach)."""
        for _ in range(copies):
            self.tours.append(seed_tour.clone())
    
    def get_fittest(self) -> Tour:
        """Find and return the fittest tour in the population."""
        return max(self.tours, key=lambda tour: tour.get_fitness())
    
    def get_average_distance(self) -> float:
        """Calculate average distance across all tours."""
        return np.mean([tour.get_total_distance() for tour in self.tours])


class GeneticAlgorithmSolver:
    """
    Genetic Algorithm solver for TSP.
    Uses tournament selection, ordered crossover, and swap mutation.
    """
    
    def __init__(
        self,
        cities: List[City],
        population_size: int = 100,
        mutation_rate: float = 0.015,
        tournament_size: int = 5,
        elitism: bool = True,
        elite_size: int = 1
    ):
        self.cities = cities
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.elite_size = elite_size
        
        self.population = None
        self.generation = 0
        self.best_distance_history = []
        self.avg_distance_history = []
    
    def initialize(self, seed_tour: Tour = None):
        """Initialize the population."""
        self.population = Population(self.population_size, self.cities)
        
        if seed_tour:
            # Hybrid approach: seed with a good solution
            print("Seeding population with provided tour...")
            self.population.seed_with_tour(seed_tour, copies=self.elite_size)
            # Fill the rest with random tours
            remaining = self.population_size - self.elite_size
            for _ in range(remaining):
                shuffled_cities = self.cities.copy()
                random.shuffle(shuffled_cities)
                self.population.tours.append(Tour(shuffled_cities))
        else:
            # Start from scratch with random tours
            self.population.initialize()
        
        self.generation = 0
        self.best_distance_history = []
        self.avg_distance_history = []
        
        # Record initial statistics
        best = self.population.get_fittest()
        self.best_distance_history.append(best.get_total_distance())
        self.avg_distance_history.append(self.population.get_average_distance())
    
    def tournament_selection(self) -> Tour:
        """
        Tournament selection: pick random tours and return the best one.
        """
        tournament = random.sample(self.population.tours, self.tournament_size)
        return max(tournament, key=lambda tour: tour.get_fitness())
    
    def ordered_crossover(self, parent1: Tour, parent2: Tour) -> Tour:
        """
        Ordered Crossover (OX): Combine two parent tours to create a child.
        Preserves a subsequence from parent1 and fills the rest from parent2.
        """
        size = len(parent1.cities)
        
        # Select random start and end positions
        start = random.randint(0, size - 1)
        end = random.randint(0, size - 1)
        
        if start > end:
            start, end = end, start
        
        # Initialize child with None
        child_cities = [None] * size
        
        # Copy subsequence from parent1
        for i in range(start, end + 1):
            child_cities[i] = parent1.cities[i]
        
        # Fill remaining positions with cities from parent2
        parent2_index = 0
        for i in range(size):
            if child_cities[i] is None:
                # Find next city from parent2 that's not already in child
                while parent2.cities[parent2_index] in child_cities:
                    parent2_index += 1
                child_cities[i] = parent2.cities[parent2_index]
                parent2_index += 1
        
        return Tour(child_cities)
    
    def swap_mutation(self, tour: Tour) -> Tour:
        """
        Swap Mutation: Randomly swap two cities in the tour.
        Applied with a probability of mutation_rate.
        """
        mutated_tour = tour.clone()
        
        for i in range(len(mutated_tour.cities)):
            if random.random() < self.mutation_rate:
                j = random.randint(0, len(mutated_tour.cities) - 1)
                # Swap cities at positions i and j
                mutated_tour.cities[i], mutated_tour.cities[j] = \
                    mutated_tour.cities[j], mutated_tour.cities[i]
                mutated_tour.invalidate_cache()
        
        return mutated_tour
    
    def evolve_generation(self):
        """Evolve the population by one generation."""
        new_population = Population(self.population_size, self.cities)
        new_population.tours = []
        
        # Elitism: Keep the best tours from the previous generation
        if self.elitism:
            # Sort by fitness and keep the elite
            sorted_tours = sorted(
                self.population.tours,
                key=lambda t: t.get_fitness(),
                reverse=True
            )
            for i in range(self.elite_size):
                new_population.tours.append(sorted_tours[i].clone())
        
        # Fill the rest of the new population
        while len(new_population.tours) < self.population_size:
            # Selection
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            
            # Crossover
            child = self.ordered_crossover(parent1, parent2)
            
            # Mutation
            child = self.swap_mutation(child)
            
            new_population.tours.append(child)
        
        # Replace old population
        self.population = new_population
        self.generation += 1
        
        # Record statistics
        best = self.population.get_fittest()
        self.best_distance_history.append(best.get_total_distance())
        self.avg_distance_history.append(self.population.get_average_distance())
    
    def solve(self, generations: int = 1000, verbose: bool = True, callback=None) -> Tour:
        """
        Run the genetic algorithm for a specified number of generations.
        
        Args:
            generations: Number of generations to evolve
            verbose: Print progress information
            callback: Optional callback function called after each generation
        
        Returns:
            The best tour found
        """
        if self.population is None:
            self.initialize()
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Starting Genetic Algorithm")
            print(f"{'='*60}")
            print(f"Population Size: {self.population_size}")
            print(f"Mutation Rate: {self.mutation_rate}")
            print(f"Tournament Size: {self.tournament_size}")
            print(f"Elitism: {self.elitism} (Elite Size: {self.elite_size})")
            print(f"Generations: {generations}")
            print(f"{'='*60}\n")
            
            initial_best = self.population.get_fittest()
            print(f"Initial Best Distance: {initial_best.get_total_distance():.2f}")
        
        # Evolution loop
        for gen in range(generations):
            self.evolve_generation()
            
            if callback:
                callback(self)
            
            if verbose and (gen + 1) % 100 == 0:
                best = self.population.get_fittest()
                avg = self.population.get_average_distance()
                print(f"Generation {gen + 1:4d} | "
                      f"Best: {best.get_total_distance():8.2f} | "
                      f"Avg: {avg:8.2f}")
        
        best_tour = self.population.get_fittest()
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Evolution Complete!")
            print(f"{'='*60}")
            print(f"Final Best Distance: {best_tour.get_total_distance():.2f}")
            improvement = ((self.best_distance_history[0] - best_tour.get_total_distance()) 
                          / self.best_distance_history[0] * 100)
            print(f"Improvement: {improvement:.2f}%")
            print(f"{'='*60}\n")
        
        return best_tour
    
    def get_best_tour(self) -> Tour:
        """Get the current best tour."""
        return self.population.get_fittest() if self.population else None