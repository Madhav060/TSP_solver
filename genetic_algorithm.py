"""
Genetic Algorithm Solver with Time-Limit Support + Convergence Logging
Compatible with multi-instance benchmark.
"""

import random
import time
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
        self.tours = []
        for _ in range(self.population_size):
            shuffled = self.cities.copy()
            random.shuffle(shuffled)
            self.tours.append(Tour(shuffled))
    
    def seed_with_tour(self, seed_tour: Tour, copies: int = 1):
        for _ in range(copies):
            self.tours.append(seed_tour.clone())
    
    def get_fittest(self) -> Tour:
        return max(self.tours, key=lambda tour: tour.get_fitness())
    
    def get_average_distance(self) -> float:
        return np.mean([t.get_total_distance() for t in self.tours])


class GeneticAlgorithmSolver:
    """
    Genetic Algorithm solver for TSP
    Upgraded for:
    - time-limit execution
    - convergence logging
    - compatibility with multi-instance benchmark
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

        # GA state
        self.population = None
        self.generation = 0

    # ---------------------------------------
    # Initialization
    # ---------------------------------------

    def initialize(self, seed_tour: Tour = None):
        self.population = Population(self.population_size, self.cities)

        if seed_tour:
            self.population.seed_with_tour(seed_tour, copies=self.elite_size)
            for _ in range(self.population_size - self.elite_size):
                shuffled = self.cities.copy()
                random.shuffle(shuffled)
                self.population.tours.append(Tour(shuffled))
        else:
            self.population.initialize()

        self.generation = 0

    # ---------------------------------------
    # Genetic operators
    # ---------------------------------------

    def tournament_selection(self) -> Tour:
        candidates = random.sample(self.population.tours, self.tournament_size)
        return max(candidates, key=lambda t: t.get_fitness())

    def ordered_crossover(self, parent1: Tour, parent2: Tour) -> Tour:
        size = len(parent1.cities)
        start = random.randint(0, size - 1)
        end = random.randint(0, size - 1)
        if start > end:
            start, end = end, start

        child = [None] * size
        for i in range(start, end + 1):
            child[i] = parent1.cities[i]

        p2_idx = 0
        for i in range(size):
            if child[i] is None:
                while parent2.cities[p2_idx] in child:
                    p2_idx += 1
                child[i] = parent2.cities[p2_idx]
                p2_idx += 1

        return Tour(child)

    def swap_mutation(self, tour: Tour):
        mutated = tour.clone()
        for i in range(len(mutated.cities)):
            if random.random() < self.mutation_rate:
                j = random.randint(0, len(mutated.cities) - 1)
                mutated.cities[i], mutated.cities[j] = (
                    mutated.cities[j],
                    mutated.cities[i],
                )
                mutated.invalidate_cache()
        return mutated

    # ---------------------------------------
    # Single generation evolution
    # ---------------------------------------

    def evolve_generation(self):
        new_pop = Population(self.population_size, self.cities)
        new_pop.tours = []

        # --- elitism ---
        if self.elitism:
            elites = sorted(
                self.population.tours,
                key=lambda t: t.get_fitness(),
                reverse=True,
            )[: self.elite_size]
            for elite in elites:
                new_pop.tours.append(elite.clone())

        # --- generate rest ---
        while len(new_pop.tours) < self.population_size:
            p1 = self.tournament_selection()
            p2 = self.tournament_selection()
            child = self.ordered_crossover(p1, p2)
            child = self.swap_mutation(child)
            new_pop.tours.append(child)

        self.population = new_pop
        self.generation += 1

    # ---------------------------------------
    # TIME-LIMITED SOLVE FUNCTION
    # ---------------------------------------

    def solve(self, time_limit=None, verbose=False, generations=None) -> Tuple[Tour, list]:
        """
        Returns:
            best_tour
            log = [(time, best_distance)]
        """
        if self.population is None:
            self.initialize()

        start = time.time()
        log = []

        # Initial best
        best = self.population.get_fittest()
        best_dist = best.get_total_distance()
        log.append((0.0, best_dist))

        gen = 0
        while True:

            # stop if time finished
            if time_limit is not None and time.time() - start >= time_limit:
                break

            # stop if fixed generation requested
            if generations is not None and gen >= generations:
                break

            # evolve generation
            self.evolve_generation()
            gen += 1

            # log best
            best = self.population.get_fittest()
            dist = best.get_total_distance()

            if dist < best_dist:
                best_dist = dist
                log.append((time.time() - start, best_dist))

        # clip log
        if time_limit is not None:
            log = [(t, d) for (t, d) in log if t <= time_limit]

        return best, log

    def get_best_tour(self):
        return self.population.get_fittest() if self.population else None
