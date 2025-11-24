import numpy as np
import random
import time
from typing import List, Optional, Callable

from tsp_core import City, Tour, DistanceMatrix


class AntColonyOptimizer:
    """
    ACO with:
    - seed_tour support
    - time_limit support
    - returns (best_tour, log)
    """

    def __init__(
        self,
        cities: List[City],
        n_ants: int = 50,
        alpha=1.0,
        beta=3.0,
        rho=0.5,
        q=100.0,
        elite_weight=2.0,
        seed_weight=50.0    # IMPORTANT CHANGE (strong seeding)
    ):
        self.cities = cities
        self.n_cities = len(cities)
        self.n_ants = n_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.elite_weight = elite_weight
        self.seed_weight = seed_weight

        self.distance_matrix = DistanceMatrix(cities)
        self.pheromones = None

        self.best_tour = None
        self.best_distance = float("inf")

    # --------------------------------------------------------
    def _initialize_pheromones(self, seed_tour=None):
        base = 0.01
        self.pheromones = np.ones((self.n_cities, self.n_cities)) * base

        if seed_tour is None:
            return

        dep = (self.q / seed_tour.get_total_distance()) * self.seed_weight

        for i in range(self.n_cities):
            c1 = seed_tour.cities[i]
            c2 = seed_tour.cities[(i+1) % self.n_cities]
            i1 = self.distance_matrix.city_to_index[c1]
            i2 = self.distance_matrix.city_to_index[c2]

            self.pheromones[i1][i2] += dep
            self.pheromones[i2][i1] += dep

    # --------------------------------------------------------
    def _construct_tour(self):
        start = random.randint(0, self.n_cities - 1)
        tour = [start]
        unvisited = set(range(self.n_cities))
        unvisited.remove(start)

        cur = start
        while unvisited:
            probs = {}
            total = 0.0

            for j in unvisited:
                tau = self.pheromones[cur][j] ** self.alpha
                dist = self.distance_matrix.get_distance_by_index(cur, j)
                eta = (1.0 / dist) ** self.beta
                p = tau * eta
                probs[j] = p
                total += p

            keys = list(probs.keys())
            vals = [probs[k] / total for k in keys]

            nxt = np.random.choice(keys, p=vals)
            tour.append(nxt)
            unvisited.remove(nxt)
            cur = nxt

        return Tour([self.cities[i] for i in tour])

    # --------------------------------------------------------
    def _update_pheromones(self, ant_tours):
        self.pheromones *= (1 - self.rho)

        for tour in ant_tours:
            d = tour.get_total_distance()
            dep = self.q / d

            for i in range(len(tour.cities)):
                c1 = tour.cities[i]
                c2 = tour.cities[(i + 1) % len(tour.cities)]
                i1 = self.distance_matrix.city_to_index[c1]
                i2 = self.distance_matrix.city_to_index[c2]

                self.pheromones[i1][i2] += dep
                self.pheromones[i2][i1] += dep

        if self.best_tour:
            dep = (self.q / self.best_tour.get_total_distance()) * self.elite_weight
            for i in range(len(self.best_tour.cities)):
                c1 = self.best_tour.cities[i]
                c2 = self.best_tour.cities[(i + 1) % len(self.best_tour.cities)]
                i1 = self.distance_matrix.city_to_index[c1]
                i2 = self.distance_matrix.city_to_index[c2]
                self.pheromones[i1][i2] += dep
                self.pheromones[i2][i1] += dep

    # --------------------------------------------------------
    def solve(
        self,
        iterations=1000,
        verbose=False,
        callback=None,
        seed_tour=None,
        time_limit=None
    ):
        t0 = time.time()
        log = []

        self._initialize_pheromones(seed_tour)

        if seed_tour:
            self.best_tour = seed_tour.clone()
            self.best_distance = self.best_tour.get_total_distance()
            log.append((0.0, self.best_distance))
        else:
            self.best_tour = None
            self.best_distance = float("inf")

        for it in range(iterations):
            if time_limit and time.time() - t0 >= time_limit:
                break

            ant_tours = []
            for _ in range(self.n_ants):

                if time_limit and time.time() - t0 >= time_limit:
                    break

                t = self._construct_tour()
                ant_tours.append(t)

                d = t.get_total_distance()
                if d < self.best_distance:
                    self.best_distance = d
                    self.best_tour = t.clone()
                    log.append((time.time() - t0, d))

            if not ant_tours:
                break

            self._update_pheromones(ant_tours)

            if callback:
                callback(self, it)

            if verbose and (it + 1) % 100 == 0:
                print(f"Iter {it+1} | Best = {self.best_distance:.2f}")

        if time_limit:
            log = [(t, d) for (t, d) in log if t <= time_limit]

        return self.best_tour, log

    # --------------------------------------------------------
    def get_best_tour(self):
        return self.best_tour
