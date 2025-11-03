"""
TSP Solver - Core Module
Contains the fundamental data structures for representing the TSP problem.
"""

import numpy as np
import random
from typing import List, Tuple
import copy


class City:
    """Represents a city with x, y coordinates."""
    
    def __init__(self, x: float, y: float, name: str = None):
        self.x = x
        self.y = y
        self.name = name or f"City_{id(self)}"
    
    def distance_to(self, city: 'City') -> float:
        """Calculate Euclidean distance to another city."""
        dx = self.x - city.x
        dy = self.y - city.y
        return np.sqrt(dx * dx + dy * dy)
    
    def __repr__(self):
        return f"City({self.x:.2f}, {self.y:.2f})"
    
    def __eq__(self, other):
        if not isinstance(other, City):
            return False
        return self.x == other.x and self.y == other.y
    
    def __hash__(self):
        return hash((self.x, self.y))


class Tour:
    """Represents a tour (solution) as an ordered sequence of cities."""
    
    def __init__(self, cities: List[City] = None):
        self.cities = cities if cities else []
        self._distance = None
        self._fitness = None
    
    def get_total_distance(self) -> float:
        """Calculate the total distance of the tour."""
        if self._distance is not None:
            return self._distance
        
        if len(self.cities) == 0:
            return 0.0
        
        distance = 0.0
        for i in range(len(self.cities)):
            from_city = self.cities[i]
            to_city = self.cities[(i + 1) % len(self.cities)]
            distance += from_city.distance_to(to_city)
        
        self._distance = distance
        return distance
    
    def get_fitness(self) -> float:
        """Calculate fitness (inverse of distance)."""
        if self._fitness is not None:
            return self._fitness
        
        distance = self.get_total_distance()
        self._fitness = 1.0 / distance if distance > 0 else 0.0
        return self._fitness
    
    def clone(self) -> 'Tour':
        """Create a deep copy of the tour."""
        return Tour(self.cities.copy())
    
    def invalidate_cache(self):
        """Invalidate cached distance and fitness values."""
        self._distance = None
        self._fitness = None
    
    def __len__(self):
        return len(self.cities)
    
    def __repr__(self):
        return f"Tour(cities={len(self.cities)}, distance={self.get_total_distance():.2f})"
    
    def __getitem__(self, index):
        return self.cities[index]
    
    def __setitem__(self, index, value):
        self.cities[index] = value
        self.invalidate_cache()


class DistanceMatrix:
    """Precomputed distance matrix for efficient distance lookups."""
    
    def __init__(self, cities: List[City]):
        self.cities = cities
        self.n = len(cities)
        self.matrix = np.zeros((self.n, self.n))
        self.city_to_index = {city: i for i, city in enumerate(cities)}
        
        # Precompute all distances
        for i in range(self.n):
            for j in range(i + 1, self.n):
                dist = cities[i].distance_to(cities[j])
                self.matrix[i][j] = dist
                self.matrix[j][i] = dist
    
    def get_distance(self, city1: City, city2: City) -> float:
        """Get precomputed distance between two cities."""
        i = self.city_to_index[city1]
        j = self.city_to_index[city2]
        return self.matrix[i][j]
    
    def get_distance_by_index(self, i: int, j: int) -> float:
        """Get distance by city indices."""
        return self.matrix[i][j]