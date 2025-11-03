"""
TSP Solver - Visualization Module
Create beautiful visualizations of tours and solver progress.
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch
import numpy as np
from typing import List, Optional
from tsp_core import Tour, City


class TSPVisualizer:
    """Visualize TSP tours and optimization progress."""
    
    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize
    
    def plot_tour(
        self,
        tour: Tour,
        title: str = "TSP Tour",
        show_arrows: bool = True,
        save_path: str = None
    ):
        """
        Plot a single tour.
        
        Args:
            tour: The tour to visualize
            title: Plot title
            show_arrows: Show direction arrows on edges
            save_path: Optional path to save the figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if len(tour.cities) == 0:
            ax.text(0.5, 0.5, 'No cities in tour', 
                   ha='center', va='center', fontsize=16)
            return
        
        # Extract coordinates
        x_coords = [city.x for city in tour.cities]
        y_coords = [city.y for city in tour.cities]
        
        # Close the loop
        x_coords.append(tour.cities[0].x)
        y_coords.append(tour.cities[0].y)
        
        # Plot cities
        ax.scatter(x_coords[:-1], y_coords[:-1], 
                  c='red', s=200, zorder=3, edgecolors='darkred', linewidth=2)
        
        # Plot tour path
        ax.plot(x_coords, y_coords, 
               'b-', linewidth=2, alpha=0.6, zorder=1)
        
        # Add city labels
        for i, city in enumerate(tour.cities):
            ax.annotate(str(i), 
                       (city.x, city.y),
                       fontsize=9,
                       ha='center',
                       va='center',
                       color='white',
                       weight='bold')
        
        # Add arrows to show direction
        if show_arrows and len(tour.cities) > 1:
            for i in range(len(tour.cities)):
                start = tour.cities[i]
                end = tour.cities[(i + 1) % len(tour.cities)]
                
                # Calculate arrow position (middle of edge)
                mid_x = (start.x + end.x) / 2
                mid_y = (start.y + end.y) / 2
                
                # Calculate direction
                dx = end.x - start.x
                dy = end.y - start.y
                
                # Draw small arrow
                ax.annotate('', 
                           xy=(mid_x + dx*0.1, mid_y + dy*0.1),
                           xytext=(mid_x - dx*0.1, mid_y - dy*0.1),
                           arrowprops=dict(arrowstyle='->', 
                                         color='blue',
                                         lw=2,
                                         alpha=0.7))
        
        # Highlight start city
        start_city = tour.cities[0]
        ax.scatter([start_city.x], [start_city.y], 
                  c='green', s=300, zorder=4, 
                  marker='*', edgecolors='darkgreen', linewidth=2)
        
        # Set labels and title
        distance = tour.get_total_distance()
        ax.set_title(f"{title}\nTotal Distance: {distance:.2f}", 
                    fontsize=14, weight='bold')
        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Y Coordinate', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Tour saved to {save_path}")
        
        plt.show()
    
    def plot_comparison(
        self,
        tours: List[Tour],
        titles: List[str],
        save_path: str = None
    ):
        """
        Plot multiple tours side by side for comparison.
        
        Args:
            tours: List of tours to compare
            titles: List of titles for each tour
            save_path: Optional path to save the figure
        """
        n_tours = len(tours)
        fig, axes = plt.subplots(1, n_tours, figsize=(6*n_tours, 6))
        
        if n_tours == 1:
            axes = [axes]
        
        for ax, tour, title in zip(axes, tours, titles):
            if len(tour.cities) == 0:
                ax.text(0.5, 0.5, 'No cities', ha='center', va='center')
                continue
            
            # Extract coordinates
            x_coords = [city.x for city in tour.cities]
            y_coords = [city.y for city in tour.cities]
            x_coords.append(tour.cities[0].x)
            y_coords.append(tour.cities[0].y)
            
            # Plot
            ax.scatter(x_coords[:-1], y_coords[:-1], 
                      c='red', s=150, zorder=3, edgecolors='darkred', linewidth=1.5)
            ax.plot(x_coords, y_coords, 'b-', linewidth=2, alpha=0.6)
            
            # Highlight start
            ax.scatter([tour.cities[0].x], [tour.cities[0].y],
                      c='green', s=200, zorder=4, marker='*')
            
            distance = tour.get_total_distance()
            ax.set_title(f"{title}\nDistance: {distance:.2f}", 
                        fontsize=12, weight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison saved to {save_path}")
        
        plt.show()
    
    def plot_convergence(
        self,
        history: List[float],
        title: str = "Convergence History",
        xlabel: str = "Generation/Iteration",
        ylabel: str = "Best Distance",
        save_path: str = None
    ):
        """
        Plot the convergence history of an optimization algorithm.
        
        Args:
            history: List of best distances over time
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            save_path: Optional path to save the figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        iterations = range(len(history))
        
        ax.plot(iterations, history, 'b-', linewidth=2, label='Best Distance')
        ax.fill_between(iterations, history, alpha=0.3)
        
        # Add initial and final values
        initial = history[0]
        final = history[-1]
        improvement = ((initial - final) / initial) * 100
        
        ax.axhline(y=final, color='g', linestyle='--', 
                  linewidth=1.5, label=f'Final: {final:.2f}')
        ax.axhline(y=initial, color='r', linestyle='--', 
                  linewidth=1.5, label=f'Initial: {initial:.2f}')
        
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f"{title}\nImprovement: {improvement:.2f}%", 
                    fontsize=14, weight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Convergence plot saved to {save_path}")
        
        plt.show()
    
    def plot_multiple_convergence(
        self,
        histories: dict,
        title: str = "Algorithm Comparison",
        save_path: str = None
    ):
        """
        Plot convergence histories of multiple algorithms.
        
        Args:
            histories: Dict mapping algorithm names to history lists
            title: Plot title
            save_path: Optional path to save the figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = ['b', 'r', 'g', 'orange', 'purple']
        
        for i, (name, history) in enumerate(histories.items()):
            color = colors[i % len(colors)]
            iterations = range(len(history))
            ax.plot(iterations, history, 
                   linewidth=2, 
                   label=f"{name} (Final: {history[-1]:.2f})",
                   color=color)
        
        ax.set_xlabel('Generation/Iteration', fontsize=12)
        ax.set_ylabel('Best Distance', fontsize=12)
        ax.set_title(title, fontsize=14, weight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to {save_path}")
        
        plt.show()


def generate_random_cities(n: int, width: float = 100, height: float = 100) -> List[City]:
    """
    Generate random cities for testing.
    
    Args:
        n: Number of cities to generate
        width: Width of the area
        height: Height of the area
    
    Returns:
        List of randomly placed cities
    """
    cities = []
    for i in range(n):
        x = np.random.uniform(0, width)
        y = np.random.uniform(0, height)
        cities.append(City(x, y, name=f"City_{i}"))
    return cities


def generate_circle_cities(n: int, radius: float = 50, center_x: float = 50, center_y: float = 50) -> List[City]:
    """
    Generate cities arranged in a circle (for testing).
    
    Args:
        n: Number of cities
        radius: Circle radius
        center_x: Circle center X coordinate
        center_y: Circle center Y coordinate
    
    Returns:
        List of cities arranged in a circle
    """
    cities = []
    for i in range(n):
        angle = 2 * np.pi * i / n
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        cities.append(City(x, y, name=f"City_{i}"))
    return cities