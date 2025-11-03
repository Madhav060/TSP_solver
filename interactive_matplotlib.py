"""
Interactive TSP Solver - Matplotlib Version
Click to place cities, then solve using different algorithms!
Works with matplotlib - no additional dependencies needed.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.widgets import Button
import matplotlib.patches as mpatches
import numpy as np
from typing import List, Optional
import time

from tsp_core import City, Tour
from genetic_algorithm import GeneticAlgorithmSolver
from ant_colony import AntColonyOptimizer
from drl_solver import DRLSolver, HybridSolver


class InteractiveTSPMatplotlib:
    """Interactive TSP solver using Matplotlib."""
    
    def __init__(self):
        self.cities: List[City] = []
        self.current_solution: Optional[Tour] = None
        self.solutions = {}
        self.solving = False
        
        # Create figure and axes
        self.fig = plt.figure(figsize=(16, 9))
        self.fig.canvas.manager.set_window_title('Interactive TSP Solver')
        
        # Main canvas for drawing
        self.ax_canvas = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=3)
        self.ax_canvas.set_xlim(0, 100)
        self.ax_canvas.set_ylim(0, 100)
        self.ax_canvas.set_aspect('equal')
        self.ax_canvas.grid(True, alpha=0.3)
        self.ax_canvas.set_title('🗺️ Click to Place Cities', fontsize=14, fontweight='bold')
        
        # Stats panel
        self.ax_stats = plt.subplot2grid((3, 3), (0, 2))
        self.ax_stats.axis('off')
        
        # Results panel
        self.ax_results = plt.subplot2grid((3, 3), (1, 2))
        self.ax_results.axis('off')
        
        # Comparison panel
        self.ax_comparison = plt.subplot2grid((3, 3), (2, 2))
        self.ax_comparison.axis('off')
        
        # Create buttons
        self.create_buttons()
        
        # Connect events
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        # Initial display
        self.update_stats()
        self.update_results()
        
        plt.tight_layout()
        plt.show()
    
    def create_buttons(self):
        """Create control buttons."""
        button_width = 0.15
        button_height = 0.03
        button_spacing = 0.035
        start_x = 0.70
        start_y = 0.60
        
        # Add Random Cities
        ax_random = plt.axes([start_x, start_y, button_width, button_height])
        self.btn_random = Button(ax_random, 'Add 10 Random', color='#f39c12', hovercolor='#e67e22')
        self.btn_random.on_clicked(self.add_random_cities)
        
        # Clear All
        ax_clear = plt.axes([start_x, start_y - button_spacing, button_width, button_height])
        self.btn_clear = Button(ax_clear, 'Clear All', color='#e74c3c', hovercolor='#c0392b')
        self.btn_clear.on_clicked(self.clear_all)
        
        # DRL Fast
        ax_drl = plt.axes([start_x, start_y - 2*button_spacing - 0.02, button_width, button_height])
        self.btn_drl = Button(ax_drl, '⚡ DRL Fast', color='#2ecc71', hovercolor='#27ae60')
        self.btn_drl.on_clicked(lambda event: self.solve_with_algorithm('drl'))
        
        # Genetic Algorithm
        ax_ga = plt.axes([start_x, start_y - 3*button_spacing - 0.02, button_width, button_height])
        self.btn_ga = Button(ax_ga, '🧬 Genetic Alg', color='#3498db', hovercolor='#2980b9')
        self.btn_ga.on_clicked(lambda event: self.solve_with_algorithm('ga'))
        
        # Ant Colony
        ax_aco = plt.axes([start_x, start_y - 4*button_spacing - 0.02, button_width, button_height])
        self.btn_aco = Button(ax_aco, '🐜 Ant Colony', color='#3498db', hovercolor='#2980b9')
        self.btn_aco.on_clicked(lambda event: self.solve_with_algorithm('aco'))
        
        # Hybrid
        ax_hybrid = plt.axes([start_x, start_y - 5*button_spacing - 0.02, button_width, button_height])
        self.btn_hybrid = Button(ax_hybrid, '🚀 Hybrid', color='#2ecc71', hovercolor='#27ae60')
        self.btn_hybrid.on_clicked(lambda event: self.solve_with_algorithm('hybrid'))
        
        # Compare All
        ax_compare = plt.axes([start_x, start_y - 6*button_spacing - 0.04, button_width, button_height])
        self.btn_compare = Button(ax_compare, '📊 Compare All', color='#9b59b6', hovercolor='#8e44ad')
        self.btn_compare.on_clicked(self.compare_all)
    
    def on_click(self, event):
        """Handle mouse click on canvas."""
        if event.inaxes != self.ax_canvas:
            return
        
        if self.solving:
            return
        
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        
        city = City(x, y, f"City_{len(self.cities) + 1}")
        self.cities.append(city)
        
        self.draw_cities()
        self.update_stats()
    
    def draw_cities(self):
        """Draw all cities on the canvas."""
        self.ax_canvas.clear()
        self.ax_canvas.set_xlim(0, 100)
        self.ax_canvas.set_ylim(0, 100)
        self.ax_canvas.set_aspect('equal')
        self.ax_canvas.grid(True, alpha=0.3)
        self.ax_canvas.set_title('🗺️ Click to Place Cities', fontsize=14, fontweight='bold')
        
        # Draw cities
        for i, city in enumerate(self.cities):
            circle = Circle((city.x, city.y), 0.8, color='#e74c3c', ec='#c0392b', linewidth=2, zorder=3)
            self.ax_canvas.add_patch(circle)
            self.ax_canvas.text(city.x, city.y, str(i+1), ha='center', va='center', 
                              color='white', fontsize=8, fontweight='bold', zorder=4)
        
        # Draw tour if exists
        if self.current_solution:
            self.draw_tour(self.current_solution)
        
        self.fig.canvas.draw_idle()
    
    def draw_tour(self, tour: Tour):
        """Draw the tour on the canvas."""
        if len(tour.cities) < 2:
            return
        
        # Draw edges
        for i in range(len(tour.cities)):
            from_city = tour.cities[i]
            to_city = tour.cities[(i + 1) % len(tour.cities)]
            
            self.ax_canvas.plot([from_city.x, to_city.x], 
                              [from_city.y, to_city.y],
                              'b-', linewidth=2, alpha=0.6, zorder=1)
            
            # Draw arrow
            mid_x = (from_city.x + to_city.x) / 2
            mid_y = (from_city.y + to_city.y) / 2
            dx = to_city.x - from_city.x
            dy = to_city.y - from_city.y
            
            arrow = FancyArrowPatch(
                (mid_x - dx*0.05, mid_y - dy*0.05),
                (mid_x + dx*0.05, mid_y + dy*0.05),
                arrowstyle='->', mutation_scale=15, 
                color='blue', linewidth=2, zorder=2
            )
            self.ax_canvas.add_patch(arrow)
        
        # Highlight start city
        start = tour.cities[0]
        star = Circle((start.x, start.y), 1.2, color='#2ecc71', ec='#27ae60', linewidth=2, zorder=5)
        self.ax_canvas.add_patch(star)
        self.ax_canvas.text(start.x, start.y, '★', ha='center', va='center',
                          color='white', fontsize=14, fontweight='bold', zorder=6)
    
    def add_random_cities(self, event):
        """Add 10 random cities."""
        if self.solving:
            return
        
        for _ in range(10):
            x = np.random.uniform(5, 95)
            y = np.random.uniform(5, 95)
            city = City(x, y, f"City_{len(self.cities) + 1}")
            self.cities.append(city)
        
        self.draw_cities()
        self.update_stats()
    
    def clear_all(self, event):
        """Clear all cities and solutions."""
        if self.solving:
            return
        
        self.cities = []
        self.current_solution = None
        self.solutions = {}
        self.draw_cities()
        self.update_stats()
        self.update_results()
        self.update_comparison()
    
    def update_stats(self):
        """Update statistics panel."""
        self.ax_stats.clear()
        self.ax_stats.axis('off')
        
        text = "PROBLEM INFO\n"
        text += "=" * 25 + "\n\n"
        text += f"Cities: {len(self.cities)}\n\n"
        
        if self.solving:
            text += "Status: Solving...\n"
        else:
            text += "Status: Ready\n"
        
        self.ax_stats.text(0.1, 0.5, text, fontsize=10, verticalalignment='center',
                          fontfamily='monospace', transform=self.ax_stats.transAxes)
    
    def update_results(self, algorithm: str = "None", tour: Optional[Tour] = None, time_taken: float = 0):
        """Update results panel."""
        self.ax_results.clear()
        self.ax_results.axis('off')
        
        text = "CURRENT SOLUTION\n"
        text += "=" * 25 + "\n\n"
        text += f"Algorithm:\n  {algorithm}\n\n"
        
        if tour:
            distance = tour.get_total_distance()
            text += f"Distance:\n  {distance:.2f}\n\n"
            text += f"Time:\n  {time_taken:.3f}s\n"
        else:
            text += "Distance:\n  -\n\n"
            text += "Time:\n  -\n"
        
        self.ax_results.text(0.1, 0.5, text, fontsize=10, verticalalignment='center',
                           fontfamily='monospace', transform=self.ax_results.transAxes)
        self.fig.canvas.draw_idle()
    
    def update_comparison(self, results: dict = None):
        """Update comparison panel."""
        self.ax_comparison.clear()
        self.ax_comparison.axis('off')
        
        if not results:
            text = "COMPARISON\n"
            text += "=" * 25 + "\n\n"
            text += "Run 'Compare All'\nto see results"
            self.ax_comparison.text(0.1, 0.5, text, fontsize=10, verticalalignment='center',
                                  fontfamily='monospace', transform=self.ax_comparison.transAxes)
            return
        
        text = "COMPARISON\n"
        text += "=" * 25 + "\n\n"
        
        for algo, data in sorted(results.items(), key=lambda x: x[1]['distance']):
            name = data['name'][:12]
            distance = data['distance']
            time_val = data['time']
            marker = "★" if algo == min(results.items(), key=lambda x: x[1]['distance'])[0] else " "
            
            text += f"{marker} {name}\n"
            text += f"  D:{distance:.1f} T:{time_val:.2f}s\n"
        
        self.ax_comparison.text(0.1, 0.5, text, fontsize=9, verticalalignment='center',
                              fontfamily='monospace', transform=self.ax_comparison.transAxes)
        self.fig.canvas.draw_idle()
    
    def solve_with_algorithm(self, algorithm: str):
        """Solve with specified algorithm."""
        if len(self.cities) < 2:
            print("Please add at least 2 cities!")
            return
        
        if self.solving:
            print("Already solving...")
            return
        
        self.solving = True
        self.update_stats()
        plt.pause(0.01)
        
        algo_names = {
            'drl': 'DRL Fast',
            'ga': 'Genetic Algorithm',
            'aco': 'Ant Colony',
            'hybrid': 'Hybrid (DRL + GA)'
        }
        
        print(f"Solving with {algo_names[algorithm]}...")
        
        start_time = time.time()
        
        try:
            if algorithm == 'drl':
                solver = DRLSolver(self.cities)
                tour = solver.solve_fast(use_2opt=True, verbose=False)
            
            elif algorithm == 'ga':
                solver = GeneticAlgorithmSolver(
                    self.cities,
                    population_size=100,
                    mutation_rate=0.015,
                    tournament_size=5
                )
                solver.initialize()
                tour = solver.solve(generations=300, verbose=False)
            
            elif algorithm == 'aco':
                solver = AntColonyOptimizer(
                    self.cities,
                    n_ants=30,
                    n_iterations=200
                )
                tour = solver.solve(verbose=False)
            
            elif algorithm == 'hybrid':
                solver = HybridSolver(self.cities)
                tour = solver.solve(quick_generations=200, verbose=False)
            
            elapsed_time = time.time() - start_time
            
            self.current_solution = tour
            self.solutions[algorithm] = {
                'name': algo_names[algorithm],
                'tour': tour,
                'time': elapsed_time,
                'distance': tour.get_total_distance()
            }
            
            self.draw_cities()
            self.update_results(algo_names[algorithm], tour, elapsed_time)
            
            print(f"✓ Completed! Distance: {tour.get_total_distance():.2f}, Time: {elapsed_time:.3f}s")
        
        except Exception as e:
            print(f"Error: {e}")
        
        finally:
            self.solving = False
            self.update_stats()
    
    def compare_all(self, event):
        """Compare all algorithms."""
        if len(self.cities) < 2:
            print("Please add at least 2 cities!")
            return
        
        if self.solving:
            print("Already solving...")
            return
        
        self.solving = True
        self.update_stats()
        plt.pause(0.01)
        
        print("\n" + "="*50)
        print("COMPARING ALL ALGORITHMS")
        print("="*50)
        
        algorithms = ['drl', 'ga', 'aco', 'hybrid']
        results = {}
        
        for algo in algorithms:
            algo_names = {
                'drl': 'DRL Fast',
                'ga': 'Genetic Algorithm',
                'aco': 'Ant Colony',
                'hybrid': 'Hybrid (DRL + GA)'
            }
            
            print(f"\nRunning {algo_names[algo]}...")
            
            start_time = time.time()
            
            try:
                if algo == 'drl':
                    solver = DRLSolver(self.cities)
                    tour = solver.solve_fast(use_2opt=True, verbose=False)
                elif algo == 'ga':
                    solver = GeneticAlgorithmSolver(self.cities, population_size=50)
                    solver.initialize()
                    tour = solver.solve(generations=200, verbose=False)
                elif algo == 'aco':
                    solver = AntColonyOptimizer(self.cities, n_ants=30, n_iterations=150)
                    tour = solver.solve(verbose=False)
                elif algo == 'hybrid':
                    solver = HybridSolver(self.cities)
                    tour = solver.solve(quick_generations=150, verbose=False)
                
                elapsed_time = time.time() - start_time
                
                results[algo] = {
                    'name': algo_names[algo],
                    'tour': tour,
                    'time': elapsed_time,
                    'distance': tour.get_total_distance()
                }
                
                print(f"  Distance: {tour.get_total_distance():.2f}, Time: {elapsed_time:.3f}s")
            
            except Exception as e:
                print(f"  Error: {e}")
        
        # Find best
        if results:
            best_algo = min(results.items(), key=lambda x: x[1]['distance'])
            best_tour = best_algo[1]['tour']
            
            self.current_solution = best_tour
            self.solutions = results
            
            self.draw_cities()
            self.update_results(best_algo[1]['name'], best_tour, best_algo[1]['time'])
            self.update_comparison(results)
            
            print("\n" + "="*50)
            print("RESULTS SUMMARY")
            print("="*50)
            for algo, data in sorted(results.items(), key=lambda x: x[1]['distance']):
                marker = "★" if algo == best_algo[0] else " "
                print(f"{marker} {data['name']:<20} Distance: {data['distance']:<8.2f} Time: {data['time']:.3f}s")
            print("="*50 + "\n")
        
        self.solving = False
        self.update_stats()


def main():
    """Main entry point."""
    print("="*60)
    print("INTERACTIVE TSP SOLVER - Matplotlib Version")
    print("="*60)
    print("\nInstructions:")
    print("1. Click on the canvas to add cities")
    print("2. Use buttons to solve or compare algorithms")
    print("3. Close the window to exit")
    print("\n" + "="*60 + "\n")
    
    app = InteractiveTSPMatplotlib()


if __name__ == "__main__":
    main()