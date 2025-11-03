"""
Simple Example - Quick Start Guide
Run this to see the TSP solver in action!
"""

from tsp_core import City
from genetic_algorithm import GeneticAlgorithmSolver
from visualization import TSPVisualizer, generate_random_cities

def simple_example():
    """Simple example showing basic usage."""
    print("\n" + "="*60)
    print("TSP SOLVER - SIMPLE EXAMPLE")
    print("="*60 + "\n")
    
    # Generate 20 random cities
    print("Step 1: Generating 20 random cities...")
    cities = generate_random_cities(20, width=100, height=100)
    print(f"✓ Created {len(cities)} cities\n")
    
    # Create and run Genetic Algorithm solver
    print("Step 2: Creating Genetic Algorithm solver...")
    solver = GeneticAlgorithmSolver(
        cities=cities,
        population_size=50,
        mutation_rate=0.015,
        tournament_size=5
    )
    print("✓ Solver created\n")
    
    # Initialize population
    print("Step 3: Initializing random population...")
    solver.initialize()
    initial_best = solver.get_best_tour()
    print(f"✓ Initial best distance: {initial_best.get_total_distance():.2f}\n")
    
    # Solve
    print("Step 4: Running evolution (500 generations)...")
    best_tour = solver.solve(generations=500, verbose=False)
    print(f"✓ Final best distance: {best_tour.get_total_distance():.2f}\n")
    
    # Calculate improvement
    improvement = ((initial_best.get_total_distance() - best_tour.get_total_distance()) 
                   / initial_best.get_total_distance()) * 100
    
    print("="*60)
    print("RESULTS")
    print("="*60)
    print(f"Initial Distance: {initial_best.get_total_distance():.2f}")
    print(f"Final Distance:   {best_tour.get_total_distance():.2f}")
    print(f"Improvement:      {improvement:.2f}%")
    print(f"Generations:      500")
    print("="*60 + "\n")
    
    # Visualize
    print("Step 5: Creating visualization...")
    visualizer = TSPVisualizer()
    visualizer.plot_tour(best_tour, title="Simple Example Solution")
    visualizer.plot_convergence(
        solver.best_distance_history,
        title="Optimization Progress"
    )
    print("✓ Done!\n")


def manual_cities_example():
    """Example with manually defined cities."""
    print("\n" + "="*60)
    print("MANUAL CITIES EXAMPLE")
    print("="*60 + "\n")
    
    # Define specific cities
    cities = [
        City(10, 20, "City A"),
        City(30, 40, "City B"),
        City(50, 25, "City C"),
        City(70, 60, "City D"),
        City(40, 80, "City E"),
        City(90, 30, "City F"),
        City(20, 70, "City G"),
        City(60, 10, "City H")
    ]
    
    print(f"Created {len(cities)} cities manually:")
    for city in cities:
        print(f"  - {city.name}: ({city.x:.1f}, {city.y:.1f})")
    
    print("\nSolving with Genetic Algorithm...")
    solver = GeneticAlgorithmSolver(cities, population_size=50)
    solver.initialize()
    best_tour = solver.solve(generations=300, verbose=False)
    
    print(f"\nBest distance found: {best_tour.get_total_distance():.2f}")
    print("\nTour order:")
    for i, city in enumerate(best_tour.cities, 1):
        print(f"  {i}. {city.name}")
    
    # Visualize
    visualizer = TSPVisualizer()
    visualizer.plot_tour(best_tour, title="Manual Cities Solution")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("TSP SOLVER - QUICK START EXAMPLES")
    print("="*60)
    print("\nThis script demonstrates basic usage of the TSP solver.")
    print("\nChoose an example:")
    print("  1. Simple random cities example")
    print("  2. Manual cities example")
    print("  3. Run both\n")
    
    choice = input("Enter choice (1-3) [default: 1]: ").strip() or "1"
    
    if choice == "1":
        simple_example()
    elif choice == "2":
        manual_cities_example()
    elif choice == "3":
        simple_example()
        manual_cities_example()
    else:
        print("Invalid choice. Running simple example...")
        simple_example()
    
    print("\n" + "="*60)
    print("Try running the full benchmark:")
    print("  python main.py --benchmark --cities 30")
    print("="*60 + "\n")