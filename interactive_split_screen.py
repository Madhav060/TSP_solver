"""
Interactive TSP Solver with Split Screen Visualization
OPTIMIZED: Fast solving with adaptive iteration counts
"""

# ============================================
# CRITICAL: Set environment variables FIRST
# ============================================
import os
import sys

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'

# ============================================
# STEP 1: Import TensorFlow modules FIRST
# ============================================
print("Loading TensorFlow modules...")
try:
    import tensorflow as tf
    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    
    from drl_solver import DRLSolver
    from hybrid_solver import HybridSolverACO
    DRL_AVAILABLE = True
    print("✓ TensorFlow and DRL modules loaded successfully.")
except Exception as e:
    print(f"✗ TensorFlow/DRL modules failed to load: {e}")
    print("  Continuing with GA and ACO only.")
    DRLSolver = None
    HybridSolverACO = None
    DRL_AVAILABLE = False

# ============================================
# STEP 2: Force matplotlib backend
# ============================================
import matplotlib
matplotlib.use('Agg')

# ============================================
# STEP 3: Import other modules
# ============================================
from tsp_core import City, Tour
from genetic_algorithm import GeneticAlgorithmSolver
from ant_colony import AntColonyOptimizer

import threading
import time
from typing import List, Optional
import queue
import traceback

# ============================================
# STEP 4: NOW import Tkinter (LAST)
# ============================================
print("Initializing Tkinter...")
import tkinter as tk
from tkinter import ttk, messagebox
print("✓ Tkinter initialized successfully.")


class FullScreenTSPSolver:
    """Interactive TSP solver with optimized performance."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("TSP Solver - Interactive Visualization (Optimized)")
        self.root.geometry("1600x1000")
        self.root.configure(bg='#2c3e50')
        
        # Data
        self.cities: List[City] = []
        self.solving = False
        self.drl_solver_instance: Optional[DRLSolver] = None
        self.drl_initial_tour: Optional[Tour] = None
        self.drl_calculation_time: float = 0.0
        
        # Adaptive parameters (will be set when solving starts)
        self.ga_generations = 200
        self.aco_iterations = 200
        self.hybrid_iterations = 100
        
        # Determine available algorithms
        self.available_algorithms = ['ga', 'aco']
        if DRL_AVAILABLE:
            self.available_algorithms.extend(['drl', 'hybrid'])
        
        self.update_queues = {
            'drl': queue.Queue(),
            'ga': queue.Queue(),
            'aco': queue.Queue(),
            'hybrid': queue.Queue()
        }
        
        # Colors
        self.colors = {
            'bg': '#2c3e50',
            'panel_bg': '#34495e',
            'canvas': '#ffffff',
            'city': '#e74c3c',
            'city_outline': '#c0392b',
            'path_drl': '#2ecc71',
            'path_ga': '#3498db',
            'path_aco': '#f39c12',
            'path_hybrid': '#9b59b6',
            'start': '#27ae60',
            'grid': '#ecf0f1',
            'text': '#ecf0f1'
        }
        
        # Setup UI
        self.setup_placement_mode()
        self.schedule_updates()
        
    def setup_placement_mode(self):
        """Create the full-screen city placement UI."""
        self.placement_container = tk.Frame(self.root, bg=self.colors['bg'])
        self.placement_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        title_frame = tk.Frame(self.placement_container, bg=self.colors['bg'])
        title_frame.pack(fill=tk.X, pady=(0, 20))
        
        title = tk.Label(
            title_frame,
            text="🗺️ Traveling Salesman Problem - City Placement",
            font=('Arial', 24, 'bold'),
            bg=self.colors['bg'],
            fg=self.colors['text']
        )
        title.pack()
        
        subtitle_text = "Click anywhere on the canvas to place cities, or use quick actions below"
        if not DRL_AVAILABLE:
            subtitle_text += "\n⚠️ DRL/Hybrid unavailable (running GA + ACO only)"
        
        subtitle = tk.Label(
            title_frame,
            text=subtitle_text,
            font=('Arial', 12),
            bg=self.colors['bg'],
            fg='#95a5a6' if DRL_AVAILABLE else '#e74c3c'
        )
        subtitle.pack()
        
        canvas_frame = tk.Frame(self.placement_container, bg=self.colors['panel_bg'], relief=tk.RIDGE, borderwidth=3)
        canvas_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        self.placement_canvas = tk.Canvas(
            canvas_frame,
            bg=self.colors['canvas'],
            highlightthickness=0
        )
        self.placement_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.placement_canvas.bind('<Button-1>', self.on_canvas_click)
        
        control_frame = tk.Frame(self.placement_container, bg=self.colors['panel_bg'], relief=tk.RIDGE, borderwidth=2)
        control_frame.pack(fill=tk.X)
        
        left_controls = tk.Frame(control_frame, bg=self.colors['panel_bg'])
        left_controls.pack(side=tk.LEFT, padx=20, pady=15)
        
        self.city_label = tk.Label(
            left_controls,
            text="Cities: 0",
            font=('Arial', 18, 'bold'),
            bg=self.colors['panel_bg'],
            fg=self.colors['text']
        )
        self.city_label.pack()
        
        self.status_label = tk.Label(
            left_controls,
            text="Add at least 2 cities to start solving",
            font=('Arial', 11),
            bg=self.colors['panel_bg'],
            fg='#95a5a6'
        )
        self.status_label.pack(pady=(5, 0))
        
        center_controls = tk.Frame(control_frame, bg=self.colors['panel_bg'])
        center_controls.pack(side=tk.LEFT, expand=True, padx=20, pady=15)
        
        button_row = tk.Frame(center_controls, bg=self.colors['panel_bg'])
        button_row.pack()
        
        self.add_random_btn = self.create_button(
            button_row,
            "➕ Add 10 Random Cities",
            self.add_random_cities,
            '#f39c12',
            width=22
        )
        self.add_random_btn.pack(side=tk.LEFT, padx=5)
        
        self.clear_btn = self.create_button(
            button_row,
            "🗑️ Clear All",
            self.clear_all,
            '#e74c3c',
            width=15
        )
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        
        right_controls = tk.Frame(control_frame, bg=self.colors['panel_bg'])
        right_controls.pack(side=tk.RIGHT, padx=20, pady=15)
        
        solve_text = "🚀 SOLVE WITH ALL ALGORITHMS" if DRL_AVAILABLE else "🚀 SOLVE (GA + ACO)"
        self.solve_btn = self.create_button(
            right_controls,
            solve_text,
            self.transition_to_solve_mode,
            '#27ae60',
            width=30,
            height=2
        )
        self.solve_btn.pack()
        
        solve_hint_text = f"Will show {len(self.available_algorithms)} algorithms solving simultaneously"
        solve_hint = tk.Label(
            right_controls,
            text=solve_hint_text,
            font=('Arial', 9),
            bg=self.colors['panel_bg'],
            fg='#95a5a6'
        )
        solve_hint.pack(pady=(5, 0))
    
    def setup_solving_mode(self):
        """Create the split-screen solving UI with adaptive parameters."""
        self.solving_container = tk.Frame(self.root, bg=self.colors['bg'])
        
        top_bar = tk.Frame(self.solving_container, bg=self.colors['panel_bg'], height=60)
        top_bar.pack(fill=tk.X, padx=10, pady=(10, 5))
        top_bar.pack_propagate(False)
        
        title = tk.Label(
            top_bar,
            text=f"🔄 Solving TSP with {len(self.cities)} Cities - Real-Time Comparison",
            font=('Arial', 16, 'bold'),
            bg=self.colors['panel_bg'],
            fg=self.colors['text']
        )
        title.pack(side=tk.LEFT, padx=20, pady=15)
        
        right_info = tk.Frame(top_bar, bg=self.colors['panel_bg'])
        right_info.pack(side=tk.RIGHT, padx=20)
        
        self.solving_status_label = tk.Label(
            right_info,
            text="Solving in progress...",
            font=('Arial', 11),
            bg=self.colors['panel_bg'],
            fg='#95a5a6'
        )
        self.solving_status_label.pack()
        
        self.progress = ttk.Progressbar(
            right_info,
            mode='indeterminate',
            length=300
        )
        self.progress.pack(pady=(5, 0))
        
        self.back_btn = self.create_button(
            top_bar,
            "← Back to Placement",
            self.transition_to_placement_mode,
            '#7f8c8d',
            width=18
        )
        self.back_btn.pack(side=tk.RIGHT, padx=(0, 20))
        self.back_btn.config(state='disabled')
        
        grid_frame = tk.Frame(self.solving_container, bg=self.colors['bg'])
        grid_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5, 10))
        
        # *** ADAPTIVE PARAMETERS based on problem size ***
        n_cities = len(self.cities)
        if n_cities <= 15:
            ga_gen = 200
            aco_iter = 200
            hybrid_iter = 100
        elif n_cities <= 30:
            ga_gen = 500
            aco_iter = 500
            hybrid_iter = 250
        elif n_cities <= 50:
            ga_gen = 800
            aco_iter = 800
            hybrid_iter = 400
        else:
            ga_gen = 1000
            aco_iter = 1000
            hybrid_iter = 500
        
        # Configure grid
        if DRL_AVAILABLE:
            # 2x2 grid
            for i in range(2):
                grid_frame.grid_rowconfigure(i, weight=1)
                grid_frame.grid_columnconfigure(0, weight=1)
                grid_frame.grid_columnconfigure(1, weight=1)
            
            panels = [
                {'name': 'Deep Reinforcement Learning', 'key': 'drl', 'color': self.colors['path_drl'], 'row': 0, 'col': 0},
                {'name': f'Genetic Algorithm ({ga_gen} gen)', 'key': 'ga', 'color': self.colors['path_ga'], 'row': 0, 'col': 1},
                {'name': f'Ant Colony ({aco_iter} iter)', 'key': 'aco', 'color': self.colors['path_aco'], 'row': 1, 'col': 0},
                {'name': f'Hybrid (DRL + ACO {hybrid_iter})', 'key': 'hybrid', 'color': self.colors['path_hybrid'], 'row': 1, 'col': 1}
            ]
        else:
            # 1x2 grid
            grid_frame.grid_rowconfigure(0, weight=1)
            grid_frame.grid_columnconfigure(0, weight=1)
            grid_frame.grid_columnconfigure(1, weight=1)
            
            panels = [
                {'name': f'Genetic Algorithm ({ga_gen} gen)', 'key': 'ga', 'color': self.colors['path_ga'], 'row': 0, 'col': 0},
                {'name': f'Ant Colony ({aco_iter} iter)', 'key': 'aco', 'color': self.colors['path_aco'], 'row': 0, 'col': 1}
            ]
        
        self.algorithm_panels = {}
        
        for panel_info in panels:
            panel = self.create_algorithm_panel(
                grid_frame,
                panel_info['name'],
                panel_info['key'],
                panel_info['color']
            )
            panel.grid(
                row=panel_info['row'],
                column=panel_info['col'],
                padx=5,
                pady=5,
                sticky='nsew'
            )
            self.algorithm_panels[panel_info['key']] = panel
    
    def create_algorithm_panel(self, parent, name, key, color):
        """Create a panel for one algorithm."""
        frame = tk.Frame(parent, bg=self.colors['panel_bg'], relief=tk.RIDGE, borderwidth=3)
        header = tk.Frame(frame, bg=color, height=50)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        title_frame = tk.Frame(header, bg=color)
        title_frame.pack(side=tk.LEFT, padx=15, pady=8)
        title = tk.Label(
            title_frame,
            text=name,
            font=('Arial', 13, 'bold'),
            bg=color,
            fg='white'
        )
        title.pack(anchor='w')
        stats_frame = tk.Frame(header, bg=color)
        stats_frame.pack(side=tk.RIGHT, padx=15, pady=8)
        stats_row1 = tk.Frame(stats_frame, bg=color)
        stats_row1.pack()
        generation_label = tk.Label(
            stats_row1,
            text="Iteration: 0",
            font=('Arial', 10, 'bold'),
            bg=color,
            fg='white'
        )
        generation_label.pack(side=tk.LEFT, padx=8)
        time_label = tk.Label(
            stats_row1,
            text="Time: 0.0s",
            font=('Arial', 10, 'bold'),
            bg=color,
            fg='white'
        )
        time_label.pack(side=tk.LEFT, padx=8)
        stats_row2 = tk.Frame(stats_frame, bg=color)
        stats_row2.pack()
        distance_label = tk.Label(
            stats_row2,
            text="Distance: -",
            font=('Arial', 11, 'bold'),
            bg=color,
            fg='white'
        )
        distance_label.pack()
        canvas = tk.Canvas(
            frame,
            bg=self.colors['canvas'],
            highlightthickness=0
        )
        canvas.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)
        frame.canvas = canvas
        frame.generation_label = generation_label
        frame.distance_label = distance_label
        frame.time_label = time_label
        frame.color = color
        return frame
    
    def create_button(self, parent, text, command, color, width=15, height=1):
        """Create a styled button."""
        btn = tk.Button(
            parent,
            text=text,
            command=command,
            font=('Arial', 10, 'bold'),
            bg=color,
            fg='white',
            activebackground=self.darken_color(color),
            activeforeground='white',
            relief=tk.RAISED,
            borderwidth=2,
            cursor='hand2',
            width=width,
            height=height
        )
        btn.bind('<Enter>', lambda e: btn.config(bg=self.darken_color(color)))
        btn.bind('<Leave>', lambda e: btn.config(bg=color))
        return btn
    
    def darken_color(self, color):
        """Darken a hex color."""
        if color.startswith('#'):
            r = max(0, int(color[1:3], 16) - 30)
            g = max(0, int(color[3:5], 16) - 30)
            b = max(0, int(color[5:7], 16) - 30)
            return f'#{r:02x}{g:02x}{b:02x}'
        return color
    
    def on_canvas_click(self, event):
        """Handle canvas click to add a city."""
        if self.solving:
            return
        x, y = event.x, event.y
        city = City(x, y, f"City_{len(self.cities) + 1}")
        self.cities.append(city)
        self.draw_city_on_placement_canvas(city, len(self.cities))
        self.update_city_count()
    
    def draw_city_on_placement_canvas(self, city: City, index: int):
        """Draw a city on the placement canvas."""
        radius = 8
        x1, y1 = city.x - radius, city.y - radius
        x2, y2 = city.x + radius, city.y + radius
        self.placement_canvas.create_oval(
            x1, y1, x2, y2,
            fill=self.colors['city'],
            outline=self.colors['city_outline'],
            width=2,
            tags='city'
        )
        self.placement_canvas.create_text(
            city.x, city.y,
            text=str(index),
            fill='white',
            font=('Arial', 9, 'bold'),
            tags='city'
        )
    
    def add_random_cities(self):
        """Add 10 random cities."""
        if self.solving:
            return
        width = self.placement_canvas.winfo_width()
        height = self.placement_canvas.winfo_height()
        if width <= 1:
            width = 800
            height = 600
        margin = 50
        import random
        for _ in range(10):
            x = random.randint(margin, width - margin)
            y = random.randint(margin, height - margin)
            city = City(x, y, f"City_{len(self.cities) + 1}")
            self.cities.append(city)
            self.draw_city_on_placement_canvas(city, len(self.cities))
        self.update_city_count()
    
    def clear_all(self):
        """Clear all cities."""
        if self.solving:
            return
        self.cities = []
        self.placement_canvas.delete('all')
        self.update_city_count()
    
    def update_city_count(self):
        """Update city count label."""
        self.city_label.config(text=f"Cities: {len(self.cities)}")
        if len(self.cities) >= 2:
            self.status_label.config(text="Ready to solve! Click the SOLVE button →")
        else:
            self.status_label.config(text="Add at least 2 cities to start solving")

    def transition_to_solve_mode(self):
        """Transition from placement mode to solving mode."""
        if len(self.cities) < 2:
            messagebox.showwarning("Not Enough Cities", "Please add at least 2 cities!")
            return
        if self.solving:
            return
        
        self.placement_container.pack_forget()
        self.setup_solving_mode()
        self.solving_container.pack(fill=tk.BOTH, expand=True)
        
        self.solving = True
        
        if DRL_AVAILABLE:
            self.solving_status_label.config(text="Loading DRL Model...")
            self.progress.start()
            self.back_btn.config(state='disabled')
            self.root.after(100, self.load_model_and_run_drl)
        else:
            self.solving_status_label.config(text="Solving in progress...")
            self.progress.start()
            self.back_btn.config(state='disabled')
            
            for key, panel in self.algorithm_panels.items():
                self.draw_cities_on_panel(panel.canvas)
            
            self.root.after(100, self.start_solver_threads)
    
    def transition_to_placement_mode(self):
        """Transition back to placement mode."""
        if self.solving:
            messagebox.showwarning("Solving in Progress", "Please wait for the solvers to complete.")
            return
        
        if hasattr(self, 'solving_container'):
            self.solving_container.pack_forget()
            self.solving_container.destroy()
        
        self.drl_solver_instance = None
        self.drl_initial_tour = None
        self.drl_calculation_time = 0.0
        
        self.placement_container.pack(fill=tk.BOTH, expand=True)

    def load_model_and_run_drl(self):
        """Load DRL model and run inference in main thread."""
        try:
            print("--- Main Thread: Loading DRL Model... ---")
            
            # 1. Load model (This is the UNTIMED setup cost)
            self.drl_solver_instance = DRLSolver(self.cities)
            print("--- Main Thread: DRL Model Loaded. ---")
            
            print("--- Main Thread: Running DRL inference... ---")
            
            # 2. Time ONLY the inference step
            drl_start_time = time.time() # <--- TIMER STARTS HERE (After loading)
            self.drl_initial_tour = self.drl_solver_instance.solve_fast(use_2opt=True, verbose=False)
            self.drl_calculation_time = time.time() - drl_start_time # <--- TIMER STOPS
            
            print(f"--- Main Thread: DRL tour ready ({self.drl_initial_tour.get_total_distance():.2f}) in {self.drl_calculation_time:.2f}s. ---")
            
        except Exception as e:
            print(f"ERROR: DRL Model/Inference failed: {e}")
            traceback.print_exc()
            
        self.solving_status_label.config(text="Solving in progress...")
        
        for key, panel in self.algorithm_panels.items():
            self.draw_cities_on_panel(panel.canvas)

        self.start_solver_threads()

    def start_solver_threads(self):
        """Launch solver threads with adaptive parameters."""
        print("--- Main Thread: Launching solver threads. ---")
        
        # *** ADAPTIVE PARAMETERS based on problem size ***
        n_cities = len(self.cities)
        
        if n_cities <= 15:
            self.ga_generations = 200
            self.aco_iterations = 200
            self.hybrid_iterations = 100
        elif n_cities <= 30:
            self.ga_generations = 500
            self.aco_iterations = 500
            self.hybrid_iterations = 250
        elif n_cities <= 50:
            self.ga_generations = 800
            self.aco_iterations = 800
            self.hybrid_iterations = 400
        else:
            self.ga_generations = 1000
            self.aco_iterations = 1000
            self.hybrid_iterations = 500
        
        print(f"Problem size: {n_cities} cities")
        print(f"GA generations: {self.ga_generations}")
        print(f"ACO iterations: {self.aco_iterations}")
        if DRL_AVAILABLE:
            print(f"Hybrid ACO iterations: {self.hybrid_iterations}")
        
        threads = []
        threads.append(threading.Thread(target=self._solve_ga, daemon=True))
        threads.append(threading.Thread(target=self._solve_aco, daemon=True))
        
        if DRL_AVAILABLE and self.drl_initial_tour is not None:
            threads.append(threading.Thread(target=self._solve_drl, daemon=True))
            threads.append(threading.Thread(target=self._solve_hybrid, daemon=True))
        
        for thread in threads:
            thread.start()
    
    def _solve_drl(self):
        """DRL solver thread."""
        try:
            print("[Thread DRL] Using pre-calculated DRL tour...")
            
            tour = self.drl_initial_tour.clone()
            elapsed = self.drl_calculation_time
            
            self.update_queues['drl'].put({
                'type': 'solution',
                'tour': tour,
                'generation': 1,
                'distance': tour.get_total_distance(),
                'time': elapsed,
                'current_iter': 1,
                'max_iters': 1
            })
            print(f"[Thread DRL] Finished.")
            
        except Exception as e:
            print(f"DRL Error: {e}")
            traceback.print_exc()
    
    def _solve_ga(self):
        """Genetic Algorithm solver thread - OPTIMIZED."""
        try:
            print("[Thread GA] Starting Genetic Algorithm...")
            solver = GeneticAlgorithmSolver(
                self.cities, population_size=100, mutation_rate=0.015, tournament_size=5
            )
            solver.initialize()
            
            start_time = time.time()
            generations = self.ga_generations  # *** Use adaptive value ***
            
            best_dist_so_far = float('inf')
            best_time_so_far = 0.0
            best_gen_so_far = 0

            def ga_callback(solver_instance):
                nonlocal best_dist_so_far, best_time_so_far, best_gen_so_far
                
                gen = solver_instance.generation
                best_tour_of_gen = solver_instance.get_best_tour()
                current_dist = best_tour_of_gen.get_total_distance()
                elapsed_time = time.time() - start_time
                
                if current_dist < best_dist_so_far:
                    best_dist_so_far = current_dist
                    best_time_so_far = elapsed_time
                    best_gen_so_far = gen

                is_last_generation = (gen == generations)
                
                # *** Update every 10 generations (was 50) ***
                if (gen % 10 == 0) or is_last_generation:
                    self.update_queues['ga'].put({
                        'type': 'solution',
                        'tour': solver_instance.get_best_tour(),
                        'generation': best_gen_so_far,
                        'distance': best_dist_so_far,
                        'time': best_time_so_far,
                        'current_iter': gen,
                        'max_iters': generations
                    })
                    # *** REMOVED: time.sleep(0.05) ***
        
            solver.solve(generations=generations, verbose=False, callback=ga_callback)
            print(f"[Thread GA] Finished in {time.time() - start_time:.2f}s.")

        except Exception as e:
            print(f"GA Error: {e}")
            traceback.print_exc()
    
    def _solve_aco(self):
        """Ant Colony Optimization solver thread - OPTIMIZED."""
        try:
            print("[Thread ACO] Starting Ant Colony...")
            solver = AntColonyOptimizer(self.cities, n_ants=30)
            
            start_time = time.time()
            iterations = self.aco_iterations  # *** Use adaptive value ***
            
            best_dist_so_far = float('inf')
            best_time_so_far = 0.0
            best_iter_so_far = 0

            def aco_callback(solver_instance, iteration):
                nonlocal best_dist_so_far, best_time_so_far, best_iter_so_far
                
                current_dist = solver_instance.best_distance 
                elapsed_time = time.time() - start_time

                if current_dist < best_dist_so_far:
                    best_dist_so_far = current_dist
                    best_time_so_far = elapsed_time
                    best_iter_so_far = iteration + 1

                is_last_iteration = (iteration == iterations - 1)
                
                # *** Update every 10 iterations (was 50) ***
                if ((iteration + 1) % 10 == 0) or is_last_iteration:
                    self.update_queues['aco'].put({
                        'type': 'solution',
                        'tour': solver_instance.get_best_tour(),
                        'generation': best_iter_so_far,
                        'distance': best_dist_so_far,
                        'time': best_time_so_far,
                        'current_iter': iteration + 1, 
                        'max_iters': iterations
                    })
                    # *** REMOVED: time.sleep(0.05) ***
        
            solver.solve(iterations=iterations, verbose=False, callback=aco_callback, seed_tour=None)
            print(f"[Thread ACO] Finished in {time.time() - start_time:.2f}s.")
            
        except Exception as e:
            print(f"ACO Error: {e}")
            traceback.print_exc()

    def _solve_hybrid(self):
        """Hybrid solver thread - OPTIMIZED."""
        try:
            print("[Thread Hybrid] Starting Hybrid (DRL+ACO)...")
            
            solver = HybridSolverACO(self.cities, drl_solver=None)
            start_time = time.time()
            quick_iterations = self.hybrid_iterations  # *** Use adaptive value ***

            initial_tour = self.drl_initial_tour.clone()
            best_dist_so_far = initial_tour.get_total_distance()
            best_time_so_far = time.time() - start_time
            best_iter_so_far = 1
            
            print(f"[Thread Hybrid] Using pre-calculated DRL seed. Starting ACO phase...")

            def hybrid_callback(solver_instance, phase, iteration, tour):
                nonlocal best_dist_so_far, best_time_so_far, best_iter_so_far
                
                elapsed_time = time.time() - start_time
                
                if phase == 'drl':
                    self.update_queues['hybrid'].put({
                        'type': 'solution', 'tour': tour,
                        'generation': 1, 'distance': best_dist_so_far,
                        'time': best_time_so_far, 'phase': phase,
                        'current_iter': 1, 'max_iters': quick_iterations
                    })
                    return

                current_dist = tour.get_total_distance()

                if current_dist < best_dist_so_far:
                    best_dist_so_far = current_dist
                    best_time_so_far = elapsed_time
                    best_iter_so_far = iteration

                is_last_iteration = (iteration == quick_iterations)
                
                # *** Update every 5 iterations (was 25) ***
                if ((iteration % 5 == 0) or is_last_iteration):
                    self.update_queues['hybrid'].put({
                        'type': 'solution',
                        'tour': tour,
                        'generation': best_iter_so_far,
                        'distance': best_dist_so_far,
                        'time': best_time_so_far,
                        'phase': phase,
                        'current_iter': iteration,
                        'max_iters': quick_iterations
                    })
                    # *** REMOVED: time.sleep(0.05) ***
        
            solver.solve_step_by_step(
                quick_iterations=quick_iterations, 
                callback=hybrid_callback,
                initial_tour=initial_tour
            )
            print(f"[Thread Hybrid] Finished in {time.time() - start_time:.2f}s.")

        except Exception as e:
            print(f"Hybrid (ACO) Error: {e}")
            traceback.print_exc()
    
    def schedule_updates(self):
        """Schedule periodic UI updates."""
        self.process_update_queues()
        self.root.after(50, self.schedule_updates)
    
    def process_update_queues(self):
        """Process updates from solver threads."""
        all_threads_appear_done = True
        
        for key, q in self.update_queues.items():
            try:
                while not q.empty():
                    update = q.get_nowait()
                    self.update_algorithm_panel(key, update)
                    all_threads_appear_done = False
            except queue.Empty:
                pass
        
        if self.solving and all_threads_appear_done:
            all_empty = all(q.empty() for q in self.update_queues.values())
            if all_empty:
                if not hasattr(self, '_done_check_time'):
                    self._done_check_time = time.time()
                
                elif time.time() - self._done_check_time > 1.5:  # Reduced from 2.0
                    self.finish_solving()
            else:
                if hasattr(self, '_done_check_time'):
                    delattr(self, '_done_check_time');

    def update_algorithm_panel(self, key: str, update: dict):
        """Update panel with solver progress."""
        if not hasattr(self, 'algorithm_panels') or key not in self.algorithm_panels:
            return
        
        panel = self.algorithm_panels[key]
        
        if 'distance' in update and update['distance'] > 0:
            panel.distance_label.config(text=f"Distance: {update['distance']:.2f}")
        
        if 'time' in update:
            panel.time_label.config(text=f"Best Time: {update['time']:.2f}s")
        
        current_iter = update.get('current_iter', update.get('generation', 1))
        max_iters = update.get('max_iters', 1000)
        best_gen = update.get('generation', 1)

        if key == 'ga':
            gen_text = f"Prog: {current_iter}/{max_iters} (Best: {best_gen})"
        elif key == 'aco':
            gen_text = f"Prog: {current_iter}/{max_iters} (Best: {best_gen})"
        elif key == 'drl':
            gen_text = f"Iter: 1/1 (Complete)"
        elif key == 'hybrid' and 'phase' in update:
            if update['phase'] == 'drl':
                gen_text = f"Phase: DRL (1/1)"
            else:
                max_iters = update.get('max_iters', 500)
                gen_text = f"Prog: {current_iter}/{max_iters} (Best: {best_gen})"
        else:
            gen_text = f"Iteration: {current_iter}"
        
        panel.generation_label.config(text=gen_text)
        
        if update['type'] == 'solution' and 'tour' in update and update['tour'] is not None:
            self.draw_tour_on_panel(panel.canvas, update['tour'], panel.color)
    
    def draw_cities_on_panel(self, canvas: tk.Canvas):
        """Draw all cities on a panel canvas."""
        canvas.update()
        width = canvas.winfo_width()
        height = canvas.winfo_height()
        
        if width <= 1 or height <= 1: return
        if not self.cities: return
        
        min_x = min(city.x for city in self.cities)
        max_x = max(city.x for city in self.cities)
        min_y = min(city.y for city in self.cities)
        max_y = max(city.y for city in self.cities)
        
        padding = 30
        delta_x = max(1, max_x - min_x)
        delta_y = max(1, max_y - min_y)
        
        scale_x = (width - 2 * padding) / delta_x
        scale_y = (height - 2 * padding) / delta_y
        scale = min(scale_x, scale_y)
        
        offset_x = (width - 2 * padding - delta_x * scale) / 2
        offset_y = (height - 2 * padding - delta_y * scale) / 2
        
        for city in self.cities:
            x = padding + offset_x + (city.x - min_x) * scale
            y = padding + offset_y + (city.y - min_y) * scale
            
            radius = 5
            canvas.create_oval(
                x - radius, y - radius, x + radius, y + radius,
                fill=self.colors['city'],
                outline=self.colors['city_outline'],
                width=2,
                tags='city'
            )
    
    def draw_tour_on_panel(self, canvas: tk.Canvas, tour: Tour, color: str):
        """Draw a tour on a panel canvas."""
        canvas.delete('tour')
        
        width = canvas.winfo_width()
        height = canvas.winfo_height()
        
        if width <= 1 or height <= 1 or len(tour.cities) < 2: return
        
        min_x = min(city.x for city in self.cities)
        max_x = max(city.x for city in self.cities)
        min_y = min(city.y for city in self.cities)
        max_y = max(city.y for city in self.cities)
        
        padding = 30
        delta_x = max(1, max_x - min_x)
        delta_y = max(1, max_y - min_y)
        
        scale_x = (width - 2 * padding) / delta_x
        scale_y = (height - 2 * padding) / delta_y
        scale = min(scale_x, scale_y)
        
        offset_x = (width - 2 * padding - delta_x * scale) / 2
        offset_y = (height - 2 * padding - delta_y * scale) / 2
        
        for i in range(len(tour.cities)):
            from_city = tour.cities[i]
            to_city = tour.cities[(i + 1) % len(tour.cities)]
            
            x1 = padding + offset_x + (from_city.x - min_x) * scale
            y1 = padding + offset_y + (from_city.y - min_y) * scale
            x2 = padding + offset_x + (to_city.x - min_x) * scale
            y2 = padding + offset_y + (to_city.y - min_y) * scale
            
            canvas.create_line(
                x1, y1, x2, y2,
                fill=color,
                width=3,
                tags='tour'
            )
        
        canvas.tag_raise('city')
    
    def finish_solving(self):
        """Called when all algorithms are done."""
        print("--- Main Thread: All solvers finished. ---")
        self.solving = False
        self.progress.stop()
        self.solving_status_label.config(text="✓ All algorithms completed! Compare the results.")
        self.back_btn.config(state='normal')
        
        if hasattr(self, '_done_check_time'):
            delattr(self, '_done_check_time')
        
        best_distance = float('inf')
        best_algo = None
        
        for key, panel in self.algorithm_panels.items():
            dist_text = panel.distance_label.cget('text')
            if 'Distance:' in dist_text:
                try:
                    dist = float(dist_text.split(':')[1].strip())
                    if dist > 0 and dist < best_distance:
                        best_distance = dist
                        best_algo = key
                except:
                    pass
        
        if best_algo:
            algo_names = {
                'drl': 'DRL',
                'ga': 'Genetic Algorithm',
                'aco': 'Ant Colony',
                'hybrid': 'Hybrid (DRL+ACO)'
            }
            self.solving_status_label.config(
                text=f"✓ Complete! Best: {algo_names.get(best_algo, best_algo)} with distance {best_distance:.2f}"
            )


def main():
    """Main entry point."""
    print("\n" + "="*60)
    print("TSP Solver - Interactive Visualization (OPTIMIZED)")
    print("="*60)
    if DRL_AVAILABLE:
        print("✓ All algorithms available: DRL, GA, ACO, Hybrid")
    else:
        print("⚠ DRL unavailable - running with GA and ACO only")
    print("="*60 + "\n")
    
    root = tk.Tk()
    app = FullScreenTSPSolver(root)
    root.mainloop()


if __name__ == "__main__":
    main()