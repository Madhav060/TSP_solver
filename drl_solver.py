import numpy as np
import tensorflow as tf
from typing import List
import random
import os # Import os to check for file existence

from tsp_core import City, Tour, DistanceMatrix
from attention_model import AttentionModelTF # Import your TF model

class DRLSolver:
    """
    Deep Reinforcement Learning Solver for TSP.
    
    This version loads a pre-trained TensorFlow Attention Model.
    If no model is found, it falls back to a heuristic.
    """
    
    def __init__(
        self,
        cities: List[City],
        # *** This path is correct ***
        model_path: str = "tsp_checkpoints"
    ):
        self.cities = cities
        self.n_cities = len(cities)
        self.distance_matrix = DistanceMatrix(cities)
        self.model_path = model_path
        self.model = None # This will hold the trained Keras model
        
        if model_path and os.path.exists(model_path):
            try:
                self._load_model(model_path)
            except Exception as e:
                print(f"[DRL Solver Warning] Found model path but failed to load: {e}")
                print("Falling back to heuristic.")
        else:
            print("[DRL Solver Info] No trained model file found. Using heuristic fallback.")

    def _load_model(self, model_path: str):
        """
        Load a pre-trained TensorFlow/Keras model weights.
        """
        print(f"[DRL Solver Info] Loading trained model from: {model_path} ...")
        
        # 1. Instantiate the model architecture
        # These parameters must match the ones used during training
        self.model = AttentionModelTF(
            embed_dim=128,
            num_encoder_layers=3,
            num_heads=8,
            ff_dim=512
        )
        
        # 2. Build the model by passing a sample input
        # Use n_cities from *this* problem instance to build
        dummy_input = tf.random.uniform((1, self.n_cities, 2))
        _ = self.model(dummy_input, return_log_probs=False)
        
        # 3. Load the saved weights
        # ==================== THE FIX ====================
        # Check if model_path is a directory (like 'tsp_checkpoints')
        if os.path.isdir(model_path):
            # If it is, find the latest checkpoint file *inside* it
            latest_ckpt = tf.train.latest_checkpoint(model_path)
            if latest_ckpt is None:
                raise FileNotFoundError(f"No checkpoint found in directory: {model_path}")
            print(f"[DRL Solver Info] Found latest checkpoint: {latest_ckpt}")
            self.model.load_weights(latest_ckpt)
        else:
            # Otherwise, assume it's a single file (like .h5) and load directly
            self.model.load_weights(model_path)
        # ===============================================
        
        print("[DRL Solver Info] Model loaded successfully.")
    
    def _predict_with_model(self) -> Tour:
        """
        Use the trained neural network model to predict a tour.
        """
        # 1. Format cities into a NumPy array
        # Shape: (1, n_cities, 2) - Add a batch dimension of 1
        city_coords = np.array(
            [[city.x, city.y] for city in self.cities],
            dtype=np.float32
        ).reshape(1, self.n_cities, 2)

        # 2. Convert to TensorFlow tensor
        inputs_tensor = tf.convert_to_tensor(city_coords)

        # 3. Run inference
        # We use return_log_probs=False to get the tour indices
        tour_indices_tensor, tour_length_tensor, _ = self.model(
            inputs_tensor,
            return_log_probs=False,
            training=False # Set model to inference mode
        )

        # 4. Convert tensor result back to a list of indices
        # Squeeze batch dimension [0]
        tour_indices = tour_indices_tensor.numpy()[0]
        
        # 5. Create a Tour object
        tour_cities = [self.cities[idx] for idx in tour_indices]
        return Tour(tour_cities)

    # --- Heuristic and 2-Opt functions remain unchanged ---

    def nearest_neighbor_heuristic(self, start_index: int = 0) -> Tour:
        """
        Nearest Neighbor Heuristic: A fast greedy algorithm.
        (Fallback if no model is loaded)
        """
        n = len(self.cities)
        unvisited = set(range(n))
        
        current_idx = start_index
        tour_indices = [current_idx]
        unvisited.remove(current_idx)
        
        while unvisited:
            nearest_idx = min(
                unvisited,
                key=lambda idx: self.distance_matrix.get_distance_by_index(current_idx, idx)
            )
            tour_indices.append(nearest_idx)
            unvisited.remove(nearest_idx)
            current_idx = nearest_idx
        
        tour_cities = [self.cities[idx] for idx in tour_indices]
        return Tour(tour_cities)
    
    def two_opt_improvement(self, tour: Tour, max_iterations: int = 100) -> Tour:
        """
        2-opt local search: Improve a tour by swapping edges.
        """
        improved_tour = tour.clone()
        improved = True
        iteration = 0
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            for i in range(1, len(improved_tour.cities) - 1):
                for j in range(i + 1, len(improved_tour.cities)):
                    new_tour = improved_tour.clone()
                    new_tour.cities[i:j+1] = reversed(new_tour.cities[i:j+1])
                    new_tour.invalidate_cache()
                    
                    if new_tour.get_total_distance() < improved_tour.get_total_distance():
                        improved_tour = new_tour
                        improved = True
                        break
                
                if improved:
                    break
        
        return improved_tour
    
    def solve_fast(self, use_2opt: bool = True, verbose: bool = True) -> Tour:
        """
        Fast "instant" solution using the trained DRL model or heuristic fallback.
        """
        if verbose:
            print(f"\n{'='*60}")
            if self.model:
                print(f"DRL Fast Solver (Trained Model Mode)")
            else:
                print(f"DRL Fast Solver (Heuristic Fallback Mode)")
            print(f"{'='*60}")
        
        if self.model:
            # *** USE THE TRAINED MODEL ***
            tour = self._predict_with_model()
        else:
            # *** FALLBACK TO HEURISTIC ***
            if verbose:
                print("Using Nearest Neighbor Heuristic...")
            
            best_tour = None
            best_distance = float('inf')
            start_points = random.sample(range(len(self.cities)), min(5, len(self.cities)))
            
            for start_idx in start_points:
                h_tour = self.nearest_neighbor_heuristic(start_idx)
                distance = h_tour.get_total_distance()
                
                if distance < best_distance:
                    best_distance = distance
                    best_tour = h_tour
            tour = best_tour
        
        initial_distance = tour.get_total_distance()
        
        # 2-Opt is still very useful! The model gives a great starting
        # point, and 2-Opt finishes the job.
        if use_2opt:
            if verbose:
                print("Applying 2-opt local search...")
            tour = self.two_opt_improvement(tour)
        
        if verbose:
            final_distance = tour.get_total_distance()
            print(f"\nInitial Distance: {initial_distance:.2f}")
            if use_2opt:
                improvement = ((initial_distance - final_distance) / initial_distance) * 100
                print(f"After 2-opt: {final_distance:.2f} ({improvement:.2f}% improvement)")
            print(f"{'='*60}\n")
        
        return tour