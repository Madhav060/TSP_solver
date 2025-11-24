import numpy as np
import tensorflow as tf
from typing import List
import random
import os
import time

from tsp_core import City, Tour, DistanceMatrix
from attention_model import AttentionModelTF


# --------------------------------------------------------
# GLOBAL MODEL CACHE (important)
# --------------------------------------------------------
GLOBAL_DRL_MODEL = None


class DRLSolver:
    """
    DRL Solver now:
    - loads model ONCE globally
    - supports time-limit + logging
    - returns (tour, log) always
    """

    def __init__(self, cities: List[City], model_path: str = "tsp_checkpoints"):
        self.cities = cities
        self.n_cities = len(cities)
        self.model_path = model_path
        self.distance_matrix = DistanceMatrix(cities)

        global GLOBAL_DRL_MODEL
        self.model = None

        if GLOBAL_DRL_MODEL is not None:
            self.model = GLOBAL_DRL_MODEL
            return

        if not (model_path and os.path.exists(model_path)):
            print("[DRL] No trained model found → using heuristic.")
            return

        try:
            self.model = self._load_model(model_path)
            GLOBAL_DRL_MODEL = self.model
        except Exception as e:
            print(f"[DRL] Model load failed: {e}")
            print("[DRL] Falling back to heuristic.")

    # --------------------------------------------------------
    def _load_model(self, model_path):
        print(f"[DRL] Loading model from {model_path} ...")

        model = AttentionModelTF(
            embed_dim=128,
            num_encoder_layers=3,
            num_heads=8,
            ff_dim=512
        )

        dummy = tf.random.uniform((1, self.n_cities, 2))
        _ = model(dummy, return_log_probs=False)

        if os.path.isdir(model_path):
            ckpt = tf.train.latest_checkpoint(model_path)
            if ckpt is None:
                raise FileNotFoundError("No checkpoint found.")
            print(f"[DRL] Using checkpoint: {ckpt}")
            model.load_weights(ckpt)
        else:
            model.load_weights(model_path)

        print("[DRL] Model ready.")
        return model

    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------
    def _predict_with_model(self) -> Tour:
        coords = np.array([[c.x, c.y] for c in self.cities], dtype=np.float32)
        coords = coords.reshape(1, self.n_cities, 2)
        tensor = tf.convert_to_tensor(coords)

        tour_idx, _, _ = self.model(tensor, return_log_probs=False, training=False)
        order = tour_idx.numpy()[0]
        return Tour([self.cities[i] for i in order])

    # --------------------------------------------------------
    # HEURISTIC
    # --------------------------------------------------------
    def nearest_neighbor_heuristic(self, s=0) -> Tour:
        n = len(self.cities)
        unvisited = set(range(n))
        cur = s
        order = [cur]
        unvisited.remove(cur)

        while unvisited:
            nxt = min(unvisited,
                      key=lambda i: self.distance_matrix.get_distance_by_index(cur, i))
            unvisited.remove(nxt)
            order.append(nxt)
            cur = nxt

        return Tour([self.cities[i] for i in order])

    # --------------------------------------------------------
    # TIME-LIMITED 2-OPT
    # --------------------------------------------------------
    def two_opt_timed(self, tour: Tour, time_limit, t0):
        log = []
        best = tour
        best_d = best.get_total_distance()
        log.append((0.0, best_d))

        n = len(tour.cities)
        improved = True

        while improved:
            improved = False

            if time_limit and time.time() - t0 >= time_limit:
                break

            for i in range(1, n - 2):
                for j in range(i + 1, n - 1):

                    if time_limit and time.time() - t0 >= time_limit:
                        return best, log

                    new = best.clone()
                    new.cities[i:j+1] = reversed(new.cities[i:j+1])
                    new.invalidate_cache()
                    d = new.get_total_distance()

                    if d < best_d:
                        best = new
                        best_d = d
                        log.append((time.time() - t0, best_d))
                        improved = True
                        break
                if improved:
                    break

        return best, log

    # --------------------------------------------------------
    # MAIN FAST SOLVER
    # --------------------------------------------------------
    def solve_fast(self, use_2opt=True, verbose=False, time_limit=None):
        """
        ALWAYS returns:
            tour, log
        """
        t0 = time.time()
        log = []

        # ------------- initial solution -------------
        if self.model:
            tour = self._predict_with_model()
        else:
            best = None
            best_d = float("inf")
            for start in random.sample(range(self.n_cities), min(5, self.n_cities)):
                t = self.nearest_neighbor_heuristic(start)
                d = t.get_total_distance()
                if d < best_d:
                    best = t
                    best_d = d
            tour = best

        init_d = tour.get_total_distance()
        log.append((0.0, init_d))

        # ------------- 2-opt refinement -------------
        if use_2opt:
            tour, log2 = self.two_opt_timed(tour, time_limit, t0)
            log.extend(log2)

        # clip by time
        if time_limit:
            log = [(t, d) for (t, d) in log if t <= time_limit]

        return tour, log
