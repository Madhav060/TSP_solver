"""
Fixed Model Validation - Works with your DRLSolver
"""

import os
import sys
import numpy as np
import tensorflow as tf
import time
from tqdm import tqdm

from tsp_core import City, Tour
from drl_solver import DRLSolver

class TSPModelValidator:
    """Validator for trained TSP models."""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.results = {'tests_passed': [], 'tests_failed': [], 'metrics': {}}
        
    def run_all_tests(self):
        print("\n" + "="*70)
        print("🔍 TSP MODEL VALIDATION SUITE")
        print("="*70)
        print(f"Model: {self.model_path}\n")
        
        tests = [
            ("File Existence", self.test_1_file_exists),
            ("Model Loading", self.test_2_model_loads),
            ("Forward Pass", self.test_3_forward_pass),
            ("Tour Validity", self.test_4_valid_tours),
            ("Solution Quality", self.test_5_solution_quality),
        ]
        
        for i, (name, test_func) in enumerate(tests, 1):
            print(f"\n{'─'*70}")
            print(f"[Test {i}/{len(tests)}] {name}")
            print('─'*70)
            
            try:
                passed = test_func()
                if passed:
                    self.results['tests_passed'].append(name)
                    print(f"✅ PASSED: {name}")
                else:
                    self.results['tests_failed'].append(name)
                    print(f"❌ FAILED: {name}")
                    if name in ["File Existence", "Model Loading"]:
                        break
            except Exception as e:
                self.results['tests_failed'].append(name)
                print(f"❌ FAILED: {name} - {e}")
                import traceback
                traceback.print_exc()
        
        self.generate_report()
        return len(self.results['tests_failed']) == 0
    
    def test_1_file_exists(self):
        if not os.path.exists(self.model_path):
            print(f"❌ Model not found: {self.model_path}")
            return False
        
        file_size = os.path.getsize(self.model_path) / (1024 * 1024)
        print(f"✓ Model file found: {file_size:.2f} MB")
        self.results['metrics']['file_size_mb'] = file_size
        return True
    
    def test_2_model_loads(self):
        try:
            dummy_cities = [City(np.random.rand()*100, np.random.rand()*100, f"C{i}") 
                          for i in range(20)]
            
            self.test_solver = DRLSolver(dummy_cities, model_path=self.model_path)
            
            if self.test_solver.model is None:
                print("❌ Model is None after loading")
                return False
            
            print(f"✓ Model loaded successfully")
            print(f"✓ Model type: {type(self.test_solver.model).__name__}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load: {e}")
            return False
    
    def test_3_forward_pass(self):
        try:
            n_cities = 25
            test_cities = [City(np.random.rand()*100, np.random.rand()*100, f"C{i}") 
                          for i in range(n_cities)]
            
            start = time.time()
            tour = self.test_solver.solve_fast(use_2opt=False, verbose=False)
            elapsed = time.time() - start
            
            if tour is None:
                print("❌ Returned None")
                return False
            
            print(f"✓ Forward pass successful ({elapsed*1000:.1f}ms)")
            print(f"✓ Tour: {len(tour.cities)} cities, length {tour.get_total_distance():.2f}")
            self.results['metrics']['inference_time_ms'] = elapsed * 1000
            return True
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            return False
    
    def test_4_valid_tours(self):
        print("Testing tour validity on 20 problems...")
        
        n_tests = 20
        invalid = 0
        
        for i in tqdm(range(n_tests), desc="Validating"):
            n_cities = np.random.randint(15, 40)
            cities = [City(np.random.rand()*100, np.random.rand()*100, f"C{j}") 
                     for j in range(n_cities)]
            
            solver = DRLSolver(cities, model_path=self.model_path)
            tour = solver.solve_fast(use_2opt=False, verbose=False)
            
            if len(tour.cities) != n_cities:
                invalid += 1
                continue
            
            tour_length = tour.get_total_distance()
            if not (0 < tour_length < float('inf')):
                invalid += 1
        
        valid = n_tests - invalid
        print(f"✓ Valid tours: {valid}/{n_tests} ({valid/n_tests*100:.1f}%)")
        
        self.results['metrics']['validity_rate'] = valid / n_tests
        return valid == n_tests
    
    def test_5_solution_quality(self):
        print("Comparing solution quality (10 problems, 30 cities)...")
        
        n_tests = 10
        drl_lengths = []
        nn_lengths = []
        
        for i in tqdm(range(n_tests), desc="Quality test"):
            cities = [City(np.random.rand()*100, np.random.rand()*100, f"C{j}") 
                     for j in range(30)]
            
            solver = DRLSolver(cities, model_path=self.model_path)
            
            # DRL solution
            drl_tour = solver.solve_fast(use_2opt=False, verbose=False)
            drl_lengths.append(drl_tour.get_total_distance())
            
            # Nearest Neighbor baseline
            nn_tour = solver.nearest_neighbor_heuristic(start_index=0)
            nn_lengths.append(nn_tour.get_total_distance())
        
        drl_avg = np.mean(drl_lengths)
        nn_avg = np.mean(nn_lengths)
        improvement = ((nn_avg - drl_avg) / nn_avg) * 100
        
        print(f"\nResults:")
        print(f"  DRL Model:        {drl_avg:.2f}")
        print(f"  Nearest Neighbor: {nn_avg:.2f}")
        print(f"  Improvement:      {improvement:+.2f}%")
        
        if improvement > 0:
            print(f"✅ Model BEATS Nearest Neighbor!")
            quality = "Excellent"
        elif improvement > -10:
            print(f"✓ Model is competitive")
            quality = "Good"
        else:
            print(f"⚠ Model worse than baseline")
            quality = "Needs improvement"
        
        self.results['metrics']['improvement_vs_nn'] = improvement
        self.results['metrics']['quality'] = quality
        
        return improvement > -10
    
    def generate_report(self):
        print("\n" + "="*70)
        print("📊 VALIDATION REPORT")
        print("="*70)
        
        total = len(self.results['tests_passed']) + len(self.results['tests_failed'])
        passed = len(self.results['tests_passed'])
        
        print(f"\nTests Passed: {passed}/{total}")
        
        if self.results['tests_passed']:
            print(f"\n✅ Passed:")
            for t in self.results['tests_passed']:
                print(f"   • {t}")
        
        if self.results['tests_failed']:
            print(f"\n❌ Failed:")
            for t in self.results['tests_failed']:
                print(f"   • {t}")
        
        if self.results['metrics']:
            print(f"\n📈 Metrics:")
            for k, v in self.results['metrics'].items():
                if isinstance(v, float):
                    print(f"   • {k}: {v:.2f}")
                else:
                    print(f"   • {k}: {v}")
        
        print("\n" + "="*70)
        if len(self.results['tests_failed']) == 0:
            print("🎉 ALL TESTS PASSED!")
            print("Your model is working correctly and ready to use!")
        else:
            print("⚠️  VALIDATION COMPLETED WITH ISSUES")
        print("="*70 + "\n")


def main():
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "trained_models_tf/tsp_2_50_attention_model_tf.weights.h5"
    
    validator = TSPModelValidator(model_path)
    success = validator.run_all_tests()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()