"""
Fixed Model Validation - Works with DRLSolver and Checkpoint Directories
"""

import os
import sys
import numpy as np
import tensorflow as tf
import time
from tqdm import tqdm

# Suppress TensorFlow warnings for cleaner validation output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tsp_core import City, Tour
from drl_solver import DRLSolver

class TSPModelValidator:
    """Validator for trained TSP models."""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.results = {'tests_passed': [], 'tests_failed': [], 'metrics': {}}
        self.test_solver = None # To be initialized in Test 2
        
    def run_all_tests(self):
        print("\n" + "="*70)
        print("🔍 TSP MODEL VALIDATION SUITE (RIGOROUS)")
        print("="*70)
        print(f"Model Path: {self.model_path}\n")
        
        tests = [
            ("File/Directory Existence", self.test_1_file_exists),
            ("Model Loading", self.test_2_model_loads),
            ("Forward Pass", self.test_3_forward_pass),
            ("Tour Validity (100 Tests, 10-60 Cities)", self.test_4_valid_tours),
            ("Solution Quality (50 Tests, 50 Cities)", self.test_5_solution_quality),
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
                    # Stop if basic loading fails
                    if name in ["File/Directory Existence", "Model Loading"]:
                        print("\nAborting further tests: Basic loading failed.")
                        break
            except Exception as e:
                self.results['tests_failed'].append(name)
                print(f"❌ FAILED: {name} - An exception occurred: {e}")
                import traceback
                traceback.print_exc()
                if name in ["File/Directory Existence", "Model Loading"]:
                    print("\nAborting further tests: Basic loading failed.")
                    break
        
        self.generate_report()
        return len(self.results['tests_failed']) == 0
    
    def test_1_file_exists(self):
        """Test if the model path exists (file or directory)."""
        if not os.path.exists(self.model_path):
            print(f"❌ Model path not found: {self.model_path}")
            return False
        
        if os.path.isdir(self.model_path):
            print(f"✓ Model checkpoint directory found.")
            total_size = 0
            try:
                for dirpath, dirnames, filenames in os.walk(self.model_path):
                    for f in filenames:
                        fp = os.path.join(dirpath, f)
                        if not os.path.islink(fp):
                            total_size += os.path.getsize(fp)
                file_size = total_size / (1024 * 1024)
                print(f"✓ Total checkpoint size: {file_size:.2f} MB")
            except Exception as e:
                print(f"✓ Could not calculate directory size: {e}")
                file_size = 0.0

        elif os.path.isfile(self.model_path):
            print(f"✓ Model file found.")
            file_size = os.path.getsize(self.model_path) / (1024 * 1024)
            print(f"✓ Model file size: {file_size:.2f} MB")
        
        else:
            print(f"❌ Path exists but is not a file or directory: {self.model_path}")
            return False

        self.results['metrics']['file_size_mb'] = file_size
        return True
    
    def test_2_model_loads(self):
        """Test if the model can be loaded by DRLSolver."""
        try:
            # DRLSolver init will build and load weights
            dummy_cities = [City(np.random.rand()*100, np.random.rand()*100, f"C{i}") 
                          for i in range(20)]
            
            # This instance will be used by other tests
            self.test_solver = DRLSolver(dummy_cities, model_path=self.model_path)
            
            if self.test_solver.model is None:
                print("❌ DRLSolver initialized, but its 'model' attribute is None.")
                print("   This might mean it fell back to heuristic mode.")
                return False
            
            print(f"✓ Model loaded successfully by DRLSolver.")
            print(f"✓ Model type: {type(self.test_solver.model).__name__}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model via DRLSolver: {e}")
            return False
    
    def test_3_forward_pass(self):
        """Test a single prediction (inference)."""
        try:
            n_cities = 25
            # We need a new solver for the correct city count
            test_cities = [City(np.random.rand()*100, np.random.rand()*100, f"C{i}") 
                          for i in range(n_cities)]
            solver = DRLSolver(test_cities, model_path=self.model_path)
            
            if solver.model is None:
                print("❌ Cannot run forward pass: Model is None (fell back to heuristic).")
                return False
            
            start = time.time()
            # Use _predict_with_model directly to isolate model performance
            tour = solver._predict_with_model()
            elapsed = time.time() - start
            
            if tour is None:
                print("❌ Model returned None")
                return False
            
            print(f"✓ Forward pass successful ({elapsed*1000:.1f}ms)")
            print(f"✓ Tour: {len(tour.cities)} cities, length {tour.get_total_distance():.2f}")
            self.results['metrics']['inference_time_ms_25_cities'] = elapsed * 1000
            return True
            
        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
            return False
    
    def test_4_valid_tours(self):
        """
        *** RIGOROUS TEST ***
        Test tour validity on 100 problems of 10-60 cities.
        """
        print("Testing tour validity (100 problems, 10-60 cities)...")
        
        n_tests = 100
        invalid = 0
        
        for i in tqdm(range(n_tests), desc="Validating"):
            n_cities = np.random.randint(10, 61) # Test full 10-60 range
            cities = [City(np.random.rand()*100, np.random.rand()*100, f"C{j}") 
                     for j in range(n_cities)]
            
            solver = DRLSolver(cities, model_path=self.model_path)
            
            if solver.model is None:
                invalid += 1 # Count as invalid if model fails to load
                continue

            # Test the model's direct output
            tour = solver._predict_with_model() 
            
            # Check 1: Is the tour a complete permutation?
            if len(tour.cities) != n_cities:
                invalid += 1
                continue
            
            city_set = set(tour.cities)
            if len(city_set) != n_cities:
                invalid += 1 # Duplicate cities in tour
                continue

            # Check 2: Is the distance valid?
            tour_length = tour.get_total_distance()
            if not (0 < tour_length < float('inf')):
                invalid += 1
        
        valid = n_tests - invalid
        valid_rate = (valid / n_tests)
        print(f"✓ Valid tours: {valid}/{n_tests} ({valid_rate*100:.1f}%)")
        
        self.results['metrics']['validity_rate'] = valid_rate
        # Pass if at least 99% of tours are valid
        return valid_rate >= 0.99
    
    def test_5_solution_quality(self):
        """
        *** RIGOROUS TEST ***
        Compare solution quality on 50 problems of 50 cities.
        """
        print("Comparing solution quality (50 problems, 50 cities)...")
        
        n_tests = 50
        drl_lengths = []
        nn_lengths = []
        
        for i in tqdm(range(n_tests), desc="Quality test"):
            n_cities = 50
            cities = [City(np.random.rand()*100, np.random.rand()*100, f"C{j}") 
                     for j in range(n_cities)]
            
            solver = DRLSolver(cities, model_path=self.model_path)
            
            if solver.model is None:
                print("\nWarning: Model failed to load, quality test is invalid.")
                # Add dummy values to avoid crash, but this test will fail
                drl_lengths.append(float('inf'))
                nn_lengths.append(1.0)
                continue

            # DRL solution (model only, no 2-opt)
            drl_tour = solver._predict_with_model()
            drl_lengths.append(drl_tour.get_total_distance())
            
            # Nearest Neighbor baseline
            nn_tour = solver.nearest_neighbor_heuristic(start_index=0)
            nn_lengths.append(nn_tour.get_total_distance())
        
        drl_avg = np.mean(drl_lengths)
        nn_avg = np.mean(nn_lengths)
        
        if nn_avg == 0: # Avoid division by zero
            improvement = 0.0
        else:
            improvement = ((nn_avg - drl_avg) / nn_avg) * 100
        
        print(f"\nResults (Avg. of {n_tests} runs, {n_cities} cities):")
        print(f"  DRL Model (Raw):  {drl_avg:.2f}")
        print(f"  Nearest Neighbor: {nn_avg:.2f}")
        print(f"  Improvement:      {improvement:+.2f}%")
        
        if improvement > 5: # Stricter requirement
            print(f"✅ Model STRONGLY BEATS Nearest Neighbor!")
            quality = "Excellent"
        elif improvement > 0:
            print(f"✓ Model BEATS Nearest Neighbor.")
            quality = "Good"
        elif improvement > -10:
            print(f"✓ Model is competitive (not worse than 10%).")
            quality = "Acceptable"
        else:
            print(f"⚠ Model is significantly worse than baseline.")
            quality = "Needs improvement"
        
        self.results['metrics']['improvement_vs_nn_pct'] = improvement
        self.results['metrics']['quality'] = quality
        
        # Pass if model is at least "Acceptable"
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
        # *** MODIFIED: Set default to your new checkpoint directory ***
        model_path = "tsp_checkpoints"
    
    print(f"--- Starting validation for model at: {model_path} ---")
    
    validator = TSPModelValidator(model_path)
    success = validator.run_all_tests()
    
    # Exit with 0 on success, 1 on failure
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()