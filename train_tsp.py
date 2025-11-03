"""
Enhanced TSP Training Script for 10-60 Cities
Fixed version - handles variable city counts efficiently without retracing
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import json
from typing import Tuple, List, Dict
import time

# Import your existing model
from attention_model import AttentionModelTF


# ============================================================================
# Data Generation with Fixed Padding
# ============================================================================

class TSPDataGenerator:
    """Generate TSP training data with fixed padding to avoid retracing"""
    
    def __init__(self, min_cities=10, max_cities=60, seed=None):
        self.min_cities = min_cities
        self.max_cities = max_cities
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            tf.random.set_seed(seed)
    
    def generate_batch_fixed_size(self, batch_size, num_cities):
        """Generate a batch with fixed number of cities"""
        cities = np.random.rand(batch_size, num_cities, 2).astype(np.float32)
        return tf.constant(cities)
    
    def generate_batch_curriculum(self, batch_size, epoch, total_epochs):
        """Curriculum learning: start with smaller problems, gradually increase"""
        # Start with min_cities, gradually reach max_cities
        progress = min(epoch / (total_epochs * 0.7), 1.0)  # Reach max at 70% of training
        current_max = int(self.min_cities + (self.max_cities - self.min_cities) * progress)
        
        num_cities = np.random.randint(self.min_cities, current_max + 1)
        cities = np.random.rand(batch_size, num_cities, 2).astype(np.float32)
        return tf.constant(cities), num_cities
    
    def generate_dataset_fixed_sizes(self, num_samples_per_size, city_sizes):
        """Generate datasets with specific city counts"""
        datasets = {}
        for n_cities in city_sizes:
            problems = []
            for _ in range(num_samples_per_size):
                cities = np.random.rand(n_cities, 2).astype(np.float32)
                problems.append(cities)
            datasets[n_cities] = problems
        return datasets


# ============================================================================
# Training Utilities
# ============================================================================

class ExponentialBaseline:
    """
    Exponential moving average baseline per problem size.
    
    This version uses tf.Variable to correctly manage state within tf.function.
    """
    
    def __init__(self, beta=0.8):
        self.beta = beta
        # This Python dictionary will store tf.Variable objects, not tensors.
        self.baselines = {}  
    
    def eval(self, tour_lengths, n_cities):
        """Update and return baseline for specific problem size"""
        
        # Use Python int for the dictionary key
        n_cities_key = int(n_cities) 
        
        if n_cities_key not in self.baselines:
            # Create a new tf.Variable. This happens *once* per
            # problem size when the tf.function is first traced.
            self.baselines[n_cities_key] = tf.Variable(
                tf.reduce_mean(tour_lengths),
                trainable=False,
                name=f'baseline_{n_cities_key}'
            )
        else:
            # Update the existing tf.Variable using .assign()
            # This creates a stateful operation in the graph.
            new_baseline = (
                self.beta * self.baselines[n_cities_key] + 
                (1 - self.beta) * tf.reduce_mean(tour_lengths)
            )
            self.baselines[n_cities_key].assign(new_baseline)
        
        # Return the current value of the variable
        return self.baselines[n_cities_key]
    
    def reset(self):
        self.baselines = {}


# ============================================================================
# REINFORCE Loss
# ============================================================================

def reinforce_loss(log_probs, tour_lengths, baseline):
    """
    REINFORCE loss with baseline
    Args:
        log_probs: (batch_size, n_cities) - log probabilities of actions
        tour_lengths: (batch_size,) - tour lengths (negative reward)
        baseline: (batch_size,) or scalar - baseline for variance reduction
    """
    advantages = tour_lengths - baseline
    log_probs_sum = tf.reduce_sum(log_probs, axis=1)
    loss = tf.reduce_mean(log_probs_sum * advantages)
    return loss


# ============================================================================
# Training Loop - Fixed to avoid retracing
# ============================================================================

class TSPTrainer:
    """Enhanced trainer with curriculum learning and fixed problem sizes"""
    
    def __init__(
        self,
        model,
        learning_rate=1e-4,
        grad_clip=1.0
    ):
        self.model = model
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.grad_clip = grad_clip
        self.exponential_baseline = ExponentialBaseline(beta=0.8)
        
        # Metrics
        self.train_loss_history = []
        self.train_distance_history = []
        self.val_distance_history = []
        self.best_val_distance = float('inf')
        
        # Cache for compiled functions per problem size
        self.train_step_cache = {}
    
    def get_train_step_fn(self, n_cities):
        """Get or create cached train step function for specific problem size"""
        if n_cities not in self.train_step_cache:
            @tf.function
            def train_step_fixed(cities):
                with tf.GradientTape() as tape:
                    # Forward pass - sample tours
                    log_probs, tour_lengths, _ = self.model(
                        cities, training=True, return_log_probs=True
                    )
                    
                    # Use exponential baseline
                    # This now correctly uses tf.Variable.assign() internally
                    baseline = self.exponential_baseline.eval(tour_lengths, n_cities)
                    
                    # Compute loss
                    loss = reinforce_loss(log_probs, tour_lengths, baseline)
                
                # Backward pass
                gradients = tape.gradient(loss, self.model.trainable_variables)
                
                # Gradient clipping
                if self.grad_clip > 0:
                    gradients, _ = tf.clip_by_global_norm(gradients, self.grad_clip)
                
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                
                return loss, tour_lengths
            
            self.train_step_cache[n_cities] = train_step_fixed
        
        return self.train_step_cache[n_cities]
    
    def train_epoch_curriculum(self, data_generator, epoch, total_epochs, 
                              steps_per_epoch=1000, batch_size=128):
        """Train for one epoch with curriculum learning"""
        total_loss = 0.0
        total_distance = 0.0
        
        # Determine city sizes to train on this epoch (curriculum)
        progress = min(epoch / (total_epochs * 0.7), 1.0)
        current_max = int(data_generator.min_cities + 
                         (data_generator.max_cities - data_generator.min_cities) * progress)
        
        # Focus on 3-4 problem sizes per epoch to reduce retracing
        city_sizes = np.linspace(data_generator.min_cities, current_max, 4, dtype=int)
        city_sizes = list(set(city_sizes))  # Remove duplicates
        
        print(f"Training on city sizes: {city_sizes}")
        
        steps_per_size = steps_per_epoch // len(city_sizes)
        
        for n_cities in city_sizes:
            train_step_fn = self.get_train_step_fn(n_cities)
            
            for step in range(steps_per_size):
                # Generate batch with fixed size
                cities = data_generator.generate_batch_fixed_size(batch_size, n_cities)
                
                # Training step
                loss, tour_lengths = train_step_fn(cities)
                
                total_loss += loss.numpy()
                total_distance += tf.reduce_mean(tour_lengths).numpy()
            
            if len(city_sizes) <= 5:
                avg_loss = total_loss / ((city_sizes.index(n_cities) + 1) * steps_per_size)
                avg_dist = total_distance / ((city_sizes.index(n_cities) + 1) * steps_per_size)
                print(f"  {n_cities} cities | Loss: {avg_loss:.4f} | Avg Dist: {avg_dist:.4f}")
        
        total_steps = len(city_sizes) * steps_per_size
        avg_loss = total_loss / total_steps
        avg_distance = total_distance / total_steps
        
        return avg_loss, avg_distance
    
    def train_epoch_fixed_sizes(self, data_generator, city_sizes, 
                               steps_per_size=250, batch_size=128):
        """Train for one epoch on specific city sizes"""
        total_loss = 0.0
        total_distance = 0.0
        total_steps = 0
        
        for n_cities in city_sizes:
            train_step_fn = self.get_train_step_fn(n_cities)
            
            for step in range(steps_per_size):
                cities = data_generator.generate_batch_fixed_size(batch_size, n_cities)
                loss, tour_lengths = train_step_fn(cities)
                
                total_loss += loss.numpy()
                total_distance += tf.reduce_mean(tour_lengths).numpy()
                total_steps += 1
            
            avg_loss = total_loss / total_steps
            avg_dist = total_distance / total_steps
            print(f"  {n_cities} cities | Loss: {avg_loss:.4f} | Avg Dist: {avg_dist:.4f}")
        
        avg_loss = total_loss / total_steps
        avg_distance = total_distance / total_steps
        
        return avg_loss, avg_distance
    
    def validate(self, val_datasets, batch_size=128):
        """Validate on datasets grouped by city count"""
        all_distances = []
        
        print("Validating...")
        
        for n_cities, problems in val_datasets.items():
            # Batch problems of same size
            for i in range(0, len(problems), batch_size):
                batch = problems[i:i+batch_size]
                cities_batch = tf.constant(np.array(batch))
                
                # Greedy decoding
                tour_indices, tour_lengths, _ = self.model(
                    cities_batch, training=False, return_log_probs=False
                )
                
                all_distances.extend(tour_lengths.numpy())
        
        avg_distance = np.mean(all_distances)
        return avg_distance
    
    def train(
        self,
        data_generator,
        epochs=100,
        batch_size=128,
        val_samples_per_size=100,
        checkpoint_dir='checkpoints',
        early_stopping_patience=20,
        training_mode='curriculum'  # 'curriculum' or 'fixed_sizes'
    ):
        """Full training loop"""
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Generate validation set (specific sizes for consistent evaluation)
        print("Generating validation set...")
        val_city_sizes = [15, 20, 30, 40, 50, 55]
        val_datasets = data_generator.generate_dataset_fixed_sizes(
            val_samples_per_size, val_city_sizes
        )
        
        # Training configuration
        if training_mode == 'curriculum':
            steps_per_epoch = 1000
        else:
            # Fixed sizes mode
            train_city_sizes = [15, 20, 25, 30, 35, 40, 50, 60]
            steps_per_size = 125
            steps_per_epoch = len(train_city_sizes) * steps_per_size
        
        # Training info
        print("\n" + "="*70)
        print("TSP TRAINING CONFIGURATION")
        print("="*70)
        print(f"Model: Attention-based TSP Solver")
        print(f"City range: {data_generator.min_cities}-{data_generator.max_cities}")
        print(f"Training mode: {training_mode}")
        print(f"Epochs: {epochs}")
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Batch size: {batch_size}")
        print(f"Validation sizes: {val_city_sizes} ({val_samples_per_size} each)")
        print(f"Learning rate: {self.optimizer.learning_rate.numpy()}")
        print(f"Trainable parameters: {sum([tf.size(v).numpy() for v in self.model.trainable_variables]):,}")
        print("="*70 + "\n")
        
        patience_counter = 0
        
        for epoch in range(epochs):
            start_time = time.time()
            
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 70)
            
            # Train
            if training_mode == 'curriculum':
                train_loss, train_distance = self.train_epoch_curriculum(
                    data_generator, epoch, epochs, steps_per_epoch, batch_size
                )
            else:
                train_loss, train_distance = self.train_epoch_fixed_sizes(
                    data_generator, train_city_sizes, steps_per_size, batch_size
                )
            
            # Validate
            val_distance = self.validate(val_datasets, batch_size)
            
            epoch_time = time.time() - start_time
            
            # Store metrics
            self.train_loss_history.append(train_loss)
            self.train_distance_history.append(train_distance)
            self.val_distance_history.append(val_distance)
            
            # Print summary
            print(f"\nEpoch Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Train Distance: {train_distance:.4f}")
            print(f"  Val Distance: {val_distance:.4f}")
            print(f"  Time: {epoch_time:.2f}s")
            
            # Save best model
            if val_distance < self.best_val_distance:
                self.best_val_distance = val_distance
                patience_counter = 0
                
                checkpoint_path = os.path.join(checkpoint_dir, 'best_model')
                self.model.save_weights(checkpoint_path)
                print(f"  ✓ New best model saved! Val distance: {val_distance:.4f}")
            else:
                patience_counter += 1
                print(f"  No improvement ({patience_counter}/{early_stopping_patience})")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}')
                self.model.save_weights(checkpoint_path)
                print(f"  Checkpoint saved: epoch {epoch+1}")
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE!")
        print(f"Best validation distance: {self.best_val_distance:.4f}")
        print("="*70)
        
        # Save training history
        self.save_training_history(checkpoint_dir)
        self.plot_training_curves(checkpoint_dir)
    
    def save_training_history(self, save_dir):
        """Save training metrics to JSON"""
        history = {
            'train_loss': [float(x) for x in self.train_loss_history],
            'train_distance': [float(x) for x in self.train_distance_history],
            'val_distance': [float(x) for x in self.val_distance_history],
            'best_val_distance': float(self.best_val_distance)
        }
        
        with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"Training history saved to {save_dir}/training_history.json")
    
    def plot_training_curves(self, save_dir):
        """Plot and save training curves"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curve
        axes[0].plot(self.train_loss_history, linewidth=2, label='Train Loss', color='#2E86AB')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(fontsize=10)
        
        # Distance curves
        axes[1].plot(self.train_distance_history, linewidth=2, label='Train Distance', color='#2E86AB')
        axes[1].plot(self.val_distance_history, linewidth=2, label='Val Distance', color='#A23B72')
        axes[1].axhline(y=self.best_val_distance, color='#F18F01', linestyle='--', 
                       linewidth=2, label=f'Best Val: {self.best_val_distance:.4f}')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Tour Distance', fontsize=12)
        axes[1].set_title('Tour Distance Progress', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved to {save_dir}/training_curves.png")


# ============================================================================
# Main Training Script
# ============================================================================

def main():
    """Main training function with fixed problem sizes to avoid retracing"""
    
    # Configuration
    MIN_CITIES = 10
    MAX_CITIES = 60
    EMBED_DIM = 128
    NUM_ENCODER_LAYERS = 3
    NUM_HEADS = 8
    FF_DIM = 512
    
    EPOCHS = 100
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-4
    VAL_SAMPLES_PER_SIZE = 100
    
    CHECKPOINT_DIR = 'tsp_checkpoints'
    SEED = 42
    
    # Choose training mode
    TRAINING_MODE = 'fixed_sizes'  # 'curriculum' or 'fixed_sizes'
    # 'fixed_sizes' is more stable and faster (recommended)
    # 'curriculum' starts easier and gradually increases difficulty
    
    # Set seeds
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    
    # Create model
    print("Creating model...")
    model = AttentionModelTF(
        embed_dim=EMBED_DIM,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        tanh_clipping=10.0,
        softmax_temperature=1.0
    )
    
    # Build model with dummy input
    dummy_input = tf.random.uniform((2, 20, 2))
    _ = model(dummy_input, training=False, return_log_probs=False)
    
    print(f"Model created with {sum([tf.size(v).numpy() for v in model.trainable_variables]):,} parameters")
    
    # Create data generator
    data_generator = TSPDataGenerator(
        min_cities=MIN_CITIES,
        max_cities=MAX_CITIES,
        seed=SEED
    )
    
    # Create trainer
    trainer = TSPTrainer(
        model=model,
        learning_rate=LEARNING_RATE,
        grad_clip=1.0
    )
    
    # Train
    trainer.train(
        data_generator=data_generator,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        val_samples_per_size=VAL_SAMPLES_PER_SIZE,
        checkpoint_dir=CHECKPOINT_DIR,
        early_stopping_patience=20,
        training_mode=TRAINING_MODE
    )
    
    print("\n✓ Training complete! Model saved to:", CHECKPOINT_DIR)


if __name__ == '__main__':
    main()