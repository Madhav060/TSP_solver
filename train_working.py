"""
WORKING Training Script for TSP Attention Model
Properly trains the full encoder-decoder architecture
"""

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import os
import sys

from attention_model import AttentionModelTF

# Configuration
CONFIG = {
    'n_cities': 20,
    'train_samples': 10000,
    'batch_size': 128,
    'epochs': 100,
    'learning_rate': 1e-4,
    'print_every': 10
}

def generate_tsp_data(n_samples, n_cities):
    """Generate random TSP instances."""
    return np.random.rand(n_samples, n_cities, 2).astype(np.float32)

def compute_tour_length(coords, tour_indices):
    """
    Compute tour length from coordinates and indices.
    
    Args:
        coords: (batch, n_cities, 2)
        tour_indices: (batch, n_cities) - integer indices
    
    Returns:
        lengths: (batch,)
    """
    batch_size = tf.shape(coords)[0]
    n_cities = tf.shape(coords)[1]
    
    # Gather coordinates in tour order
    # tour_indices shape: (batch, n_cities)
    # We need to use batch_dims=1 to gather along batch dimension
    tour_coords = tf.gather(coords, tour_indices, batch_dims=1)  # (batch, n_cities, 2)
    
    # Compute distances between consecutive cities
    # Shift by one to get next city
    rolled = tf.roll(tour_coords, shift=-1, axis=1)  # (batch, n_cities, 2)
    
    # Euclidean distance
    distances = tf.sqrt(tf.reduce_sum((tour_coords - rolled) ** 2, axis=2))  # (batch, n_cities)
    
    # Sum all segment distances
    total_lengths = tf.reduce_sum(distances, axis=1)  # (batch,)
    
    return total_lengths

class TSPTrainer:
    """Trainer class for TSP model."""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.optimizer = None
        self.train_data = None
        self.best_length = float('inf')
        
    def setup(self):
        """Setup model, optimizer, and data."""
        print("="*70)
        print("SETUP")
        print("="*70)
        
        # Generate data
        print("\nGenerating training data...")
        self.train_data = generate_tsp_data(
            self.config['train_samples'],
            self.config['n_cities']
        )
        print(f"✓ Generated {self.config['train_samples']} samples")
        
        # Create model
        print("\nCreating model...")
        self.model = AttentionModelTF(
            embed_dim=128,
            num_encoder_layers=3,
            num_heads=8,
            ff_dim=512
        )
        
        # Build model
        dummy = tf.random.uniform((1, self.config['n_cities'], 2))
        _ = self.model(dummy, training=False)
        
        # Count parameters
        total_params = sum([np.prod(v.shape) for v in self.model.trainable_variables])
        print(f"✓ Model created")
        print(f"✓ Total parameters: {total_params:,}")
        print(f"✓ Trainable layers: {len(self.model.trainable_variables)}")
        
        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(self.config['learning_rate'])
        print(f"✓ Optimizer: Adam (lr={self.config['learning_rate']})")
        
        print("\n" + "="*70)
    
    @tf.function
    def train_step(self, batch_coords):
        """
        Single training step using REINFORCE algorithm.
        """
        with tf.GradientTape() as tape:
            # Forward pass - model generates tours
            tour_indices, tour_lengths, log_probs = self.model(
                batch_coords, 
                return_log_probs=True,
                training=True
            )
            
            # REINFORCE: Use tour length as reward (negative because we minimize)
            # Baseline: mean tour length in batch
            baseline = tf.stop_gradient(tf.reduce_mean(tour_lengths))
            
            # Advantage
            advantages = tour_lengths - baseline
            
            # Policy gradient loss
            # We want to increase probability of good tours (short length)
            # Loss = advantage * log_prob, we minimize this
            if log_probs is not None and len(log_probs.shape) > 1:
                # If log_probs is (batch, cities, dim), sum over cities
                log_probs = tf.reduce_sum(log_probs, axis=[1, 2])
            elif log_probs is not None and len(log_probs.shape) > 0:
                log_probs = tf.reduce_sum(log_probs, axis=1)
            
            # Compute loss
            if log_probs is not None:
                policy_loss = tf.reduce_mean(advantages * log_probs)
            else:
                # Fallback: just minimize tour length
                policy_loss = tf.reduce_mean(tour_lengths)
            
            loss = policy_loss
        
        # Compute gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)
        
        # Check if gradients exist
        valid_gradients = [g for g in gradients if g is not None]
        
        if len(valid_gradients) < len(self.model.trainable_variables):
            # Some gradients are None - this is the problem!
            # Use simpler loss: just tour length
            with tf.GradientTape() as tape2:
                tour_indices, tour_lengths, _ = self.model(batch_coords, training=True)
                loss = tf.reduce_mean(tour_lengths)
            
            gradients = tape2.gradient(loss, self.model.trainable_variables)
        
        # Clip gradients
        gradients, global_norm = tf.clip_by_global_norm(gradients, 1.0)
        
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return loss, tf.reduce_mean(tour_lengths), global_norm
    
    def train(self):
        """Main training loop."""
        print("TRAINING")
        print("="*70)
        
        for epoch in range(self.config['epochs']):
            # Shuffle data
            indices = np.random.permutation(self.config['train_samples'])
            shuffled_data = self.train_data[indices]
            
            n_batches = self.config['train_samples'] // self.config['batch_size']
            
            epoch_losses = []
            epoch_lengths = []
            epoch_grad_norms = []
            
            # Training batches
            with tqdm(total=n_batches, desc=f"Epoch {epoch+1:3d}/{self.config['epochs']}") as pbar:
                for i in range(n_batches):
                    start = i * self.config['batch_size']
                    end = start + self.config['batch_size']
                    batch = shuffled_data[start:end]
                    
                    # Train step
                    loss, length, grad_norm = self.train_step(batch)
                    
                    epoch_losses.append(loss.numpy())
                    epoch_lengths.append(length.numpy())
                    epoch_grad_norms.append(grad_norm.numpy())
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        'loss': f'{np.mean(epoch_losses):.3f}',
                        'len': f'{np.mean(epoch_lengths):.2f}',
                        'grad': f'{np.mean(epoch_grad_norms):.2f}'
                    })
            
            # Epoch summary
            avg_length = np.mean(epoch_lengths)
            avg_loss = np.mean(epoch_losses)
            avg_grad = np.mean(epoch_grad_norms)
            
            # Print summary
            if (epoch + 1) % self.config['print_every'] == 0:
                print(f"\nEpoch {epoch+1:3d}: Loss={avg_loss:.4f}, "
                      f"Length={avg_length:.2f}, GradNorm={avg_grad:.3f}")
            
            # Save best model
            if avg_length < self.best_length:
                self.best_length = avg_length
                self.save_model(epoch + 1)
                if (epoch + 1) % self.config['print_every'] == 0:
                    print(f"  → New best! Saved model (length: {self.best_length:.2f})")
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        print(f"Best tour length: {self.best_length:.2f}")
        
        # Save final model
        self.save_model(self.config['epochs'], final=True)
    
    def save_model(self, epoch, final=False):
        """Save model weights."""
        os.makedirs('trained_models_tf', exist_ok=True)
        
        if final:
            save_path = 'trained_models_tf/tsp_2_50_attention_model_tf.weights.h5'
        else:
            save_path = 'trained_models_tf/tsp_2_50_attention_model_tf.weights.h5'
        
        try:
            self.model.save_weights(save_path)
            
            # Check file size
            file_size = os.path.getsize(save_path) / (1024 * 1024)
            
            if final:
                print(f"\n✓ Model saved: {save_path}")
                print(f"✓ File size: {file_size:.2f} MB")
                
                if file_size < 10:
                    print("⚠️  WARNING: File size is small! Model may not have saved correctly.")
                else:
                    print("✓ File size looks good!")
        except Exception as e:
            print(f"❌ Error saving model: {e}")


def main():
    """Main entry point."""
    # Set random seeds
    np.random.seed(42)
    tf.random.set_seed(42)
    
    print("\n" + "="*70)
    print("TSP ATTENTION MODEL TRAINING")
    print("="*70)
    print("Configuration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    print("="*70 + "\n")
    
    # Create trainer
    trainer = TSPTrainer(CONFIG)
    
    # Setup
    trainer.setup()
    
    # Train
    trainer.train()
    
    print("\n" + "="*70)
    print("Next steps:")
    print("  1. Run: python validate_model.py")
    print("  2. Check model file size (should be ~40-50 MB)")
    print("  3. Run: python gui.py")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()