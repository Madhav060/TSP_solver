"""
PROPER REINFORCE Training for Your Attention Model
Uses log_probs correctly to train the decoder
"""

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import os

from attention_model import AttentionModelTF

CONFIG = {
    'n_cities': 20,
    'train_samples': 10000,
    'batch_size': 128,
    'epochs': 100,
    'learning_rate': 1e-4,
    'baseline_decay': 0.9  # Exponential moving average for baseline
}

def generate_tsp_data(n_samples, n_cities):
    """Generate random TSP instances."""
    return np.random.rand(n_samples, n_cities, 2).astype(np.float32)

print("="*70)
print("PROPER REINFORCE TRAINING FOR TSP")
print("="*70)
print("Configuration:")
for k, v in CONFIG.items():
    print(f"  {k}: {v}")
print("="*70)

# Generate data
print("\nGenerating training data...")
train_data = generate_tsp_data(CONFIG['train_samples'], CONFIG['n_cities'])
print(f"✓ Generated {CONFIG['train_samples']} instances")

# Create model
print("\nCreating model...")
model = AttentionModelTF(
    embed_dim=128,
    num_encoder_layers=3,
    num_heads=8,
    ff_dim=512
)

# Build model
dummy = tf.random.uniform((1, CONFIG['n_cities'], 2))
_ = model(dummy, training=True, return_log_probs=True)

total_params = sum([np.prod(v.shape) for v in model.trainable_variables])
print(f"✓ Model created with {total_params:,} parameters")
print(f"✓ Number of trainable variables: {len(model.trainable_variables)}")

# Check layer names
print("\nModel layers:")
for v in model.trainable_variables[:5]:
    print(f"  - {v.name}")
print("  ...")
for v in model.trainable_variables[-5:]:
    print(f"  - {v.name}")

# Optimizer
optimizer = tf.keras.optimizers.Adam(CONFIG['learning_rate'])

# Baseline for REINFORCE
baseline = tf.Variable(10.0, trainable=False, dtype=tf.float32)

@tf.function
def train_step(batch_coords):
    """
    REINFORCE training step.
    
    The key insight:
    - Model samples tour and returns log_probs for each step
    - Loss = (tour_length - baseline) * sum(log_probs)
    - Gradients flow through log_probs back to decoder!
    """
    with tf.GradientTape() as tape:
        # Forward pass with sampling (return_log_probs=True)
        # Returns: (batch, n_cities) log_probs for each selection
        tours_log_probs, tour_lengths, _ = model(
            batch_coords, 
            training=True, 
            return_log_probs=True
        )
        
        # REINFORCE loss computation
        # Advantage: tour_length - baseline (we want to minimize length)
        advantages = tour_lengths - baseline
        
        # Sum log probabilities over the tour
        # tours_log_probs shape: (batch, n_cities)
        sum_log_probs = tf.reduce_sum(tours_log_probs, axis=1)  # (batch,)
        
        # Policy gradient loss: advantage * log_prob
        # We minimize this, which means:
        # - If advantage > 0 (tour worse than baseline): decrease log_prob (make it less likely)
        # - If advantage < 0 (tour better than baseline): increase log_prob (make it more likely)
        pg_loss = tf.reduce_mean(advantages * sum_log_probs)
        
        loss = pg_loss
    
    # Compute gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    
    # Check for None gradients
    none_count = sum(1 for g in gradients if g is None)
    
    if none_count > 0:
        # This shouldn't happen now, but just in case
        print(f"\n⚠️  WARNING: {none_count} gradients are None!")
        for i, (g, v) in enumerate(zip(gradients, model.trainable_variables)):
            if g is None:
                print(f"  - {v.name}")
    
    # Clip gradients
    gradients, global_norm = tf.clip_by_global_norm(gradients, 1.0)
    
    # Apply gradients
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    # Update baseline (exponential moving average)
    baseline.assign(
        CONFIG['baseline_decay'] * baseline + 
        (1 - CONFIG['baseline_decay']) * tf.reduce_mean(tour_lengths)
    )
    
    return loss, tf.reduce_mean(tour_lengths), global_norm, none_count

# Training loop
print("\n" + "="*70)
print("TRAINING")
print("="*70)

best_length = float('inf')

for epoch in range(CONFIG['epochs']):
    # Shuffle data
    indices = np.random.permutation(CONFIG['train_samples'])
    shuffled_data = train_data[indices]
    
    n_batches = CONFIG['train_samples'] // CONFIG['batch_size']
    
    epoch_losses = []
    epoch_lengths = []
    epoch_grad_norms = []
    
    with tqdm(total=n_batches, desc=f"Epoch {epoch+1:3d}/{CONFIG['epochs']}") as pbar:
        for i in range(n_batches):
            start = i * CONFIG['batch_size']
            end = start + CONFIG['batch_size']
            batch = shuffled_data[start:end]
            
            loss, length, grad_norm, none_count = train_step(batch)
            
            epoch_losses.append(loss.numpy())
            epoch_lengths.append(length.numpy())
            epoch_grad_norms.append(grad_norm.numpy())
            
            pbar.update(1)
            pbar.set_postfix({
                'loss': f'{np.mean(epoch_losses):.3f}',
                'len': f'{np.mean(epoch_lengths):.2f}',
                'grad': f'{np.mean(epoch_grad_norms):.2f}',
                'baseline': f'{baseline.numpy():.2f}'
            })
    
    avg_length = np.mean(epoch_lengths)
    avg_loss = np.mean(epoch_losses)
    avg_grad = np.mean(epoch_grad_norms)
    
    # Print every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"\nEpoch {epoch+1:3d}:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Avg Length: {avg_length:.2f}")
        print(f"  Baseline: {baseline.numpy():.2f}")
        print(f"  Grad Norm: {avg_grad:.3f}")
    
    # Save best model
    if avg_length < best_length:
        best_length = avg_length
        os.makedirs('trained_models_tf', exist_ok=True)
        save_path = 'trained_models_tf/tsp_2_50_attention_model_tf.weights.h5'
        model.save_weights(save_path)
        
        if (epoch + 1) % 10 == 0:
            print(f"  ✓ New best! Saved model (best: {best_length:.2f})")

print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)
print(f"Best tour length: {best_length:.2f}")

# Save final model
save_path = 'trained_models_tf/tsp_2_50_attention_model_tf.weights.h5'
model.save_weights(save_path)

file_size = os.path.getsize(save_path) / (1024 * 1024)
print(f"\n✓ Final model saved: {save_path}")
print(f"✓ File size: {file_size:.2f} MB")

if file_size < 10:
    print("\n⚠️  WARNING: File is small - model may not have saved all weights")
else:
    print("✓ File size looks correct!")

print("\n" + "="*70)
print("Next steps:")
print("  1. Run: python validate_model.py")
print("  2. The model should now work properly!")
print("="*70)