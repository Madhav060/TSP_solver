import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
from tqdm import tqdm # Progress bar
import numpy as np
import os
import random # *** MODIFIED: Import random for sampling n_cities ***

from attention_model import AttentionModelTF # Import the TF model
from data_generator import generate_tsp_data

def train_attention_model_tf(
    # *** MODIFIED: Define min/max cities instead of fixed n_cities ***
    min_cities: int = 2,
    max_cities: int = 50,
    # ------------------------------------------------------------------
    embed_dim: int = 128,
    num_encoder_layers: int = 3,
    num_heads: int = 8,
    ff_dim: int = 512,
    n_epochs: int = 100,
    batch_size: int = 512,
    steps_per_epoch: int = 500,
    lr: float = 1e-4,
    baseline_alpha: float = 0.99, # EMA decay factor for baseline
    save_path: str = "trained_models_tf", # Separate folder for TF models
    # *** CORRECTED: Use the required .weights.h5 suffix ***
    model_filename: str = "tsp_2_50_attention_model_tf.weights.h5"
    # --------------------------------------------------------
):
    """
    Trains the AttentionModelTF for TSP using REINFORCE with baseline (TensorFlow version).
    Trains on a range of city sizes.
    """
    # --- Ensure GPU is used if available ---
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print(f"Using GPU: {physical_devices[0]}")
    else:
        print("Using CPU")

    # --- Initialize Model, Optimizer, Baseline ---
    model = AttentionModelTF(
        embed_dim=embed_dim,
        num_encoder_layers=num_encoder_layers,
        num_heads=num_heads,
        ff_dim=ff_dim
    )
    # Build the model by calling it once with dummy data (use max_cities for build)
    dummy_input = tf.random.uniform((1, max_cities, 2))
    _ = model(dummy_input) # This initializes weights

    optimizer = optimizers.Adam(learning_rate=lr)

    baseline_reward = tf.Variable(0.0, trainable=False, dtype=tf.float32)

    print(f"\n--- Starting Training for TSP-{min_cities}-{max_cities} (TensorFlow) ---")
    print(f"Hyperparameters:")
    print(f"  Epochs: {n_epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Steps per Epoch: {steps_per_epoch}")
    print(f"  Learning Rate: {lr}")
    print(f"  EMA Alpha: {baseline_alpha}")
    model.summary() # Print model structure

    best_avg_length = tf.Variable(np.inf, trainable=False, dtype=tf.float32)

    # --- Training Loop ---
    for epoch in range(n_epochs):
        epoch_total_loss = tf.Variable(0.0, dtype=tf.float32)
        epoch_total_length = tf.Variable(0.0, dtype=tf.float32)
        epoch_avg_cities = tf.Variable(0.0, dtype=tf.float32)

        print(f"\nEpoch {epoch+1}/{n_epochs}")
        pbar = tqdm(range(steps_per_epoch))

        for step in pbar:
            current_n_cities = random.randint(min_cities, max_cities)
            inputs_np = generate_tsp_data(batch_size, current_n_cities)
            inputs = tf.convert_to_tensor(inputs_np)

            with tf.GradientTape() as tape:
                tours_log_probs, tour_lengths, _ = model(inputs, return_log_probs=True, training=True)
                reward = -tour_lengths
                current_mean_reward = tf.reduce_mean(reward)
                baseline_reward.assign(baseline_alpha * baseline_reward +
                                       (1.0 - baseline_alpha) * current_mean_reward)
                advantage = reward - baseline_reward
                total_log_probs = tf.reduce_sum(tours_log_probs, axis=1)
                loss = -tf.reduce_mean(advantage * total_log_probs)

            gradients = tape.gradient(loss, model.trainable_variables)
            gradients = [tf.clip_by_norm(g, 1.0) if g is not None else None for g in gradients]
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            epoch_total_loss.assign_add(loss)
            epoch_total_length.assign_add(tf.reduce_mean(tour_lengths))
            epoch_avg_cities.assign_add(tf.cast(current_n_cities, tf.float32))

            pbar.set_postfix({
                'Loss': f"{loss.numpy():.4f}",
                'Avg Len': f"{tf.reduce_mean(tour_lengths).numpy():.2f}",
                'N_Cities': current_n_cities,
                'Baseline': f"{-baseline_reward.numpy():.2f}"
            })

        avg_loss = epoch_total_loss / steps_per_epoch
        avg_length = epoch_total_length / steps_per_epoch
        avg_cities = epoch_avg_cities / steps_per_epoch
        print(f"End of Epoch {epoch+1}: Avg Loss = {avg_loss.numpy():.4f}, Avg Tour Length = {avg_length.numpy():.2f}, Avg Cities = {avg_cities.numpy():.1f}")

        if avg_length < best_avg_length:
            best_avg_length.assign(avg_length)
            os.makedirs(save_path, exist_ok=True)
            full_save_path = os.path.join(save_path, model_filename)
            model.save_weights(full_save_path) # Now uses the corrected filename
            print(f"✓ New best model saved with avg length {best_avg_length.numpy():.2f} to {full_save_path}")

    print(f"\n--- Training Finished for TSP-{min_cities}-{max_cities} (TensorFlow) ---")
    print(f"Best average tour length achieved (across sizes): {best_avg_length.numpy():.2f}")

# --- Main Execution Block ---
if __name__ == "__main__":
    train_attention_model_tf(
        min_cities=2,
        max_cities=50,
        n_epochs=100,
        batch_size=256,
        steps_per_epoch=500,
        # *** CORRECTED: Use the required .weights.h5 suffix in the call too ***
        model_filename="tsp_2_50_attention_model_tf.weights.h5"
        # ----------------------------------------------------------------------
    )