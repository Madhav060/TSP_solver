import numpy as np
import tensorflow as tf # Import tf mainly for testing dtype

def generate_tsp_data(batch_size: int, n_cities: int) -> np.ndarray:
    """
    Generates a batch of random TSP instances using NumPy.
    Cities are represented by 2D coordinates sampled uniformly from [0, 1].

    Args:
        batch_size (int): Number of TSP instances in the batch.
        n_cities (int): Number of cities in each instance.

    Returns:
        np.ndarray: Batch of city coordinates, shape (batch_size, n_cities, 2), dtype float32.
    """
    # Use float32 as it's standard for TF models
    return np.random.rand(batch_size, n_cities, 2).astype(np.float32)

# --- Example Usage ---
if __name__ == '__main__':
    batch_size = 4
    n_cities = 20
    data = generate_tsp_data(batch_size, n_cities)

    print("--- TSP Data Generation (NumPy/TF) ---")
    print("Batch size:", batch_size)
    print("Number of cities:", n_cities)
    print("Data shape:", data.shape) # Should be (batch_size, n_cities, 2)
    print("Data type:", data.dtype) # Should be float32
    print("First instance (first 5 cities):\n", data[0, :5, :])

    # Test conversion to TensorFlow tensor
    tf_data = tf.convert_to_tensor(data)
    print("\nTensorFlow tensor shape:", tf_data.shape)
    print("TensorFlow tensor dtype:", tf_data.dtype)