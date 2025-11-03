import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import math

class GraphEmbedding(layers.Layer):
    """Initial embedding layer for city coordinates using Dense layer."""
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        # Dense layer for projection
        self.dense = layers.Dense(embed_dim, name='graph_embedding')

    def call(self, inputs):
        # inputs shape: (batch_size, n_cities, 2)
        return self.dense(inputs)
        # Output shape: (batch_size, n_cities, embed_dim)

class MultiHeadAttentionTF(layers.Layer):
    """Keras MultiHeadAttention layer wrapper for convenience."""
    def __init__(self, embed_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        # Use the built-in Keras MHA layer
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads, name='multi_head_attention')
        # Layer normalization applied after MHA
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)
        # Add layer for residual connection
        self.add = layers.Add()

    def call(self, inputs, attention_mask=None):
        # Keras MHA expects query, value, key. For self-attention, they are the same.
        attn_output, attn_scores = self.mha(
            query=inputs, value=inputs, key=inputs,
            attention_mask=attention_mask,
            return_attention_scores=True
        )
        # Residual connection and layer normalization
        out1 = self.layernorm(self.add([inputs, attn_output]))
        return out1, attn_scores

class FeedForwardTF(layers.Layer):
    """Position-wise Feed-Forward Network using Dense layers."""
    def __init__(self, embed_dim, ff_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        # Two dense layers with ReLU activation in between
        self.dense1 = layers.Dense(ff_dim, activation="relu", name='ff_dense_1')
        self.dense2 = layers.Dense(embed_dim, name='ff_dense_2')
        # Layer normalization applied after FF
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)
        # Add layer for residual connection
        self.add = layers.Add()

    def call(self, inputs):
        # 1. Pass inputs through the first dense layer
        x = self.dense1(inputs)
        
        # 2. Pass the result 'x' (not 'inputs'!) through the second dense layer
        #    This was the fix for the 'None' gradient bug
        x = self.dense2(x)
        
        # 3. Residual connection and layer normalization
        out = self.layernorm(self.add([inputs, x]))
        return out

class EncoderLayerTF(layers.Layer):
    """Single Encoder layer combining MHA and FF."""
    def __init__(self, embed_dim, num_heads, ff_dim, **kwargs):
        super().__init__(**kwargs)
        self.mha = MultiHeadAttentionTF(embed_dim, num_heads)
        self.ffn = FeedForwardTF(embed_dim, ff_dim)

    def call(self, inputs, attention_mask=None):
        attn_output, _ = self.mha(inputs, attention_mask=attention_mask)
        ffn_output = self.ffn(attn_output)
        return ffn_output

class AttentionEncoderTF(layers.Layer):
    """Stacks multiple Encoder layers."""
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.encoder_layers = [
            EncoderLayerTF(embed_dim, num_heads, ff_dim, name=f'encoder_layer_{i}')
            for i in range(num_layers)
        ]

    def call(self, inputs, attention_mask=None):
        x = inputs
        for layer in self.encoder_layers:
            x = layer(x, attention_mask=attention_mask)
        return x

class PointerDecoderTF(layers.Layer):
    """Decoder using Pointer Network mechanism."""
    def __init__(self, embed_dim, num_heads, tanh_clipping=10.0, softmax_temperature=1.0, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.tanh_clipping = tanh_clipping
        self.softmax_temperature = softmax_temperature

        # This layer projects the context features into a query vector
        self.query_projection = layers.Dense(embed_dim, name='decoder_query_proj')
        
        # Note: Removed the unused self.mha layer from here

    @tf.function
    def call(self, encoder_output, graph_embedding_pooled, current_city_embedding, first_city_embedding, mask):
        # Concatenate context features: (graph_avg, current_city, first_city)
        context_features = tf.concat([
            graph_embedding_pooled,
            current_city_embedding,
            first_city_embedding
        ], axis=-1)

        # Project context to get the query vector
        query = self.query_projection(context_features)
        query = tf.expand_dims(query, axis=1) # Shape: (batch, 1, embed_dim)

        # Expand mask for matmul
        mha_mask = tf.expand_dims(mask, axis=1) # Shape: (batch, 1, n_cities)

        # Simplified dot-product based scoring (Pointer Network)
        # (batch, 1, embed_dim) @ (batch, embed_dim, n_cities) -> (batch, 1, n_cities)
        scores = tf.matmul(query, encoder_output, transpose_b=True) / tf.sqrt(tf.cast(self.embed_dim, tf.float32))
        
        # Clip scores as in the original paper
        scores = self.tanh_clipping * tf.tanh(scores)
        
        # Apply the mask (set visited cities to -inf)
        scores_masked = tf.where(mha_mask, scores, tf.fill(tf.shape(scores), -np.inf))
        
        # Get log probabilities
        log_probs = tf.nn.log_softmax(scores_masked / self.softmax_temperature, axis=-1)
        
        # Squeeze out the "query" dimension
        log_probs = tf.squeeze(log_probs, axis=1) # Shape: (batch, n_cities)

        return log_probs

class AttentionModelTF(keras.Model):
    """Main TSP Solver model using TensorFlow Keras."""
    def __init__(
        self,
        embed_dim=128,
        num_encoder_layers=3,
        num_heads=8,
        ff_dim=512,
        tanh_clipping=10.0,
        softmax_temperature=1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.embedding = GraphEmbedding(embed_dim)
        self.encoder = AttentionEncoderTF(num_encoder_layers, embed_dim, num_heads, ff_dim)
        self.decoder = PointerDecoderTF(embed_dim, num_heads, tanh_clipping, softmax_temperature)

    def _get_pooled_embedding(self, embeddings):
        return tf.reduce_mean(embeddings, axis=1)

    @tf.function
    def call(self, inputs, training=False, return_log_probs=True):
        batch_size = tf.shape(inputs)[0]
        n_cities = tf.shape(inputs)[1]

        initial_embeddings = self.embedding(inputs)
        encoder_output = self.encoder(initial_embeddings)
        graph_embedding_pooled = self._get_pooled_embedding(encoder_output)

        # MUST CREATE BOTH TENSOR ARRAYS for graph consistency
        tour_indices_ta = tf.TensorArray(tf.int64, size=n_cities)
        tours_log_probs_ta = tf.TensorArray(tf.float32, size=n_cities)

        mask = tf.ones((batch_size, n_cities), dtype=tf.bool)
        current_city_idx = tf.zeros((batch_size,), dtype=tf.int64) # Start at city 0
        first_city_embedding = encoder_output[:, 0, :]
        
        # Create a constant zero tensor for non-log_prob steps
        zeros = tf.zeros((batch_size,))

        for step in tf.range(n_cities):
            # 1. Store the current city in the tour
            tour_indices_ta = tour_indices_ta.write(step, current_city_idx)

            # 2. Get embedding for the current city
            current_city_embedding = tf.gather(
                encoder_output, current_city_idx, batch_dims=1
            )
            
            # 3. Update the mask to forbid visiting this city again
            mask = tf.tensor_scatter_nd_update(
                mask,
                tf.stack([tf.cast(tf.range(batch_size), dtype=tf.int64), current_city_idx], axis=1),
                tf.zeros((batch_size,), dtype=tf.bool)
            )

            # 4. If this is not the last step, run the decoder to pick the *next* city
            if step < n_cities - 1:
                log_probs = self.decoder(
                    encoder_output,
                    graph_embedding_pooled,
                    current_city_embedding,
                    first_city_embedding,
                    mask
                )

                if return_log_probs:
                    # Training: Sample next city
                    next_city_idx = tf.random.categorical(log_probs, 1)
                    next_city_idx = tf.squeeze(next_city_idx, axis=-1)
                    
                    # Store the log_prob of the choice we just made
                    step_log_prob = tf.gather(log_probs, next_city_idx, batch_dims=1)
                    tours_log_probs_ta = tours_log_probs_ta.write(step, step_log_prob)
                else:
                    # Inference: Greedy
                    next_city_idx = tf.argmax(log_probs, axis=1, output_type=tf.int64)
                    
                    # FIX: MUST write to log_probs_ta to keep graph consistent
                    tours_log_probs_ta = tours_log_probs_ta.write(step, zeros)

                # 5. Set the *next* city as the *current* city for the next loop iteration
                current_city_idx = next_city_idx
            
            else:
                # 4b. This is the last step (step == n_cities - 1). No more choices.
                
                # FIX: MUST write to log_probs_ta to keep graph consistent
                tours_log_probs_ta = tours_log_probs_ta.write(step, zeros)

        # Stack the results
        tour_indices = tf.transpose(tour_indices_ta.stack())
        tour_lengths = self.calculate_tour_lengths(inputs, tour_indices)

        if return_log_probs:
            tours_log_probs = tf.transpose(tours_log_probs_ta.stack())
            # Return 3 items for consistency
            return tours_log_probs, tour_lengths, encoder_output
        else:
            # Return 3 items for consistency
            return tour_indices, tour_lengths, encoder_output

    @tf.function
    def calculate_tour_lengths(self, coords, tour_indices):
        batch_size = tf.shape(coords)[0]
        n_cities = tf.shape(coords)[1]

        batch_indices = tf.range(batch_size)[:, tf.newaxis]
        batch_indices = tf.tile(batch_indices, [1, n_cities])
        
        gather_indices = tf.stack([batch_indices, tf.cast(tour_indices, tf.int32)], axis=-1)
        tour_coords = tf.gather_nd(coords, gather_indices)

        # Roll coordinates to calculate segment lengths (city 1-2, 2-3, ..., N-1)
        rolled_coords = tf.roll(tour_coords, shift=-1, axis=1)
        segment_lengths = tf.sqrt(tf.reduce_sum(tf.square(tour_coords - rolled_coords), axis=2))
        
        # Sum all segment lengths for the total tour distance
        tour_lengths = tf.reduce_sum(segment_lengths, axis=1)
        return tour_lengths