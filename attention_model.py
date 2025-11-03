import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class GraphEmbedding(nn.Module):
    """
    Initial embedding layer for city coordinates.
    Projects 2D coordinates to a higher-dimensional space.
    """
    def __init__(self, input_dim: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, n_cities, input_dim=2)
        return self.embedding(x)
        # Output shape: (batch_size, n_cities, embed_dim)

class MultiHeadAttention(nn.Module):
    """
    Standard Multi-Head Attention layer.
    Allows the model to jointly attend to information from different
    representation subspaces at different positions.
    """
    def __init__(self, embed_dim: int, n_heads: int):
        super().__init__()
        assert embed_dim % n_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        
        # Linear layers for Q, K, V projections
        self.Wq = nn.Linear(embed_dim, embed_dim)
        self.Wk = nn.Linear(embed_dim, embed_dim)
        self.Wv = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.Wo = nn.Linear(embed_dim, embed_dim)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, embed_dim)
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        # Output shape: (batch_size, n_heads, seq_len, head_dim)

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, n_heads, seq_len, head_dim)
        batch_size, _, seq_len, _ = x.shape
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        # Output shape: (batch_size, seq_len, embed_dim)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Project Q, K, V
        Q = self.Wq(q) # (batch_size, q_len, embed_dim)
        K = self.Wk(k) # (batch_size, k_len, embed_dim)
        V = self.Wv(v) # (batch_size, v_len, embed_dim)
        
        # Split into heads
        Q = self._split_heads(Q) # (batch_size, n_heads, q_len, head_dim)
        K = self._split_heads(K) # (batch_size, n_heads, k_len, head_dim)
        V = self._split_heads(V) # (batch_size, n_heads, v_len, head_dim)
        
        # Scaled Dot-Product Attention
        # (batch_size, n_heads, q_len, head_dim) @ (batch_size, n_heads, head_dim, k_len)
        # -> (batch_size, n_heads, q_len, k_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            # Apply mask (e.g., for decoder or padding)
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention weights to V
        # (batch_size, n_heads, q_len, k_len) @ (batch_size, n_heads, v_len, head_dim)
        # -> (batch_size, n_heads, q_len, head_dim)
        # Note: k_len == v_len
        context = torch.matmul(attention_weights, V)
        
        # Combine heads
        context = self._combine_heads(context) # (batch_size, q_len, embed_dim)
        
        # Final linear projection
        output = self.Wo(context) # (batch_size, q_len, embed_dim)
        
        return output

class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    Applies two linear transformations with a ReLU activation in between.
    """
    def __init__(self, embed_dim: int, ff_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, embed_dim)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
        # Output shape: (batch_size, seq_len, embed_dim)

class EncoderLayer(nn.Module):
    """
    A single layer of the Transformer Encoder.
    Consists of Multi-Head Attention and Feed-Forward sublayers,
    each followed by Add & Norm (Residual Connection + Layer Normalization).
    """
    def __init__(self, embed_dim: int, n_heads: int, ff_dim: int):
        super().__init__()
        self.mha = MultiHeadAttention(embed_dim, n_heads)
        self.ff = FeedForward(embed_dim, ff_dim)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, n_cities, embed_dim)
        
        # Multi-Head Attention sublayer
        attn_output = self.mha(x, x, x) # Self-attention
        x = self.norm1(x + attn_output) # Add & Norm
        
        # Feed-Forward sublayer
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output) # Add & Norm
        
        return x
        # Output shape: (batch_size, n_cities, embed_dim)

class AttentionEncoder(nn.Module):
    """
    The Transformer Encoder.
    Stacks multiple EncoderLayers.
    """
    def __init__(self, n_layers: int, embed_dim: int, n_heads: int, ff_dim: int):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(embed_dim, n_heads, ff_dim) for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, n_cities, embed_dim)
        for layer in self.layers:
            x = layer(x)
        return x
        # Output shape: (batch_size, n_cities, embed_dim)

class PointerNetworkDecoder(nn.Module):
    """
    Decoder using Pointer Network mechanism.
    At each step, it uses attention to point to one of the input cities.
    """
    def __init__(self, embed_dim: int, n_heads: int):
        super().__init__()
        self.mha = MultiHeadAttention(embed_dim, n_heads)
        
        # For creating the query vector in MHA
        # This will combine graph embedding, last city embedding, and first city embedding
        self.query_projection = nn.Linear(embed_dim * 3, embed_dim)
        
        # Clipping value for log-softmax, prevents numerical instability
        self.tanh_clipping = 10.0
        
        # Temperature parameter for exploration vs exploitation (can be learned or fixed)
        self.softmax_temperature = 1.0

    def forward(
        self,
        encoder_output: torch.Tensor,
        graph_embedding_pooled: torch.Tensor, # Pooled embedding of the whole graph (e.g., mean)
        current_city_embedding: torch.Tensor,
        first_city_embedding: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # encoder_output shape: (batch_size, n_cities, embed_dim)
        # graph_embedding_pooled shape: (batch_size, embed_dim)
        # current_city_embedding shape: (batch_size, embed_dim)
        # first_city_embedding shape: (batch_size, embed_dim)
        # mask shape: (batch_size, n_cities) - 1 for allowed, 0 for visited/masked

        # Unsqueeze embeddings to add sequence length dimension (of 1)
        graph_emb_unsqueezed = graph_embedding_pooled.unsqueeze(1) # (bs, 1, embed_dim)
        current_emb_unsqueezed = current_city_embedding.unsqueeze(1) # (bs, 1, embed_dim)
        first_emb_unsqueezed = first_city_embedding.unsqueeze(1) # (bs, 1, embed_dim)
        
        # Concatenate context features to form the MHA query
        query_input = torch.cat(
            (graph_emb_unsqueezed, current_emb_unsqueezed, first_emb_unsqueezed),
            dim=-1 # Concatenate along the embedding dimension
        )
        # query_input shape: (batch_size, 1, embed_dim * 3)

        # Project concatenated features to get the final query vector
        query = self.query_projection(query_input) # (batch_size, 1, embed_dim)

        # Attention mechanism
        # Q = query from context, K = V = encoder output (city embeddings)
        # attn_output shape: (batch_size, 1, embed_dim)
        attn_output = self.mha(query, encoder_output, encoder_output, mask=mask.unsqueeze(1).unsqueeze(2)) 
        # Note: Mask needs shape (bs, n_heads=1 (broadcasted), q_len=1, k_len=n_cities) for MHA

        # --- Pointer Mechanism ---
        # Calculate scores (logits) by projecting the attention output onto encoder outputs
        # This is equivalent to the final attention score calculation *before* softmax
        # We need the compatibility between the context vector (attn_output) and each node embedding
        
        # Re-compute simplified attention scores (dot product) using the attention output
        # Q = attn_output, K = encoder_output
        # attn_output: (bs, 1, embed_dim)
        # encoder_output.transpose(-2, -1): (bs, embed_dim, n_cities)
        # scores shape: (bs, 1, n_cities)
        scores = torch.matmul(attn_output, encoder_output.transpose(-2, -1)) / math.sqrt(attn_output.size(-1))
        
        # Apply clipping (as done in the paper)
        scores = self.tanh_clipping * torch.tanh(scores)

        # Apply mask *before* softmax
        # Ensure mask has shape (bs, 1, n_cities) for broadcasting
        scores_masked = scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))
        
        # Apply temperature scaling before softmax for exploration
        log_probs = F.log_softmax(scores_masked / self.softmax_temperature, dim=-1) # (bs, 1, n_cities)
        
        # Squeeze out the middle dimension
        log_probs = log_probs.squeeze(1) # (bs, n_cities)
        
        return log_probs # Return log probabilities for training (using REINFORCE)

class AttentionModel(nn.Module):
    """
    The main TSP Solver model combining Encoder and Decoder.
    """
    def __init__(
        self,
        embed_dim: int = 128,
        n_encode_layers: int = 3,
        n_heads: int = 8,
        ff_dim: int = 512, # FeedForward hidden dim (usually 4*embed_dim)
        tanh_clipping: float = 10.0,
        softmax_temperature: float = 1.0
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Initial embedding layer
        self.embedding = GraphEmbedding(input_dim=2, embed_dim=embed_dim)
        
        # Encoder
        self.encoder = AttentionEncoder(n_encode_layers, embed_dim, n_heads, ff_dim)
        
        # Decoder
        self.decoder = PointerNetworkDecoder(embed_dim, n_heads)
        
        # Parameters from Decoder moved here for convenience
        self.decoder.tanh_clipping = tanh_clipping
        self.decoder.softmax_temperature = softmax_temperature

    def _get_pooled_embedding(self, embeddings: torch.Tensor) -> torch.Tensor:
        # Simple mean pooling over the city dimension
        return embeddings.mean(dim=1)

    def forward(
        self,
        inputs: torch.Tensor,
        return_log_probs: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model. Generates tours step-by-step.

        Args:
            inputs (torch.Tensor): City coordinates, shape (batch_size, n_cities, 2).
            return_log_probs (bool): If True, returns log probabilities (for training).
                                      If False, returns sampled tour indices (for inference).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - tour_indices or tours_log_probs: Shape (batch_size, n_cities). Indices if sampling, log-probs if training.
            - tour_lengths: Calculated lengths of the generated tours, shape (batch_size,).
            - encoder_output: For potential use in baseline/critic, shape (batch_size, n_cities, embed_dim).
        """
        batch_size, n_cities, _ = inputs.shape
        device = inputs.device

        # 1. Initial Embedding
        # shape: (batch_size, n_cities, embed_dim)
        initial_embeddings = self.embedding(inputs)

        # 2. Encoder
        # shape: (batch_size, n_cities, embed_dim)
        encoder_output = self.encoder(initial_embeddings)
        
        # 3. Pooled Graph Embedding (for decoder context)
        # shape: (batch_size, embed_dim)
        graph_embedding_pooled = self._get_pooled_embedding(encoder_output)

        # 4. Decoder Step-by-Step Tour Construction
        tours_log_probs = [] # Stores log_probs for each step (for training)
        tour_indices = []    # Stores chosen city indices for each step

        # Initialize mask (all cities available initially)
        mask = torch.ones(batch_size, n_cities, device=device)

        # Start from a fixed city (or learn starting city - simpler to fix for now)
        # For simplicity, let's assume we start at city 0
        current_city_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
        first_city_embedding = encoder_output[:, 0, :] # Embedding of the starting city

        # Collect embeddings based on current indices
        current_city_embedding = torch.gather(
            encoder_output, 1, current_city_idx.view(-1, 1, 1).expand(-1, -1, self.embed_dim)
        ).squeeze(1) # Get embedding of the current city

        for step in range(n_cities):
            # Update mask: Mask the current city
            mask.scatter_(1, current_city_idx.unsqueeze(1), 0)

            # Decoder forward pass
            log_probs = self.decoder(
                encoder_output,
                graph_embedding_pooled,
                current_city_embedding,
                first_city_embedding,
                mask
            )
            # log_probs shape: (batch_size, n_cities)

            if return_log_probs:
                # Training: Sample actions based on probabilities
                probs = torch.exp(log_probs)
                # Ensure no NaN values in probabilities due to masking
                probs[mask == 0] = 0.0 
                # Renormalize probabilities if needed
                probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-8) 

                next_city_idx = torch.multinomial(probs, 1).squeeze(1) # Sample based on probs
                
                # Get the log_prob of the *chosen* action
                step_log_prob = torch.gather(log_probs, 1, next_city_idx.unsqueeze(-1)).squeeze(-1)
                tours_log_probs.append(step_log_prob)

            else:
                # Inference: Choose the most likely city (greedy)
                next_city_idx = torch.argmax(log_probs, dim=1) # Greedy selection

            # Store the chosen city index
            tour_indices.append(next_city_idx)
            
            # Update current city for the next step
            current_city_idx = next_city_idx
            current_city_embedding = torch.gather(
                encoder_output, 1, current_city_idx.view(-1, 1, 1).expand(-1, -1, self.embed_dim)
            ).squeeze(1)

        # Stack results from all steps
        if return_log_probs:
            tours_log_probs = torch.stack(tours_log_probs, dim=1) # (batch_size, n_cities)
        tour_indices = torch.stack(tour_indices, dim=1)         # (batch_size, n_cities)

        # Calculate tour lengths
        tour_lengths = self.calculate_tour_lengths(inputs, tour_indices)

        if return_log_probs:
            return tours_log_probs, tour_lengths, encoder_output
        else:
            return tour_indices, tour_lengths, encoder_output
            
    def calculate_tour_lengths(self, coords: torch.Tensor, tour_indices: torch.Tensor) -> torch.Tensor:
        """Calculate the length of tours given coordinates and indices."""
        batch_size, n_cities, _ = coords.shape
        device = coords.device

        # Gather coordinates in tour order
        # Shape: (batch_size, n_cities, 2)
        tour_coords = torch.gather(
            coords, 1, tour_indices.unsqueeze(-1).expand(-1, -1, 2)
        )

        # Calculate step distances (including return to start)
        # Shift coordinates to get pairs (city_i, city_{i+1})
        rolled_coords = torch.roll(tour_coords, shifts=-1, dims=1)
        
        # Calculate Euclidean distances between consecutive cities
        # Shape: (batch_size, n_cities)
        segment_lengths = torch.sqrt(
            ((tour_coords - rolled_coords)**2).sum(dim=2)
        )
        
        # Sum segment lengths to get total tour length
        # Shape: (batch_size,)
        tour_lengths = segment_lengths.sum(dim=1)
        
        return tour_lengths

# --- Example Usage (for testing) ---
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = AttentionModel(embed_dim=128, n_encode_layers=3).to(device)
    
    # Generate dummy data
    batch_size = 4
    n_cities = 10
    dummy_input = torch.rand(batch_size, n_cities, 2, device=device) # Random coords [0, 1]

    # Test training mode
    model.train()
    tours_log_probs, tour_lengths, _ = model(dummy_input, return_log_probs=True)
    print("\n--- Training Mode Output ---")
    print("Log Probs shape:", tours_log_probs.shape) # Should be (batch_size, n_cities)
    print("Tour Lengths shape:", tour_lengths.shape) # Should be (batch_size,)
    print("Example Tour Lengths:", tour_lengths.detach().cpu().numpy())

    # Test inference mode
    model.eval()
    with torch.no_grad():
        tour_indices, tour_lengths, _ = model(dummy_input, return_log_probs=False)
    print("\n--- Inference Mode Output ---")
    print("Tour Indices shape:", tour_indices.shape) # Should be (batch_size, n_cities)
    print("Tour Lengths shape:", tour_lengths.shape) # Should be (batch_size,)
    print("Example Tour Indices (first batch):\n", tour_indices[0].detach().cpu().numpy())
    print("Example Tour Lengths:", tour_lengths.detach().cpu().numpy())
    
    # Check if lengths match calculated lengths
    recalculated_lengths = model.calculate_tour_lengths(dummy_input, tour_indices)
    print("\nRecalculated Lengths:", recalculated_lengths.detach().cpu().numpy())
    print("Lengths Match:", torch.allclose(tour_lengths, recalculated_lengths))