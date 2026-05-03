"""
Task 12: Transformer Architecture — Built from Scratch in PyTorch

This implements the complete Transformer from "Attention Is All You Need" (Vaswani et al., 2017).
Every component is coded from scratch with detailed explanations.

Components:
1. Multi-Head Self-Attention
2. Position-wise Feed-Forward Network
3. Positional Encoding (sinusoidal)
4. Encoder Layer + Encoder Stack
5. Decoder Layer + Decoder Stack
6. Full Transformer (Encoder-Decoder)

Run: python transformer_from_scratch.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. SCALED DOT-PRODUCT ATTENTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def scaled_dot_product_attention(
    query: torch.Tensor,    # (batch, heads, seq_len, d_k)
    key: torch.Tensor,      # (batch, heads, seq_len, d_k)
    value: torch.Tensor,    # (batch, heads, seq_len, d_v)
    mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    Core attention mechanism:
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

    Why scale by sqrt(d_k)?
    → Without scaling, dot products grow large for high d_k,
      pushing softmax into regions with tiny gradients (vanishing gradient problem).
    """
    d_k = query.size(-1)

    # Step 1: Compute attention scores — QK^T / sqrt(d_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # Shape: (batch, heads, seq_len_q, seq_len_k)

    # Step 2: Apply mask (for decoder — prevent attending to future tokens)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    # Step 3: Softmax — convert scores to probabilities
    attention_weights = F.softmax(scores, dim=-1)

    # Step 4: Weighted sum of values
    output = torch.matmul(attention_weights, value)
    # Shape: (batch, heads, seq_len_q, d_v)

    return output


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. MULTI-HEAD ATTENTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MultiHeadAttention(nn.Module):
    """
    Instead of one big attention, we run h parallel attention "heads"
    each with d_k = d_model / h dimensions. This lets the model attend
    to information from different representation subspaces.

    MultiHead(Q,K,V) = Concat(head_1, ..., head_h) * W_O
    where head_i = Attention(Q*W_Qi, K*W_Ki, V*W_Vi)
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head

        # Linear projections for Q, K, V and output
        self.W_q = nn.Linear(d_model, d_model)  # Projects input → queries
        self.W_k = nn.Linear(d_model, d_model)  # Projects input → keys
        self.W_v = nn.Linear(d_model, d_model)  # Projects input → values
        self.W_o = nn.Linear(d_model, d_model)  # Projects concatenated heads → output

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Step 1: Linear projections
        Q = self.W_q(query)  # (batch, seq_len, d_model)
        K = self.W_k(key)
        V = self.W_v(value)

        # Step 2: Split into multiple heads
        # Reshape: (batch, seq_len, d_model) → (batch, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Step 3: Apply attention
        attn_output = scaled_dot_product_attention(Q, K, V, mask)

        # Step 4: Concatenate heads
        # (batch, heads, seq_len, d_k) → (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )

        # Step 5: Final linear projection
        return self.W_o(attn_output)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. POSITION-WISE FEED-FORWARD NETWORK
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class FeedForward(nn.Module):
    """
    FFN(x) = ReLU(x * W_1 + b_1) * W_2 + b_2

    Two linear layers with ReLU activation.
    d_ff is typically 4x d_model (e.g., 2048 for d_model=512).
    This is where most of the model's "thinking" happens.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. POSITIONAL ENCODING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PositionalEncoding(nn.Module):
    """
    Since transformers have no recurrence/convolution, they don't know
    word ORDER. We inject position information using sinusoidal functions:

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Why sin/cos? They allow the model to easily learn relative positions
    because PE(pos+k) can be expressed as a linear function of PE(pos).
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)  # Even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd dimensions

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. ENCODER LAYER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class EncoderLayer(nn.Module):
    """
    One encoder layer:
    1. Multi-head self-attention (each token attends to all other tokens)
    2. Add & Norm (residual connection + layer normalization)
    3. Feed-forward network
    4. Add & Norm
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention + residual + norm
        attn_out = self.self_attn(x, x, x, mask)  # Q=K=V=x (self-attention)
        x = self.norm1(x + self.dropout1(attn_out))

        # Feed-forward + residual + norm
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_out))

        return x


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. DECODER LAYER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DecoderLayer(nn.Module):
    """
    One decoder layer:
    1. Masked multi-head self-attention (can only attend to earlier positions)
    2. Add & Norm
    3. Cross-attention (attends to encoder output)
    4. Add & Norm
    5. Feed-forward
    6. Add & Norm
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Masked self-attention (prevents looking ahead)
        attn_out = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(attn_out))

        # Cross-attention (Q from decoder, K&V from encoder)
        cross_out = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout2(cross_out))

        # Feed-forward
        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_out))

        return x


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. FULL TRANSFORMER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class Transformer(nn.Module):
    """
    Complete Transformer model (Encoder-Decoder architecture).

    Architecture:
    Input → Embedding → Positional Encoding → Encoder Stack → 
    Output → Embedding → Positional Encoding → Decoder Stack → Linear → Softmax

    Default config matches the original paper:
    d_model=512, num_heads=8, num_layers=6, d_ff=2048
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 5000,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)

        # Encoder stack
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])

        # Decoder stack
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])

        # Final output projection
        self.output_linear = nn.Linear(d_model, tgt_vocab_size)

        self.d_model = d_model

    def encode(self, src, src_mask=None):
        """Encode source sequence."""
        x = self.src_embedding(src) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        """Decode target sequence using encoder output."""
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return x

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        output = self.output_linear(decoder_output)
        return output

    @staticmethod
    def generate_causal_mask(size: int) -> torch.Tensor:
        """
        Create a causal (look-ahead) mask for the decoder.
        Prevents position i from attending to positions > i.
        
        Example for size=4:
        [[1, 0, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 0],
         [1, 1, 1, 1]]
        """
        return torch.tril(torch.ones(size, size)).unsqueeze(0).unsqueeze(0)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DEMO
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    # Create a small transformer
    model = Transformer(
        src_vocab_size=1000,
        tgt_vocab_size=1000,
        d_model=256,
        num_heads=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        d_ff=512,
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Transformer created: {total_params:,} parameters")
    print(f"  Encoder: {sum(p.numel() for l in model.encoder_layers for p in l.parameters()):,}")
    print(f"  Decoder: {sum(p.numel() for l in model.decoder_layers for p in l.parameters()):,}")

    # Test forward pass
    src = torch.randint(0, 1000, (2, 10))   # batch=2, src_len=10
    tgt = torch.randint(0, 1000, (2, 8))    # batch=2, tgt_len=8
    tgt_mask = Transformer.generate_causal_mask(8)

    output = model(src, tgt, tgt_mask=tgt_mask)
    print(f"\n  Input:  src={list(src.shape)}, tgt={list(tgt.shape)}")
    print(f"  Output: {list(output.shape)}")
    print(f"  (batch=2, tgt_len=8, vocab=1000)")
    print("\nTransformer from scratch — working!")
