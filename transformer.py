import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ─── Multi-Head Self-Attention ───────────────────────────────────────────────

class MultiHeadSelfAttention(nn.Module):
    """Scaled dot-product multi-head self-attention (implemented from scratch)."""

    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads  # dimension per head

        # Linear projections for Q, K, V and the final output
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: optional (batch, 1, 1, seq_len) – 0 for positions to attend, 1 for positions to mask
        Returns:
            out: (batch, seq_len, d_model)
            attn_weights: (batch, num_heads, seq_len, seq_len)
        """
        B, T, C = x.size()

        # Project to Q, K, V then split into heads
        # (B, T, C) -> (B, T, num_heads, head_dim) -> (B, num_heads, T, head_dim)
        Q = self.W_q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # (B, num_heads, T, T)

        # Apply mask (set masked positions to -inf so softmax gives ~0)
        if mask is not None:
            scores = scores.masked_fill(mask == 1, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)  # (B, num_heads, T, T)

        # Weighted sum of values
        out = torch.matmul(attn_weights, V)  # (B, num_heads, T, head_dim)

        # Concatenate heads and project
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, d_model)
        out = self.W_o(out)

        return out, attn_weights


# ─── Feedforward Network ────────────────────────────────────────────────────

class FeedForwardNetwork(nn.Module):
    """Position-wise feedforward network: d_model → d_ff → d_model with ReLU."""

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


# ─── Single Encoder Layer ───────────────────────────────────────────────────

class EncoderLayer(nn.Module):
    """One transformer encoder block: self-attention + feedforward,
       each with a residual connection and LayerNorm."""

    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attention = MultiHeadSelfAttention(d_model, num_heads)
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # Self-attention with residual + LayerNorm
        attn_out, attn_weights = self.attention(x, mask)
        x = self.ln1(x + attn_out)

        # Feedforward with residual + LayerNorm
        ff_out = self.ffn(x)
        x = self.ln2(x + ff_out)

        return x, attn_weights


# ─── Full Transformer Encoder ───────────────────────────────────────────────

class TransformerEncoder(nn.Module):
    """Transformer encoder: token embeddings + positional embeddings →
       num_layers × EncoderLayer.  Returns per-token embeddings and
       a list of attention matrices (one per layer) for sanity checking."""

    def __init__(self, vocab_size, d_model, num_heads, num_layers,
                 max_seq_len, d_ff=None):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model  # default feed-forward expansion

        # 1. Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)

        # 2. Learnable positional embeddings
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        # 6. Stack encoder layers
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]
        )

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len) – token indices
            mask: optional padding mask; if None, one is built from x==0
        Returns:
            embeddings: (batch, seq_len, d_model)
            attn_maps: list of (batch, num_heads, seq_len, seq_len)
        """
        B, T = x.size()

        # Build padding mask if not provided: shape (B, 1, 1, T)
        if mask is None:
            mask = (x == 0).unsqueeze(1).unsqueeze(2)  # True where padding

        # 1 & 2. Token + positional embeddings
        positions = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)
        embeddings = self.token_embedding(x) + self.position_embedding(positions)

        # 3–6. Pass through each encoder layer, collecting attention maps
        attn_maps = []
        for layer in self.layers:
            embeddings, attn_weights = layer(embeddings, mask)
            attn_maps.append(attn_weights)

        return embeddings, attn_maps


# ─── Feedforward Classifier ─────────────────────────────────────────────────

class FeedForwardClassifier(nn.Module):
    """Simple one-hidden-layer feedforward classifier that sits on top of
    the transformer encoder.  It takes the mean-pooled encoder output and
    predicts which politician spoke the segment."""

    def __init__(self, d_model, hidden_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, encoder_output, mask=None):
        """
        Args:
            encoder_output: (batch, seq_len, d_model) from TransformerEncoder
            mask: optional (batch, seq_len) padding mask – True/1 where padded
        Returns:
            logits: (batch, num_classes)
        """
        # Mean-pool over the sequence dimension, ignoring padding tokens
        if mask is not None:
            # mask: (B, T) -> (B, T, 1) for broadcasting
            mask_expanded = (~mask).unsqueeze(-1).float()        # 1 for real tokens, 0 for pad
            summed = (encoder_output * mask_expanded).sum(dim=1) # (B, d_model)
            lengths = mask_expanded.sum(dim=1).clamp(min=1)      # (B, 1)
            pooled = summed / lengths                            # (B, d_model)
        else:
            pooled = encoder_output.mean(dim=1)                  # (B, d_model)

        # Feedforward: d_model → hidden_dim → num_classes
        x = F.relu(self.fc1(pooled))
        logits = self.fc2(x)
        return logits


# ═══════════════════════════════════════════════════════════════════════════
#  PART 2 — Transformer Decoder (GPT-like, autoregressive)
# ═══════════════════════════════════════════════════════════════════════════

# ─── Masked (Causal) Multi-Head Self-Attention ───────────────────────────

class CausalMultiHeadSelfAttention(nn.Module):
    """Scaled dot-product multi-head self-attention with a causal mask
    so that position i can only attend to positions ≤ i.
    If window_size is set, attention is further restricted to the
    window_size most recent positions (local window attention).
    If use_alibi is True, ALiBi position biases are added to attention
    scores instead of relying on positional embeddings.
    If block_sparse_size is set, attention is restricted to tokens within
    the same block and the immediately preceding block (blockwise sparse)."""

    def __init__(self, d_model, num_heads, max_seq_len, window_size=None,
                 use_alibi=False, block_sparse_size=None):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.window_size = window_size
        self.use_alibi = use_alibi
        self.block_sparse_size = block_sparse_size

        # Linear projections for Q, K, V and output
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # Build attention mask: True = positions to MASK OUT
        # Start with causal mask (upper triangle)
        positions = torch.arange(max_seq_len)
        # distance[i, j] = i - j  (positive means j is in the past)
        distance = positions.unsqueeze(1) - positions.unsqueeze(0)  # (T, T)

        # Causal constraint: mask out future (distance < 0)
        attn_mask = distance < 0  # True where j > i (future)

        # Window constraint: also mask out tokens beyond window_size
        if window_size is not None:
            attn_mask = attn_mask | (distance > window_size)  # True where too far in past

        # Block-sparse constraint: only attend within same or previous block
        if block_sparse_size is not None:
            block_ids = positions // block_sparse_size
            i_blocks = block_ids.unsqueeze(1)  # (T, 1)
            j_blocks = block_ids.unsqueeze(0)  # (1, T)
            same_block = (i_blocks == j_blocks)
            prev_block = (i_blocks - j_blocks == 1)
            block_allow = same_block | prev_block
            attn_mask = attn_mask | (~block_allow)  # mask out positions not in same/prev block

        self.register_buffer("attn_mask", attn_mask)  # (max_seq_len, max_seq_len)

        # ── ALiBi: per-head slopes (non-learnable) ────────────────────────
        if use_alibi:
            # Slopes follow the geometric sequence from the ALiBi paper:
            #   slope_h = 2^(-8h/H)  for h = 1, …, H
            slopes = torch.pow(2.0, -8.0 * torch.arange(1, num_heads + 1, dtype=torch.float32) / num_heads)
            self.register_buffer("alibi_slopes", slopes)  # (num_heads,)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            out: (batch, seq_len, d_model)
            attn_weights: (batch, num_heads, seq_len, seq_len)
        """
        B, T, C = x.size()

        # Project to Q, K, V then split into heads
        Q = self.W_q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # (B, num_heads, T, T)

        # ALiBi: add linear distance bias per head
        if self.use_alibi:
            positions = torch.arange(T, device=x.device)
            dist = positions.unsqueeze(0) - positions.unsqueeze(1)  # (T, T), dist[i,j] = j - i
            # bias = -|distance| but for causal (lower-triangular) dist[i,j]<=0 when j<=i
            # We want penalty = -slope * (i - j) for j <= i  (non-negative distance)
            alibi_bias = -dist.float()  # (T, T): positive for j < i (past tokens)
            # Multiply each head by its slope: slopes (H,) → (H, 1, 1)
            alibi_bias = self.alibi_slopes.view(self.num_heads, 1, 1) * alibi_bias.unsqueeze(0)  # (H, T, T)
            scores = scores + alibi_bias.unsqueeze(0)  # broadcast over batch: (B, H, T, T)

        # Apply causal (& window) mask: prevent attending to masked positions
        mask = self.attn_mask[:T, :T]  # crop to current seq length
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)  # (B, num_heads, T, T)

        # Weighted sum of values
        out = torch.matmul(attn_weights, V)  # (B, num_heads, T, head_dim)

        # Concatenate heads and project
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.W_o(out)

        return out, attn_weights


# ─── Single Decoder Layer ───────────────────────────────────────────────

class DecoderLayer(nn.Module):
    """One transformer decoder block: causal self-attention + feedforward,
       each with a residual connection and LayerNorm."""

    def __init__(self, d_model, num_heads, d_ff, max_seq_len, window_size=None,
                 use_alibi=False, block_sparse_size=None):
        super().__init__()
        self.attention = CausalMultiHeadSelfAttention(d_model, num_heads, max_seq_len,
                                                     window_size, use_alibi,
                                                     block_sparse_size)
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Causal self-attention with residual + LayerNorm
        attn_out, attn_weights = self.attention(x)
        x = self.ln1(x + attn_out)

        # Feedforward with residual + LayerNorm
        ff_out = self.ffn(x)
        x = self.ln2(x + ff_out)

        return x, attn_weights


# ─── Full Transformer Decoder (GPT-like) ────────────────────────────────

class TransformerDecoder(nn.Module):
    """Autoregressive transformer decoder: token embeddings + positional
    embeddings → num_layers × DecoderLayer → LayerNorm → linear head.

    Forward pass returns logits (B, T, vocab_size) and, when targets are
    provided, computes and returns the cross-entropy loss directly."""

    def __init__(self, vocab_size, d_model, num_heads, num_layers,
                 max_seq_len, d_ff=None, window_size=None, use_alibi=False,
                 block_sparse_size=None):
        super().__init__()
        if d_ff is None:
            d_ff = 100  # assignment specifies feedforward hidden dim = 100

        self.window_size = window_size
        self.use_alibi = use_alibi
        self.block_sparse_size = block_sparse_size

        # Token and positional embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)  # kept for equal param count

        # Stack of decoder layers
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, max_seq_len, window_size, use_alibi,
                          block_sparse_size)
             for _ in range(num_layers)]
        )

        # Final LayerNorm after all decoder blocks
        self.ln_final = nn.LayerNorm(d_model)

        # Linear projection to vocabulary
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x, targets=None):
        """
        Args:
            x: (batch, seq_len) – token indices
            targets: optional (batch, seq_len) – target token indices for loss
        Returns:
            If targets is None:  logits (B, T, vocab_size), attn_maps
            If targets provided: loss (scalar), attn_maps
        """
        B, T = x.size()

        # Token + positional embeddings (skip positional when using ALiBi)
        h = self.token_embedding(x)
        if not self.use_alibi:
            positions = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)
            h = h + self.position_embedding(positions)

        # Pass through each decoder layer, collecting attention maps
        attn_maps = []
        for layer in self.layers:
            h, attn_weights = layer(h)
            attn_maps.append(attn_weights)

        # Final LayerNorm + linear projection to vocab
        h = self.ln_final(h)
        logits = self.head(h)  # (B, T, vocab_size)

        if targets is not None:
            # Flatten for cross-entropy: (B*T, vocab_size) vs (B*T,)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return loss
        else:
            return logits, attn_maps