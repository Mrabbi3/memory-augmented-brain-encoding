"""
Memory Module for TRIBE v2 Extension
=====================================
Author: MD Rabbi
Project: Memory-Augmented Brain Encoding

This module adds long-range context to TRIBE v2 by storing compressed
representations from past context windows and retrieving relevant ones
via cosine similarity, then integrating them via cross-attention.

Insertion point in TRIBE v2's forward() pass:
    1. aggregate_features()    → [B, T, 1152]
    2. transformer_forward()   → [B, T, 1152]
    >>> MEMORY MODULE HERE <<<
    3. low_rank_head()          → [B, T, 2048]
    4. predictor()              → [B, 20484, T]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MemoryBuffer:
    """Stores compressed latent representations from past context windows.
    
    Each window's output from the transformer encoder gets mean-pooled
    across time to produce a single summary vector (dim=hidden_dim).
    These summaries are stored in a FIFO buffer per timeline.
    
    During inference, the current window queries the buffer via cosine
    similarity and retrieves the top-k most relevant past representations.
    
    Args:
        buffer_size: Maximum number of past windows to store
        hidden_dim: Dimension of each stored vector (1152 for TRIBE v2)
        top_k: Number of vectors to retrieve per query
    """
    
    def __init__(self, buffer_size: int = 100, hidden_dim: int = 1152, top_k: int = 5):
        self.buffer_size = buffer_size
        self.hidden_dim = hidden_dim
        self.top_k = top_k
        self.reset()
    
    def reset(self):
        """Clear the buffer. Call between different timelines/movies."""
        self.buffer = []  # List of tensors, each shape [hidden_dim]
    
    def store(self, window_latents: torch.Tensor):
        """Store a compressed representation of the current window.
        
        Args:
            window_latents: Transformer output for current window [B, T, H]
                           We mean-pool across time to get [B, H], then
                           store the first batch element (single timeline).
        """
        # Mean-pool across time dimension
        summary = window_latents.mean(dim=1)  # [B, H]
        
        # Detach to prevent gradient flow through entire history
        summary = summary.detach()
        
        # Store first batch element (assumes single timeline per batch)
        self.buffer.append(summary[0])  # [H]
        
        # Maintain FIFO: remove oldest if over capacity
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
    
    def retrieve(self, query: torch.Tensor) -> Optional[torch.Tensor]:
        """Retrieve the top-k most similar past representations.
        
        Args:
            query: Current window's mean-pooled latent [B, H]
        
        Returns:
            Retrieved memories tensor [B, top_k, H] or None if buffer empty
        """
        if len(self.buffer) == 0:
            return None
        
        # Stack buffer into matrix [N, H] where N = number of stored windows
        buffer_tensor = torch.stack(self.buffer)  # [N, H]
        buffer_tensor = buffer_tensor.to(query.device)
        
        # Compute cosine similarity between query and all stored vectors
        # query[0] shape: [H], buffer_tensor shape: [N, H]
        query_vec = query[0]  # Use first batch element
        similarities = F.cosine_similarity(
            query_vec.unsqueeze(0),  # [1, H]
            buffer_tensor,           # [N, H]
            dim=1
        )  # [N]
        
        # Get top-k indices
        k = min(self.top_k, len(self.buffer))
        top_indices = similarities.topk(k).indices  # [k]
        
        # Retrieve top-k vectors
        retrieved = buffer_tensor[top_indices]  # [k, H]
        
        # Expand for batch dimension [B, k, H]
        retrieved = retrieved.unsqueeze(0).expand(query.shape[0], -1, -1)
        
        return retrieved
    
    @property
    def size(self) -> int:
        """Number of windows currently stored."""
        return len(self.buffer)
    
    def __repr__(self):
        return f"MemoryBuffer(stored={self.size}/{self.buffer_size}, dim={self.hidden_dim}, top_k={self.top_k})"


class MemoryAttention(nn.Module):
    """Cross-attention layer that integrates retrieved memories.
    
    Current window latents (Q) attend to retrieved memory vectors (K, V).
    The output is added residually to the current latents, preserving
    the original signal while enriching it with past context.
    
    Args:
        hidden_dim: Dimension of input/output (1152 for TRIBE v2)
        num_heads: Number of attention heads
        dropout: Attention dropout rate
    """
    
    def __init__(self, hidden_dim: int = 1152, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        assert hidden_dim % num_heads == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        
        self.head_dim = hidden_dim // num_heads
        
        # Query projection (from current window)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Key and Value projections (from retrieved memories)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Layer norm (applied before attention, pre-norm style)
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Learnable gate to control how much memory influences output
        # Initialized to 0 so the model starts as vanilla TRIBE v2
        self.gate = nn.Parameter(torch.zeros(1))
    
    def forward(self, x: torch.Tensor, memories: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            x: Current window latents [B, T, H] from transformer output
            memories: Retrieved past representations [B, K, H] or None
        
        Returns:
            Enriched latents [B, T, H] (same shape as input)
        """
        # If no memories available, return input unchanged
        if memories is None:
            return x
        
        B, T, H = x.shape
        K = memories.shape[1]  # number of retrieved memories
        
        # Pre-norm
        x_normed = self.norm(x)
        
        # Project queries from current window
        Q = self.q_proj(x_normed)             # [B, T, H]
        
        # Project keys and values from memories
        K_proj = self.k_proj(memories)         # [B, K, H]
        V_proj = self.v_proj(memories)         # [B, K, H]
        
        # Reshape for multi-head attention
        # [B, T, H] → [B, num_heads, T, head_dim]
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K_proj = K_proj.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        V_proj = V_proj.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn_weights = torch.matmul(Q, K_proj.transpose(-2, -1)) * scale  # [B, heads, T, K]
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V_proj)  # [B, heads, T, head_dim]
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, H)  # [B, T, H]
        
        # Output projection
        attn_output = self.out_proj(attn_output)  # [B, T, H]
        
        # Gated residual connection
        # sigmoid(gate) starts near 0.5, but gate is initialized to 0
        # so tanh(gate) starts at 0 → no memory influence initially
        gate_value = torch.tanh(self.gate)
        
        # Residual: x + gate * attention_output
        output = x + gate_value * attn_output
        
        return output


class MemoryAugmentedEncoder:
    """Wraps a TRIBE v2 model with memory capabilities.
    
    This is the high-level interface you'll use. It:
    1. Takes the TRIBE v2 model
    2. Adds a MemoryBuffer and MemoryAttention
    3. Modifies the forward pass to store/retrieve memories
    
    Usage:
        model = TribeModel.from_pretrained('facebook/tribev2')
        memory_encoder = MemoryAugmentedEncoder(model._model)
        
        # Process windows sequentially
        for window_batch in timeline_windows:
            predictions = memory_encoder.forward_with_memory(window_batch)
        
        # Reset between different movies/timelines
        memory_encoder.reset_memory()
    
    Args:
        brain_model: The FmriEncoderModel from TRIBE v2
        buffer_size: How many past windows to store
        top_k: How many past windows to retrieve
        num_heads: Attention heads in memory attention
    """
    
    def __init__(
        self,
        brain_model: nn.Module,
        buffer_size: int = 100,
        top_k: int = 5,
        num_heads: int = 8,
    ):
        self.brain_model = brain_model
        self.hidden_dim = brain_model.config.hidden  # 1152
        
        # Create memory components
        self.memory_buffer = MemoryBuffer(
            buffer_size=buffer_size,
            hidden_dim=self.hidden_dim,
            top_k=top_k,
        )
        
        self.memory_attention = MemoryAttention(
            hidden_dim=self.hidden_dim,
            num_heads=num_heads,
        ).to(brain_model.device)
    
    def reset_memory(self):
        """Reset the memory buffer. Call between timelines."""
        self.memory_buffer.reset()
    
    def forward_with_memory(self, batch, pool_outputs: bool = True) -> torch.Tensor:
        """Modified forward pass that includes memory retrieval.
        
        This follows the exact same pipeline as FmriEncoderModel.forward(),
        but inserts memory operations between transformer and low_rank_head.
        
        Args:
            batch: SegmentData batch from the dataloader
            pool_outputs: Whether to pool outputs (default True)
        
        Returns:
            Predicted brain activity [B, n_outputs, T']
        """
        model = self.brain_model
        
        # === Step 1: Aggregate features (unchanged) ===
        x = model.aggregate_features(batch)  # [B, T, 1152]
        subject_id = batch.data.get("subject_id", None)
        
        # Temporal smoothing if configured
        if hasattr(model, "temporal_smoothing"):
            x = model.temporal_smoothing(x.transpose(1, 2)).transpose(1, 2)
        
        # === Step 2: Transformer forward (unchanged) ===
        if not model.config.linear_baseline:
            x = model.transformer_forward(x, subject_id)  # [B, T, 1152]
        
        # === Step 3: MEMORY OPERATIONS (NEW!) ===
        # 3a. Query the memory buffer with current window's mean representation
        query = x.mean(dim=1, keepdim=False)  # [B, H]
        retrieved_memories = self.memory_buffer.retrieve(query)  # [B, K, H] or None
        
        # 3b. Cross-attend current window to retrieved memories
        x = self.memory_attention(x, retrieved_memories)  # [B, T, 1152] (unchanged shape)
        
        # 3c. Store current window in buffer for future retrieval
        self.memory_buffer.store(x)
        
        # === Step 4: Low-rank head (unchanged) ===
        x = x.transpose(1, 2)  # [B, H, T]
        if model.config.low_rank_head is not None:
            x = model.low_rank_head(x.transpose(1, 2)).transpose(1, 2)
        
        # === Step 5: Subject prediction (unchanged) ===
        x = model.predictor(x, subject_id)  # [B, O, T]
        
        # === Step 6: Temporal pooling (unchanged) ===
        if pool_outputs:
            out = model.pooler(x)  # [B, O, T']
        else:
            out = x
        
        return out
    
    def get_memory_stats(self) -> dict:
        """Return current memory buffer statistics."""
        return {
            "buffer_size": self.memory_buffer.size,
            "buffer_capacity": self.memory_buffer.buffer_size,
            "top_k": self.memory_buffer.top_k,
            "gate_value": torch.tanh(self.memory_attention.gate).item(),
        }


# ============================================================
# Alternative Memory Strategies (for ablation experiments)
# ============================================================

class SlidingWindowBuffer:
    """Strategy B: Simply extend the context by concatenating recent windows.
    
    Instead of retrieval, just keep the last N windows' representations
    and concatenate them. Simpler but uses more memory.
    """
    
    def __init__(self, n_windows: int = 5, hidden_dim: int = 1152):
        self.n_windows = n_windows
        self.hidden_dim = hidden_dim
        self.reset()
    
    def reset(self):
        self.buffer = []
    
    def store(self, window_latents: torch.Tensor):
        summary = window_latents.mean(dim=1).detach()[0]  # [H]
        self.buffer.append(summary)
        if len(self.buffer) > self.n_windows:
            self.buffer.pop(0)
    
    def retrieve(self, query: torch.Tensor) -> Optional[torch.Tensor]:
        if len(self.buffer) == 0:
            return None
        retrieved = torch.stack(self.buffer)  # [N, H]
        retrieved = retrieved.unsqueeze(0).expand(query.shape[0], -1, -1)
        return retrieved.to(query.device)
    
    @property
    def size(self):
        return len(self.buffer)


class HierarchicalSummaryBuffer:
    """Strategy C: Maintain a running summary via exponential moving average.
    
    Instead of storing individual windows, maintain a single summary
    vector that gets updated with each new window. This is the most
    memory-efficient approach but loses the ability to retrieve
    specific past events.
    """
    
    def __init__(self, hidden_dim: int = 1152, decay: float = 0.9):
        self.hidden_dim = hidden_dim
        self.decay = decay
        self.reset()
    
    def reset(self):
        self.summary = None
        self.n_updates = 0
    
    def store(self, window_latents: torch.Tensor):
        current = window_latents.mean(dim=1).detach()[0]  # [H]
        if self.summary is None:
            self.summary = current
        else:
            self.summary = self.decay * self.summary + (1 - self.decay) * current
        self.n_updates += 1
    
    def retrieve(self, query: torch.Tensor) -> Optional[torch.Tensor]:
        if self.summary is None:
            return None
        # Return summary as a single "memory" [B, 1, H]
        retrieved = self.summary.unsqueeze(0).unsqueeze(0)
        retrieved = retrieved.expand(query.shape[0], -1, -1)
        return retrieved.to(query.device)
    
    @property
    def size(self):
        return self.n_updates


class RandomRetrievalBuffer(MemoryBuffer):
    """Control: Retrieves random past windows instead of most similar ones.
    
    Used as an ablation control to test whether similarity-based
    retrieval matters, or if any past context helps.
    """
    
    def retrieve(self, query: torch.Tensor) -> Optional[torch.Tensor]:
        if len(self.buffer) == 0:
            return None
        
        buffer_tensor = torch.stack(self.buffer).to(query.device)
        k = min(self.top_k, len(self.buffer))
        
        # Random indices instead of similarity-based
        indices = torch.randperm(len(self.buffer))[:k]
        retrieved = buffer_tensor[indices]
        retrieved = retrieved.unsqueeze(0).expand(query.shape[0], -1, -1)
        
        return retrieved
