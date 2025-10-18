"""Base transformer architecture using PyTorch."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention layer."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """Initialize multi-head attention.
        
        Args:
            d_model: Model dimension.
            n_heads: Number of attention heads.
            dropout: Dropout probability.
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass with optional KV caching.
        
        Args:
            query: Query tensor [batch_size, seq_len, d_model].
            key: Key tensor [batch_size, seq_len, d_model].
            value: Value tensor [batch_size, seq_len, d_model].
            mask: Attention mask [batch_size, 1, seq_len, seq_len] (boolean).
            past_kv: Cached (key, value) from previous forward passes.
            use_cache: Whether to return updated cache for next iteration.
            
        Returns:
            Tuple of (output tensor [batch_size, seq_len, d_model], 
                     optional cached (K, V) tensors).
        """
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Concatenate with past if provided
        if past_kv is not None:
            past_K, past_V = past_kv
            K = torch.cat([past_K, K], dim=2)  # Concat along sequence dimension
            V = torch.cat([past_V, V], dim=2)
        
        # Store K, V for next iteration if caching
        present_kv = (K, V) if use_cache else None
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            # Mask should be boolean; if True, attend; if False, mask out
            scores = scores.masked_fill(~mask, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        output = torch.matmul(attn, V)
        
        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final linear projection
        output = self.w_o(output)
        
        return output, present_kv


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """Initialize feed-forward network.
        
        Args:
            d_model: Model dimension.
            d_ff: Feed-forward dimension.
            dropout: Dropout probability.
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model].
            
        Returns:
            Output tensor [batch_size, seq_len, d_model].
        """
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Transformer decoder block."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        """Initialize transformer block.
        
        Args:
            d_model: Model dimension.
            n_heads: Number of attention heads.
            d_ff: Feed-forward dimension.
            dropout: Dropout probability.
        """
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass with optional KV caching.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model].
            mask: Attention mask [batch_size, 1, seq_len, seq_len].
            past_kv: Cached (key, value) from previous forward pass.
            use_cache: Whether to return updated cache.
            
        Returns:
            Tuple of (output tensor [batch_size, seq_len, d_model],
                     optional cached (K, V) tensors).
        """
        # Self-attention with residual connection
        attn_output, present_kv = self.attn(x, x, x, mask, past_kv, use_cache)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # Feed-forward with residual connection
        ff_output = self.ff(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)
        
        return x, present_kv


class TransformerLM(nn.Module):
    """Transformer language model."""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        pad_token_id: int = 0
    ):
        """Initialize transformer language model.
        
        Args:
            vocab_size: Vocabulary size.
            d_model: Model dimension.
            n_heads: Number of attention heads.
            n_layers: Number of transformer layers.
            d_ff: Feed-forward dimension.
            max_seq_len: Maximum sequence length.
            dropout: Dropout probability.
            pad_token_id: Padding token id.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_kvs: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]]:
        """Forward pass with optional KV caching.
        
        Args:
            input_ids: Input token ids [batch_size, seq_len].
            attention_mask: Attention mask [batch_size, total_seq_len] where total_seq_len
                           includes past cached tokens.
            past_kvs: Tuple of cached (key, value) pairs for each layer.
            use_cache: Whether to return updated cache.
            
        Returns:
            Tuple of (hidden states [batch_size, seq_len, d_model],
                     optional tuple of cached (K, V) pairs for each layer).
        """
        batch_size, seq_len = input_ids.size()
        device = input_ids.device
        
        # Determine position offset from past_kvs
        past_length = 0
        if past_kvs is not None and len(past_kvs) > 0:
            # All layers should have same past length
            past_length = past_kvs[0][0].size(2)  # Key tensor shape: [batch, heads, past_len, d_k]
        
        total_seq_len = past_length + seq_len
        
        # Validate sequence length
        if total_seq_len > self.max_seq_len:
            raise ValueError(
                f"Total sequence length {total_seq_len} exceeds maximum {self.max_seq_len}. "
                f"Please truncate input or increase max_seq_len."
            )
        
        # Convert and validate attention mask dtype
        if attention_mask is not None:
            if attention_mask.dtype not in (torch.bool, torch.uint8, torch.int32, torch.int64, torch.float32, torch.float16):
                raise ValueError(f"attention_mask must be bool, int, or float type, got {attention_mask.dtype}")
            if attention_mask.dtype != torch.bool:
                attention_mask = attention_mask.bool()
        
        # Create position ids with offset for cached positions
        position_ids = torch.arange(
            past_length, past_length + seq_len, device=device
        ).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        x = self.dropout(token_embeds + position_embeds)
        
        # Create causal mask for current tokens attending to all tokens (past + current)
        # Optimized: when using KV cache with single token, avoid recomputing full mask
        if past_length > 0 and seq_len == 1:
            # Fast path: single new token can attend to all previous tokens
            # Shape: [1, total_seq_len] - all True
            causal_mask = torch.ones(1, total_seq_len, device=device, dtype=torch.bool)
            attn_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, 1, total_seq_len]
        else:
            # Standard path: full causal mask computation
            # Shape: [seq_len, total_seq_len]
            causal_mask = torch.zeros(seq_len, total_seq_len, device=device, dtype=torch.bool)
            # Current tokens can attend to all past tokens
            if past_length > 0:
                causal_mask[:, :past_length] = True
            # Current tokens use causal masking among themselves
            causal_mask[:, past_length:] = torch.tril(
                torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)
            )
            attn_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, total_seq_len]
        
        # Apply padding mask if provided
        if attention_mask is not None:
            # attention_mask should have shape [batch_size, total_seq_len]
            if attention_mask.size(1) != total_seq_len:
                raise ValueError(
                    f"attention_mask length {attention_mask.size(1)} doesn't match "
                    f"total sequence length {total_seq_len}"
                )
            # Mask keys: don't attend TO padded positions
            key_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, total_seq_len]
            attn_mask = attn_mask & key_mask
        
        # Transformer blocks with KV caching
        present_kvs = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            past_kv = past_kvs[i] if past_kvs is not None else None
            x, present_kv = block(x, attn_mask, past_kv, use_cache)
            if use_cache:
                present_kvs.append(present_kv)
        
        x = self.norm(x)
        
        return x, tuple(present_kvs) if use_cache else None
    
    def get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal attention mask.
        
        Args:
            seq_len: Sequence length.
            device: Device to create mask on.
            
        Returns:
            Causal mask [1, 1, seq_len, seq_len] (boolean, True = attend, False = mask).
        """
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        return mask.unsqueeze(0).unsqueeze(0)

