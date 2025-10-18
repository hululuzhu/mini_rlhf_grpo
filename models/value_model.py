"""Value network for PPO."""

import torch
import torch.nn as nn
from typing import Optional
from .transformer import TransformerLM


class ValueModel(nn.Module):
    """Value network for predicting token-level values."""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,  # Typically smaller than policy
        d_ff: int = 1024,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        pad_token_id: int = 0
    ):
        """Initialize value model.
        
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
        self.transformer = TransformerLM(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout,
            pad_token_id=pad_token_id
        )
        
        # Value head - outputs scalar value per token
        self.value_head = nn.Linear(d_model, 1)
        
        self.pad_token_id = pad_token_id
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass to compute values.
        
        Args:
            input_ids: Input token ids [batch_size, seq_len].
            attention_mask: Attention mask [batch_size, seq_len].
            
        Returns:
            Values per token [batch_size, seq_len].
        """
        # Get hidden states from transformer (don't use cache for value model)
        hidden_states, _ = self.transformer(input_ids, attention_mask, past_kvs=None, use_cache=False)
        
        # Compute values for each token
        values = self.value_head(hidden_states).squeeze(-1)  # [batch_size, seq_len]
        
        return values

