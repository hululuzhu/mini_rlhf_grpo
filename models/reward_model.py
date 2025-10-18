"""Reward model for preference-based scoring."""

import torch
import torch.nn as nn
from typing import Optional
from .transformer import TransformerLM


class RewardModel(nn.Module):
    """Reward model that outputs scalar rewards for sequences."""
    
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
        """Initialize reward model.
        
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
        
        # Reward head - outputs scalar reward
        self.reward_head = nn.Linear(d_model, 1)
        
        self.pad_token_id = pad_token_id
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass to compute rewards.
        
        Args:
            input_ids: Input token ids [batch_size, seq_len].
            attention_mask: Attention mask [batch_size, seq_len].
            
        Returns:
            Rewards per token [batch_size, seq_len].
        """
        # Get hidden states from transformer (don't use cache for reward model)
        hidden_states, _ = self.transformer(input_ids, attention_mask, past_kvs=None, use_cache=False)
        
        # Compute rewards for each token
        rewards = self.reward_head(hidden_states).squeeze(-1)  # [batch_size, seq_len]
        
        return rewards
    
    def get_sequence_reward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        reduction: str = 'last'
    ) -> torch.Tensor:
        """Get reward for entire sequence.
        
        Args:
            input_ids: Input token ids [batch_size, seq_len].
            attention_mask: Attention mask [batch_size, seq_len].
            reduction: How to aggregate token rewards ('last' is standard for preference learning).
                      'mean' and 'sum' are length-biased; use with caution.
            
        Returns:
            Sequence rewards [batch_size].
        """
        # Validate reduction type
        if reduction not in ('last', 'mean', 'sum'):
            raise ValueError(f"Unknown reduction: {reduction}. Must be 'last', 'mean', or 'sum'.")
        
        # For 'last' reduction, strictly require attention_mask to avoid silent errors
        if reduction == 'last' and attention_mask is None:
            raise ValueError(
                "'last' reduction requires attention_mask for variable-length batches. "
                "This ensures correct identification of the last non-padding position and "
                "prevents silent label leakage from padded positions."
            )
        
        rewards = self.forward(input_ids, attention_mask)
        
        if reduction == 'last':
            # Use reward of last non-padding token (standard for preference learning)
            # Find last non-padding position
            seq_lengths = attention_mask.long().sum(dim=1) - 1  # -1 for 0-indexing
            batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
            return rewards[batch_indices, seq_lengths.clamp(min=0)]
        elif reduction == 'mean':
            # Mean aggregation (length-biased, not standard for preference learning)
            if attention_mask is not None:
                mask_sum = attention_mask.sum(dim=1).clamp(min=1)  # Avoid division by zero
                return (rewards * attention_mask).sum(dim=1) / mask_sum
            else:
                return rewards.mean(dim=1)
        elif reduction == 'sum':
            # Sum aggregation (highly length-biased, use with explicit length penalty)
            if attention_mask is not None:
                return (rewards * attention_mask).sum(dim=1)
            else:
                return rewards.sum(dim=1)
    
    def compute_preference_loss(
        self,
        chosen_ids: torch.Tensor,
        rejected_ids: torch.Tensor,
        chosen_mask: Optional[torch.Tensor] = None,
        rejected_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute preference loss for training.
        
        Args:
            chosen_ids: Chosen sequence token ids [batch_size, seq_len].
            rejected_ids: Rejected sequence token ids [batch_size, seq_len].
            chosen_mask: Attention mask for chosen [batch_size, seq_len].
            rejected_mask: Attention mask for rejected [batch_size, seq_len].
            
        Returns:
            Preference loss scalar.
        """
        # Get rewards for chosen and rejected sequences
        chosen_rewards = self.get_sequence_reward(chosen_ids, chosen_mask)
        rejected_rewards = self.get_sequence_reward(rejected_ids, rejected_mask)
        
        # Bradley-Terry model loss (numerically stable version)
        # We want chosen to have higher reward than rejected
        # Using softplus(-x) which is equivalent to -log(sigmoid(x)) but numerically stable
        diff = chosen_rewards - rejected_rewards
        loss = torch.nn.functional.softplus(-diff).mean()
        
        return loss

