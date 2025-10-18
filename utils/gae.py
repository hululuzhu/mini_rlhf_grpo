"""Generalized Advantage Estimation (GAE) utilities."""

import torch
from typing import Tuple


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
    normalize: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Generalized Advantage Estimation with proper masking.
    
    Args:
        rewards: Rewards per timestep [batch_size, seq_len].
        values: Value predictions per timestep [batch_size, seq_len].
        mask: Mask indicating valid positions (1=valid, 0=padding) [batch_size, seq_len].
        gamma: Discount factor.
        lam: GAE lambda parameter.
        normalize: Whether to normalize advantages.
        
    Returns:
        advantages: Computed advantages [batch_size, seq_len].
        returns: Computed returns (advantages + values) [batch_size, seq_len].
    """
    batch_size, seq_len = rewards.size()
    device = rewards.device
    
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(batch_size, device=device)
    
    # Compute GAE in reverse order
    for t in reversed(range(seq_len)):
        # Get next value (0 if at end or next position is masked)
        if t < seq_len - 1:
            next_value = values[:, t + 1]
            next_mask = mask[:, t + 1]
        else:
            next_value = torch.zeros(batch_size, device=device)
            next_mask = torch.zeros(batch_size, device=device)
        
        # TD error: delta = r + gamma * V(s') * mask - V(s)
        delta = rewards[:, t] + gamma * next_value * next_mask - values[:, t]
        
        # GAE: A = delta + gamma * lambda * A' * mask
        last_gae = delta + gamma * lam * next_mask * last_gae
        advantages[:, t] = last_gae
    
    # Apply mask to advantages
    advantages = advantages * mask
    
    # Returns are advantages + values
    returns = advantages + values
    
    # Optional: normalize advantages per batch (only over valid positions)
    if normalize:
        valid_mask = mask > 0
        if valid_mask.any():
            mean = advantages[valid_mask].mean()
            std = advantages[valid_mask].std(unbiased=False)
            advantages = (advantages - mean) / (std + 1e-8)
            # Re-apply mask after normalization
            advantages = advantages * mask
    
    return advantages, returns


def normalize_advantages(advantages: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize advantages to have zero mean and unit variance.
    
    Args:
        advantages: Advantage values [batch_size, seq_len].
        eps: Small epsilon for numerical stability.
        
    Returns:
        Normalized advantages [batch_size, seq_len].
    """
    mean = advantages.mean()
    std = advantages.std()
    return (advantages - mean) / (std + eps)


def compute_returns_to_go(
    rewards: torch.Tensor,
    gamma: float = 0.99
) -> torch.Tensor:
    """Compute discounted returns-to-go.
    
    Args:
        rewards: Rewards per timestep [batch_size, seq_len].
        gamma: Discount factor.
        
    Returns:
        Returns-to-go [batch_size, seq_len].
    """
    batch_size, seq_len = rewards.size()
    returns = torch.zeros_like(rewards)
    
    running_return = torch.zeros(batch_size, device=rewards.device)
    
    # Compute returns in reverse order
    for t in reversed(range(seq_len)):
        running_return = rewards[:, t] + gamma * running_return
        returns[:, t] = running_return
    
    return returns

