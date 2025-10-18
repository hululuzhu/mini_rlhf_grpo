"""Reinforcement learning utilities for PPO/GRPO training."""

import torch
from typing import Tuple, Optional


def token_kl(
    current_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor
) -> torch.Tensor:
    """Compute per-token KL divergence between current and reference policies.
    
    Args:
        current_logprobs: Log probabilities from current policy [batch_size, seq_len].
        ref_logprobs: Log probabilities from reference policy [batch_size, seq_len].
        
    Returns:
        Per-token KL divergence [batch_size, seq_len].
    """
    # KL(current || ref) = E[log(current) - log(ref)]
    # For a single sample: log p_current - log p_ref
    return current_logprobs - ref_logprobs


def shaped_rewards(
    base_seq_reward: torch.Tensor,
    token_kl_terms: torch.Tensor,
    beta: float,
    attn_mask: torch.Tensor
) -> torch.Tensor:
    """Shape rewards with KL penalty for RL training.
    
    Distributes sequence reward to the last valid token and applies per-token KL penalty.
    
    Args:
        base_seq_reward: Scalar reward per sequence [batch_size].
        token_kl_terms: Per-token KL divergence [batch_size, seq_len].
        beta: KL penalty coefficient (higher = stronger penalty).
        attn_mask: Attention mask [batch_size, seq_len] (1 for valid, 0 for padding).
        
    Returns:
        Shaped rewards per token [batch_size, seq_len].
    """
    batch_size, seq_len = token_kl_terms.shape
    device = token_kl_terms.device
    
    # Apply KL penalty per token
    kl_penalty = -beta * token_kl_terms
    kl_penalty = kl_penalty * attn_mask
    
    # Put terminal reward at the last valid token
    last_idx = attn_mask.sum(dim=1) - 1  # Get last valid position
    last_idx = last_idx.clamp(min=0, max=seq_len - 1).long()  # Clamp and convert to long
    
    r = torch.zeros_like(token_kl_terms)
    r.scatter_(1, last_idx.unsqueeze(1), base_seq_reward.unsqueeze(1))
    
    return r + kl_penalty


def compute_sequence_kl(
    current_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    attn_mask: torch.Tensor
) -> torch.Tensor:
    """Compute total KL divergence per sequence.
    
    Args:
        current_logprobs: Log probabilities from current policy [batch_size, seq_len].
        ref_logprobs: Log probabilities from reference policy [batch_size, seq_len].
        attn_mask: Attention mask [batch_size, seq_len].
        
    Returns:
        Total KL divergence per sequence [batch_size].
    """
    token_kl_div = token_kl(current_logprobs, ref_logprobs)
    # Sum over valid tokens only
    return (token_kl_div * attn_mask).sum(dim=1)


def get_action_log_probs(
    logits: torch.Tensor,
    actions: torch.Tensor,
    mask: torch.Tensor = None
) -> torch.Tensor:
    """Extract log probabilities of taken actions.
    
    Args:
        logits: Model logits [batch_size, seq_len, vocab_size].
        actions: Taken actions (token ids) [batch_size, seq_len].
        mask: Optional mask for valid positions [batch_size, seq_len].
        
    Returns:
        Log probabilities of actions [batch_size, seq_len].
    """
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    # Gather log probs for taken actions
    action_log_probs = torch.gather(
        log_probs,
        dim=2,
        index=actions.unsqueeze(-1)
    ).squeeze(-1)
    
    if mask is not None:
        action_log_probs = action_log_probs * mask
    
    return action_log_probs


def clip_ratio_loss(
    ratio: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float = 0.2,
    mask: torch.Tensor = None
) -> torch.Tensor:
    """Compute PPO clipped ratio loss.
    
    Args:
        ratio: Importance sampling ratio (new_prob / old_prob) [batch_size, seq_len].
        advantages: Advantage estimates [batch_size, seq_len].
        clip_epsilon: Clipping parameter for PPO.
        mask: Optional mask for valid positions [batch_size, seq_len].
        
    Returns:
        Clipped policy loss scalar.
    """
    # Clipped surrogate objective
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2)
    
    if mask is not None:
        policy_loss = policy_loss * mask
        loss = policy_loss.sum() / mask.sum().clamp(min=1)
    else:
        loss = policy_loss.mean()
    
    return loss


def value_loss_with_clipping(
    values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    clip_epsilon: float = 0.2,
    mask: torch.Tensor = None
) -> torch.Tensor:
    """Compute PPO value loss with clipping.
    
    Args:
        values: Current value predictions [batch_size, seq_len].
        old_values: Old value predictions [batch_size, seq_len].
        returns: Target returns [batch_size, seq_len].
        clip_epsilon: Clipping parameter.
        mask: Optional mask for valid positions [batch_size, seq_len].
        
    Returns:
        Value loss scalar.
    """
    # Clipped value loss
    value_loss_unclipped = (values - returns) ** 2
    values_clipped = old_values + torch.clamp(
        values - old_values,
        -clip_epsilon,
        clip_epsilon
    )
    value_loss_clipped = (values_clipped - returns) ** 2
    value_loss = torch.max(value_loss_unclipped, value_loss_clipped)
    
    if mask is not None:
        value_loss = value_loss * mask
        loss = value_loss.sum() / mask.sum().clamp(min=1)
    else:
        loss = value_loss.mean()
    
    return loss


def scatter_terminal_reward(
    seq_reward: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    """Scatter sequence-level reward to the last valid token position.
    
    This is useful for distributing terminal rewards in episodic RL tasks
    where only the final outcome provides a reward signal.
    
    Args:
        seq_reward: Sequence-level reward [batch_size] or [batch_size, 1].
        mask: Attention mask indicating valid positions [batch_size, seq_len].
        
    Returns:
        Per-token rewards with terminal reward at last position [batch_size, seq_len].
    """
    batch_size, seq_len = mask.shape
    
    # Ensure seq_reward is [batch_size]
    if seq_reward.dim() > 1:
        seq_reward = seq_reward.squeeze(-1)
    
    # Find last valid token position
    idx = mask.long().sum(dim=1, keepdim=True) - 1  # [batch_size, 1]
    idx = idx.clamp(min=0)
    
    # Create zero tensor and scatter terminal reward to last position
    rewards = torch.zeros_like(mask, dtype=seq_reward.dtype)
    rewards.scatter_(1, idx, seq_reward.unsqueeze(1))
    
    return rewards


def gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor,
    gamma: float = 1.0,
    lam: float = 0.95
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Generalized Advantage Estimation (GAE).
    
    This is a helper that provides the exact implementation recommended
    in the expert review for stable PPO/GRPO training.
    
    Args:
        rewards: Per-timestep rewards [batch_size, seq_len].
        values: Value predictions [batch_size, seq_len].
        mask: Attention mask [batch_size, seq_len].
        gamma: Discount factor (typically 1.0 for language tasks).
        lam: GAE lambda parameter for bias-variance tradeoff.
        
    Returns:
        advantages: Normalized advantages [batch_size, seq_len].
        returns: Target returns (advantages + values) [batch_size, seq_len].
    """
    batch_size, seq_len = rewards.shape
    device = rewards.device
    
    advantages = torch.zeros_like(rewards)
    lastgaelam = torch.zeros(batch_size, device=device)
    
    # Compute GAE in reverse order
    for t in reversed(range(seq_len)):
        # Get next value and mask (0 if at end or next is masked)
        next_v = values[:, t + 1] if t < seq_len - 1 else torch.zeros_like(values[:, 0])
        next_m = mask[:, t + 1] if t < seq_len - 1 else torch.zeros_like(mask[:, 0])
        
        # TD error: delta = r_t + gamma * V(s_{t+1}) * mask - V(s_t)
        delta = rewards[:, t] + gamma * next_v * next_m - values[:, t]
        
        # GAE: A_t = delta_t + gamma * lambda * mask * A_{t+1}
        lastgaelam = delta + gamma * lam * next_m * lastgaelam
        advantages[:, t] = lastgaelam
    
    # Compute returns
    returns = advantages + values
    
    # Normalize advantages over valid positions only
    m = mask.bool()
    if m.any():
        adv_mean = advantages[m].mean()
        adv_std = advantages[m].std(unbiased=False)
        advantages = (advantages - adv_mean) / (adv_std + 1e-8)
    
    # Apply mask to both
    advantages = advantages * mask
    returns = returns * mask
    
    return advantages, returns


def clip_value(
    new_v: torch.Tensor,
    old_v: torch.Tensor,
    eps: float = 0.2
) -> torch.Tensor:
    """Clip value predictions to prevent large updates (PPO-style).
    
    This helps stabilize value learning by preventing the value function
    from changing too rapidly during training.
    
    Args:
        new_v: New value predictions [batch_size, seq_len].
        old_v: Old value predictions [batch_size, seq_len].
        eps: Clipping epsilon (typically 0.2).
        
    Returns:
        Clipped value predictions [batch_size, seq_len].
    """
    return old_v + (new_v - old_v).clamp(min=-eps, max=eps)

