"""DeepSeek GRPO (Group Relative Policy Optimization) training.

GRPO is a variant that eliminates the value network by using
group-relative advantages computed from sampled trajectories.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
import copy

from models import CharTokenizer, PolicyModel, RewardModel
from utils import MetricsLogger, print_training_stats, get_device


class PromptDataset(Dataset):
    """Dataset of prompts for RL generation."""
    
    def __init__(self, tokenizer, text, num_prompts=1000, prompt_len=20):
        """Initialize prompt dataset."""
        self.prompts = []
        
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        import random
        random.seed(42)
        
        for _ in range(num_prompts):
            line = random.choice(lines)
            if len(line) > prompt_len:
                start = random.randint(0, len(line) - prompt_len)
                prompt = line[start:start + prompt_len]
            else:
                prompt = line[:prompt_len]
            
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)[:prompt_len]
            self.prompts.append(prompt_ids)
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return torch.tensor(self.prompts[idx], dtype=torch.long)


def compute_group_advantages(rewards, baseline='mean'):
    """Compute group-relative advantages.
    
    In GRPO, we sample multiple completions for each prompt and compute
    advantages relative to the group statistics (e.g., mean reward).
    
    Args:
        rewards: Rewards for each completion [batch_size].
        baseline: Baseline for advantage computation ('mean', 'median').
        
    Returns:
        Advantages [batch_size].
    """
    if baseline == 'mean':
        baseline_value = rewards.mean()
    elif baseline == 'median':
        baseline_value = rewards.median()
    else:
        raise ValueError(f"Unknown baseline: {baseline}")
    
    advantages = rewards - baseline_value
    
    # Normalize
    std = advantages.std()
    if std > 1e-8:
        advantages = advantages / (std + 1e-8)
    
    return advantages


def grpo_train(
    policy_checkpoint: str,
    reward_checkpoint: str,
    tokenizer_path: str,
    data_path: str,
    output_dir: str,
    d_model: int = 256,
    n_heads: int = 8,
    n_layers: int = 6,
    d_ff: int = 1024,
    prompt_len: int = 20,
    gen_len: int = 40,
    max_seq_len: int = 128,
    batch_size: int = 16,
    num_samples_per_prompt: int = 4,
    learning_rate: float = 1e-5,
    num_steps: int = 100,
    ppo_epochs: int = 4,
    clip_epsilon: float = 0.2,
    kl_coef: float = 0.1,
    temperature: float = 1.0,
    device: str = 'auto'
):
    """GRPO training without value network.
    
    Args:
        policy_checkpoint: Path to pretrained policy checkpoint.
        reward_checkpoint: Path to trained reward model checkpoint.
        tokenizer_path: Path to tokenizer.
        data_path: Path to data for prompts.
        output_dir: Directory to save outputs.
        d_model: Model dimension.
        n_heads: Number of attention heads.
        n_layers: Number of policy layers.
        d_ff: Feed-forward dimension.
        prompt_len: Prompt length.
        gen_len: Generation length.
        max_seq_len: Maximum sequence length.
        batch_size: Batch size (number of prompts).
        num_samples_per_prompt: Number of completions per prompt.
        learning_rate: Policy learning rate.
        num_steps: Number of training steps.
        ppo_epochs: Number of optimization epochs per step.
        clip_epsilon: PPO clip epsilon.
        kl_coef: KL divergence coefficient.
        temperature: Sampling temperature.
        device: Device to train on.
    """
    # Get device
    device = get_device(device)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = CharTokenizer.load(tokenizer_path)
    vocab_size = tokenizer.vocab_size
    
    # Load policy model
    print("Loading policy model...")
    policy = PolicyModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        dropout=0.0,
        pad_token_id=tokenizer.pad_token_id
    ).to(device)
    
    checkpoint = torch.load(policy_checkpoint, map_location=device)
    policy.load_state_dict(checkpoint['model_state_dict'])
    
    # Create reference model (frozen copy)
    print("Creating reference model...")
    ref_policy = copy.deepcopy(policy)
    ref_policy.eval()
    for param in ref_policy.parameters():
        param.requires_grad = False
    
    # Load reward model
    print("Loading reward model...")
    reward_model = RewardModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=4,  # Reward model was trained with 4 layers
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        dropout=0.0,
        pad_token_id=tokenizer.pad_token_id
    ).to(device)
    
    reward_checkpoint_data = torch.load(reward_checkpoint, map_location=device)
    reward_model.load_state_dict(reward_checkpoint_data['model_state_dict'])
    reward_model.eval()
    for param in reward_model.parameters():
        param.requires_grad = False
    
    print(f"Policy parameters: {sum(p.numel() for p in policy.parameters()):,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(policy.parameters(), lr=learning_rate)
    
    # Load prompts
    print("Loading prompts...")
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    prompt_dataset = PromptDataset(tokenizer, text, num_prompts=500, prompt_len=prompt_len)
    
    # Custom collate function to pad prompts to same length
    def collate_fn(batch):
        max_len = max(len(x) for x in batch)
        padded = []
        for x in batch:
            if len(x) < max_len:
                padding = torch.full((max_len - len(x),), tokenizer.pad_token_id, dtype=torch.long)
                x = torch.cat([x, padding])
            padded.append(x)
        return torch.stack(padded)
    
    prompt_loader = DataLoader(prompt_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    # Logger
    logger = MetricsLogger(log_dir=output_dir, experiment_name='grpo')
    
    # Training loop
    print(f"\nStarting GRPO training for {num_steps} steps...")
    print(f"Sampling {num_samples_per_prompt} completions per prompt")
    
    for step in range(num_steps):
        policy.eval()
        
        # Sample prompts
        prompts = next(iter(prompt_loader)).to(device)  # [batch_size, prompt_len]
        
        # Generate multiple completions per prompt
        all_completions = []
        all_log_probs = []
        all_ref_log_probs = []
        all_rewards = []
        
        for _ in range(num_samples_per_prompt):
            with torch.no_grad():
                # Generate completions
                generated = policy.generate(
                    prompts,
                    max_length=prompt_len + gen_len,
                    temperature=temperature,
                    do_sample=True
                )
                
                # Get log probs
                policy_log_probs = policy.get_log_probs(generated)
                ref_log_probs = ref_policy.get_log_probs(generated)
                
                # Gather log probs for generated tokens
                generated_log_probs = torch.gather(
                    policy_log_probs,
                    2,
                    generated.unsqueeze(-1)
                ).squeeze(-1)
                
                ref_generated_log_probs = torch.gather(
                    ref_log_probs,
                    2,
                    generated.unsqueeze(-1)
                ).squeeze(-1)
                
                # Create attention mask for generated sequences
                attention_mask = (generated != tokenizer.pad_token_id).long()
                
                # Compute rewards (sequence-level)
                sequence_rewards = reward_model.get_sequence_reward(generated, attention_mask=attention_mask, reduction='last')
                
                # Compute KL penalty
                kl_penalty = kl_coef * (generated_log_probs - ref_generated_log_probs).sum(dim=1)
                
                # Final rewards with KL penalty
                final_rewards = sequence_rewards - kl_penalty
                
                all_completions.append(generated)
                all_log_probs.append(generated_log_probs)
                all_ref_log_probs.append(ref_generated_log_probs)
                all_rewards.append(final_rewards)
        
        # Stack all samples
        all_completions = torch.cat(all_completions, dim=0)  # [batch_size * num_samples, seq_len]
        all_log_probs = torch.cat(all_log_probs, dim=0)
        all_ref_log_probs = torch.cat(all_ref_log_probs, dim=0)
        all_rewards = torch.cat(all_rewards, dim=0)  # [batch_size * num_samples]
        
        # Compute group-relative advantages
        # Group by prompt (each group has num_samples_per_prompt completions)
        advantages_list = []
        for i in range(batch_size):
            start_idx = i * num_samples_per_prompt
            end_idx = start_idx + num_samples_per_prompt
            group_rewards = all_rewards[start_idx:end_idx]
            group_advantages = compute_group_advantages(group_rewards, baseline='mean')
            advantages_list.append(group_advantages)
        
        advantages = torch.cat(advantages_list, dim=0)  # [batch_size * num_samples]
        
        # Expand advantages to token level (same advantage for all tokens in a sequence)
        advantages_expanded = advantages.unsqueeze(1).expand_as(all_log_probs)
        
        # GRPO updates (similar to PPO but without value function)
        policy.train()
        
        for epoch in range(ppo_epochs):
            # Get new log probs
            policy_log_probs_new = policy.get_log_probs(all_completions)
            generated_log_probs_new = torch.gather(
                policy_log_probs_new,
                2,
                all_completions.unsqueeze(-1)
            ).squeeze(-1)
            
            # Compute ratio
            ratio = torch.exp(generated_log_probs_new - all_log_probs.detach())
            
            # Clipped surrogate objective
            surr1 = ratio * advantages_expanded
            surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages_expanded
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Backward pass
            optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
        
        # Logging
        with torch.no_grad():
            mean_reward = all_rewards.mean().item()
            std_reward = all_rewards.std().item()
            mean_advantage = advantages.mean().item()
            kl_div = (all_log_probs - all_ref_log_probs).sum(dim=1).mean().item()
        
        metrics = {
            'reward_mean': mean_reward,
            'reward_std': std_reward,
            'advantage': mean_advantage,
            'kl_divergence': kl_div,
            'policy_loss': policy_loss.item(),
        }
        
        logger.log_scalars(metrics, step)
        
        if step % 10 == 0:
            print_training_stats(step, metrics)
            
            # Generate sample
            with torch.no_grad():
                sample_prompt = prompts[0:1]
                sample_gen = policy.generate(
                    sample_prompt,
                    max_length=prompt_len + gen_len,
                    temperature=0.8,
                    do_sample=True
                )
                sample_text = tokenizer.decode(sample_gen[0].cpu().tolist())
                print(f"Sample generation:\n{sample_text}\n")
                logger.log_text('samples/generation', sample_text, step)
        
        # Save checkpoint
        if step % 50 == 0 and step > 0:
            checkpoint_path = os.path.join(output_dir, f'grpo_checkpoint_step_{step}.pt')
            torch.save({
                'step': step,
                'policy_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    final_policy_path = os.path.join(output_dir, 'final_grpo_policy.pt')
    torch.save({'model_state_dict': policy.state_dict()}, final_policy_path)
    
    print(f"\nSaved final model to {final_policy_path}")
    
    logger.close()
    print("GRPO training complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GRPO training')
    parser.add_argument('--policy_checkpoint', type=str, default='../checkpoints/pretrain/best_model.pt')
    parser.add_argument('--reward_checkpoint', type=str, default='../checkpoints/reward/best_reward_model.pt')
    parser.add_argument('--tokenizer_path', type=str, default='../checkpoints/pretrain/tokenizer.pkl')
    parser.add_argument('--data_path', type=str, default='../data/shakespeare.txt')
    parser.add_argument('--output_dir', type=str, default='../checkpoints/grpo')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_samples_per_prompt', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--num_steps', type=int, default=100)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--d_ff', type=int, default=1024)
    parser.add_argument('--max_seq_len', type=int, default=128)
    parser.add_argument('--device', type=str, default='auto')
    
    args = parser.parse_args()
    
    grpo_train(
        policy_checkpoint=args.policy_checkpoint,
        reward_checkpoint=args.reward_checkpoint,
        tokenizer_path=args.tokenizer_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_samples_per_prompt=args.num_samples_per_prompt,
        learning_rate=args.learning_rate,
        num_steps=args.num_steps,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        device=args.device
    )

