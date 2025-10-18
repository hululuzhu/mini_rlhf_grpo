"""PPO-based RLHF training loop."""

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

from models import CharTokenizer, PolicyModel, RewardModel, ValueModel
from utils import compute_gae, normalize_advantages, MetricsLogger, print_training_stats, get_device


class PromptDataset(Dataset):
    """Dataset of prompts for RL generation."""
    
    def __init__(self, tokenizer, text, num_prompts=1000, prompt_len=20):
        """Initialize prompt dataset.
        
        Args:
            tokenizer: Character tokenizer.
            text: Source text.
            num_prompts: Number of prompts to generate.
            prompt_len: Prompt length.
        """
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


def compute_kl_divergence(log_probs, ref_log_probs):
    """Compute KL divergence between policy and reference.
    
    Args:
        log_probs: Log probs from policy [batch_size, seq_len, vocab_size].
        ref_log_probs: Log probs from reference [batch_size, seq_len, vocab_size].
        
    Returns:
        KL divergence [batch_size, seq_len].
    """
    # KL(P || Q) = sum(P * (log P - log Q))
    probs = torch.exp(log_probs)
    kl = (probs * (log_probs - ref_log_probs)).sum(dim=-1)
    return kl


def ppo_rlhf_train(
    policy_checkpoint: str,
    reward_checkpoint: str,
    tokenizer_path: str,
    data_path: str,
    output_dir: str,
    d_model: int = 256,
    n_heads: int = 8,
    n_layers: int = 6,
    value_n_layers: int = 4,
    d_ff: int = 1024,
    prompt_len: int = 20,
    gen_len: int = 40,
    max_seq_len: int = 128,
    batch_size: int = 16,
    learning_rate: float = 1e-5,
    value_lr: float = 3e-5,
    num_steps: int = 100,
    ppo_epochs: int = 4,
    clip_epsilon: float = 0.2,
    gamma: float = 0.99,
    lam: float = 0.95,
    kl_coef: float = 0.1,
    value_coef: float = 0.5,
    device: str = 'auto'
):
    """PPO RLHF training.
    
    Args:
        policy_checkpoint: Path to pretrained policy checkpoint.
        reward_checkpoint: Path to trained reward model checkpoint.
        tokenizer_path: Path to tokenizer.
        data_path: Path to data for prompts.
        output_dir: Directory to save outputs.
        d_model: Model dimension.
        n_heads: Number of attention heads.
        n_layers: Number of policy layers.
        value_n_layers: Number of value layers.
        d_ff: Feed-forward dimension.
        prompt_len: Prompt length.
        gen_len: Generation length.
        max_seq_len: Maximum sequence length.
        batch_size: Batch size.
        learning_rate: Policy learning rate.
        value_lr: Value learning rate.
        num_steps: Number of training steps.
        ppo_epochs: Number of PPO epochs per step.
        clip_epsilon: PPO clip epsilon.
        gamma: Discount factor.
        lam: GAE lambda.
        kl_coef: KL divergence coefficient.
        value_coef: Value loss coefficient.
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
    
    # Load models
    print("Loading policy model...")
    policy = PolicyModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        dropout=0.0,  # No dropout during RL
        pad_token_id=tokenizer.pad_token_id
    ).to(device)
    
    checkpoint = torch.load(policy_checkpoint, map_location=device)
    policy.load_state_dict(checkpoint['model_state_dict'])
    
    # Create reference model (frozen copy of policy)
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
    
    # Initialize value model
    print("Initializing value model...")
    value_model = ValueModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=value_n_layers,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        dropout=0.0,
        pad_token_id=tokenizer.pad_token_id
    ).to(device)
    
    print(f"Policy parameters: {sum(p.numel() for p in policy.parameters()):,}")
    print(f"Value parameters: {sum(p.numel() for p in value_model.parameters()):,}")
    
    # Optimizers
    policy_optimizer = torch.optim.AdamW(policy.parameters(), lr=learning_rate)
    value_optimizer = torch.optim.AdamW(value_model.parameters(), lr=value_lr)
    
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
    logger = MetricsLogger(log_dir=output_dir, experiment_name='ppo_rlhf')
    
    # Training loop
    print(f"\nStarting PPO RLHF training for {num_steps} steps...")
    
    for step in range(num_steps):
        policy.eval()
        
        # Sample prompts
        prompts = next(iter(prompt_loader)).to(device)
        
        # Generate completions
        with torch.no_grad():
            generated = policy.generate(
                prompts,
                max_length=prompt_len + gen_len,
                temperature=1.0,
                do_sample=True
            )
        
        # Get log probs from policy and reference
        with torch.no_grad():
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
        
        # Compute rewards
        with torch.no_grad():
            rewards_per_token = reward_model(generated)
            
            # Compute KL penalty
            kl_penalty = kl_coef * (generated_log_probs - ref_generated_log_probs)
            
            # Final rewards with KL penalty
            rewards = rewards_per_token - kl_penalty
            
            # Get values
            values = value_model(generated)
            
            # Compute advantages using GAE
            dones = torch.zeros_like(rewards)
            dones[:, -1] = 1.0  # Mark last token as done
            
            advantages, returns = compute_gae(
                rewards, values, dones, gamma=gamma, lam=lam
            )
            
            # Normalize advantages
            advantages = normalize_advantages(advantages)
        
        # PPO updates
        policy.train()
        value_model.train()
        
        for ppo_epoch in range(ppo_epochs):
            # Policy update
            policy_log_probs_new = policy.get_log_probs(generated)
            generated_log_probs_new = torch.gather(
                policy_log_probs_new,
                2,
                generated.unsqueeze(-1)
            ).squeeze(-1)
            
            # Compute ratio for PPO
            ratio = torch.exp(generated_log_probs_new - generated_log_probs.detach())
            
            # Clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value update
            values_new = value_model(generated)
            value_loss = F.mse_loss(values_new, returns.detach())
            
            # Combined loss
            loss = policy_loss + value_coef * value_loss
            
            # Backward pass
            policy_optimizer.zero_grad()
            value_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(value_model.parameters(), 1.0)
            policy_optimizer.step()
            value_optimizer.step()
        
        # Logging
        with torch.no_grad():
            mean_reward = rewards.mean().item()
            mean_kl = kl_penalty.mean().item()
            mean_value = values.mean().item()
            mean_advantage = advantages.mean().item()
        
        metrics = {
            'reward': mean_reward,
            'kl_divergence': mean_kl,
            'value': mean_value,
            'advantage': mean_advantage,
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
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
            checkpoint_path = os.path.join(output_dir, f'ppo_checkpoint_step_{step}.pt')
            torch.save({
                'step': step,
                'policy_state_dict': policy.state_dict(),
                'value_state_dict': value_model.state_dict(),
                'policy_optimizer_state_dict': policy_optimizer.state_dict(),
                'value_optimizer_state_dict': value_optimizer.state_dict(),
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final models
    final_policy_path = os.path.join(output_dir, 'final_ppo_policy.pt')
    final_value_path = os.path.join(output_dir, 'final_ppo_value.pt')
    
    torch.save({'model_state_dict': policy.state_dict()}, final_policy_path)
    torch.save({'model_state_dict': value_model.state_dict()}, final_value_path)
    
    print(f"\nSaved final models to {output_dir}")
    
    logger.close()
    print("PPO RLHF training complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PPO RLHF training')
    parser.add_argument('--policy_checkpoint', type=str, default='../checkpoints/pretrain/best_model.pt')
    parser.add_argument('--reward_checkpoint', type=str, default='../checkpoints/reward/best_reward_model.pt')
    parser.add_argument('--tokenizer_path', type=str, default='../checkpoints/pretrain/tokenizer.pkl')
    parser.add_argument('--data_path', type=str, default='../data/shakespeare.txt')
    parser.add_argument('--output_dir', type=str, default='../checkpoints/ppo_rlhf')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--num_steps', type=int, default=100)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--value_n_layers', type=int, default=4)
    parser.add_argument('--d_ff', type=int, default=1024)
    parser.add_argument('--max_seq_len', type=int, default=128)
    parser.add_argument('--device', type=str, default='auto')
    
    args = parser.parse_args()
    
    ppo_rlhf_train(
        policy_checkpoint=args.policy_checkpoint,
        reward_checkpoint=args.reward_checkpoint,
        tokenizer_path=args.tokenizer_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_steps=args.num_steps,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        value_n_layers=args.value_n_layers,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        device=args.device
    )

