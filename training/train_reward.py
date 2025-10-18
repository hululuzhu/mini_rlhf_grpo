"""Train reward model on preference data."""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse

from models import CharTokenizer, RewardModel
from utils import MetricsLogger, print_training_stats, get_device


class SyntheticPreferenceDataset(Dataset):
    """Synthetic preference dataset for character-level LM.
    
    Since HuggingFace doesn't have many char-level preference datasets,
    we create synthetic preferences based on simple heuristics:
    - Prefer sequences with proper capitalization
    - Prefer sequences with balanced punctuation
    - Prefer sequences that look more Shakespeare-like
    """
    
    def __init__(self, tokenizer, text, num_samples=1000, seq_len=64):
        """Initialize synthetic preference dataset.
        
        Args:
            tokenizer: Character tokenizer.
            text: Source text (Shakespeare).
            num_samples: Number of preference pairs to generate.
            seq_len: Sequence length.
        """
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.pairs = []
        
        # Split text into sentences/lines
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Generate preference pairs
        import random
        random.seed(42)
        
        for _ in range(num_samples):
            # Sample a good line (from actual text)
            good_line = random.choice(lines)[:seq_len]
            
            # Create a degraded version
            bad_line = self._degrade_text(good_line)
            
            # Tokenize
            chosen_ids = tokenizer.encode(good_line, add_special_tokens=True)[:seq_len]
            rejected_ids = tokenizer.encode(bad_line, add_special_tokens=True)[:seq_len]
            
            # Pad to seq_len
            chosen_ids += [tokenizer.pad_token_id] * (seq_len - len(chosen_ids))
            rejected_ids += [tokenizer.pad_token_id] * (seq_len - len(rejected_ids))
            
            self.pairs.append((chosen_ids, rejected_ids))
    
    def _degrade_text(self, text):
        """Create a degraded version of text."""
        import random
        
        # Random degradation strategies
        strategy = random.choice(['lowercase', 'uppercase', 'remove_punct', 'scramble'])
        
        if strategy == 'lowercase':
            return text.lower()
        elif strategy == 'uppercase':
            return text.upper()
        elif strategy == 'remove_punct':
            return ''.join(c if c.isalnum() or c.isspace() else '' for c in text)
        elif strategy == 'scramble':
            chars = list(text)
            random.shuffle(chars)
            return ''.join(chars)
        
        return text
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        chosen_ids, rejected_ids = self.pairs[idx]
        
        # Create attention masks (1 for real tokens, 0 for padding)
        chosen_mask = [1 if token_id != self.tokenizer.pad_token_id else 0 for token_id in chosen_ids]
        rejected_mask = [1 if token_id != self.tokenizer.pad_token_id else 0 for token_id in rejected_ids]
        
        return {
            'chosen_ids': torch.tensor(chosen_ids, dtype=torch.long),
            'rejected_ids': torch.tensor(rejected_ids, dtype=torch.long),
            'chosen_mask': torch.tensor(chosen_mask, dtype=torch.long),
            'rejected_mask': torch.tensor(rejected_mask, dtype=torch.long),
        }


def train_reward_model(
    data_path: str,
    tokenizer_path: str,
    output_dir: str,
    d_model: int = 256,
    n_heads: int = 8,
    n_layers: int = 4,
    d_ff: int = 1024,
    seq_len: int = 64,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    epochs: int = 5,
    num_samples: int = 5000,
    device: str = 'auto'
):
    """Train reward model.
    
    Args:
        data_path: Path to Shakespeare text file.
        tokenizer_path: Path to tokenizer.
        output_dir: Directory to save outputs.
        d_model: Model dimension.
        n_heads: Number of attention heads.
        n_layers: Number of transformer layers.
        d_ff: Feed-forward dimension.
        seq_len: Sequence length.
        batch_size: Batch size.
        learning_rate: Learning rate.
        epochs: Number of epochs.
        num_samples: Number of preference pairs.
        device: Device to train on.
    """
    # Get device
    device = get_device(device)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = CharTokenizer.load(tokenizer_path)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Load data
    print("Loading data...")
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Create synthetic preference dataset
    print("Creating synthetic preference dataset...")
    dataset = SyntheticPreferenceDataset(tokenizer, text, num_samples, seq_len)
    
    # Split into train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Initialize model
    print("Initializing reward model...")
    model = RewardModel(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_seq_len=seq_len,
        dropout=0.1,
        pad_token_id=tokenizer.pad_token_id
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Logger
    logger = MetricsLogger(log_dir=output_dir, experiment_name='reward_training')
    
    # Training loop
    print("\nStarting training...")
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, batch in enumerate(pbar):
            chosen_ids = batch['chosen_ids'].to(device)
            rejected_ids = batch['rejected_ids'].to(device)
            chosen_mask = batch['chosen_mask'].to(device)
            rejected_mask = batch['rejected_mask'].to(device)
            
            # Compute preference loss
            loss = model.compute_preference_loss(chosen_ids, rejected_ids, chosen_mask, rejected_mask)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Compute accuracy
            with torch.no_grad():
                chosen_rewards = model.get_sequence_reward(chosen_ids, chosen_mask)
                rejected_rewards = model.get_sequence_reward(rejected_ids, rejected_mask)
                acc = (chosen_rewards > rejected_rewards).float().mean()
            
            # Logging
            train_loss += loss.item()
            train_acc += acc.item()
            global_step += 1
            
            pbar.set_postfix({'loss': loss.item(), 'acc': acc.item()})
            
            logger.log_scalar('train/loss', loss.item(), global_step)
            logger.log_scalar('train/accuracy', acc.item(), global_step)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                chosen_ids = batch['chosen_ids'].to(device)
                rejected_ids = batch['rejected_ids'].to(device)
                chosen_mask = batch['chosen_mask'].to(device)
                rejected_mask = batch['rejected_mask'].to(device)
                
                loss = model.compute_preference_loss(chosen_ids, rejected_ids, chosen_mask, rejected_mask)
                val_loss += loss.item()
                
                chosen_rewards = model.get_sequence_reward(chosen_ids, chosen_mask)
                rejected_rewards = model.get_sequence_reward(rejected_ids, rejected_mask)
                acc = (chosen_rewards > rejected_rewards).float().mean()
                val_acc += acc.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_acc / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc / len(val_loader)
        
        print_training_stats(epoch + 1, {
            'Train Loss': avg_train_loss,
            'Train Accuracy': avg_train_acc,
            'Val Loss': avg_val_loss,
            'Val Accuracy': avg_val_acc,
        })
        
        logger.log_scalar('val/loss', avg_val_loss, global_step)
        logger.log_scalar('val/accuracy', avg_val_acc, global_step)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(output_dir, 'best_reward_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': best_val_loss,
            }, best_model_path)
            print(f"Saved best model with val loss: {best_val_loss:.4f}")
    
    # Save final model
    final_model_path = os.path.join(output_dir, 'final_reward_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
    }, final_model_path)
    print(f"\nSaved final model to {final_model_path}")
    
    logger.close()
    print("Training complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train reward model')
    parser.add_argument('--data_path', type=str, default='../data/shakespeare.txt')
    parser.add_argument('--tokenizer_path', type=str, default='../checkpoints/pretrain/tokenizer.pkl')
    parser.add_argument('--output_dir', type=str, default='../checkpoints/reward')
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--d_ff', type=int, default=1024)
    parser.add_argument('--seq_len', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--num_samples', type=int, default=5000)
    parser.add_argument('--device', type=str, default='auto')
    
    args = parser.parse_args()
    
    train_reward_model(
        data_path=args.data_path,
        tokenizer_path=args.tokenizer_path,
        output_dir=args.output_dir,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        num_samples=args.num_samples,
        device=args.device
    )

