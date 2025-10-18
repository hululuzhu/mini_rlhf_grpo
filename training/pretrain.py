"""Pretrain policy model on Shakespeare data."""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse

from models import CharTokenizer, PolicyModel
from utils import MetricsLogger, print_training_stats, get_device


class TextDataset(Dataset):
    """Dataset for character-level language modeling."""
    
    def __init__(self, token_ids, seq_len=128):
        """Initialize dataset.
        
        Args:
            token_ids: List of token ids.
            seq_len: Sequence length.
        """
        self.token_ids = token_ids
        self.seq_len = seq_len
    
    def __len__(self):
        return len(self.token_ids) // self.seq_len
    
    def __getitem__(self, idx):
        start_idx = idx * self.seq_len
        end_idx = start_idx + self.seq_len + 1  # +1 for target
        
        sequence = self.token_ids[start_idx:end_idx]
        
        # Pad if necessary
        if len(sequence) < self.seq_len + 1:
            sequence = sequence + [0] * (self.seq_len + 1 - len(sequence))
        
        input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
        labels = torch.tensor(sequence[1:], dtype=torch.long)
        
        return input_ids, labels


def train_policy_model(
    data_path: str,
    output_dir: str,
    d_model: int = 256,
    n_heads: int = 8,
    n_layers: int = 6,
    d_ff: int = 1024,
    seq_len: int = 128,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    epochs: int = 10,
    device: str = 'auto',
    save_every: int = 1000
):
    """Train policy model.
    
    Args:
        data_path: Path to Shakespeare text file.
        output_dir: Directory to save outputs.
        d_model: Model dimension.
        n_heads: Number of attention heads.
        n_layers: Number of transformer layers.
        d_ff: Feed-forward dimension.
        seq_len: Sequence length.
        batch_size: Batch size.
        learning_rate: Learning rate.
        epochs: Number of epochs.
        device: Device to train on.
        save_every: Save checkpoint every N steps.
    """
    # Get device
    device = get_device(device)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and prepare data
    print("Loading data...")
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Data size: {len(text)} characters")
    
    # Build tokenizer
    print("Building tokenizer...")
    tokenizer = CharTokenizer()
    tokenizer.build_vocab(text)
    tokenizer.save(os.path.join(output_dir, 'tokenizer.pkl'))
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Tokenize text
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    
    # Split into train/val
    split_idx = int(0.9 * len(token_ids))
    train_ids = token_ids[:split_idx]
    val_ids = token_ids[split_idx:]
    
    # Create datasets
    train_dataset = TextDataset(train_ids, seq_len)
    val_dataset = TextDataset(val_ids, seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Initialize model
    print("Initializing model...")
    model = PolicyModel(
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
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs * len(train_loader)
    )
    
    # Logger
    logger = MetricsLogger(log_dir=output_dir, experiment_name='pretrain')
    
    # Training loop
    print("\nStarting training...")
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (input_ids, labels) in enumerate(pbar):
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits, loss = model(input_ids, labels=labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            # Logging
            train_loss += loss.item()
            global_step += 1
            
            pbar.set_postfix({'loss': loss.item(), 'lr': scheduler.get_last_lr()[0]})
            
            logger.log_scalar('train/loss', loss.item(), global_step)
            logger.log_scalar('train/lr', scheduler.get_last_lr()[0], global_step)
            
            # Save checkpoint
            if global_step % save_every == 0:
                checkpoint_path = os.path.join(output_dir, f'checkpoint_step_{global_step}.pt')
                torch.save({
                    'step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                }, checkpoint_path)
                print(f"\nSaved checkpoint to {checkpoint_path}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for input_ids, labels in tqdm(val_loader, desc="Validation"):
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                
                logits, loss = model(input_ids, labels=labels)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print_training_stats(epoch + 1, {
            'Train Loss': avg_train_loss,
            'Val Loss': avg_val_loss,
            'Learning Rate': scheduler.get_last_lr()[0],
        })
        
        logger.log_scalar('val/loss', avg_val_loss, global_step)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(output_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': best_val_loss,
            }, best_model_path)
            print(f"Saved best model with val loss: {best_val_loss:.4f}")
        
        # Generate sample
        model.eval()
        with torch.no_grad():
            prompt = "ROMEO:"
            prompt_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
            generated_ids = model.generate(
                prompt_ids,
                max_length=100,
                temperature=0.8,
                do_sample=True
            )
            generated_text = tokenizer.decode(generated_ids[0].cpu().tolist())
            print(f"\nGenerated sample:\n{generated_text}\n")
            logger.log_text('samples/generation', generated_text, global_step)
    
    # Save final model
    final_model_path = os.path.join(output_dir, 'final_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
    }, final_model_path)
    print(f"\nSaved final model to {final_model_path}")
    
    logger.close()
    print("Training complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pretrain policy model')
    parser.add_argument('--data_path', type=str, default='../data/shakespeare.txt')
    parser.add_argument('--output_dir', type=str, default='../checkpoints/pretrain')
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--d_ff', type=int, default=1024)
    parser.add_argument('--seq_len', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--device', type=str, default='auto')
    
    args = parser.parse_args()
    
    train_policy_model(
        data_path=args.data_path,
        output_dir=args.output_dir,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        device=args.device
    )

