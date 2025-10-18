"""Unit tests for model components."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import pytest
from models import CharTokenizer, PolicyModel, RewardModel, ValueModel


def test_tokenizer():
    """Test character tokenizer."""
    text = "Hello World! This is a test."
    
    tokenizer = CharTokenizer()
    tokenizer.build_vocab(text)
    
    # Test encoding
    encoded = tokenizer.encode(text, add_special_tokens=True)
    assert len(encoded) > 0
    assert encoded[0] == tokenizer.bos_token_id
    assert encoded[-1] == tokenizer.eos_token_id
    
    # Test decoding
    decoded = tokenizer.decode(encoded, skip_special_tokens=True)
    assert decoded == text
    
    # Test vocab size
    assert tokenizer.vocab_size > 0
    
    print("✓ Tokenizer tests passed")


def test_policy_model():
    """Test policy model."""
    vocab_size = 100
    batch_size = 4
    seq_len = 32
    
    model = PolicyModel(
        vocab_size=vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=512,
        max_seq_len=seq_len
    )
    
    # Test forward pass
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits, loss = model(input_ids)
    
    assert logits.shape == (batch_size, seq_len, vocab_size)
    assert loss is None  # No labels provided
    
    # Test with labels
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits, loss = model(input_ids, labels=labels)
    
    assert loss is not None
    assert loss.item() > 0
    
    # Test generation
    model.eval()
    with torch.no_grad():
        prompt = torch.randint(0, vocab_size, (1, 10))
        generated = model.generate(prompt, max_length=20, do_sample=False)
        assert generated.shape[1] == 20
    
    print("✓ Policy model tests passed")


def test_reward_model():
    """Test reward model."""
    vocab_size = 100
    batch_size = 4
    seq_len = 32
    
    model = RewardModel(
        vocab_size=vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=512,
        max_seq_len=seq_len
    )
    
    # Test forward pass (avoid pad token 0)
    input_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
    mask = torch.ones(batch_size, seq_len)
    rewards = model(input_ids)
    
    assert rewards.shape == (batch_size, seq_len)
    
    # Test sequence reward
    seq_reward = model.get_sequence_reward(input_ids, attention_mask=mask, reduction='last')
    assert seq_reward.shape == (batch_size,)
    
    # Test preference loss (avoid pad token 0)
    chosen_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
    rejected_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
    
    loss = model.compute_preference_loss(chosen_ids, rejected_ids, mask, mask)
    assert loss.item() > 0
    
    print("✓ Reward model tests passed")


def test_value_model():
    """Test value model."""
    vocab_size = 100
    batch_size = 4
    seq_len = 32
    
    model = ValueModel(
        vocab_size=vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=512,
        max_seq_len=seq_len
    )
    
    # Test forward pass
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    values = model(input_ids)
    
    assert values.shape == (batch_size, seq_len)
    
    print("✓ Value model tests passed")


def test_gae():
    """Test GAE computation."""
    from utils import compute_gae, normalize_advantages
    
    batch_size = 4
    seq_len = 10
    
    rewards = torch.randn(batch_size, seq_len)
    values = torch.randn(batch_size, seq_len)
    mask = torch.ones(batch_size, seq_len)  # All positions valid
    
    advantages, returns = compute_gae(rewards, values, mask, gamma=0.99, lam=0.95)
    
    assert advantages.shape == (batch_size, seq_len)
    assert returns.shape == (batch_size, seq_len)
    
    # Test normalization
    normalized = normalize_advantages(advantages)
    assert torch.abs(normalized.mean()) < 1e-5
    assert torch.abs(normalized.std() - 1.0) < 1e-5
    
    print("✓ GAE tests passed")


if __name__ == '__main__':
    print("Running model unit tests...\n")
    
    test_tokenizer()
    test_policy_model()
    test_reward_model()
    test_value_model()
    test_gae()
    
    print("\n" + "="*60)
    print("All unit tests passed!")
    print("="*60)

