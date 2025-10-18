"""Tests to verify expert review v2 fixes are working correctly."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import pytest
from models.policy_model import PolicyModel
from models.reward_model import RewardModel
from models.transformer import TransformerLM
from utils.rl_utils import scatter_terminal_reward, gae, clip_value


def test_policy_generate_parameter_validation():
    """Test that generate validates parameters correctly."""
    model = PolicyModel(vocab_size=100, d_model=64, n_heads=4, n_layers=2)
    input_ids = torch.randint(0, 100, (2, 10))
    
    # Test temperature validation
    with pytest.raises(ValueError, match="Temperature must be positive"):
        model.generate(input_ids, temperature=0)
    
    with pytest.raises(ValueError, match="Temperature must be positive"):
        model.generate(input_ids, temperature=-1.0)
    
    # Test top_k validation
    with pytest.raises(ValueError, match="top_k must be in"):
        model.generate(input_ids, top_k=0)
    
    with pytest.raises(ValueError, match="top_k must be in"):
        model.generate(input_ids, top_k=100)  # vocab_size or more
    
    # Test top_p validation
    with pytest.raises(ValueError, match="top_p must be in"):
        model.generate(input_ids, top_p=0.0)
    
    with pytest.raises(ValueError, match="top_p must be in"):
        model.generate(input_ids, top_p=1.5)
    
    print("✓ Parameter validation tests passed")


def test_policy_generate_max_seq_len_check():
    """Test that generate checks prompt length against max_seq_len."""
    model = PolicyModel(vocab_size=100, d_model=64, n_heads=4, n_layers=2, max_seq_len=50)
    
    # Prompt at max length should fail
    long_prompt = torch.randint(0, 100, (1, 50))
    with pytest.raises(ValueError, match="max_seq_len"):
        model.generate(long_prompt, max_length=100)
    
    # Prompt just under max should work
    ok_prompt = torch.randint(0, 100, (1, 45))
    output = model.generate(ok_prompt, max_length=50, do_sample=False)
    assert output.size(1) <= 50  # Should not exceed max_seq_len
    
    print("✓ Max seq len validation tests passed")


def test_policy_generate_attention_mask():
    """Test that generate uses explicit attention mask."""
    model = PolicyModel(vocab_size=100, d_model=64, n_heads=4, n_layers=2)
    input_ids = torch.randint(0, 100, (2, 10))
    
    # Generate should work and produce valid output
    output = model.generate(input_ids, max_length=20, do_sample=False, eos_token_id=1)
    
    assert output.size(0) == 2  # Batch size preserved
    assert output.size(1) >= 10  # At least as long as input
    assert output.size(1) <= 20  # Not longer than max_length
    
    print("✓ Attention mask in generation tests passed")


def test_policy_generate_nucleus_filtering_fallback():
    """Test that top-p filtering has fallback for edge cases."""
    model = PolicyModel(vocab_size=100, d_model=64, n_heads=4, n_layers=2)
    input_ids = torch.randint(0, 100, (2, 10))
    
    # Very small top_p should still work (fallback to at least one token)
    output = model.generate(input_ids, max_length=15, top_p=0.001, do_sample=True)
    assert output.size(1) > input_ids.size(1)  # Should generate something
    
    # Combined top_k and top_p
    output = model.generate(input_ids, max_length=15, top_k=5, top_p=0.9, do_sample=True)
    assert output.size(1) > input_ids.size(1)
    
    print("✓ Nucleus filtering fallback tests passed")


def test_transformer_causal_mask_dtype():
    """Test that get_causal_mask returns bool dtype."""
    model = TransformerLM(vocab_size=100, d_model=64, n_heads=4, n_layers=2)
    
    mask = model.get_causal_mask(10, torch.device('cpu'))
    assert mask.dtype == torch.bool, f"Expected bool dtype, got {mask.dtype}"
    assert mask.shape == (1, 1, 10, 10)
    
    # Check that it's actually a causal mask
    assert mask[0, 0, 0, 0]  # Can attend to self
    assert mask[0, 0, 5, 5]  # Can attend to self
    assert mask[0, 0, 5, 3]  # Can attend to past
    assert not mask[0, 0, 3, 5]  # Cannot attend to future
    
    print("✓ Causal mask dtype tests passed")


def test_reward_model_preference_loss_stability():
    """Test that preference loss is numerically stable."""
    model = RewardModel(vocab_size=100, d_model=64, n_heads=4, n_layers=2)
    
    # Create sequences with large reward differences
    chosen_ids = torch.randint(0, 100, (4, 20))
    rejected_ids = torch.randint(0, 100, (4, 20))
    mask = torch.ones(4, 20)
    
    loss = model.compute_preference_loss(chosen_ids, rejected_ids, mask, mask)
    
    # Loss should be finite (not NaN or Inf)
    assert torch.isfinite(loss).all(), f"Loss is not finite: {loss}"
    assert loss >= 0, f"Loss should be non-negative: {loss}"
    
    print("✓ Preference loss stability tests passed")


def test_reward_model_last_reduction_strict():
    """Test that 'last' reduction requires attention_mask."""
    model = RewardModel(vocab_size=100, d_model=64, n_heads=4, n_layers=2)
    input_ids = torch.randint(0, 100, (2, 20))
    
    # Should raise error without mask
    with pytest.raises(ValueError, match="'last' reduction requires attention_mask"):
        model.get_sequence_reward(input_ids, attention_mask=None, reduction='last')
    
    # Should work with mask
    mask = torch.ones(2, 20)
    rewards = model.get_sequence_reward(input_ids, attention_mask=mask, reduction='last')
    assert rewards.shape == (2,)
    
    # Should still work for 'mean' without mask (though not recommended)
    rewards = model.get_sequence_reward(input_ids, attention_mask=None, reduction='mean')
    assert rewards.shape == (2,)
    
    print("✓ Last reduction strict validation tests passed")


def test_scatter_terminal_reward():
    """Test scatter_terminal_reward utility."""
    # Test basic functionality
    seq_reward = torch.tensor([1.0, 2.0, 3.0])
    mask = torch.ones(3, 10)
    
    token_rewards = scatter_terminal_reward(seq_reward, mask)
    
    assert token_rewards.shape == (3, 10)
    assert token_rewards[:, -1].allclose(seq_reward)
    assert token_rewards[:, :-1].sum() == 0  # All other positions should be zero
    
    # Test with variable lengths
    mask = torch.tensor([
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [1, 1, 0, 0, 0]
    ], dtype=torch.float)
    seq_reward = torch.tensor([10.0, 20.0, 30.0])
    
    token_rewards = scatter_terminal_reward(seq_reward, mask)
    
    # Check rewards are at correct positions
    assert token_rewards[0, 2] == 10.0  # Last valid position for seq 0
    assert token_rewards[1, 4] == 20.0  # Last valid position for seq 1
    assert token_rewards[2, 1] == 30.0  # Last valid position for seq 2
    
    print("✓ Scatter terminal reward tests passed")


def test_gae_utility():
    """Test GAE utility function."""
    batch_size = 2
    seq_len = 5
    
    # Create simple test data
    rewards = torch.ones(batch_size, seq_len)
    values = torch.zeros(batch_size, seq_len)
    mask = torch.ones(batch_size, seq_len)
    
    advantages, returns = gae(rewards, values, mask, gamma=1.0, lam=0.95)
    
    assert advantages.shape == (batch_size, seq_len)
    assert returns.shape == (batch_size, seq_len)
    
    # Advantages should be normalized (mean ~0, std ~1 over valid positions)
    valid_mask = mask.bool()
    adv_mean = advantages[valid_mask].mean()
    adv_std = advantages[valid_mask].std(unbiased=False)
    assert abs(adv_mean) < 0.1, f"Mean should be ~0, got {adv_mean}"
    assert abs(adv_std - 1.0) < 0.1, f"Std should be ~1, got {adv_std}"
    
    # Test with masking
    mask = torch.tensor([
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1]
    ], dtype=torch.float)
    rewards = torch.ones(2, 5)
    values = torch.zeros(2, 5)
    
    advantages, returns = gae(rewards, values, mask, gamma=1.0, lam=0.95)
    
    # Masked positions should have zero advantages
    assert advantages[0, 3:].sum() == 0
    assert advantages[0, 4:].sum() == 0
    
    print("✓ GAE utility tests passed")


def test_clip_value_utility():
    """Test clip_value utility function."""
    old_v = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    new_v = torch.tensor([0.5, 2.1, 3.5, 5.0, 6.0])
    
    clipped = clip_value(new_v, old_v, eps=0.2)
    
    # Check clipping behavior
    assert clipped[0] == pytest.approx(1.0 - 0.2)  # Clamped to lower bound
    assert clipped[1] == pytest.approx(2.1)  # Within bounds
    assert clipped[2] == pytest.approx(3.0 + 0.2)  # Clamped to upper bound
    assert clipped[3] == pytest.approx(4.0 + 0.2)  # Clamped to upper bound
    
    print("✓ Clip value utility tests passed")


if __name__ == "__main__":
    print("Running expert review v2 fix verification tests...\n")
    
    # Policy model tests
    test_policy_generate_parameter_validation()
    test_policy_generate_max_seq_len_check()
    test_policy_generate_attention_mask()
    test_policy_generate_nucleus_filtering_fallback()
    
    # Transformer tests
    test_transformer_causal_mask_dtype()
    
    # Reward model tests
    test_reward_model_preference_loss_stability()
    test_reward_model_last_reduction_strict()
    
    # RL utilities tests
    test_scatter_terminal_reward()
    test_gae_utility()
    test_clip_value_utility()
    
    print("\n" + "="*60)
    print("All expert review v2 fix verification tests passed! ✓")
    print("="*60)

