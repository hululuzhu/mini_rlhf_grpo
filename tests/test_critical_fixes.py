"""Unit tests for critical fixes based on expert review."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import pytest
from models.policy_model import PolicyModel
from models.reward_model import RewardModel
from models.value_model import ValueModel
from models.transformer import TransformerLM
from utils.gae import compute_gae
from utils.rl_utils import (
    token_kl, shaped_rewards, compute_sequence_kl,
    get_action_log_probs, clip_ratio_loss, value_loss_with_clipping
)


class TestMaskingSemantics:
    """Test proper masking behavior."""
    
    def test_boolean_masks(self):
        """Test that masks are properly converted to boolean."""
        model = TransformerLM(vocab_size=100, d_model=64, n_heads=4, n_layers=2)
        
        input_ids = torch.randint(0, 100, (2, 10))
        
        # Test with int mask
        int_mask = torch.ones(2, 10, dtype=torch.int64)
        int_mask[:, 5:] = 0  # Mask out second half
        
        # Should not raise an error
        output, _ = model(input_ids, int_mask)
        assert output.shape == (2, 10, 64)
    
    def test_causal_plus_padding_mask(self):
        """Test that causal mask and padding mask work together."""
        model = PolicyModel(vocab_size=100, d_model=64, n_heads=4, n_layers=2)
        
        batch_size = 4
        seq_len = 10
        input_ids = torch.randint(1, 100, (batch_size, seq_len))
        
        # Create attention mask with varying lengths
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[0, 7:] = 0  # First sequence ends at position 7
        attention_mask[1, 5:] = 0  # Second sequence ends at position 5
        attention_mask[2, 9:] = 0  # Third sequence ends at position 9
        # Fourth sequence uses full length
        
        # Should work without error
        logits, _ = model(input_ids, attention_mask)
        assert logits.shape == (batch_size, seq_len, 100)
        
        # Check no NaN or Inf in output
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()
    
    def test_padded_positions_no_nan(self):
        """Test that padded positions produce valid outputs (not NaN).
        
        Note: We don't mask query positions (only keys), so padded positions
        will compute attention and produce logits. These are ignored in the loss.
        """
        model = PolicyModel(vocab_size=100, d_model=64, n_heads=4, n_layers=2)
        
        input_ids = torch.randint(1, 100, (2, 10))
        # Set some positions to padding
        input_ids[:, 8:] = 0  # Pad token
        
        attention_mask = torch.ones(2, 10)
        attention_mask[:, 8:] = 0
        
        logits, _ = model(input_ids, attention_mask)
        
        # Padded positions will produce valid logits (not NaN)
        # This is acceptable because we ignore them in the loss via padding mask
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()


class TestGenerationEOS:
    """Test per-sample EOS handling in generation."""
    
    def test_per_sample_eos_stopping(self):
        """Test that sequences stop independently when they hit EOS."""
        model = PolicyModel(vocab_size=100, d_model=64, n_heads=4, n_layers=2, pad_token_id=0)
        model.eval()
        
        # Start with simple prompts
        input_ids = torch.tensor([[1, 2], [3, 4]])  # Batch of 2
        
        # Generate with EOS
        eos_token_id = 5
        generated = model.generate(
            input_ids,
            max_length=20,
            temperature=1.0,
            do_sample=False,  # Greedy for determinism
            eos_token_id=eos_token_id
        )
        
        # Check that generation happened
        assert generated.size(1) >= input_ids.size(1)
        
        # If any sequence hit EOS, subsequent tokens should be EOS
        for i in range(generated.size(0)):
            seq = generated[i]
            eos_positions = (seq == eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                first_eos = eos_positions[0].item()
                # All positions after first EOS should be EOS
                if first_eos < seq.size(0) - 1:
                    assert (seq[first_eos:] == eos_token_id).all(), \
                        f"Sequence {i} continued after EOS at position {first_eos}"
    
    def test_temperature_validation(self):
        """Test that invalid temperature raises error."""
        model = PolicyModel(vocab_size=100, d_model=64, n_heads=4, n_layers=2)
        model.eval()
        
        input_ids = torch.tensor([[1, 2, 3]])
        
        # Zero temperature should fail
        with pytest.raises(ValueError, match="Temperature must be positive"):
            model.generate(input_ids, temperature=0.0)
        
        # Negative temperature should fail
        with pytest.raises(ValueError, match="Temperature must be positive"):
            model.generate(input_ids, temperature=-1.0)
        
        # Positive should work
        try:
            model.generate(input_ids, temperature=0.5, max_length=5)
        except ValueError as e:
            if "Temperature" in str(e):
                pytest.fail("Valid temperature raised error")


class TestRewardModel:
    """Test reward model behavior."""
    
    def test_preference_loss_invariant(self):
        """Test that loss(chosen, rejected) creates correct ordering."""
        model = RewardModel(vocab_size=100, d_model=64, n_heads=4, n_layers=2)
        
        # Create chosen and rejected sequences
        chosen = torch.randint(1, 100, (4, 10))
        rejected = torch.randint(1, 100, (4, 10))
        mask = torch.ones(4, 10)
        
        # Get rewards (without training, these are random, but we can test structure)
        with torch.no_grad():
            chosen_rewards = model.get_sequence_reward(chosen, attention_mask=mask)
            rejected_rewards = model.get_sequence_reward(rejected, attention_mask=mask)
        
        assert chosen_rewards.shape == (4,)
        assert rejected_rewards.shape == (4,)
        
        # Test loss computation
        loss = model.compute_preference_loss(chosen, rejected, mask, mask)
        assert loss.ndim == 0  # Scalar
        assert not torch.isnan(loss)
        assert loss >= 0  # Bradley-Terry loss is non-negative
    
    def test_get_sequence_reward_last_requires_mask(self):
        """Test that 'last' reduction with padding requires attention mask."""
        model = RewardModel(vocab_size=100, d_model=64, n_heads=4, n_layers=2, pad_token_id=0)
        
        # Create sequence with padding
        input_ids = torch.randint(1, 100, (2, 10))
        input_ids[:, 8:] = 0  # Add padding
        
        # Should raise error without mask
        with pytest.raises(ValueError, match="'last' reduction requires attention_mask"):
            model.get_sequence_reward(input_ids, attention_mask=None, reduction='last')
        
        # Should work with mask
        mask = torch.ones(2, 10)
        mask[:, 8:] = 0
        rewards = model.get_sequence_reward(input_ids, attention_mask=mask, reduction='last')
        assert rewards.shape == (2,)
    
    def test_reduction_validation(self):
        """Test that invalid reduction raises error."""
        model = RewardModel(vocab_size=100, d_model=64, n_heads=4, n_layers=2)
        input_ids = torch.randint(1, 100, (2, 10))
        
        with pytest.raises(ValueError, match="Unknown reduction"):
            model.get_sequence_reward(input_ids, reduction='invalid')


class TestValueModel:
    """Test value model behavior."""
    
    def test_value_head_shape(self):
        """Test that value head produces correct shapes."""
        model = ValueModel(vocab_size=100, d_model=64, n_heads=4, n_layers=2)
        
        batch_size = 4
        seq_len = 10
        input_ids = torch.randint(1, 100, (batch_size, seq_len))
        
        values = model(input_ids)
        
        assert values.shape == (batch_size, seq_len)
        assert not torch.isnan(values).any()
    
    def test_value_with_masking(self):
        """Test that value model respects masking."""
        model = ValueModel(vocab_size=100, d_model=64, n_heads=4, n_layers=2, pad_token_id=0)
        
        input_ids = torch.randint(1, 100, (2, 10))
        input_ids[:, 7:] = 0  # Padding
        
        mask = torch.ones(2, 10)
        mask[:, 7:] = 0
        
        values = model(input_ids, attention_mask=mask)
        
        assert values.shape == (2, 10)
        # Values for padded positions should still be computed (they'll be masked out in loss)
        assert not torch.isnan(values).any()


class TestGAE:
    """Test GAE computation."""
    
    def test_gae_with_mask(self):
        """Test GAE properly handles masking."""
        batch_size = 4
        seq_len = 10
        
        rewards = torch.randn(batch_size, seq_len)
        values = torch.randn(batch_size, seq_len)
        mask = torch.ones(batch_size, seq_len)
        
        # Mask out different amounts per sequence
        mask[0, 7:] = 0
        mask[1, 5:] = 0
        mask[2, 9:] = 0
        
        advantages, returns = compute_gae(rewards, values, mask)
        
        assert advantages.shape == (batch_size, seq_len)
        assert returns.shape == (batch_size, seq_len)
        
        # Advantages for masked positions should be zero
        assert (advantages[0, 7:] == 0).all()
        assert (advantages[1, 5:] == 0).all()
        assert (advantages[2, 9:] == 0).all()


class TestRLUtilities:
    """Test RL utility functions."""
    
    def test_token_kl(self):
        """Test per-token KL computation."""
        current = torch.randn(4, 10)
        ref = torch.randn(4, 10)
        
        kl = token_kl(current, ref)
        
        assert kl.shape == (4, 10)
        # KL should be just the difference in log-probs
        assert torch.allclose(kl, current - ref)
    
    def test_shaped_rewards(self):
        """Test reward shaping with KL penalty."""
        batch_size = 4
        seq_len = 10
        
        base_reward = torch.randn(batch_size)
        token_kl_terms = torch.randn(batch_size, seq_len).abs()  # Positive KL
        beta = 0.1
        mask = torch.ones(batch_size, seq_len)
        mask[:, 7:] = 0  # Mask last positions
        
        rewards = shaped_rewards(base_reward, token_kl_terms, beta, mask)
        
        assert rewards.shape == (batch_size, seq_len)
        
        # Masked positions should have zero reward
        assert (rewards[:, 7:] == 0).all()
    
    def test_clip_ratio_loss(self):
        """Test PPO clipped ratio loss."""
        ratio = torch.tensor([[1.5, 0.8, 1.2], [0.7, 1.1, 1.3]])
        advantages = torch.randn(2, 3)
        mask = torch.ones(2, 3)
        
        loss = clip_ratio_loss(ratio, advantages, clip_epsilon=0.2, mask=mask)
        
        assert loss.ndim == 0  # Scalar
        assert not torch.isnan(loss)


class TestPositionalEmbedding:
    """Test positional embedding bounds checking."""
    
    def test_exceeds_max_seq_len(self):
        """Test that exceeding max_seq_len raises error."""
        model = TransformerLM(
            vocab_size=100,
            d_model=64,
            n_heads=4,
            n_layers=2,
            max_seq_len=20
        )
        
        # This should work
        input_ids = torch.randint(0, 100, (2, 20))
        output, _ = model(input_ids)
        assert output.shape == (2, 20, 64)
        
        # This should fail
        input_ids_too_long = torch.randint(0, 100, (2, 25))
        with pytest.raises(ValueError, match="exceeds maximum"):
            model(input_ids_too_long)


class TestInputValidation:
    """Test input validation in forward passes."""
    
    def test_policy_input_validation(self):
        """Test that PolicyModel validates input shapes."""
        model = PolicyModel(vocab_size=100, d_model=64, n_heads=4, n_layers=2)
        
        # Wrong dimensions
        with pytest.raises(ValueError, match="must be"):
            model(torch.randint(0, 100, (2, 10, 3)))  # 3D instead of 2D
        
        # Mismatched labels shape
        input_ids = torch.randint(0, 100, (2, 10))
        labels = torch.randint(0, 100, (2, 8))  # Wrong length
        
        with pytest.raises(ValueError, match="must match"):
            model(input_ids, labels=labels)


class TestGradientFlow:
    """Test that gradients flow correctly."""
    
    def test_policy_backward(self):
        """Test basic gradient flow through policy model."""
        model = PolicyModel(vocab_size=100, d_model=64, n_heads=4, n_layers=2)
        
        input_ids = torch.randint(1, 100, (2, 10))
        labels = torch.randint(1, 100, (2, 10))
        
        logits, loss = model(input_ids, labels=labels)
        
        # Ensure loss is scalar and finite
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        
        # Backward should work
        loss.backward()
        
        # Check some gradients exist and are finite
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient in {name}"
    
    def test_reward_model_backward(self):
        """Test gradient flow through reward model."""
        model = RewardModel(vocab_size=100, d_model=64, n_heads=4, n_layers=2)
        
        chosen = torch.randint(1, 100, (4, 10))
        rejected = torch.randint(1, 100, (4, 10))
        mask = torch.ones(4, 10)
        
        loss = model.compute_preference_loss(chosen, rejected, mask, mask)
        
        assert torch.isfinite(loss)
        
        loss.backward()
        
        # Check gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient in {name}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

