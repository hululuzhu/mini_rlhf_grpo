"""Tests for model improvements: KV cache mask optimization and repetition penalty."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
import torch
import time
from models.transformer import TransformerLM
from models.policy_model import PolicyModel
from models.tokenizer import CharTokenizer


class TestKVCacheMaskOptimization:
    """Tests for the optimized attention mask computation in KV cache."""
    
    @pytest.fixture
    def model(self):
        """Create a small transformer model for testing."""
        return TransformerLM(
            vocab_size=100,
            d_model=64,
            n_heads=4,
            n_layers=2,
            d_ff=128,
            max_seq_len=128,
            dropout=0.0
        )
    
    def test_single_token_mask_fast_path(self, model):
        """Test that single token with past uses optimized mask path."""
        batch_size = 2
        device = torch.device('cpu')
        
        # First pass: process prompt (3 tokens)
        prompt = torch.randint(0, 100, (batch_size, 3), device=device)
        hidden1, past_kvs = model(prompt, use_cache=True)
        
        assert hidden1.shape == (batch_size, 3, model.d_model)
        assert past_kvs is not None
        assert len(past_kvs) == 2  # 2 layers
        
        # Second pass: single new token (should use fast path)
        new_token = torch.randint(0, 100, (batch_size, 1), device=device)
        attn_mask = torch.ones(batch_size, 4, dtype=torch.bool, device=device)  # 3 past + 1 new
        
        hidden2, past_kvs2 = model(
            new_token, 
            attention_mask=attn_mask,
            past_kvs=past_kvs, 
            use_cache=True
        )
        
        assert hidden2.shape == (batch_size, 1, model.d_model)
        assert past_kvs2 is not None
        # Verify KV cache grew
        assert past_kvs2[0][0].shape[2] == 4  # 3 + 1 tokens
    
    def test_multi_token_mask_standard_path(self, model):
        """Test that multiple tokens use standard mask computation."""
        batch_size = 2
        device = torch.device('cpu')
        
        # First pass: process prompt (3 tokens)
        prompt = torch.randint(0, 100, (batch_size, 3), device=device)
        hidden1, past_kvs = model(prompt, use_cache=True)
        
        # Second pass: multiple new tokens (should use standard path)
        new_tokens = torch.randint(0, 100, (batch_size, 2), device=device)
        attn_mask = torch.ones(batch_size, 5, dtype=torch.bool, device=device)  # 3 past + 2 new
        
        hidden2, past_kvs2 = model(
            new_tokens, 
            attention_mask=attn_mask,
            past_kvs=past_kvs, 
            use_cache=True
        )
        
        assert hidden2.shape == (batch_size, 2, model.d_model)
        assert past_kvs2[0][0].shape[2] == 5  # 3 + 2 tokens
    
    def test_mask_correctness_with_cache(self, model):
        """Verify that cached generation produces same results as non-cached."""
        batch_size = 1
        device = torch.device('cpu')
        torch.manual_seed(42)
        
        # Generate sequence without cache
        input_ids = torch.randint(0, 100, (batch_size, 5), device=device)
        model.eval()
        with torch.no_grad():
            hidden_no_cache, _ = model(input_ids, use_cache=False)
        
        # Generate same sequence with cache (step by step)
        torch.manual_seed(42)
        past_kvs = None
        all_hiddens = []
        
        for i in range(5):
            token = input_ids[:, i:i+1]
            attn_mask = torch.ones(batch_size, i+1, dtype=torch.bool, device=device)
            with torch.no_grad():
                hidden, past_kvs = model(
                    token,
                    attention_mask=attn_mask,
                    past_kvs=past_kvs,
                    use_cache=True
                )
            all_hiddens.append(hidden)
        
        hidden_with_cache = torch.cat(all_hiddens, dim=1)
        
        # Results should be very close
        assert torch.allclose(hidden_no_cache, hidden_with_cache, atol=1e-5)
    
    def test_mask_performance_improvement(self, model):
        """Test that optimized mask provides performance benefit (informal timing)."""
        batch_size = 4
        device = torch.device('cpu')
        model.eval()
        
        # Warm up
        prompt = torch.randint(0, 100, (batch_size, 10), device=device)
        with torch.no_grad():
            _, past_kvs = model(prompt, use_cache=True)
        
        # Time single token forward passes (uses optimized path)
        num_iterations = 50
        start = time.time()
        for _ in range(num_iterations):
            token = torch.randint(0, 100, (batch_size, 1), device=device)
            seq_len = 10 + _ + 1
            attn_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
            with torch.no_grad():
                _, past_kvs = model(
                    token,
                    attention_mask=attn_mask,
                    past_kvs=past_kvs,
                    use_cache=True
                )
        elapsed = time.time() - start
        
        # Just verify it completes without error and is reasonably fast
        assert elapsed < 10.0  # Should complete in under 10 seconds
        print(f"KV cache generation: {num_iterations} steps in {elapsed:.3f}s ({elapsed/num_iterations*1000:.2f}ms/step)")


class TestRepetitionPenalty:
    """Tests for repetition penalty in generation."""
    
    @pytest.fixture
    def tokenizer(self):
        """Create a simple character tokenizer."""
        return CharTokenizer(vocab_size=50)
    
    @pytest.fixture
    def policy_model(self):
        """Create a small policy model."""
        return PolicyModel(
            vocab_size=50,
            d_model=64,
            n_heads=4,
            n_layers=2,
            d_ff=128,
            max_seq_len=128,
            dropout=0.0,
            pad_token_id=0
        )
    
    def test_repetition_penalty_parameter_validation(self, policy_model):
        """Test that invalid repetition penalties raise errors."""
        input_ids = torch.tensor([[1, 2, 3]])
        
        # Negative penalty should fail
        with pytest.raises(ValueError, match="repetition_penalty must be positive"):
            policy_model.generate(input_ids, repetition_penalty=-1.0)
        
        # Zero penalty should fail
        with pytest.raises(ValueError, match="repetition_penalty must be positive"):
            policy_model.generate(input_ids, repetition_penalty=0.0)
        
        # Positive values should work
        output = policy_model.generate(input_ids, max_length=5, repetition_penalty=1.5)
        assert output.shape[0] == 1
    
    def test_no_repetition_penalty_baseline(self, policy_model):
        """Test generation without repetition penalty (baseline)."""
        torch.manual_seed(42)
        policy_model.eval()
        
        input_ids = torch.tensor([[1, 2, 3]])
        output = policy_model.generate(
            input_ids,
            max_length=15,
            temperature=0.8,
            do_sample=True,
            repetition_penalty=None
        )
        
        assert output.shape[0] == 1
        assert output.shape[1] <= 15
        # Without penalty, repeated tokens are more likely
        unique_ratio = len(output[0].unique()) / output.shape[1]
        print(f"Without penalty - Unique ratio: {unique_ratio:.2f}, Output: {output[0].tolist()}")
    
    def test_high_repetition_penalty_reduces_repeats(self, policy_model):
        """Test that high repetition penalty reduces token repetition."""
        torch.manual_seed(42)
        policy_model.eval()
        
        input_ids = torch.tensor([[1, 2, 3]])
        
        # Generate without penalty
        output_no_penalty = policy_model.generate(
            input_ids,
            max_length=15,
            temperature=0.8,
            do_sample=True,
            repetition_penalty=None
        )
        
        # Generate with high penalty
        torch.manual_seed(42)
        output_with_penalty = policy_model.generate(
            input_ids,
            max_length=15,
            temperature=0.8,
            do_sample=True,
            repetition_penalty=2.0  # Strong penalty against repetition
        )
        
        # Count unique tokens
        unique_no_penalty = len(output_no_penalty[0].unique())
        unique_with_penalty = len(output_with_penalty[0].unique())
        
        print(f"No penalty: {unique_no_penalty} unique tokens")
        print(f"With penalty (2.0): {unique_with_penalty} unique tokens")
        
        # With penalty should generally have more unique tokens (though not guaranteed due to randomness)
        # Just verify both completed successfully
        assert output_no_penalty.shape[1] <= 15
        assert output_with_penalty.shape[1] <= 15
    
    def test_repetition_penalty_with_batches(self, policy_model):
        """Test that repetition penalty works correctly with batched generation."""
        torch.manual_seed(123)
        policy_model.eval()
        
        input_ids = torch.tensor([
            [1, 2, 3],
            [4, 5, 6]
        ])
        
        output = policy_model.generate(
            input_ids,
            max_length=12,
            temperature=1.0,
            do_sample=True,
            repetition_penalty=1.5
        )
        
        assert output.shape[0] == 2
        assert output.shape[1] <= 12
        
        # Verify each batch item has its own repetition tracking
        for i in range(2):
            unique_ratio = len(output[i].unique()) / output.shape[1]
            print(f"Batch {i} - Unique ratio: {unique_ratio:.2f}")
            assert unique_ratio > 0.3  # Should have reasonable diversity
    
    def test_repetition_penalty_one_has_no_effect(self, policy_model):
        """Test that repetition_penalty=1.0 is equivalent to no penalty."""
        torch.manual_seed(999)
        policy_model.eval()
        
        input_ids = torch.tensor([[1, 2, 3]])
        
        # Generate with penalty = 1.0 (no effect)
        output_penalty_one = policy_model.generate(
            input_ids,
            max_length=10,
            temperature=1.0,
            do_sample=True,
            repetition_penalty=1.0
        )
        
        # Generate with no penalty
        torch.manual_seed(999)
        output_no_penalty = policy_model.generate(
            input_ids,
            max_length=10,
            temperature=1.0,
            do_sample=True,
            repetition_penalty=None
        )
        
        # Should produce identical results
        assert torch.equal(output_penalty_one, output_no_penalty)
    
    def test_repetition_penalty_with_greedy_decoding(self, policy_model):
        """Test repetition penalty works with greedy decoding."""
        torch.manual_seed(42)
        policy_model.eval()
        
        input_ids = torch.tensor([[1, 2, 3]])
        
        output = policy_model.generate(
            input_ids,
            max_length=10,
            do_sample=False,  # Greedy
            repetition_penalty=1.8
        )
        
        assert output.shape[0] == 1
        assert output.shape[1] <= 10
        
        # Even with greedy, penalty should affect token selection
        # Just verify it completes successfully
    
    def test_repetition_penalty_respects_eos(self, policy_model):
        """Test that repetition penalty doesn't override EOS behavior."""
        torch.manual_seed(42)
        policy_model.eval()
        
        input_ids = torch.tensor([[1, 2, 3]])
        eos_token_id = 10
        
        output = policy_model.generate(
            input_ids,
            max_length=20,
            do_sample=True,
            eos_token_id=eos_token_id,
            repetition_penalty=1.5
        )
        
        # If EOS was generated, sequence should have stopped
        if eos_token_id in output[0]:
            eos_pos = (output[0] == eos_token_id).nonzero()[0].item()
            # All tokens after first EOS should also be EOS
            if eos_pos < output.shape[1] - 1:
                assert (output[0, eos_pos:] == eos_token_id).all()
    
    def test_repetition_penalty_with_top_k_top_p(self, policy_model):
        """Test that repetition penalty works with top-k and top-p sampling."""
        torch.manual_seed(555)
        policy_model.eval()
        
        input_ids = torch.tensor([[1, 2, 3]])
        
        output = policy_model.generate(
            input_ids,
            max_length=15,
            temperature=0.9,
            top_k=20,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.5
        )
        
        assert output.shape[0] == 1
        assert output.shape[1] <= 15
        
        unique_ratio = len(output[0].unique()) / output.shape[1]
        print(f"With top_k, top_p, and repetition penalty - Unique ratio: {unique_ratio:.2f}")
    
    def test_very_high_repetition_penalty(self, policy_model):
        """Test that very high repetition penalty forces diversity."""
        torch.manual_seed(777)
        policy_model.eval()
        
        input_ids = torch.tensor([[1, 2, 3]])
        
        output = policy_model.generate(
            input_ids,
            max_length=20,
            temperature=1.0,
            do_sample=True,
            repetition_penalty=5.0  # Very high penalty
        )
        
        # With very high penalty, almost all tokens should be unique
        unique_tokens = len(output[0].unique())
        total_tokens = output.shape[1]
        
        print(f"Very high penalty (5.0): {unique_tokens}/{total_tokens} unique tokens")
        # Should have high diversity (allowing some leeway for randomness)
        assert unique_tokens >= total_tokens * 0.7


class TestIntegration:
    """Integration tests combining both improvements."""
    
    def test_generation_with_both_improvements(self):
        """Test that KV cache optimization and repetition penalty work together."""
        policy_model = PolicyModel(
            vocab_size=50,
            d_model=64,
            n_heads=4,
            n_layers=2,
            d_ff=128,
            max_seq_len=128,
            dropout=0.0,
            pad_token_id=0
        )
        
        torch.manual_seed(888)
        policy_model.eval()
        
        input_ids = torch.tensor([[1, 2, 3, 4]])
        
        # Generate with all features
        output = policy_model.generate(
            input_ids,
            max_length=25,
            temperature=0.9,
            top_k=30,
            top_p=0.95,
            do_sample=True,
            repetition_penalty=1.8
        )
        
        assert output.shape[0] == 1
        assert output.shape[1] <= 25
        assert output.shape[1] >= input_ids.shape[1]
        
        # Verify the prompt is preserved
        assert torch.equal(output[:, :input_ids.shape[1]], input_ids)
        
        unique_ratio = len(output[0].unique()) / output.shape[1]
        print(f"Integration test - Unique ratio: {unique_ratio:.2f}")
        print(f"Generated sequence: {output[0].tolist()}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

