"""Overfitting tests to ensure models can converge on small datasets."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn.functional as F
from models import CharTokenizer, PolicyModel, RewardModel, ValueModel
from utils import compute_gae


def test_policy_overfit():
    """Test that policy model can overfit a short sentence."""
    print("\nTesting policy model overfitting...")
    
    # Create a simple dataset - one short sentence
    text = "To be or not to be"
    
    tokenizer = CharTokenizer()
    tokenizer.build_vocab(text)
    
    token_ids = tokenizer.encode(text, add_special_tokens=True)
    
    # Create model
    model = PolicyModel(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=512,
        max_seq_len=len(token_ids),
        dropout=0.0
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Prepare data
    input_ids = torch.tensor([token_ids], dtype=torch.long)
    
    # Train
    initial_loss = None
    final_loss = None
    
    for step in range(200):
        logits, loss = model(input_ids, labels=input_ids)
        
        if step == 0:
            initial_loss = loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 50 == 0:
            print(f"  Step {step}: Loss = {loss.item():.6f}")
    
    final_loss = loss.item()
    
    # Check convergence
    assert final_loss < initial_loss * 0.1, \
        f"Model did not converge: initial={initial_loss:.4f}, final={final_loss:.4f}"
    
    # Test generation
    model.eval()
    with torch.no_grad():
        prompt = input_ids[:, :5]
        generated = model.generate(prompt, max_length=len(token_ids), temperature=0.1, do_sample=False)
        generated_text = tokenizer.decode(generated[0].tolist())
        
        print(f"  Original: {text}")
        print(f"  Generated: {generated_text}")
    
    print("✓ Policy model can overfit successfully")
    return True


def test_reward_overfit():
    """Test that reward model can learn to distinguish good from bad."""
    print("\nTesting reward model overfitting...")
    
    # Create simple preference data
    good_text = "Hello World"
    bad_text = "hElLo wOrLd"  # Mixed case - worse
    
    tokenizer = CharTokenizer()
    tokenizer.build_vocab(good_text + bad_text)
    
    good_ids = tokenizer.encode(good_text, add_special_tokens=True)
    bad_ids = tokenizer.encode(bad_text, add_special_tokens=True)
    
    # Pad to same length
    max_len = max(len(good_ids), len(bad_ids))
    good_ids += [tokenizer.pad_token_id] * (max_len - len(good_ids))
    bad_ids += [tokenizer.pad_token_id] * (max_len - len(bad_ids))
    
    chosen = torch.tensor([good_ids], dtype=torch.long)
    rejected = torch.tensor([bad_ids], dtype=torch.long)
    
    # Create model
    model = RewardModel(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=512,
        max_seq_len=max_len,
        dropout=0.0
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Create attention masks
    batch_size = chosen.size(0)
    mask = torch.ones(batch_size, max_len)
    
    # Train
    initial_loss = None
    final_loss = None
    
    for step in range(200):
        loss = model.compute_preference_loss(chosen, rejected, mask, mask)
        
        if step == 0:
            initial_loss = loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 50 == 0:
            with torch.no_grad():
                good_reward = model.get_sequence_reward(chosen, attention_mask=mask).mean().item()
                bad_reward = model.get_sequence_reward(rejected, attention_mask=mask).mean().item()
                print(f"  Step {step}: Loss = {loss.item():.6f}, "
                      f"Good = {good_reward:.4f}, Bad = {bad_reward:.4f}")
    
    final_loss = loss.item()
    
    # Check convergence
    assert final_loss < initial_loss * 0.5, \
        f"Model did not converge: initial={initial_loss:.4f}, final={final_loss:.4f}"
    
    # Check that good has higher reward than bad
    with torch.no_grad():
        good_reward = model.get_sequence_reward(chosen, attention_mask=mask).mean().item()
        bad_reward = model.get_sequence_reward(rejected, attention_mask=mask).mean().item()
        
        assert good_reward > bad_reward, \
            f"Good reward ({good_reward:.4f}) should be > bad reward ({bad_reward:.4f})"
    
    print("✓ Reward model can overfit successfully")
    return True


def test_value_overfit():
    """Test that value model can learn to predict returns."""
    print("\nTesting value model overfitting...")
    
    # Create simple sequence
    text = "Test sequence"
    
    tokenizer = CharTokenizer()
    tokenizer.build_vocab(text)
    
    token_ids = tokenizer.encode(text, add_special_tokens=True)
    input_ids = torch.tensor([token_ids], dtype=torch.long)
    
    # Create synthetic returns (e.g., decreasing)
    seq_len = len(token_ids)
    target_returns = torch.linspace(10.0, 0.0, seq_len).unsqueeze(0)
    
    # Create model
    model = ValueModel(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=512,
        max_seq_len=seq_len,
        dropout=0.0
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Train
    initial_loss = None
    final_loss = None
    
    for step in range(200):
        predicted_values = model(input_ids)
        loss = F.mse_loss(predicted_values, target_returns)
        
        if step == 0:
            initial_loss = loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 50 == 0:
            print(f"  Step {step}: Loss = {loss.item():.6f}")
    
    final_loss = loss.item()
    
    # Check convergence
    assert final_loss < initial_loss * 0.1, \
        f"Model did not converge: initial={initial_loss:.4f}, final={final_loss:.4f}"
    
    print("✓ Value model can overfit successfully")
    return True


def test_ppo_components():
    """Test PPO components work together."""
    print("\nTesting PPO component integration...")
    
    vocab_size = 50
    batch_size = 2
    seq_len = 10
    
    # Create models
    policy = PolicyModel(
        vocab_size=vocab_size,
        d_model=64,
        n_heads=2,
        n_layers=2,
        d_ff=256,
        max_seq_len=seq_len,
        dropout=0.0
    )
    
    value_model = ValueModel(
        vocab_size=vocab_size,
        d_model=64,
        n_heads=2,
        n_layers=2,
        d_ff=256,
        max_seq_len=seq_len,
        dropout=0.0
    )
    
    reward_model = RewardModel(
        vocab_size=vocab_size,
        d_model=64,
        n_heads=2,
        n_layers=2,
        d_ff=256,
        max_seq_len=seq_len,
        dropout=0.0
    )
    
    # Generate fake data
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Test forward passes
    with torch.no_grad():
        # Policy
        logits, _ = policy(input_ids)
        log_probs = policy.get_log_probs(input_ids)
        
        # Value
        values = value_model(input_ids)
        
        # Rewards
        rewards = reward_model(input_ids)
        
        # GAE
        dones = torch.zeros(batch_size, seq_len)
        dones[:, -1] = 1.0
        advantages, returns = compute_gae(rewards, values, dones)
    
    assert logits.shape == (batch_size, seq_len, vocab_size)
    assert values.shape == (batch_size, seq_len)
    assert rewards.shape == (batch_size, seq_len)
    assert advantages.shape == (batch_size, seq_len)
    assert returns.shape == (batch_size, seq_len)
    
    print("✓ PPO components integrate successfully")
    return True


if __name__ == '__main__':
    print("="*60)
    print("Running overfitting tests...")
    print("="*60)
    
    results = []
    
    try:
        results.append(("Policy Overfit", test_policy_overfit()))
    except Exception as e:
        print(f"✗ Policy overfit test failed: {e}")
        results.append(("Policy Overfit", False))
    
    try:
        results.append(("Reward Overfit", test_reward_overfit()))
    except Exception as e:
        print(f"✗ Reward overfit test failed: {e}")
        results.append(("Reward Overfit", False))
    
    try:
        results.append(("Value Overfit", test_value_overfit()))
    except Exception as e:
        print(f"✗ Value overfit test failed: {e}")
        results.append(("Value Overfit", False))
    
    try:
        results.append(("PPO Integration", test_ppo_components()))
    except Exception as e:
        print(f"✗ PPO integration test failed: {e}")
        results.append(("PPO Integration", False))
    
    print("\n" + "="*60)
    print("Overfitting Test Results:")
    print("="*60)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:30s}: {status}")
    
    all_passed = all(passed for _, passed in results)
    if all_passed:
        print("\n" + "="*60)
        print("All overfitting tests passed!")
        print("="*60)
    else:
        print("\nSome tests failed. Please check the output above.")
        sys.exit(1)

