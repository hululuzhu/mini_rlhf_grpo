"""Demo script to test generation from trained models."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from models import CharTokenizer, PolicyModel

def generate_samples(model, tokenizer, prompt, max_length=100, temperature=0.8):
    """Generate text from a prompt."""
    model.eval()
    device = next(model.parameters()).device
    
    # Encode prompt
    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    
    # Generate
    with torch.no_grad():
        for _ in range(max_length):
            logits, _ = model(tokens)  # model returns (logits, loss)
            next_token_logits = logits[0, -1, :] / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
            
            # Stop at end token if exists
            if next_token.item() == tokenizer.pad_token_id:
                break
    
    # Decode
    generated_tokens = tokens[0].cpu().tolist()
    return tokenizer.decode(generated_tokens)


def main():
    """Test generation from different model checkpoints."""
    # Load tokenizer
    tokenizer = CharTokenizer()
    with open('data/shakespeare.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    tokenizer.build_vocab(text)
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Test prompt
    prompt = "ROMEO:"
    print(f"Prompt: {prompt}")
    print("=" * 80)
    
    # Load and test pretrained model
    print("\n1. PRETRAINED MODEL (before RLHF):")
    print("-" * 80)
    policy_pretrained = PolicyModel(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ff=512,
        max_seq_len=64
    ).to(device)
    
    checkpoint = torch.load('checkpoints/pretrain/final_model.pt', map_location=device)
    policy_pretrained.load_state_dict(checkpoint['model_state_dict'])
    
    text_pretrained = generate_samples(policy_pretrained, tokenizer, prompt, max_length=50)
    print(text_pretrained[:200] + "...")
    
    # Load and test PPO model
    print("\n2. PPO-RLHF MODEL (after reinforcement learning):")
    print("-" * 80)
    policy_ppo = PolicyModel(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ff=512,
        max_seq_len=64
    ).to(device)
    
    checkpoint = torch.load('checkpoints/ppo_rlhf/final_ppo_policy.pt', map_location=device)
    policy_ppo.load_state_dict(checkpoint['model_state_dict'])
    
    text_ppo = generate_samples(policy_ppo, tokenizer, prompt, max_length=50)
    print(text_ppo[:200] + "...")
    
    # Load and test GRPO model
    print("\n3. GRPO MODEL (Group Relative Policy Optimization):")
    print("-" * 80)
    policy_grpo = PolicyModel(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ff=512,
        max_seq_len=64
    ).to(device)
    
    checkpoint = torch.load('checkpoints/grpo/final_grpo_policy.pt', map_location=device)
    policy_grpo.load_state_dict(checkpoint['model_state_dict'])
    
    text_grpo = generate_samples(policy_grpo, tokenizer, prompt, max_length=50)
    print(text_grpo[:200] + "...")
    
    print("\n" + "=" * 80)
    print("✅ Generation test complete! All models loaded and generated text successfully.")


if __name__ == '__main__':
    main()

