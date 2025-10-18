# Architecture Overview

## System Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                        RLHF TRAINING SYSTEM                    │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│  PHASE 1: PRETRAINING                                          │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Shakespeare.txt ──► [Tokenizer] ──► [Policy Model]            │
│                                          │                     │
│                                          ▼                     │
│                               [Next-Token Prediction]          │
│                                          │                     │
│                                          ▼                     │
│                                [Trained Policy Model]          │
│                                                                │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│  PHASE 2: REWARD MODEL TRAINING                                │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Good Text ──┐                                                 │
│              ├──► [Reward Model] ──► [Preference Loss]         │
│  Bad Text ───┘           │                                     │
│                          ▼                                     │
│                  [Trained Reward Model]                        │
│                                                                │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│  PHASE 3: PPO RLHF                                             │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌─────────────┐      ┌──────────────┐                         │
│  │   Policy    │◄────►│  Reference   │ (KL Divergence)         │
│  │  (train)    │      │  (frozen)    │                         │
│  └──────┬──────┘      └──────────────┘                         │
│         │                                                      │
│         │ generates                                            │
│         ▼                                                      │
│    [Sequences]                                                 │
│         │                                                      │
│         ├──► [Reward Model] ──► Rewards                        │
│         │        (frozen)                                      │
│         │                                                      │
│         └──► [Value Model]  ──► Values                         │
│                  (train)         │                             │
│                                  ▼                             │
│                          [GAE: Advantages]                     │
│                                  │                             │
│                                  ▼                             │
│                           [PPO Update]                         │
│                            │          │                        │
│                            ▼          ▼                        │
│                       Policy    Value Model                    │
│                       Updated   Updated                        │
│                                                                │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│  PHASE 4: GRPO (Alternative to PPO)                            │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌─────────────┐      ┌──────────────┐                         │
│  │   Policy    │◄────►│  Reference   │ (KL Divergence)         │
│  │  (train)    │      │  (frozen)    │                         │
│  └──────┬──────┘      └──────────────┘                         │
│         │                                                      │
│         │ generates N samples per prompt                       │
│         ▼                                                      │
│    [Multiple Sequences]                                        │
│         │                                                      │
│         └──► [Reward Model] ──► Group of Rewards               │
│                  (frozen)         │                            │
│                                   ▼                            │
│                       [Group-Relative Advantages]              │
│                       (no value network!)                      │
│                                  │                             │
│                                  ▼                             │
│                           [PPO-style Update]                   │
│                                  │                             │
│                                  ▼                             │
│                           Policy Updated                       │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## Model Architectures

### Transformer Block

```
Input Embeddings (Token + Position)
         │
         ▼
┌──────────────────┐
│  Multi-Head      │
│  Self-Attention  │
│  (Causal Mask)   │
└────────┬─────────┘
         │
    Residual +
         │
         ▼
┌──────────────────┐
│  Layer Norm      │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Feed-Forward    │
│  (GELU)          │
└────────┬─────────┘
         │
    Residual +
         │
         ▼
┌──────────────────┐
│  Layer Norm      │
└────────┬─────────┘
         │
         ▼
      Output
```

### Policy Model

```
Input IDs
    │
    ▼
[Token Embedding]
    │
    ▼
[Position Embedding]
    │
    ▼
[Transformer Blocks] × N
    │
    ▼
[Layer Norm]
    │
    ▼
[LM Head] ──► Logits
```

### Reward Model

```
Input IDs
    │
    ▼
[Token Embedding]
    │
    ▼
[Position Embedding]
    │
    ▼
[Transformer Blocks] × N
    │
    ▼
[Layer Norm]
    │
    ▼
[Reward Head] ──► Scalar Rewards
```

### Value Model

```
Input IDs
    │
    ▼
[Token Embedding]
    │
    ▼
[Position Embedding]
    │
    ▼
[Transformer Blocks] × N
    │
    ▼
[Layer Norm]
    │
    ▼
[Value Head] ──► Value Estimates
```

## Data Flow

### Pretraining

```
Shakespeare Text
      │
      ▼
[Tokenizer.build_vocab()]
      │
      ▼
  Vocabulary
      │
      ▼
[Tokenizer.encode()]
      │
      ▼
  Token IDs
      │
      ▼
[Policy Model]
      │
      ▼
  Logits
      │
      ▼
[Cross-Entropy Loss]
      │
      ▼
[Backpropagation]
      │
      ▼
Updated Policy
```

### Reward Training

```
Good/Bad Text Pairs
      │
      ▼
[Tokenizer.encode()]
      │
      ▼
Chosen/Rejected IDs
      │
      ▼
[Reward Model]
      │
      ▼
Reward Scores
      │
      ▼
[Bradley-Terry Loss]
      │
      ▼
[Backpropagation]
      │
      ▼
Updated Reward Model
```

### PPO RLHF Loop

```
1. Sample Prompts
      │
      ▼
2. Policy.generate()
      │
      ▼
3. Compute Log Probs
   (Policy & Reference)
      │
      ▼
4. Get Rewards
   (Reward Model)
      │
      ▼
5. Compute KL Penalty
      │
      ▼
6. Final Rewards = 
   Rewards - KL
      │
      ▼
7. Get Values
   (Value Model)
      │
      ▼
8. Compute GAE
   (Advantages & Returns)
      │
      ▼
9. PPO Update
   (Policy & Value)
      │
      ▼
10. Repeat
```

### GRPO Loop

```
1. Sample Prompts
      │
      ▼
2. Generate N Completions
   per Prompt
      │
      ▼
3. Compute Log Probs
   (Policy & Reference)
      │
      ▼
4. Get Rewards
   (Reward Model)
      │
      ▼
5. Compute KL Penalty
      │
      ▼
6. Final Rewards = 
   Rewards - KL
      │
      ▼
7. Group-Relative
   Advantages
   (No Value Model!)
      │
      ▼
8. PPO-style Update
   (Policy Only)
      │
      ▼
9. Repeat
```

## Key Algorithms

### GAE (Generalized Advantage Estimation)

```python
δₜ = rₜ + γ·V(sₜ₊₁) - V(sₜ)
Aₜ = δₜ + γ·λ·Aₜ₊₁

where:
  δₜ = TD error
  Aₜ = advantage
  γ = discount factor
  λ = GAE lambda
```

### PPO Clipped Objective

```python
ratio = π_new(a|s) / π_old(a|s)
L = E[min(ratio·A, clip(ratio, 1-ε, 1+ε)·A)]

where:
  ratio = probability ratio
  A = advantage
  ε = clip epsilon (0.2)
```

### KL Divergence Penalty

```python
KL = E[log(π_policy(a|s)) - log(π_ref(a|s))]
Final_Reward = Reward - β·KL

where:
  β = KL coefficient (0.1)
```

### Group-Relative Advantage (GRPO)

```python
For each prompt:
  Generate N samples
  Get rewards: [r₁, r₂, ..., rₙ]
  Baseline = mean(rewards)
  Advantages = rewards - baseline
  Normalize advantages
```

## File Dependencies

```
models/tokenizer.py
    └─► (standalone)

models/transformer.py
    └─► (PyTorch only)

models/policy_model.py
    └─► transformer.py

models/reward_model.py
    └─► transformer.py

models/value_model.py
    └─► transformer.py

utils/gae.py
    └─► (PyTorch only)

utils/logging_utils.py
    └─► (TensorBoard)

training/pretrain.py
    └─► tokenizer.py
    └─► policy_model.py
    └─► logging_utils.py

training/train_reward.py
    └─► tokenizer.py
    └─► reward_model.py
    └─► logging_utils.py

training/ppo_rlhf.py
    └─► tokenizer.py
    └─► policy_model.py
    └─► reward_model.py
    └─► value_model.py
    └─► gae.py
    └─► logging_utils.py

training/grpo.py
    └─► tokenizer.py
    └─► policy_model.py
    └─► reward_model.py
    └─► logging_utils.py
```

## Configuration

### Model Sizes (Default)

```
Policy Model:
  - d_model: 256
  - n_heads: 8
  - n_layers: 6
  - d_ff: 1024
  - params: ~2M

Reward Model:
  - d_model: 256
  - n_heads: 8
  - n_layers: 4 (smaller)
  - d_ff: 1024
  - params: ~1.5M

Value Model:
  - d_model: 256
  - n_heads: 8
  - n_layers: 4
  - d_ff: 1024
  - params: ~1M
```

### Training Hyperparameters

```
Pretraining:
  - epochs: 10
  - batch_size: 32
  - learning_rate: 3e-4
  - seq_len: 128

Reward Training:
  - epochs: 5
  - batch_size: 32
  - learning_rate: 1e-4
  - num_samples: 5000

PPO RLHF:
  - num_steps: 100
  - batch_size: 16
  - learning_rate: 1e-5
  - value_lr: 3e-5
  - ppo_epochs: 4
  - clip_epsilon: 0.2
  - gamma: 0.99
  - lambda: 0.95
  - kl_coef: 0.1

GRPO:
  - num_steps: 100
  - batch_size: 16
  - learning_rate: 1e-5
  - num_samples_per_prompt: 4
  - ppo_epochs: 4
  - clip_epsilon: 0.2
  - kl_coef: 0.1
```

## Resource Requirements

### Minimal (CPU)
- RAM: 8GB
- Disk: 1GB
- Time: 3-4 hours full training

### Recommended (GPU)
- RAM: 16GB
- GPU: 6GB VRAM
- Disk: 2GB
- Time: 20-30 minutes full training

### Storage Breakdown
- Code: ~50MB
- Data: ~1MB (shakespeare.txt)
- Checkpoints: ~500MB (all models)
- Logs: ~100MB (TensorBoard)
- Virtual env: ~2GB (dependencies)

## Performance Metrics

### Pretraining
- Throughput: ~1000 tokens/sec (CPU)
- Throughput: ~10000 tokens/sec (GPU)
- Loss: 4.0 → 1.5-2.0

### RL Training
- Samples/sec: ~50 (PPO, CPU)
- Samples/sec: ~200 (PPO, GPU)
- Samples/sec: ~30 (GRPO, CPU) - more samples per prompt
- Samples/sec: ~120 (GRPO, GPU)

## Extensibility Points

1. **Custom Tokenizers**: Replace CharTokenizer
2. **Different Architectures**: Modify transformer.py
3. **New Reward Functions**: Extend reward_model.py
4. **Alternative RL Algorithms**: Add new trainers
5. **Evaluation Metrics**: Add to logging_utils.py
6. **Distributed Training**: Add DDP/FSDP wrappers
7. **Mixed Precision**: Add AMP support

---

This architecture is designed to be:
- **Modular**: Each component is independent
- **Extensible**: Easy to add new features
- **Educational**: Clear structure for learning
- **Production-Ready**: Can scale to larger models

