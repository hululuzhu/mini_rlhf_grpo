# Mini Transformer LLM with PPO-RLHF and GRPO

A complete PyTorch implementation of a character-level transformer language model with Reinforcement Learning from Human Feedback (RLHF) using PPO and DeepSeek GRPO variants. Features modular architecture, comprehensive testing, and educational design for understanding RLHF training dynamics.
- Code is also optimized for Mac MPS

## 🏗️ Architecture Overview

This project implements a full mini-RLHF pipeline with multiple training phases:

1. **Pretraining**: Character-level transformer trained on Shakespeare text via next-token prediction
2. **Reward Modeling**: Preference-based reward model trained on human feedback data
3. **PPO RLHF**: Proximal Policy Optimization with value networks and GAE advantage estimation
4. **GRPO Alternative**: DeepSeek GRPO variant eliminating value network dependency

**Key Innovation**: Implements both PPO and GRPO algorithms for comparative study of RLHF approaches.

## 🚀 Quick Start

**New to this project? Start here:**
1. 📖 Read [`QUICK_START.md`](QUICK_START.md) for step-by-step setup
2. 🎯 Check [`PROJECT_SUMMARY.md`](PROJECT_SUMMARY.md) for complete overview
3. 📝 Review [`IMPLEMENTATION_NOTES.md`](IMPLEMENTATION_NOTES.md) for technical details
4. 🏗️ See [`markdowns/architecture.md`](markdowns/architecture.md) for detailed architecture

**TL;DR:**
```bash
./setup.sh && source venv/bin/activate
python3 quick_demo.py          # Verify installation
python3 tests/test_models.py   # Run tests
python3 run_all.py --quick     # Train everything
```

## ✅ Status: COMPLETE

All components implemented, tested, and ready to use!

## Project Structure

```
rl_demos/
├── models/                  # Core model implementations
│   ├── __init__.py          # Package initialization
│   ├── config.py            # Model configuration utilities
│   ├── transformer.py       # Base transformer architecture with causal attention
│   ├── policy_model.py      # Policy model for text generation
│   ├── reward_model.py      # Reward model for preference scoring
│   ├── value_model.py       # Value network for PPO advantage estimation
│   └── tokenizer.py         # Character-level tokenizer with vocab management
│
├── training/                # Training pipelines and scripts
│   ├── __init__.py          # Package initialization
│   ├── pretrain.py          # Policy model pretraining on Shakespeare text
│   ├── train_reward.py      # Reward model training on preference pairs
│   ├── ppo_rlhf.py          # PPO-based RLHF with value networks & GAE
│   └── grpo.py              # DeepSeek GRPO without value network dependency
│
├── utils/                   # Utility functions and helpers
│   ├── __init__.py          # Package initialization
│   ├── device_utils.py      # GPU/CPU device management utilities
│   ├── gae.py               # Generalized Advantage Estimation implementation
│   ├── logging_utils.py     # TensorBoard logging and metrics tracking
│   └── rl_utils.py          # RL-specific utilities and helper functions
│
├── tests/                   # Comprehensive test suite
│   ├── __init__.py          # Package initialization
│   ├── test_models.py       # Model architecture and functionality tests
│   ├── test_overfit.py      # Overfitting tests for training convergence
│   ├── test_improvements.py # Improvement validation tests
│   ├── demo_generation.py   # Text generation demonstration
│   ├── demo_kv_cache.py     # KV-cache optimization demo
│   └── demo_improvements.py # Improvement showcase scripts
│
├── architecture.md          # Detailed system architecture and data flow
├── quick_demo.py            # Quick installation verification script
├── requirements.txt         # Python dependencies
├── run_all.py               # Complete training pipeline runner
├── setup.sh                 # Environment setup script
└── README.md                # This file
```

## Installation

```bash
pip install -r requirements.txt
```

## Core Components

### 1. **Base Transformer Architecture** (`models/transformer.py`)
Causal transformer implementation with multi-head self-attention, feed-forward networks, and layer normalization. Serves as the foundation for all model variants.

### 2. **Character-Level Tokenizer** (`models/tokenizer.py`)
Custom tokenizer handling character-level tokenization with vocabulary management, encoding/decoding, and special token support.

### 3. **Policy Model** (`models/policy_model.py`)
Language model for text generation using the transformer architecture with causal attention masking and next-token prediction.

### 4. **Reward Model** (`models/reward_model.py`)
Preference-based reward model that scores text quality using Bradley-Terry loss on paired preference data.

> **⚠️ Note**: Current reward model implementation is basic and needs significant improvement for production use. The model struggles with preference consistency and may require architectural enhancements, better training data, or alternative reward modeling approaches.

### 5. **Value Network** (`models/value_model.py`)
Token-level value function estimator for PPO training, providing stable advantage estimation through GAE.

### 6. **PPO RLHF Training** (`training/ppo_rlhf.py`)
Complete PPO implementation featuring:
- Trainable policy model with frozen reference model (KL divergence penalty)
- Frozen reward model for preference scoring
- Trainable value network with GAE advantage estimation
- Clipped probability ratio objective for stable training

### 7. **GRPO Training** (`training/grpo.py`)
DeepSeek GRPO variant that eliminates value network dependency through:
- Group-relative advantage estimation across multiple samples per prompt
- Baseline computation from reward distribution within each prompt group
- Advantage normalization for stable policy updates

## Usage

### Complete Training Pipeline
```bash
# Run all training phases sequentially
python run_all.py --quick     # Quick mode with reduced steps
python run_all.py            # Full training pipeline

# Individual training phases
python training/pretrain.py     # Pretrain policy model
python training/train_reward.py # Train reward model
python training/ppo_rlhf.py     # PPO RLHF training
python training/grpo.py         # GRPO training
```

### Verification and Demos
```bash
python quick_demo.py              # Verify installation and basic functionality
python tests/demo_generation.py   # Text generation demonstration
python tests/demo_improvements.py # Show RLHF improvements
python tests/demo_kv_cache.py     # KV-cache optimization demo
```

## Testing

### Unit Tests
```bash
pytest tests/test_models.py       # Model architecture tests
pytest tests/test_improvements.py # Improvement validation
pytest tests/                     # Run complete test suite
```

### Training Validation
```bash
pytest tests/test_overfit.py -v   # Overfitting tests for convergence
pytest tests/test_critical_fixes.py # Critical functionality tests
pytest tests/test_review_v2_fixes.py # Review-based validation
```

### Continuous Integration
```bash
python run_all.py --test-only    # Run tests without training
```

## Key Features

- ✅ **Character-level tokenization** with custom vocabulary management
- ✅ **Causal transformer architecture** with multi-head attention and feed-forward networks
- ✅ **Preference-based reward modeling** using Bradley-Terry loss on human feedback pairs
- ✅ **PPO RLHF implementation** with GAE advantage estimation and value networks
- ✅ **KL divergence regularization** with frozen reference model for training stability
- ✅ **DeepSeek GRPO variant** eliminating value network dependency through group-relative advantages
- ✅ **Comprehensive logging** with TensorBoard integration for training metrics
- ✅ **Extensive test suite** including unit tests, overfitting validation, and improvement demonstrations
- ✅ **Modular architecture** with clean separation of concerns across models, training, and utilities
- ✅ **Device management** utilities for GPU/CPU training with automatic mixed precision support
- ✅ **KV-cache optimization** demonstrations for efficient inference
- ✅ **Complete training pipeline** automation with configurable execution modes

## 📚 Documentation

- **[`QUICK_START.md`](QUICK_START.md)**: Step-by-step setup and usage guide
- **[`PROJECT_SUMMARY.md`](PROJECT_SUMMARY.md)**: Complete project overview and status
- **[`IMPLEMENTATION_NOTES.md`](IMPLEMENTATION_NOTES.md)**: Detailed technical documentation
- **[`markdowns/architecture.md`](markdowns/architecture.md)**: Comprehensive system architecture and data flow diagrams
- **README.md**: This file - general information and project overview

## 🔮 Future Work & Next Steps

### Immediate Improvements Needed
- **Enhanced Reward Modeling**: Improve preference consistency and explore alternative reward architectures
- **Larger Scale Training**: Scale to larger datasets and model sizes for better performance
- **Evaluation Metrics**: Implement comprehensive evaluation beyond basic generation quality

### RLVF Integration (DeepSeek R1)
Next phase involves implementing **Reinforcement Learning with Verifiable Feedback (RLVF)** from DeepSeek, so that we will further replace the expensive reward model with some verifiable feedback like math/lean/general_code execution feedback if applicable.

See [DeepSeek R1](https://arxiv.org/abs/2501.12948) for detailed methodology.

## 🎓 References

### Core Papers
- [Attention Is All You Need (Transformer)](https://arxiv.org/abs/1706.03762) - Vaswani et al. (2017)
- [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) - Schulman et al. (2017)
- [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155) - Ouyang et al. (2022)
- [High-dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) - Schulman et al. (2016)

### GRPO & DeepSeek
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300) - DeepSeek AI (2024)
- [DeepSeek-Coder: When the Large Language Model Meets Programming](https://arxiv.org/abs/2401.14196) - DeepSeek AI (2024)
- [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434) - DeepSeek AI (2024)

### RLVF & Verifiable Feedback
- [DeepSeek R1](https://arxiv.org/abs/2501.12948) - DeepSeek AI (2025)

## 🤝 Contributing

This is a complete educational implementation. Feel free to:
- Experiment with different hyperparameters
- Try different model architectures
- Implement additional RL algorithms
- Scale to larger datasets
- Add evaluation metrics

## 📄 License

MIT License - feel free to use for learning and research.

## 🙏 Acknowledgments

- Shakespeare text from [Andrej Karpathy's char-rnn](https://github.com/karpathy/char-rnn)
- Inspired by papers from OpenAI, DeepMind, and DeepSeek
- Built with PyTorch and HuggingFace ecosystem
- AI Coding assistants
  - Cursor v1.7.52 + Sonnet V4.5