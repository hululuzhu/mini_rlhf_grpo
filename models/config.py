"""Configuration classes for training pipeline."""

from dataclasses import dataclass, asdict
from typing import Literal


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    d_model: int
    n_heads: int
    n_layers_policy: int
    n_layers_reward: int
    d_ff: int
    seq_len: int
    
    def __str__(self):
        """Pretty print configuration."""
        lines = ["Model Configuration:"]
        lines.append(f"  • Model dimension (d_model): {self.d_model}")
        lines.append(f"  • Attention heads (n_heads): {self.n_heads}")
        lines.append(f"  • Policy layers: {self.n_layers_policy}")
        lines.append(f"  • Reward layers: {self.n_layers_reward}")
        lines.append(f"  • Feed-forward dimension (d_ff): {self.d_ff}")
        lines.append(f"  • Sequence length: {self.seq_len}")
        return "\n".join(lines)


@dataclass
class TrainingConfig:
    """Training hyperparameters configuration."""
    pretrain_epochs: int
    reward_epochs: int
    rl_steps: int
    batch_size: int
    
    def __str__(self):
        """Pretty print configuration."""
        lines = ["Training Configuration:"]
        lines.append(f"  • Pretrain epochs: {self.pretrain_epochs}")
        lines.append(f"  • Reward epochs: {self.reward_epochs}")
        lines.append(f"  • RL steps: {self.rl_steps}")
        lines.append(f"  • Batch size: {self.batch_size}")
        return "\n".join(lines)


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""
    model: ModelConfig
    training: TrainingConfig
    device: str
    mode: Literal["quick", "normal"]
    
    def __str__(self):
        """Pretty print complete configuration."""
        from utils.logging_utils import Colors
        
        lines = []
        lines.append(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.ENDC}")
        lines.append(f"{Colors.BOLD}{Colors.CYAN}Pipeline Configuration ({self.mode.upper()} mode){Colors.ENDC}")
        lines.append(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.ENDC}\n")
        lines.append(f"{Colors.BOLD}Device:{Colors.ENDC} {Colors.GREEN}{self.device}{Colors.ENDC}\n")
        lines.append(str(self.model))
        lines.append("")
        lines.append(str(self.training))
        lines.append(f"\n{Colors.CYAN}{'='*60}{Colors.ENDC}\n")
        return "\n".join(lines)
    
    def to_dict(self):
        """Convert to dictionary for easy parameter passing."""
        return {
            **asdict(self.model),
            **asdict(self.training),
            'device': self.device
        }


# Predefined configurations
QUICK_CONFIG = PipelineConfig(
    model=ModelConfig(
        d_model=128,
        n_heads=4,
        n_layers_policy=4,
        n_layers_reward=2,
        d_ff=512,
        seq_len=64
    ),
    training=TrainingConfig(
        pretrain_epochs=2,
        reward_epochs=2,
        rl_steps=50,
        batch_size=8
    ),
    device="auto",  # Will be set at runtime
    mode="quick"
)

NORMAL_CONFIG = PipelineConfig(
    model=ModelConfig(
        d_model=256,
        n_heads=8,
        n_layers_policy=6,
        n_layers_reward=4,
        d_ff=1024,
        seq_len=128
    ),
    training=TrainingConfig(
        pretrain_epochs=10,
        reward_epochs=5,
        rl_steps=100,
        batch_size=32
    ),
    device="auto",  # Will be set at runtime
    mode="normal"
)


def detect_device() -> str:
    """Auto-detect the best available device (MPS > CUDA > CPU)."""
    import torch
    
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


def get_config(quick: bool = False, device: str = "auto") -> PipelineConfig:
    """Get pipeline configuration.
    
    Args:
        quick: If True, use quick configuration, else use normal configuration
        device: Device to use ('auto', 'mps', 'cuda', or 'cpu')
    
    Returns:
        PipelineConfig with appropriate settings
    """
    config = QUICK_CONFIG if quick else NORMAL_CONFIG
    
    # Set device
    if device == "auto":
        config.device = detect_device()
    else:
        config.device = device
    
    return config

