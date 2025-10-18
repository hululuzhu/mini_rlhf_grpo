"""Logging utilities for training."""

import os
import json
from typing import Dict, Any, Optional
from torch.utils.tensorboard import SummaryWriter
import torch


class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class MetricsLogger:
    """Logger for training metrics."""
    
    def __init__(
        self,
        log_dir: str,
        experiment_name: str = "experiment",
        use_tensorboard: bool = True
    ):
        """Initialize metrics logger.
        
        Args:
            log_dir: Directory for logs.
            experiment_name: Name of the experiment.
            use_tensorboard: Whether to use TensorBoard.
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.use_tensorboard = use_tensorboard
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize TensorBoard writer
        if use_tensorboard:
            self.writer = SummaryWriter(log_dir=os.path.join(log_dir, experiment_name))
        else:
            self.writer = None
        
        # Metrics storage
        self.metrics = []
    
    def log_scalar(self, name: str, value: float, step: int):
        """Log scalar metric.
        
        Args:
            name: Metric name.
            value: Metric value.
            step: Training step.
        """
        if self.writer is not None:
            self.writer.add_scalar(name, value, step)
        
        self.metrics.append({
            'name': name,
            'value': value,
            'step': step
        })
    
    def log_scalars(self, metrics: Dict[str, float], step: int):
        """Log multiple scalar metrics.
        
        Args:
            metrics: Dictionary of metric name to value.
            step: Training step.
        """
        for name, value in metrics.items():
            self.log_scalar(name, value, step)
    
    def log_text(self, name: str, text: str, step: int):
        """Log text.
        
        Args:
            name: Text name.
            text: Text content.
            step: Training step.
        """
        if self.writer is not None:
            self.writer.add_text(name, text, step)
    
    def log_histogram(self, name: str, values: torch.Tensor, step: int):
        """Log histogram.
        
        Args:
            name: Histogram name.
            values: Tensor of values.
            step: Training step.
        """
        if self.writer is not None:
            self.writer.add_histogram(name, values, step)
    
    def save_metrics(self):
        """Save metrics to JSON file."""
        metrics_file = os.path.join(self.log_dir, f"{self.experiment_name}_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def close(self):
        """Close logger."""
        if self.writer is not None:
            self.writer.close()
        self.save_metrics()


def print_training_stats(step: int, metrics: Dict[str, Any]):
    """Pretty print training statistics.
    
    Args:
        step: Training step.
        metrics: Dictionary of metrics to print.
    """
    print(f"\n{'='*60}")
    print(f"Step {step}")
    print(f"{'='*60}")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:30s}: {value:.6f}")
        else:
            print(f"{key:30s}: {value}")
    print(f"{'='*60}\n")

