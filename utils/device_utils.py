"""Device utilities for cross-platform GPU support."""

import torch


def get_device(device_str='auto'):
    """Get the best available device.
    
    Args:
        device_str: Device string ('auto', 'cpu', 'cuda', 'mps', or specific device).
        
    Returns:
        torch.device: Selected device.
    """
    if device_str == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
            print("Using Apple Metal (MPS) GPU")
        else:
            device = torch.device('cpu')
            print("Using CPU")
    else:
        device = torch.device(device_str)
        print(f"Using device: {device}")
    
    return device


def optimize_for_device(model, device):
    """Apply device-specific optimizations.
    
    Args:
        model: PyTorch model.
        device: torch.device.
        
    Returns:
        Optimized model.
    """
    model = model.to(device)
    
    # MPS-specific optimizations
    if device.type == 'mps':
        # Ensure model is in eval mode when not training for MPS
        # MPS can be more memory efficient with float32
        pass
    
    return model

