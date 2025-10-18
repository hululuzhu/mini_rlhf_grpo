"""Quick demo script to verify installation and run minimal tests."""

import os
import sys

# ANSI color codes
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.ENDC}\n")

def print_section(number, text):
    print(f"{Colors.BOLD}{Colors.BLUE}{number}. {text}{Colors.ENDC}")

def print_success(text):
    print(f"{Colors.GREEN}   ✓ {text}{Colors.ENDC}")

def print_error(text):
    print(f"{Colors.RED}   ✗ {text}{Colors.ENDC}")

def print_info(text):
    print(f"   {text}")

print_header("RL Demos - Quick Demo")

# Check Python version
print_section(1, "Checking Python version...")
print_info(f"Python {sys.version}")
print_success("Python OK")
print()

# Check if PyTorch is available
print_section(2, "Checking PyTorch installation...")
try:
    import torch
    print_info(f"PyTorch {torch.__version__}")
    
    # Check for Mac GPU (MPS), then CUDA, then CPU
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        accelerator = "Mac GPU (MPS)"
        device = "mps"
        print_success(f"Accelerator: {accelerator}")
    elif torch.cuda.is_available():
        accelerator = "CUDA GPU"
        device = "cuda"
        print_success(f"Accelerator: {accelerator}")
    else:
        accelerator = "CPU"
        device = "cpu"
        print_info(f"Accelerator: {accelerator}")
    
    print_success("PyTorch OK")
except ImportError:
    print_error("PyTorch not installed")
    print_info("Please run: pip install -r requirements.txt")
    sys.exit(1)
print()

# Check project structure
print_section(3, "Checking project structure...")
required_dirs = ['data', 'models', 'training', 'utils', 'tests']
all_dirs_exist = True
for dir_name in required_dirs:
    if os.path.exists(dir_name):
        print_success(f"{dir_name}/ exists")
    else:
        print_error(f"{dir_name}/ missing")
        all_dirs_exist = False
print()

# Check if Shakespeare data exists
print_section(4, "Checking data...")
if os.path.exists('data/shakespeare.txt'):
    size = os.path.getsize('data/shakespeare.txt')
    print_success(f"shakespeare.txt exists ({size:,} bytes)")
else:
    print_error("shakespeare.txt not found")
    print_info("Download with:")
    print_info("  curl -o data/shakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")
print()

# Run minimal model test
print_section(5, "Running minimal model test...")
try:
    from models import CharTokenizer, PolicyModel
    
    # Test tokenizer
    tokenizer = CharTokenizer()
    tokenizer.build_vocab("Hello World!")
    encoded = tokenizer.encode("Hello")
    decoded = tokenizer.decode(encoded)
    print_info(f"Tokenizer: 'Hello' -> {encoded} -> '{decoded}'")
    print_success("Tokenizer OK")
    
    # Test policy model
    model = PolicyModel(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        n_heads=2,
        n_layers=2,
        d_ff=256,
        max_seq_len=32
    )
    
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    logits, _ = model(input_ids)
    print_info(f"Policy model output shape: {logits.shape}")
    print_success("Policy model OK")
    
except Exception as e:
    print_error(f"Model test failed: {e}")
    import traceback
    traceback.print_exc()
print()

print_header("Quick Demo Complete!")
print(f"{Colors.BOLD}Next steps:{Colors.ENDC}")
print(f"{Colors.YELLOW}1. Run unit tests: {Colors.ENDC}python3 tests/test_models.py")
print(f"{Colors.YELLOW}2. Run overfitting tests: {Colors.ENDC}python3 tests/test_overfit.py")
print(f"{Colors.YELLOW}3. Run full pipeline: {Colors.ENDC}python3 run_all.py --quick")
print()

