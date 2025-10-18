"""Master script to run the entire training pipeline."""

import subprocess
import argparse
import sys
from models.config import get_config
from utils.logging_utils import Colors

# Use the same Python interpreter that's running this script
PYTHON = sys.executable


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}Running: {description}{Colors.ENDC}")
    print(f"{Colors.CYAN}{'='*60}{Colors.ENDC}")
    print(f"{Colors.YELLOW}Command:{Colors.ENDC} {cmd}")
    print()
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"\n{Colors.RED}❌ Failed: {description}{Colors.ENDC}")
        return False
    else:
        print(f"\n{Colors.GREEN}✅ Completed: {description}{Colors.ENDC}")
        return True


def main():
    parser = argparse.ArgumentParser(
        description='Run complete training pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline in quick mode
  python3 run_all.py --quick
  
  # Run only pretraining and reward training
  python3 run_all.py --skip-tests --skip-ppo --skip-grpo
  
  # Use CPU for all training
  python3 run_all.py --cpu
        """
    )
    
    # Pipeline stage flags
    parser.add_argument('--skip-tests', action='store_true', 
                        help='Skip unit and overfitting tests')
    parser.add_argument('--skip-pretrain', action='store_true', 
                        help='Skip pretraining stage')
    parser.add_argument('--skip-reward', action='store_true', 
                        help='Skip reward model training')
    parser.add_argument('--skip-ppo', action='store_true', 
                        help='Skip PPO RLHF training')
    parser.add_argument('--skip-grpo', action='store_true', 
                        help='Skip GRPO training')
    
    # Configuration flags
    parser.add_argument('--quick', action='store_true', 
                        help='Use quick mode (reduced epochs/steps, smaller model)')
    
    # Device flags (mutually exclusive)
    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument('--mps', action='store_true', 
                              help='Force Mac GPU (MPS) device')
    device_group.add_argument('--cuda', action='store_true', 
                              help='Force CUDA GPU device')
    device_group.add_argument('--cpu', action='store_true', 
                              help='Force CPU device')
    
    args = parser.parse_args()
    
    # Determine device
    if args.mps:
        device = 'mps'
    elif args.cuda:
        device = 'cuda'
    elif args.cpu:
        device = 'cpu'
    else:
        device = 'auto'
    
    # Get configuration
    config = get_config(quick=args.quick, device=device)
    
    # Print configuration
    print(config)
    
    results = []
    
    # Step 1: Run unit tests
    if not args.skip_tests:
        success = run_command(
            f"{PYTHON} tests/test_models.py",
            "Unit Tests"
        )
        results.append(("Unit Tests", success))
        
        if not success:
            print(f"\n{Colors.YELLOW}⚠️  Unit tests failed, but continuing...{Colors.ENDC}")
    
    # Step 2: Run overfitting tests
    if not args.skip_tests:
        success = run_command(
            f"{PYTHON} tests/test_overfit.py",
            "Overfitting Tests"
        )
        results.append(("Overfitting Tests", success))
        
        if not success:
            print(f"\n{Colors.YELLOW}⚠️  Overfitting tests failed, but continuing...{Colors.ENDC}")
    
    # Step 3: Pretrain policy model
    if not args.skip_pretrain:
        success = run_command(
            f"{PYTHON} training/pretrain.py "
            f"--data_path data/shakespeare.txt "
            f"--output_dir checkpoints/pretrain "
            f"--epochs {config.training.pretrain_epochs} "
            f"--batch_size {config.training.batch_size} "
            f"--d_model {config.model.d_model} "
            f"--n_heads {config.model.n_heads} "
            f"--n_layers {config.model.n_layers_policy} "
            f"--d_ff {config.model.d_ff} "
            f"--seq_len {config.model.seq_len} "
            f"--device {config.device}",
            "Pretrain Policy Model"
        )
        results.append(("Pretrain", success))
        
        if not success:
            print(f"\n{Colors.RED}❌ Pretraining failed. Cannot continue.{Colors.ENDC}")
            return
    
    # Step 4: Train reward model
    if not args.skip_reward:
        success = run_command(
            f"{PYTHON} training/train_reward.py "
            f"--data_path data/shakespeare.txt "
            f"--tokenizer_path checkpoints/pretrain/tokenizer.pkl "
            f"--output_dir checkpoints/reward "
            f"--epochs {config.training.reward_epochs} "
            f"--batch_size {config.training.batch_size} "
            f"--d_model {config.model.d_model} "
            f"--n_heads {config.model.n_heads} "
            f"--n_layers {config.model.n_layers_reward} "
            f"--d_ff {config.model.d_ff} "
            f"--seq_len {config.model.seq_len} "
            f"--device {config.device}",
            "Train Reward Model"
        )
        results.append(("Reward Training", success))
        
        if not success:
            print(f"\n{Colors.RED}❌ Reward training failed. Cannot continue.{Colors.ENDC}")
            return
    
    # Step 5: PPO RLHF training
    if not args.skip_ppo:
        success = run_command(
            f"{PYTHON} training/ppo_rlhf.py "
            f"--policy_checkpoint checkpoints/pretrain/best_model.pt "
            f"--reward_checkpoint checkpoints/reward/best_reward_model.pt "
            f"--tokenizer_path checkpoints/pretrain/tokenizer.pkl "
            f"--data_path data/shakespeare.txt "
            f"--output_dir checkpoints/ppo_rlhf "
            f"--num_steps {config.training.rl_steps} "
            f"--batch_size {config.training.batch_size // 2} "
            f"--d_model {config.model.d_model} "
            f"--n_heads {config.model.n_heads} "
            f"--n_layers {config.model.n_layers_policy} "
            f"--value_n_layers {config.model.n_layers_reward} "
            f"--d_ff {config.model.d_ff} "
            f"--max_seq_len {config.model.seq_len} "
            f"--device {config.device}",
            "PPO RLHF Training"
        )
        results.append(("PPO RLHF", success))
    
    # Step 6: GRPO training
    if not args.skip_grpo:
        success = run_command(
            f"{PYTHON} training/grpo.py "
            f"--policy_checkpoint checkpoints/pretrain/best_model.pt "
            f"--reward_checkpoint checkpoints/reward/best_reward_model.pt "
            f"--tokenizer_path checkpoints/pretrain/tokenizer.pkl "
            f"--data_path data/shakespeare.txt "
            f"--output_dir checkpoints/grpo "
            f"--num_steps {config.training.rl_steps} "
            f"--batch_size {config.training.batch_size // 2} "
            f"--d_model {config.model.d_model} "
            f"--n_heads {config.model.n_heads} "
            f"--n_layers {config.model.n_layers_policy} "
            f"--d_ff {config.model.d_ff} "
            f"--max_seq_len {config.model.seq_len} "
            f"--device {config.device}",
            "GRPO Training"
        )
        results.append(("GRPO", success))
    
    # Print summary
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}TRAINING PIPELINE SUMMARY{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.ENDC}\n")
    
    for name, success in results:
        if success:
            status = f"{Colors.GREEN}✅ SUCCESS{Colors.ENDC}"
        else:
            status = f"{Colors.RED}❌ FAILED{Colors.ENDC}"
        print(f"{Colors.BOLD}{name:30s}{Colors.ENDC}: {status}")
    
    all_success = all(success for _, success in results)
    
    if all_success:
        print(f"\n{Colors.GREEN}{Colors.BOLD}{'🎉 ' * 10}{Colors.ENDC}")
        print(f"{Colors.GREEN}{Colors.BOLD}ALL STAGES COMPLETED SUCCESSFULLY!{Colors.ENDC}")
        print(f"{Colors.GREEN}{Colors.BOLD}{'🎉 ' * 10}{Colors.ENDC}\n")
    else:
        print(f"\n{Colors.YELLOW}⚠️  Some stages failed. Please check the logs above.{Colors.ENDC}\n")


if __name__ == '__main__':
    main()

