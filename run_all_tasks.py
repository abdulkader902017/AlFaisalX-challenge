"""
Main script to run all tasks sequentially.
This provides a unified interface for the complete pipeline.
"""
import os
import sys
import argparse
import subprocess


def run_task1(args):
    """Run Task 1: CNN Classification."""
    print("\n" + "="*70)
    print("TASK 1: CNN CLASSIFICATION")
    print("="*70 + "\n")
    
    cmd = [
        "python", "task1_classification/run_experiment.py",
        "--model", args.model,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--output_dir", args.output_dir,
    ]
    
    if args.augment:
        cmd.append("--augment")
    if args.weighted_loss:
        cmd.append("--weighted_loss")
    
    subprocess.run(cmd)


def run_task2(args):
    """Run Task 2: Medical Report Generation."""
    print("\n" + "="*70)
    print("TASK 2: MEDICAL REPORT GENERATION")
    print("="*70 + "\n")
    
    cmd = [
        "python", "task2_report_generation/generate_reports.py",
        "--model", args.vlm_model,
        "--n_samples", str(args.n_samples),
        "--output_dir", args.report_dir,
    ]
    
    if args.compare_prompts:
        cmd.append("--compare_prompts")
    
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(
        description='Run Medical AI Challenge tasks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run Task 1 only
  python run_all_tasks.py --task 1
  
  # Run Task 2 only
  python run_all_tasks.py --task 2
  
  # Run both tasks
  python run_all_tasks.py --task all
  
  # Full pipeline with custom settings
  python run_all_tasks.py --task all --epochs 100 --n_samples 20
        """
    )
    
    parser.add_argument('--task', type=str, default='all',
                        choices=['1', '2', 'all'],
                        help='Which task to run (1, 2, or all)')
    
    # Task 1 arguments
    parser.add_argument('--model', type=str, default='resnet18',
                        choices=['cnn', 'resnet18', 'efficientnet_b0'],
                        help='CNN model architecture')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--augment', action='store_true', default=True,
                        help='Use data augmentation')
    parser.add_argument('--weighted_loss', action='store_true', default=True,
                        help='Use weighted loss')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Output directory for models')
    
    # Task 2 arguments
    parser.add_argument('--vlm_model', type=str, default='google/medgemma-4b-pt',
                        help='VLM model name')
    parser.add_argument('--n_samples', type=int, default=10,
                        help='Number of samples for report generation')
    parser.add_argument('--report_dir', type=str, default='reports',
                        help='Output directory for reports')
    parser.add_argument('--compare_prompts', action='store_true',
                        help='Compare prompting strategies')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.report_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("MEDICAL AI CHALLENGE - POSTDOCTORAL TECHNICAL CHALLENGE")
    print("Alfaisal University, MedX Research Unit")
    print("="*70)
    
    if args.task == '1':
        run_task1(args)
    elif args.task == '2':
        run_task2(args)
    elif args.task == 'all':
        run_task1(args)
        run_task2(args)
    
    print("\n" + "="*70)
    print("ALL TASKS COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nOutputs saved to:")
    print(f"  - Models: {args.output_dir}/")
    print(f"  - Reports: {args.report_dir}/")


if __name__ == '__main__':
    main()
