"""
Main script to run the complete training and evaluation pipeline.
"""
import os
import sys
import torch
import argparse
import json

from train import train_model
from evaluate import evaluate_model, plot_training_history
from data_loader import get_data_loaders
from model import get_model


def main():
    parser = argparse.ArgumentParser(description='Run complete pneumonia classification experiment')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='resnet18',
                        choices=['cnn', 'resnet18', 'efficientnet_b0'],
                        help='Model architecture')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained weights')
    parser.add_argument('--no_pretrained', action='store_true',
                        help='Do not use pretrained weights')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate')
    
    # Data arguments
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--augment', action='store_true', default=True,
                        help='Use data augmentation')
    parser.add_argument('--no_augment', action='store_true',
                        help='Disable data augmentation')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'adamw', 'sgd'],
                        help='Optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--scheduler', type=str, default='plateau',
                        choices=['plateau', 'cosine', 'none'],
                        help='Learning rate scheduler')
    parser.add_argument('--weighted_loss', action='store_true', default=True,
                        help='Use weighted loss for class imbalance')
    parser.add_argument('--no_weighted_loss', action='store_true',
                        help='Disable weighted loss')
    
    # Other arguments
    parser.add_argument('--output_dir', type=str, default='../models',
                        help='Output directory for models')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--early_stopping', type=int, default=10,
                        help='Early stopping patience (0 to disable)')
    
    # Evaluation only
    parser.add_argument('--eval_only', action='store_true',
                        help='Only run evaluation (requires --checkpoint)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint for evaluation')
    
    args = parser.parse_args()
    
    # Handle negated flags
    if args.no_pretrained:
        args.pretrained = False
    if args.no_augment:
        args.augment = False
    if args.no_weighted_loss:
        args.weighted_loss = False
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if args.eval_only:
        # Evaluation only mode
        if not args.checkpoint:
            print("Error: --checkpoint required for evaluation mode")
            sys.exit(1)
        
        print(f"Loading model from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        # Get model
        model = get_model(
            model_name=args.model,
            num_classes=2,
            pretrained=False,
            dropout_rate=args.dropout
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        # Get data loaders
        _, _, test_loader = get_data_loaders(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            augment=False
        )
        
        # Evaluate
        results = evaluate_model(model, test_loader, device, save_dir=args.output_dir)
        
        # Print results
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Accuracy:  {results['accuracy']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall:    {results['recall']:.4f}")
        print(f"F1-Score:  {results['f1_score']:.4f}")
        print(f"AUC:       {results['auc']:.4f}")
        print("="*50)
        
    else:
        # Training mode
        print("Starting training experiment...")
        print(f"Model: {args.model}")
        print(f"Pretrained: {args.pretrained}")
        print(f"Epochs: {args.epochs}")
        print(f"Batch size: {args.batch_size}")
        print(f"Learning rate: {args.lr}")
        print(f"Optimizer: {args.optimizer}")
        print(f"Scheduler: {args.scheduler}")
        print(f"Data augmentation: {args.augment}")
        print(f"Weighted loss: {args.weighted_loss}")
        
        # Train model
        model, history, results = train_model(args)
        
        # Plot training history
        plot_training_history(
            history,
            save_path=os.path.join(args.output_dir, 'training_curves.png')
        )
        
        print("\n" + "="*50)
        print("FINAL TEST RESULTS")
        print("="*50)
        print(f"Accuracy:  {results['accuracy']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall:    {results['recall']:.4f}")
        print(f"F1-Score:  {results['f1_score']:.4f}")
        print(f"AUC:       {results['auc']:.4f}")
        print("="*50)
        
        print(f"\nAll outputs saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
