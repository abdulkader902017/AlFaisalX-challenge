"""
Main script for generating medical reports from chest X-ray images.
"""
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import json
from PIL import Image
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from task1_classification.data_loader import get_data_loaders
from task2_report_generation.vlm_report_generator import MedicalReportGenerator, PromptEngineer


def load_sample_images(test_loader, n_samples=10, balance_classes=True):
    """
    Load sample images from the test set.
    
    Args:
        test_loader: DataLoader for test set
        n_samples: Number of samples to load
        balance_classes: Whether to balance normal and pneumonia cases
        
    Returns:
        List of (image, label, index) tuples
    """
    samples = []
    normal_samples = []
    pneumonia_samples = []
    
    # Collect samples
    idx = 0
    for images, labels in test_loader:
        for i in range(len(images)):
            img = images[i].numpy()
            label = labels[i].item()
            
            if label == 0:
                normal_samples.append((img, label, idx))
            else:
                pneumonia_samples.append((img, label, idx))
            
            idx += 1
            
            # Stop if we have enough
            if not balance_classes and len(normal_samples) + len(pneumonia_samples) >= n_samples * 2:
                break
        
        if not balance_classes and len(normal_samples) + len(pneumonia_samples) >= n_samples * 2:
            break
    
    # Balance and select samples
    if balance_classes:
        n_per_class = n_samples // 2
        normal_selected = normal_samples[:n_per_class]
        pneumonia_selected = pneumonia_samples[:n_per_class]
        samples = normal_selected + pneumonia_selected
    else:
        all_samples = normal_samples + pneumonia_samples
        np.random.shuffle(all_samples)
        samples = all_samples[:n_samples]
    
    np.random.shuffle(samples)
    return samples


def visualize_reports(samples, reports, save_path=None):
    """
    Visualize images with their generated reports.
    
    Args:
        samples: List of (image, label, index) tuples
        reports: List of report dictionaries
        save_path: Path to save visualization
    """
    n_samples = len(samples)
    n_cols = 2
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(16, 5 * n_rows))
    
    label_names = ['Normal', 'Pneumonia']
    
    for i, ((img, label, idx), report_data) in enumerate(zip(samples, reports)):
        # Image subplot
        ax_img = plt.subplot(n_rows, n_cols * 2, i * 2 + 1)
        
        # Denormalize and display image
        img_display = img.squeeze()
        img_display = img_display * 0.5 + 0.5  # Reverse normalization
        img_display = np.clip(img_display, 0, 1)
        
        ax_img.imshow(img_display, cmap='gray')
        ax_img.set_title(f"Image {idx} - True: {label_names[label]}", fontsize=10)
        ax_img.axis('off')
        
        # Report text subplot
        ax_text = plt.subplot(n_rows, n_cols * 2, i * 2 + 2)
        ax_text.axis('off')
        
        report = report_data['report']
        # Truncate if too long
        if len(report) > 800:
            report = report[:800] + "..."
        
        ax_text.text(
            0.05, 0.95, report,
            transform=ax_text.transAxes,
            fontsize=8,
            verticalalignment='top',
            wrap=True,
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved report visualization to {save_path}")
    else:
        plt.show()
    plt.close()


def save_reports_to_file(samples, reports, output_path):
    """
    Save generated reports to a structured file.
    
    Args:
        samples: List of (image, label, index) tuples
        reports: List of report dictionaries
        output_path: Path to save reports
    """
    label_names = ['Normal', 'Pneumonia']
    
    with open(output_path, 'w') as f:
        f.write("# Medical Report Generation Results\n\n")
        f.write(f"Model: {reports[0]['model']}\n")
        f.write(f"Number of samples: {len(samples)}\n\n")
        f.write("---\n\n")
        
        for i, ((img, label, idx), report_data) in enumerate(zip(samples, reports)):
            f.write(f"## Sample {i+1} (Image Index: {idx})\n\n")
            f.write(f"**True Label:** {label_names[label]}\n\n")
            f.write(f"**Generated Report:**\n\n")
            f.write(f"```\n{report_data['report']}\n```\n\n")
            
            if 'mock_classification' in report_data:
                f.write(f"**Mock Classification:** {report_data['mock_classification']}\n\n")
            
            f.write("---\n\n")
    
    print(f"Saved reports to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate medical reports from chest X-ray images')
    
    parser.add_argument('--model', type=str, default='google/medgemma-4b-pt',
                        help='VLM model name from HuggingFace')
    parser.add_argument('--n_samples', type=int, default=10,
                        help='Number of samples to generate reports for')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for data loading')
    parser.add_argument('--output_dir', type=str, default='../reports',
                        help='Output directory for reports')
    parser.add_argument('--compare_prompts', action='store_true',
                        help='Compare different prompting strategies')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Custom prompt to use')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Generation temperature')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum generation length')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*60)
    print("MEDICAL REPORT GENERATION")
    print("="*60)
    
    # Load data
    print("\nLoading test data...")
    _, _, test_loader = get_data_loaders(batch_size=args.batch_size, num_workers=2)
    
    # Load sample images
    print(f"\nLoading {args.n_samples} sample images...")
    samples = load_sample_images(test_loader, n_samples=args.n_samples, balance_classes=True)
    print(f"Loaded {len(samples)} samples")
    
    # Initialize report generator
    print(f"\nInitializing report generator with model: {args.model}")
    generator = MedicalReportGenerator(model_name=args.model)
    
    if args.compare_prompts:
        # Compare prompting strategies on first image
        print("\n" + "="*60)
        print("COMPARING PROMPTING STRATEGIES")
        print("="*60)
        
        comparison_results = PromptEngineer.compare_prompts(
            generator,
            samples[0][0],  # First image
            save_path=os.path.join(args.output_dir, 'prompt_comparison.json')
        )
        
        # Print comparison
        print("\n" + "-"*60)
        for name, result in comparison_results.items():
            print(f"\nPrompt: '{name}'")
            print("-"*40)
            print(result['report'][:300] + "..." if len(result['report']) > 300 else result['report'])
    
    # Generate reports for all samples
    print("\n" + "="*60)
    print("GENERATING REPORTS FOR ALL SAMPLES")
    print("="*60)
    
    reports = []
    for i, (img, label, idx) in enumerate(samples):
        print(f"\nGenerating report for sample {i+1}/{len(samples)} (Image {idx}, Label: {'Normal' if label == 0 else 'Pneumonia'})...")
        
        report = generator.generate_report(
            img,
            prompt=args.prompt,
            temperature=args.temperature,
            max_length=args.max_length
        )
        reports.append(report)
        
        print(f"Report generated:")
        print(report['report'][:200] + "..." if len(report['report']) > 200 else report['report'])
    
    # Visualize results
    print("\nCreating visualizations...")
    visualize_reports(
        samples, reports,
        save_path=os.path.join(args.output_dir, 'generated_reports.png')
    )
    
    # Save reports to file
    save_reports_to_file(
        samples, reports,
        output_path=os.path.join(args.output_dir, 'generated_reports.md')
    )
    
    # Save structured data
    structured_data = []
    for (img, label, idx), report in zip(samples, reports):
        structured_data.append({
            'image_index': idx,
            'true_label': 'Normal' if label == 0 else 'Pneumonia',
            'report': report['report'],
            'model': report['model']
        })
    
    with open(os.path.join(args.output_dir, 'reports_data.json'), 'w') as f:
        json.dump(structured_data, f, indent=2)
    
    print("\n" + "="*60)
    print("REPORT GENERATION COMPLETE")
    print("="*60)
    print(f"\nOutputs saved to: {args.output_dir}")
    print(f"  - generated_reports.png: Visual comparison of images and reports")
    print(f"  - generated_reports.md: Full text of all reports")
    print(f"  - reports_data.json: Structured data for further analysis")
    if args.compare_prompts:
        print(f"  - prompt_comparison.json: Comparison of different prompts")


if __name__ == '__main__':
    main()
