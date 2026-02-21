"""
Evaluation script for pneumonia classification model.
"""
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, classification_report
)
from tqdm import tqdm
import json

from data_loader import get_data_loaders
from model import get_model


def evaluate_model(model, data_loader, device, save_dir=None):
    """
    Evaluate model and compute all metrics.
    
    Returns:
        Dictionary containing all evaluation results
    """
    model.eval()
    
    all_labels = []
    all_predictions = []
    all_probabilities = []
    all_images = []
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc='Evaluating'):
            images = images.to(device)
            labels = labels.cpu().numpy()
            
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            
            all_labels.extend(labels)
            all_predictions.extend(predictions)
            all_probabilities.extend(probabilities.cpu().numpy())
            all_images.extend(images.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    all_images = np.array(all_images)
    
    # Compute metrics
    results = compute_metrics(all_labels, all_predictions, all_probabilities)
    
    # Find failure cases
    failure_cases = find_failure_cases(
        all_labels, all_predictions, all_probabilities, all_images
    )
    
    # Generate visualizations
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        # Confusion matrix
        plot_confusion_matrix(
            all_labels, all_predictions, 
            save_path=os.path.join(save_dir, 'confusion_matrix.png')
        )
        
        # ROC curve
        plot_roc_curve(
            all_labels, all_probabilities[:, 1],
            save_path=os.path.join(save_dir, 'roc_curve.png')
        )
        
        # Failure cases
        plot_failure_cases(
            failure_cases,
            save_path=os.path.join(save_dir, 'failure_cases.png')
        )
        
        # Sample predictions
        plot_sample_predictions(
            all_images, all_labels, all_predictions, all_probabilities,
            save_path=os.path.join(save_dir, 'sample_predictions.png')
        )
        
        # Save results
        with open(os.path.join(save_dir, 'evaluation_results.json'), 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            json_results = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in results.items()
            }
            json.dump(json_results, f, indent=2)
    
    results['failure_cases'] = failure_cases
    
    return results


def compute_metrics(labels, predictions, probabilities):
    """Compute classification metrics."""
    
    # Basic metrics
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='binary')
    recall = recall_score(labels, predictions, average='binary')
    f1 = f1_score(labels, predictions, average='binary')
    
    # AUC-ROC (use probability of positive class)
    try:
        auc = roc_auc_score(labels, probabilities[:, 1])
    except:
        auc = 0.5  # Default if computation fails
    
    # Per-class metrics
    report = classification_report(labels, predictions, output_dict=True)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc,
        'classification_report': report,
        'confusion_matrix': confusion_matrix(labels, predictions).tolist()
    }


def find_failure_cases(labels, predictions, probabilities, images, max_cases=20):
    """
    Find and analyze failure cases.
    
    Returns:
        List of failure case dictionaries
    """
    failure_indices = np.where(labels != predictions)[0]
    
    # Sort by confidence (highest confidence wrong predictions first)
    wrong_confidences = np.max(probabilities[failure_indices], axis=1)
    sorted_indices = failure_indices[np.argsort(wrong_confidences)[::-1]]
    
    failure_cases = []
    for idx in sorted_indices[:max_cases]:
        failure_cases.append({
            'index': int(idx),
            'image': images[idx],
            'true_label': int(labels[idx]),
            'predicted_label': int(predictions[idx]),
            'confidence': float(np.max(probabilities[idx])),
            'probabilities': probabilities[idx].tolist()
        })
    
    return failure_cases


def plot_confusion_matrix(labels, predictions, save_path=None):
    """Plot confusion matrix."""
    cm = confusion_matrix(labels, predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Pneumonia'],
                yticklabels=['Normal', 'Pneumonia'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_roc_curve(labels, probabilities, save_path=None):
    """Plot ROC curve."""
    fpr, tpr, thresholds = roc_curve(labels, probabilities)
    auc = roc_auc_score(labels, probabilities)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved ROC curve to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_failure_cases(failure_cases, save_path=None, max_cases=12):
    """Visualize failure cases."""
    if len(failure_cases) == 0:
        print("No failure cases to visualize")
        return
    
    n_cases = min(len(failure_cases), max_cases)
    n_cols = 4
    n_rows = (n_cases + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    label_names = ['Normal', 'Pneumonia']
    
    for i in range(n_cases):
        case = failure_cases[i]
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        # Denormalize image
        img = case['image'].squeeze()
        img = img * 0.5 + 0.5  # Reverse normalization
        img = np.clip(img, 0, 1)
        
        ax.imshow(img, cmap='gray')
        ax.set_title(
            f"True: {label_names[case['true_label']]}\n"
            f"Pred: {label_names[case['predicted_label']]}\n"
            f"Conf: {case['confidence']:.2%}",
            fontsize=9
        )
        ax.axis('off')
        
        # Color title based on error type
        if case['true_label'] == 1 and case['predicted_label'] == 0:
            ax.title.set_color('red')  # False negative (missed pneumonia)
        else:
            ax.title.set_color('orange')  # False positive
    
    # Hide empty subplots
    for i in range(n_cases, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')
    
    plt.suptitle('Failure Cases Analysis\n(Red = Missed Pneumonia, Orange = False Alarm)', 
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved failure cases to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_sample_predictions(images, labels, predictions, probabilities, 
                            save_path=None, n_samples=16):
    """Visualize sample predictions with confidence."""
    n_cols = 4
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    # Randomly select samples
    indices = np.random.choice(len(images), min(n_samples, len(images)), replace=False)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    label_names = ['Normal', 'Pneumonia']
    
    for i, idx in enumerate(indices):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        # Denormalize image
        img = images[idx].squeeze()
        img = img * 0.5 + 0.5
        img = np.clip(img, 0, 1)
        
        pred_label = predictions[idx]
        true_label = labels[idx]
        confidence = np.max(probabilities[idx])
        
        ax.imshow(img, cmap='gray')
        
        # Color based on correctness
        color = 'green' if pred_label == true_label else 'red'
        
        ax.set_title(
            f"True: {label_names[true_label]} | Pred: {label_names[pred_label]}\n"
            f"Confidence: {confidence:.2%}",
            fontsize=9, color=color
        )
        ax.axis('off')
    
    # Hide empty subplots
    for i in range(len(indices), n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')
    
    plt.suptitle('Sample Predictions (Green=Correct, Red=Incorrect)', 
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved sample predictions to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_training_history(history, save_path=None):
    """Plot training history curves."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss curves
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy curves
    axes[0, 1].plot(history['train_acc'], label='Train Acc')
    axes[0, 1].plot(history['val_acc'], label='Val Acc')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate
    axes[1, 0].plot(history['lr'])
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Validation metrics summary
    final_val_acc = history['val_acc'][-1]
    best_val_acc = max(history['val_acc'])
    best_epoch = history['val_acc'].index(best_val_acc) + 1
    
    axes[1, 1].axis('off')
    summary_text = f"""
    Training Summary:
    
    Final Val Accuracy: {final_val_acc:.2f}%
    Best Val Accuracy: {best_val_acc:.2f}%
    Best Epoch: {best_epoch}
    
    Total Epochs: {len(history['train_loss'])}
    """
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, 
                    verticalalignment='center', family='monospace')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training history to {save_path}")
    else:
        plt.show()
    plt.close()
