# File: src/utils/visualization.py
# Description: Visualization utilities for model evaluation and results analysis

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
import torch
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm

def plot_confusion_matrix(y_true, y_pred, classes=['IDC Negative', 'IDC Positive']):
    """
    Plot confusion matrix with seaborn
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes,
                yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def plot_roc_curve(y_true, y_pred_proba):
    """
    Plot ROC curve
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('roc_curve.png')
    plt.close()

def plot_precision_recall_curve(y_true, y_pred_proba):
    """
    Plot Precision-Recall curve
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    average_precision = average_precision_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'PR curve (AP = {average_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig('precision_recall_curve.png')
    plt.close()

def plot_training_history(history):
    """
    Plot training and validation metrics
    """
    metrics = ['loss', 'accuracy', 'precision', 'recall', 'f1']
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        plt.plot(history[f'train_{metric}'], label=f'Training {metric}')
        plt.plot(history[f'val_{metric}'], label=f'Validation {metric}')
        plt.title(f'Training and Validation {metric.capitalize()}')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{metric}_history.png')
        plt.close()

def plot_class_distribution(y_true, y_pred):
    """
    Plot class distribution comparison
    """
    plt.figure(figsize=(10, 6))
    x = np.arange(2)
    width = 0.35
    
    plt.bar(x - width/2, np.bincount(y_true), width, label='True')
    plt.bar(x + width/2, np.bincount(y_pred), width, label='Predicted')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Class Distribution: True vs Predicted')
    plt.xticks(x, ['IDC Negative', 'IDC Positive'])
    plt.legend()
    plt.tight_layout()
    plt.savefig('class_distribution.png')
    plt.close()

def evaluate_model(model, test_loader, device):
    """
    Evaluate model and generate all visualizations
    """
    model.eval()
    all_preds = []
    all_preds_proba = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating'):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            preds_proba = outputs.cpu().numpy()
            preds = (preds_proba > 0.5).astype(int)
            
            all_preds.extend(preds)
            all_preds_proba.extend(preds_proba)
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_preds_proba = np.array(all_preds_proba)
    all_labels = np.array(all_labels)
    
    # Generate classification report
    report = classification_report(all_labels, all_preds, 
                                 target_names=['IDC Negative', 'IDC Positive'],
                                 output_dict=True)
    
    # Convert to DataFrame for better visualization
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv('classification_report.csv')
    
    # Generate all plots
    plot_confusion_matrix(all_labels, all_preds)
    plot_roc_curve(all_labels, all_preds_proba)
    plot_precision_recall_curve(all_labels, all_preds_proba)
    plot_class_distribution(all_labels, all_preds)
    
    return report_df

def visualize_sample_predictions(model, test_loader, device, num_samples=5):
    """
    Visualize sample predictions with their true labels
    """
    model.eval()
    images, labels = next(iter(test_loader))
    images = images[:num_samples].to(device)
    labels = labels[:num_samples]
    
    with torch.no_grad():
        outputs = model(images)
        preds = (outputs > 0.5).cpu().numpy()
    
    plt.figure(figsize=(15, 3))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i+1)
        plt.imshow(images[i].cpu().permute(1, 2, 0))
        plt.title(f'True: {labels[i]}\nPred: {preds[i][0]}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_predictions.png')
    plt.close()

def save_evaluation_results(report_df, history, save_dir='evaluation_results'):
    """
    Save all evaluation results to a directory
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Save classification report
    report_df.to_csv(os.path.join(save_dir, 'classification_report.csv'))
    
    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(save_dir, 'training_history.csv'))
    
    # Move all plots to the directory
    import shutil
    for file in ['confusion_matrix.png', 'roc_curve.png', 
                 'precision_recall_curve.png', 'class_distribution.png',
                 'sample_predictions.png']:
        if os.path.exists(file):
            shutil.move(file, os.path.join(save_dir, file))
    
    print(f"All evaluation results saved to {save_dir}/") 