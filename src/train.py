import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from datetime import datetime

from models.model import get_model
from data.dataloader import get_data_loaders, get_class_weights

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directory
        self.output_dir = os.path.join('outputs', datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize model
        self.model = get_model(
            model_name=config['model_name'],
            pretrained=True,
            num_classes=2,
            device=self.device
        )
        
        # Get data loaders
        self.train_loader, self.val_loader, self.test_loader = get_data_loaders(
            data_dir=config['data_dir'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers']
        )
        
        # Initialize loss function and optimizer
        self.criterion = nn.CrossEntropyLoss(weight=get_class_weights(config['data_dir']).to(self.device))
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.1,
            patience=3,
            min_lr=1e-6
        )
        
        # Initialize metrics
        self.best_val_auc = 0
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc='Training')
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        # Calculate metrics
        epoch_loss = total_loss / len(self.train_loader)
        epoch_metrics = self.calculate_metrics(all_labels, all_preds)
        
        return epoch_loss, epoch_metrics
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc='Validation'):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Update metrics
                total_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
        
        # Calculate metrics
        epoch_loss = total_loss / len(self.val_loader)
        epoch_metrics = self.calculate_metrics(all_labels, all_preds, all_probs)
        
        return epoch_loss, epoch_metrics
    
    def calculate_metrics(self, labels, preds, probs=None):
        metrics = {
            'accuracy': accuracy_score(labels, preds),
            'precision': precision_score(labels, preds),
            'recall': recall_score(labels, preds),
            'f1': f1_score(labels, preds)
        }
        
        if probs is not None:
            metrics['auc'] = roc_auc_score(labels, probs)
        
        return metrics
    
    def save_checkpoint(self, epoch, val_metrics, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_metrics': val_metrics,
            'config': self.config
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, os.path.join(self.output_dir, 'latest_checkpoint.pth'))
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, os.path.join(self.output_dir, 'best_checkpoint.pth'))
    
    def plot_metrics(self):
        epochs = range(1, len(self.train_losses) + 1)
        
        # Plot losses
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, self.train_losses, label='Train Loss')
        plt.plot(epochs, self.val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, 'loss_plot.png'))
        plt.close()
        
        # Plot metrics
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        plt.figure(figsize=(10, 5))
        for metric in metrics:
            train_metric = [m[metric] for m in self.train_metrics]
            val_metric = [m[metric] for m in self.val_metrics]
            plt.plot(epochs, train_metric, label=f'Train {metric.capitalize()}')
            plt.plot(epochs, val_metric, label=f'Val {metric.capitalize()}')
        plt.title('Training and Validation Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, 'metrics_plot.png'))
        plt.close()
    
    def train(self):
        print(f"Training on {self.device}")
        print(f"Output directory: {self.output_dir}")
        
        for epoch in range(self.config['num_epochs']):
            print(f"\nEpoch {epoch + 1}/{self.config['num_epochs']}")
            
            # Train
            train_loss, train_metrics = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_metrics.append(train_metrics)
            
            # Validate
            val_loss, val_metrics = self.validate()
            self.val_losses.append(val_loss)
            self.val_metrics.append(val_metrics)
            
            # Update learning rate
            self.scheduler.step(val_metrics['auc'])
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Train Metrics: {train_metrics}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Metrics: {val_metrics}")
            
            # Save checkpoint
            is_best = val_metrics['auc'] > self.best_val_auc
            if is_best:
                self.best_val_auc = val_metrics['auc']
            self.save_checkpoint(epoch, val_metrics, is_best)
        
        # Plot metrics
        self.plot_metrics()
        
        print("\nTraining completed!")
        print(f"Best validation AUC: {self.best_val_auc:.4f}")

if __name__ == "__main__":
    # Training configuration
    config = {
        'model_name': 'efficientnet_b0',
        'data_dir': 'data/processed',
        'batch_size': 8,
        'num_workers': 2,
        'learning_rate': 1e-4,
        'num_epochs': 20
    }
    
    # Initialize and train
    trainer = Trainer(config)
    trainer.train() 