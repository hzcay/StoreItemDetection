"""Training utilities for store item detection."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional
from pathlib import Path
from tqdm import tqdm


class Trainer:
    """Trainer class for object detection."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: str = 'cuda'
    ):
        """
        Initialize trainer.
        
        Args:
            model: Detection model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            device: Device to train on
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Initialize optimizer
        self.optimizer = self._get_optimizer()
        
        # Initialize learning rate scheduler
        self.scheduler = self._get_scheduler()
        
        # Training state
        self.current_epoch = 0
        self.best_metric = 0.0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_map': []
        }
    
    def _get_optimizer(self) -> optim.Optimizer:
        """Get optimizer."""
        optimizer_name = self.config.get('optimizer', 'adam').lower()
        lr = self.config.get('learning_rate', 0.001)
        
        if optimizer_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer_name == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=0.0005
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _get_scheduler(self):
        """Get learning rate scheduler."""
        return optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.1,
            patience=5,
            verbose=True
        )
    
    def train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            loss_dict = self.model(images, targets)
            
            # Calculate total loss
            losses = sum(loss for loss in loss_dict.values())
            
            # Backward pass
            losses.backward()
            self.optimizer.step()
            
            total_loss += losses.item()
            pbar.set_postfix({'loss': losses.item()})
        
        return total_loss / len(self.train_loader)
    
    def validate(self) -> Dict[str, float]:
        """
        Validate model.
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc='Validation'):
                images = images.to(self.device)
                
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                total_loss += losses.item()
        
        avg_loss = total_loss / len(self.val_loader)
        
        return {
            'val_loss': avg_loss,
            'val_map': 0.0  # Placeholder - would calculate mAP
        }
    
    def train(self, num_epochs: int, save_dir: str):
        """
        Train model.
        
        Args:
            num_epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            self.history['train_loss'].append(train_loss)
            
            # Validate
            val_metrics = self.validate()
            self.history['val_loss'].append(val_metrics['val_loss'])
            self.history['val_map'].append(val_metrics['val_map'])
            
            # Update scheduler
            self.scheduler.step(val_metrics['val_map'])
            
            # Save checkpoint
            if val_metrics['val_map'] > self.best_metric:
                self.best_metric = val_metrics['val_map']
                checkpoint_path = save_path / 'best_model.pth'
                self.model.save_checkpoint(
                    str(checkpoint_path),
                    epoch,
                    self.optimizer.state_dict(),
                    val_metrics
                )
            
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, "
                  f"val_loss={val_metrics['val_loss']:.4f}, "
                  f"val_map={val_metrics['val_map']:.4f}")
