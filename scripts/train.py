#!/usr/bin/env python3
"""Training script for store item detection."""

import argparse
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from store_detection.config import Config
from store_detection.data import StoreItemDataset, get_augmentation_pipeline
from store_detection.models import StoreItemDetector
from store_detection.models.trainer import Trainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train store item detection model')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/raw',
        help='Path to data directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs',
        help='Path to output directory'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of epochs (overrides config)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size (overrides config)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for training'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    config = Config(args.config)
    
    # Override config with command line args
    if args.epochs is not None:
        config.update('training.epochs', args.epochs)
    if args.batch_size is not None:
        config.update('training.batch_size', args.batch_size)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config.save(str(output_dir / 'config.yaml'))
    
    # Create datasets
    print("Loading datasets...")
    train_transform = get_augmentation_pipeline(
        image_size=config.get('model.input_size', 640),
        train=True
    )
    val_transform = get_augmentation_pipeline(
        image_size=config.get('model.input_size', 640),
        train=False
    )
    
    train_dataset = StoreItemDataset(
        data_dir=f"{args.data_dir}/train",
        annotation_file=f"{args.data_dir}/annotations/train.json",
        transform=train_transform
    )
    
    val_dataset = StoreItemDataset(
        data_dir=f"{args.data_dir}/val",
        annotation_file=f"{args.data_dir}/annotations/val.json",
        transform=val_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('training.batch_size', 16),
        shuffle=True,
        num_workers=config.get('data.num_workers', 4),
        collate_fn=lambda x: x  # Custom collate function
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('training.batch_size', 16),
        shuffle=False,
        num_workers=config.get('data.num_workers', 4),
        collate_fn=lambda x: x
    )
    
    # Create model
    print("Creating model...")
    model = StoreItemDetector(
        num_classes=config.get('model.num_classes', 10),
        model_name=config.get('model.name', 'yolov8'),
        pretrained=config.get('model.pretrained', True),
        input_size=config.get('model.input_size', 640)
    )
    
    # Resume from checkpoint if provided
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        model.load_pretrained(args.resume)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config.config['training'],
        device=args.device
    )
    
    # Train
    print("Starting training...")
    trainer.train(
        num_epochs=config.get('training.epochs', 100),
        save_dir=str(output_dir / 'checkpoints')
    )
    
    print("Training complete!")


if __name__ == '__main__':
    main()
