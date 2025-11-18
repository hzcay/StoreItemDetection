#!/usr/bin/env python3
"""Script to evaluate model on test set."""

import argparse
import json
import torch
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from store_detection.config import Config
from store_detection.data import StoreItemDataset
from store_detection.models import StoreItemDetector
from store_detection.utils.metrics import calculate_metrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate model on test set')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/raw/test',
        help='Path to test data directory'
    )
    parser.add_argument(
        '--annotation-file',
        type=str,
        default='data/annotations/test.json',
        help='Path to test annotations'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/evaluation_results.json',
        help='Path to save evaluation results'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use'
    )
    
    return parser.parse_args()


def evaluate(model, dataset, device, config):
    """Evaluate model on dataset."""
    model.eval()
    
    all_predictions = {}
    all_ground_truths = {}
    
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc='Evaluating'):
            image, annotations = dataset[idx]
            
            # Prepare image
            if isinstance(image, torch.Tensor):
                image = image.unsqueeze(0).to(device)
            
            # Get predictions
            predictions = model(image)
            
            # Process predictions and ground truths
            # This is a placeholder - actual implementation would depend on model output format
            pass
    
    # Calculate metrics
    num_classes = config.get('model.num_classes', 10)
    metrics = calculate_metrics(
        all_predictions,
        all_ground_truths,
        num_classes=num_classes
    )
    
    return metrics


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Load configuration
    config = Config(args.config)
    
    # Load model
    print(f"Loading model from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    
    model = StoreItemDetector(
        num_classes=checkpoint['model_config']['num_classes'],
        model_name=checkpoint['model_config']['model_name'],
        input_size=checkpoint['model_config']['input_size']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(args.device)
    
    # Load dataset
    print(f"Loading dataset from: {args.data_dir}")
    dataset = StoreItemDataset(
        data_dir=args.data_dir,
        annotation_file=args.annotation_file
    )
    
    # Evaluate
    print("Running evaluation...")
    metrics = evaluate(model, dataset, args.device, config)
    
    # Print results
    print("\nEvaluation Results:")
    print(f"mAP: {metrics['mAP']:.4f}")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
