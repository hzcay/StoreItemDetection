#!/usr/bin/env python3
"""Inference script for store item detection."""

import argparse
import torch
import cv2
import numpy as np
from pathlib import Path

from store_detection.config import Config
from store_detection.models import StoreItemDetector
from store_detection.utils.visualization import visualize_predictions


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run inference on store item detection model')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input image or directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/predictions',
        help='Path to output directory'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.5,
        help='Confidence threshold'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for inference'
    )
    
    return parser.parse_args()


def load_model(checkpoint_path: str, config: Config, device: str):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = StoreItemDetector(
        num_classes=checkpoint['model_config']['num_classes'],
        model_name=checkpoint['model_config']['model_name'],
        input_size=checkpoint['model_config']['input_size']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


def preprocess_image(image_path: str, input_size: int):
    """Preprocess image for inference."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize
    original_shape = image.shape[:2]
    image_resized = cv2.resize(image, (input_size, input_size))
    
    # Normalize
    image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float()
    image_tensor = image_tensor / 255.0
    image_tensor = image_tensor.unsqueeze(0)
    
    return image, image_tensor, original_shape


def run_inference(
    model,
    image_path: str,
    output_dir: Path,
    config: Config,
    confidence_threshold: float,
    device: str
):
    """Run inference on single image."""
    # Load and preprocess image
    original_image, image_tensor, original_shape = preprocess_image(
        image_path,
        config.get('model.input_size', 640)
    )
    
    # Run inference
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        predictions = model(image_tensor)
    
    # Extract predictions
    boxes = predictions['boxes'][0].cpu().numpy()
    scores = predictions['scores'][0].cpu().numpy()
    labels = predictions['labels'][0].cpu().numpy()
    
    # Get class names
    class_names = [f"item_{i}" for i in range(config.get('model.num_classes', 10))]
    
    # Visualize
    output_path = output_dir / f"pred_{Path(image_path).name}"
    vis_image = visualize_predictions(
        original_image,
        boxes,
        labels,
        scores,
        class_names,
        score_threshold=confidence_threshold,
        save_path=str(output_path)
    )
    
    print(f"Saved prediction to: {output_path}")
    return boxes, scores, labels


def main():
    """Main inference function."""
    args = parse_args()
    
    # Load configuration
    config = Config(args.config)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model = load_model(args.checkpoint, config, args.device)
    
    # Process input
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single image
        print(f"Processing image: {input_path}")
        run_inference(
            model,
            str(input_path),
            output_dir,
            config,
            args.confidence,
            args.device
        )
    elif input_path.is_dir():
        # Directory of images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = [
            f for f in input_path.iterdir()
            if f.suffix.lower() in image_extensions
        ]
        
        print(f"Processing {len(image_files)} images...")
        for image_file in image_files:
            print(f"Processing: {image_file.name}")
            run_inference(
                model,
                str(image_file),
                output_dir,
                config,
                args.confidence,
                args.device
            )
    else:
        raise ValueError(f"Invalid input path: {input_path}")
    
    print("Inference complete!")


if __name__ == '__main__':
    main()
