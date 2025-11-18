#!/usr/bin/env python3
"""Script to export model to ONNX format."""

import argparse
import torch
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from store_detection.models import StoreItemDetector


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Export model to ONNX')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/model.onnx',
        help='Path to save ONNX model'
    )
    parser.add_argument(
        '--opset-version',
        type=int,
        default=11,
        help='ONNX opset version'
    )
    
    return parser.parse_args()


def export_to_onnx(model, output_path, input_size, opset_version):
    """Export model to ONNX format."""
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, input_size, input_size)
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['boxes', 'scores', 'labels'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'boxes': {0: 'batch_size'},
            'scores': {0: 'batch_size'},
            'labels': {0: 'batch_size'}
        }
    )


def main():
    """Main export function."""
    args = parse_args()
    
    # Load checkpoint
    print(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # Create model
    model = StoreItemDetector(
        num_classes=checkpoint['model_config']['num_classes'],
        model_name=checkpoint['model_config']['model_name'],
        input_size=checkpoint['model_config']['input_size']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Export
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Exporting to ONNX: {output_path}")
    export_to_onnx(
        model,
        str(output_path),
        checkpoint['model_config']['input_size'],
        args.opset_version
    )
    
    print("Export complete!")
    print(f"Model saved to: {output_path}")


if __name__ == '__main__':
    main()
