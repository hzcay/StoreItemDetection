# Store Item Detection - Project Overview

## 🎯 Purpose

This project provides a complete framework for building, training, and deploying object detection models specifically designed for recognizing items in retail store environments.

## 📊 Key Features

### 1. Modular Architecture
- **Separation of Concerns**: Data, models, and utilities are separated into distinct modules
- **Easy to Extend**: Add new models, datasets, or utilities without touching existing code
- **Configuration-Driven**: All settings managed through YAML files

### 2. Complete Pipeline
- **Data Loading**: COCO format support with custom dataset class
- **Augmentation**: Advanced augmentation pipeline using Albumentations
- **Training**: Full training loop with validation and checkpointing
- **Inference**: Easy-to-use inference scripts for single images or batches
- **Evaluation**: mAP calculation and other detection metrics
- **Export**: ONNX export for deployment

### 3. Development Tools
- **Jupyter Notebooks**: Interactive exploration and experimentation
- **Unit Tests**: Ensure code quality and correctness
- **Scripts**: Command-line tools for common tasks
- **Documentation**: Comprehensive docs and examples

## 🏗️ Architecture

### Core Components

```
StoreItemDetection/
├── Data Pipeline (src/store_detection/data/)
│   ├── Dataset loading (COCO format)
│   └── Augmentation pipeline
│
├── Model Framework (src/store_detection/models/)
│   ├── Detector architecture
│   └── Training utilities
│
├── Utilities (src/store_detection/utils/)
│   ├── Visualization tools
│   └── Evaluation metrics
│
└── Configuration (src/store_detection/config.py)
    └── Centralized config management
```

### Workflow

1. **Prepare Data**: Organize images and create COCO-format annotations
2. **Configure**: Set model architecture, hyperparameters in `config.yaml`
3. **Train**: Run training script with your configuration
4. **Evaluate**: Assess model performance on validation/test sets
5. **Deploy**: Export to ONNX and integrate into applications

## 🔧 Technology Stack

- **Deep Learning**: PyTorch
- **Computer Vision**: OpenCV
- **Augmentation**: Albumentations
- **Visualization**: Matplotlib, Seaborn
- **Configuration**: PyYAML
- **Testing**: pytest

## 📁 Directory Structure Explained

### `src/store_detection/`
Main package with all core functionality:
- `config.py`: Configuration management
- `data/`: Dataset and augmentation
- `models/`: Model definitions and training
- `utils/`: Visualization and metrics

### `configs/`
YAML configuration files for different experiments

### `scripts/`
Command-line tools:
- `train.py`: Train models
- `inference.py`: Run predictions
- `evaluate.py`: Calculate metrics
- `export_model.py`: Export to ONNX

### `notebooks/`
Jupyter notebooks for:
- Data exploration
- Training demos
- Result visualization

### `tests/`
Unit tests for core functionality

### `data/`
Data storage:
- `raw/`: Original images
- `processed/`: Processed data
- `annotations/`: COCO format annotations

### `models/`
Model storage:
- `checkpoints/`: Training checkpoints
- `pretrained/`: Pretrained weights

## 🚀 Getting Started

See `QUICKSTART.md` for installation and usage instructions.

## 📈 Typical Use Cases

### 1. Retail Inventory Management
Detect and count items on store shelves

### 2. Automated Checkout
Recognize items at point of sale

### 3. Planogram Compliance
Verify shelf arrangement matches planogram

### 4. Stock Monitoring
Track product availability and placement

## 🔄 Extension Points

### Adding a New Model
1. Create new detector class in `src/store_detection/models/`
2. Update configuration to support new model type
3. Modify trainer if needed

### Custom Data Format
1. Create new dataset class inheriting from base
2. Implement `__getitem__` for your format
3. Update data loader configuration

### Additional Metrics
1. Add metric functions to `src/store_detection/utils/metrics.py`
2. Update evaluation script to use new metrics

## 📚 Resources

- **PyTorch Documentation**: https://pytorch.org/docs/
- **COCO Format**: https://cocodataset.org/#format-data
- **Albumentations**: https://albumentations.ai/docs/

## 🤝 Contributing

See `CONTRIBUTING.md` for guidelines on contributing to this project.

## 📄 License

MIT License - See `LICENSE` file for details.
