# Store Item Detection

A deep learning-based system for detecting and recognizing items in retail store environments. This project provides a complete framework for training, evaluating, and deploying object detection models specifically designed for store item recognition.

## 🎯 Features

- **Modular Architecture**: Clean, maintainable code structure with separate modules for data, models, and utilities
- **Configurable Pipeline**: YAML-based configuration for easy experimentation
- **Data Augmentation**: Comprehensive augmentation pipeline using Albumentations
- **Multiple Model Support**: Framework supports various detection architectures (YOLO, Faster R-CNN, etc.)
- **Training & Inference Scripts**: Ready-to-use scripts for training and running inference
- **Visualization Tools**: Built-in tools for visualizing predictions and training metrics
- **Jupyter Notebooks**: Interactive notebooks for exploration and experimentation

## 📁 Project Structure

```
StoreItemDetection/
├── configs/                    # Configuration files
│   └── config.yaml            # Main configuration file
├── data/                      # Data directory
│   ├── raw/                   # Raw images
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── processed/             # Processed data
│   └── annotations/           # Annotation files (COCO format)
├── models/                    # Model storage
│   ├── checkpoints/           # Training checkpoints
│   └── pretrained/            # Pretrained models
├── notebooks/                 # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   └── 02_training_demo.ipynb
├── scripts/                   # Utility scripts
│   ├── train.py              # Training script
│   └── inference.py          # Inference script
├── src/                       # Source code
│   └── store_detection/
│       ├── __init__.py
│       ├── config.py         # Configuration management
│       ├── data/             # Data loading and processing
│       │   ├── __init__.py
│       │   ├── dataset.py
│       │   └── augmentation.py
│       ├── models/           # Model definitions
│       │   ├── __init__.py
│       │   ├── detector.py
│       │   └── trainer.py
│       └── utils/            # Utility functions
│           ├── __init__.py
│           ├── visualization.py
│           └── metrics.py
├── tests/                     # Unit tests
│   ├── __init__.py
│   ├── test_config.py
│   └── test_utils.py
├── logs/                      # Training logs
├── outputs/                   # Output directory
├── .gitignore
├── requirements.txt           # Python dependencies
├── setup.py                   # Package installation
└── README.md                  # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional but recommended for training)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/hzcay/StoreItemDetection.git
cd StoreItemDetection
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install the package in development mode:
```bash
pip install -e .
```

## 📊 Data Preparation

1. Organize your data in the following structure:
```
data/
├── raw/
│   ├── train/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── val/
│   └── test/
└── annotations/
    ├── train.json  # COCO format
    ├── val.json
    └── test.json
```

2. Annotations should be in COCO format with the following structure:
```json
{
  "images": [...],
  "annotations": [...],
  "categories": [...]
}
```

## 🏋️ Training

### Using the Training Script

```bash
python scripts/train.py \
    --config configs/config.yaml \
    --data-dir data/raw \
    --output-dir outputs/experiment1 \
    --epochs 100 \
    --batch-size 16
```

### Using Jupyter Notebook

Open `notebooks/02_training_demo.ipynb` for an interactive training experience.

### Configuration

Edit `configs/config.yaml` to customize:
- Model architecture and parameters
- Training hyperparameters
- Data augmentation settings
- Paths and directories

## 🔍 Inference

### Run Inference on Single Image

```bash
python scripts/inference.py \
    --checkpoint outputs/experiment1/checkpoints/best_model.pth \
    --input path/to/image.jpg \
    --output outputs/predictions \
    --confidence 0.5
```

### Run Inference on Directory

```bash
python scripts/inference.py \
    --checkpoint outputs/experiment1/checkpoints/best_model.pth \
    --input path/to/image/directory \
    --output outputs/predictions \
    --confidence 0.5
```

## 🧪 Testing

Run unit tests:
```bash
pytest tests/
```

Run specific test file:
```bash
pytest tests/test_config.py
```

## 📈 Model Evaluation

The framework includes metrics for object detection:
- **mAP (mean Average Precision)**: Primary metric for detection performance
- **IoU (Intersection over Union)**: For measuring bounding box overlap
- **Per-class AP**: Detailed performance per item category

## 🛠️ Development

### Code Style

This project follows PEP 8 guidelines. Format code using:
```bash
black src/
isort src/
```

Lint code using:
```bash
flake8 src/
```

## 📝 Usage Examples

### Load Configuration
```python
from store_detection.config import Config

config = Config('configs/config.yaml')
model_name = config.get('model.name')
```

### Create Dataset
```python
from store_detection.data import StoreItemDataset

dataset = StoreItemDataset(
    data_dir='data/raw/train',
    annotation_file='data/annotations/train.json'
)
```

### Initialize Model
```python
from store_detection.models import StoreItemDetector

model = StoreItemDetector(
    num_classes=10,
    model_name='yolov8',
    pretrained=True
)
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with PyTorch
- Uses Albumentations for data augmentation
- Supports COCO format annotations

## 📞 Contact

For questions or issues, please open an issue on GitHub.

---

**Note**: This project is designed for educational and research purposes in retail item detection and recognition.