# Quick Start Guide

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/hzcay/StoreItemDetection.git
cd StoreItemDetection
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Install the package:**
```bash
pip install -e .
```

## Verify Installation

```python
from store_detection.config import Config

config = Config('configs/config.yaml')
print(f"Model: {config.get('model.name')}")
print(f"Input size: {config.get('model.input_size')}")
```

## Quick Examples

### 1. Training a Model

```bash
python scripts/train.py \
    --config configs/config.yaml \
    --data-dir data/raw \
    --output-dir outputs/my_experiment \
    --epochs 50
```

### 2. Running Inference

```bash
python scripts/inference.py \
    --checkpoint outputs/my_experiment/checkpoints/best_model.pth \
    --input path/to/test/image.jpg \
    --output outputs/predictions
```

### 3. Evaluating the Model

```bash
python scripts/evaluate.py \
    --checkpoint outputs/my_experiment/checkpoints/best_model.pth \
    --data-dir data/raw/test \
    --annotation-file data/annotations/test.json
```

### 4. Exporting to ONNX

```bash
python scripts/export_model.py \
    --checkpoint outputs/my_experiment/checkpoints/best_model.pth \
    --output outputs/model.onnx
```

## Using Jupyter Notebooks

```bash
jupyter notebook notebooks/
```

Then open:
- `01_data_exploration.ipynb` - For data exploration
- `02_training_demo.ipynb` - For interactive training

## Running Tests

```bash
pytest tests/
```

## Next Steps

1. **Prepare your data**: Organize images and annotations in COCO format
2. **Customize config**: Edit `configs/config.yaml` for your use case
3. **Train**: Run the training script with your data
4. **Evaluate**: Test model performance on validation set
5. **Deploy**: Export model and integrate into your application

## Common Issues

### Import Errors
Make sure to install the package: `pip install -e .`

### CUDA Not Available
The project works on CPU too, but training will be slower. Use `--device cpu` flag.

### Data Format
Ensure annotations follow COCO format. See `data/annotations/sample_annotations.json` for reference.

## Documentation

- See `README.md` for full documentation
- Check `CONTRIBUTING.md` for contribution guidelines
- Review module docstrings for API details
