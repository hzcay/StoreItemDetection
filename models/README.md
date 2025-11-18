# Models Directory

This directory stores model-related files.

## Structure

```
models/
├── checkpoints/     # Training checkpoints
│   └── best_model.pth
└── pretrained/      # Pretrained model weights
```

## Checkpoint Format

Model checkpoints contain:
- `epoch`: Training epoch number
- `model_state_dict`: Model weights
- `optimizer_state_dict`: Optimizer state
- `model_config`: Model configuration
- `metrics`: Training metrics

## Loading a Checkpoint

```python
from store_detection.models import StoreItemDetector

# Create model
model = StoreItemDetector(num_classes=10)

# Load checkpoint
model.load_pretrained('models/checkpoints/best_model.pth')
```
