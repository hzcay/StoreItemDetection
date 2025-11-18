# Data Directory

This directory contains all data for the store item detection project.

## Structure

```
data/
├── raw/              # Raw image data
│   ├── train/       # Training images
│   ├── val/         # Validation images
│   └── test/        # Test images
├── processed/       # Processed/augmented data
└── annotations/     # Annotation files in COCO format
    ├── train.json
    ├── val.json
    └── test.json
```

## Annotation Format

Annotations should follow the COCO format:

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "image1.jpg",
      "width": 640,
      "height": 480
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, width, height],
      "area": 1000,
      "iscrowd": 0
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "item_name",
      "supercategory": "store_item"
    }
  ]
}
```

## Adding Your Data

1. Place your images in the appropriate subdirectory (train/val/test)
2. Create or update the corresponding annotation JSON file
3. Ensure image paths in annotations match the actual file locations
