"""Data augmentation pipeline for store item detection."""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_augmentation_pipeline(
    image_size: int = 640,
    train: bool = True
):
    """
    Get augmentation pipeline.
    
    Args:
        image_size: Target image size
        train: Whether this is for training (applies augmentations)
        
    Returns:
        Albumentations compose pipeline
    """
    if train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=15,
                p=0.5
            ),
            A.OneOf([
                A.GaussNoise(p=1.0),
                A.GaussianBlur(p=1.0),
                A.MotionBlur(p=1.0),
            ], p=0.3),
            A.OneOf([
                A.OpticalDistortion(p=1.0),
                A.GridDistortion(p=1.0),
            ], p=0.2),
            A.HueSaturationValue(p=0.3),
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ], bbox_params=A.BboxParams(
            format='coco',
            label_fields=['class_labels']
        ))
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ], bbox_params=A.BboxParams(
            format='coco',
            label_fields=['class_labels']
        ))
