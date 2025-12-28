"""
Script nhanh để trực quan hóa một vài ảnh mẫu
Sử dụng khi cần xem nhanh kết quả
"""
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import numpy as np
import random


def quick_visualize_situ(num_images=3):
    """Visualize nhanh situ data"""
    project_root = Path(__file__).parent.parent
    raw_dir = project_root / "data" / "raw" / "inSitu" / "inSitu"
    processed_dir = project_root / "data" / "processing" / "inSitu" / "inSitu"
    
    if not raw_dir.exists() or not processed_dir.exists():
        print("❌ Không tìm thấy thư mục situ")
        return
    
    # Tìm ảnh
    images = []
    for class_dir in sorted(raw_dir.iterdir())[:3]:  # 3 class đầu
        if not class_dir.is_dir():
            continue
        video_dir = class_dir / "video"
        if video_dir.exists():
            png_files = list(video_dir.glob("*.png"))[:2]  # 2 ảnh mỗi class
            for png in png_files:
                processed = processed_dir / class_dir.name / "video" / png.name
                if processed.exists():
                    images.append((png, processed))
    
    if not images:
        print("❌ Không tìm thấy ảnh")
        return
    
    images = random.sample(images, min(num_images, len(images)))
    
    fig, axes = plt.subplots(len(images), 2, figsize=(12, 4 * len(images)))
    if len(images) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (raw, processed) in enumerate(images):
        img_raw = cv2.imread(str(raw))
        img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        
        img_proc = cv2.imread(str(processed))
        img_proc = cv2.cvtColor(img_proc, cv2.COLOR_BGR2RGB)
        
        axes[idx, 0].imshow(img_raw)
        axes[idx, 0].set_title(f'Trước: {raw.name}')
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(img_proc)
        axes[idx, 1].set_title(f'Sau: {processed.name}')
        axes[idx, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('situ_quick_visualize.png', dpi=150)
    print("✅ Đã lưu: situ_quick_visualize.png")
    plt.show()


def quick_visualize_vitro(num_images=3):
    """Visualize nhanh vitro data"""
    project_root = Path(__file__).parent.parent
    raw_dir = project_root / "data" / "raw" / "inVitro" / "inVitro"
    processed_dir = project_root / "data" / "processing" / "inVitro" / "inVitro"
    
    if not raw_dir.exists() or not processed_dir.exists():
        print("❌ Không tìm thấy thư mục vitro")
        return
    
    # Tìm ảnh
    images = []
    for class_dir in sorted(raw_dir.iterdir())[:3]:  # 3 class đầu
        if not class_dir.is_dir():
            continue
        web_dir = class_dir / "web" / "JPEG"
        if web_dir.exists():
            jpg_files = list(web_dir.glob("*.jpg"))[:2]  # 2 ảnh mỗi class
            for jpg in jpg_files:
                processed = processed_dir / class_dir.name / "web" / "JPEG" / jpg.name
                if processed.exists():
                    images.append((jpg, processed))
    
    if not images:
        print("❌ Không tìm thấy ảnh")
        return
    
    images = random.sample(images, min(num_images, len(images)))
    
    fig, axes = plt.subplots(len(images), 2, figsize=(12, 4 * len(images)))
    if len(images) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (raw, processed) in enumerate(images):
        img_raw = cv2.imread(str(raw))
        img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        
        img_proc = cv2.imread(str(processed))
        img_proc = cv2.cvtColor(img_proc, cv2.COLOR_BGR2RGB)
        
        axes[idx, 0].imshow(img_raw)
        axes[idx, 0].set_title(f'Trước: {raw.name}')
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(img_proc)
        axes[idx, 1].set_title(f'Sau: {processed.name}')
        axes[idx, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('vitro_quick_visualize.png', dpi=150)
    print("✅ Đã lưu: vitro_quick_visualize.png")
    plt.show()


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        dataset = sys.argv[1].lower()
        if dataset == 'situ':
            quick_visualize_situ()
        elif dataset == 'vitro':
            quick_visualize_vitro()
        else:
            print("Usage: python quick_visualize.py [situ|vitro]")
    else:
        print("Visualizing both datasets...")
        quick_visualize_situ()
        quick_visualize_vitro()

