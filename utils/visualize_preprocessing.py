"""
Script trực quan hóa dữ liệu trước và sau khi xử lý
Hiển thị so sánh ảnh gốc và ảnh đã được enhance cho cả situ và vitro
"""
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import cv2
import numpy as np
from typing import List, Tuple, Optional
import random


def load_image_safe(image_path: Path) -> Optional[np.ndarray]:
    """Load ảnh an toàn, trả về None nếu lỗi"""
    try:
        if image_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            img = cv2.imread(str(image_path))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        return None
    except Exception as e:
        print(f"   ⚠️  Lỗi load ảnh {image_path.name}: {e}")
        return None


def find_image_pairs_situ(
    raw_dir: Path, 
    processed_dir: Path, 
    num_samples: int = 5,
    num_classes: int = 5
) -> List[Tuple[Path, Path]]:
    """
    Tìm các cặp ảnh situ (gốc và đã xử lý)
    
    Args:
        raw_dir: Thư mục ảnh gốc (data/raw/inSitu/inSitu)
        processed_dir: Thư mục ảnh đã xử lý (data/processing/inSitu/inSitu)
        num_samples: Số ảnh mẫu mỗi class
        num_classes: Số class để lấy mẫu
    
    Returns:
        List các tuple (raw_path, processed_path)
    """
    pairs = []
    
    # Lấy các class có sẵn
    class_dirs = sorted([d for d in raw_dir.iterdir() if d.is_dir()])
    
    for class_dir in class_dirs[:num_classes]:
        try:
            class_id = int(class_dir.name)
        except ValueError:
            continue
        
        video_dir_raw = class_dir / "video"
        video_dir_processed = processed_dir / class_dir.name / "video"
        
        if not video_dir_raw.exists():
            continue
        
        # Tìm các ảnh PNG
        png_files = list(video_dir_raw.glob("*.png"))
        if not png_files:
            continue
        
        # Lấy mẫu ngẫu nhiên
        sample_files = random.sample(png_files, min(num_samples, len(png_files)))
        
        for png_file in sample_files:
            raw_path = png_file
            processed_path = video_dir_processed / png_file.name
            
            if processed_path.exists():
                pairs.append((raw_path, processed_path))
    
    return pairs


def find_image_pairs_vitro(
    raw_dir: Path, 
    processed_dir: Path, 
    num_samples: int = 5,
    num_classes: int = 5
) -> List[Tuple[Path, Path]]:
    """
    Tìm các cặp ảnh vitro (gốc và đã xử lý)
    
    Args:
        raw_dir: Thư mục ảnh gốc (data/raw/inVitro/inVitro)
        processed_dir: Thư mục ảnh đã xử lý (data/processing/inVitro/inVitro)
        num_samples: Số ảnh mẫu mỗi class
        num_classes: Số class để lấy mẫu
    
    Returns:
        List các tuple (raw_path, processed_path)
    """
    pairs = []
    
    # Lấy các class có sẵn
    class_dirs = sorted([d for d in raw_dir.iterdir() if d.is_dir()])
    
    for class_dir in class_dirs[:num_classes]:
        try:
            class_id = int(class_dir.name)
        except ValueError:
            continue
        
        # Tìm trong web/JPEG hoặc web/PNG
        web_dir_raw = class_dir / "web"
        web_dir_processed = processed_dir / class_dir.name / "web"
        
        if not web_dir_raw.exists():
            continue
        
        # Tìm ảnh JPEG
        jpeg_dir_raw = web_dir_raw / "JPEG"
        jpeg_dir_processed = web_dir_processed / "JPEG"
        
        image_files = []
        processed_files = []
        
        if jpeg_dir_raw.exists():
            image_files.extend(list(jpeg_dir_raw.glob("*.jpg")))
        
        # Tìm ảnh PNG
        png_dir_raw = web_dir_raw / "PNG"
        if png_dir_raw.exists():
            image_files.extend(list(png_dir_raw.glob("*.png")))
        
        if not image_files:
            continue
        
        # Lấy mẫu ngẫu nhiên
        sample_files = random.sample(image_files, min(num_samples, len(image_files)))
        
        for img_file in sample_files:
            raw_path = img_file
            
            # Tìm file tương ứng trong processed
            if img_file.suffix.lower() == '.jpg':
                processed_path = jpeg_dir_processed / img_file.name
            else:
                processed_path = web_dir_processed / "PNG" / img_file.name
            
            if processed_path.exists():
                pairs.append((raw_path, processed_path))
    
    return pairs


def visualize_comparison(
    pairs: List[Tuple[Path, Path]],
    title: str,
    max_images: int = 10,
    save_path: Optional[Path] = None
):
    """
    Visualize so sánh ảnh gốc và ảnh đã xử lý
    
    Args:
        pairs: List các tuple (raw_path, processed_path)
        title: Tiêu đề cho visualization
        max_images: Số ảnh tối đa để hiển thị
        save_path: Đường dẫn lưu ảnh (None = chỉ hiển thị)
    """
    if not pairs:
        print(f"   ⚠️  Không tìm thấy cặp ảnh nào cho {title}")
        return
    
    # Giới hạn số ảnh
    pairs = pairs[:max_images]
    
    # Tính số hàng và cột
    num_images = len(pairs)
    cols = 2  # Mỗi ảnh có 2 cột: gốc và đã xử lý
    rows = num_images
    
    # Tạo figure
    fig = plt.figure(figsize=(16, 4 * rows))
    fig.suptitle(f'{title} - So sánh trước và sau xử lý', fontsize=16, fontweight='bold')
    
    gs = gridspec.GridSpec(rows, cols, figure=fig, hspace=0.3, wspace=0.2)
    
    for idx, (raw_path, processed_path) in enumerate(pairs):
        # Load ảnh gốc
        img_raw = load_image_safe(raw_path)
        if img_raw is None:
            continue
        
        # Load ảnh đã xử lý
        img_processed = load_image_safe(processed_path)
        if img_processed is None:
            continue
        
        # Hiển thị ảnh gốc
        ax1 = fig.add_subplot(gs[idx, 0])
        ax1.imshow(img_raw)
        ax1.set_title(f'Trước xử lý\n{raw_path.name}', fontsize=10)
        ax1.axis('off')
        
        # Hiển thị ảnh đã xử lý
        ax2 = fig.add_subplot(gs[idx, 1])
        ax2.imshow(img_processed)
        ax2.set_title(f'Sau xử lý\n{processed_path.name}', fontsize=10)
        ax2.axis('off')
    
    # Lưu hoặc hiển thị
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   ✅ Đã lưu: {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_grid_comparison(
    pairs: List[Tuple[Path, Path]],
    title: str,
    grid_size: Tuple[int, int] = (3, 4),
    save_path: Optional[Path] = None
):
    """
    Visualize so sánh dạng grid (nhiều ảnh cùng lúc)
    
    Args:
        pairs: List các tuple (raw_path, processed_path)
        title: Tiêu đề cho visualization
        grid_size: Kích thước grid (rows, cols) - mỗi ảnh chiếm 2 cột
        save_path: Đường dẫn lưu ảnh
    """
    if not pairs:
        print(f"   ⚠️  Không tìm thấy cặp ảnh nào cho {title}")
        return
    
    rows, cols_per_pair = grid_size
    max_images = rows * (cols_per_pair // 2)  # Mỗi ảnh chiếm 2 cột
    pairs = pairs[:max_images]
    
    fig, axes = plt.subplots(rows, cols_per_pair, figsize=(20, 5 * rows))
    fig.suptitle(f'{title} - Grid Comparison', fontsize=16, fontweight='bold')
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    pair_idx = 0
    for row in range(rows):
        for col in range(0, cols_per_pair, 2):
            if pair_idx >= len(pairs):
                break
            
            raw_path, processed_path = pairs[pair_idx]
            
            # Load và hiển thị ảnh gốc
            img_raw = load_image_safe(raw_path)
            if img_raw is not None:
                axes[row, col].imshow(img_raw)
                axes[row, col].set_title('Trước', fontsize=9)
                axes[row, col].axis('off')
            
            # Load và hiển thị ảnh đã xử lý
            img_processed = load_image_safe(processed_path)
            if img_processed is not None:
                axes[row, col + 1].imshow(img_processed)
                axes[row, col + 1].set_title('Sau', fontsize=9)
                axes[row, col + 1].axis('off')
            
            pair_idx += 1
        
        if pair_idx >= len(pairs):
            break
    
    # Ẩn các subplot không sử dụng
    for idx in range(pair_idx * 2, rows * cols_per_pair):
        row = idx // cols_per_pair
        col = idx % cols_per_pair
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   ✅ Đã lưu: {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_datasets(
    project_root: Optional[Path] = None,
    num_samples: int = 5,
    num_classes: int = 5,
    save_dir: Optional[Path] = None,
    grid_mode: bool = False
):
    """
    Visualize dữ liệu cho cả situ và vitro
    
    Args:
        project_root: Thư mục gốc của project (None = tự động tìm)
        num_samples: Số ảnh mẫu mỗi class
        num_classes: Số class để lấy mẫu
        save_dir: Thư mục lưu ảnh visualization (None = chỉ hiển thị)
        grid_mode: True = dạng grid, False = dạng list
    """
    if project_root is None:
        project_root = Path(__file__).parent.parent
    
    # Đường dẫn thư mục
    situ_raw = project_root / "data" / "raw" / "inSitu" / "inSitu"
    situ_processed = project_root / "data" / "processing" / "inSitu" / "inSitu"
    vitro_raw = project_root / "data" / "raw" / "inVitro" / "inVitro"
    vitro_processed = project_root / "data" / "processing" / "inVitro" / "inVitro"
    
    print("="*70)
    print("🖼️  TRỰC QUAN HÓA DỮ LIỆU TRƯỚC VÀ SAU XỬ LÝ")
    print("="*70)
    
    # Tạo thư mục lưu nếu cần
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n📁 Thư mục lưu: {save_dir}")
    
    # Xử lý Situ
    print("\n📊 Xử lý dữ liệu Situ...")
    if situ_raw.exists() and situ_processed.exists():
        situ_pairs = find_image_pairs_situ(situ_raw, situ_processed, num_samples, num_classes)
        print(f"   ✅ Tìm thấy {len(situ_pairs)} cặp ảnh situ")
        
        if situ_pairs:
            if grid_mode:
                save_path = save_dir / "situ_comparison_grid.png" if save_dir else None
                visualize_grid_comparison(situ_pairs, "Situ Dataset", (3, 4), save_path)
            else:
                save_path = save_dir / "situ_comparison.png" if save_dir else None
                visualize_comparison(situ_pairs, "Situ Dataset", max_images=10, save_path=save_path)
    else:
        print(f"   ⚠️  Không tìm thấy thư mục situ")
        if not situ_raw.exists():
            print(f"      Raw: {situ_raw}")
        if not situ_processed.exists():
            print(f"      Processed: {situ_processed}")
    
    # Xử lý Vitro
    print("\n📊 Xử lý dữ liệu Vitro...")
    if vitro_raw.exists() and vitro_processed.exists():
        vitro_pairs = find_image_pairs_vitro(vitro_raw, vitro_processed, num_samples, num_classes)
        print(f"   ✅ Tìm thấy {len(vitro_pairs)} cặp ảnh vitro")
        
        if vitro_pairs:
            if grid_mode:
                save_path = save_dir / "vitro_comparison_grid.png" if save_dir else None
                visualize_grid_comparison(vitro_pairs, "Vitro Dataset", (3, 4), save_path)
            else:
                save_path = save_dir / "vitro_comparison.png" if save_dir else None
                visualize_comparison(vitro_pairs, "Vitro Dataset", max_images=10, save_path=save_path)
    else:
        print(f"   ⚠️  Không tìm thấy thư mục vitro")
        if not vitro_raw.exists():
            print(f"      Raw: {vitro_raw}")
        if not vitro_processed.exists():
            print(f"      Processed: {vitro_processed}")
    
    print("\n" + "="*70)
    print("✅ Hoàn thành!")
    print("="*70)


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Trực quan hóa dữ liệu trước và sau khi xử lý'
    )
    parser.add_argument(
        '--project-root',
        type=str,
        default=None,
        help='Thư mục gốc của project (mặc định: tự động tìm)'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=5,
        help='Số ảnh mẫu mỗi class (mặc định: 5)'
    )
    parser.add_argument(
        '--num-classes',
        type=int,
        default=5,
        help='Số class để lấy mẫu (mặc định: 5)'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default='visualizations',
        help='Thư mục lưu ảnh visualization (mặc định: visualizations)'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Không lưu ảnh, chỉ hiển thị'
    )
    parser.add_argument(
        '--grid',
        action='store_true',
        help='Hiển thị dạng grid (nhiều ảnh cùng lúc)'
    )
    
    args = parser.parse_args()
    
    project_root = Path(args.project_root) if args.project_root else None
    save_dir = None if args.no_save else Path(args.save_dir)
    
    visualize_datasets(
        project_root=project_root,
        num_samples=args.num_samples,
        num_classes=args.num_classes,
        save_dir=save_dir,
        grid_mode=args.grid
    )


if __name__ == '__main__':
    main()

