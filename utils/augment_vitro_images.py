"""
Script tăng cường dữ liệu (augmentation) cho ảnh đã làm nét
Tạo nhiều biến thể từ mỗi ảnh để tăng dataset
"""
import os
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
from torchvision import transforms
import random


def create_augmentation_transforms(num_augmentations: int = 15):
    """
    Tạo danh sách các transform augmentation
    
    Args:
        num_augmentations: Số lượng augmentation mỗi ảnh
    
    Returns:
        List các transform
    """
    augmentation_list = []
    
    # Base transform (luôn có)
    base_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Tạo nhiều biến thể augmentation
    for i in range(num_augmentations):
        # Random seed để mỗi lần khác nhau
        random.seed(i)
        np.random.seed(i)
        
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.ColorJitter(
                brightness=random.uniform(0.2, 0.4),
                contrast=random.uniform(0.2, 0.4),
                saturation=random.uniform(0.2, 0.4),
                hue=random.uniform(0.05, 0.2)
            ),
            transforms.RandomRotation(degrees=random.randint(15, 30)),
            transforms.RandomAffine(
                degrees=0,
                translate=(random.uniform(0.1, 0.2), random.uniform(0.1, 0.2))
            ),
            transforms.RandomPerspective(
                distortion_scale=random.uniform(0.1, 0.2),
                p=0.3
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.1)
        ])
        
        augmentation_list.append(transform)
    
    return augmentation_list


def augment_single_image(image_path: Path, output_dir: Path, num_augmentations: int = 15):
    """
    Tạo augmentation cho 1 ảnh
    
    Args:
        image_path: Đường dẫn ảnh gốc
        output_dir: Thư mục output
        num_augmentations: Số lượng augmentation
    
    Returns:
        Số lượng ảnh đã tạo thành công
    """
    try:
        # Đọc ảnh
        image = Image.open(image_path).convert('RGB')
        
        # Tạo transforms
        transforms_list = create_augmentation_transforms(num_augmentations)
        
        # Tạo thư mục output
        output_dir.mkdir(parents=True, exist_ok=True)
        
        success_count = 0
        
        # Tạo các biến thể
        for i, transform in enumerate(transforms_list):
            try:
                # Áp dụng transform
                augmented_tensor = transform(image)
                
                # Convert tensor về PIL Image để lưu
                # Denormalize
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                
                augmented_array = augmented_tensor.permute(1, 2, 0).numpy()
                augmented_array = augmented_array * std + mean
                augmented_array = np.clip(augmented_array, 0, 1)
                augmented_array = (augmented_array * 255).astype(np.uint8)
                
                augmented_image = Image.fromarray(augmented_array)
                
                # Tên file: {original_name}_aug{i}.png
                output_filename = f"{image_path.stem}_aug{i+1:02d}.png"
                output_path = output_dir / output_filename
                
                # Lưu ảnh
                augmented_image.save(output_path, quality=95)
                success_count += 1
                
            except Exception as e:
                print(f"   ⚠️  Lỗi khi tạo augmentation {i+1}: {e}")
                continue
        
        return success_count
        
    except Exception as e:
        print(f"   ❌ Lỗi khi đọc ảnh {image_path.name}: {e}")
        return 0


def process_vitro_augmentation(
    input_dir: str,
    output_dir: str = None,
    num_augmentations: int = 15,
    dry_run: bool = False
):
    """
    Tăng cường dữ liệu cho tất cả ảnh trong data vitro đã làm nét
    
    Args:
        input_dir: Đường dẫn đến thư mục ảnh đã làm nét (processing/inVitro/inVitro)
        output_dir: Thư mục output (None = lưu vào cùng thư mục input)
        num_augmentations: Số lượng augmentation mỗi ảnh
        dry_run: Chỉ đếm, không xử lý
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"❌ Không tìm thấy thư mục: {input_dir}")
        return
    
    # Nếu không có output_dir, dùng cùng thư mục input
    if output_dir is None:
        output_dir = input_dir
    
    # Tìm tất cả ảnh PNG
    png_files = []
    for class_dir in sorted(input_path.iterdir()):
        if not class_dir.is_dir():
            continue
        
        try:
            class_id = int(class_dir.name)
        except ValueError:
            continue
        
        web_dir = class_dir / "web" / "PNG"
        if web_dir.exists():
            for png_file in web_dir.glob("*.png"):
                # Bỏ qua ảnh đã được augment (có _aug trong tên)
                if "_aug" not in png_file.stem and png_file.name.lower() != "thumbs.db":
                    png_files.append(png_file)
    
    print(f"\n📊 Thống kê:")
    print(f"   Tổng số ảnh gốc: {len(png_files)}")
    print(f"   Số augmentation mỗi ảnh: {num_augmentations}")
    print(f"   Tổng số ảnh sẽ tạo: {len(png_files) * num_augmentations}")
    print(f"   Output: {output_dir}")
    print(f"   Mode: {'DRY RUN' if dry_run else 'XỬ LÝ THẬT'}\n")
    
    if dry_run:
        print("⚠️  DRY RUN: Không tạo augmentation nào")
        print("   Để tạo thật, chạy lại với --execute")
        return
    
    # Xử lý augmentation
    total_success = 0
    total_failed = 0
    
    print(f"🖼️  Đang tạo augmentation...")
    for png_file in tqdm(png_files, desc="Augmenting"):
        # Tạo thư mục output giữ nguyên cấu trúc
        rel_path = png_file.relative_to(input_path)
        output_file_dir = Path(output_dir) / rel_path.parent
        
        success = augment_single_image(png_file, output_file_dir, num_augmentations)
        total_success += success
        if success < num_augmentations:
            total_failed += (num_augmentations - success)
    
    print(f"\n✅ Hoàn thành!")
    print(f"   Tổng số ảnh đã tạo: {total_success}")
    print(f"   Lỗi: {total_failed}")
    print(f"   Ảnh đã lưu vào: {output_dir}")
    print(f"\n💡 Tổng số ảnh sau augmentation:")
    print(f"   Gốc: {len(png_files)}")
    print(f"   Augmented: {total_success}")
    print(f"   Tổng: {len(png_files) + total_success}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Tăng cường dữ liệu (augmentation) cho ảnh đã làm nét'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='data/processing/inVitro/inVitro',
        help='Đường dẫn đến thư mục ảnh đã làm nét'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Thư mục output (None = lưu vào cùng thư mục input)'
    )
    parser.add_argument(
        '--num-augmentations',
        type=int,
        default=15,
        help='Số lượng augmentation mỗi ảnh (mặc định: 15)'
    )
    parser.add_argument(
        '--execute',
        action='store_true',
        help='Thực sự tạo augmentation (mặc định chỉ dry-run)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("🔄 AUGMENT VITRO IMAGES - Tăng cường dữ liệu")
    print("="*70)
    
    process_vitro_augmentation(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        num_augmentations=args.num_augmentations,
        dry_run=not args.execute
    )
    
    print("\n" + "="*70)
    print("✅ Hoàn thành!")
    print("="*70)
    print("\n💡 Lưu ý:")
    print("   - Mỗi ảnh sẽ tạo thêm N ảnh augmented")
    print("   - Ảnh gốc được giữ nguyên")
    print("   - Tên file: {original}_aug01.png, {original}_aug02.png, ...")
    print("   - Augmentation bao gồm: flip, rotation, color jitter, perspective, erasing")


if __name__ == '__main__':
    main()

