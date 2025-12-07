"""
Script tăng cường dữ liệu (augmentation) cho situ - CHỈ AUGMENT CÁC LỚP ÍT ẢNH
Tăng số lượng ảnh cho các classes có < threshold ảnh
"""
import os
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
from torchvision import transforms
import random
from collections import defaultdict


def create_augmentation_transforms(num_augmentations: int = 15):
    """
    Tạo danh sách các transform augmentation cho situ
    Situ ảnh từ video nên augmentation mạnh hơn một chút
    
    Args:
        num_augmentations: Số lượng augmentation mỗi ảnh
    
    Returns:
        List các transform
    """
    augmentation_list = []
    
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
                brightness=random.uniform(0.2, 0.5),  # Mạnh hơn cho situ
                contrast=random.uniform(0.2, 0.5),
                saturation=random.uniform(0.2, 0.5),
                hue=random.uniform(0.05, 0.2)
            ),
            transforms.RandomRotation(degrees=random.randint(15, 35)),  # Rotation lớn hơn
            transforms.RandomAffine(
                degrees=0,
                translate=(random.uniform(0.1, 0.25), random.uniform(0.1, 0.25))
            ),
            transforms.RandomPerspective(
                distortion_scale=random.uniform(0.1, 0.25),
                p=0.4
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.15)  # Erasing cao hơn
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


def analyze_situ_classes(data_dir: Path, threshold: int = 50):
    """
    Phân tích situ data để tìm các lớp ít ảnh
    
    Args:
        data_dir: Thư mục situ data
        threshold: Ngưỡng số ảnh (classes < threshold sẽ được augment)
    
    Returns:
        Dict: {class_id: image_count} cho các lớp < threshold
    """
    class_counts = defaultdict(int)
    
    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        
        try:
            class_id = int(class_dir.name)
        except ValueError:
            continue
        
        video_dir = class_dir / "video"
        if video_dir.exists():
            video_files = [f for f in video_dir.glob("*.png") 
                          if f.name.lower() != "thumbs.db" and "_aug" not in f.stem]
            class_counts[class_id] = len(video_files)
    
    # Chỉ lấy các lớp < threshold
    classes_to_augment = {
        class_id: count 
        for class_id, count in class_counts.items() 
        if count < threshold
    }
    
    return classes_to_augment, class_counts


def calculate_augmentation_count(current_count: int, target_count: int = 50):
    """
    Tính số lượng augmentation cần thiết để đạt target_count
    
    Args:
        current_count: Số ảnh hiện tại
        target_count: Số ảnh mục tiêu
    
    Returns:
        Số augmentation mỗi ảnh
    """
    if current_count == 0:
        return 0
    
    # Số ảnh cần tạo thêm
    needed = target_count - current_count
    
    # Số augmentation mỗi ảnh (làm tròn lên)
    aug_per_image = max(1, (needed + current_count - 1) // current_count)
    
    # Giới hạn tối đa 20 augmentation/ảnh để tránh quá nhiều
    return min(aug_per_image, 20)


def process_situ_augmentation(
    input_dir: str,
    output_dir: str = None,
    threshold: int = 50,
    target_count: int = 50,
    dry_run: bool = False
):
    """
    Tăng cường dữ liệu cho situ - CHỈ AUGMENT CÁC LỚP ÍT ẢNH
    
    Args:
        input_dir: Đường dẫn đến thư mục situ data
        output_dir: Thư mục output (None = lưu vào cùng thư mục input)
        threshold: Chỉ augment classes có < threshold ảnh
        target_count: Số ảnh mục tiêu cho mỗi class (sau augmentation)
        dry_run: Chỉ đếm, không xử lý
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"❌ Không tìm thấy thư mục: {input_dir}")
        return
    
    # Nếu không có output_dir, dùng cùng thư mục input
    if output_dir is None:
        output_dir = input_dir
    
    print(f"\n📊 Phân tích situ data...")
    classes_to_augment, all_class_counts = analyze_situ_classes(input_path, threshold)
    
    print(f"\n📈 Thống kê:")
    print(f"   Tổng số classes: {len(all_class_counts)}")
    print(f"   Classes < {threshold} ảnh: {len(classes_to_augment)}")
    print(f"   Classes >= {threshold} ảnh: {len(all_class_counts) - len(classes_to_augment)}")
    
    if not classes_to_augment:
        print(f"\n✅ Không có class nào cần augment (tất cả đều >= {threshold} ảnh)")
        return
    
    # Tính toán augmentation
    total_original = 0
    total_augmented = 0
    augmentation_plan = {}
    
    print(f"\n📋 Kế hoạch augmentation:")
    print(f"   {'Class ID':<10} {'Hiện tại':<12} {'Cần tạo':<12} {'Aug/ảnh':<10} {'Sau augment':<12}")
    print(f"   {'-'*60}")
    
    for class_id, current_count in sorted(classes_to_augment.items()):
        aug_per_image = calculate_augmentation_count(current_count, target_count)
        total_aug = current_count * aug_per_image
        final_count = current_count + total_aug
        
        augmentation_plan[class_id] = {
            'current': current_count,
            'aug_per_image': aug_per_image,
            'total_aug': total_aug,
            'final': final_count
        }
        
        total_original += current_count
        total_augmented += total_aug
        
        print(f"   {class_id:<10} {current_count:<12} {total_aug:<12} {aug_per_image:<10} {final_count:<12}")
    
    print(f"\n   {'TỔNG:':<10} {total_original:<12} {total_augmented:<12} {'':<10} {total_original + total_augmented:<12}")
    print(f"\n   Output: {output_dir}")
    print(f"   Mode: {'DRY RUN' if dry_run else 'XỬ LÝ THẬT'}\n")
    
    if dry_run:
        print("⚠️  DRY RUN: Không tạo augmentation nào")
        print("   Để tạo thật, chạy lại với --execute")
        return
    
    # Xử lý augmentation
    total_success = 0
    total_failed = 0
    
    print(f"🖼️  Đang tạo augmentation cho {len(classes_to_augment)} classes...")
    
    for class_id, plan in tqdm(augmentation_plan.items(), desc="Processing classes"):
        class_dir = input_path / str(class_id)
        video_dir = class_dir / "video"
        
        if not video_dir.exists():
            continue
        
        # Lấy tất cả ảnh gốc (bỏ qua ảnh đã augment)
        video_files = [
            f for f in sorted(video_dir.glob("*.png"))
            if f.name.lower() != "thumbs.db" and "_aug" not in f.stem
        ]
        
        # Tạo thư mục output
        output_video_dir = Path(output_dir) / str(class_id) / "video"
        
        # Augment mỗi ảnh
        for video_file in video_files:
            success = augment_single_image(
                video_file, 
                output_video_dir, 
                plan['aug_per_image']
            )
            total_success += success
            if success < plan['aug_per_image']:
                total_failed += (plan['aug_per_image'] - success)
    
    print(f"\n✅ Hoàn thành!")
    print(f"   Tổng số ảnh đã tạo: {total_success}")
    print(f"   Lỗi: {total_failed}")
    print(f"   Ảnh đã lưu vào: {output_dir}")
    
    # Thống kê sau augmentation
    print(f"\n💡 Tổng số ảnh sau augmentation:")
    print(f"   Gốc: {total_original}")
    print(f"   Augmented: {total_success}")
    print(f"   Tổng: {total_original + total_success}")
    
    # Kiểm tra lại
    print(f"\n🔍 Kiểm tra lại sau augmentation...")
    final_classes, _ = analyze_situ_classes(Path(output_dir), threshold)
    if final_classes:
        print(f"   ⚠️  Vẫn còn {len(final_classes)} classes < {threshold} ảnh")
        print(f"      {list(final_classes.keys())[:10]}...")
    else:
        print(f"   ✅ Tất cả classes đều >= {threshold} ảnh!")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Tăng cường dữ liệu (augmentation) cho situ - CHỈ AUGMENT CÁC LỚP ÍT ẢNH'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/raw/inSitu/inSitu',
        help='Đường dẫn đến thư mục situ data'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Thư mục output (None = lưu vào cùng thư mục input)'
    )
    parser.add_argument(
        '--threshold',
        type=int,
        default=50,
        help='Chỉ augment classes có < threshold ảnh (mặc định: 50)'
    )
    parser.add_argument(
        '--target-count',
        type=int,
        default=50,
        help='Số ảnh mục tiêu cho mỗi class sau augmentation (mặc định: 50)'
    )
    parser.add_argument(
        '--execute',
        action='store_true',
        help='Thực sự tạo augmentation (mặc định chỉ dry-run)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("🔄 AUGMENT SITU IMAGES - Tăng cường dữ liệu cho các lớp ít ảnh")
    print("="*70)
    
    process_situ_augmentation(
        input_dir=args.data_dir,
        output_dir=args.output_dir,
        threshold=args.threshold,
        target_count=args.target_count,
        dry_run=not args.execute
    )
    
    print("\n" + "="*70)
    print("✅ Hoàn thành!")
    print("="*70)
    print("\n💡 Lưu ý:")
    print("   - Chỉ augment các classes có < threshold ảnh")
    print("   - Số augmentation phụ thuộc vào số ảnh hiện tại")
    print("   - Mục tiêu: đạt target_count ảnh/class")
    print("   - Ảnh gốc được giữ nguyên")
    print("   - Tên file: {original}_aug01.png, {original}_aug02.png, ...")
    print("\n💡 Kết hợp với weighted sampling để tối ưu training!")


if __name__ == '__main__':
    main()

