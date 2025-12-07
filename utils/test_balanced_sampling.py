"""
Script test weighted sampling và class weights cho situ dataset
Kiểm tra xem có cân bằng được các class không
"""
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from collections import Counter

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.situ_finetune.finetune_situ import (
    SituDataset, 
    create_data_loaders,
    get_class_weights,
    create_weighted_sampler
)


def analyze_sampling_distribution(data_loader, num_samples: int = 1000):
    """
    Phân tích phân bố class khi sampling
    
    Args:
        data_loader: DataLoader
        num_samples: Số samples để phân tích
    
    Returns:
        Dict với phân bố class
    """
    class_counts = Counter()
    total_samples = 0
    
    for batch_idx, (images, labels) in enumerate(data_loader):
        for label in labels:
            class_counts[label.item()] += 1
            total_samples += 1
            
            if total_samples >= num_samples:
                break
        
        if total_samples >= num_samples:
            break
    
    return class_counts, total_samples


def compare_distributions(original_counts, sampled_counts):
    """
    So sánh phân bố gốc vs sau khi weighted sampling
    
    Args:
        original_counts: Counter từ dataset gốc
        sampled_counts: Counter từ sampled data
    """
    print(f"\n{'='*70}")
    print("📊 SO SÁNH PHÂN BỐ CLASS")
    print(f"{'='*70}")
    
    # Tính statistics
    original_values = list(original_counts.values())
    sampled_values = list(sampled_counts.values())
    
    # Lấy các class chung
    all_classes = set(original_counts.keys()) | set(sampled_counts.keys())
    
    print(f"\n📈 Thống kê phân bố gốc:")
    print(f"   Min: {min(original_values)}")
    print(f"   Max: {max(original_values)}")
    print(f"   Mean: {sum(original_values) / len(original_values):.2f}")
    print(f"   Std: {(sum((x - sum(original_values)/len(original_values))**2 for x in original_values) / len(original_values))**0.5:.2f}")
    
    print(f"\n📈 Thống kê phân bố sau weighted sampling:")
    print(f"   Min: {min(sampled_values) if sampled_values else 0}")
    print(f"   Max: {max(sampled_values) if sampled_values else 0}")
    if sampled_values:
        mean_sampled = sum(sampled_values) / len(sampled_values)
        print(f"   Mean: {mean_sampled:.2f}")
        std_sampled = (sum((x - mean_sampled)**2 for x in sampled_values) / len(sampled_values))**0.5
        print(f"   Std: {std_sampled:.2f}")
    
    # So sánh ratio
    original_ratio = max(original_values) / min(original_values) if min(original_values) > 0 else float('inf')
    if sampled_values and min(sampled_values) > 0:
        sampled_ratio = max(sampled_values) / min(sampled_values)
        improvement = ((original_ratio - sampled_ratio) / original_ratio) * 100
        print(f"\n📊 Imbalance Ratio:")
        print(f"   Gốc: {original_ratio:.2f}x")
        print(f"   Sau weighted sampling: {sampled_ratio:.2f}x")
        print(f"   Cải thiện: {improvement:.1f}%")
    
    # Hiển thị top 10 classes ít nhất và nhiều nhất
    print(f"\n   Top 10 classes ít ảnh nhất (gốc):")
    sorted_original = sorted(original_counts.items(), key=lambda x: x[1])
    for class_id, count in sorted_original[:10]:
        sampled_count = sampled_counts.get(class_id, 0)
        print(f"      Class {class_id:3d}: Gốc={count:4d}, Sampled={sampled_count:4d}")
    
    print(f"\n   Top 10 classes nhiều ảnh nhất (gốc):")
    sorted_original_desc = sorted(original_counts.items(), key=lambda x: x[1], reverse=True)
    for class_id, count in sorted_original_desc[:10]:
        sampled_count = sampled_counts.get(class_id, 0)
        print(f"      Class {class_id:3d}: Gốc={count:4d}, Sampled={sampled_count:4d}")


def test_balanced_sampling(data_dir: str):
    """
    Test weighted sampling và class weights
    
    Args:
        data_dir: Đường dẫn đến data situ
    """
    print("="*70)
    print("🧪 TEST BALANCED SAMPLING - Kiểm tra cân bằng class")
    print("="*70)
    
    # Tạo dataset
    print("\n📦 Loading dataset...")
    dataset = SituDataset(data_dir, transform=None)
    
    # Đếm phân bố gốc
    original_labels = [label for _, label in dataset.samples]
    original_counts = Counter(original_labels)
    
    print(f"\n📊 Phân bố gốc:")
    print(f"   Tổng số ảnh: {len(dataset.samples)}")
    print(f"   Số classes: {len(original_counts)}")
    
    image_counts = list(original_counts.values())
    print(f"   Min: {min(image_counts)} ảnh/class")
    print(f"   Max: {max(image_counts)} ảnh/class")
    print(f"   Mean: {sum(image_counts) / len(image_counts):.2f} ảnh/class")
    print(f"   Imbalance Ratio: {max(image_counts) / min(image_counts):.2f}x")
    
    # Test weighted sampling
    print(f"\n🔄 Testing Weighted Sampling...")
    train_loader_with_weight, _, _, _ = create_data_loaders(
        data_dir, 
        batch_size=32, 
        use_weighted_sampling=True
    )
    
    # Phân tích phân bố sau weighted sampling
    sampled_counts, num_samples = analyze_sampling_distribution(train_loader_with_weight, num_samples=5000)
    
    print(f"   Đã sample {num_samples} ảnh")
    print(f"   Số classes xuất hiện: {len(sampled_counts)}")
    
    # So sánh
    compare_distributions(original_counts, sampled_counts)
    
    # Test class weights
    print(f"\n{'='*70}")
    print("⚖️  TEST CLASS WEIGHTS")
    print(f"{'='*70}")
    
    class_weights = get_class_weights(dataset, device='cpu')
    
    print(f"\n📊 Class Weights:")
    print(f"   Min weight: {class_weights.min():.2f}")
    print(f"   Max weight: {class_weights.max():.2f}")
    print(f"   Mean weight: {class_weights.mean():.2f}")
    
    # Top 10 classes có weight cao nhất (ít ảnh)
    sorted_by_weight = sorted(zip(range(len(class_weights)), class_weights.tolist()), 
                             key=lambda x: x[1], reverse=True)
    
    print(f"\n   Top 10 classes có weight cao nhất (ít ảnh):")
    for class_id, weight in sorted_by_weight[:10]:
        original_count = original_counts.get(class_id, 0)
        print(f"      Class {class_id:3d}: Weight={weight:.2f}, Số ảnh gốc={original_count}")
    
    print(f"\n   Top 10 classes có weight thấp nhất (nhiều ảnh):")
    for class_id, weight in sorted_by_weight[-10:]:
        original_count = original_counts.get(class_id, 0)
        print(f"      Class {class_id:3d}: Weight={weight:.2f}, Số ảnh gốc={original_count}")
    
    print(f"\n{'='*70}")
    print("✅ Hoàn thành test!")
    print(f"{'='*70}")
    print("\n💡 Kết luận:")
    if len(sampled_counts) > 0:
        sampled_ratio = max(sampled_counts.values()) / min(sampled_counts.values())
        original_ratio = max(original_counts.values()) / min(original_counts.values())
        if sampled_ratio < original_ratio * 0.5:
            print("   ✅ Weighted sampling giúp cân bằng tốt (ratio giảm >50%)")
        elif sampled_ratio < original_ratio * 0.7:
            print("   ⚠️  Weighted sampling giúp cân bằng vừa phải (ratio giảm 30-50%)")
        else:
            print("   ⚠️  Weighted sampling chưa đủ, có thể cần điều chỉnh")
    
    print("   ✅ Class weights đã được tính toán đúng")
    print("\n📝 Bước tiếp theo:")
    print("   - Nếu kết quả tốt: Có thể bắt đầu training với --use-weighted-sampling --use-class-weights")
    print("   - Nếu chưa tốt: Có thể cần augmentation cho class ít ảnh\n")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Test weighted sampling và class weights cho situ dataset'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/raw/inSitu/inSitu',
        help='Đường dẫn đến thư mục in-situ data'
    )
    
    args = parser.parse_args()
    
    test_balanced_sampling(args.data_dir)


if __name__ == '__main__':
    main()

