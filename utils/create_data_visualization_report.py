"""
Script tạo báo cáo trực quan hóa dữ liệu với biểu đồ và hình ảnh minh chứng
Cho cả situ và vitro, trước và sau khi xử lý
"""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import cv2
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import random
import seaborn as sns

# Set style cho đẹp hơn
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('ggplot')
sns.set_palette("husl")


def count_images_situ(data_dir: Path) -> Dict[int, int]:
    """Đếm số ảnh trong mỗi class của situ"""
    class_counts = {}
    
    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        
        try:
            class_id = int(class_dir.name)
        except ValueError:
            continue
        
        video_dir = class_dir / "video"
        if video_dir.exists():
            png_files = [f for f in video_dir.glob("*.png") 
                         if f.name.lower() != "thumbs.db"]
            class_counts[class_id] = len(png_files)
    
    return class_counts


def count_images_vitro(data_dir: Path) -> Dict[int, int]:
    """Đếm số ảnh trong mỗi class của vitro"""
    class_counts = {}
    
    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        
        try:
            class_id = int(class_dir.name)
        except ValueError:
            continue
        
        web_dir = class_dir / "web"
        if web_dir.exists():
            jpeg_dir = web_dir / "JPEG"
            png_dir = web_dir / "PNG"
            
            count = 0
            # Đếm JPEG
            if jpeg_dir.exists():
                count += len([f for f in jpeg_dir.glob("*.jpg") 
                             if f.name.lower() != "thumbs.db"])
            # Đếm PNG (vitro processed thường lưu ở đây)
            if png_dir.exists():
                count += len([f for f in png_dir.glob("*.png") 
                             if f.name.lower() != "thumbs.db"])
            
            if count > 0:
                class_counts[class_id] = count
    
    return class_counts


def load_image_safe(image_path: Path) -> Optional[np.ndarray]:
    """Load ảnh an toàn"""
    try:
        if image_path.exists():
            img = cv2.imread(str(image_path))
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return None
    except:
        return None


def find_sample_images_situ(
    raw_dir: Path, 
    processed_dir: Path, 
    num_samples: int = 6
) -> List[Tuple[Path, Path]]:
    """Tìm ảnh mẫu situ"""
    pairs = []
    
    for class_dir in sorted(raw_dir.iterdir())[:10]:  # 10 class đầu
        if not class_dir.is_dir():
            continue
        
        try:
            class_id = int(class_dir.name)
        except ValueError:
            continue
        
        video_dir_raw = class_dir / "video"
        video_dir_processed = processed_dir / class_dir.name / "video"
        
        if not video_dir_raw.exists():
            continue
        
        png_files = list(video_dir_raw.glob("*.png"))[:2]  # 2 ảnh mỗi class
        
        for png_file in png_files:
            processed_path = video_dir_processed / png_file.name
            if processed_path.exists():
                pairs.append((png_file, processed_path))
                if len(pairs) >= num_samples:
                    return pairs
    
    return pairs[:num_samples]


def find_sample_images_vitro(
    raw_dir: Path, 
    processed_dir: Path, 
    num_samples: int = 6
) -> List[Tuple[Path, Path]]:
    """Tìm ảnh mẫu vitro - tìm cả JPEG và PNG, bỏ qua augmented images"""
    pairs = []
    
    for class_dir in sorted(raw_dir.iterdir())[:20]:  # Tăng lên 20 class để tìm đủ
        if not class_dir.is_dir():
            continue
        
        try:
            class_id = int(class_dir.name)
        except ValueError:
            continue
        
        # Tìm trong JPEG (raw thường có)
        web_dir_raw_jpeg = class_dir / "web" / "JPEG"
        web_dir_processed_png = processed_dir / class_dir.name / "web" / "PNG"
        
        # Tìm trong PNG (raw có thể có)
        web_dir_raw_png = class_dir / "web" / "PNG"
        
        # Tìm JPEG trong raw và PNG trong processed (bỏ qua augmented)
        if web_dir_raw_jpeg.exists() and web_dir_processed_png.exists():
            jpg_files = [f for f in web_dir_raw_jpeg.glob("*.jpg") 
                        if f.name.lower() != "thumbs.db"][:3]
            
            for jpg_file in jpg_files:
                # Tìm file PNG tương ứng trong processed (có thể là web1.png, web2.png, ...)
                base_name = jpg_file.stem  # ví dụ: "web1"
                
                # Thử tìm file PNG với tên tương ứng (không có _aug)
                processed_path = web_dir_processed_png / f"{base_name}.png"
                if processed_path.exists():
                    pairs.append((jpg_file, processed_path))
                    if len(pairs) >= num_samples:
                        return pairs
        
        # Tìm PNG trong raw và processed
        if web_dir_raw_png.exists() and web_dir_processed_png.exists():
            # Chỉ lấy file không phải augmented (không có _aug trong tên)
            png_files = [f for f in web_dir_raw_png.glob("*.png") 
                        if f.name.lower() != "thumbs.db" and "_aug" not in f.stem][:3]
            
            for png_file in png_files:
                processed_path = web_dir_processed_png / png_file.name
                if processed_path.exists():
                    pairs.append((png_file, processed_path))
                    if len(pairs) >= num_samples:
                        return pairs
    
    return pairs[:num_samples]


def create_statistics_plots(
    situ_raw_counts: Dict[int, int],
    situ_processed_counts: Dict[int, int],
    vitro_raw_counts: Dict[int, int],
    vitro_processed_counts: Dict[int, int],
    save_path: Path
):
    """Tạo biểu đồ thống kê"""
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Bar chart so sánh tổng số ảnh
    ax1 = fig.add_subplot(gs[0, :])
    categories = ['Situ\n(Raw)', 'Situ\n(Processed)', 'Vitro\n(Raw)', 'Vitro\n(Processed)']
    totals = [
        sum(situ_raw_counts.values()),
        sum(situ_processed_counts.values()),
        sum(vitro_raw_counts.values()),
        sum(vitro_processed_counts.values())
    ]
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    bars = ax1.bar(categories, totals, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Số lượng ảnh', fontsize=12, fontweight='bold')
    ax1.set_title('Tổng số ảnh trước và sau xử lý', fontsize=14, fontweight='bold', pad=20)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Thêm giá trị trên cột
    for bar, total in zip(bars, totals):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{total:,}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 2. Histogram phân bố số ảnh mỗi class - Situ
    ax2 = fig.add_subplot(gs[1, 0])
    situ_raw_values = list(situ_raw_counts.values())
    situ_processed_values = list(situ_processed_counts.values())
    
    ax2.hist([situ_raw_values, situ_processed_values], 
             bins=30, alpha=0.7, label=['Raw', 'Processed'], 
             color=['#3498db', '#2ecc71'], edgecolor='black')
    ax2.set_xlabel('Số ảnh mỗi class', fontsize=11)
    ax2.set_ylabel('Số lượng class', fontsize=11)
    ax2.set_title('Phân bố số ảnh mỗi class - Situ', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 3. Histogram phân bố số ảnh mỗi class - Vitro
    ax3 = fig.add_subplot(gs[1, 1])
    vitro_raw_values = list(vitro_raw_counts.values())
    vitro_processed_values = list(vitro_processed_counts.values())
    
    ax3.hist([vitro_raw_values, vitro_processed_values], 
             bins=30, alpha=0.7, label=['Raw', 'Processed'], 
             color=['#e74c3c', '#f39c12'], edgecolor='black')
    ax3.set_xlabel('Số ảnh mỗi class', fontsize=11)
    ax3.set_ylabel('Số lượng class', fontsize=11)
    ax3.set_title('Phân bố số ảnh mỗi class - Vitro', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 4. Box plot so sánh
    ax4 = fig.add_subplot(gs[2, :])
    data_to_plot = [
        situ_raw_values,
        situ_processed_values,
        vitro_raw_values,
        vitro_processed_values
    ]
    labels = ['Situ Raw', 'Situ Processed', 'Vitro Raw', 'Vitro Processed']
    
    bp = ax4.boxplot(data_to_plot, tick_labels=labels, patch_artist=True, 
                     showmeans=True, meanline=True)
    
    # Tô màu
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax4.set_ylabel('Số ảnh mỗi class', fontsize=11)
    ax4.set_title('Box Plot: Phân bố số ảnh mỗi class', fontsize=12, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Đã lưu biểu đồ: {save_path}")
    plt.close()


def create_image_comparison(
    pairs: List[Tuple[Path, Path]],
    title: str,
    save_path: Path,
    max_images: int = 6
):
    """Tạo so sánh ảnh trước/sau"""
    pairs = pairs[:max_images]
    
    if not pairs:
        return
    
    rows = len(pairs)
    fig, axes = plt.subplots(rows, 2, figsize=(12, 4 * rows))
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    
    for idx, (raw_path, processed_path) in enumerate(pairs):
        img_raw = load_image_safe(raw_path)
        img_processed = load_image_safe(processed_path)
        
        if img_raw is not None:
            axes[idx, 0].imshow(img_raw)
            axes[idx, 0].set_title(f'Trước xử lý\n{raw_path.name}', 
                                  fontsize=10, fontweight='bold')
            axes[idx, 0].axis('off')
        
        if img_processed is not None:
            axes[idx, 1].imshow(img_processed)
            axes[idx, 1].set_title(f'Sau xử lý\n{processed_path.name}', 
                                  fontsize=10, fontweight='bold')
            axes[idx, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Đã lưu so sánh ảnh: {save_path}")
    plt.close()


def create_comprehensive_report(
    project_root: Optional[Path] = None,
    save_dir: Optional[Path] = None
):
    """Tạo báo cáo tổng hợp với biểu đồ và hình ảnh"""
    
    if project_root is None:
        project_root = Path(__file__).parent.parent
    
    # Đường dẫn
    situ_raw = project_root / "data" / "raw" / "inSitu" / "inSitu"
    situ_processed = project_root / "data" / "processing" / "inSitu" / "inSitu"
    vitro_raw = project_root / "data" / "raw" / "inVitro" / "inVitro"
    # Thử cả 2 đường dẫn có thể
    vitro_processed = project_root / "data" / "processing" / "vitro"
    if not vitro_processed.exists():
        vitro_processed = project_root / "data" / "processing" / "inVitro" / "inVitro"
    
    if save_dir is None:
        save_dir = project_root / "visualizations"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("📊 TẠO BÁO CÁO TRỰC QUAN HÓA DỮ LIỆU")
    print("="*70)
    
    # Đếm ảnh
    print("\n📈 Đang thống kê dữ liệu...")
    
    situ_raw_counts = {}
    situ_processed_counts = {}
    vitro_raw_counts = {}
    vitro_processed_counts = {}
    
    if situ_raw.exists():
        situ_raw_counts = count_images_situ(situ_raw)
        print(f"   ✅ Situ Raw: {len(situ_raw_counts)} classes, {sum(situ_raw_counts.values())} ảnh")
    
    if situ_processed.exists():
        situ_processed_counts = count_images_situ(situ_processed)
        print(f"   ✅ Situ Processed: {len(situ_processed_counts)} classes, {sum(situ_processed_counts.values())} ảnh")
    
    if vitro_raw.exists():
        vitro_raw_counts = count_images_vitro(vitro_raw)
        print(f"   ✅ Vitro Raw: {len(vitro_raw_counts)} classes, {sum(vitro_raw_counts.values())} ảnh")
    
    if vitro_processed.exists():
        vitro_processed_counts = count_images_vitro(vitro_processed)
        print(f"   ✅ Vitro Processed: {len(vitro_processed_counts)} classes, {sum(vitro_processed_counts.values())} ảnh")
    else:
        print(f"   ⚠️  Vitro Processed: Không tìm thấy thư mục {vitro_processed}")
        print(f"      💡 Gợi ý: Chạy script enhance_vitro_images.py để tạo dữ liệu đã xử lý")
    
    # Tạo biểu đồ thống kê
    print("\n📊 Đang tạo biểu đồ thống kê...")
    create_statistics_plots(
        situ_raw_counts,
        situ_processed_counts,
        vitro_raw_counts,
        vitro_processed_counts,
        save_dir / "statistics_comparison.png"
    )
    
    # Tạo so sánh ảnh Situ
    print("\n🖼️  Đang tạo so sánh ảnh Situ...")
    if situ_raw.exists() and situ_processed.exists():
        situ_pairs = find_sample_images_situ(situ_raw, situ_processed, num_samples=6)
        if situ_pairs:
            create_image_comparison(
                situ_pairs,
                "Situ Dataset - So sánh trước và sau xử lý",
                save_dir / "situ_image_comparison.png"
            )
    
    # Tạo so sánh ảnh Vitro
    print("\n🖼️  Đang tạo so sánh ảnh Vitro...")
    if vitro_raw.exists() and vitro_processed.exists():
        vitro_pairs = find_sample_images_vitro(vitro_raw, vitro_processed, num_samples=6)
        if vitro_pairs:
            create_image_comparison(
                vitro_pairs,
                "Vitro Dataset - So sánh trước và sau xử lý",
                save_dir / "vitro_image_comparison.png"
            )
        else:
            print(f"   ⚠️  Không tìm thấy cặp ảnh Vitro để so sánh")
            print(f"      💡 Có thể dữ liệu processed chưa được tạo hoặc đường dẫn không khớp")
    else:
        if not vitro_raw.exists():
            print(f"   ⚠️  Không tìm thấy thư mục Vitro Raw: {vitro_raw}")
        if not vitro_processed.exists():
            print(f"   ⚠️  Không tìm thấy thư mục Vitro Processed: {vitro_processed}")
            print(f"      💡 Chạy lệnh sau để tạo dữ liệu processed:")
            print(f"         python utils/enhance_vitro_images.py --execute")
    
    # Tạo báo cáo text
    print("\n📝 Đang tạo báo cáo text...")
    report_path = save_dir / "data_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("BÁO CÁO THỐNG KÊ DỮ LIỆU\n")
        f.write("="*70 + "\n\n")
        
        f.write("SITU DATASET\n")
        f.write("-"*70 + "\n")
        f.write(f"Raw: {len(situ_raw_counts)} classes, {sum(situ_raw_counts.values())} ảnh\n")
        f.write(f"Processed: {len(situ_processed_counts)} classes, {sum(situ_processed_counts.values())} ảnh\n")
        if situ_raw_counts and situ_processed_counts:
            f.write(f"Tăng: {sum(situ_processed_counts.values()) - sum(situ_raw_counts.values())} ảnh\n")
        f.write("\n")
        
        f.write("VITRO DATASET\n")
        f.write("-"*70 + "\n")
        f.write(f"Raw: {len(vitro_raw_counts)} classes, {sum(vitro_raw_counts.values())} ảnh\n")
        f.write(f"Processed: {len(vitro_processed_counts)} classes, {sum(vitro_processed_counts.values())} ảnh\n")
        if vitro_raw_counts and vitro_processed_counts:
            f.write(f"Tăng: {sum(vitro_processed_counts.values()) - sum(vitro_raw_counts.values())} ảnh\n")
        f.write("\n")
        
        f.write("TỔNG KẾT\n")
        f.write("-"*70 + "\n")
        total_raw = sum(situ_raw_counts.values()) + sum(vitro_raw_counts.values())
        total_processed = sum(situ_processed_counts.values()) + sum(vitro_processed_counts.values())
        f.write(f"Tổng Raw: {total_raw} ảnh\n")
        f.write(f"Tổng Processed: {total_processed} ảnh\n")
        f.write(f"Tổng tăng: {total_processed - total_raw} ảnh\n")
    
    print(f"   ✅ Đã lưu báo cáo: {report_path}")
    
    print("\n" + "="*70)
    print("✅ Hoàn thành!")
    print("="*70)
    print(f"\n📁 Tất cả file đã được lưu vào: {save_dir}")
    print("   - statistics_comparison.png: Biểu đồ thống kê")
    print("   - situ_image_comparison.png: So sánh ảnh Situ")
    print("   - vitro_image_comparison.png: So sánh ảnh Vitro")
    print("   - data_report.txt: Báo cáo text")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Tạo báo cáo trực quan hóa dữ liệu với biểu đồ và hình ảnh'
    )
    parser.add_argument(
        '--project-root',
        type=str,
        default=None,
        help='Thư mục gốc của project'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default='visualizations',
        help='Thư mục lưu kết quả (mặc định: visualizations)'
    )
    
    args = parser.parse_args()
    
    project_root = Path(args.project_root) if args.project_root else None
    save_dir = Path(args.save_dir)
    
    create_comprehensive_report(project_root=project_root, save_dir=save_dir)


if __name__ == '__main__':
    main()

