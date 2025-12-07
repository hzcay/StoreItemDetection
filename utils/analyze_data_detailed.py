"""
Script phân tích chi tiết dữ liệu - kiểm tra từng vấn đề cụ thể
Chạy từ từ, không vội, để hiểu rõ dữ liệu trước khi xử lý
"""
import os
from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image
import json


def check_image_validity(image_path: Path) -> tuple[bool, str]:
    """
    Kiểm tra ảnh có hợp lệ không
    
    Returns:
        (is_valid, error_message)
    """
    try:
        with Image.open(image_path) as img:
            img.verify()  # Verify image integrity
            return True, ""
    except Exception as e:
        return False, str(e)


def analyze_vitro_detailed(data_dir: Path):
    """Phân tích chi tiết in-vitro data"""
    print("\n" + "="*70)
    print("🔬 PHÂN TÍCH CHI TIẾT IN-VITRO DATA")
    print("="*70)
    
    issues = {
        'empty_classes': [],
        'very_few_images': [],  # < 5 ảnh
        'few_images': [],       # 5-10 ảnh
        'invalid_images': [],
        'missing_web_dir': [],
        'missing_jpeg_dir': [],
        'missing_png_dir': [],
        'class_details': {}
    }
    
    total_valid_images = 0
    total_invalid_images = 0
    
    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        
        try:
            class_id = int(class_dir.name)
        except ValueError:
            continue
        
        class_info = {
            'class_id': class_id,
            'jpeg_count': 0,
            'png_count': 0,
            'total': 0,
            'invalid': 0,
            'paths': []
        }
        
        # Kiểm tra web/
        web_dir = class_dir / "web"
        if not web_dir.exists():
            issues['missing_web_dir'].append(class_id)
            continue
        
        # Kiểm tra JPEG/
        jpeg_dir = web_dir / "JPEG"
        if jpeg_dir.exists():
            jpeg_files = [f for f in jpeg_dir.glob("*.jpg") if f.name.lower() != "thumbs.db"]
            for img_file in jpeg_files:
                is_valid, error = check_image_validity(img_file)
                if is_valid:
                    class_info['jpeg_count'] += 1
                    total_valid_images += 1
                else:
                    class_info['invalid'] += 1
                    total_invalid_images += 1
                    issues['invalid_images'].append((str(img_file), error))
        else:
            issues['missing_jpeg_dir'].append(class_id)
        
        # Kiểm tra PNG/
        png_dir = web_dir / "PNG"
        if png_dir.exists():
            png_files = [f for f in png_dir.glob("*.png") if f.name.lower() != "thumbs.db"]
            for img_file in png_files:
                is_valid, error = check_image_validity(img_file)
                if is_valid:
                    class_info['png_count'] += 1
                    total_valid_images += 1
                else:
                    class_info['invalid'] += 1
                    total_invalid_images += 1
                    issues['invalid_images'].append((str(img_file), error))
        else:
            issues['missing_png_dir'].append(class_id)
        
        class_info['total'] = class_info['jpeg_count'] + class_info['png_count']
        issues['class_details'][class_id] = class_info
        
        # Phân loại theo số lượng ảnh
        if class_info['total'] == 0:
            issues['empty_classes'].append(class_id)
        elif class_info['total'] < 5:
            issues['very_few_images'].append(class_id)
        elif class_info['total'] <= 10:
            issues['few_images'].append(class_id)
    
    # In kết quả
    print(f"\n📊 Tổng quan:")
    print(f"   Tổng số classes: {len(issues['class_details'])}")
    print(f"   Tổng số ảnh hợp lệ: {total_valid_images}")
    print(f"   Tổng số ảnh không hợp lệ: {total_invalid_images}")
    
    print(f"\n⚠️  Vấn đề phát hiện:")
    print(f"   Classes không có ảnh: {len(issues['empty_classes'])}")
    if issues['empty_classes']:
        print(f"      {issues['empty_classes'][:10]}..." if len(issues['empty_classes']) > 10 else f"      {issues['empty_classes']}")
    
    print(f"   Classes có rất ít ảnh (<5): {len(issues['very_few_images'])}")
    if issues['very_few_images']:
        print(f"      {issues['very_few_images'][:10]}..." if len(issues['very_few_images']) > 10 else f"      {issues['very_few_images']}")
    
    print(f"   Classes có ít ảnh (5-10): {len(issues['few_images'])}")
    if issues['few_images']:
        print(f"      {issues['few_images'][:10]}..." if len(issues['few_images']) > 10 else f"      {issues['few_images']}")
    
    print(f"   Classes thiếu web/JPEG/: {len(issues['missing_jpeg_dir'])}")
    print(f"   Classes thiếu web/PNG/: {len(issues['missing_png_dir'])}")
    print(f"   Ảnh không hợp lệ: {len(issues['invalid_images'])}")
    
    if issues['invalid_images']:
        print(f"\n   Chi tiết ảnh không hợp lệ (5 đầu tiên):")
        for img_path, error in issues['invalid_images'][:5]:
            print(f"      {Path(img_path).name}: {error}")
    
    # Phân bố chi tiết
    image_counts = [info['total'] for info in issues['class_details'].values()]
    if image_counts:
        print(f"\n📈 Phân bố số ảnh/class:")
        print(f"   Min: {min(image_counts)}")
        print(f"   Max: {max(image_counts)}")
        print(f"   Mean: {sum(image_counts) / len(image_counts):.2f}")
        print(f"   Median: {sorted(image_counts)[len(image_counts)//2]}")
        
        # Top 10 classes có ít ảnh nhất
        sorted_classes = sorted(issues['class_details'].items(), key=lambda x: x[1]['total'])
        print(f"\n   Top 10 classes có ít ảnh nhất:")
        for class_id, info in sorted_classes[:10]:
            print(f"      Class {class_id}: {info['total']} ảnh (JPEG: {info['jpeg_count']}, PNG: {info['png_count']})")
        
        # Top 10 classes có nhiều ảnh nhất
        sorted_classes_desc = sorted(issues['class_details'].items(), key=lambda x: x[1]['total'], reverse=True)
        print(f"\n   Top 10 classes có nhiều ảnh nhất:")
        for class_id, info in sorted_classes_desc[:10]:
            print(f"      Class {class_id}: {info['total']} ảnh (JPEG: {info['jpeg_count']}, PNG: {info['png_count']})")
    
    return issues


def analyze_situ_detailed(data_dir: Path):
    """Phân tích chi tiết in-situ data"""
    print("\n" + "="*70)
    print("🔬 PHÂN TÍCH CHI TIẾT IN-SITU DATA")
    print("="*70)
    
    issues = {
        'empty_classes': [],
        'very_few_images': [],  # < 20 ảnh
        'few_images': [],       # 20-50 ảnh
        'many_images': [],      # 100+ ảnh
        'invalid_images': [],
        'missing_video_dir': [],
        'class_details': {}
    }
    
    total_valid_images = 0
    total_invalid_images = 0
    
    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        
        try:
            class_id = int(class_dir.name)
        except ValueError:
            continue
        
        class_info = {
            'class_id': class_id,
            'total': 0,
            'invalid': 0,
            'paths': []
        }
        
        # Kiểm tra video/
        video_dir = class_dir / "video"
        if not video_dir.exists():
            issues['missing_video_dir'].append(class_id)
            continue
        
        video_files = [f for f in video_dir.glob("*.png") if f.name.lower() != "thumbs.db"]
        for img_file in video_files:
            is_valid, error = check_image_validity(img_file)
            if is_valid:
                class_info['total'] += 1
                total_valid_images += 1
            else:
                class_info['invalid'] += 1
                total_invalid_images += 1
                issues['invalid_images'].append((str(img_file), error))
        
        issues['class_details'][class_id] = class_info
        
        # Phân loại theo số lượng ảnh
        if class_info['total'] == 0:
            issues['empty_classes'].append(class_id)
        elif class_info['total'] < 20:
            issues['very_few_images'].append(class_id)
        elif class_info['total'] <= 50:
            issues['few_images'].append(class_id)
        elif class_info['total'] >= 100:
            issues['many_images'].append(class_id)
    
    # In kết quả
    print(f"\n📊 Tổng quan:")
    print(f"   Tổng số classes: {len(issues['class_details'])}")
    print(f"   Tổng số ảnh hợp lệ: {total_valid_images}")
    print(f"   Tổng số ảnh không hợp lệ: {total_invalid_images}")
    
    print(f"\n⚠️  Vấn đề phát hiện:")
    print(f"   Classes không có ảnh: {len(issues['empty_classes'])}")
    if issues['empty_classes']:
        print(f"      {issues['empty_classes'][:10]}..." if len(issues['empty_classes']) > 10 else f"      {issues['empty_classes']}")
    
    print(f"   Classes có rất ít ảnh (<20): {len(issues['very_few_images'])}")
    if issues['very_few_images']:
        print(f"      {issues['very_few_images'][:10]}..." if len(issues['very_few_images']) > 10 else f"      {issues['very_few_images']}")
    
    print(f"   Classes có ít ảnh (20-50): {len(issues['few_images'])}")
    print(f"   Classes có nhiều ảnh (100+): {len(issues['many_images'])}")
    if issues['many_images']:
        print(f"      {issues['many_images'][:10]}..." if len(issues['many_images']) > 10 else f"      {issues['many_images']}")
    
    print(f"   Classes thiếu video/: {len(issues['missing_video_dir'])}")
    print(f"   Ảnh không hợp lệ: {len(issues['invalid_images'])}")
    
    if issues['invalid_images']:
        print(f"\n   Chi tiết ảnh không hợp lệ (5 đầu tiên):")
        for img_path, error in issues['invalid_images'][:5]:
            print(f"      {Path(img_path).name}: {error}")
    
    # Phân bố chi tiết
    image_counts = [info['total'] for info in issues['class_details'].values()]
    if image_counts:
        print(f"\n📈 Phân bố số ảnh/class:")
        print(f"   Min: {min(image_counts)}")
        print(f"   Max: {max(image_counts)}")
        print(f"   Mean: {sum(image_counts) / len(image_counts):.2f}")
        print(f"   Median: {sorted(image_counts)[len(image_counts)//2]}")
        
        # Tính imbalance ratio
        max_count = max(image_counts)
        min_count = min([c for c in image_counts if c > 0])
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        print(f"   Imbalance Ratio (Max/Min): {imbalance_ratio:.2f}x")
        
        # Top 10 classes có ít ảnh nhất
        sorted_classes = sorted(issues['class_details'].items(), key=lambda x: x[1]['total'])
        print(f"\n   Top 10 classes có ít ảnh nhất:")
        for class_id, info in sorted_classes[:10]:
            print(f"      Class {class_id}: {info['total']} ảnh")
        
        # Top 10 classes có nhiều ảnh nhất
        sorted_classes_desc = sorted(issues['class_details'].items(), key=lambda x: x[1]['total'], reverse=True)
        print(f"\n   Top 10 classes có nhiều ảnh nhất:")
        for class_id, info in sorted_classes_desc[:10]:
            print(f"      Class {class_id}: {info['total']} ảnh")
    
    return issues


def compare_datasets(vitro_issues, situ_issues):
    """So sánh 2 datasets"""
    print("\n" + "="*70)
    print("🔄 SO SÁNH IN-VITRO vs IN-SITU")
    print("="*70)
    
    vitro_classes = set(vitro_issues['class_details'].keys())
    situ_classes = set(situ_issues['class_details'].keys())
    
    common_classes = vitro_classes & situ_classes
    only_vitro = vitro_classes - situ_classes
    only_situ = situ_classes - vitro_classes
    
    print(f"\n📋 Classes:")
    print(f"   In-vitro: {len(vitro_classes)} classes")
    print(f"   In-situ: {len(situ_classes)} classes")
    print(f"   Common: {len(common_classes)} classes")
    print(f"   Only in-vitro: {len(only_vitro)} classes")
    print(f"   Only in-situ: {len(only_situ)} classes")
    
    if only_vitro:
        print(f"\n   Classes chỉ có trong in-vitro: {sorted(list(only_vitro))[:10]}...")
    if only_situ:
        print(f"   Classes chỉ có trong in-situ: {sorted(list(only_situ))[:10]}...")
    
    # So sánh số lượng ảnh cho các class chung
    print(f"\n📊 So sánh số ảnh cho các class chung:")
    common_comparison = []
    for class_id in sorted(common_classes):
        vitro_count = vitro_issues['class_details'][class_id]['total']
        situ_count = situ_issues['class_details'][class_id]['total']
        ratio = situ_count / vitro_count if vitro_count > 0 else float('inf')
        common_comparison.append((class_id, vitro_count, situ_count, ratio))
    
    # Sắp xếp theo ratio (situ/vitro)
    common_comparison.sort(key=lambda x: x[3], reverse=True)
    
    print(f"\n   Top 10 classes có chênh lệch lớn nhất (situ/vitro):")
    for class_id, vitro_count, situ_count, ratio in common_comparison[:10]:
        print(f"      Class {class_id}: vitro={vitro_count}, situ={situ_count}, ratio={ratio:.2f}x")
    
    print(f"\n   Top 10 classes có chênh lệch nhỏ nhất (situ/vitro):")
    for class_id, vitro_count, situ_count, ratio in common_comparison[-10:]:
        print(f"      Class {class_id}: vitro={vitro_count}, situ={situ_count}, ratio={ratio:.2f}x")


def main():
    """Main function"""
    project_root = Path(__file__).parent.parent
    vitro_dir = project_root / "data" / "processing" / "vitro"
    situ_dir = project_root / "data" / "processing" / "inSitu" / "inSitu"
    
    print("\n" + "="*70)
    print("🔍 PHÂN TÍCH DỮ LIỆU CHI TIẾT")
    print("="*70)
    print("\n⚠️  Lưu ý: Script này sẽ kiểm tra từng ảnh, có thể mất thời gian...")
    print("   Hãy kiên nhẫn, chúng ta cần hiểu rõ dữ liệu trước khi xử lý!\n")
    
    vitro_issues = None
    situ_issues = None
    
    if vitro_dir.exists():
        vitro_issues = analyze_vitro_detailed(vitro_dir)
    else:
        print(f"❌ Không tìm thấy: {vitro_dir}")
    
    if situ_dir.exists():
        situ_issues = analyze_situ_detailed(situ_dir)
    else:
        print(f"❌ Không tìm thấy: {situ_dir}")
    
    if vitro_issues and situ_issues:
        compare_datasets(vitro_issues, situ_issues)
    
    print("\n" + "="*70)
    print("✅ Hoàn thành phân tích!")
    print("="*70)
    print("\n💡 Bước tiếp theo:")
    print("   1. Xem lại các vấn đề phát hiện ở trên")
    print("   2. Quyết định cách xử lý từng vấn đề")
    print("   3. Tạo script xử lý dữ liệu nếu cần")
    print("   4. Sau đó mới bắt đầu training\n")


if __name__ == '__main__':
    main()

