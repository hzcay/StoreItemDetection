"""
Script làm nét và cải thiện chất lượng ảnh trong data situ
Xử lý trước khi training
"""
import os
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
from tqdm import tqdm


def sharpen_image_pil(image: Image.Image, factor: float = 2.0) -> Image.Image:
    """
    Làm nét ảnh bằng PIL (Unsharp Mask)
    
    Args:
        image: PIL Image
        factor: Độ nét (1.0 = không đổi, 2.0 = nét hơn)
    
    Returns:
        PIL Image đã được làm nét
    """
    # Unsharp mask filter
    image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
    
    # Sharpness enhancer
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(factor)
    
    return image


def enhance_contrast_pil(image: Image.Image, factor: float = 1.2) -> Image.Image:
    """
    Tăng độ tương phản
    
    Args:
        image: PIL Image
        factor: Độ tương phản (1.0 = không đổi, >1.0 = tăng)
    
    Returns:
        PIL Image đã được tăng tương phản
    """
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)


def denoise_and_sharpen_cv2(image_path: Path, output_path: Path, strength: str = 'balanced'):
    """
    Làm nét và giảm nhiễu bằng OpenCV (phiên bản cân bằng - giữ chi tiết)
    
    Args:
        image_path: Đường dẫn ảnh gốc
        output_path: Đường dẫn ảnh output
        strength: 'light', 'balanced', 'medium', 'strong'
    """
    # Đọc ảnh
    img = cv2.imread(str(image_path))
    if img is None:
        return False
    
    # Chuyển sang RGB (OpenCV dùng BGR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Cấu hình theo strength - cân bằng giữa làm nét và giữ chi tiết
    if strength == 'light':
        denoise_h = 3
        denoise_hcolor = 3
        sharp_weight1, sharp_weight2 = 1.2, -0.2
        clahe_limit = 1.3
        unsharp_radius = 1.0
        use_bilateral = True
    elif strength == 'balanced':  # Mặc định - cân bằng tốt nhất
        denoise_h = 5
        denoise_hcolor = 5
        sharp_weight1, sharp_weight2 = 1.4, -0.4
        clahe_limit = 1.8
        unsharp_radius = 1.5
        use_bilateral = True
    elif strength == 'medium':
        denoise_h = 8
        denoise_hcolor = 8
        sharp_weight1, sharp_weight2 = 1.6, -0.5
        clahe_limit = 2.2
        unsharp_radius = 2.0
        use_bilateral = False
    else:  # strong
        denoise_h = 10
        denoise_hcolor = 10
        sharp_weight1, sharp_weight2 = 1.8, -0.6
        clahe_limit = 2.5
        unsharp_radius = 2.0
        use_bilateral = False
    
    # Giảm nhiễu NHẸ để giữ chi tiết (hoặc dùng bilateral filter để giữ edge)
    if use_bilateral:
        # Bilateral filter giữ edge tốt hơn
        img_denoised = cv2.bilateralFilter(img_rgb, 5, 50, 50)
    else:
        # Non-local means - nhẹ hơn
        img_denoised = cv2.fastNlMeansDenoisingColored(img_rgb, None, denoise_h, denoise_hcolor, 7, 21)
    
    # Làm nét bằng Unsharp Mask (vừa phải để không mất chi tiết)
    gaussian = cv2.GaussianBlur(img_denoised, (0, 0), unsharp_radius)
    img_sharpened = cv2.addWeighted(img_denoised, sharp_weight1, gaussian, sharp_weight2, 0)
    
    # Tăng độ tương phản (CLAHE) - vừa phải để không làm mất chi tiết
    lab = cv2.cvtColor(img_sharpened, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clahe_limit, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img_enhanced = cv2.merge([l, a, b])
    img_enhanced = cv2.cvtColor(img_enhanced, cv2.COLOR_LAB2RGB)
    
    # Làm nét nhẹ lần cuối (rất nhẹ để không làm mất chi tiết)
    gaussian_final = cv2.GaussianBlur(img_enhanced, (0, 0), 0.8)
    img_final = cv2.addWeighted(img_enhanced, 1.1, gaussian_final, -0.1, 0)
    
    # Đảm bảo giá trị trong range [0, 255]
    img_final = np.clip(img_final, 0, 255).astype(np.uint8)
    
    # Lưu ảnh
    img_bgr = cv2.cvtColor(img_final, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    return True


def enhance_image_simple(image_path: Path, output_path: Path, method: str = 'cv2', strength: str = 'balanced'):
    """
    Xử lý ảnh đơn giản
    
    Args:
        image_path: Đường dẫn ảnh gốc
        output_path: Đường dẫn ảnh output
        method: 'pil' hoặc 'cv2'
        strength: 'light', 'balanced', 'medium', 'strong'
    """
    try:
        if method == 'cv2':
            return denoise_and_sharpen_cv2(image_path, output_path, strength=strength)
        else:
            # Method PIL (nhanh hơn nhưng chất lượng thấp hơn)
            image = Image.open(image_path).convert('RGB')
            
            # Làm nét
            image = sharpen_image_pil(image, factor=1.5)
            
            # Tăng tương phản
            image = enhance_contrast_pil(image, factor=1.1)
            
            # Lưu
            image.save(output_path, quality=95)
            return True
    except Exception as e:
        print(f"   ❌ Lỗi khi xử lý {image_path.name}: {e}")
        return False


def process_situ_data(
    data_dir: str,
    output_dir: str = None,
    method: str = 'cv2',
    strength: str = 'balanced',
    dry_run: bool = False
):
    """
    Xử lý tất cả ảnh trong data situ
    
    Args:
        data_dir: Đường dẫn đến thư mục in-situ
        output_dir: Thư mục output (None = dùng data/processing/inSitu/inSitu)
        method: 'cv2' (chất lượng tốt) hoặc 'pil' (nhanh)
        strength: 'light', 'balanced', 'medium', 'strong'
        dry_run: Chỉ đếm, không xử lý
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"❌ Không tìm thấy thư mục: {data_dir}")
        return
    
    # Nếu không có output_dir, dùng processing folder
    if output_dir is None:
        # Tạo đường dẫn processing: data/processing/inSitu/inSitu
        project_root = data_path.parent.parent.parent  # Từ inSitu/inSitu -> data
        output_dir = str(project_root / "processing" / "inSitu" / "inSitu")
        print(f"📁 Sử dụng thư mục processing mặc định: {output_dir}")
    
    output_path = Path(output_dir)
    
    # Tìm tất cả ảnh PNG trong video/
    png_files = []
    for class_dir in sorted(data_path.iterdir()):
        if not class_dir.is_dir():
            continue
        
        try:
            class_id = int(class_dir.name)
        except ValueError:
            continue
        
        video_dir = class_dir / "video"
        if video_dir.exists():
            for png_file in video_dir.glob("*.png"):
                if png_file.name.lower() != "thumbs.db":
                    png_files.append(png_file)
    
    print(f"\n📊 Thống kê:")
    print(f"   Tổng số ảnh PNG: {len(png_files)}")
    print(f"   Method: {method.upper()}")
    print(f"   Strength: {strength.upper()}")
    print(f"   Output: {output_dir}")
    print(f"   Mode: {'DRY RUN' if dry_run else 'XỬ LÝ THẬT'}\n")
    
    if dry_run:
        print("⚠️  DRY RUN: Không xử lý ảnh nào")
        print("   Để xử lý thật, chạy lại với --execute")
        return
    
    # Xử lý ảnh
    success_count = 0
    failed_count = 0
    
    print(f"🖼️  Đang xử lý ảnh...")
    for png_file in tqdm(png_files, desc="Enhancing"):
        # Tạo đường dẫn output giữ nguyên cấu trúc thư mục
        rel_path = png_file.relative_to(data_path)
        output_file = output_path / rel_path
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if enhance_image_simple(png_file, output_file, method=method, strength=strength):
            success_count += 1
        else:
            failed_count += 1
    
    print(f"\n✅ Hoàn thành!")
    print(f"   Thành công: {success_count} ảnh")
    print(f"   Ảnh đã lưu vào: {output_dir}")
    if failed_count > 0:
        print(f"   Lỗi: {failed_count} ảnh")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Làm nét và cải thiện chất lượng ảnh trong data situ'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/raw/inSitu/inSitu',
        help='Đường dẫn đến thư mục in-situ data'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Thư mục output (None = dùng data/processing/inSitu/inSitu)'
    )
    parser.add_argument(
        '--method',
        type=str,
        choices=['cv2', 'pil'],
        default='cv2',
        help='Method xử lý: cv2 (chất lượng tốt) hoặc pil (nhanh)'
    )
    parser.add_argument(
        '--strength',
        type=str,
        choices=['light', 'balanced', 'medium', 'strong'],
        default='balanced',
        help='Độ mạnh xử lý: light, balanced (giữ chi tiết - khuyến nghị), medium, strong (mặc định: balanced)'
    )
    parser.add_argument(
        '--execute',
        action='store_true',
        help='Thực sự xử lý ảnh (mặc định chỉ dry-run)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("🖼️  ENHANCE SITU IMAGES - Làm nét và cải thiện chất lượng")
    print("="*70)
    
    process_situ_data(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        method=args.method,
        strength=args.strength,
        dry_run=not args.execute
    )
    
    print("\n" + "="*70)
    print("✅ Hoàn thành!")
    print("="*70)
    print("\n💡 Lưu ý:")
    print("   - Method cv2: Chất lượng tốt hơn (denoising + sharpening + CLAHE)")
    print("   - Method pil: Nhanh hơn nhưng chất lượng thấp hơn")
    print("   - Ảnh đã xử lý được lưu vào data/processing/inSitu/inSitu (giữ nguyên cấu trúc)")
    print("   - Ảnh gốc không bị thay đổi")


if __name__ == '__main__':
    main()

