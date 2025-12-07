"""
Script test làm nét ảnh trên 1 ảnh để kiểm tra thuật toán
Hiển thị kết quả so sánh trước/sau
"""
import sys
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
import matplotlib.pyplot as plt


def enhance_image_cv2(image_path: Path, strength: str = 'balanced') -> np.ndarray:
    """
    Làm nét và cải thiện ảnh bằng OpenCV (phiên bản cân bằng - giữ chi tiết)
    
    Args:
        image_path: Đường dẫn ảnh
        strength: 'light', 'balanced', 'medium', 'strong'
    
    Returns:
        numpy array của ảnh đã xử lý (RGB)
    """
    # Đọc ảnh
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Không thể đọc ảnh: {image_path}")
    
    # Chuyển sang RGB
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
    
    return img_final


def enhance_image_pil(image_path: Path) -> Image.Image:
    """
    Làm nét và cải thiện ảnh bằng PIL
    
    Returns:
        PIL Image đã xử lý
    """
    image = Image.open(image_path).convert('RGB')
    
    # Làm nét
    image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.5)
    
    # Tăng tương phản
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.1)
    
    return image


def compare_images(original_path: Path, method: str = 'cv2', strength: str = 'strong', save_output: bool = True):
    """
    So sánh ảnh gốc và ảnh đã xử lý
    
    Args:
        original_path: Đường dẫn ảnh gốc
        method: 'cv2' hoặc 'pil'
        save_output: Lưu ảnh output không
    """
    print(f"\n{'='*70}")
    print(f"🧪 TEST ENHANCE IMAGE")
    print(f"{'='*70}")
    print(f"Ảnh gốc: {original_path}")
    print(f"Method: {method.upper()}")
    print(f"Strength: {strength.upper()}\n")
    
    # Đọc ảnh gốc
    try:
        original_img = Image.open(original_path).convert('RGB')
        original_array = np.array(original_img)
    except Exception as e:
        print(f"❌ Lỗi đọc ảnh gốc: {e}")
        return
    
    # Xử lý ảnh
    print("🖼️  Đang xử lý ảnh...")
    try:
        if method == 'cv2':
            enhanced_array = enhance_image_cv2(original_path, strength=strength)
            enhanced_img = Image.fromarray(enhanced_array)
        else:
            enhanced_img = enhance_image_pil(original_path)
            enhanced_array = np.array(enhanced_img)
    except Exception as e:
        print(f"❌ Lỗi xử lý ảnh: {e}")
        return
    
    print("✅ Xử lý xong!\n")
    
    # Lưu ảnh output nếu cần
    if save_output:
        output_path = original_path.parent / f"{original_path.stem}_enhanced{original_path.suffix}"
        if method == 'cv2':
            cv2.imwrite(str(output_path), cv2.cvtColor(enhanced_array, cv2.COLOR_RGB2BGR), 
                       [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            enhanced_img.save(output_path, quality=95)
        print(f"💾 Đã lưu ảnh output: {output_path}\n")
    
    # Hiển thị so sánh
    print("📊 So sánh:")
    print(f"   Kích thước gốc: {original_img.size}")
    print(f"   Kích thước sau: {enhanced_img.size}")
    
    # Tính toán một số metrics
    original_gray = cv2.cvtColor(original_array, cv2.COLOR_RGB2GRAY) if method == 'cv2' else np.array(original_img.convert('L'))
    enhanced_gray = cv2.cvtColor(enhanced_array, cv2.COLOR_RGB2GRAY) if method == 'cv2' else np.array(enhanced_img.convert('L'))
    
    # Laplacian variance (đo độ nét)
    laplacian_original = cv2.Laplacian(original_gray, cv2.CV_64F).var()
    laplacian_enhanced = cv2.Laplacian(enhanced_gray, cv2.CV_64F).var()
    
    print(f"   Độ nét gốc (Laplacian variance): {laplacian_original:.2f}")
    print(f"   Độ nét sau: {laplacian_enhanced:.2f}")
    print(f"   Cải thiện: {((laplacian_enhanced / laplacian_original - 1) * 100):.1f}%")
    
    # Hiển thị ảnh (nếu có matplotlib)
    try:
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        
        axes[0].imshow(original_array)
        axes[0].set_title('Ảnh Gốc', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(enhanced_array)
        axes[1].set_title('Ảnh Đã Xử Lý', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        # Lưu comparison image
        comparison_path = original_path.parent / f"{original_path.stem}_comparison.png"
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        print(f"💾 Đã lưu ảnh so sánh: {comparison_path}")
        
        plt.show()
        print("\n✅ Đã hiển thị ảnh so sánh!")
        
    except Exception as e:
        print(f"\n⚠️  Không thể hiển thị ảnh (có thể thiếu matplotlib hoặc display): {e}")
        print("   Nhưng ảnh đã được lưu, bạn có thể xem bằng image viewer")
    
    print(f"\n{'='*70}")
    print("✅ Hoàn thành test!")
    print(f"{'='*70}\n")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Test làm nét ảnh trên 1 ảnh để kiểm tra thuật toán'
    )
    parser.add_argument(
        'image_path',
        type=str,
        help='Đường dẫn đến ảnh cần test'
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
        help='Độ mạnh xử lý: light, balanced (giữ chi tiết), medium, strong (mặc định: balanced)'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Không lưu ảnh output'
    )
    
    args = parser.parse_args()
    
    image_path = Path(args.image_path)
    
    if not image_path.exists():
        print(f"❌ Không tìm thấy ảnh: {image_path}")
        print("\n💡 Ví dụ sử dụng:")
        print("   python utils/test_enhance_single_image.py data/raw/inVitro/inVitro/1/web/PNG/web1.png")
        print("   python utils/test_enhance_single_image.py data/raw/inVitro/inVitro/1/web/PNG/web1.png --method pil")
        return
    
    compare_images(
        original_path=image_path,
        method=args.method,
        strength=args.strength,
        save_output=not args.no_save
    )


if __name__ == '__main__':
    main()

