import os
import cv2
import numpy as np

root_folder = r"D:/digital-image-processing/StoreItemDetection/data/raw/inVitro/inVitro"

def extract_number(name):
    """Lấy số cuối cùng trong tên file"""
    digits = ''.join([c for c in name if c.isdigit()])
    return int(digits) if digits else None

for folder_name in os.listdir(root_folder):
    if not folder_name.isdigit():
        continue

    print(f"\n📂 Đang xử lý folder số: {folder_name}")

    current_folder = os.path.join(root_folder, folder_name, "web")
    jpeg_folder = os.path.join(current_folder, "JPEG")
    masks_folder = os.path.join(current_folder, "masks")
    label_folder = os.path.join(current_folder, "LABEL")

    os.makedirs(label_folder, exist_ok=True)

    # Load mask mapping
    mask_dict = {}
    for mask_name in os.listdir(masks_folder):
        if mask_name.endswith(".png"):
            num = extract_number(mask_name)
            mask_dict[num] = mask_name

    for image_file in os.listdir(jpeg_folder):
        if not image_file.endswith(".jpg"):
            continue

        img_num = extract_number(image_file)
        if img_num not in mask_dict:
            print(f"❌ Không tìm thấy mask cho {image_file}")
            continue

        image_path = os.path.join(jpeg_folder, image_file)
        mask_path = os.path.join(masks_folder, mask_dict[img_num])

        img = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            print(f"⚠️ Lỗi đọc file: {image_file}")
            continue

        h, w = mask.shape

        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            print(f"⚠️ Mask rỗng cho {image_file}")
            continue

        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()

        cx = (xmin + xmax) / 2 / w
        cy = (ymin + ymax) / 2 / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h

        label_path = os.path.join(label_folder, os.path.splitext(image_file)[0] + ".txt")
        with open(label_path, "w") as f:
            f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        print(f"✅ Saved label: {label_path}")
