import os
import shutil
from sklearn.model_selection import train_test_split

root_folder = r"D:/digital-image-processing/StoreItemDetection/data/raw/inVitro/inVitro"
output_folder = r"D:/digital-image-processing/StoreItemDetection/data/yolo_dataset"

# Tạo cấu trúc YOLO
for f in ["images/train", "images/val", "labels/train", "labels/val"]:
    os.makedirs(os.path.join(output_folder, f), exist_ok=True)

all_pairs = []  # list các (img_path, label_path)

# Gom dữ liệu
for folder_name in os.listdir(root_folder):
    if not folder_name.isdigit():
        continue

    print(f"📂 Đang gom dữ liệu từ folder: {folder_name}")

    jpeg_dir = os.path.join(root_folder, folder_name, "web", "JPEG")
    label_dir = os.path.join(root_folder, folder_name, "web", "LABEL")

    if not os.path.exists(jpeg_dir) or not os.path.exists(label_dir):
        print(f"⚠️ Thiếu JPEG hoặc LABEL → bỏ qua {folder_name}")
        continue

    for img_file in os.listdir(jpeg_dir):
        if img_file.endswith(".jpg"):
            img_path = os.path.join(jpeg_dir, img_file)
            label_path = os.path.join(label_dir, img_file.replace(".jpg", ".txt"))

            if os.path.exists(label_path):
                all_pairs.append((img_path, label_path))
            else:
                print(f"⚠️ Không tìm thấy label cho {img_file}")

print(f"👉 Tổng số ảnh + label hợp lệ: {len(all_pairs)}")


# CHIA TRAIN / VAL
pairs_train, pairs_val = train_test_split(
    all_pairs, test_size=0.2, random_state=42
)

# ---- GLOBAL COUNTER ----
global_id = 1     # bắt đầu từ 1
def get_new_name():
    global global_id
    new_name = f"{global_id:05d}"  # vd: 00001, 00002
    global_id += 1
    return new_name


# COPY + ĐỔI TÊN KHÔNG TRÙNG
def copy_with_new_name(pairs, split):
    for img_src, lbl_src in pairs:
        new_id = get_new_name()
        new_img_name = new_id + ".jpg"
        new_lbl_name = new_id + ".txt"

        img_dst = os.path.join(output_folder, f"images/{split}", new_img_name)
        lbl_dst = os.path.join(output_folder, f"labels/{split}", new_lbl_name)

        shutil.copy(img_src, img_dst)
        shutil.copy(lbl_src, lbl_dst)


print("📦 Copy TRAIN...")
copy_with_new_name(pairs_train, "train")

print("📦 Copy VAL...")
copy_with_new_name(pairs_val, "val")

print("🎉 DONE! Dataset YOLO đã được chuẩn hóa tên và không trùng file.")
