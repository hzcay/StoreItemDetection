from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# 1️⃣ Load model YOLO pretrained (dùng để phát hiện vật thể)
model = YOLO(r"D:\digital-image-processing\StoreItemDetection\test_yolo\best.pt")  # hoặc yolov11n.pt

# 2️⃣ Load ảnh
image_path = r"D:\digital-image-processing\StoreItemDetection\test_yolo\products.jpeg"
img = cv2.imread(image_path)

# 3️⃣ Chạy detect
results = model(img)

# 4️⃣ Vẽ bounding box
annotated = results[0].plot()   # YOLO tự vẽ bbox + label + confidence

# 5️⃣ Hiển thị ảnh
plt.figure(figsize=(8,8))
plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()