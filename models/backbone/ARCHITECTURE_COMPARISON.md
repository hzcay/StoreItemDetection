# So sánh kiến trúc ResMobileNet: V1 vs V2

## 📊 Tổng quan

| Aspect | **ResMobileNet V1** (Hiện tại) | **ResMobileNet V2** (Đề xuất) |
|--------|--------------------------------|-------------------------------|
| **Stem** | MobileNet (Conv3x3, stride=2) | ResNet (Conv7x7 + MaxPool) |
| **Mid** | MobileNet Inverted Residuals | MobileNet Inverted Residuals |
| **Tail** | ResNet Bottleneck (6 blocks) | ResNet Bottleneck (6 blocks) |
| **Head** | AvgPool + FC + BN + ReLU + Norm | AvgPool + FC + BN + ReLU + Norm |

---

## 🔍 Phân tích chi tiết

### 1. **Stem Layer (Low-level Feature Extraction)**

#### V1: MobileNet Stem
```python
Conv2dNormActivation(3, 16, kernel_size=3, stride=2, ...)
```
- **Kích thước kernel**: 3×3 (nhỏ, nhẹ)
- **Stride**: 2 (downsample 1 lần)
- **Output size**: `[B, 16, H/2, W/2]`
- **FLOPs**: ~0.1M (rất nhẹ)
- **Ưu điểm**: 
  - ✅ Rất nhanh, ít tham số
  - ✅ Phù hợp mobile/edge devices
- **Nhược điểm**:
  - ❌ Receptive field nhỏ (3×3) → bắt ít context
  - ❌ Feature extraction yếu hơn ở low-level

#### V2: ResNet Stem
```python
Conv2d(3, 64, kernel_size=7, stride=2, ...)
BN + ReLU
MaxPool2d(kernel_size=3, stride=2, ...)
```
- **Kích thước kernel**: 7×7 (lớn, mạnh)
- **Stride**: 2 + MaxPool stride=2 (downsample 2 lần)
- **Output size**: `[B, 64, H/4, W/4]`
- **FLOPs**: ~0.5M (nặng hơn 5x)
- **Ưu điểm**:
  - ✅ Receptive field lớn (7×7) → bắt nhiều context hơn
  - ✅ Feature extraction mạnh ở low-level (edges, textures)
  - ✅ MaxPool giúp robust hơn với noise
- **Nhược điểm**:
  - ❌ Nặng hơn, chậm hơn
  - ❌ Nhiều tham số hơn

**Kết luận**: V2 mạnh hơn ở low-level, nhưng đánh đổi tốc độ.

---

### 2. **Mid Blocks (Mid-level Feature Processing)**

**Cả hai đều dùng MobileNet Inverted Residuals** → Không khác biệt.

- 15 blocks với depthwise separable convolution
- Hiệu quả về compute
- Tốt cho mid-level features

---

### 3. **Tail Blocks (High-level Feature Refinement)**

**Cả hai đều dùng 6 ResNet Bottleneck blocks** → Không khác biệt.

- Refinement mạnh cho high-level semantics
- Giữ nguyên số kênh qua các block

---

### 4. **Embedding Head**

**Cả hai giống hệt nhau** → Không khác biệt.

---

## 📈 So sánh Performance (Dự đoán)

| Metric | V1 (MobileNet Stem) | V2 (ResNet Stem) |
|--------|---------------------|------------------|
| **Inference Speed** | ⚡⚡⚡ Nhanh hơn | ⚡⚡ Chậm hơn ~10-15% |
| **Model Size** | 📦 Nhỏ hơn (~5MB) | 📦 Lớn hơn (~7MB) |
| **FLOPs** | ~300M | ~350M |
| **Accuracy (dự đoán)** | Tốt | Tốt hơn ~1-2% |
| **Low-level Features** | Trung bình | Mạnh hơn |
| **Mobile-friendly** | ✅ Rất tốt | ⚠️ Tốt (nhưng nặng hơn) |

---

## 🎯 Khi nào dùng V1 vs V2?

### Dùng **ResMobileNet V1** (MobileNet Stem) khi:
- ✅ **Ưu tiên tốc độ**: Real-time inference, mobile/edge devices
- ✅ **Tài nguyên hạn chế**: GPU yếu, RAM ít
- ✅ **Dataset đơn giản**: Không cần quá nhiều low-level detail
- ✅ **Latency quan trọng**: Cần inference < 50ms

### Dùng **ResMobileNet V2** (ResNet Stem) khi:
- ✅ **Ưu tiên accuracy**: Muốn tối đa hóa độ chính xác
- ✅ **Dataset phức tạp**: Nhiều texture, pattern phức tạp
- ✅ **GPU khỏe**: Có đủ tài nguyên để trade-off tốc độ
- ✅ **Offline processing**: Không cần real-time

---

## 🔬 Thử nghiệm đề xuất

Để quyết định chính xác, bạn nên:

1. **Train cả 2 kiến trúc** trên cùng dataset `vitro`
2. **So sánh metrics**:
   - Precision@1, Recall@5, mAP
   - Inference time (ms)
   - Model size (MB)
   - FLOPs
3. **Visualize embeddings** (t-SNE) để xem clustering quality

---

## 💡 Kết luận

- **V1 (hiện tại)**: Cân bằng tốt giữa tốc độ và accuracy, phù hợp production
- **V2 (mới)**: Mạnh hơn ở low-level, có thể tốt hơn 1-2% accuracy nhưng đánh đổi tốc độ

**Khuyến nghị**: Thử cả 2 và chọn dựa trên kết quả thực tế trên dataset của bạn!

