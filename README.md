# StoreItemDetection

Dự án phát hiện và nhận dạng sản phẩm trong cửa hàng sử dụng Machine Learning và Computer Vision. Hệ thống này được thiết kế để nhận dạng các sản phẩm trong môi trường cửa hàng thực tế (in-situ) và trong điều kiện phòng thí nghiệm (in-vitro).

## 📋 Mục lục

- [Tổng quan dự án](#tổng-quan-dự-án)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Cài đặt](#cài-đặt)
- [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)
- [API Documentation](#api-documentation)
- [Notebooks](#notebooks)
- [Cấu trúc dữ liệu](#cấu-trúc-dữ-liệu)
- [Models](#models)
- [Web UI](#web-ui)
- [Đóng góp](#đóng-góp)

## 🎯 Tổng quan dự án

StoreItemDetection là một hệ thống AI tiên tiến được phát triển để:

- **Phát hiện sản phẩm**: Nhận dạng và định vị các sản phẩm trong hình ảnh cửa hàng
- **Phân loại sản phẩm**: Phân loại các loại sản phẩm khác nhau
- **Embedding vectors**: Tạo ra các vector đặc trưng cho việc tìm kiếm và so sánh sản phẩm
- **API service**: Cung cấp REST API để tích hợp vào các hệ thống khác
- **Web interface**: Giao diện web để test và demo

## 📁 Cấu trúc dự án

```
StoreItemDetection/
├── 📂 api/                     # REST API service
│   ├── __init__.py
│   └── main.py                 # FastAPI application chính
│
├── 📂 data/                    # Dữ liệu training và testing
│   ├── 📂 raw/                 # Dữ liệu thô
│   │   ├── 📂 inSitu/          # Dữ liệu môi trường thực tế (11,434 files)
│   │   │   └── 📂 inSitu/      # Images (.png) và labels (.txt)
│   │   └── 📂 inVitro/         # Dữ liệu phòng thí nghiệm (2,181 files)
│   │       └── 📂 inVitro/     # Images (.png, .jpg) và labels (.txt)
│   ├── 📂 processing/          # Dữ liệu đã xử lý
│   └── 📂 test/                # Dữ liệu test
│
├── 📂 embeddings/              # Vector embeddings
│   └── metadata.json           # Metadata của embeddings
│
├── 📂 models/                  # Các model ML/DL
│   ├── 📂 backbone/            # Base models (ResNet, EfficientNet, etc.)
│   ├── 📂 situ_finetune/       # Models fine-tuned cho in-situ data
│   └── 📂 vitro_pretrain/      # Models pre-trained trên in-vitro data
│
├── 📂 notebooks/               # Jupyter notebooks
│   ├── finetune_situ.ipynb     # Fine-tuning cho dữ liệu in-situ
│   ├── test_embeddings.ipynb   # Test và đánh giá embeddings
│   └── train_vitro.ipynb       # Training trên dữ liệu in-vitro
│
├── 📂 qdrant_client/           # Vector database client
│   └── __init__.py
│
├── 📂 utils/                   # Utility functions
│   └── __init__.py
│
├── 📂 web_ui/                  # Web interface
│   ├── index.html              # Trang chủ
│   ├── 📂 src/
│   │   ├── 📂 components/      # React/Vue components
│   │   ├── 📂 pages/           # Các trang web
│   │   └── 📂 services/        # API services
│   └── 📂 static/              # Static files (CSS, JS, images)
│
├── 📄 requirements.txt         # Python dependencies
├── 📄 setup.py                # Package setup
└── 📄 README.md               # Documentation này
```

## 💻 Yêu cầu hệ thống

### Phần cứng khuyến nghị:
- **CPU**: Intel i5+ hoặc AMD Ryzen 5+
- **RAM**: 8GB+ (16GB khuyến nghị cho training)
- **GPU**: NVIDIA GTX 1060+ (cho training deep learning)
- **Storage**: 10GB+ dung lượng trống

### Phần mềm:
- **Python**: 3.8+
- **CUDA**: 11.0+ (nếu sử dụng GPU)
- **Git**: Để clone repository

## 🚀 Cài đặt

### 1. Clone repository

```bash
git clone https://github.com/hzcay/StoreItemDetection.git
cd StoreItemDetection
```

### 2. Tạo virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 3. Cài đặt package

```bash
pip install .
```

**Happy Coding! 🚀**
