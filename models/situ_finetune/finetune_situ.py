"""
Fine-tuning script for ResMobileNetV2 on in-situ dataset
Chiến lược: Load checkpoint từ vitro pretrain, unfreeze tất cả layers và train với LR nhỏ hơn
"""
import os
import sys
import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Subset, WeightedRandomSampler
from collections import Counter

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.backbone.ResMobileNetV2 import ResMobileNetV2, res_mobilenet_conf
from models.losses.supcon_loss import SupConLoss
class SituDataset(Dataset):
    def __init__(self, data_dir: str, transform_visual=None, transform_color=None, is_train=True):
        self.data_dir = Path(data_dir)
        # [FIX] Nhận đúng tham số transform
        self.transform_visual = transform_visual
        self.transform_color = transform_color
        self.is_train = is_train
        self.samples = []
        
        # Logic load ảnh Situ (Giữ nguyên)
        for class_dir in sorted(self.data_dir.iterdir()):
            if not class_dir.is_dir(): continue
            try:
                class_id = int(class_dir.name)
            except ValueError: continue
            
            video_dir = class_dir / "video"
            if video_dir.exists():
                for img_file in sorted(video_dir.glob("*.png")):
                    if img_file.name.lower() != "thumbs.db":
                        self.samples.append((str(img_file), class_id))
        
        print(f"   [Situ] Loaded {len(self.samples)} samples from {len(set([s[1] for s in self.samples]))} classes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.is_train:
            img_vis = self.transform_visual(image) if self.transform_visual else image
            img_col = self.transform_color(image) if self.transform_color else image
            label = label - 1
            return (img_vis, img_col), label
        else:
            img = self.transform_color(image) if self.transform_color else image
            label = label - 1
            return (img, img), label

def create_weighted_sampler(dataset):
    labels = [label for _, label in dataset.samples]
    class_counts = Counter(labels)
    
    total_samples = len(labels)
    num_classes = len(class_counts)
    
    weights = []
    for _, label in dataset.samples:
        weight = total_samples / (num_classes * class_counts[label])
        weights.append(weight)
    
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )
    
    return sampler


def get_class_weights(dataset, num_classes=None, device='cpu'):
    """
    Tính class weights dựa trên inverse frequency
    
    Args:
        dataset: SituDataset
        num_classes: Số classes (nếu None thì dùng từ dataset)
        device: Device để tạo tensor
    
    Returns:
        Tensor class weights (size = num_classes, indices từ 0 đến num_classes-1)
    """
    labels = [label for _, label in dataset.samples]
    class_counts = Counter(labels)
    
    total_samples = len(labels)
    dataset_num_classes = len(class_counts)

    if num_classes is None:
        num_classes = dataset_num_classes
    
    min_class_id = min(class_counts.keys())
    max_class_id = max(class_counts.keys())

    class_weights = torch.ones(num_classes, device=device)
    
    for class_id, count in class_counts.items():
        if 0 <= class_id < num_classes:
            weight = total_samples / (dataset_num_classes * count)
            class_weights[class_id] = weight
    
    print(f"   Dataset classes: {dataset_num_classes}, Model classes: {num_classes}")
    print(f"   Class weights shape: {class_weights.shape}")
    print(f"   Min class_id: {min_class_id}, Max class_id: {max_class_id}")
    
    return class_weights

def unfreeze_all_layers(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = True
    print("✅ Unfrozen: All layers (full fine-tuning)")

@torch.no_grad()
def compute_arcface_accuracy(embeddings, labels, arcface_head):
    """
    Precision@1 trên cosine thuần (không margin) để phản ánh retrieval.
    """
    cosine = F.linear(F.normalize(embeddings), F.normalize(arcface_head.weight))
    _, predicted = cosine.max(1)
    correct = predicted.eq(labels).sum().item()
    return correct


def load_pretrained_checkpoint(model: nn.Module, checkpoint_path: str, device: str, strict: bool = False):

    print(f"📂 Loading pretrained checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    state_dict = checkpoint['model_state_dict']
    model_state_dict = model.state_dict()
    
    if not strict:
        # Handle ArcFace head weight shape mismatch
        # [FIX] AdaptiveSubCenterArcFace có weight shape (out_features * k, in_features)
        if 'arcface_head.weight' in state_dict:
            pretrained_weight = state_dict['arcface_head.weight']
            current_weight = model_state_dict['arcface_head.weight']
            
            # Tính num_classes từ weight shape (weight shape = num_classes * k)
            # Giả sử k=3 (default của AdaptiveSubCenterArcFace)
            pretrained_k = 3  # Có thể detect từ checkpoint nếu cần
            current_k = getattr(model.arcface_head, 'k', 3)
            
            pretrained_num_classes = pretrained_weight.shape[0] // pretrained_k
            current_num_classes = current_weight.shape[0] // current_k
            
            if pretrained_num_classes != current_num_classes:
                print(f"   ⚠️  num_classes mismatch: {pretrained_num_classes} vs {current_num_classes}")
                print(f"   ⚠️  Skipping arcface_head.weight and arcface_head.m, will initialize randomly")
                state_dict.pop('arcface_head.weight', None)
                # Cũng skip margin buffer nếu có
                if 'arcface_head.m' in state_dict:
                    state_dict.pop('arcface_head.m', None)
    
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    # [FIX] Kiểm tra xem ArcFace head có load được không
    arcface_not_loaded = 'arcface_head.weight' in missing_keys or 'arcface_head.m' in missing_keys
    
    if missing_keys:
        print(f"   ⚠️  Missing keys (will use random init): {len(missing_keys)}")
        for key in missing_keys[:5]:
            print(f"      - {key}")
        if len(missing_keys) > 5:
            print(f"      ... and {len(missing_keys) - 5} more")
    
    if unexpected_keys:
        print(f"   ⚠️  Unexpected keys (ignored): {len(unexpected_keys)}")
    
    print(f"   ✅ Loaded pretrained weights from epoch {checkpoint.get('epoch', '?')}")
    print(f"   ✅ Pretrained val acc: {checkpoint.get('val_acc', 0):.2f}%")
    

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, use_amp=False):
    model.train()
    supcon_lambda = getattr(model, "supcon_lambda", 0.0)
    supcon_criterion = getattr(model, "supcon_criterion", None)
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
    
    for batch_data, labels in pbar:
        # [FIX] Unpack tuple ảnh từ Dataset
        # Dataset trả về: ((img_vis, img_col), label)
        if isinstance(batch_data, (list, tuple)):
            images_vis, images_col = batch_data
        else:
            images_vis = images_col = batch_data # Fallback nếu lỗi
            
        images_vis = images_vis.to(device, non_blocking=True) # Ảnh phá màu
        images_col = images_col.to(device, non_blocking=True) # Ảnh màu chuẩn
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                # [FIX] Truyền cả 2 loại ảnh vào model
                logits, visual_emb, color_emb = model(images_vis, labels, x_color=images_col)
                
                loss_arc = criterion(logits, labels)
                
                # SupCon học trên Color Embedding (từ ảnh màu chuẩn) -> Vẫn học được màu đúng!
                if supcon_lambda > 0 and supcon_criterion is not None:
                    loss_sup = supcon_criterion(color_emb, labels)
                else:
                    loss_sup = loss_arc.new_tensor(0.0)

                loss = loss_arc + (supcon_lambda * loss_sup)

            if not torch.isfinite(loss):
                optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            if not torch.isfinite(torch.tensor(grad_norm, device='cpu')):
                optimizer.zero_grad()
                continue 

             # --- BACKWARD ---
            scaler.scale(loss).backward()
            
            # [FIX] Unscale để tính norm
            scaler.unscale_(optimizer)
            
            # Clip gradient
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # [FIX QUAN TRỌNG] Check Gradient NaN
            if not torch.isfinite(torch.tensor(grad_norm, device='cpu')):
                if epoch == 0 and total < 32:
                    print(f"   ⚠️  Skipping step (Grad NaN)")
                
                optimizer.zero_grad()
                # [BẮT BUỘC] Phải gọi update() để reset scaler state dù không step()
                scaler.update() 
                continue 
            
            scaler.step(optimizer)
            scaler.update()
        else:
            # FP32 Mode (Code tương tự)
            logits, visual_emb, color_emb = model(images_vis, labels, x_color=images_col)
            loss_arc = criterion(logits, labels)
            loss_sup = supcon_criterion(color_emb, labels) if (supcon_lambda > 0) else 0.0
            loss = loss_arc + (supcon_lambda * loss_sup)
            
            if not torch.isfinite(loss): continue
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if not torch.isfinite(torch.tensor(grad_norm, device='cpu')): continue
            optimizer.step()

        running_loss += loss.item()
        total += labels.size(0)
        with torch.no_grad():
            pred = logits.argmax(dim=1)
            correct += pred.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    epoch_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0
    epoch_acc = 100. * correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc

@torch.no_grad()
def validate(model, val_loader, criterion, device, epoch: int = 0, use_amp=False):
    """
    Validate model với cơ chế tính Accuracy có Margin y hệt lúc Train
    để theo dõi chính xác độ hội tụ trên tập Vitro/Situ.
    """
    model.eval()
    supcon_lambda = getattr(model, "supcon_lambda", 0.0)
    supcon_criterion = getattr(model, "supcon_criterion", None)
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False)
    
    for batch_data, labels in pbar:
        # 1. Unpack dữ liệu: Dataset trả về ((img_vis, img_col), label)
        if isinstance(batch_data, (list, tuple)):
            # Lúc Val ta dùng ảnh visual (vis) làm chuẩn cho cả 2 nhánh
            images_vis, images_col = batch_data
        else:
            images_vis = images_col = batch_data
            
        images_vis = images_vis.to(device, non_blocking=True)
        images_col = images_col.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # 2. Forward pass
        if use_amp:
            with torch.amp.autocast('cuda'):
                # Ở mode eval(), model trả về 2 giá trị: (visual_emb, color_emb)
                visual_emb, color_emb = model(images_vis)
                
                # [QUAN TRỌNG] Tự tính logits có margin để tính Acc giống Train
                logits = model.arcface_head(visual_emb, labels)
                
                # Tính Loss ArcFace
                loss_arc = criterion(logits, labels)
                
                # Tính Loss SupCon (nếu có)
                if supcon_lambda > 0 and supcon_criterion is not None:
                    loss_sup = supcon_criterion(color_emb, labels)
                else:
                    loss_sup = loss_arc.new_tensor(0.0)
                    
                loss = loss_arc + (supcon_lambda * loss_sup)
        else:
            # Mode FP32
            visual_emb, color_emb = model(images_vis)
            logits = model.arcface_head(visual_emb, labels)
            
            loss_arc = criterion(logits, labels)
            loss_sup = supcon_criterion(color_emb, labels) if supcon_lambda > 0 else 0.0
            loss = loss_arc + (supcon_lambda * loss_sup)

        # 3. Thống kê
        running_loss += loss.item()
        total += labels.size(0)
        
        # Accuracy tính bằng argmax của Logits (có Margin)
        with torch.no_grad():
            pred = logits.argmax(dim=1)
            correct += pred.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    # Trả về trung bình loss và accuracy của epoch
    epoch_loss = running_loss / len(val_loader) if len(val_loader) > 0 else 0.0
    epoch_acc = 100. * correct / total if total > 0 else 0.0
    
    return epoch_loss, epoch_acc

@torch.no_grad()
def extract_embeddings(model, data_loader, device):
    model.eval()
    visual_embeddings = []
    labels = []
    
    for batch_data, batch_labels in tqdm(data_loader, desc="Extracting"):
        # [FIX] Handle tuple
        if isinstance(batch_data, (list, tuple)):
            images, _ = batch_data
        else:
            images = batch_data
            
        images = images.to(device)
        visual_emb, _ = model(images)
        visual_embeddings.append(visual_emb.cpu())
        labels.append(batch_labels)
    
    return torch.cat(visual_embeddings, dim=0).numpy(), torch.cat(labels, dim=0).numpy()


def compute_recall_at_k(test_emb, gallery_emb, test_labels, gallery_labels, k=5, device="cpu"):
    """Tính Recall@k"""
    test_emb = torch.tensor(test_emb, dtype=torch.float32, device=device)
    gallery_emb = torch.tensor(gallery_emb, dtype=torch.float32, device=device)
    test_labels = torch.tensor(test_labels, device=device)
    gallery_labels = torch.tensor(gallery_labels, device=device)
    
    # Normalize embeddings
    test_emb = torch.nn.functional.normalize(test_emb, p=2, dim=1)
    gallery_emb = torch.nn.functional.normalize(gallery_emb, p=2, dim=1)
    
    # Compute similarity
    similarity = test_emb @ gallery_emb.T
    
    # Get top-k
    topk = torch.topk(similarity, k=k, dim=1).indices
    
    # Check if true label in top-k
    correct = 0
    for i in range(len(test_labels)):
        true_label = test_labels[i]
        retrieved_labels = gallery_labels[topk[i]]
        if (retrieved_labels == true_label).any():
            correct += 1
    
    recall = correct / len(test_labels)
    return recall


def compute_precision_at_1(test_emb, gallery_emb, test_labels, gallery_labels, device="cpu"):
    """Tính Precision@1 (1-NN Accuracy)"""
    test_emb = torch.tensor(test_emb, dtype=torch.float32, device=device)
    gallery_emb = torch.tensor(gallery_emb, dtype=torch.float32, device=device)
    test_labels = torch.tensor(test_labels, device=device)
    gallery_labels = torch.tensor(gallery_labels, device=device)
    
    # Normalize embeddings
    test_emb = torch.nn.functional.normalize(test_emb, p=2, dim=1)
    gallery_emb = torch.nn.functional.normalize(gallery_emb, p=2, dim=1)
    
    # Compute similarity
    similarity = test_emb @ gallery_emb.T
    
    # Get top-1
    _, top1 = torch.topk(similarity, k=1, dim=1)
    top1 = top1.squeeze(1)
    
    # Check if predicted label matches true label
    predicted_labels = gallery_labels[top1]
    correct = (predicted_labels == test_labels).sum().item()
    precision = correct / len(test_labels)
    
    return precision

def compute_mean_average_precision(test_emb, gallery_emb, test_labels, gallery_labels, device="cpu"):
    """Tính Mean Average Precision (mAP)"""
    test_emb = torch.tensor(test_emb, dtype=torch.float32, device=device)
    gallery_emb = torch.tensor(gallery_emb, dtype=torch.float32, device=device)
    test_labels = torch.tensor(test_labels, device=device)
    gallery_labels = torch.tensor(gallery_labels, device=device)
    
    # Normalize embeddings
    test_emb = torch.nn.functional.normalize(test_emb, p=2, dim=1)
    gallery_emb = torch.nn.functional.normalize(gallery_emb, p=2, dim=1)
    
    # Compute similarity
    similarity = test_emb @ gallery_emb.T
    
    # Sort by similarity
    _, indices = torch.sort(similarity, dim=1, descending=True)
    
    # Compute AP for each query
    aps = []
    for i in range(len(test_labels)):
        query_label = test_labels[i]
        retrieved_labels = gallery_labels[indices[i]]
        
        # Find relevant items (same label)
        relevant = (retrieved_labels == query_label).float()
        
        if relevant.sum() == 0:
            aps.append(0.0)
            continue
        
        # Compute precision at each relevant position
        cumulative_relevant = torch.cumsum(relevant, dim=0)
        precision_at_k = cumulative_relevant / torch.arange(1, len(relevant) + 1, dtype=torch.float32, device=device)
        
        # Average precision
        ap = (precision_at_k * relevant).sum() / relevant.sum()
        aps.append(ap.item())
    
    map_score = sum(aps) / len(aps)
    return map_score


def plot_training_curves(train_acc_history, val_acc_history, train_loss_history, val_loss_history, save_path=None):
    """
    Plot biểu đồ training curves: accuracy và loss
    Nếu save_path=None thì chỉ hiển thị, không lưu file
    """
    epochs = range(1, len(train_acc_history) + 1)
    
    # Tạo figure với 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot Accuracy
    ax1.plot(epochs, train_acc_history, 'b-', label='Train Accuracy', linewidth=2, marker='o', markersize=4)
    ax1.plot(epochs, val_acc_history, 'r-', label='Validation Accuracy', linewidth=2, marker='s', markersize=4)
    
    # Highlight best validation accuracy
    if val_acc_history:
        best_val_acc = max(val_acc_history)
        best_epoch = val_acc_history.index(best_val_acc) + 1
        ax1.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.5, linewidth=1.5)
        ax1.plot(best_epoch, best_val_acc, 'go', markersize=10, label=f'Best Val Acc: {best_val_acc:.2f}%')
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([1, len(epochs)])
    
    # Plot Loss
    ax2.plot(epochs, train_loss_history, 'b-', label='Train Loss', linewidth=2, marker='o', markersize=4)
    ax2.plot(epochs, val_loss_history, 'r-', label='Validation Loss', linewidth=2, marker='s', markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([1, len(epochs)])
    
    plt.tight_layout()
    
    # Hiển thị biểu đồ thay vì lưu ảnh
    print(f"   📈 Training curves plotted successfully")
    print(f"   👁️  Đang hiển thị biểu đồ...")
    plt.show()


@torch.no_grad()
def evaluate_test(model, test_loader, train_loader, criterion, device):
    """Evaluate on test set với các metrics: Precision@1, Recall@5, mAP"""
    model.eval()
    
    print("\n🧪 Evaluating on TEST set...")
    print("   Extracting embeddings...")
    
    # Extract embeddings từ test set
    test_embeddings, test_labels = extract_embeddings(model, test_loader, device)
    
    # Extract embeddings từ train set (dùng làm gallery)
    gallery_embeddings, gallery_labels = extract_embeddings(model, train_loader, device)
    
    print("   Computing metrics...")
    
    # Compute metrics
    precision_at_1 = compute_precision_at_1(
        test_embeddings, gallery_embeddings, test_labels, gallery_labels, device=device
    )
    
    recall_at_5 = compute_recall_at_k(
        test_embeddings, gallery_embeddings, test_labels, gallery_labels, k=5, device=device
    )
    
    map_score = compute_mean_average_precision(
        test_embeddings, gallery_embeddings, test_labels, gallery_labels, device=device
    )
    
    # Print results
    print(f"\n📊 TEST Results:")
    print(f"   1-NN Accuracy (Precision@1): {precision_at_1 * 100:.2f}%")
    print(f"   Recall@5: {recall_at_5 * 100:.2f}%")
    print(f"   Mean Average Precision (mAP): {map_score:.4f}")
    
    return {
        'precision_at_1': precision_at_1,
        'recall_at_5': recall_at_5,
        'mean_average_precision': map_score,
        'test_embeddings': test_embeddings,
        'test_labels': test_labels,
        'gallery_embeddings': gallery_embeddings,
        'gallery_labels': gallery_labels
    }

def create_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 8, 
    train_split: float = 0.7,
    val_split: float = 0.10,
    test_split: float = 0.20,
    use_weighted_sampling: bool = True,
    dataset_type: str = 'situ' 
):
    if dataset_type == 'situ':
        transform_visual_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            # --- Augmentation màu mạnh ---
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.1)
        ])
    else:
        transform_visual_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.1)
        ])
    
    # 2. Transform chuẩn cho Color (Train) và All (Val/Test)
    transform_clean = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224), # Val/Test dùng CenterCrop
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Transform chuẩn cho Color branch lúc Train (Có thể RandomCrop nhẹ)
    transform_color_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Chọn class Dataset
    DatasetClass = SituDataset
    full_dataset = SituDataset(data_dir, transform_visual=transform_visual_train)

    # [FIX QUAN TRỌNG] Tạo 2 dataset riêng biệt để tránh lỗi reference
    # Dataset 1: Dùng cho Train (is_train=True)
    full_train_dataset = DatasetClass(
        data_dir, 
        transform_visual=transform_visual_train, 
        transform_color=transform_color_train, 
        is_train=True
    )
    
    # Dataset 2: Dùng cho Val/Test (is_train=False)
    full_val_dataset = DatasetClass(
        data_dir, 
        transform_visual=None, 
        transform_color=transform_clean, 
        is_train=False
    )
    
    # Tính toán indices để split
    total_size = len(full_train_dataset)
    train_size = int(total_size * train_split)
    val_size = int(total_size * val_split)
    test_size = total_size - train_size - val_size
    
    # Tạo indices ngẫu nhiên
    indices = torch.randperm(total_size).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Tạo Subset từ đúng dataset nguồn
    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_val_dataset, val_indices)   # Lấy từ dataset Val
    test_dataset = Subset(full_val_dataset, test_indices) # Lấy từ dataset Val
    
    # Weighted Sampler logic
    if use_weighted_sampling:
        # Lấy label từ full_train_dataset thông qua indices
        train_labels = [full_train_dataset.samples[i][1] for i in train_indices]
        class_counts = Counter(train_labels)
        
        weights = []
        for i in train_indices:
            label = full_train_dataset.samples[i][1]
            # Tránh chia cho 0 nếu count lỗi
            count = class_counts[label] if class_counts[label] > 0 else 1
            weights.append(1.0 / count)
            
        train_sampler = WeightedRandomSampler(weights, len(weights))
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, 
                                  num_workers=num_workers, pin_memory=True, persistent_workers=False)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                  num_workers=num_workers, pin_memory=True, persistent_workers=False)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=True, persistent_workers=False)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                             num_workers=num_workers, pin_memory=True, persistent_workers=False)
    
    raw_ids = [cid for _, cid in full_train_dataset.samples] 
    max_label = max(raw_ids) - 1                        
    num_classes = int(max_label + 1)
    return train_loader, val_loader, test_loader, num_classes, full_dataset

def main():
    parser = argparse.ArgumentParser(description='Fine-tune ResMobileNetV2 on in-situ dataset')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to in-situ data directory')
    parser.add_argument('--pretrained', type=str, required=True,
                        help='Path to pretrained checkpoint from vitro')
    parser.add_argument('--output-dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of fine-tuning epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001,  # LR nhỏ hơn cho fine-tune
                        help='Learning rate (should be smaller than pretrain)')
    parser.add_argument('--embedding-size', type=int, default=512,
                        help='Embedding dimension (must match pretrained model)')
    parser.add_argument('--use-color-embedding', action='store_true',
                        help='Enable ColorEncoder and final embedding concat (visual ⊕ α·color)')
    parser.add_argument('--color-embedding-size', type=int, default=64,
                        help='Color embedding dimension (default: 64)')
    parser.add_argument('--color-alpha', type=float, default=0.3,
                        help='Scale for color embedding in final embedding (default: 0.3)')
    parser.add_argument('--supcon-lambda', type=float, default=0.0,
                        help='Weight for SupCon loss on final embedding (default: 0.0 = off)')
    parser.add_argument('--supcon-temp', type=float, default=0.07,
                        help='Temperature for SupCon loss (default: 0.07)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--use-weighted-sampling', action='store_true',
                        help='Use weighted sampling to balance classes (khuyến nghị cho imbalanced data)')
    parser.add_argument('--use-class-weights', action='store_true',
                        help='Use class weights in loss function (khuyến nghị cho imbalanced data)')

    # ArcFace margin warm-up (off by default)
    parser.add_argument('--arcface-margin-start', type=float, default=None,
                        help='ArcFace margin warmup start (e.g., 0.0). If set, enables warm-up.')
    parser.add_argument('--arcface-margin-end', type=float, default=0.35,
                        help='ArcFace margin warmup end (e.g., 0.35 or 0.5)')
    parser.add_argument('--arcface-warmup-epochs', type=int, default=10,
                        help='Number of epochs for linear margin warmup')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("📦 Loading in-situ dataset...")
    train_loader, val_loader, test_loader, num_classes, full_dataset = create_data_loaders(
        args.data_dir, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_weighted_sampling=args.use_weighted_sampling
    )
    print(f"   Classes: {num_classes}")
    print(f"   Train samples: {len(train_loader.dataset)}")
    print(f"   Val samples: {len(val_loader.dataset)}")
    print(f"   Test samples: {len(test_loader.dataset)}")
    
    print("\n🏗️  Creating model...")
    inverted_residual_setting, last_channel = res_mobilenet_conf(width_mult=1.0)
    
    # Tính class counts cho adaptive margin (nếu cần)
    class_counts = None  # Có thể tính từ dataset nếu cần
    
    model = ResMobileNetV2(
        inverted_residual_setting=inverted_residual_setting,
        embedding_size=args.embedding_size,
        num_classes=num_classes,
        use_color_embedding=args.use_color_embedding,
        color_embedding_size=args.color_embedding_size,
        arcface_s=22.5,  # Giảm từ 30.0 → 22.5 để margin mềm hơn, model linh hoạt hơn
        class_counts=class_counts,
        dropout_rate=0.45  # Tăng từ 0.3 → 0.45 để giảm overfitting
    ).to(args.device)

    model.supcon_lambda = float(args.supcon_lambda)
    model.supcon_criterion = SupConLoss(temperature=args.supcon_temp) if args.supcon_lambda > 0 else None
    
    load_pretrained_checkpoint(model, args.pretrained, args.device, strict=False)
    
    unfreeze_all_layers(model)
    
    # Label smoothing để giảm overfitting (giảm confidence quá cao)
    # Tăng từ 0.1 → 0.25 để model linh hoạt hơn, không quá khắt khe
    label_smoothing = 0.25
    if args.use_class_weights:
        class_weights = get_class_weights(full_dataset, num_classes=num_classes, device=args.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        print(f"\n   ✅ Using class weights + label smoothing ({label_smoothing}) in loss function")
        print(f"   Weight range: {class_weights.min():.2f} - {class_weights.max():.2f}")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        print(f"\n   ✅ Using label smoothing ({label_smoothing}) to make model more flexible")
    
    stem_params = list(model.conv1.parameters()) + list(model.bn1.parameters()) + \
                  list(model.transition.parameters())
    mobile_params = list(model.mobile_features.parameters())
    tail_params = list(model.res_block.parameters()) + list(model.res_block_2.parameters()) + \
                  list(model.res_block_3.parameters()) + list(model.res_block_4.parameters()) + \
                  list(model.res_block_5.parameters()) + list(model.res_block_6.parameters())
    head_params = list(model.fc_1.parameters()) + list(model.batch_norm_1.parameters()) + \
                  list(model.arcface_head.parameters())
    
    optimizer = optim.AdamW([
        {'params': stem_params, 'lr': args.lr * 0.1},
        {'params': mobile_params, 'lr': args.lr * 0.1},
        {'params': tail_params, 'lr': args.lr},
        {'params': head_params, 'lr': args.lr * 2.0},
    ], weight_decay=5e-4)  # Tăng weight decay để giảm overfitting
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    print(f"\n📊 Learning rates:")
    print(f"   Stem + MobileNet: {args.lr * 0.1:.6f}")
    print(f"   ResNet Tail: {args.lr:.6f}")
    print(f"   Embedding Head: {args.lr * 2.0:.6f}")
    
    start_epoch = 0
    best_val_acc = 0.0
    
    # Lưu history để plot biểu đồ
    train_acc_history = []
    val_acc_history = []
    train_loss_history = []
    val_loss_history = []
    
    if args.resume:
        print(f"\n📂 Loading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        print(f"   Resumed from epoch {start_epoch}, best val acc: {best_val_acc:.2f}%")
    
    print(f"\n🚀 Starting fine-tuning for {args.epochs} epochs...")
    print(f"   Device: {args.device}")
    print(f"   Base learning rate: {args.lr}")
    print(f"   Batch size: {args.batch_size}\n")
    
    for epoch in range(start_epoch, args.epochs):
        # ArcFace margin warm-up: AdaptiveSubCenterArcFace không hỗ trợ margin warmup trực tiếp
        # Margin được tính từ class_counts trong __init__, không cần warmup
        pass

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, args.device, epoch)
        
        val_loss, val_acc = validate(model, val_loader, criterion, args.device)
        
        scheduler.step()
        
        print(f"\nEpoch {epoch+1}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}\n")
        
        # Lưu vào history
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'best_val_acc': best_val_acc,
            'num_classes': num_classes,
            'embedding_size': args.embedding_size,
            'pretrained_from': args.pretrained,
        }
        
        torch.save(checkpoint, os.path.join(args.output_dir, 'latest.pth'))
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint['best_val_acc'] = best_val_acc
            torch.save(checkpoint, os.path.join(args.output_dir, 'best.pth'))
            print(f"  ✅ Saved best model (val_acc: {val_acc:.2f}%)\n")
    
    # Evaluate on test set sau khi training xong
    print(f"\n{'='*70}")
    print("🧪 FINAL TEST EVALUATION")
    print(f"{'='*70}")
    
    # Load best model để test
    best_checkpoint = torch.load(os.path.join(args.output_dir, 'best.pth'), map_location=args.device)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    test_results = evaluate_test(
        model, test_loader, train_loader, criterion, args.device
    )
    
    # Lưu test results vào checkpoint
    best_checkpoint['test_precision_at_1'] = test_results['precision_at_1']
    best_checkpoint['test_recall_at_5'] = test_results['recall_at_5']
    best_checkpoint['test_mean_average_precision'] = test_results['mean_average_precision']
    torch.save(best_checkpoint, os.path.join(args.output_dir, 'best.pth'))
    
    # Plot biểu đồ training history
    print(f"\n📊 Plotting training curves...")
    plot_training_curves(
        train_acc_history, val_acc_history,
        train_loss_history, val_loss_history,
        save_path=None  # Không lưu ảnh, chỉ hiển thị
    )
    
    print(f"\n🎉 Fine-tuning completed!")
    print(f"   Best Val Acc: {best_val_acc:.2f}%")
    print(f"   Test Precision@1: {test_results['precision_at_1'] * 100:.2f}%")
    print(f"   Test Recall@5: {test_results['recall_at_5'] * 100:.2f}%")
    print(f"   Test mAP: {test_results['mean_average_precision']:.4f}")
    print(f"   Checkpoints saved to: {args.output_dir}")


if __name__ == '__main__':
    main()

