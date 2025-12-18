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
from collections import Counter
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.backbone.ResMobileNetV2 import ResMobileNetV2, res_mobilenet_conf
from models.losses.supcon_loss import SupConLoss
class SituDataset(Dataset):
    def __init__(self, data_dir: str, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        
        for class_dir in sorted(self.data_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            
            try:
                class_id = int(class_dir.name)
            except ValueError:
                continue
            
            video_dir = class_dir / "video"
            if video_dir.exists():
                for img_file in sorted(video_dir.glob("*.png")):
                    if img_file.name.lower() != "thumbs.db":
                        self.samples.append((str(img_file), class_id))
        
        print(f"   Loaded {len(self.samples)} samples from {len(set([s[1] for s in self.samples]))} classes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        label = label - 1

        return image, label


def create_weighted_sampler(dataset):
    """
    
    Args:
        dataset: SituDataset
    
    Returns:
        WeightedRandomSampler
    """
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


def create_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 8,  # Tăng từ 4 lên 8-16 để tăng tốc data loading
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    use_weighted_sampling: bool = True
):
    """
    Tạo train/val/test loaders cho situ
    
    Args:
        data_dir: Đường dẫn đến data situ
        batch_size: Batch size
        num_workers: Số workers
        train_split: Tỷ lệ train (mặc định: 0.7)
        val_split: Tỷ lệ validation (mặc định: 0.15)
        test_split: Tỷ lệ test (mặc định: 0.15)
        use_weighted_sampling: Dùng weighted sampling để cân bằng class
    """
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
    
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
       
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    full_dataset = SituDataset(data_dir, transform=transform_train)
    
    # Split: train/val/test
    total_size = len(full_dataset)
    train_size = int(total_size * train_split)
    val_size = int(total_size * val_split)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Update transforms
    val_dataset.dataset.transform = transform_val
    test_dataset.dataset.transform = transform_test
    
    if use_weighted_sampling:
        sampler = create_weighted_sampler(full_dataset)
        
        train_indices = train_dataset.indices
        train_labels = [full_dataset.samples[i][1] for i in train_indices]
        train_class_counts = Counter(train_labels)
        total_train = len(train_indices)
        num_classes = len(train_class_counts)
        
        train_weights = []
        for idx in train_indices:
            label = full_dataset.samples[idx][1]
            weight = total_train / (num_classes * train_class_counts[label])
            train_weights.append(weight)
        
        train_sampler = WeightedRandomSampler(
            weights=train_weights,
            num_samples=len(train_weights),
            replacement=True
        )
        
        # Trên Kaggle/Colab, persistent_workers có thể gây lỗi
        # Tắt persistent_workers để tránh worker crash (trade-off: hơi chậm hơn một chút)
        use_persistent = False  # Set False để tránh lỗi trên Kaggle/Colab
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler,
            num_workers=num_workers, pin_memory=True,
            persistent_workers=use_persistent,
            prefetch_factor=2 if num_workers > 0 else None  # None nếu num_workers=0
        )
        print("   ✅ Using weighted sampling for balanced training")
    else:
        use_persistent = False  # Set False để tránh lỗi trên Kaggle/Colab
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True,
            persistent_workers=use_persistent,
            prefetch_factor=2 if num_workers > 0 else None
        )
    
    use_persistent = False  # Set False để tránh lỗi trên Kaggle/Colab
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=use_persistent,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=use_persistent,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    return train_loader, val_loader, test_loader, len(set([s[1] for s in full_dataset.samples])), full_dataset


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
    """
    
    Args:
    """
    print(f"📂 Loading pretrained checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    state_dict = checkpoint['model_state_dict']
    model_state_dict = model.state_dict()
    
    if not strict:
        # Handle ArcFace head weight shape mismatch
        if 'arcface_head.weight' in state_dict:
            pretrained_num_classes = state_dict['arcface_head.weight'].shape[0]
            current_num_classes = model_state_dict['arcface_head.weight'].shape[0]
            if pretrained_num_classes != current_num_classes:
                print(f"   ⚠️  num_classes mismatch: {pretrained_num_classes} vs {current_num_classes}")
                print(f"   ⚠️  Skipping arcface_head, will initialize randomly")
                state_dict.pop('arcface_head.weight', None)
    
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
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


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, use_amp=True):
    """Train 1 epoch với mixed precision để tăng tốc"""
    model.train()
    supcon_lambda = getattr(model, "supcon_lambda", 0.0)
    supcon_criterion = getattr(model, "supcon_criterion", None)
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Handle device: support both string and torch.device
    if isinstance(device, str):
        is_cuda = device == 'cuda' and torch.cuda.is_available()
    else:
        is_cuda = device.type == 'cuda'
    
    # Mixed precision scaler (dùng API mới để tránh deprecation warning)
    scaler = torch.amp.GradScaler('cuda') if use_amp and is_cuda else None
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)  # leave=False để không spam terminal
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)  # non_blocking=True để tăng tốc transfer
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Mixed precision forward
        if scaler is not None:
            with torch.cuda.amp.autocast():
                embeddings = model(images)
                logits = model.arcface_head(embeddings, labels)
                loss_arc = criterion(logits, labels)

                if supcon_lambda > 0 and supcon_criterion is not None and getattr(model, "use_color_embedding", False):
                    final_emb = model.get_final_embedding(images)
                    loss_sup = supcon_criterion(final_emb, labels)
                else:
                    loss_sup = loss_arc.new_tensor(0.0)

                loss = loss_arc + (supcon_lambda * loss_sup)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            embeddings = model(images)
            logits = model.arcface_head(embeddings, labels)
            loss_arc = criterion(logits, labels)

            if supcon_lambda > 0 and supcon_criterion is not None and getattr(model, "use_color_embedding", False):
                final_emb = model.get_final_embedding(images)
                loss_sup = supcon_criterion(final_emb, labels)
            else:
                loss_sup = loss_arc.new_tensor(0.0)

            loss = loss_arc + (supcon_lambda * loss_sup)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        running_loss += loss.item()
        total += labels.size(0)
        # Tính accuracy mỗi batch (cần thiết cho tracking)
        with torch.no_grad():
            correct += compute_arcface_accuracy(embeddings.detach(), labels, model.arcface_head)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, val_loader, criterion, device, use_amp=True):
    """Validate model với mixed precision để tăng tốc"""
    model.eval()
    supcon_lambda = getattr(model, "supcon_lambda", 0.0)
    supcon_criterion = getattr(model, "supcon_criterion", None)
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Handle device: support both string and torch.device
    if isinstance(device, str):
        is_cuda = device == 'cuda' and torch.cuda.is_available()
    else:
        is_cuda = device.type == 'cuda'
    
    pbar = tqdm(val_loader, desc="[Val]", leave=False)  # leave=False
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)  # non_blocking=True
        labels = labels.to(device, non_blocking=True)
        
        if use_amp and is_cuda:
            with torch.cuda.amp.autocast():
                embeddings = model(images)
                logits = model.arcface_head(embeddings, labels)
                loss_arc = criterion(logits, labels)

                if supcon_lambda > 0 and supcon_criterion is not None and getattr(model, "use_color_embedding", False):
                    final_emb = model.get_final_embedding(images)
                    loss_sup = supcon_criterion(final_emb, labels)
                else:
                    loss_sup = loss_arc.new_tensor(0.0)

                loss = loss_arc + (supcon_lambda * loss_sup)
        else:
            embeddings = model(images)
            logits = model.arcface_head(embeddings, labels)
            loss_arc = criterion(logits, labels)

            if supcon_lambda > 0 and supcon_criterion is not None and getattr(model, "use_color_embedding", False):
                final_emb = model.get_final_embedding(images)
                loss_sup = supcon_criterion(final_emb, labels)
            else:
                loss_sup = loss_arc.new_tensor(0.0)

            loss = loss_arc + (supcon_lambda * loss_sup)
        
        running_loss += loss.item()
        total += labels.size(0)
        correct += compute_arcface_accuracy(embeddings, labels, model.arcface_head)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def extract_embeddings(model, data_loader, device):
    """Extract embeddings và labels từ data_loader"""
    model.eval()
    embeddings = []
    labels = []
    
    for images, batch_labels in tqdm(data_loader, desc="Extracting embeddings"):
        images = images.to(device)
        batch_embeddings = model(images)
        embeddings.append(batch_embeddings.cpu())
        labels.append(batch_labels)
    
    embeddings = torch.cat(embeddings, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    
    return embeddings, labels


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
    model = ResMobileNetV2(
        inverted_residual_setting=inverted_residual_setting,
        embedding_size=args.embedding_size,
        num_classes=num_classes,
        use_color_embedding=args.use_color_embedding,
        color_embedding_size=args.color_embedding_size
    ).to(args.device)

    model.color_alpha = args.color_alpha if hasattr(model, "color_alpha") else args.color_alpha
    model.supcon_lambda = float(args.supcon_lambda)
    model.supcon_criterion = SupConLoss(temperature=args.supcon_temp) if args.supcon_lambda > 0 else None
    
    load_pretrained_checkpoint(model, args.pretrained, args.device, strict=False)
    
    unfreeze_all_layers(model)
    
    # Label smoothing để giảm overfitting (giảm confidence quá cao)
    if args.use_class_weights:
        class_weights = get_class_weights(full_dataset, num_classes=num_classes, device=args.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        print(f"\n   ✅ Using class weights + label smoothing in loss function")
        print(f"   Weight range: {class_weights.min():.2f} - {class_weights.max():.2f}")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        print(f"\n   ✅ Using label smoothing (0.1) to reduce overfitting")
    
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
        # ArcFace margin warm-up
        if args.arcface_margin_start is not None and hasattr(model, "arcface_head"):
            warm = max(int(args.arcface_warmup_epochs), 1)
            t = min(max(epoch / warm, 0.0), 1.0)
            m = float(args.arcface_margin_start + t * (args.arcface_margin_end - args.arcface_margin_start))
            if hasattr(model.arcface_head, "set_margin"):
                model.arcface_head.set_margin(m)
            else:
                model.arcface_head.m = m

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

